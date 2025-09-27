#!/usr/bin/env python3
"""
Sign-to-Text (minimal, self-trainable)

This program lets you:
  1) Record your own sign samples for target words (labels)
  2) Run real-time recognition using a lightweight KNN classifier on MediaPipe hand landmarks

No pre-trained model is required. You can quickly collect ~30 samples per label and start.

Usage examples:
  Record samples for a label:
    python sign_to_text.py record --label Hello --num-samples 40

  Add another label later (data is appended to the same dataset file):
    python sign_to_text.py record --label Thanks --num-samples 40

  Run real-time prediction:
    python sign_to_text.py run

Controls (record):
  - Press 'c' to capture current frame as a sample
  - Press 'q' to stop recording early

Controls (run):
  - Shows predicted label and confidence
  - Press SPACE to append current stable prediction to the phrase buffer
  - Press BACKSPACE to delete last word in the phrase buffer
  - Press ENTER to print the phrase to console and clear it
  - Press 'q' to quit

Requirements:
  - Python 3.10+
  - Packages: mediapipe, opencv-python, numpy
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    print("OpenCV (cv2) is required. Install via: pip install opencv-python", file=sys.stderr)
    raise

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover
    print("MediaPipe is required. Install via: pip install mediapipe", file=sys.stderr)
    raise


DEFAULT_DATASET_PATH = Path("sign_dataset.npz")


def _ensure_dir(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def _put_text(image: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int] = (0, 255, 0), scale: float = 0.7, thickness: int = 2) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)


class HandLandmarkExtractor:
    """
    Wraps MediaPipe Hands to extract 21 hand landmarks.
    Produces normalized features robust to scale and handedness.
    """

    def __init__(self, max_num_hands: int = 1, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
        )

    def close(self) -> None:
        self._hands.close()

    def extract(self, bgr_frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Returns (features, handedness) where features is a 63-dim vector (21 keypoints * (x,y,z)).
        The coordinates are normalized: origin at wrist, scale by hand size, flipped so left/right align.
        If no hand is detected, returns (None, None).
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None, None

        # Take the most confident hand (first)
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness_label: Optional[str] = None
        if results.multi_handedness:
            handedness_label = results.multi_handedness[0].classification[0].label  # "Left" or "Right"

        # Convert to numpy
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)  # shape (21,3)

        # Normalize: translate to wrist-origin, scale by max distance
        wrist = pts[0].copy()
        centered = pts - wrist  # shape (21,3)
        # Mirror so that both hands map to a canonical orientation along x
        # If label is "Left", flip x to align with "Right"
        if handedness_label and handedness_label.lower().startswith("left"):
            centered[:, 0] *= -1.0

        # Scale normalization by max pairwise distance magnitude along x,y; z kept relative
        xy = centered[:, :2]
        scale = np.linalg.norm(xy, axis=1).max()
        if scale < 1e-6:
            return None, None
        normalized = centered / scale

        # Flatten to 63-dim feature
        feat = normalized.reshape(-1)
        return feat, handedness_label


class SimpleKNNClassifier:
    """A tiny KNN classifier (Euclidean), implemented to avoid heavyweight deps."""

    def __init__(self, k_neighbors: int = 7) -> None:
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be positive")
        self.k = k_neighbors
        self._x: Optional[np.ndarray] = None
        self._y_indices: Optional[np.ndarray] = None
        self._labels: List[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._x is not None and self._y_indices is not None and len(self._labels) > 0

    @property
    def labels(self) -> List[str]:
        return list(self._labels)

    def fit(self, x: np.ndarray, y_indices: np.ndarray, label_names: List[str]) -> None:
        if x.ndim != 2:
            raise ValueError("x must be 2D [n_samples, n_features]")
        if len(x) != len(y_indices):
            raise ValueError("x and y_indices must have same length")
        self._x = x.astype(np.float32)
        self._y_indices = y_indices.astype(np.int64)
        self._labels = list(label_names)

    def predict(self, x_query: np.ndarray) -> Tuple[str, float]:
        if not self.is_fitted or self._x is None or self._y_indices is None:
            raise RuntimeError("Classifier not fitted")
        diffs = self._x - x_query[None, :]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))  # shape [n_samples]
        k = min(self.k, len(dists))
        nn_idx = np.argpartition(dists, k - 1)[:k]
        nn_labels = self._y_indices[nn_idx]
        counts = np.bincount(nn_labels, minlength=len(self._labels))
        pred_idx = int(np.argmax(counts))
        confidence = float(counts[pred_idx]) / float(k)
        pred_label = self._labels[pred_idx]
        return pred_label, confidence


class DatasetManager:
    """Manages loading/saving of dataset features and labels."""

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path
        self.x: Optional[np.ndarray] = None  # [n_samples, n_features]
        self.y_indices: Optional[np.ndarray] = None  # [n_samples]
        self.label_names: List[str] = []

    def load(self) -> bool:
        if not self.dataset_path.exists():
            return False
        data = np.load(self.dataset_path, allow_pickle=False)
        self.x = data["x"].astype(np.float32)
        self.y_indices = data["y_indices"].astype(np.int64)
        self.label_names = [str(s) for s in data["label_names"]]
        return True

    def save(self) -> None:
        if self.x is None or self.y_indices is None:
            raise RuntimeError("Nothing to save; collect data first")
        _ensure_dir(self.dataset_path)
        np.savez_compressed(
            self.dataset_path,
            x=self.x,
            y_indices=self.y_indices,
            label_names=np.array(self.label_names, dtype=object),
        )

    def add_samples(self, features: np.ndarray, label: str) -> None:
        if features.ndim != 2:
            raise ValueError("features must be 2D [n_samples, n_features]")
        if label not in self.label_names:
            self.label_names.append(label)
        label_idx = self.label_names.index(label)
        y_new = np.full((features.shape[0],), label_idx, dtype=np.int64)
        if self.x is None or self.y_indices is None:
            self.x = features.astype(np.float32)
            self.y_indices = y_new
        else:
            self.x = np.concatenate([self.x, features.astype(np.float32)], axis=0)
            self.y_indices = np.concatenate([self.y_indices, y_new], axis=0)


@dataclass
class RunConfig:
    dataset_path: Path = DEFAULT_DATASET_PATH
    k_neighbors: int = 7
    smoothing_window: int = 7
    min_confidence: float = 0.6


def record_mode(label: str, num_samples: int, dataset_path: Path, min_hands_conf: float = 0.5) -> None:
    extractor = HandLandmarkExtractor(min_detection_confidence=min_hands_conf, min_tracking_confidence=min_hands_conf)
    dm = DatasetManager(dataset_path)
    dm.load()  # best-effort; ignore absence

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        extractor.close()
        raise RuntimeError("Cannot open webcam (device 0)")

    captured_features: List[np.ndarray] = []
    per_second_hint_last = 0.0

    print(f"Recording samples for label='{label}' — target {num_samples} samples")
    print("Hold the sign steady. Press 'c' to capture a sample, 'q' to quit early.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from camera; stopping.")
            break

        feat, handed = extractor.extract(frame)

        # UI overlays
        _put_text(frame, f"Label: {label}", (10, 30))
        _put_text(frame, f"Samples: {len(captured_features)}/{num_samples}", (10, 60))
        _put_text(frame, "Press 'c' to capture, 'q' to stop", (10, 90), (255, 255, 255))

        now = time.time()
        if feat is None:
            if now - per_second_hint_last > 1.0:
                _put_text(frame, "No hand detected", (10, 120), (0, 0, 255))
                per_second_hint_last = now
        else:
            # Draw a small indicator when a hand is detected
            _put_text(frame, f"Hand: {handed or 'Unknown'}", (10, 120), (0, 255, 255))

        cv2.imshow("Record - Sign-to-Text", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c') and feat is not None:
            captured_features.append(feat.copy())
            if len(captured_features) >= num_samples:
                break

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()

    if len(captured_features) == 0:
        print("No samples captured; nothing saved.")
        return

    features_arr = np.stack(captured_features, axis=0)
    dm.add_samples(features_arr, label)
    dm.save()
    print(f"Saved dataset to {dataset_path} with total samples: {len(dm.y_indices) if dm.y_indices is not None else 0}")


def run_mode(cfg: RunConfig) -> None:
    dm = DatasetManager(cfg.dataset_path)
    if not dm.load():
        raise RuntimeError(f"Dataset not found at {cfg.dataset_path}. Please run record mode first.")
    if dm.x is None or dm.y_indices is None or len(dm.label_names) == 0:
        raise RuntimeError("Dataset is empty or invalid. Record samples first.")

    clf = SimpleKNNClassifier(k_neighbors=cfg.k_neighbors)
    clf.fit(dm.x, dm.y_indices, dm.label_names)

    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        extractor.close()
        raise RuntimeError("Cannot open webcam (device 0)")

    pred_window: Deque[str] = deque(maxlen=cfg.smoothing_window)
    stable_label: Optional[str] = None
    phrase_tokens: List[str] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from camera; stopping.")
            break

        feat, _ = extractor.extract(frame)
        pred_label: str = ""
        pred_conf: float = 0.0
        if feat is not None:
            pred_label, pred_conf = clf.predict(feat)
            if pred_conf >= cfg.min_confidence:
                pred_window.append(pred_label)
            else:
                pred_window.append("")
        else:
            pred_window.append("")

        # Majority vote for stability
        non_empty = [p for p in pred_window if p]
        if non_empty:
            counts = Counter(non_empty)
            stable_label, _ = counts.most_common(1)[0]
        else:
            stable_label = ""

        # UI overlays
        _put_text(frame, f"Pred: {pred_label or '-'} ({pred_conf:.0%})", (10, 30))
        _put_text(frame, f"Stable: {stable_label or '-'}", (10, 60))
        _put_text(frame, f"Phrase: {' '.join(phrase_tokens) if phrase_tokens else '-'}", (10, 90), (255, 255, 255))
        _put_text(frame, "SPACE=add  BACKSPACE=del  ENTER=print  q=quit", (10, 120), (200, 200, 200), 0.6, 1)

        cv2.imshow("Run - Sign-to-Text", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == 32:  # Space
            if stable_label:
                phrase_tokens.append(stable_label)
        if key == 8:  # Backspace
            if phrase_tokens:
                phrase_tokens.pop()
        if key == 13:  # Enter
            phrase = " ".join(phrase_tokens)
            if phrase:
                print(f"Phrase: {phrase}")
                phrase_tokens.clear()

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sign-to-Text: record and run real-time recognition")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("record", help="Record samples for a label")
    p_rec.add_argument("--label", required=True, type=str, help="Word label to record (e.g., Hello)")
    p_rec.add_argument("--num-samples", type=int, default=40, help="Number of samples to capture")
    p_rec.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to dataset .npz")
    p_rec.add_argument("--min-hands-conf", type=float, default=0.5, help="Min confidence for hand detection/tracking")

    p_run = sub.add_parser("run", help="Run real-time recognition")
    p_run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to dataset .npz")
    p_run.add_argument("--k", type=int, default=7, help="K for KNN")
    p_run.add_argument("--smooth", type=int, default=7, help="Smoothing window (frames)")
    p_run.add_argument("--min-conf", type=float, default=0.6, help="Min confidence to accept a frame prediction")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.cmd == "record":
        record_mode(
            label=args.label,
            num_samples=max(1, int(args.num_samples)),
            dataset_path=args.dataset,
            min_hands_conf=float(args.min_hands_conf),
        )
        return

    if args.cmd == "run":
        cfg = RunConfig(
            dataset_path=args.dataset,
            k_neighbors=max(1, int(args.k)),
            smoothing_window=max(1, int(args.smooth)),
            min_confidence=float(args.min_conf),
        )
        run_mode(cfg)
        return

    parser.print_help()


if __name__ == "__main__":
    main()

