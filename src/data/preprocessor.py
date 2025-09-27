"""
Data preprocessing module for sign language recognition
Handles video/image preprocessing, augmentation, and feature extraction
"""
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from typing import List, Tuple, Optional, Dict
import albumentations as A
from pathlib import Path
import logging

from config.config import config

class MediaPipeDetector:
    """MediaPipe-based hand and pose detection"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe models
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=config.ui.detection_confidence,
            min_tracking_confidence=config.ui.tracking_confidence
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=config.ui.detection_confidence,
            min_tracking_confidence=config.ui.tracking_confidence
        )
    
    def extract_landmarks(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract hand and pose landmarks from image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary containing extracted landmarks
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process hands
        hand_results = self.hands.process(image_rgb)
        
        # Process pose
        pose_results = self.pose.process(image_rgb)
        
        landmarks = {
            'hands': [],
            'pose': []
        }
        
        # Extract hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_coords = []
                for landmark in hand_landmarks.landmark:
                    hand_coords.extend([landmark.x, landmark.y, landmark.z])
                landmarks['hands'].append(np.array(hand_coords))
        
        # Extract pose landmarks (upper body focus)
        if pose_results.pose_landmarks:
            pose_coords = []
            # Focus on upper body landmarks (0-10: face, 11-16: arms)
            for i in range(17):  # Upper body landmarks
                landmark = pose_results.pose_landmarks.landmark[i]
                pose_coords.extend([landmark.x, landmark.y, landmark.z])
            landmarks['pose'] = np.array(pose_coords)
        
        return landmarks
    
    def close(self):
        """Clean up MediaPipe resources"""
        self.hands.close()
        self.pose.close()

class VideoProcessor:
    """Process video files for sign language recognition"""
    
    def __init__(self):
        self.detector = MediaPipeDetector()
        
        # Setup data augmentation pipeline
        self.augmentation = A.Compose([
            A.Rotate(limit=config.data.rotation_range, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=config.data.rotation_range,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3
            )
        ])
    
    def extract_frames(self, video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract frames from video file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of extracted frames
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if max_frames is None:
            max_frames = config.data.sequence_length
        
        frame_count = 0
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (config.data.frame_width, config.data.frame_height))
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        return frames
    
    def process_video_sequence(self, video_path: str, label: str) -> Dict:
        """
        Process a complete video sequence for training
        
        Args:
            video_path: Path to video file
            label: Ground truth label
            
        Returns:
            Processed sequence data
        """
        frames = self.extract_frames(video_path)
        
        if len(frames) == 0:
            logging.warning(f"No frames extracted from {video_path}")
            return None
        
        # Pad or truncate sequence to fixed length
        target_length = config.data.sequence_length
        
        if len(frames) < target_length:
            # Pad with last frame
            last_frame = frames[-1]
            frames.extend([last_frame] * (target_length - len(frames)))
        elif len(frames) > target_length:
            # Sample frames uniformly
            indices = np.linspace(0, len(frames) - 1, target_length, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Extract landmarks for each frame
        landmark_sequence = []
        frame_sequence = []
        
        for frame in frames:
            # Extract landmarks
            landmarks = self.detector.extract_landmarks(frame)
            landmark_sequence.append(landmarks)
            
            # Store processed frame
            frame_sequence.append(frame)
        
        return {
            'frames': np.array(frame_sequence),
            'landmarks': landmark_sequence,
            'label': label,
            'video_path': video_path
        }
    
    def augment_data(self, frames: np.ndarray, apply_augmentation: bool = True) -> np.ndarray:
        """
        Apply data augmentation to frame sequence
        
        Args:
            frames: Input frame sequence
            apply_augmentation: Whether to apply augmentation
            
        Returns:
            Augmented frame sequence
        """
        if not apply_augmentation:
            return frames
        
        augmented_frames = []
        for frame in frames:
            # Apply augmentation
            augmented = self.augmentation(image=frame)['image']
            augmented_frames.append(augmented)
        
        return np.array(augmented_frames)

class DatasetBuilder:
    """Build and manage sign language datasets"""
    
    def __init__(self):
        self.processor = VideoProcessor()
        self.data_cache = {}
    
    def process_dataset_folder(self, dataset_path: str, output_path: str):
        """
        Process a complete dataset folder
        
        Args:
            dataset_path: Path to dataset folder
            output_path: Path to save processed data
        """
        dataset_path = Path(dataset_path)
        processed_data = []
        
        # Process each class folder
        for class_folder in dataset_path.iterdir():
            if not class_folder.is_dir():
                continue
            
            class_name = class_folder.name
            logging.info(f"Processing class: {class_name}")
            
            # Process each video in the class
            for video_file in class_folder.glob("*.mp4"):
                try:
                    sequence_data = self.processor.process_video_sequence(
                        str(video_file), 
                        class_name
                    )
                    
                    if sequence_data is not None:
                        processed_data.append(sequence_data)
                        
                except Exception as e:
                    logging.error(f"Error processing {video_file}: {e}")
        
        # Save processed data
        self._save_processed_data(processed_data, output_path)
        
        return processed_data
    
    def _save_processed_data(self, data: List[Dict], output_path: str):
        """Save processed data to disk"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy arrays for efficient loading
        frames_data = []
        landmarks_data = []
        labels = []
        
        for sequence in data:
            frames_data.append(sequence['frames'])
            landmarks_data.append(sequence['landmarks'])
            labels.append(sequence['label'])
        
        np.save(output_path / 'frames.npy', np.array(frames_data))
        np.save(output_path / 'landmarks.npy', np.array(landmarks_data))
        np.save(output_path / 'labels.npy', np.array(labels))
        
        # Save metadata
        metadata = {
            'num_sequences': len(data),
            'classes': list(set(labels)),
            'sequence_length': config.data.sequence_length,
            'frame_shape': config.model.input_shape
        }
        
        pd.DataFrame([metadata]).to_json(output_path / 'metadata.json')
        
        logging.info(f"Saved {len(data)} sequences to {output_path}")

# Utility functions
def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize landmark coordinates"""
    if len(landmarks) == 0:
        return landmarks
    
    # Center and scale landmarks
    landmarks = landmarks - np.mean(landmarks, axis=0)
    landmarks = landmarks / (np.std(landmarks) + 1e-8)
    
    return landmarks

def create_synthetic_data_sample():
    """Create a sample dataset for testing purposes"""
    # This function would create synthetic sign language data
    # for initial testing and development
    pass