# Sign-to-Text (Minimal, Self-Trainable)

This lightweight Python app learns your own signs and translates them to text in real time using a webcam. It uses MediaPipe hand landmarks plus a tiny KNN classifier.

## Quickstart
1) Install dependencies (in a virtualenv is recommended):
```bash
python3 -m venv .venv && . .venv/bin/activate && python -m pip install -U pip && python -m pip install -r requirements.txt
```

2) Record samples for a label (e.g., Hello):
```bash
python sign_to_text.py record --label Hello --num-samples 40
```
- Press `c` to capture a sample, `q` to stop.

3) Add another label (e.g., Thanks):
```bash
python sign_to_text.py record --label Thanks --num-samples 40
```

4) Run real-time prediction:
```bash
python sign_to_text.py run
```
- SPACE: add stable word to phrase
- BACKSPACE: delete last word
- ENTER: print and clear phrase
- q: quit

Tips: collect 30–60 samples per word, keep lighting consistent, center your hand.

## Options
- Change dataset path: `--dataset /path/to/data.npz`
- K for KNN: `--k 9`
- Smoothing window: `--smooth 9`
- Min confidence: `--min-conf 0.7`

If OpenCV GUI fails on Linux, try:
```bash
sudo apt-get install -y libgl1
```

License: MIT
