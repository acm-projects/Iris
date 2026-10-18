# Standalone WLASL-100 camera inference

This manual-clip demo proves that camera frames can travel through MediaPipe
and the WLASL-100 TGCN model. The checkpoint, preprocessing, and label contract
are reviewed in `PREPROCESSING_STATUS.md`; real-camera recognition accuracy is
still experimental and must be measured separately.

## Windows setup

Use 64-bit CPython 3.12. The Python 3.14 installation currently present on the
development machine is not a supported runtime for the required MediaPipe
wheel.

```powershell
py -3.12 -m venv .venv-wlasl
.\.venv-wlasl\Scripts\python.exe -m pip install --upgrade pip
.\.venv-wlasl\Scripts\python.exe -m pip install -r .\backend\sign_inference\requirements-inference.txt
.\.venv-wlasl\Scripts\python.exe .\backend\sign_inference\setup_assets.py
```

`setup_assets.py` downloads:

- the revision-pinned WLASL-100 checkpoint and verifies its SHA-256;
- MediaPipe Pose Landmarker Lite;
- MediaPipe Hand Landmarker.

The downloaded binaries are intentionally ignored by Git. To check an existing
installation without downloading anything:

```powershell
.\.venv-wlasl\Scripts\python.exe .\backend\sign_inference\setup_assets.py --verify-only
```

## Run

```powershell
.\.venv-wlasl\Scripts\python.exe .\backend\sign_inference\camera_inference.py --landmarks
```

- `R`: start recording a sign
- `S`: stop and predict the top three WLASL-100 glosses
- `Q`: quit
- `--camera 1`: select a different camera
- omit `--landmarks` to hide landmark dots

Record the complete sign between `R` and `S` with the upper body and both hands
visible. The model receives unmirrored RGB frames; only the preview is mirrored.
Scores are raw softmax model scores, not calibrated probabilities.

## Test

```powershell
.\.venv-wlasl\Scripts\python.exe -m unittest discover -s .\backend\sign_inference -p "test_*inference*.py" -v
```
