# Standalone WLASL-100 camera inference

This is an experimental manual-clip demo. It proves that camera frames can
travel through MediaPipe and the WLASL-100 TGCN model; it does not establish
that the predicted glosses are accurate. The checkpoint's original
preprocessing and class-index provenance are still unverified.

## How it works

1. `camera_inference.py` reads RGB frames from the selected webcam.
2. `sign_inference.py` runs MediaPipe Pose and Hand Landmarkers and
   `inference_utils.py` remaps their output to the model's 55-joint layout.
3. Frames recorded between `R` and `S` are centered or padded to 50 frames,
   normalized, and packed as a float32 tensor shaped `(1, 55, 100)`.
4. `tgcn_model.py` runs the WLASL-100 TGCN checkpoint and returns the three
   highest-scoring glosses.

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

## macOS setup

Install CPython 3.12 with [Homebrew](https://brew.sh/), then create the same
isolated environment:

```bash
brew install python@3.12
python3.12 -m venv .venv-wlasl
./.venv-wlasl/bin/python -m pip install --upgrade pip
./.venv-wlasl/bin/python -m pip install -r ./backend/sign_inference/requirements-inference.txt
./.venv-wlasl/bin/python ./backend/sign_inference/setup_assets.py
```

The first camera launch may prompt for access. Allow camera access for Terminal
or your IDE under **System Settings → Privacy & Security → Camera**.

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

Windows:

```powershell
.\.venv-wlasl\Scripts\python.exe .\backend\sign_inference\camera_inference.py --experimental-legacy --landmarks
```

macOS:

```bash
./.venv-wlasl/bin/python ./backend/sign_inference/camera_inference.py --experimental-legacy --landmarks
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

Windows:

```powershell
.\.venv-wlasl\Scripts\python.exe -m unittest discover -s .\backend\sign_inference -p "test_*inference*.py" -v
```

macOS:

```bash
./.venv-wlasl/bin/python -m unittest discover -s ./backend/sign_inference -p "test_*inference*.py" -v
```
