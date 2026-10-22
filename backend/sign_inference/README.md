# Standalone WLASL-100 camera inference

This manual-clip demo proves that camera frames can travel through MediaPipe
and the WLASL-100 TGCN model. The checkpoint, preprocessing, and label contract
are reviewed in `PREPROCESSING_STATUS.md`; real-camera recognition accuracy is
still experimental and must be measured separately.

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
.\.venv-wlasl\Scripts\python.exe .\backend\sign_inference\camera_inference.py --landmarks
```

macOS:

```bash
./.venv-wlasl/bin/python ./backend/sign_inference/camera_inference.py --landmarks
```

- `R`: start recording a sign
- `S`: stop and predict the top three WLASL-100 glosses
- `Q`: quit
- `--camera 1`: select a different camera
- omit `--landmarks` to hide landmark dots

Record the complete sign between `R` and `S` with the upper body and both hands
visible. The model receives unmirrored RGB frames; only the preview is mirrored.
Scores are raw softmax model scores, not calibrated probabilities.

## Live confidence calibration

Live acceptance thresholds must come from labelled camera clips. Iris no
longer learns confidence ranges from its own unlabelled predictions. Use
`camera_evaluation.py` to run a versioned clip manifest through the same
sampling, segmentation, and inference path used by the desktop app, then
place the generated artifact at:

```text
backend/sign_inference/calibration/asl100-camera-v1.json
```

Without that artifact, live recognition reports calibration as missing and
uses an explicit bootstrap cutoff. See `../BACKEND_RUNTIME.md` for the
manifest format, command, runtime architecture, and current measured status.

## Test

Windows:

```powershell
.\.venv-wlasl\Scripts\python.exe -m unittest discover -s .\backend\sign_inference -p "test_*inference*.py" -v
```

macOS:

```bash
./.venv-wlasl/bin/python -m unittest discover -s ./backend/sign_inference -p "test_*inference*.py" -v
```
