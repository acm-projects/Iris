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

## Gloss phrases to English

Install `backend/requirements.txt` for the HTTP server, or
`backend/sign_inference/requirements-inference.txt` for the camera runners.
Create `backend/.env` locally using `backend/.env.example` as a template, then
enter your Gemini Developer API key on the `GEMINI_API_KEY=` line. Do not overwrite
existing keys. Set `GLOSS_LLM_MODEL=gemini-flash-latest`, replacing any old OpenAI
model setting. OpenAI keys are no longer used by this service. Model access
depends on your API account. Process environment variables override this file.
The backend reads this explicit path regardless of the working directory.
`.env` is ignored by Git; credentials never belong in frontend/Vite variables.

Run `python backend/sign_inference/continuous_camera.py --model meeting100 --landmarks`
for continuous recognition. By default a two-second pause without detected signing
automatically submits the accepted phrase. Lower your hands out of view when done;
keeping hands raised may keep signing active. Adjust with `--submit-pause 3`.
Use `--manual-submit` to disable automatic submission. Enter also submits the phrase;
C clears it. R/S in `camera_inference.py` instead manually delimits each sign,
accepting its highest-scoring gloss, including intentionally recorded repeats.
Enter submits that runner's accumulated phrase too.

Continuous recognition uses a 60-frame landmark FIFO, separate from the ordered
accepted-gloss list. Each window's winner is determined by confidence, then the
three-window smoothing and two-window stability checks decide whether
to commit it. Duplicate suppression rearms after rest, confidence release, or a
different committed gloss. These are heuristics: there is no trained sign-boundary
or blank detector. Hands disappearing can falsely signal rest; repeated signs
without a detectable release may be missed. Manual R/S supplies explicit boundaries.

The default acceptance confidence is 30% (previously 45%), with two consecutive
winning predictions (previously three) and three smoothing windows (previously
five). This accepts shorter, less confident signs sooner, at the cost of more
false positives. Duplicate suppression still applies. Tune with `--min-conf`,
`--stable`, and `--smooth`; restore the old settings with
`--min-conf 0.45 --stable 3 --smooth 5`. The preview reports acceptance status,
including insufficient confidence, unstable winners, and suppressed duplicates.

The phrase now persists across pauses and is not silently truncated to 12 words.
Automatic pause submission (or Enter) snapshots it and issues one Gemini request in a worker (SDK retries
disabled), and leaves recognition running. Further submissions are blocked while
pending. Success consumes only the snapshot; newly accepted words remain. Failure
retains the words for explicit retry. C discards the active phrase but does not
cancel an already submitted request. Translation status appears in the preview;
the complete English result is also printed to the terminal.

The existing FastAPI server also exposes `POST /gloss/translate` with
`{"glosses":["i","want","help"]}`, returning `{"english":"..."}`. It uses the
same backend service, with one provider request per HTTP invocation. The Electron
engine's `recognize_sign` remains a placeholder; desktop recognition and submit
controls are not wired to these camera runners. No LLM calls occur per frame.
Automatic submission attempts each phrase revision once, including on failure;
it does not retry continuously while you rest. Recognition of an additional word
allows another automatic attempt, or use Enter for an explicit retry.

Both model paths already use descending softmax `topk`; the manual selector uses
the maximum numeric `confidence` (pose) or `score` (experimental legacy) explicitly.
The legacy scores and label mapping retain their existing experimental caveats.

## Test

Windows:

```powershell
.\.venv-wlasl\Scripts\python.exe -m unittest discover -s .\backend\sign_inference -p "test_*inference*.py" -v
```

macOS:

```bash
./.venv-wlasl/bin/python -m unittest discover -s ./backend/sign_inference -p "test_*inference*.py" -v
```
