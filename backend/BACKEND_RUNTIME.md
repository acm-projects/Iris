# Iris backend runtime

This document covers the desktop translation backend. The desktop flow does
not require a separately started HTTP server.

## Runtime ownership

Electron starts `engine/engine.py` and communicates with it using one JSON
object per line. The engine owns:

- virtual-camera lifecycle and frame forwarding;
- live MediaPipe/TGCN recognition;
- sentence assembly;
- operating-system speech synthesis; and
- virtual-microphone playback.

`server/api.py` remains an optional standalone development API. The Electron
application does not call it. Keeping desktop speech in the engine gives
camera controls, test tones, and speech one process and one audio lock.

## Setup

Use 64-bit CPython 3.12:

```powershell
py -3.12 -m venv .venv-wlasl
.\.venv-wlasl\Scripts\python.exe -m pip install -r .\backend\requirements.txt
.\.venv-wlasl\Scripts\python.exe .\backend\sign_inference\setup_assets.py
```

The model checkpoint and MediaPipe task files are downloaded assets and are
not stored in Git. Verify them without downloading:

```powershell
.\.venv-wlasl\Scripts\python.exe .\backend\sign_inference\setup_assets.py --verify-only
```

The engine's `system_check` request reports missing Python modules, virtual
devices, inference assets, and invalid calibration files. A missing
calibration is reported separately and does not pretend the checkpoint is
camera-calibrated.

Packaged builds currently locate a project virtual environment first and then
fall back to system `python`/`python3`. Python itself is not embedded in the
installer yet.

## Labelled camera evaluation

Confidence acceptance must be fitted from labelled camera clips, never from
the model's own unlabelled predictions. Create a manifest next to isolated
sign clips:

```json
{
  "clips": [
    { "path": "clips/bed-01.mp4", "label": "bed" },
    { "path": "clips/help-01.mp4", "label": "help" }
  ]
}
```

Run the clips through the same 10 FPS MediaPipe, motion-segmentation, TGCN,
and top-one path used live:

```powershell
.\.venv-wlasl\Scripts\python.exe .\backend\sign_inference\camera_evaluation.py `
  .\local-evaluation\manifest.json `
  --observations .\local-evaluation\observations.jsonl `
  --calibration .\backend\sign_inference\calibration\asl100-camera-v1.json `
  --dataset team-camera-v1
```

The calibration artifact records its schema, model/preprocessing contract,
dataset name, source-data SHA-256, global threshold, supported per-gloss
thresholds, and precision/recall counts. Re-running with the same labelled
observations produces the same artifact. Commit a calibration only together
with the dataset/version and measured report used to create it.

If a valid artifact is present at
`sign_inference/calibration/asl100-camera-v1.json`, the live recognizer loads
it once at startup. Without one it uses the explicit uncalibrated bootstrap
cutoff and emits a `sign_calibration: missing` event; it does not learn from
the session.

## Tests

```powershell
.\.venv-wlasl\Scripts\python.exe -m unittest discover -s .\backend\engine -p "test_*.py" -v
.\.venv-wlasl\Scripts\python.exe -m unittest discover -s .\backend\sign_inference -p "test_*.py" -v
.\.venv-wlasl\Scripts\python.exe -m unittest discover -s .\server -p "test_*.py" -v
cd frontend
npm run typecheck
npm run build
```

The engine suite includes a real subprocess JSON-protocol check, stuck-camera
shutdown protection, and an accepted-sign → phrase → speech contract test.
Phrase events report `assemblyLatencyMs`; speech replies report
`synthesisMs`, `playbackMs`, and `totalLatencyMs`.

## Verification status and limitations

Verified on 2026-10-21:

- 22 engine/sentence/protocol tests pass;
- 28 inference/calibration tests pass;
- 7 optional HTTP speech-service tests pass; and
- frontend type checking and production build pass.

These tests verify contracts and deterministic model behavior. They do not
measure real-camera recognition accuracy or hardware round-trip latency.
There are no labelled camera clips in the repository, so no production
calibration or honest live accuracy number is included yet. Capture the team
dataset and run the command above before claiming calibrated confidence.

The pinned static fixture proves checkpoint and preprocessing compatibility,
not webcam accuracy. Hardware latency must also be measured on the target
OBS/VB-CABLE or BlackHole setup; the emitted timing fields provide the
measurement points.
