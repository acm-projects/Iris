"""Iris translation engine.

A long-running helper process that Electron starts (see frontend/src/electron
/main.ts). It owns the two virtual devices Google Meet uses:

  * Virtual camera  - frames are written with pyvirtualcam into the OBS
                      Virtual Camera driver. In Meet: Camera -> "OBS Virtual
                      Camera".
  * Virtual microphone - audio is played with sounddevice into a loopback
                      driver (BlackHole 2ch on macOS, VB-CABLE on Windows).
                      In Meet: Microphone -> "BlackHole 2ch" / "CABLE Output".

The Iris app owns the real camera. It draws the preview and caption overlay
and streams JPEG frames here; this process forwards them to the virtual
camera and is where sign recognition will run (see `recognize_sign`).

Protocol (one JSON object per line, same as server/main.py):
  stdin  request : {"id": "1", "method": "health", "params": {...}}
  stdout reply   : {"id": "1", "result": {...}}  or  {"id": "1", "error": "..."}
  stdin  frame   : {"method": "frame", "data": "<base64 JPEG>"}   (no reply)
  stdout event   : {"event": "status", ...}                         (unsolicited)
Diagnostics go to stderr so they never mix with replies.

Run manually:  python3 backend/engine/engine.py
Then type:     {"id": "1", "method": "system_check"}
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

# The engine is executed as a script from backend/engine. Add the repository
# roots needed by the sibling sign-inference and confidence modules. The same
# layout is preserved under Electron's packaged resources directory.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_ROOT.parent
SIGN_INFERENCE_ROOT = BACKEND_ROOT / "sign_inference"
for import_root in (str(SIGN_INFERENCE_ROOT), str(REPO_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

# Third-party modules are imported lazily so `health` and `system_check`
# still work (and can report what is missing) before `pip install`.
REQUIRED_MODULES = {
    "numpy": "numpy",
    "cv2": "opencv-contrib-python",
    "pyvirtualcam": "pyvirtualcam",
    "sounddevice": "sounddevice",
    "torch": "torch",
    "mediapipe": "mediapipe",
}

# Substrings that identify loopback audio drivers usable as a virtual mic.
VIRTUAL_MIC_HINTS = ("blackhole", "cable input", "vb-audio", "loopback")

_print_lock = threading.Lock()


def send(message: dict) -> None:
    """Write one JSON message to stdout (thread-safe)."""
    with _print_lock:
        sys.stdout.write(json.dumps(message) + "\n")
        sys.stdout.flush()


def log(message: str) -> None:
    print(f"[iris-engine] {message}", file=sys.stderr, flush=True)


def module_status() -> dict:
    """{module: True | "error message"} for each required package."""
    status = {}
    for module in REQUIRED_MODULES:
        try:
            status[module] = True if importlib.util.find_spec(module) else "not installed"
        except Exception as error:  # noqa: BLE001 - report any import failure
            status[module] = f"{type(error).__name__}: {error}"
    return status


# ---------------------------------------------------------------------------
# Sign recognition worker
# ---------------------------------------------------------------------------

_recognizer = None
_phrase_worker = None
_recognizer_lock = threading.Lock()


def _handle_recognition_event(message: dict) -> None:
    """Forward recognition feedback and assemble only accepted glosses."""
    send(message)
    if message.get("event") != "sign" or not message.get("accepted"):
        return
    worker = _phrase_worker
    text = message.get("text")
    if worker is not None and isinstance(text, str):
        worker.submit(text)


def sign_model_status() -> dict:
    """Report whether the reviewed sign model and runtime assets are available."""
    missing_modules = []
    for module in ("torch", "mediapipe"):
        try:
            available = importlib.util.find_spec(module) is not None
        except Exception:
            available = False
        if not available:
            missing_modules.append(module)
    required_assets = (
        SIGN_INFERENCE_ROOT / "checkpoints" / "asl100" / "pytorch_model.bin",
        SIGN_INFERENCE_ROOT / "mediapipe_models" / "pose_landmarker.task",
        SIGN_INFERENCE_ROOT / "mediapipe_models" / "hand_landmarker.task",
        SIGN_INFERENCE_ROOT / "asl100.json",
    )
    missing_assets = [path.name for path in required_assets if not path.is_file()]
    if missing_modules or missing_assets:
        return {
            "ok": False,
            "status": "missing",
            "missingModules": missing_modules,
            "missingAssets": missing_assets,
        }
    try:
        from checkpoint_compatibility import ASL100_AUDIT, require_compatible

        require_compatible(ASL100_AUDIT)
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "status": "blocked", "error": str(error)}
    return {"ok": True, "status": "ready", "model": "Pose-TGCN WLASL-100"}


def start_sign_recognition() -> None:
    global _phrase_worker, _recognizer
    with _recognizer_lock:
        if _recognizer is not None and _recognizer.running:
            return
        try:
            from live_recognizer import LiveSignRecognizer
            from sentence_assembly import PhraseAssemblyWorker

            _phrase_worker = PhraseAssemblyWorker(send)
            _phrase_worker.start()
            _recognizer = LiveSignRecognizer(_handle_recognition_event)
            _recognizer.start()
        except Exception as error:  # noqa: BLE001
            if _phrase_worker is not None:
                _phrase_worker.stop()
            _phrase_worker = None
            _recognizer = None
            log(f"sign recognizer could not start: {error}")
            send({"event": "sign_status", "state": "error", "error": str(error)})


def stop_sign_recognition() -> None:
    global _phrase_worker, _recognizer
    with _recognizer_lock:
        if _recognizer is not None:
            _recognizer.stop()
        if _phrase_worker is not None:
            _phrase_worker.stop()
        _recognizer = None
        _phrase_worker = None


def recognize_sign(frame_rgb) -> None:
    """Submit the newest frame without blocking virtual-camera delivery."""
    worker = _recognizer
    if worker is not None:
        worker.submit(frame_rgb)


def finalize_phrase() -> dict:
    """Ask the phrase worker to finish the current sentence immediately."""
    worker = _phrase_worker
    return {"queued": bool(worker is not None and worker.finalize())}


def clear_phrase() -> dict:
    """Discard the current sentence without speaking it."""
    worker = _phrase_worker
    return {"queued": bool(worker is not None and worker.clear())}


# ---------------------------------------------------------------------------
# Virtual camera
# ---------------------------------------------------------------------------


class VirtualCameraStream:
    """Sends the newest frame from the app to the virtual camera at a fixed FPS."""

    def __init__(self) -> None:
        self._latest = None  # newest RGB frame (numpy array)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.device: str | None = None
        self.size = (1280, 720)
        self.fps = 24
        self.frames_in = 0
        self.frames_out = 0
        self.last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, width: int, height: int, fps: int) -> dict:
        if self.running:
            return {
                "device": self.device,
                "width": self.size[0],
                "height": self.size[1],
                "starting": self.device is None,
            }
        self.size = (int(width), int(height))
        self.fps = max(5, min(int(fps), 30))
        self._stop.clear()
        self.last_error = None
        self.device = None
        # Opening the OBS backend can block inside native code for longer than
        # Electron's request timeout. Delay it just enough for this request to
        # reply; readiness or failure is delivered as a status event.
        self._thread = threading.Timer(0.1, self._run)
        self._thread.daemon = True
        self._thread.start()
        return {
            "device": self.device,
            "width": self.size[0],
            "height": self.size[1],
            "starting": True,
        }

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None
        self.device = None
        stop_sign_recognition()

    def push_jpeg(self, data_b64: str) -> None:
        """Decode a JPEG from the app and keep it as the newest frame."""
        import cv2
        import numpy as np

        raw = np.frombuffer(base64.b64decode(data_b64), dtype=np.uint8)
        frame_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.size:
            frame_bgr = cv2.resize(frame_bgr, self.size)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            self._latest = frame_rgb
        self.frames_in += 1
        recognize_sign(frame_rgb)

    def _placeholder(self):
        """Deep-navy frame shown before the first real frame arrives."""
        import cv2
        import numpy as np

        width, height = self.size
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (0, 34, 78)  # Deep Navy #00224E (RGB)
        cv2.putText(frame, "Iris is starting...", (width // 2 - 200, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230, 237, 247), 2, cv2.LINE_AA)
        return frame

    def _run(self) -> None:
        import pyvirtualcam

        try:
            with pyvirtualcam.Camera(width=self.size[0], height=self.size[1], fps=self.fps) as cam:
                self.device = cam.device
                placeholder = self._placeholder()
                # The OBS backend can construct a camera object while another
                # producer still owns its output, then fail on the first send.
                # Verify one frame before telling Electron that startup worked.
                cam.send(placeholder)
                self.frames_out += 1
                log(f"virtual camera started on {cam.device}")
                send({"event": "status", "camera": "running", "device": self.device,
                      "framesIn": self.frames_in, "framesOut": self.frames_out})
                # Model imports and initialization happen on their own worker;
                # they must never delay camera startup or frame delivery.
                threading.Thread(target=start_sign_recognition, daemon=True).start()
                last_status = 0.0
                while not self._stop.is_set():
                    with self._lock:
                        frame = self._latest if self._latest is not None else placeholder
                    cam.send(frame)
                    self.frames_out += 1
                    cam.sleep_until_next_frame()
                    if time.time() - last_status > 2:
                        last_status = time.time()
                        send({"event": "status", "camera": "running", "device": self.device,
                              "framesIn": self.frames_in, "framesOut": self.frames_out})
        except Exception as error:  # noqa: BLE001
            self.last_error = f"Could not open the virtual camera: {error}"
            log(self.last_error)
            send({"event": "status", "camera": "error", "error": self.last_error})
        finally:
            with self._lock:
                self._latest = None
            stop_sign_recognition()
            send({"event": "status", "camera": "stopped"})


camera = VirtualCameraStream()


# ---------------------------------------------------------------------------
# Virtual microphone (speech and test sounds)
# ---------------------------------------------------------------------------


_audio_lock = threading.Lock()


def refresh_audio_devices() -> None:
    """Re-read the OS audio device list.

    PortAudio (used by sounddevice) only lists devices once, when it starts,
    so drivers installed while the engine is running (e.g. BlackHole) would
    stay invisible. Restarting PortAudio picks them up. Skipped while audio
    is playing so speech is never cut off.
    """
    import sounddevice as sd

    if not _audio_lock.acquire(blocking=False):
        return
    try:
        sd._terminate()
        sd._initialize()
    except Exception as error:  # noqa: BLE001 - keep the old list if this fails
        log(f"could not refresh audio devices: {error}")
    finally:
        _audio_lock.release()


def find_output_device(name: str | None) -> tuple[int, str]:
    """Find an output device by (partial) name; defaults to a loopback driver.

    On Windows each device is listed once per audio API (MME, DirectSound,
    WASAPI, WDM-KS). MME is tried first because it accepts any sample rate;
    its names are cut to 31 characters (e.g. "CABLE Input (VB-Audio Virtual C"),
    so matching is done on the part before " (".
    """
    import sounddevice as sd

    refresh_audio_devices()
    devices = sd.query_devices()
    outputs = [(i, d["name"]) for i, d in enumerate(devices) if d["max_output_channels"] > 0]
    if platform.system() == "Windows":
        apis = sd.query_hostapis()
        mme = [i for i, api in enumerate(apis) if api["name"] == "MME"]
        outputs.sort(key=lambda item: 0 if devices[item[0]]["hostapi"] in mme else 1)
    wanted = (name or "").lower().split(" (")[0].strip()
    for index, device_name in outputs:
        lower = device_name.lower()
        if wanted and (wanted in lower or lower in wanted):
            return index, device_name
    for index, device_name in outputs:
        if any(hint in device_name.lower() for hint in VIRTUAL_MIC_HINTS):
            return index, device_name
    raise RuntimeError(
        "No virtual microphone found. Install BlackHole 2ch (macOS) or VB-CABLE (Windows)."
    )


def synthesize_speech(text: str) -> tuple:
    """Turn text into (float32 samples, sample_rate) with the OS voice."""
    import numpy as np

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "speech.wav"
        system = platform.system()
        if system == "Darwin":
            subprocess.run(["say", "-o", str(path), "--data-format=LEI16@48000", text],
                           check=True, timeout=60)
        elif system == "Windows":
            # The text goes in on stdin (as UTF-8) so quotes and accents are safe;
            # single quotes in the file path are doubled for PowerShell.
            safe_path = str(path).replace("'", "''")
            script = (
                "[Console]::InputEncoding = [System.Text.Encoding]::UTF8;"
                "Add-Type -AssemblyName System.Speech;"
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                f"$s.SetOutputToWaveFile('{safe_path}');"
                "$s.Speak([Console]::In.ReadToEnd());$s.Dispose()"
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", script],
                           input=text, text=True, encoding="utf-8", check=True, timeout=60)
        else:
            raise RuntimeError("Text-to-speech is only set up for macOS and Windows.")
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        audio = samples.astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels)
        return audio, rate


def resample(audio, rate: int, target: int):
    """Linear resampling, enough for speech (e.g. Windows' 22.05 kHz voice -> 48 kHz)."""
    import numpy as np

    if rate == target or len(audio) == 0:
        return audio
    length = int(round(len(audio) * target / rate))
    old = np.linspace(0, 1, len(audio), endpoint=False)
    new = np.linspace(0, 1, length, endpoint=False)
    if audio.ndim == 1:
        return np.interp(new, old, audio).astype(np.float32)
    return np.stack([np.interp(new, old, audio[:, c]) for c in range(audio.shape[1])], axis=1).astype(np.float32)


def play(audio, rate: int, device_name: str | None) -> dict:
    import sounddevice as sd

    index, resolved = find_output_device(device_name)
    # Play at the device's own rate; some Windows audio APIs reject others.
    device_rate = int(sd.query_devices(index)["default_samplerate"] or rate)
    audio, rate = resample(audio, rate, device_rate), device_rate
    # Hold the lock while playing so a device refresh can't interrupt it.
    with _audio_lock:
        sd.play(audio, rate, device=index)
        sd.wait()
    return {"device": resolved, "durationMs": int(len(audio) / rate * 1000)}


def test_tone() -> tuple:
    """A short, soft two-note chime."""
    import numpy as np

    rate = 48000
    notes = []
    for frequency in (660, 880):
        t = np.linspace(0, 0.35, int(rate * 0.35), endpoint=False)
        envelope = np.minimum(1, t * 40) * np.exp(-t * 6)
        notes.append(0.25 * np.sin(2 * np.pi * frequency * t) * envelope)
    return np.concatenate(notes).astype(np.float32), rate


# ---------------------------------------------------------------------------
# System check
# ---------------------------------------------------------------------------


def system_check() -> dict:
    modules = module_status()
    result: dict = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "modules": modules,
        "missingPackages": [REQUIRED_MODULES[m] for m, ok in modules.items() if ok is not True],
    }

    # Do not open the driver as a readiness probe. Some Windows virtual-camera
    # backends block indefinitely when OBS is installed but its driver is not
    # ready. Electron also enumerates the device; the real open happens when
    # the user starts translation and already has an 8-second guarded path.
    if modules.get("pyvirtualcam") is True:
        if camera.running:
            result["virtualCamera"] = {"ok": True, "device": camera.device, "inUse": True}
        else:
            result["virtualCamera"] = {
                "ok": False,
                "status": "device-enumeration-required",
                "error": "Virtual camera runtime installed; waiting for the OBS Virtual Camera device.",
            }
    else:
        result["virtualCamera"] = {"ok": False, "error": "pyvirtualcam is not installed."}

    # Virtual microphone: look for a loopback output device.
    if modules.get("sounddevice") is True:
        try:
            _, name = find_output_device(None)
            result["virtualMic"] = {"ok": True, "device": name}
        except Exception as error:  # noqa: BLE001
            result["virtualMic"] = {"ok": False, "error": str(error)}
    else:
        result["virtualMic"] = {"ok": False, "error": "sounddevice is not installed."}

    # Text-to-speech uses the voice built into the OS.
    system = platform.system()
    tts_ok = (system == "Darwin" and shutil.which("say") is not None) or system == "Windows"
    result["speech"] = {"ok": tts_ok, "engine": "macOS say" if system == "Darwin" else "Windows SAPI"}

    result["signModel"] = sign_model_status()
    return result


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------


def handle(method: str, params: dict):
    if method == "health":
        return {"status": "ok", "service": "iris-engine", "python": platform.python_version()}
    if method == "system_check":
        return system_check()
    if method == "start_virtual_camera":
        return camera.start(params.get("width", 1280), params.get("height", 720), params.get("fps", 24))
    if method == "stop_virtual_camera":
        camera.stop()
        return {"stopped": True}
    if method == "finalize_phrase":
        return finalize_phrase()
    if method == "clear_phrase":
        return clear_phrase()
    if method == "speak":
        text = str(params.get("text", "")).strip()[:500]
        if not text:
            raise ValueError("Nothing to say.")
        audio, rate = synthesize_speech(text)
        return play(audio, rate, params.get("device"))
    if method == "test_tone":
        audio, rate = test_tone()
        return play(audio, rate, params.get("device"))
    raise ValueError(f"Unknown method: {method}")


def respond(request: dict) -> None:
    """Run one request (in its own thread so slow speech never blocks frames)."""
    request_id = request.get("id")
    try:
        send({"id": request_id, "result": handle(request.get("method", ""), request.get("params") or {})})
    except Exception as error:  # noqa: BLE001
        send({"id": request_id, "error": str(error)})


def main() -> None:
    log("ready")
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            send({"id": None, "error": "Invalid JSON."})
            continue
        if request.get("method") == "frame":
            # Frames are fire-and-forget; drop any that fail to decode.
            try:
                camera.push_jpeg(request.get("data", ""))
            except Exception as error:  # noqa: BLE001
                log(f"frame dropped: {error}")
            continue
        if not isinstance(request.get("id"), str):
            send({"id": None, "error": "Request requires a string ID."})
            continue
        threading.Thread(target=respond, args=(request,), daemon=True).start()
    camera.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
