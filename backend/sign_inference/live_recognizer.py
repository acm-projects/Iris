"""Non-blocking live sign segmentation and inference for the Iris engine."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SegmentUpdate:
    state: str
    completed: list[tuple[int, np.ndarray]] | None = None


class MotionSegmenter:
    """Split a landmark stream into signs using motion followed by a pause."""

    def __init__(
        self,
        *,
        start_motion: float = 0.006,
        stop_motion: float = 0.0025,
        start_frames: int = 2,
        quiet_frames: int = 5,
        min_frames: int = 10,
        max_frames: int = 50,
        pre_roll: int = 3,
    ):
        self.start_motion = start_motion
        self.stop_motion = stop_motion
        self.start_frames = start_frames
        self.quiet_frames = quiet_frames
        self.min_frames = min_frames
        self.max_frames = max_frames
        self._pre_roll = deque(maxlen=pre_roll)
        self._previous_normalized: np.ndarray | None = None
        self._active: list[tuple[int, np.ndarray]] | None = None
        self._moving = 0
        self._quiet = 0
        self._cooldown = 0

    @staticmethod
    def _normalized(points: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
        height, width = frame_shape[:2]
        scale = np.array([max(width, 1), max(height, 1)], dtype=np.float32)
        return np.asarray(points, dtype=np.float32) / scale

    @staticmethod
    def _body_visible(points: np.ndarray) -> bool:
        return int(np.isfinite(points[:13]).all(axis=1).sum()) >= 5

    def _motion(self, normalized: np.ndarray) -> float:
        previous = self._previous_normalized
        self._previous_normalized = normalized
        if previous is None:
            return 0.0
        valid = np.isfinite(previous).all(axis=1) & np.isfinite(normalized).all(axis=1)
        if int(valid.sum()) < 5:
            return 0.0
        # A one-handed sign moves fewer than half of the 55 joints, so median
        # motion would remain zero. The upper quartile catches that hand while
        # still ignoring a few noisy landmarks.
        distances = np.linalg.norm(normalized[valid] - previous[valid], axis=1)
        return float(np.percentile(distances, 75))

    def observe(
        self,
        timestamp_ms: int,
        points: np.ndarray,
        frame_shape: tuple[int, ...],
    ) -> SegmentUpdate:
        points = np.asarray(points, dtype=np.float32)
        visible = self._body_visible(points)
        motion = self._motion(self._normalized(points, frame_shape)) if visible else 0.0
        record = (int(timestamp_ms), points.copy())

        if self._active is None:
            if visible:
                self._pre_roll.append(record)
            if self._cooldown:
                self._cooldown -= 1
                return SegmentUpdate("listening")
            self._moving = self._moving + 1 if motion >= self.start_motion else 0
            if self._moving < self.start_frames:
                return SegmentUpdate("listening")
            self._active = list(self._pre_roll)
            self._moving = 0
            self._quiet = 0
            return SegmentUpdate("capturing")

        if visible:
            self._active.append(record)
        self._quiet = self._quiet + 1 if motion < self.stop_motion else 0
        enough = len(self._active) >= self.min_frames
        reached_pause = enough and self._quiet >= self.quiet_frames
        reached_limit = len(self._active) >= self.max_frames
        if not reached_pause and not reached_limit:
            return SegmentUpdate("capturing")

        completed = self._active
        if reached_pause and len(completed) - self._quiet >= self.min_frames:
            completed = completed[:-self._quiet]
        self._active = None
        self._pre_roll.clear()
        self._quiet = 0
        self._cooldown = 3
        return SegmentUpdate("predicting", completed)


class LiveSignRecognizer:
    """Own model state on a worker thread and keep only the newest video frame."""

    SAMPLE_INTERVAL_MS = 100
    BOOTSTRAP_MIN_SCORE = 0.55

    def __init__(self, emit: Callable[[dict], None]):
        self._emit = emit
        self._frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_sample_ms = -1
        self._last_state = ""

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="iris-sign-recognizer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._replace_frame(None)
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None

    def submit(self, frame_rgb: np.ndarray) -> None:
        if self.running:
            self._replace_frame(frame_rgb)

    def _replace_frame(self, frame: np.ndarray | None) -> None:
        try:
            self._frames.put_nowait(frame)
            return
        except queue.Full:
            pass
        try:
            self._frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            pass

    def _status(self, state: str, **extra) -> None:
        if state == self._last_state and not extra:
            return
        self._last_state = state
        self._emit({"event": "sign_status", "state": state, **extra})

    def _run(self) -> None:
        session = None
        try:
            # Imported here so engine health/TTS still work when sign extras are absent.
            from sign_inference import SignInference
            from sign_confidence.sign_confidence import SignConfidence

            session = SignInference(device="cpu")
            confidence = SignConfidence()
            segmenter = MotionSegmenter()
            self._status("listening")
            while not self._stop.is_set():
                try:
                    frame = self._frames.get(timeout=0.25)
                except queue.Empty:
                    continue
                if frame is None:
                    continue
                timestamp_ms = time.monotonic_ns() // 1_000_000
                if timestamp_ms - self._last_sample_ms < self.SAMPLE_INTERVAL_MS:
                    continue
                self._last_sample_ms = max(timestamp_ms, self._last_sample_ms + 1)
                session.process_frame(frame, self._last_sample_ms)
                points = session.last_model_points
                if points is None:
                    continue
                update = segmenter.observe(
                    self._last_sample_ms,
                    points,
                    (256, 256, 3),
                )
                self._status(update.state)
                if not update.completed:
                    continue
                result = session.predict_keypoints(update.completed)
                top = result["predictions"][0]
                evaluated = confidence.evaluate(top["gloss"], top["score"])
                calibrated = bool(evaluated["calibrated"])
                relative = evaluated["relative"]
                accepted = (
                    bool(evaluated["successful"])
                    if calibrated
                    else float(top["score"]) >= self.BOOTSTRAP_MIN_SCORE
                )
                self._emit(
                    {
                        "event": "sign",
                        "text": top["gloss"],
                        "score": float(top["score"]),
                        "confidence": float(relative if relative is not None else top["score"]),
                        "calibrated": calibrated,
                        "accepted": accepted,
                        "repeatNeeded": not accepted,
                        "candidates": result["predictions"],
                        "latencyMs": int((time.monotonic_ns() // 1_000_000) - self._last_sample_ms),
                    }
                )
                self._status("listening")
        except Exception as error:  # noqa: BLE001 - surface model failures to the app
            self._status("error", error=f"{type(error).__name__}: {error}")
        finally:
            if session is not None:
                session.close()
