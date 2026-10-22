"""Non-blocking live sign segmentation and inference for the Iris engine."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

LEFT_HAND = slice(13, 34)
RIGHT_HAND = slice(34, 55)
MODEL_POINT_SHAPE = (55, 2)
MIN_INFERENCE_HAND_FRAMES = 10
MIN_INFERENCE_HAND_RATIO = 0.8
DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parent / "calibration" / "asl100-camera-v1.json"
)


def has_detected_hand(points: np.ndarray) -> bool:
    """Whether MediaPipe supplied one complete 21-landmark hand."""
    points = np.asarray(points)
    if points.shape != MODEL_POINT_SHAPE:
        return False
    finite = np.isfinite(points).all(axis=1)
    return bool(finite[LEFT_HAND].all() or finite[RIGHT_HAND].all())


def segment_has_sufficient_hands(
    records: list[tuple[int, np.ndarray]],
    *,
    min_frames: int = MIN_INFERENCE_HAND_FRAMES,
    min_ratio: float = MIN_INFERENCE_HAND_RATIO,
) -> bool:
    """Reject sequences that do not contain enough hand-visible samples."""
    if not records:
        return False
    visible = sum(has_detected_hand(points) for _, points in records)
    return visible >= min_frames and visible / len(records) >= min_ratio


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
        hand_loss_grace_frames: int = 2,
    ):
        self.start_motion = start_motion
        self.stop_motion = stop_motion
        self.start_frames = start_frames
        self.quiet_frames = quiet_frames
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.hand_loss_grace_frames = hand_loss_grace_frames
        self._pre_roll = deque(maxlen=pre_roll)
        self._previous_normalized: np.ndarray | None = None
        self._active: list[tuple[int, np.ndarray]] | None = None
        self._moving = 0
        self._quiet = 0
        self._missing_hand = 0
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
        hand_valid = valid.copy()
        hand_valid[:LEFT_HAND.start] = False
        if int(hand_valid.sum()) < 21:
            return 0.0
        # The upper quartile catches motion in a subset of finger joints while
        # ignoring a few noisy points. Measure only hands so body motion cannot
        # begin or extend a sign segment.
        distances = np.linalg.norm(
            normalized[hand_valid] - previous[hand_valid],
            axis=1,
        )
        return float(np.percentile(distances, 75))

    def _reset_active(self) -> None:
        self._active = None
        self._pre_roll.clear()
        self._moving = 0
        self._quiet = 0
        self._missing_hand = 0

    def _finish(self, *, trim_quiet: bool = False) -> SegmentUpdate:
        completed = self._active or []
        if (
            trim_quiet
            and self._quiet
            and len(completed) - self._quiet >= self.min_frames
        ):
            completed = completed[:-self._quiet]
        self._reset_active()
        self._cooldown = 3
        return SegmentUpdate("predicting", completed)

    def flush(self) -> SegmentUpdate:
        """Finish a sufficiently long active segment at an input boundary."""
        if self._active is not None and len(self._active) >= self.min_frames:
            return self._finish()
        self._reset_active()
        return SegmentUpdate("listening")

    def observe(
        self,
        timestamp_ms: int,
        points: np.ndarray,
        frame_shape: tuple[int, ...],
    ) -> SegmentUpdate:
        points = np.asarray(points, dtype=np.float32)
        visible = self._body_visible(points)
        hand_visible = visible and has_detected_hand(points)
        motion = (
            self._motion(self._normalized(points, frame_shape))
            if hand_visible
            else 0.0
        )
        record = (int(timestamp_ms), points.copy())

        if self._active is None:
            if not hand_visible:
                self._pre_roll.clear()
                self._previous_normalized = None
                self._moving = 0
                if self._cooldown:
                    self._cooldown -= 1
                return SegmentUpdate("listening")
            if hand_visible:
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
            self._missing_hand = 0
            return SegmentUpdate("capturing")

        if not hand_visible:
            self._missing_hand += 1
            if self._missing_hand <= self.hand_loss_grace_frames:
                return SegmentUpdate("capturing")
            if len(self._active) >= self.min_frames:
                return self._finish()
            self._reset_active()
            return SegmentUpdate("listening")

        self._missing_hand = 0
        self._active.append(record)
        self._quiet = self._quiet + 1 if motion < self.stop_motion else 0
        enough = len(self._active) >= self.min_frames
        reached_pause = enough and self._quiet >= self.quiet_frames
        reached_limit = len(self._active) >= self.max_frames
        if not reached_pause and not reached_limit:
            return SegmentUpdate("capturing")

        return self._finish(trim_quiet=reached_pause)


class LiveSignRecognizer:
    """Own model state on a worker thread and keep only the newest video frame."""

    SAMPLE_INTERVAL_MS = 100
    BOOTSTRAP_MIN_SCORE = 0.55

    def __init__(
        self,
        emit: Callable[[dict], None],
        *,
        calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
    ):
        self._emit = emit
        self._calibration_path = Path(calibration_path)
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
            from confidence_calibration import ConfidenceCalibration

            session = SignInference(device="cpu")
            calibration = None
            if self._calibration_path.is_file():
                calibration = ConfidenceCalibration.load(self._calibration_path)
                if calibration.preprocessing != session.preprocessing:
                    raise ValueError(
                        "confidence calibration preprocessing does not match the model"
                    )
                self._emit(
                    {
                        "event": "sign_calibration",
                        "state": "ready",
                        "dataset": calibration.dataset,
                        "sourceSha256": calibration.source_sha256,
                    }
                )
            else:
                self._emit(
                    {
                        "event": "sign_calibration",
                        "state": "missing",
                        "path": str(self._calibration_path),
                    }
                )
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
                if not segment_has_sufficient_hands(update.completed):
                    self._status("listening")
                    continue
                result = session.predict_keypoints(update.completed)
                top = result["predictions"][0]
                raw_score = float(top["score"])
                evaluated = (
                    calibration.evaluate(top["gloss"], raw_score)
                    if calibration is not None
                    else {
                        "accepted": raw_score >= self.BOOTSTRAP_MIN_SCORE,
                        "threshold": self.BOOTSTRAP_MIN_SCORE,
                        "thresholdSource": "bootstrap",
                    }
                )
                calibrated = calibration is not None
                accepted = bool(evaluated["accepted"])
                self._emit(
                    {
                        "event": "sign",
                        "text": top["gloss"],
                        "score": raw_score,
                        "confidence": raw_score,
                        "calibrated": calibrated,
                        "accepted": accepted,
                        "repeatNeeded": not accepted,
                        "threshold": float(evaluated["threshold"]),
                        "thresholdSource": evaluated["thresholdSource"],
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
