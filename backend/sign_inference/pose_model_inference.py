"""Camera inference with our own pose classifiers (ml/two_word), chosen by name.

Same session shape as SignInference (start_sequence / process_frame /
finish_and_predict / reset / close), but the model is any checkpoint from
pose_models.py, and the result reports confidence per word plus diagnostics.

Frames are uint8 RGB images and timestamps are strictly increasing milliseconds.
"""
from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from inference_utils import MODEL_DIR, remap_landmarks
from pose_models import PoseModel

MIN_RECORDED_FRAMES = 8  # shorter than this is almost certainly a mis-press, not a sign


class PoseModelInference:
    """One instance per session; methods must be called serially."""

    def __init__(self, model_name: str = "meeting100", *, model_dir=MODEL_DIR,
                 device: str | None = None, hand_assignment: str = "wrist", top_k: int = 5):
        model_dir = Path(model_dir)
        missing = [str(p) for p in (model_dir / "pose_landmarker.task", model_dir / "hand_landmarker.task")
                   if not p.is_file()]
        if missing:
            raise FileNotFoundError("Missing MediaPipe models:\n  - " + "\n  - ".join(missing)
                                    + "\nRun backend/sign_inference/setup_assets.py, then retry.")
        self.model = PoseModel(model_name, device=device)
        self.hand_assignment = hand_assignment
        self.top_k = top_k
        self._pose = self._hands = None
        self._closed = False
        self._timestamp = -1
        self.reset()
        try:
            self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_dir / "pose_landmarker.task")),
                running_mode=vision.RunningMode.VIDEO, num_poses=1))
            self._hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_dir / "hand_landmarker.task")),
                running_mode=vision.RunningMode.VIDEO, num_hands=2))
        except Exception:
            self.close()
            raise

    # ---- session state -------------------------------------------------------

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Inference session is closed.")

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def captured_frame_count(self) -> int:
        return self._captured

    def reset(self):
        self._ensure_open()
        self._frames: list[np.ndarray] = []
        self._captured = 0
        self._recording = False
        self._previous = None
        self._seen = {"body": 0, "left_hand": 0, "right_hand": 0}

    def start_sequence(self):
        self._ensure_open()
        if self._recording:
            raise RuntimeError("Already recording; finish or reset first.")
        self.reset()
        self._recording = True

    # ---- per frame -------------------------------------------------------------

    def process_frame(self, rgb_frame, timestamp_ms):
        """Track one frame; buffer it while recording. Returns (55, 2) pixel points for overlays."""
        self._ensure_open()
        frame = np.asarray(rgb_frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) == 0:
            raise ValueError("Expected a nonempty uint8 RGB image shaped (H,W,3).")
        if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, (int, np.integer)):
            raise ValueError("Timestamp must be integer milliseconds.")
        timestamp_ms = int(timestamp_ms)
        if timestamp_ms < 0 or timestamp_ms <= self._timestamp:
            raise ValueError("Timestamps must be nonnegative and strictly increasing across the session.")
        self._timestamp = timestamp_ms

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
        pose = self._pose.detect_for_video(image, timestamp_ms)
        hands = self._hands.detect_for_video(image, timestamp_ms)
        points, _ = remap_landmarks(pose, hands, hand_assignment=self.hand_assignment)
        # Pixels, like the OpenPose training data (normalization needs equal x/y units).
        points *= np.array([frame.shape[1], frame.shape[0]], dtype=np.float32)

        if self._recording:
            self._captured += 1
            if pose.pose_landmarks:
                self._previous = points.copy()
                self._seen["body"] += 1
                self._seen["left_hand"] += int(np.isfinite(points[13:34]).all())
                self._seen["right_hand"] += int(np.isfinite(points[34:55]).all())
            # Same policy as the training data: frames before the first body
            # detection are skipped; later frames without a body repeat the last one.
            if self._previous is not None:
                self._frames.append(self._previous.copy())
        return points.copy()

    # ---- prediction --------------------------------------------------------------

    def finish_and_predict(self) -> dict:
        self._ensure_open()
        if not self._recording:
            raise RuntimeError("No active sequence; start recording first.")
        self._recording = False
        if not self._frames:
            raise ValueError("No body detected in the recording; make sure your upper body is in view.")
        if len(self._frames) < MIN_RECORDED_FRAMES:
            raise ValueError(f"Recording too short ({len(self._frames)} frames); hold the sign a bit longer.")

        predictions = self.model.predict(np.stack(self._frames), top_k=self.top_k)
        top = predictions[0]["confidence"]
        second = predictions[1]["confidence"] if len(predictions) > 1 else 0.0
        body = max(self._seen["body"], 1)
        return {
            "model": self.model.name,
            "model_file": str(self.model.path),
            "vocabulary_size": len(self.model.labels),
            "predictions": predictions,                 # [{"gloss", "confidence"}], best first
            "top_confidence": top,
            "margin": top - second,                     # how far ahead the best guess is
            "captured_frame_count": self._captured,
            "usable_frame_count": len(self._frames),
            "model_frame_count": self.model.num_frames,
            "detection_rate": {                          # share of body-detected frames with each part found
                "body": self._seen["body"] / max(self._captured, 1),
                "left_hand": self._seen["left_hand"] / body,
                "right_hand": self._seen["right_hand"] / body,
            },
            "hand_assignment": self.hand_assignment,
            "score_type": "softmax_probability",
        }

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._frames = []
        self._recording = False
        try:
            if self._pose is not None:
                self._pose.close()
        finally:
            if self._hands is not None:
                self._hands.close()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *args):
        self.close()
