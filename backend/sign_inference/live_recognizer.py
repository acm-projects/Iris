"""Continuous sign recognition for the app.

Two modes:

**Fixed window, started by your hands** (`window_frames=50`, what the app
uses): nothing is recorded until a hand comes up into signing space. Then
`window_frames` frames are recorded and scored as one sign, and the top word is
reported. For the next `buffer_frames` frames nothing can start, so lowering
your hands isn't read as a new sign; after that it waits for your hands again
(if they're still up, the next window starts straight away).

**Automatic segmentation** (`window_frames=None`): the camera demo uses R / S
keys to mark a sign; without keys this watches the hands instead:

  - a sign STARTS when a hand comes up into signing space (its wrist above
    roughly mid-chest);
  - it ENDS when the hands drop or disappear for `end_gap_ms`, or after
    `max_sign_ms` at the latest;
  - the recording is then scored, and reported only if the model is confident
    enough (`min_confidence`, `min_margin`), so idle movement doesn't turn
    into words.

Frames are uint8 RGB images with strictly increasing millisecond timestamps.
"""
from __future__ import annotations

import numpy as np

from pose_model_inference import PoseModelInference

NECK, R_SHOULDER, L_SHOULDER = 1, 2, 5
LEFT_HAND, RIGHT_HAND = slice(13, 34), slice(34, 55)


def hands_in_signing_space(points: np.ndarray) -> bool:
    """(55, 2) pixel points, NaN = missing -> is a detected hand raised into signing space?"""
    hands = [points[LEFT_HAND], points[RIGHT_HAND]]
    detected = [h for h in hands if np.isfinite(h).all()]
    if not detected:
        return False
    body = points[[NECK, R_SHOULDER, L_SHOULDER]]
    if not np.isfinite(body).all():
        return True  # no body to compare against: any visible hand counts
    shoulder_width = float(np.linalg.norm(body[1] - body[2]))
    if shoulder_width < 1:
        return True
    # Image y grows downward: "raised" = wrist above neck + 1.3 shoulder widths.
    limit = body[0, 1] + 1.3 * shoulder_width
    return any(h[0, 1] < limit for h in detected)


class LiveSignRecognizer:
    """Feeds every frame to MediaPipe and the model; returns a result when a sign finishes."""

    def __init__(self, model_name: str, *, min_confidence: float = 0.45, min_margin: float = 0.10,
                 end_gap_ms: int = 450, max_sign_ms: int = 3000, cooldown_ms: int = 300,
                 min_active_frames: int = 5, window_frames: int | None = None,
                 buffer_frames: int = 6, top_k: int = 3, allow_paths: bool = False,
                 session=None):
        """
        model_name:        which model to load (a plain name, see pose_models.py)
        min_confidence:    lowest top-word confidence (0-1) to mark a result "accepted"
        min_margin:        how far the top word must lead the second guess
        end_gap_ms:        segmentation mode: hands down this long ends a sign
        max_sign_ms:       segmentation mode: longest a single sign can last
        cooldown_ms:       segmentation mode: pause after a sign before the next can start
        min_active_frames: frames with hands raised needed for a real sign
        window_frames:     window mode: frames recorded per sign (None = segmentation mode)
        buffer_frames:     window mode: frames ignored after each sign while hands drop
        top_k:             how many guesses to return
        allow_paths:       False = only plain model names from the model folders
        session:           a ready-made PoseModelInference (tests pass a fake one)
        """
        # allow_paths=False: only plain model names from the model folders.
        self.session = session or PoseModelInference(model_name, top_k=top_k, allow_paths=allow_paths)
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.end_gap_ms = end_gap_ms
        self.max_sign_ms = max_sign_ms
        self.cooldown_ms = cooldown_ms
        # Frames with hands raised needed for a real sign (a quick twitch isn't one).
        self.min_active_frames = min_active_frames
        self._active_frames = 0
        # Fixed-window mode: score every `window_frames` frames (None = segment by hand movement).
        self.window_frames = window_frames
        self._window_count = 0
        # Frames to ignore after each window, while you lower your hands.
        self.buffer_frames = buffer_frames
        self._buffer_left = 0
        self._started_at = None    # timestamp the current sign started, or None
        self._last_active = -1     # last timestamp a hand was in signing space
        self._last_finish = -10**9

    @property
    def model(self):
        return self.session.model

    def process(self, rgb_frame, timestamp_ms: int) -> dict | None:
        """Track one frame. Returns a result dict when a sign has just finished, else None.

        The dict has "accepted" (confident enough to show), "text" (best word),
        "confidence", "margin" and "predictions" (top-k words with confidence).
        """
        if self.window_frames:
            return self._process_window(rgb_frame, timestamp_ms)
        points = self.session.process_frame(rgb_frame, timestamp_ms)
        active = hands_in_signing_space(points)
        if active:
            self._last_active = timestamp_ms

        if self._started_at is None:
            if active and timestamp_ms - self._last_finish >= self.cooldown_ms:
                self.session.start_sequence()
                self._started_at = timestamp_ms
                self._active_frames = 0
            return None

        if active:
            self._active_frames += 1

        ended = timestamp_ms - self._last_active >= self.end_gap_ms
        too_long = timestamp_ms - self._started_at >= self.max_sign_ms
        if ended or too_long:
            return self._finish(timestamp_ms)
        return None

    def _process_window(self, rgb_frame, timestamp_ms: int) -> dict | None:
        """Window mode: wait for raised hands, record `window_frames` frames, score, then buffer."""
        if self.session.recording:
            points = self.session.process_frame(rgb_frame, timestamp_ms)
            self._window_count += 1
            if hands_in_signing_space(points):
                self._active_frames += 1
            if self._window_count >= self.window_frames:
                self._buffer_left = self.buffer_frames
                return self._finish(timestamp_ms)
            return None

        # Not recording: keep tracking (MediaPipe follows the hands frame to frame).
        points = self.session.process_frame(rgb_frame, timestamp_ms)
        if self._buffer_left > 0:
            self._buffer_left -= 1  # still lowering hands after the last sign
            return None
        if hands_in_signing_space(points):
            # Hands are up: this frame starts the next sign.
            self.session.start_sequence()
            self._window_count = 0
            self._active_frames = 0
        return None

    def _finish(self, timestamp_ms: int) -> dict | None:
        """Score the recording that just ended; None if it was too short to be a sign."""
        self._started_at = None
        self._last_finish = timestamp_ms
        if self._active_frames < self.min_active_frames:
            self.session.reset()  # hands were only up for a moment: not a sign
            return None
        try:
            result = self.session.finish_and_predict()
        except ValueError:
            return None  # too short, or no body detected: not a sign
        top = result["predictions"][0]
        accepted = result["top_confidence"] >= self.min_confidence and result["margin"] >= self.min_margin
        return {
            "accepted": accepted,
            "text": top["gloss"],
            "confidence": result["top_confidence"],
            "margin": result["margin"],
            "predictions": result["predictions"],
            "frames": result["usable_frame_count"],
        }

    def reset(self):
        """Drop any recording in progress (MediaPipe keeps tracking)."""
        self._started_at = None
        self._buffer_left = 0
        if self.session.recording:
            self.session.reset()

    def close(self):
        """Release MediaPipe; the recognizer can't be used afterwards."""
        self.session.close()
