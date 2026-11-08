"""Continuous (rolling-window) sign recognition on top of PoseModelInference.

camera_inference.py classifies one clip at a time (R ... S). This keeps the last
WINDOW_FRAMES (60) frames of 55-joint keypoints in a rolling buffer instead and
re-classifies that window every few frames, so signs are picked up as they
happen and committed one after another into a running sentence.

Nothing about the model or its inputs changes: each window is handed to
PoseModel.predict as (60, 55, 2) pixel keypoints, exactly like a recording, and
the model samples its own frame count from it. What this file adds:

  RollingBuffer    the last 60 frames (+ whether a hand was seen in each)
  SentenceBuilder  smooths the per-window word probabilities and decides when a
                   word is confident and stable enough to append to the sentence
  ContinuousInference  glues them to a PoseModelInference session (watch it
                   live with continuous_camera.py); its
                   process() method is what the app's engine calls
                   (backend/engine/engine.py, SIGN_MODE = "continuous")
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from display_confidence import load_display_scale_words, scale_top_display
from pose_model_inference import PoseModelInference

WINDOW_FRAMES = 60  # rolling buffer length


class RollingBuffer:
    """Fixed-length FIFO of (55, 2) keypoint frames; the oldest frame drops out when full."""

    def __init__(self, maxlen: int = WINDOW_FRAMES):
        self.maxlen = maxlen
        self._frames: deque[np.ndarray] = deque(maxlen=maxlen)
        self._hands: deque[bool] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def full(self) -> bool:
        return len(self._frames) == self.maxlen

    def append(self, frame: np.ndarray, hands_visible: bool):
        self._frames.append(frame)
        self._hands.append(hands_visible)

    def window(self) -> np.ndarray:
        """Oldest-to-newest frames, shaped (len, 55, 2)."""
        return np.stack(self._frames)

    def hand_fraction(self, last: int | None = None) -> float:
        """Share of the (last N) buffered frames in which at least one hand was detected."""
        flags = list(self._hands)[-last:] if last else list(self._hands)
        return sum(flags) / len(flags) if flags else 0.0

    def clear(self):
        self._frames.clear()
        self._hands.clear()


@dataclass
class SentenceBuilder:
    """Turns a stream of per-window word probabilities into a sentence.

    A word is appended when, over the last `smooth` windows, its averaged
    probability is at least `min_conf` and it has also been the single best
    guess in each of the last `stable` windows. After a word is added the same
    word is not added again until the sign has visibly ended (hands dropped, its
    probability fell under `release_conf`, or another word was committed), so
    one slow sign that stays in the buffer for a whole window isn't repeated.
    """
    smooth: int = 5
    stable: int = 3
    min_conf: float = 0.45
    release_conf: float = 0.15
    max_words: int = 12
    words: list[str] = field(default_factory=list)
    sentences: list[list[str]] = field(default_factory=list)  # finished earlier sentences

    def __post_init__(self):
        self._history: deque[dict[str, float]] = deque(maxlen=self.smooth)
        self._last: str | None = None  # last committed word
        self._armed = True             # may `_last` be committed again?

    @property
    def text(self) -> str:
        return " ".join(self.words)

    def rest(self):
        """No signing right now: forget the old window's votes and allow repeats."""
        self._history.clear()
        self._armed = True

    def end_sentence(self):
        """Close the current sentence (a long pause) and start a fresh one."""
        if self.words:
            self.sentences.append(self.words)
        self.words = []
        self._last = None
        self.rest()

    def update(self, probs: dict[str, float]) -> str | None:
        """Feed one window's probabilities {gloss: p}; returns the word it committed, if any."""
        self._history.append(probs)
        n = len(self._history)
        avg: dict[str, float] = {}
        for p in self._history:
            for gloss, value in p.items():
                avg[gloss] = avg.get(gloss, 0.0) + value / n
        top = max(avg, key=avg.get)

        if self._last is not None and avg.get(self._last, 0.0) < self.release_conf:
            self._armed = True  # the committed sign has faded from the window
        if n < self.stable or avg[top] < self.min_conf:
            return None
        recent = list(self._history)[-self.stable:]
        if any(max(p, key=p.get) != top for p in recent):
            return None
        if top == self._last and not self._armed:
            return None

        self.words.append(top)
        del self.words[:-self.max_words]
        self._last, self._armed = top, False
        self._history.clear()  # the next word must earn its votes from fresh windows
        return top


@dataclass
class Update:
    """What happened on one frame."""
    points: np.ndarray                    # (55, 2) pixel landmarks for drawing
    buffered: int                         # frames currently in the buffer
    predicted: bool                       # a window was classified this frame
    guesses: list[dict]                   # top guesses for that window (empty if none)
    committed: str | None                 # word appended to the sentence this frame
    signing: bool                         # hands are active in the window


class ContinuousInference:
    """Rolling-window prediction over a PoseModelInference session.

    Call process_frame() for every camera frame (same arguments as
    PoseModelInference.process_frame). A window is classified every
    `predict_every` frames once the buffer is full and a hand has been seen in
    it; when hands are absent the model is not run, so resting never produces
    words.
    """

    def __init__(self, session: PoseModelInference, *, window: int = WINDOW_FRAMES,
                 predict_every: int = 3, min_hand_fraction: float = 0.3,
                 recent_hand_frames: int = 10, sentence_gap_frames: int = 90,
                 builder: SentenceBuilder | None = None):
        self.session = session
        self.buffer = RollingBuffer(window)
        self.builder = builder or SentenceBuilder()
        self.predict_every = predict_every
        self.min_hand_fraction = min_hand_fraction  # of the whole window
        self.recent_hand_frames = recent_hand_frames  # a hand must also be in the newest N frames
        self.sentence_gap_frames = sentence_gap_frames  # idle frames that end a sentence (~3 s at 30 fps)
        self._last_body: np.ndarray | None = None
        self._since_predict = 0
        self._idle = 0
        self.guesses: list[dict] = []
        self._probs: dict[str, float] = {}  # every word's probability for the latest window

    @classmethod
    def for_model(cls, model_name: str, *, top_k: int = 3, allow_paths: bool = False,
                  **options) -> "ContinuousInference":
        """Open a MediaPipe + model session by name and wrap it (used by the engine).

        allow_paths=False only loads plain model names from the model folders.
        """
        session = PoseModelInference(model_name, top_k=top_k, allow_paths=allow_paths)
        return cls(session, **options)

    @property
    def model(self):
        """The loaded PoseModel (labels, accuracy), like LiveSignRecognizer.model."""
        return self.session.model

    def process(self, rgb_frame, timestamp_ms) -> dict | None:
        """Engine interface (same result shape as LiveSignRecognizer.process).

        Returns a dict only on frames where a new word was added to the
        sentence: "accepted", "text", "confidence" (the model's real score),
        "display_confidence" (cosmetic, see display_confidence.py), "margin",
        "predictions" (top guesses for that window) and "frames" (buffer length).
        Words are added based on real scores only, never the display score.
        """
        update = self.process_frame(rgb_frame, timestamp_ms)
        if update.committed is None:
            return None
        confidence = self._probs.get(update.committed, 0.0)
        others = [p for g, p in self._probs.items() if g != update.committed]
        display = scale_top_display([{"gloss": update.committed, "confidence": confidence}],
                                    load_display_scale_words())[0]["display_confidence"]
        return {
            "accepted": True,  # SentenceBuilder already required a confident, stable word
            "text": update.committed,
            "confidence": confidence,
            "display_confidence": display,
            "margin": confidence - max(others, default=0.0),
            "predictions": update.guesses,
            "frames": update.buffered,
        }

    def close(self):
        """Release MediaPipe; this object can't be used afterwards."""
        self.session.close()

    def reset(self):
        self.buffer.clear()
        self.builder.end_sentence()
        self._last_body = None
        self._since_predict = self._idle = 0
        self.guesses = []
        self._probs = {}

    def process_frame(self, rgb_frame, timestamp_ms) -> Update:
        points = self.session.process_frame(rgb_frame, timestamp_ms)
        hands = bool(np.isfinite(points[13:34]).all() or np.isfinite(points[34:55]).all())

        # Same policy as recordings and the training data: skip frames until a
        # body has been seen, then repeat the last body frame while it's lost.
        if np.isfinite(points[:13]).any():
            self._last_body = points
        if self._last_body is None:
            return Update(points, 0, False, [], None, False)
        self.buffer.append((points if np.isfinite(points[:13]).any() else self._last_body).copy(), hands)

        signing = (self.buffer.hand_fraction() >= self.min_hand_fraction
                   and self.buffer.hand_fraction(self.recent_hand_frames) > 0)
        predicted, committed = False, None
        if signing:
            self._idle = 0
            self._since_predict += 1
            if self.buffer.full and self._since_predict >= self.predict_every:
                self._since_predict = 0
                predicted = True
                committed = self._classify()
        else:
            self._since_predict = 0
            self.guesses = []
            self.builder.rest()
            self._idle += 1
            if self._idle == self.sentence_gap_frames:
                self.builder.end_sentence()
        return Update(points, len(self.buffer), predicted, self.guesses, committed, signing)

    def _classify(self) -> str | None:
        model = self.session.model
        # Ask for the whole vocabulary: the builder averages probabilities across windows.
        # predict_with_word_rules applies word_rules.json (mirrored / disabled words).
        predict = getattr(model, "predict_with_word_rules", model.predict)
        self.guesses = predict(self.buffer.window(), top_k=len(model.labels))
        probs = {g["gloss"]: g["confidence"] for g in self.guesses}
        self._probs = probs
        self.guesses = self.guesses[:self.session.top_k]
        return self.builder.update(probs)
