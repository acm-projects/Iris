"""Continuous (rolling-window) sign recognition on top of PoseModelInference.

camera_inference.py classifies one clip at a time (R ... S). This keeps the last
60 frames of 55-joint keypoints in a rolling buffer instead and re-classifies
that window every few frames, so signs are picked up as they happen and
committed one after another into a running sentence.

Nothing about the model or its inputs changes: each window is handed to
PoseModel.predict as (60, 55, 2) pixel keypoints, exactly like a recording, and
the model samples its own frame count from it. What this file adds:

  RollingBuffer    the last 60 frames (+ whether a hand was seen in each)
  SentenceBuilder  smooths the per-window word probabilities and decides when a
                   word is confident and stable enough to append to the sentence
  ContinuousInference  glues them to a PoseModelInference session

  python continuous_camera.py --model meeting100 --landmarks
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

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
    one slow sign that stays in the buffer for 50 frames isn't repeated.
    """
    smooth: int = 3
    stable: int = 2
    min_conf: float = 0.30
    release_conf: float = 0.15
    max_words: int | None = None  # no silent truncation of a manually submitted phrase
    words: list[str] = field(default_factory=list)
    sentences: list[list[str]] = field(default_factory=list)  # finished earlier sentences

    def __post_init__(self):
        if not 1 <= self.stable <= self.smooth:
            raise ValueError('Require 1 <= stable <= smooth.')
        if not 0 <= self.release_conf < self.min_conf <= 1:
            raise ValueError('Require 0 <= release_conf < min_conf <= 1.')
        self._history: deque[dict[str, float]] = deque(maxlen=self.smooth)
        self._last: str | None = None  # last committed word
        self._armed = True             # may `_last` be committed again?
        self.status = 'Waiting for predictions'

    @property
    def text(self) -> str:
        return " ".join(self.words)

    def rest(self):
        """No signing right now: forget the old window's votes and allow repeats."""
        self._history.clear()
        self._armed = True
        self.status = 'Resting: waiting for signing'

    def end_sentence(self):
        """Close the current sentence (a long pause) and start a fresh one."""
        if self.words:
            self.sentences.append(self.words)
        self.words = []
        self._last = None
        self.rest()

    def update(self, probs: dict[str, float]) -> str | None:
        """Feed one window's probabilities {gloss: p}; returns the word it committed, if any."""
        if not probs:
            self.status = 'No predictions'
            return None
        self._history.append(probs)
        n = len(self._history)
        avg: dict[str, float] = {}
        for p in self._history:
            for gloss, value in p.items():
                avg[gloss] = avg.get(gloss, 0.0) + value / n
        top = max(avg, key=avg.get)

        if self._last is not None and avg.get(self._last, 0.0) < self.release_conf:
            self._armed = True  # the committed sign has faded from the window
        if n < self.stable:
            self.status = f'{top}: collecting {n}/{self.stable} windows'
            return None
        if avg[top] < self.min_conf:
            self.status = f'{top}: average {avg[top]:.0%} below {self.min_conf:.0%}'
            return None
        recent = list(self._history)[-self.stable:]
        if any(max(p, key=p.get) != top for p in recent):
            self.status = f'{top}: waiting for {self.stable} consecutive wins'
            return None
        if top == self._last and not self._armed:
            self.status = f'{top}: duplicate held until release or another sign'
            return None

        self.words.append(top)
        if self.max_words is not None:
            del self.words[:-self.max_words]
        self._last, self._armed = top, False
        self._history.clear()  # the next word must earn its votes from fresh windows
        self.status = f'Accepted: {top}'
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
    """Rolling landmark-window prediction over a PoseModelInference session.

    Call process_frame() for every camera frame (same arguments as
    PoseModelInference.process_frame). A window is classified every
    `predict_every` frames once the buffer is full and a hand has been seen in
    it; when hands are absent the model is not run, so resting never produces
    words.
    """

    def __init__(self, session: PoseModelInference, *, window: int = WINDOW_FRAMES,
                 predict_every: int = 3, min_hand_fraction: float = 0.3,
                 recent_hand_frames: int = 10, sentence_gap_frames: int | None = None,
                 builder: SentenceBuilder | None = None):
        self.session = session
        self.buffer = RollingBuffer(window)
        self.builder = builder or SentenceBuilder()
        self.predict_every = predict_every
        self.min_hand_fraction = min_hand_fraction  # of the whole window
        self.recent_hand_frames = recent_hand_frames  # a hand must also be in the newest N frames
        self.sentence_gap_frames = sentence_gap_frames  # optional archival; manual submit by default
        self._last_body: np.ndarray | None = None
        self._since_predict = 0
        self._idle = 0
        self.guesses: list[dict] = []

    def reset(self):
        self.buffer.clear()
        self.builder.end_sentence()
        self._last_body = None
        self._since_predict = self._idle = 0
        self.guesses = []

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
        self.guesses = sorted(model.predict(self.buffer.window(), top_k=len(model.labels)),
                              key=lambda g: g["confidence"], reverse=True)
        probs = {g["gloss"]: g["confidence"] for g in self.guesses}
        self.guesses = self.guesses[:self.session.top_k]
        return self.builder.update(probs)
