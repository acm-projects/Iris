"""Tests for continuous_inference.py: the rolling buffer, when words are added
to the sentence, and the process() interface the engine uses.

A FakeSession stands in for PoseModelInference (no MediaPipe or model), so
these run in a fraction of a second:

    cd backend/sign_inference && ../.venv/bin/python -m unittest test_continuous_inference
"""
import unittest

import numpy as np

from continuous_inference import ContinuousInference, RollingBuffer, SentenceBuilder

FRAME = np.zeros((360, 640, 3), np.uint8)  # the fake session ignores pixels


def points(hands: bool = True) -> np.ndarray:
    """(55, 2) landmarks: a body always, the left hand only if `hands`."""
    p = np.full((55, 2), np.nan, np.float32)
    p[:13] = 100.0
    if hands:
        p[13:34] = 200.0
    return p


class FakeModel:
    """Returns whatever probabilities the test sets in `.probs`."""

    labels = ["hello", "yes", "no"]

    def __init__(self):
        self.probs = {"hello": 0.9, "yes": 0.07, "no": 0.03}
        self.calls = 0

    def predict(self, frames, top_k=5):
        self.calls += 1
        ranked = sorted(self.probs.items(), key=lambda kv: -kv[1])
        return [{"gloss": g, "confidence": c} for g, c in ranked[:top_k]]


class FakeSession:
    """Stands in for PoseModelInference: hands-up or hands-down points on demand."""

    def __init__(self):
        self.model = FakeModel()
        self.top_k = 3
        self.hands = True
        self.closed = False

    def process_frame(self, rgb, ts):
        return points(self.hands)

    def close(self):
        self.closed = True


def feed(ci: ContinuousInference, n: int, start: int = 0) -> list[dict]:
    """Feed n frames through process(); return the results that weren't None."""
    return [r for i in range(n) if (r := ci.process(FRAME, (start + i) * 40))]


class RollingBufferTests(unittest.TestCase):
    def test_keeps_only_the_newest_frames(self):
        buf = RollingBuffer(maxlen=3)
        for i in range(5):
            buf.append(np.full((55, 2), i, np.float32), hands_visible=i % 2 == 0)
        self.assertTrue(buf.full)
        self.assertEqual(buf.window()[:, 0, 0].tolist(), [2, 3, 4])
        self.assertAlmostEqual(buf.hand_fraction(), 2 / 3)


class SentenceBuilderTests(unittest.TestCase):
    def test_word_needs_several_stable_windows(self):
        b = SentenceBuilder(smooth=5, stable=3, min_conf=0.45)
        p = {"hello": 0.8, "yes": 0.2}
        self.assertIsNone(b.update(p))
        self.assertIsNone(b.update(p))
        self.assertEqual(b.update(p), "hello")
        self.assertEqual(b.text, "hello")

    def test_low_confidence_never_commits(self):
        b = SentenceBuilder(min_conf=0.45)
        for _ in range(10):
            self.assertIsNone(b.update({"hello": 0.4, "yes": 0.35, "no": 0.25}))

    def test_held_sign_is_not_repeated(self):
        b = SentenceBuilder(stable=3)
        words = [b.update({"hello": 0.9, "yes": 0.1}) for _ in range(20)]
        self.assertEqual([w for w in words if w], ["hello"])

    def test_same_word_again_after_hands_rest(self):
        b = SentenceBuilder(stable=3)
        for _ in range(3):
            b.update({"hello": 0.9})
        b.rest()
        words = [b.update({"hello": 0.9}) for _ in range(3)]
        self.assertEqual(b.words, ["hello", "hello"])
        self.assertEqual(words[-1], "hello")


class ContinuousInferenceTests(unittest.TestCase):
    def make(self, **options):
        session = FakeSession()
        return ContinuousInference(session, window=10, predict_every=2, **options), session

    def test_nothing_until_the_buffer_is_full(self):
        ci, session = self.make()
        self.assertEqual(feed(ci, 9), [])
        self.assertEqual(session.model.calls, 0)

    def test_signing_adds_a_word_with_engine_result_shape(self):
        ci, _ = self.make()
        results = feed(ci, 30)
        self.assertEqual(len(results), 1)  # held sign: added once, not repeated
        r = results[0]
        self.assertTrue(r["accepted"])
        self.assertEqual(r["text"], "hello")
        self.assertAlmostEqual(r["confidence"], 0.9)
        self.assertAlmostEqual(r["margin"], 0.9 - 0.07)
        self.assertAlmostEqual(r["display_confidence"], 0.9)  # already above 80%: unchanged
        self.assertEqual([p["gloss"] for p in r["predictions"]], ["hello", "yes", "no"])

    def test_resting_hands_never_run_the_model(self):
        ci, session = self.make()
        session.hands = False
        self.assertEqual(feed(ci, 40), [])
        self.assertEqual(session.model.calls, 0)

    def test_two_different_signs_in_a_row(self):
        ci, session = self.make()
        first = feed(ci, 20)
        session.model.probs = {"yes": 0.85, "hello": 0.1, "no": 0.05}
        second = feed(ci, 20, start=20)
        self.assertEqual([r["text"] for r in first + second], ["hello", "yes"])

    def test_display_score_raised_but_real_score_kept(self):
        # "hello" is in display_scale_words.json: a 60% real score displays as 80%.
        ci, session = self.make()
        session.model.probs = {"hello": 0.6, "yes": 0.25, "no": 0.15}
        r = feed(ci, 30)[0]
        self.assertAlmostEqual(r["confidence"], 0.6)
        self.assertEqual(r["display_confidence"], 0.8)

    def test_word_rules_are_used_when_the_model_has_them(self):
        ci, session = self.make()
        calls = []
        session.model.predict_with_word_rules = lambda frames, top_k: calls.append(top_k) or \
            session.model.predict(frames, top_k)
        feed(ci, 20)
        self.assertTrue(calls)
        self.assertEqual(calls[0], 3)  # whole vocabulary

    def test_model_and_close_pass_through(self):
        ci, session = self.make()
        self.assertEqual(ci.model.labels, ["hello", "yes", "no"])
        ci.close()
        self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
