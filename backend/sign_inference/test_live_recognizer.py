"""Tests for live_recognizer.py: when a sign starts and ends, and which results are accepted.

A FakeSession replays scripted keypoints instead of running MediaPipe and the
model, so these run in a fraction of a second.
"""
import unittest

import numpy as np

from live_recognizer import LiveSignRecognizer, hands_in_signing_space


def body_points(hand_y=None, hands=("left",)):
    """(55, 2) points: neck at y=200, shoulders 100 apart; optional hands with wrist at hand_y."""
    p = np.full((55, 2), np.nan, np.float32)
    p[1] = (320, 200)              # neck
    p[2], p[5] = (270, 200), (370, 200)  # shoulders (width 100)
    if hand_y is not None:
        for side in hands:
            start = 13 if side == "left" else 34
            p[start:start + 21] = np.column_stack([np.full(21, 300.0), np.linspace(hand_y, hand_y - 40, 21)])
    return p


class FakeSession:
    """Stands in for PoseModelInference: replays scripted points and a scripted prediction."""

    def __init__(self, frames, confidence=0.8, margin=0.5):
        # frames: points returned for each call; confidence / margin: the scripted prediction
        self.frames = list(frames)
        self.confidence, self.margin = confidence, margin
        self.recording = False
        self.recorded = 0
        self.model = type("M", (), {"labels": ["hello", "yes"]})()

    def process_frame(self, rgb, ts):
        """Return the next scripted points; count the frame if recording."""
        points = self.frames.pop(0)
        if self.recording:
            self.recorded += 1
        return points

    def start_sequence(self):
        """Start a new recording, like PoseModelInference.start_sequence."""
        self.recording, self.recorded = True, 0

    def finish_and_predict(self):
        """Return the scripted prediction; too-short recordings raise, like the real session."""
        self.recording = False
        if self.recorded < 8:
            raise ValueError("Recording too short")
        return {"predictions": [{"gloss": "hello", "confidence": self.confidence},
                                {"gloss": "yes", "confidence": self.confidence - self.margin}],
                "top_confidence": self.confidence, "margin": self.margin,
                "usable_frame_count": self.recorded}

    def reset(self):
        self.recording = False

    def close(self):
        pass


FRAME = np.zeros((360, 640, 3), np.uint8)  # the fake session ignores pixels


def run(frames, step_ms=66, **kwargs):
    """Feed frames (one every step_ms) to a recognizer; return [(frame index, result)] per finished sign."""
    session = FakeSession(frames, **{k: kwargs.pop(k) for k in ("confidence", "margin") if k in kwargs})
    rec = LiveSignRecognizer("unused", session=session, **kwargs)
    results = []
    for i in range(len(session.frames)):
        r = rec.process(FRAME, i * step_ms)
        if r:
            results.append((i, r))
    return results


class SigningSpaceTests(unittest.TestCase):
    """Is a hand raised high enough to count as signing?"""
    def test_raised_hand_counts(self):
        self.assertTrue(hands_in_signing_space(body_points(hand_y=250)))   # above neck + 130

    def test_resting_hand_does_not(self):
        self.assertFalse(hands_in_signing_space(body_points(hand_y=400)))  # below the limit

    def test_no_hands(self):
        self.assertFalse(hands_in_signing_space(body_points()))

    def test_hand_without_body_counts(self):
        p = body_points(hand_y=400)
        p[[1, 2, 5]] = np.nan
        self.assertTrue(hands_in_signing_space(p))


class SegmentationTests(unittest.TestCase):
    """Automatic mode: a sign starts when hands come up and ends when they drop."""
    def test_one_sign_is_reported_after_hands_drop(self):
        frames = [body_points(400)] * 3 + [body_points(250)] * 15 + [body_points(400)] * 10
        results = run(frames)
        self.assertEqual(len(results), 1)
        i, r = results[0]
        self.assertTrue(r["accepted"])
        self.assertEqual(r["text"], "hello")
        self.assertGreaterEqual(i, 18 + 6)  # only after ~450 ms without raised hands

    def test_low_confidence_is_not_accepted(self):
        frames = [body_points(250)] * 15 + [body_points(400)] * 10
        (_, r), = run(frames, confidence=0.3)
        self.assertFalse(r["accepted"])

    def test_small_margin_is_not_accepted(self):
        frames = [body_points(250)] * 15 + [body_points(400)] * 10
        (_, r), = run(frames, confidence=0.6, margin=0.05)
        self.assertFalse(r["accepted"])

    def test_blip_is_ignored(self):
        frames = [body_points(250)] * 3 + [body_points(400)] * 12  # far too short to be a sign
        self.assertEqual(run(frames), [])

    def test_long_signing_is_cut_at_max_length(self):
        frames = [body_points(250)] * 80  # hands up for 80 * 66 ms = 5.3 s
        results = run(frames, max_sign_ms=3000)
        self.assertGreaterEqual(len(results), 1)
        self.assertLessEqual(results[0][0], 3000 // 66 + 2)

    def test_two_signs_in_a_row(self):
        sign = [body_points(250)] * 15 + [body_points(400)] * 10
        self.assertEqual(len(run(sign + sign)), 2)


UP, DOWN = 250, 400  # wrist heights: raised to signing space / resting


class WindowModeTests(unittest.TestCase):
    """Window mode: wait for raised hands, record 50 frames, report, then a 6-frame buffer."""

    def window(self, frames, **kwargs):
        """run() with the app's settings: 50 frames per sign, 6-frame buffer, no confidence filter."""
        return run(frames, window_frames=50, buffer_frames=6, min_confidence=0, min_margin=0, **kwargs)

    def test_idle_records_nothing(self):
        self.assertEqual(self.window([body_points(DOWN)] * 60 + [body_points()] * 60), [])

    def test_window_starts_when_hands_come_up(self):
        frames = [body_points(DOWN)] * 20 + [body_points(UP)] * 60
        results = self.window(frames)
        # hands first up at frame 20; the next 50 frames (21-70) are the sign
        self.assertEqual([i for i, _ in results], [70])
        self.assertEqual(results[0][1]["frames"], 50)

    def test_lowering_hands_after_a_sign_is_ignored(self):
        frames = ([body_points(DOWN)] * 10 + [body_points(UP)] * 51   # sign: frames 10-60
                  + [body_points(UP)] * 5                              # still lowering (61-65)
                  + [body_points(DOWN)] * 40)
        self.assertEqual([i for i, _ in self.window(frames)], [60])

    def test_continuous_signing_starts_the_next_window_after_the_buffer(self):
        results = self.window([body_points(UP)] * 120)
        # start at 0 -> word at 50, buffer 51-56, restart at 57 -> word at 107
        self.assertEqual([i for i, _ in results], [50, 107])

    def test_low_confidence_still_reported_when_thresholds_are_zero(self):
        frames = [body_points(UP)] * 60
        (_, r), = self.window(frames, confidence=0.1, margin=0.01)
        self.assertTrue(r["accepted"])

    def test_a_brief_raise_is_skipped(self):
        frames = [body_points(UP)] + [body_points(DOWN)] * 60  # up for one frame only
        self.assertEqual(self.window(frames), [])


if __name__ == "__main__":
    unittest.main()
