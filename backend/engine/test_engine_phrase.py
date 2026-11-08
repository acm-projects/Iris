"""Tests for the engine's phrase timer: after the hands rest for
PHRASE_PAUSE_SECONDS, the words signed so far go to the app as one "phrase"
event (which the app translates into English and speaks).

Fake time, fake recognizer and a captured send(), so nothing real runs:

    cd backend/engine && ../.venv/bin/python -m unittest test_engine_phrase
"""
import unittest
from unittest import mock

import engine


class FakeBuilder:
    def __init__(self, words):
        self.words = list(words)


class FakeContinuous:
    """Just enough of ContinuousInference for the phrase timer."""

    def __init__(self, words):
        self.builder = FakeBuilder(words)
        self.taken = 0

    def take_phrase(self):
        self.taken += 1
        words, self.builder.words = self.builder.words, []
        return words


class PhraseTimerTests(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.clock = [100.0]
        self.patches = [
            mock.patch.object(engine, "send", self.sent.append),
            mock.patch.object(engine, "log", lambda *_: None),
            mock.patch.object(engine.time, "monotonic", lambda: self.clock[0]),
        ]
        for p in self.patches:
            p.start()
        self.rec = engine.SignRecognition()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def tick(self, rest, seconds, signing=False):
        """Advance the clock and run the timer once."""
        self.clock[0] += seconds
        return self.rec._maybe_send_phrase(rest, signing)

    def test_phrase_sent_once_after_the_pause(self):
        self.rec.recognizer = FakeContinuous(["i", "want", "help"])
        rest = self.tick(None, 0)            # hands just went down
        rest = self.tick(rest, 1.0)          # not long enough yet
        self.assertEqual(self.sent, [])
        rest = self.tick(rest, 1.1)          # 2.1 s of rest
        self.assertEqual(self.sent, [{"event": "phrase", "words": ["i", "want", "help"]}])
        self.tick(rest, 5.0)                 # still resting: nothing new to send
        self.assertEqual(len(self.sent), 1)

    def test_signing_restarts_the_pause(self):
        self.rec.recognizer = FakeContinuous(["help"])
        rest = self.tick(None, 0)
        rest = self.tick(rest, 1.5)
        rest = self.tick(rest, 0.1, signing=True)   # hands back up: timer resets
        self.assertIsNone(rest)
        rest = self.tick(rest, 0)
        rest = self.tick(rest, 1.5)
        self.assertEqual(self.sent, [])
        self.tick(rest, 0.6)
        self.assertEqual(len(self.sent), 1)

    def test_no_words_means_no_phrase(self):
        self.rec.recognizer = FakeContinuous([])
        rest = self.tick(None, 0)
        self.tick(rest, 10)
        self.assertEqual(self.sent, [])

    def test_window_mode_never_sends_phrases(self):
        self.rec.recognizer = object()  # LiveSignRecognizer has no take_phrase
        self.assertIsNone(self.tick(None, 0))
        self.assertEqual(self.sent, [])


if __name__ == "__main__":
    unittest.main()
