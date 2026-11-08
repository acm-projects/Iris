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

    def test_take_phrase_hands_over_words_and_starts_fresh(self):
        ci, session = self.make()
        feed(ci, 20)
        self.assertTrue(ci.signing)
        self.assertEqual(ci.take_phrase(), ["hello"])
        self.assertEqual(ci.builder.words, [])
        self.assertEqual(ci.builder.sentences, [])  # history isn't kept forever
        session.hands = False
        feed(ci, 15, start=20)
        self.assertFalse(ci.signing)

    def test_model_and_close_pass_through(self):
        ci, session = self.make()
        self.assertEqual(ci.model.labels, ["hello", "yes", "no"])
        ci.close()
        self.assertTrue(session.closed)


# Preserve the local phrase-translation and acceptance regression tests.
import unittest
from types import SimpleNamespace
from threading import Event
from unittest.mock import MagicMock, patch

import numpy as np

from continuous_inference import ContinuousInference, RollingBuffer, SentenceBuilder, WINDOW_FRAMES
from gloss_translation import PhraseTranslator, PauseSubmitter, chosen_prediction, translate_phrase


def probs(gloss, p=0.8, others=("x", "y")):
    rest = (1 - p) / len(others)
    return {gloss: p, **{o: rest for o in others}}


class TranslationRollingBufferTests(unittest.TestCase):
    def test_keeps_only_newest_50_in_order(self):
        buf = RollingBuffer()
        for i in range(70):
            buf.append(np.full((55, 2), i, np.float32), True)
        self.assertEqual(len(buf), WINDOW_FRAMES)
        self.assertTrue(buf.full)
        window = buf.window()
        self.assertEqual(window.shape, (WINDOW_FRAMES, 55, 2))
        self.assertEqual(window[0, 0, 0], 70 - WINDOW_FRAMES)
        self.assertEqual(window[-1, 0, 0], 69)

    def test_hand_fraction(self):
        buf = RollingBuffer()
        for i in range(10):
            buf.append(np.zeros((55, 2), np.float32), i >= 5)
        self.assertEqual(buf.hand_fraction(), 0.5)
        self.assertEqual(buf.hand_fraction(last=5), 1.0)


class TranslationSentenceBuilderTests(unittest.TestCase):
    def feed(self, builder, gloss, times, p=0.8):
        return [builder.update(probs(gloss, p)) for _ in range(times)]

    def test_commits_only_after_stable_confident_windows(self):
        b = SentenceBuilder()
        self.assertEqual(self.feed(b, "hello", 2), [None, "hello"])
        self.assertEqual(b.text, "hello")

    def test_low_confidence_never_commits(self):
        b = SentenceBuilder()
        low_probs = {"hello": .28, "x": .24, "y": .24, "z": .24}
        self.assertEqual([b.update(low_probs) for _ in range(8)], [None] * 8)
        self.assertIn('below 30%', b.status)

    def test_accepts_shorter_lower_confidence_signs_without_duplicates(self):
        b = SentenceBuilder()
        self.assertEqual(self.feed(b, 'hello', 2, p=.34), [None, 'hello'])
        self.assertEqual(self.feed(b, 'hello', 20, p=.34), [None] * 20)
        self.assertIn('duplicate', b.status)
        # The preceding held sign still contributes until its smoothing votes expire.
        self.assertEqual(self.feed(b, 'help', 3, p=.40), [None, None, 'help'])
        self.assertEqual(b.words, ['hello', 'help'])

    def test_original_strict_settings_remain_available(self):
        b = SentenceBuilder(smooth=5, stable=3, min_conf=.45)
        self.assertEqual(self.feed(b, 'hello', 3), [None, None, 'hello'])

    def test_invalid_acceptance_settings_are_rejected(self):
        for options in ({'stable': 0}, {'smooth': 1, 'stable': 2}, {'min_conf': .1}):
            with self.assertRaises(ValueError):
                SentenceBuilder(**options)

    def test_one_sign_is_not_repeated_while_it_stays_in_the_window(self):
        b = SentenceBuilder()
        out = self.feed(b, "hello", 12)
        self.assertEqual([w for w in out if w], ["hello"])

    def test_same_word_again_after_rest(self):
        b = SentenceBuilder()
        self.feed(b, "hello", 3)
        b.rest()
        self.feed(b, "hello", 3)
        self.assertEqual(b.words, ["hello", "hello"])

    def test_sequence_of_words_and_fragment(self):
        b = SentenceBuilder()
        for g in ("i", "want", "help"):
            self.feed(b, g, 4)
        self.assertEqual(b.text, "i want help")

    def test_flicker_does_not_commit(self):
        b = SentenceBuilder()
        out = [b.update(probs(g, 0.6)) for g in ("a", "b", "a", "b", "a", "b")]
        self.assertEqual(out, [None] * 6)

    def test_end_sentence_archives_and_clears(self):
        b = SentenceBuilder()
        self.feed(b, "hello", 3)
        b.end_sentence()
        self.assertEqual(b.words, [])
        self.assertEqual(b.sentences, [["hello"]])

    def test_max_words(self):
        b = SentenceBuilder(max_words=2)
        for g in ("a", "b", "c"):
            self.feed(b, g, 3)
        self.assertEqual(b.words, ["a", "b", "c"][1:])


class TranslationFakeSession:
    """Stands in for PoseModelInference: scripted keypoints, scripted model."""
    def __init__(self):
        self.top_k = 3
        self.hands = True
        self.body = True
        self.calls = 0
        self.model = SimpleNamespace(labels=["hello", "x", "y"], predict=self.predict)

    def process_frame(self, rgb, ts):
        pts = np.full((55, 2), np.nan, np.float32)
        if self.body:
            pts[:13] = 1.0
        if self.hands:
            pts[13:] = 2.0
        return pts

    def predict(self, frames, top_k=5):
        self.calls += 1
        assert frames.shape == (WINDOW_FRAMES, 55, 2)
        return [{"gloss": "hello", "confidence": .8}, {"gloss": "x", "confidence": .1},
                {"gloss": "y", "confidence": .1}][:top_k]


class TranslationContinuousInferenceTests(unittest.TestCase):
    def run_frames(self, live, n, start=0):
        return [live.process_frame(None, start + i) for i in range(n)]

    def test_predicts_only_with_full_window_and_hands(self):
        fake = TranslationFakeSession()
        live = ContinuousInference(fake, predict_every=5)
        updates = self.run_frames(live, WINDOW_FRAMES + 10)
        self.assertEqual(sum(u.predicted for u in updates[:WINDOW_FRAMES - 1]), 0)
        self.assertEqual(fake.calls, 3)
        self.assertEqual(live.builder.words, ["hello"])              # committed once, not three times

    def test_no_hands_means_no_predictions(self):
        fake = TranslationFakeSession()
        fake.hands = False
        live = ContinuousInference(fake)
        updates = self.run_frames(live, 80)
        self.assertEqual(fake.calls, 0)
        self.assertFalse(any(u.signing for u in updates))

    def test_nothing_buffered_until_a_body_is_seen(self):
        fake = TranslationFakeSession()
        fake.body = False
        live = ContinuousInference(fake)
        self.run_frames(live, 5)
        self.assertEqual(len(live.buffer), 0)
        fake.body = True
        self.run_frames(live, 5, start=5)
        self.assertEqual(len(live.buffer), 5)

    def test_lost_body_repeats_last_body_frame(self):
        fake = TranslationFakeSession()
        live = ContinuousInference(fake)
        self.run_frames(live, 3)
        fake.body = False
        self.run_frames(live, 2, start=3)
        self.assertTrue(np.isfinite(live.buffer.window()[-1, :13]).all())

    def test_long_pause_ends_sentence_and_allows_repeat(self):
        fake = TranslationFakeSession()
        live = ContinuousInference(fake, sentence_gap_frames=20)
        self.run_frames(live, WINDOW_FRAMES + 10)
        fake.hands = False
        self.run_frames(live, 60, start=55)
        self.assertEqual(live.builder.sentences, [["hello"]])
        fake.hands = True
        self.run_frames(live, WINDOW_FRAMES + 10, start=150)
        self.assertEqual(live.builder.words, ["hello"])

    def test_default_pause_keeps_phrase_until_manual_submission(self):
        fake = TranslationFakeSession()
        live = ContinuousInference(fake)
        self.run_frames(live, WINDOW_FRAMES + 10)
        fake.hands = False
        self.run_frames(live, 200, start=100)
        self.assertEqual(live.builder.words, ["hello"])
        self.assertEqual(live.builder.sentences, [])


class PhraseTranslationTests(unittest.TestCase):
    def test_pause_submits_once_and_requires_continuous_rest(self):
        auto = PauseSubmitter(2)
        translator = SimpleNamespace(pending=False, submit=MagicMock(return_value="submitted"))
        words = ['i', 'want', 'help']
        def update(now, signing=False, committed=False):
            return auto.update(signing=signing, committed=committed, now=now,
                               words=words, translator=translator)
        self.assertIsNone(update(0, signing=True, committed=True))
        self.assertIsNone(update(1))
        self.assertIsNone(update(2, signing=True))
        self.assertIsNone(update(3))
        self.assertIsNone(update(4.9))
        self.assertEqual(update(5), 'submitted')
        self.assertIsNone(update(10))  # failure must not trigger repeated API charges
        translator.submit.assert_called_once_with(words)
        words.append('please')
        update(11, signing=True, committed=True)
        update(12)
        translator.pending = True
        self.assertIsNone(update(14))
        translator.pending = False
        self.assertEqual(update(15), 'submitted')

    def test_provider_receives_one_ordered_phrase_with_retries_disabled(self):
        client = MagicMock()
        client.models.generate_content.return_value = SimpleNamespace(
            candidates=[SimpleNamespace(finish_reason='STOP')], text="I want help."
        )
        constructor = MagicMock()
        constructor.return_value.__enter__.return_value = client
        dotenv = SimpleNamespace(load_dotenv=MagicMock())
        with patch.dict('sys.modules', {'google': SimpleNamespace(genai=SimpleNamespace(Client=constructor)),
                                       'dotenv': dotenv}), patch.dict(
            'os.environ', {'GEMINI_API_KEY': 'placeholder-for-offline-test',
                           'GLOSS_LLM_MODEL': 'gemini-flash-latest'}
        ):
            self.assertEqual(translate_phrase(["i", "want", "help"]), "I want help.")
        constructor.assert_called_once_with(api_key='placeholder-for-offline-test', vertexai=False,
                                            http_options={'timeout': 15_000, 'retry_options': {'attempts': 1}})
        client.models.generate_content.assert_called_once()
        kwargs = client.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs['model'], 'gemini-flash-latest')
        self.assertEqual(kwargs['contents'], '{"glosses": ["i", "want", "help"]}')
        self.assertFalse(dotenv.load_dotenv.call_args.kwargs['override'])

    def test_provider_error_details_are_not_exposed(self):
        client = MagicMock()
        provider_error = RuntimeError("private provider details")
        provider_error.code = 503
        client.models.generate_content.side_effect = provider_error
        constructor = MagicMock()
        constructor.return_value.__enter__.return_value = client
        with patch.dict('sys.modules', {'google': SimpleNamespace(genai=SimpleNamespace(Client=constructor)),
                                       'dotenv': SimpleNamespace(load_dotenv=MagicMock())}), patch.dict(
            'os.environ', {'GEMINI_API_KEY': 'placeholder-for-offline-test',
                           'GLOSS_LLM_MODEL': 'gemini-flash-latest'}
        ):
            with self.assertRaises(RuntimeError) as raised:
                translate_phrase(["help"])
        self.assertNotIn("private provider details", str(raised.exception))
        self.assertIn("temporarily unavailable or overloaded (HTTP 503)", str(raised.exception))
        client.models.generate_content.assert_called_once()

    def test_selection_uses_score_not_list_order(self):
        for field in ("confidence", "score"):
            result = {"predictions": [{"gloss": "wrong", field: .1},
                                      {"gloss": "chosen", field: .9}]}
            self.assertEqual(chosen_prediction(result)["gloss"], "chosen")

    def test_pending_snapshot_does_not_block_words_or_duplicate_requests(self):
        started, release = Event(), Event()

        def fake_translate(snapshot):
            self.assertEqual(snapshot, ("i", "want"))
            started.set()
            self.assertTrue(release.wait(5))
            return "I want."

        controller = PhraseTranslator()
        try:
            with patch('gloss_translation.translate_phrase', side_effect=fake_translate) as call:
                words = ["i", "want"]
                controller.submit(words)
                self.assertTrue(started.wait(5))
                words.append("help")
                self.assertIsNone(controller.poll())
                self.assertIn("in progress", controller.submit(words))
                release.set()
                controller._future.result(timeout=5)
                self.assertEqual(controller.poll(), "English: I want.")
                self.assertEqual(words, ["help"])
                call.assert_called_once()
        finally:
            release.set()
            controller.close()

    def test_failure_retains_phrase_and_empty_submit_does_not_call(self):
        controller = PhraseTranslator()
        try:
            with patch('gloss_translation.translate_phrase', side_effect=RuntimeError("Unavailable")) as call:
                self.assertIn("empty", controller.submit([]))
                call.assert_not_called()
                words = ["help"]
                controller.submit(words)
                with self.assertRaises(RuntimeError):
                    controller._future.result(timeout=5)
                self.assertEqual(controller.poll(), "Unavailable")
                self.assertEqual(words, ["help"])
        finally:
            controller.close()

    def test_default_phrase_does_not_silently_truncate(self):
        builder = SentenceBuilder()
        for index in range(20):
            for _ in range(3):
                builder.update(probs(str(index)))
        self.assertEqual(len(builder.words), 20)



if __name__ == "__main__":
    unittest.main()
