import sys
import threading
import unittest
from pathlib import Path


ENGINE_DIR = Path(__file__).resolve().parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from sentence_assembly import (
    FallbackSentenceTranslator,
    PhraseAssembler,
    PhraseAssemblyWorker,
    TranslationResult,
    format_glosses,
    normalize_gloss,
)


class PhraseAssemblerTests(unittest.TestCase):
    def test_normalizes_and_preserves_order(self):
        assembler = PhraseAssembler()

        first = assembler.append("  WANT  ", 1_000)
        second = assembler.append("ICE_CREAM", 2_000)

        self.assertEqual(first.text, "Want")
        self.assertEqual(second.glosses, ("want", "ice cream"))
        self.assertEqual(second.text, "Want ice cream")

    def test_suppresses_only_nearby_consecutive_duplicates(self):
        assembler = PhraseAssembler(duplicate_window_ms=1_400)

        assembler.append("help", 1_000)
        duplicate = assembler.append("HELP", 2_000)
        assembler.append("please", 2_500)
        nonconsecutive = assembler.append("help", 2_700)

        self.assertFalse(duplicate.appended)
        self.assertEqual(nonconsecutive.glosses, ("help", "please", "help"))

    def test_allows_same_gloss_after_duplicate_window(self):
        assembler = PhraseAssembler(duplicate_window_ms=1_400)

        assembler.append("very", 1_000)
        repeated = assembler.append("very", 2_401)

        self.assertTrue(repeated.appended)
        self.assertEqual(repeated.glosses, ("very", "very"))

    def test_idle_deadline_finalize_and_clear(self):
        assembler = PhraseAssembler(idle_timeout_ms=2_500)
        assembler.append("thanksgiving", 10_000)

        self.assertEqual(assembler.remaining_ms(11_000), 1_500)
        self.assertFalse(assembler.is_due(12_499))
        self.assertTrue(assembler.is_due(12_500))
        self.assertEqual(assembler.finalize(), ("thanksgiving",))
        self.assertIsNone(assembler.finalize())

        assembler.append("help", 20_000)
        self.assertTrue(assembler.clear())
        self.assertFalse(assembler.clear())

    def test_rejects_invalid_configuration_and_glosses(self):
        with self.assertRaises(ValueError):
            PhraseAssembler(idle_timeout_ms=0)
        with self.assertRaises(ValueError):
            PhraseAssembler(duplicate_window_ms=-1)
        with self.assertRaises(ValueError):
            normalize_gloss("   ")
        with self.assertRaises(TypeError):
            normalize_gloss(123)  # type: ignore[arg-type]


class FallbackTranslatorTests(unittest.TestCase):
    def test_formats_cleaned_glosses_without_reordering(self):
        translator = FallbackSentenceTranslator()

        result = translator.translate(("want", "help"))

        self.assertEqual(result, TranslationResult("Want help.", translated=False))
        self.assertEqual(format_glosses(("who", "you"), final=False), "Who you")


class PhraseAssemblyWorkerTests(unittest.TestCase):
    def test_emits_building_then_one_manual_final_event(self):
        events = []
        final = threading.Event()

        def emit(message):
            events.append(message)
            if message.get("state") == "final":
                final.set()

        worker = PhraseAssemblyWorker(emit, idle_timeout_ms=5_000)
        worker.start()
        try:
            self.assertTrue(worker.submit("want"))
            self.assertTrue(worker.submit("help"))
            self.assertTrue(worker.finalize())
            self.assertTrue(final.wait(1))
        finally:
            worker.stop()

        building = [event for event in events if event["state"] == "building"]
        finalized = [event for event in events if event["state"] == "final"]
        self.assertEqual([event["text"] for event in building], ["Want", "Want help"])
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0]["text"], "Want help.")
        self.assertEqual(finalized[0]["reason"], "manual")
        self.assertFalse(finalized[0]["translated"])

    def test_idle_timeout_finalizes(self):
        events = []
        final = threading.Event()

        def emit(message):
            events.append(message)
            if message.get("state") == "final":
                final.set()

        worker = PhraseAssemblyWorker(emit, idle_timeout_ms=30)
        worker.start()
        try:
            worker.submit("hello")
            self.assertTrue(final.wait(1))
        finally:
            worker.stop()

        finalized = [event for event in events if event["state"] == "final"]
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0]["reason"], "idle")

    def test_clear_cancels_phrase_without_finalizing(self):
        events = []
        cleared = threading.Event()

        def emit(message):
            events.append(message)
            if message.get("state") == "cleared":
                cleared.set()

        worker = PhraseAssemblyWorker(emit, idle_timeout_ms=5_000)
        worker.start()
        try:
            worker.submit("help")
            worker.clear()
            self.assertTrue(cleared.wait(1))
            worker.finalize()
        finally:
            worker.stop()

        self.assertFalse(any(event["state"] == "final" for event in events))

    def test_translator_failure_falls_back_to_cleaned_sequence(self):
        class BrokenTranslator:
            def translate(self, glosses):
                raise RuntimeError("provider unavailable")

        events = []
        final = threading.Event()

        def emit(message):
            events.append(message)
            if message.get("state") == "final":
                final.set()

        worker = PhraseAssemblyWorker(
            emit,
            translator=BrokenTranslator(),
            idle_timeout_ms=5_000,
        )
        worker.start()
        try:
            worker.submit("need")
            worker.submit("medicine")
            worker.finalize()
            self.assertTrue(final.wait(1))
        finally:
            worker.stop()

        finalized = next(event for event in events if event["state"] == "final")
        self.assertEqual(finalized["text"], "Need medicine.")
        self.assertFalse(finalized["translated"])
        self.assertIn("RuntimeError", finalized["translationError"])

    def test_translation_runs_off_the_callers_thread(self):
        caller_thread = threading.get_ident()
        translated_thread = None
        final = threading.Event()

        class RecordingTranslator:
            def translate(self, glosses):
                nonlocal translated_thread
                translated_thread = threading.get_ident()
                return TranslationResult("I need help.", translated=True)

        worker = PhraseAssemblyWorker(
            lambda message: final.set() if message.get("state") == "final" else None,
            translator=RecordingTranslator(),
            idle_timeout_ms=5_000,
        )
        worker.start()
        try:
            worker.submit("need")
            worker.submit("help")
            worker.finalize()
            self.assertTrue(final.wait(1))
        finally:
            worker.stop()

        self.assertIsNotNone(translated_thread)
        self.assertNotEqual(translated_thread, caller_thread)

    def test_stop_clears_pending_phrase_without_finalizing(self):
        events = []
        worker = PhraseAssemblyWorker(events.append, idle_timeout_ms=5_000)
        worker.start()
        worker.submit("unfinished")
        worker.stop()

        self.assertTrue(any(event["state"] == "cleared" for event in events))
        self.assertFalse(any(event["state"] == "final" for event in events))


if __name__ == "__main__":
    unittest.main()
