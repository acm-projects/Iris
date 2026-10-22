import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ENGINE_DIR = Path(__file__).resolve().parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import engine
from sentence_assembly import PhraseAssemblyWorker


class EngineSignToSpeechTests(unittest.TestCase):
    def test_sign_model_contract_is_ready(self):
        status = engine.sign_model_status()
        if status["status"] == "missing":
            self.skipTest(f"Optional local sign assets are absent: {status}")
        self.assertTrue(status["ok"], status)
        self.assertEqual(status["status"], "ready")

    def test_recognition_submission_does_not_run_inference_inline(self):
        class Recognizer:
            def __init__(self):
                self.frames = []

            def submit(self, frame):
                self.frames.append(frame)

        recognizer = Recognizer()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch.object(engine, "_recognizer", recognizer):
            result = engine.recognize_sign(frame)
        self.assertIsNone(result)
        self.assertIs(recognizer.frames[0], frame)

    def test_accepted_sign_text_uses_tts_and_virtual_output_path(self):
        samples = np.zeros(480, dtype=np.float32)
        with patch.object(
            engine, "synthesize_speech", return_value=(samples, 48000)
        ) as synthesize, patch.object(
            engine,
            "play",
            return_value={"device": "CABLE Input", "durationMs": 10},
        ) as play:
            result = engine.handle(
                "speak",
                {"text": "bed", "device": "CABLE Input"},
            )
        synthesize.assert_called_once_with("bed")
        play.assert_called_once_with(samples, 48000, "CABLE Input")
        self.assertEqual(result["device"], "CABLE Input")
        self.assertGreaterEqual(result["synthesisMs"], 0)
        self.assertGreaterEqual(result["playbackMs"], 0)
        self.assertGreaterEqual(result["totalLatencyMs"], 0)

    def test_only_accepted_signs_are_submitted_for_sentence_assembly(self):
        class PhraseWorker:
            def __init__(self):
                self.glosses = []

            def submit(self, gloss):
                self.glosses.append(gloss)
                return True

        worker = PhraseWorker()
        accepted = {"event": "sign", "text": "help", "accepted": True}
        rejected = {"event": "sign", "text": "wrong", "accepted": False}
        with patch.object(engine, "_phrase_worker", worker), patch.object(
            engine, "send"
        ) as send:
            engine._handle_recognition_event(accepted)
            engine._handle_recognition_event(rejected)

        self.assertEqual(worker.glosses, ["help"])
        self.assertEqual(
            [call.args[0] for call in send.call_args_list],
            [accepted, rejected],
        )

    def test_phrase_controls_queue_work_on_the_phrase_worker(self):
        class PhraseWorker:
            def __init__(self):
                self.actions = []

            def finalize(self):
                self.actions.append("finalize")
                return True

            def clear(self):
                self.actions.append("clear")
                return True

        worker = PhraseWorker()
        with patch.object(engine, "_phrase_worker", worker):
            finalized = engine.handle("finalize_phrase", {})
            cleared = engine.handle("clear_phrase", {})

        self.assertEqual(finalized, {"queued": True})
        self.assertEqual(cleared, {"queued": True})
        self.assertEqual(worker.actions, ["finalize", "clear"])


class VirtualCameraLifecycleTests(unittest.TestCase):
    def test_stuck_native_shutdown_keeps_worker_registered(self):
        class StuckThread:
            def __init__(self):
                self.joined = False

            def is_alive(self):
                return True

            def join(self, timeout=None):
                self.joined = True

        camera = engine.VirtualCameraStream()
        thread = StuckThread()
        camera._thread = thread
        with patch.object(engine, "stop_sign_recognition"), patch.object(
            engine, "log"
        ):
            stopped = camera.stop()
            restarted = camera.start(640, 480, 24)

        self.assertFalse(stopped)
        self.assertTrue(thread.joined)
        self.assertIs(camera._thread, thread)
        self.assertTrue(restarted["starting"])
        self.assertIn("shutdown is still pending", camera.last_error)

    def test_completed_worker_is_cleared_on_stop(self):
        class CompletedThread:
            def is_alive(self):
                return False

            def join(self, timeout=None):
                return None

        camera = engine.VirtualCameraStream()
        camera._thread = CompletedThread()
        with patch.object(engine, "stop_sign_recognition"):
            stopped = camera.stop()

        self.assertTrue(stopped)
        self.assertIsNone(camera._thread)


class SignPhraseSpeechContractTests(unittest.TestCase):
    def test_accepted_sign_reaches_phrase_and_supervised_speech_path(self):
        final = threading.Event()
        events = []
        speech_result = {}
        samples = np.zeros(480, dtype=np.float32)

        def emit(message):
            events.append(message)
            if message.get("state") == "final":
                speech_result.update(engine.handle("speak", {"text": message["text"]}))
                final.set()

        worker = PhraseAssemblyWorker(emit, idle_timeout_ms=5_000)
        worker.start()
        try:
            with patch.object(engine, "_phrase_worker", worker), patch.object(
                engine, "send"
            ), patch.object(
                engine, "synthesize_speech", return_value=(samples, 48_000)
            ), patch.object(
                engine,
                "play",
                return_value={"device": "CABLE Input", "durationMs": 10},
            ):
                engine._handle_recognition_event(
                    {"event": "sign", "text": "help", "accepted": True}
                )
                self.assertTrue(worker.finalize())
                self.assertTrue(final.wait(1))
        finally:
            worker.stop()

        finalized = [event for event in events if event.get("state") == "final"]
        self.assertEqual(finalized[0]["text"], "Help.")
        self.assertGreaterEqual(finalized[0]["assemblyLatencyMs"], 0)
        self.assertEqual(speech_result["device"], "CABLE Input")
        self.assertGreaterEqual(speech_result["totalLatencyMs"], 0)


if __name__ == "__main__":
    unittest.main()
