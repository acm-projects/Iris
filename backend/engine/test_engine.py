import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ENGINE_DIR = Path(__file__).resolve().parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import engine


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


if __name__ == "__main__":
    unittest.main()
