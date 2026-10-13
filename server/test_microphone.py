"""Run: python -m unittest server.test_microphone

Exercises real MP3 decoding with simulated playback; never opens an audio device
or sends a Google request. Uses the packages in backend/requirements.txt.
"""

from io import BytesIO
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile

from server import api


class MicrophoneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A short in-memory recording exercises the installed MP3 codec.
        output = BytesIO()
        sf.write(output, np.zeros(2400, dtype=np.float32), 24000, format="MP3")
        cls.mp3 = output.getvalue()

    def upload(self, data):
        return UploadFile(filename="speech.mp3", file=BytesIO(data))

    def test_mp3_is_decoded_and_passed_to_existing_player(self):
        with patch.object(api, "play", return_value={"device": "Test cable", "durationMs": 100}) as play:
            result = api.push_audio_to_microphone(self.upload(self.mp3))
        samples, rate, device = play.call_args.args
        self.assertEqual(samples.dtype, np.float32)
        self.assertGreater(len(samples), 0)
        self.assertEqual(rate, 24000)
        self.assertIsNone(device)
        self.assertEqual(result.status, "played")
        self.assertEqual(result.device, "Test cable")

    def test_invalid_uploads_do_not_play(self):
        wav = BytesIO()
        sf.write(wav, np.zeros(100), 24000, format="WAV")
        for data, status in [(b"", 400), (b"not audio", 400), (wav.getvalue(), 415)]:
            with self.subTest(status=status), patch.object(api, "play") as play:
                with self.assertRaises(HTTPException) as error:
                    api.push_audio_to_microphone(self.upload(data))
                self.assertEqual(error.exception.status_code, status)
                play.assert_not_called()

    def test_size_limit(self):
        with patch.object(api, "MAX_MP3_BYTES", 4), patch.object(api, "play") as play:
            with self.assertRaises(HTTPException) as error:
                api.push_audio_to_microphone(self.upload(b"12345"))
            self.assertEqual(error.exception.status_code, 413)
            play.assert_not_called()

    def test_duration_limit(self):
        with patch.object(api, "MAX_AUDIO_SECONDS", 0):
            with self.assertRaises(HTTPException) as error:
                api.decode_mp3(self.mp3)
            self.assertEqual(error.exception.status_code, 413)

    def test_busy_microphone(self):
        with api._microphone_lock, patch.object(api, "play") as play:
            with self.assertRaises(HTTPException) as error:
                api.push_audio_to_microphone(self.upload(self.mp3))
            self.assertEqual(error.exception.status_code, 409)
            play.assert_not_called()

    def test_missing_device_releases_lock(self):
        with patch.object(api, "play", side_effect=RuntimeError("No virtual microphone")):
            with self.assertRaises(HTTPException) as error:
                api.push_audio_to_microphone(self.upload(self.mp3))
            self.assertEqual(error.exception.status_code, 503)
        self.assertFalse(api._microphone_lock.locked())

    def test_swagger_documents_file_upload(self):
        endpoint = api.app.openapi()["paths"]["/microphone"]["post"]
        self.assertIn("multipart/form-data", endpoint["requestBody"]["content"])


if __name__ == "__main__":
    unittest.main()
