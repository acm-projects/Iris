import json
import subprocess
import sys
import unittest
from pathlib import Path


ENGINE = Path(__file__).resolve().parent / "engine.py"


class EngineProtocolTests(unittest.TestCase):
    def setUp(self):
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(ENGINE)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

    def tearDown(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        for stream in (self.process.stdout, self.process.stderr):
            if stream:
                stream.close()

    def exchange(self, line):
        self.assertIsNotNone(self.process.stdin)
        self.assertIsNotNone(self.process.stdout)
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        self.assertTrue(response, "engine exited without a protocol response")
        return json.loads(response)

    def test_health_request_round_trip(self):
        response = self.exchange(
            json.dumps({"id": "health-1", "method": "health", "params": {}})
        )
        self.assertEqual(response["id"], "health-1")
        self.assertEqual(response["result"]["status"], "ok")
        self.assertEqual(response["result"]["service"], "iris-engine")

    def test_malformed_messages_return_protocol_errors_and_engine_recovers(self):
        invalid_json = self.exchange("{not json")
        self.assertEqual(invalid_json, {"id": None, "error": "Invalid JSON."})

        invalid_id = self.exchange(json.dumps({"method": "health"}))
        self.assertEqual(
            invalid_id,
            {"id": None, "error": "Request requires a string ID."},
        )

        recovered = self.exchange(
            json.dumps({"id": "health-2", "method": "health"})
        )
        self.assertEqual(recovered["id"], "health-2")
        self.assertEqual(recovered["result"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
