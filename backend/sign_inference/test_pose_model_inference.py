import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from inference_utils import remap_landmarks
import pose_models
from pose_models import PoseModel, list_models, resolve_model

MODEL = "pretrain_asl2000_e100"  # committed, so these tests don't depend on local training runs
HAVE_MODEL = (pose_models.PRETRAINED_DIR / f"{MODEL}.pt").is_file()


def pose_result(left_wrist=(0.7, 0.6), right_wrist=(0.3, 0.6)):
    """33 MediaPipe pose landmarks; 15 = signer's left wrist, 16 = right wrist."""
    lm = [SimpleNamespace(x=0.5, y=0.3 + i / 200) for i in range(33)]
    lm[11], lm[12] = SimpleNamespace(x=0.62, y=0.4), SimpleNamespace(x=0.38, y=0.4)  # shoulders
    lm[15] = SimpleNamespace(x=left_wrist[0], y=left_wrist[1])
    lm[16] = SimpleNamespace(x=right_wrist[0], y=right_wrist[1])
    return SimpleNamespace(pose_landmarks=[lm])


def hand_at(x, y, label, score=0.9):
    """21 hand landmarks with the wrist (index 0) at (x, y)."""
    lm = [SimpleNamespace(x=x + (i % 5) * 0.01, y=y - (i // 5) * 0.01) for i in range(21)]
    return lm, [SimpleNamespace(category_name=label, score=score)]


def hands_result(*hands):
    return SimpleNamespace(hand_landmarks=[h[0] for h in hands], handedness=[h[1] for h in hands])


class WristAssignmentTests(unittest.TestCase):
    def test_swapped_labels_are_fixed_by_wrist_distance(self):
        # MediaPipe calls the hand at the signer's left wrist "Right" (mirrored labels).
        hands = hands_result(hand_at(0.7, 0.6, "Right"), hand_at(0.3, 0.6, "Left"))
        by_label, _ = remap_landmarks(pose_result(), hands, hand_assignment="label")
        by_wrist, sides = remap_landmarks(pose_result(), hands, hand_assignment="wrist")
        self.assertAlmostEqual(by_label[13, 0], 0.3, places=5)   # label put it on the wrong side
        self.assertAlmostEqual(by_wrist[13, 0], 0.7, places=5)   # left-hand slot = hand at left wrist
        self.assertAlmostEqual(by_wrist[34, 0], 0.3, places=5)
        self.assertEqual(sorted(sides), ["Left", "Right"])

    def test_single_hand_goes_to_nearest_wrist(self):
        points, sides = remap_landmarks(pose_result(), hands_result(hand_at(0.31, 0.59, "Left")),
                                        hand_assignment="wrist")
        self.assertEqual(sides, ["Right"])
        self.assertTrue(np.isfinite(points[34:55]).all())
        self.assertTrue(np.isnan(points[13:34]).all())

    def test_no_body_falls_back_to_labels(self):
        points, _ = remap_landmarks(SimpleNamespace(pose_landmarks=[]),
                                    hands_result(hand_at(0.7, 0.6, "Right")), hand_assignment="wrist")
        self.assertTrue(np.isfinite(points[34:55]).all())

    def test_default_behavior_unchanged(self):
        hands = hands_result(hand_at(0.7, 0.6, "Right"))
        a, _ = remap_landmarks(pose_result(), hands)
        b, _ = remap_landmarks(pose_result(), hands, hand_assignment="label")
        np.testing.assert_array_equal(a, b)

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            remap_landmarks(pose_result(), hands_result(), hand_assignment="nearest")


@unittest.skipUnless(HAVE_MODEL, "pretrained model not available")
class PoseModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = PoseModel(MODEL)

    def frames(self, n=40, seed=0):
        rng = np.random.default_rng(seed)
        f = rng.uniform(100, 400, size=(n, 55, 2)).astype(np.float32)
        f[5:9, 13:34] = np.nan  # left hand missing for a few frames
        return f

    def test_resolves_by_name_and_lists_it(self):
        self.assertEqual(resolve_model(MODEL).name, f"{MODEL}.pt")
        self.assertIn(MODEL, list_models())
        with self.assertRaisesRegex(FileNotFoundError, "Available"):
            resolve_model("no_such_model")

    def test_labels_and_shapes_come_from_checkpoint(self):
        self.assertEqual(len(self.model.labels), 2000)
        self.assertEqual(self.model.num_frames, 32)
        self.assertEqual(self.model.features(self.frames()).shape, (32, self.model.feature_dim))

    def test_features_match_training_pipeline(self):
        frames = self.frames()
        raw = np.zeros((len(frames), 55, 3), np.float32)
        ok = np.isfinite(frames).all(-1)
        raw[..., :2] = np.where(ok[..., None], frames, 0)
        raw[..., 2] = ok
        idx = pose_models._data.sample_frames(len(raw), 32, False, np.random.default_rng(0))
        expected = pose_models._data.build_features(raw[idx], self.model.components)
        np.testing.assert_array_equal(self.model.features(frames), expected)

    def test_predictions_sorted_probabilities(self):
        preds = self.model.predict(self.frames(), top_k=5)
        self.assertEqual(len(preds), 5)
        confs = [p["confidence"] for p in preds]
        self.assertEqual(confs, sorted(confs, reverse=True))
        self.assertTrue(0 <= sum(confs) <= 1 + 1e-6)
        self.assertTrue(all(p["gloss"] in self.model.labels for p in preds))

    def test_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            self.model.features(np.zeros((10, 54, 2), np.float32))


class Detector:
    def __init__(self, result):
        self.result = result
        self.closed = False

    def detect_for_video(self, image, timestamp):
        return self.result

    def close(self):
        self.closed = True


@unittest.skipUnless(HAVE_MODEL, "pretrained model not available")
class SessionTests(unittest.TestCase):
    def session(self, pose, hands):
        import pose_model_inference as pmi
        # Placeholder MediaPipe files: the detectors themselves are mocked.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for name in ("pose_landmarker.task", "hand_landmarker.task"):
            (Path(tmp.name) / name).write_bytes(b"")
        with patch("pose_model_inference.vision.PoseLandmarker.create_from_options", return_value=pose), \
             patch("pose_model_inference.vision.HandLandmarker.create_from_options", return_value=hands):
            s = pmi.PoseModelInference(MODEL, model_dir=tmp.name, top_k=3)
        self.addCleanup(s.close)
        return s

    def test_record_and_predict(self):
        pose = Detector(pose_result())
        hands = Detector(hands_result(hand_at(0.7, 0.6, "Right"), hand_at(0.3, 0.6, "Left")))
        s = self.session(pose, hands)
        frame = np.zeros((240, 320, 3), np.uint8)
        s.start_sequence()
        for t in range(1, 21):
            points = s.process_frame(frame, t)
        self.assertEqual(points.shape, (55, 2))
        result = s.finish_and_predict()
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["model"], MODEL)
        self.assertEqual(result["vocabulary_size"], 2000)
        self.assertEqual(len(result["predictions"]), 3)
        self.assertEqual(result["usable_frame_count"], 20)
        self.assertEqual(result["detection_rate"]["left_hand"], 1.0)
        self.assertGreaterEqual(result["margin"], 0)

    def test_short_and_empty_recordings_are_rejected(self):
        pose = Detector(SimpleNamespace(pose_landmarks=[]))
        s = self.session(pose, Detector(hands_result()))
        frame = np.zeros((240, 320, 3), np.uint8)
        s.start_sequence()
        s.process_frame(frame, 1)
        with self.assertRaisesRegex(ValueError, "No body"):
            s.finish_and_predict()
        pose.result = pose_result()
        s.start_sequence()
        for t in range(2, 5):
            s.process_frame(frame, t)
        with self.assertRaisesRegex(ValueError, "too short"):
            s.finish_and_predict()

    def test_timestamps_must_increase(self):
        s = self.session(Detector(pose_result()), Detector(hands_result()))
        frame = np.zeros((240, 320, 3), np.uint8)
        s.process_frame(frame, 5)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            s.process_frame(frame, 5)


if __name__ == "__main__":
    unittest.main()
