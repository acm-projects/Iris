import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from inference_utils import (
    assemble_model_input,
    class_labels_from_entries,
    load_class_labels,
    prepare_recording,
    remap_landmarks,
)
from landmark_normalization import legacy_affine_keypoints


def pose_result():
    landmarks = [
        SimpleNamespace(x=index / 100, y=-index / 100)
        for index in range(33)
    ]
    return SimpleNamespace(pose_landmarks=[landmarks])


def hand(offset):
    return [
        SimpleNamespace(x=offset + index / 1000, y=offset + index / 500)
        for index in range(21)
    ]


class InferenceUtilityTests(unittest.TestCase):
    def test_remaps_body_and_hands_to_model_order(self):
        left, right = hand(0.1), hand(0.5)
        hands = SimpleNamespace(
            hand_landmarks=[right, left],
            handedness=[
                [SimpleNamespace(category_name="Right", score=0.8)],
                [SimpleNamespace(category_name="Left", score=0.9)],
            ],
        )
        points, detected = remap_landmarks(pose_result(), hands)

        self.assertEqual(points.shape, (55, 2))
        self.assertEqual(points.dtype, np.float32)
        np.testing.assert_allclose(points[1], [0.115, -0.115])
        np.testing.assert_allclose(points[13:34], [[p.x, p.y] for p in left])
        np.testing.assert_allclose(points[34:55], [[p.x, p.y] for p in right])
        self.assertEqual(detected, ["Right", "Left"])

    def test_missing_landmarks_become_legacy_mask_values(self):
        missing = np.full((55, 2), np.nan, dtype=np.float32)
        np.testing.assert_array_equal(
            legacy_affine_keypoints(missing),
            -np.ones((55, 2), dtype=np.float32),
        )

    def test_prepare_recording_centers_or_pads_to_50_frames(self):
        long_records = [
            (index * 33, np.full((55, 2), index, dtype=np.float32))
            for index in range(60)
        ]
        frames, timestamps = prepare_recording(long_records)
        self.assertEqual(frames.shape, (50, 55, 2))
        self.assertEqual(timestamps, [index * 33 for index in range(5, 55)])

        short_records = long_records[:3]
        frames, timestamps = prepare_recording(short_records)
        self.assertEqual(timestamps[3:], [timestamps[2]] * 47)
        np.testing.assert_array_equal(frames[3:], np.repeat(frames[2:3], 47, axis=0))

    def test_pack_preserves_per_joint_interleaved_xy_order(self):
        frames = np.empty((50, 55, 2), dtype=np.float32)
        for frame_index in range(50):
            for joint_index in range(55):
                value = frame_index / 100 + joint_index / 1000
                frames[frame_index, joint_index] = (value, -value)

        packed = assemble_model_input(frames)
        self.assertEqual(packed.shape, (1, 55, 100))
        self.assertEqual(packed.dtype, np.float32)
        np.testing.assert_allclose(
            packed[0, 7, :8],
            [0.007, -0.007, 0.017, -0.017, 0.027, -0.027, 0.037, -0.037],
        )

    def test_pack_rejects_wrong_shape_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, r"Expected \(50, 55, 2\)"):
            assemble_model_input(np.zeros((49, 55, 2), dtype=np.float32))
        frames = np.zeros((50, 55, 2), dtype=np.float32)
        frames[4, 2, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "cannot contain NaN"):
            assemble_model_input(frames)

    def test_loads_alphabetically_sorted_unique_labels(self):
        entries = [{"gloss": f"word_{index:03}"} for index in reversed(range(100))]
        expected = [f"word_{index:03}" for index in range(100)]
        self.assertEqual(class_labels_from_entries(entries), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asl100.json"
            path.write_text(json.dumps(entries), encoding="utf-8")
            self.assertEqual(load_class_labels(path), expected)

    def test_rejects_duplicate_label_map(self):
        with self.assertRaisesRegex(ValueError, "100 unique gloss labels"):
            class_labels_from_entries([{"gloss": "same"} for _ in range(100)])


if __name__ == "__main__":
    unittest.main()
