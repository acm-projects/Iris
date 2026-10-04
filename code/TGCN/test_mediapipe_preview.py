import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mediapipe_preview import (
    assemble_model_input,
    class_labels_from_entries,
    format_landmark_debug,
    load_class_labels,
    normalize_landmarks,
    prepare_recording,
    remap_landmarks,
)


def make_pose_result():
    landmarks = [
        SimpleNamespace(x=index / 100, y=-index / 100)
        for index in range(33)
    ]
    return SimpleNamespace(pose_landmarks=[landmarks])


def make_hand(side_offset):
    return [
        SimpleNamespace(
            x=side_offset + index / 1000,
            y=side_offset + index / 500,
        )
        for index in range(21)
    ]


def make_detection(side, score, landmarks):
    category = SimpleNamespace(category_name=side, score=score)
    return landmarks, [category]


class RemapLandmarksTests(unittest.TestCase):
    def test_maps_body_landmarks_in_model_order_and_computes_midpoints(self):
        points, detected_sides = remap_landmarks(
            make_pose_result(),
            SimpleNamespace(hand_landmarks=[], handedness=[]),
        )

        expected = np.array([
            [0.00, 0.00],                 # Nose
            [0.115, -0.115],              # Shoulder midpoint
            [0.12, -0.12],                # Right shoulder
            [0.14, -0.14],                # Right elbow
            [0.16, -0.16],                # Right wrist
            [0.11, -0.11],                # Left shoulder
            [0.13, -0.13],                # Left elbow
            [0.15, -0.15],                # Left wrist
            [0.235, -0.235],              # Hip midpoint
            [0.05, -0.05],                # Right eye
            [0.02, -0.02],                # Left eye
            [0.08, -0.08],                # Right ear
            [0.07, -0.07],                # Left ear
        ], dtype=np.float32)

        self.assertEqual(points.shape, (55, 2))
        self.assertEqual(points.dtype, np.float32)
        np.testing.assert_allclose(points[:13], expected)
        self.assertTrue(np.isnan(points[13:]).all())
        self.assertEqual(detected_sides, [])

    def test_assigns_all_hand_points_by_handedness_not_detection_order(self):
        right_hand = make_hand(0.5)
        left_hand = make_hand(0.1)
        hand_result = SimpleNamespace(
            hand_landmarks=[right_hand, left_hand],
            handedness=[
                [SimpleNamespace(category_name="Right", score=0.8)],
                [SimpleNamespace(category_name="Left", score=0.9)],
            ],
        )

        points, detected_sides = remap_landmarks(
            SimpleNamespace(pose_landmarks=[]),
            hand_result,
        )

        self.assertEqual(points.shape, (55, 2))
        self.assertEqual(points.dtype, np.float32)
        self.assertTrue(np.isnan(points[:13]).all())
        np.testing.assert_allclose(
            points[13:34],
            [[landmark.x, landmark.y] for landmark in left_hand],
        )
        np.testing.assert_allclose(
            points[34:55],
            [[landmark.x, landmark.y] for landmark in right_hand],
        )
        self.assertEqual(detected_sides, ["Right", "Left"])

    def test_leaves_absent_pose_and_hands_as_nan(self):
        points, detected_sides = remap_landmarks(
            SimpleNamespace(pose_landmarks=[]),
            SimpleNamespace(hand_landmarks=[], handedness=[]),
        )

        self.assertEqual(points.shape, (55, 2))
        self.assertEqual(points.dtype, np.float32)
        self.assertTrue(np.isnan(points).all())
        self.assertEqual(detected_sides, [])

    def test_ignores_unknown_unclassified_and_incomplete_hand_detections(self):
        valid_left = make_hand(0.2)
        hand_result = SimpleNamespace(
            hand_landmarks=[
                make_hand(0.7),
                make_hand(0.8),
                make_hand(0.9)[:20],
                valid_left,
            ],
            handedness=[
                [SimpleNamespace(category_name="Center", score=1.0)],
                [],
                [SimpleNamespace(category_name="Right", score=1.0)],
                [SimpleNamespace(category_name="Left", score=0.7)],
            ],
        )

        points, detected_sides = remap_landmarks(
            SimpleNamespace(pose_landmarks=[]),
            hand_result,
        )

        np.testing.assert_allclose(
            points[13:34],
            [[landmark.x, landmark.y] for landmark in valid_left],
        )
        self.assertTrue(np.isnan(points[:13]).all())
        self.assertTrue(np.isnan(points[34:]).all())
        self.assertEqual(detected_sides, ["Center", "Right", "Left"])

    def test_uses_highest_scoring_detection_for_duplicate_handedness(self):
        lower_score_hand = make_hand(0.1)
        higher_score_hand = make_hand(0.4)
        hand_result = SimpleNamespace(
            hand_landmarks=[lower_score_hand, higher_score_hand],
            handedness=[
                [SimpleNamespace(category_name="Left", score=0.4)],
                [SimpleNamespace(category_name="Left", score=0.9)],
            ],
        )

        points, _ = remap_landmarks(
            SimpleNamespace(pose_landmarks=[]),
            hand_result,
        )

        np.testing.assert_allclose(
            points[13:34],
            [[landmark.x, landmark.y] for landmark in higher_score_hand],
        )
        self.assertTrue(np.isnan(points[34:]).all())

    def test_normalizes_coordinates_and_maps_missing_points_to_training_sentinel(self):
        points = np.array([[0.0, 0.5], [1.0, np.nan]], dtype=np.float32)
        points = np.concatenate(
            [points, np.zeros((53, 2), dtype=np.float32)],
            axis=0,
        )

        normalized = normalize_landmarks(points)

        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_allclose(normalized[0], [-1.0, 0.0])
        np.testing.assert_allclose(normalized[1], [-1.0, -1.0])
        self.assertTrue(np.isfinite(normalized).all())

    def test_normalization_rejects_wrong_landmark_shape(self):
        with self.assertRaisesRegex(ValueError, "Expected \\(55, 2\\)"):
            normalize_landmarks(np.zeros((54, 2), dtype=np.float32))

    def test_assembles_50_frames_with_joint_and_xy_order_preserved(self):
        normalized_frames = np.empty((50, 55, 2), dtype=np.float32)
        for frame_index in range(50):
            for joint_index in range(55):
                normalized_frames[frame_index, joint_index] = (
                    frame_index / 100 + joint_index / 1000,
                    -(frame_index / 100 + joint_index / 1000),
                )

        model_input = assemble_model_input(normalized_frames)

        self.assertEqual(model_input.shape, (1, 55, 100))
        self.assertEqual(model_input.dtype, np.float32)
        self.assertTrue(np.isfinite(model_input).all())
        np.testing.assert_allclose(
            model_input[0, 7, :8],
            [
                0.007, -0.007,
                0.017, -0.017,
                0.027, -0.027,
                0.037, -0.037,
            ],
        )
        np.testing.assert_allclose(
            model_input[0, 7, -2:],
            [0.497, -0.497],
        )

    def test_missing_frame_keeps_its_temporal_position(self):
        normalized_frames = np.zeros((50, 55, 2), dtype=np.float32)
        normalized_frames[17] = normalize_landmarks(
            np.full((55, 2), np.nan, dtype=np.float32)
        )

        model_input = assemble_model_input(normalized_frames)

        self.assertEqual(model_input.shape, (1, 55, 100))
        np.testing.assert_allclose(model_input[0, 0, 34:36], [-1.0, -1.0])
        np.testing.assert_allclose(model_input[0, 0, 36:38], [0.0, 0.0])

    def test_prepare_recording_selects_centered_consecutive_frames(self):
        records = [
            (index * 33, np.full((55, 2), index, dtype=np.float32))
            for index in range(60)
        ]

        frames, timestamps = prepare_recording(records)

        self.assertEqual(frames.shape, (50, 55, 2))
        self.assertEqual(timestamps, [index * 33 for index in range(5, 55)])
        np.testing.assert_allclose(frames[:, 0, 0], np.arange(5, 55))

    def test_prepare_recording_pads_last_frame_and_timestamp(self):
        records = [
            (index * 40, np.full((55, 2), index, dtype=np.float32))
            for index in range(3)
        ]

        frames, timestamps = prepare_recording(records)

        self.assertEqual(frames.shape, (50, 55, 2))
        self.assertEqual(timestamps[:3], [0.0, 40.0, 80.0])
        self.assertEqual(timestamps[3:], [80.0] * 47)
        np.testing.assert_allclose(frames[:3, 0, 0], [0, 1, 2])
        np.testing.assert_allclose(frames[3:, 0, 0], [2] * 47)

    def test_prepare_recording_keeps_missing_frames_and_rejects_empty(self):
        missing = np.full((55, 2), np.nan, dtype=np.float32)
        frames, timestamps = prepare_recording([(12, missing)])

        self.assertEqual(timestamps, [12.0] * 50)
        self.assertTrue(np.isnan(frames).all())
        with self.assertRaisesRegex(ValueError, "No frames were recorded"):
            prepare_recording([])

    def test_prepare_recording_rejects_invalid_frame_shape(self):
        with self.assertRaisesRegex(ValueError, "landmark shape"):
            prepare_recording([(0, np.zeros((54, 2), dtype=np.float32))])

    def test_model_input_rejects_wrong_frame_count_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "Expected \\(50, 55, 2\\)"):
            assemble_model_input(np.zeros((49, 55, 2), dtype=np.float32))

        frames = np.zeros((50, 55, 2), dtype=np.float32)
        frames[12, 4, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "cannot contain NaN"):
            assemble_model_input(frames)

    def test_class_mapping_sorts_glosses_and_loads_split_file(self):
        entries = [{"gloss": f"word_{index:03}"} for index in reversed(range(100))]
        expected = [f"word_{index:03}" for index in range(100)]
        self.assertEqual(class_labels_from_entries(entries), expected)

        with tempfile.TemporaryDirectory() as directory:
            split_file = Path(directory) / "asl100.json"
            split_file.write_text(json.dumps(entries), encoding="utf-8")
            self.assertEqual(load_class_labels(split_file), expected)

    def test_class_mapping_rejects_wrong_count_and_duplicate_labels(self):
        with self.assertRaisesRegex(ValueError, "Expected 100 gloss labels"):
            class_labels_from_entries([{"gloss": "only"}])

        duplicate_entries = [{"gloss": "same"} for _ in range(100)]
        with self.assertRaisesRegex(ValueError, "100 unique gloss labels"):
            class_labels_from_entries(duplicate_entries)

    def test_debug_summary_reports_order_labels_missing_counts_and_coordinates(self):
        points = np.full((55, 2), np.nan, dtype=np.float32)
        normalized = normalize_landmarks(points)
        summary = format_landmark_debug(points, normalized, ["Right", "Left"])

        self.assertIn("Detected labels: ['Right', 'Left']", summary)
        self.assertIn("Missing body/L/R: 13/21/21", summary)
        self.assertIn("L0[13]=(-1.000,-1.000)", summary)
        self.assertIn("R20[54]=(-1.000,-1.000)", summary)


if __name__ == "__main__":
    unittest.main()
