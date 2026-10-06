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


class PivotNormalizationTests(unittest.TestCase):
    def setUp(self):
        indices = np.arange(55, dtype=np.float32)
        self.points = np.column_stack((100 + indices * 3, 250 - indices * 2))
        self.points[1] = (180, 120)
        self.points[2] = (220, 120)
        self.points[5] = (140, 120)

    def test_pivots_are_zero_and_scale_references_have_unit_length(self):
        actual = normalize_landmarks(self.points)
        for pivot in (1, 13, 34):
            np.testing.assert_array_equal(actual[pivot], [0, 0])
        for a, b in ((2, 5), (13, 22), (34, 43)):
            self.assertAlmostEqual(float(np.linalg.norm(actual[a] - actual[b])), 1, places=6)
        np.testing.assert_allclose(actual[2], [0.5, 0], atol=1e-6)
        np.testing.assert_allclose(actual[5], [-0.5, 0], atol=1e-6)
        np.testing.assert_allclose(actual[0], [-1, 1.625], atol=1e-6)

    def test_translation_and_uniform_scale_preserve_all_xy_coordinates(self):
        expected = normalize_landmarks(self.points)
        for scale in (0.25, 0.5, 2, 4):
            with self.subTest(scale=scale):
                actual = normalize_landmarks(self.points * scale + [35, -20])
                np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_each_hand_can_move_and_scale_independently(self):
        expected = normalize_landmarks(self.points)
        for start, end in ((13, 34), (34, 55)):
            with self.subTest(hand=start):
                points = self.points.copy()
                points[start:end] = points[start:end] * 2.5 + [70, -90]
                np.testing.assert_allclose(normalize_landmarks(points), expected, atol=1e-6)

    def test_missing_or_degenerate_references_mask_only_their_group(self):
        expected = normalize_landmarks(self.points)
        for start, end, pivot, a, b in ((0, 13, 1, 2, 5),
                                      (13, 34, 13, 13, 22),
                                      (34, 55, 34, 34, 43)):
            for failure in ('pivot', 'scale', 'zero scale'):
                with self.subTest(group=start, failure=failure):
                    points = self.points.copy()
                    if failure == 'pivot':
                        points[pivot] = [np.nan, 0]
                    elif failure == 'scale':
                        points[b] = [0, np.inf]
                    else:
                        points[b] = points[a]
                    actual = normalize_landmarks(points)
                    np.testing.assert_array_equal(actual[start:end], -np.ones((end-start, 2)))
                    other = np.ones(55, dtype=bool)
                    other[start:end] = False
                    np.testing.assert_array_equal(actual[other], expected[other])
                    self.assertTrue(np.isfinite(actual).all())

    def test_missing_nonreference_point_is_masked_without_mutation(self):
        self.points[18] = [np.nan, 42]
        original = self.points.copy()
        actual = normalize_landmarks(self.points)
        np.testing.assert_array_equal(actual[18], [-1, -1])
        np.testing.assert_array_equal(actual[13], [0, 0])
        np.testing.assert_array_equal(self.points, original)
        self.assertEqual(actual.dtype, np.float32)

    def test_training_json_and_live_coordinates_use_identical_normalization(self):
        from sign_dataset import read_pose_file
        retained = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18]
        body = np.zeros((25, 3), dtype=np.float32)
        body[retained, :2] = self.points[:13]
        body[retained, 2] = 1
        left = np.column_stack((self.points[13:34], np.ones(21)))
        right = np.column_stack((self.points[34:], np.ones(21)))
        # Training and MediaPipe both mark this point as missing.
        left[5] = 0
        points = self.points.copy()
        points[18] = np.nan
        content = {'people': [{
            'pose_keypoints_2d': body.ravel().tolist(),
            'hand_left_keypoints_2d': left.ravel().tolist(),
            'hand_right_keypoints_2d': right.ravel().tolist(),
        }]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image_00001_keypoints.json'
            path.write_text(json.dumps(content), encoding='utf-8')
            actual = read_pose_file(str(path)).numpy()
        np.testing.assert_allclose(actual, normalize_landmarks(points), atol=1e-6)

    def test_pixel_threshold_for_all_groups_and_dataset_parity(self):
        from landmark_normalization import MIN_SCALE_PIXELS
        from sign_dataset import read_pose_file
        groups = ((0, 13, 2, 5), (13, 34, 13, 22), (34, 55, 34, 43))
        for start, end, a, b in groups:
            for distance in (1e-5, 0.5, MIN_SCALE_PIXELS, 2.0):
                with self.subTest(group=start, distance=distance):
                    points = self.points.copy()
                    points[a] = [100, 100]
                    points[b] = [100 + distance, 100]
                    original = points.copy()
                    actual = normalize_landmarks(points)
                    baseline = normalize_landmarks(self.points)
                    other = np.ones(55, dtype=bool)
                    other[start:end] = False
                    np.testing.assert_array_equal(actual[other], baseline[other])
                    if distance < MIN_SCALE_PIXELS:
                        np.testing.assert_array_equal(actual[start:end], -np.ones((end-start, 2)))
                    else:
                        self.assertAlmostEqual(float(np.linalg.norm(actual[a]-actual[b])), 1)
                    self.assertTrue(np.isfinite(actual).all())
                    self.assertEqual(actual.dtype, np.float32)
                    np.testing.assert_array_equal(points, original)
                    body = np.ones((25, 3), dtype=np.float32)
                    body[[0,1,2,3,4,5,6,7,8,15,16,17,18], :2] = points[:13]
                    content = {'people': [{
                        'pose_keypoints_2d': body.ravel().tolist(),
                        'hand_left_keypoints_2d': np.column_stack((points[13:34], np.ones(21))).ravel().tolist(),
                        'hand_right_keypoints_2d': np.column_stack((points[34:], np.ones(21))).ravel().tolist(),
                    }]}
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / 'pose.json'
                        path.write_text(json.dumps(content))
                        np.testing.assert_array_equal(read_pose_file(path).numpy(), actual)


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_compatible_incompatible_unknown_and_missing_audits(self):
        from checkpoint_compatibility import PreprocessingAudit, require_compatible
        require_compatible(PreprocessingAudit('compatible', 'Synthetic reviewed fixture'))
        for status in ('incompatible', 'unknown', 'invalid'):
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, 'predictions blocked'):
                require_compatible(PreprocessingAudit(status, 'Synthetic fixture'))
        with self.assertRaises(RuntimeError):
            require_compatible(None)

    def test_synthetic_compatible_model_can_predict(self):
        import torch
        from mediapipe_preview import predict_top_words
        from checkpoint_compatibility import PreprocessingAudit
        class FixtureModel:
            preprocessing_audit = PreprocessingAudit('compatible', 'Synthetic reviewed fixture')
            def __call__(self, inputs):
                self.shape = tuple(inputs.shape)
                return torch.arange(100, dtype=torch.float32).reshape(1, 100)
        model = FixtureModel()
        result = predict_top_words(model, [str(i) for i in range(100)], 'cpu',
                                   np.zeros((1,55,100), dtype=np.float32))
        self.assertEqual(model.shape, (1,55,100))
        self.assertEqual([word for word, score in result], ['99', '98', '97'])

    def test_actual_checkpoint_is_blocked_before_loading_weights(self):
        from mediapipe_preview import load_asl100_model
        with self.assertRaisesRegex(RuntimeError, 'unknown.*predictions blocked'):
            load_asl100_model()

    def test_prediction_gate_blocks_before_model_execution(self):
        from mediapipe_preview import predict_top_words
        from checkpoint_compatibility import PreprocessingAudit
        for status in ('unknown', 'incompatible'):
            model = SimpleNamespace(preprocessing_audit=PreprocessingAudit(status, 'fixture'))
            with self.assertRaisesRegex(RuntimeError, 'predictions blocked'):
                predict_top_words(model, ['word'] * 100, 'cpu', np.zeros((1,55,100), dtype=np.float32))


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
