import unittest

import numpy as np

from live_recognizer import MotionSegmenter, segment_has_sufficient_hands


class MotionSegmenterTests(unittest.TestCase):
    def setUp(self):
        self.points = np.full((55, 2), 40, dtype=np.float32)
        self.shape = (100, 100, 3)

    @staticmethod
    def body_only():
        points = np.full((55, 2), np.nan, dtype=np.float32)
        points[:13] = 40
        return points

    def one_hand(self):
        points = self.body_only()
        points[13:34] = 40
        return points

    @staticmethod
    def move_hands(points, amount=1):
        moved = points.copy()
        finite_hands = np.isfinite(moved[13:]).all(axis=1)
        moved[13:][finite_hands, 0] += amount
        return moved

    def test_motion_then_pause_produces_one_segment(self):
        segmenter = MotionSegmenter(
            start_motion=0.005,
            stop_motion=0.002,
            start_frames=2,
            quiet_frames=3,
            min_frames=6,
            max_frames=20,
        )
        completed = []
        states = []
        timestamp = 0
        for _ in range(2):
            update = segmenter.observe(timestamp, self.points, self.shape)
            states.append(update.state)
            timestamp += 100
        moving = self.points.copy()
        for _ in range(8):
            moving[:, 0] += 1
            update = segmenter.observe(timestamp, moving, self.shape)
            states.append(update.state)
            if update.completed:
                completed.append(update.completed)
            timestamp += 100
        for _ in range(4):
            update = segmenter.observe(timestamp, moving, self.shape)
            states.append(update.state)
            if update.completed:
                completed.append(update.completed)
            timestamp += 100
        self.assertIn("capturing", states)
        self.assertIn("predicting", states)
        self.assertEqual(len(completed), 1)
        self.assertGreaterEqual(len(completed[0]), 6)

    def test_body_only_motion_never_starts_capture(self):
        segmenter = MotionSegmenter(start_frames=1)
        points = self.body_only()
        states = []
        for index in range(20):
            points[:13, 0] += 2
            states.append(segmenter.observe(index * 100, points, self.shape).state)
        self.assertEqual(set(states), {"listening"})

    def test_body_motion_with_stationary_hand_never_starts_capture(self):
        segmenter = MotionSegmenter(start_frames=1)
        points = self.one_hand()
        segmenter.observe(0, points, self.shape)
        states = []
        for index in range(1, 10):
            points[:13, 0] += 2
            states.append(segmenter.observe(index * 100, points, self.shape).state)
        self.assertEqual(set(states), {"listening"})

    def test_one_or_two_moving_hands_start_capture(self):
        for initial in (self.one_hand(), self.points.copy()):
            with self.subTest(hand_count=1 if np.isnan(initial[34:]).all() else 2):
                segmenter = MotionSegmenter(start_motion=0.005, start_frames=2)
                segmenter.observe(0, initial, self.shape)
                moving = initial
                states = []
                for index in range(1, 3):
                    moving = self.move_hands(moving)
                    states.append(
                        segmenter.observe(index * 100, moving, self.shape).state
                    )
                self.assertEqual(states[-1], "capturing")

    def test_two_missing_hand_frames_are_tolerated_and_omitted(self):
        segmenter = MotionSegmenter(
            start_motion=0.005,
            stop_motion=0.002,
            start_frames=1,
            quiet_frames=5,
            min_frames=4,
            max_frames=20,
            pre_roll=1,
            hand_loss_grace_frames=2,
        )
        moving = self.one_hand()
        segmenter.observe(0, moving, self.shape)
        for index in range(1, 5):
            moving = self.move_hands(moving)
            update = segmenter.observe(index * 100, moving, self.shape)
        self.assertEqual(update.state, "capturing")

        missing = self.body_only()
        first = segmenter.observe(500, missing, self.shape)
        second = segmenter.observe(600, missing, self.shape)
        self.assertEqual(first.state, "capturing")
        self.assertEqual(second.state, "capturing")
        self.assertIsNone(first.completed)
        self.assertIsNone(second.completed)

        moving = self.move_hands(moving)
        resumed = segmenter.observe(700, moving, self.shape)
        self.assertEqual(resumed.state, "capturing")
        self.assertEqual(len(segmenter._active), 5)

    def test_prolonged_hand_loss_finishes_valid_segment(self):
        segmenter = MotionSegmenter(
            start_motion=0.005,
            start_frames=1,
            quiet_frames=5,
            min_frames=4,
            pre_roll=1,
            hand_loss_grace_frames=2,
        )
        moving = self.one_hand()
        segmenter.observe(0, moving, self.shape)
        for index in range(1, 5):
            moving = self.move_hands(moving)
            segmenter.observe(index * 100, moving, self.shape)
        missing = self.body_only()
        segmenter.observe(500, missing, self.shape)
        segmenter.observe(600, missing, self.shape)
        completed = segmenter.observe(700, missing, self.shape)
        self.assertEqual(completed.state, "predicting")
        self.assertEqual(len(completed.completed), 4)

    def test_prolonged_hand_loss_cancels_short_segment(self):
        segmenter = MotionSegmenter(
            start_motion=0.005,
            start_frames=1,
            min_frames=4,
            pre_roll=1,
            hand_loss_grace_frames=2,
        )
        moving = self.one_hand()
        segmenter.observe(0, moving, self.shape)
        moving = self.move_hands(moving)
        self.assertEqual(
            segmenter.observe(100, moving, self.shape).state,
            "capturing",
        )
        missing = self.body_only()
        segmenter.observe(200, missing, self.shape)
        segmenter.observe(300, missing, self.shape)
        cancelled = segmenter.observe(400, missing, self.shape)
        self.assertEqual(cancelled.state, "listening")
        self.assertIsNone(cancelled.completed)

    def test_continuous_motion_is_bounded_by_max_frames(self):
        segmenter = MotionSegmenter(
            start_motion=0.001,
            stop_motion=0.0001,
            start_frames=1,
            quiet_frames=3,
            min_frames=3,
            max_frames=8,
            pre_roll=1,
        )
        moving = self.points.copy()
        completed = None
        for index in range(20):
            moving[:, 0] += 1
            update = segmenter.observe(index * 100, moving, self.shape)
            if update.completed:
                completed = update.completed
                break
        self.assertIsNotNone(completed)
        self.assertEqual(len(completed), 8)

    def test_flush_finishes_only_a_complete_active_segment(self):
        segmenter = MotionSegmenter(
            start_motion=0.005,
            start_frames=1,
            min_frames=4,
            pre_roll=1,
        )
        moving = self.one_hand()
        segmenter.observe(0, moving, self.shape)
        for index in range(1, 5):
            moving = self.move_hands(moving)
            segmenter.observe(index * 100, moving, self.shape)
        completed = segmenter.flush()
        self.assertEqual(completed.state, "predicting")
        self.assertEqual(len(completed.completed), 4)
        self.assertEqual(segmenter.flush().state, "listening")

    def test_inference_guard_requires_enough_hand_visible_frames(self):
        hand = self.one_hand()
        body = self.body_only()
        valid = [(index, hand) for index in range(10)]
        insufficient = [(index, hand) for index in range(9)] + [(9, body)]
        low_ratio = [(0, hand), (1, hand), (2, body), (3, body)]

        self.assertTrue(segment_has_sufficient_hands(valid))
        self.assertFalse(segment_has_sufficient_hands(insufficient))
        self.assertFalse(
            segment_has_sufficient_hands(
                low_ratio,
                min_frames=2,
                min_ratio=0.75,
            )
        )


if __name__ == "__main__":
    unittest.main()
