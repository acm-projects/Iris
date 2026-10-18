import unittest

import numpy as np

from live_recognizer import MotionSegmenter


class MotionSegmenterTests(unittest.TestCase):
    def setUp(self):
        self.points = np.full((55, 2), 40, dtype=np.float32)
        self.shape = (100, 100, 3)

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


if __name__ == "__main__":
    unittest.main()
