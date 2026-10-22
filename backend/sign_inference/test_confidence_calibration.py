import tempfile
import unittest
from pathlib import Path

from confidence_calibration import (
    ConfidenceCalibration,
    LabelledPrediction,
    choose_threshold,
    fit_calibration,
)


class ConfidenceCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            LabelledPrediction("bed", "bed", 0.90, "bed-1.mp4"),
            LabelledPrediction("bed", "bed", 0.80, "bed-2.mp4"),
            LabelledPrediction("bed", "bed", 0.70, "bed-3.mp4"),
            LabelledPrediction("table", "bed", 0.75, "table-1.mp4"),
            LabelledPrediction("table", "bed", 0.20, "table-2.mp4"),
            LabelledPrediction("medicine", "medicine", 0.95, "medicine-1.mp4"),
            LabelledPrediction("medicine", "medicine", 0.85, "medicine-2.mp4"),
            LabelledPrediction("bed", "medicine", 0.10, "bed-4.mp4"),
        ]

    def test_threshold_uses_labelled_correctness_not_score_distribution(self):
        selected = choose_threshold(
            [row for row in self.rows if row.predicted == "bed"],
            target_precision=0.8,
            min_true_positives=2,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.threshold, 0.8)
        self.assertEqual(selected.true_positives, 2)
        self.assertEqual(selected.false_positives, 0)

    def test_fit_is_deterministic_and_uses_global_fallback(self):
        first = fit_calibration(
            self.rows,
            model="Pose-TGCN WLASL-100",
            preprocessing="wlasl_affine_256_v1",
            dataset="team-camera-v1",
            target_precision=0.8,
            min_true_positives=2,
        )
        second = fit_calibration(
            reversed(self.rows),
            model="Pose-TGCN WLASL-100",
            preprocessing="wlasl_affine_256_v1",
            dataset="team-camera-v1",
            target_precision=0.8,
            min_true_positives=2,
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.threshold_for("bed"), (0.8, "per-gloss"))
        self.assertEqual(first.threshold_for("table")[1], "global")
        self.assertTrue(first.evaluate("bed", 0.81)["accepted"])
        self.assertFalse(first.evaluate("bed", 0.79)["accepted"])

    def test_versioned_artifact_round_trips(self):
        fitted = fit_calibration(
            self.rows,
            model="Pose-TGCN WLASL-100",
            preprocessing="wlasl_affine_256_v1",
            dataset="team-camera-v1",
            target_precision=0.8,
            min_true_positives=2,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibration.json"
            fitted.save(path)
            loaded = ConfidenceCalibration.load(path)
        self.assertEqual(loaded.to_dict(), fitted.to_dict())

    def test_rejects_unusable_labelled_dataset(self):
        wrong = [
            LabelledPrediction("bed", "table", 0.9, "wrong-1.mp4"),
            LabelledPrediction("bed", "table", 0.8, "wrong-2.mp4"),
        ]
        with self.assertRaisesRegex(ValueError, "cannot meet"):
            fit_calibration(
                wrong,
                model="Pose-TGCN WLASL-100",
                preprocessing="wlasl_affine_256_v1",
                dataset="bad",
                min_true_positives=1,
            )


if __name__ == "__main__":
    unittest.main()
