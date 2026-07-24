"""Tests for reviewed nearest-neighbor safety metrics."""

from pathlib import Path
import sys
import unittest


SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from ground_truth_evaluation import (  # noqa: E402
    evaluate_ground_truth_predictions,
    validate_ground_truth_document,
)


def document():
    return {
        "schema_version": 1,
        "center_tolerance_px": 3.0,
        "cases": [
            {
                "id": "valid",
                "click": {"x": 100, "y": 200},
                "ground_truth": {
                    "review_status": "confirmed",
                    "valid_pair": True,
                    "left_center_x": 90.0,
                    "right_center_x": 110.0,
                },
            },
            {
                "id": "negative",
                "click": {"x": 300, "y": 400},
                "ground_truth": {
                    "review_status": "confirmed",
                    "valid_pair": False,
                },
            },
        ],
    }


class GroundTruthEvaluationTests(unittest.TestCase):
    def test_pending_case_is_allowed_only_before_metric_evaluation(self):
        pending = document()
        pending["cases"][0]["ground_truth"]["review_status"] = "pending"

        validate_ground_truth_document(pending)
        with self.assertRaisesRegex(ValueError, "not confirmed"):
            evaluate_ground_truth_predictions(pending, {})

    def test_false_success_includes_wrong_pair_and_negative_case(self):
        predictions = {
            "valid": {
                "success": True,
                "left_center_x": 80.0,
                "right_center_x": 110.0,
                "threshold_method": "adaptive",
                "geometry": "original",
                "pitch_status": "Normal",
            },
            "negative": {
                "success": True,
                "left_center_x": 290.0,
                "right_center_x": 310.0,
                "threshold_method": "otsu",
                "geometry": "original",
                "pitch_status": "Suspicious",
            },
        }

        metrics = evaluate_ground_truth_predictions(document(), predictions)

        self.assertEqual(metrics["detection_success_rate"], 1.0)
        self.assertEqual(metrics["correct_nearest_neighbor_rate"], 0.0)
        self.assertEqual(metrics["false_success_rate"], 1.0)
        self.assertEqual(metrics["error_among_success_rate"], 1.0)
        self.assertEqual(metrics["suspicious_among_success_rate"], 0.5)
        self.assertFalse(metrics["quality_gate"]["passed"])

    def test_correct_pair_within_tolerance_is_counted(self):
        predictions = {
            "valid": {
                "success": True,
                "left_center_x": 92.5,
                "right_center_x": 107.0,
                "threshold_method": "adaptive",
                "geometry": "rotated",
                "pitch_status": "Normal",
            },
            "negative": {
                "success": False,
                "threshold_method": None,
                "geometry": None,
                "pitch_status": "Not applicable",
            },
        }

        metrics = evaluate_ground_truth_predictions(document(), predictions)

        self.assertEqual(metrics["correct_nearest_neighbor_rate"], 1.0)
        self.assertEqual(metrics["false_success_rate"], 0.0)
        self.assertEqual(metrics["left_center_error_px"]["mae"], 2.5)
        self.assertEqual(metrics["right_center_error_px"]["mae"], 3.0)
        self.assertEqual(metrics["geometry_counts"], {"rotated": 1})
        self.assertEqual(
            metrics["threshold_method_rates"],
            {"adaptive": 1.0},
        )
        self.assertEqual(metrics["geometry_rates"], {"rotated": 1.0})
        self.assertTrue(metrics["quality_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
