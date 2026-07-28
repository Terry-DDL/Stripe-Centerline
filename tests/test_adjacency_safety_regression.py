"""Safety-focused regressions for the formal adjacency result contract."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.run_ground_truth_regression import run_regression  # noqa: E402
from tools.run_adjacency_safety_scan import (  # noqa: E402
    classify_formal_result,
)


class AdjacencySafetyRegressionTests(unittest.TestCase):
    def test_scan_classifier_distinguishes_safe_failure_from_wrong_success(self):
        safe_failure = {
            "success": False,
            "left": None,
            "clicked": None,
            "right": None,
            "stripe_spacing_px": None,
        }
        wrong_success = {
            "success": True,
            "left": {"center_x_global": 647.5},
            "right": {"center_x_global": 712.0},
            "stripe_spacing_px": 64.5,
        }

        self.assertEqual(
            classify_formal_result(
                safe_failure,
                623.5,
                701.0,
                3.0,
            ),
            "safe_failure",
        )
        self.assertEqual(
            classify_formal_result(
                wrong_success,
                623.5,
                701.0,
                3.0,
            ),
            "wrong_success",
        )

    def test_stripe_9_10_ground_truth_has_no_wrong_success(self):
        truth_path = (
            PROJECT_ROOT
            / "tests"
            / "data"
            / "stripe_09_10_ground_truth.json"
        )
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        missing_images = [
            case["image_path"]
            for case in truth["cases"]
            if not (PROJECT_ROOT / case["image_path"]).is_file()
        ]
        if missing_images:
            self.skipTest("missing Stripe 9/10 regression images")

        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "interactive_pipeline.save_debug_image"
        ), patch("interactive_pipeline._save_report"):
            result = run_regression(
                truth,
                Path(temporary_directory),
            )

        metrics = result["metrics"]
        self.assertEqual(metrics["false_success_rate"], 0.0)
        self.assertEqual(metrics["error_among_success_rate"], 0.0)
        self.assertEqual(metrics["screenshot_wrong_success_count"], 0)
        self.assertFalse(
            any(row["false_success"] for row in metrics["rows"])
        )
        for case_id in ("S10-11", "S10-12", "S10-14"):
            self.assertFalse(result["predictions"][case_id]["success"])
            self.assertIsNone(
                result["predictions"][case_id]["left_center_x"]
            )
            self.assertIsNone(
                result["predictions"][case_id]["right_center_x"]
            )


if __name__ == "__main__":
    unittest.main()
