"""Safety-focused regressions for the formal adjacency result contract."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.run_ground_truth_regression import run_regression  # noqa: E402
from tools.run_adjacency_safety_scan import (  # noqa: E402
    build_coordinate_plan,
    classify_formal_result,
    clipped_pixel_ratio,
    git_provenance,
)


class AdjacencySafetyRegressionTests(unittest.TestCase):
    def test_scan_plan_counts_overlapping_coordinates_once(self):
        coordinate_plan = build_coordinate_plan()

        self.assertEqual(len(coordinate_plan), 605)
        self.assertEqual(
            sum(len(point["anchor_ids"]) for point in coordinate_plan),
            765,
        )
        self.assertEqual(
            sum(
                len(point["anchor_ids"]) > 1
                for point in coordinate_plan
            ),
            160,
        )
        overlap = next(
            point
            for point in coordinate_plan
            if (point["x_global"], point["y_global"]) == (662, 1040)
        )
        self.assertEqual(
            overlap["anchor_ids"],
            ["sample2_662_1046", "sample2_662_1065"],
        )

    def test_scan_plan_rejects_conflicting_truth_for_same_coordinate(self):
        anchors = (
            {
                "id": "first",
                "click": (100, 100),
                "left_center_x": 80.0,
                "right_center_x": 120.0,
            },
            {
                "id": "second",
                "click": (100, 100),
                "left_center_x": 81.0,
                "right_center_x": 120.0,
            },
        )

        with self.assertRaisesRegex(ValueError, "conflicting ground truth"):
            build_coordinate_plan(anchors)

    def test_clipped_pixel_ratio_counts_only_pixels_changed_by_clip(self):
        image = np.array([[100, 110], [200, 255]], dtype=np.uint8)

        self.assertEqual(clipped_pixel_ratio(image, 0), 0.0)
        self.assertEqual(clipped_pixel_ratio(image, 150), 0.75)

    @patch("tools.run_adjacency_safety_scan.subprocess.run")
    def test_scan_provenance_records_commit_and_dirty_state(self, run):
        run.side_effect = (
            SimpleNamespace(stdout="abc123\n"),
            SimpleNamespace(stdout=" M tools/scan.py\n"),
        )

        provenance = git_provenance(PROJECT_ROOT)

        self.assertEqual(provenance["git_commit"], "abc123")
        self.assertTrue(provenance["git_dirty"])

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
