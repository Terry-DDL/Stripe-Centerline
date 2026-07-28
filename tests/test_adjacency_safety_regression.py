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
    run_scan,
    summarize_clicked_hypothesis_arbitration,
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

    def test_clicked_hypothesis_decisions_have_stable_scan_outcomes(self):
        decisions = (
            ("trigger_not_met", False, "trigger_not_met"),
            ("no_eligible_alternate", True, "no_verified"),
            ("no_verified_alternate", True, "no_verified"),
            ("unique_verified_adopted", True, "adopted"),
            (
                "ambiguous_verified_clicked_hypotheses",
                True,
                "ambiguous",
            ),
        )
        for decision, attempted, expected_outcome in decisions:
            with self.subTest(decision=decision):
                report = {
                    "candidate_arbitration": {
                        "candidates": {
                            "original_otsu": {
                                "adjacency_verification": {
                                    "clicked_hypothesis_arbitration": {
                                        "attempted": attempted,
                                        "decision": decision,
                                    }
                                }
                            }
                        }
                    }
                }

                summary = summarize_clicked_hypothesis_arbitration(report)

                self.assertEqual(summary["attempted"], attempted)
                self.assertEqual(summary["decision"], decision)
                self.assertEqual(summary["outcome"], expected_outcome)
                self.assertFalse(summary["legacy_missing_field"])

    def test_missing_clicked_hypothesis_field_is_legacy_safe(self):
        summary = summarize_clicked_hypothesis_arbitration(
            {"candidate_arbitration": {"candidates": {}}}
        )

        self.assertEqual(
            summary,
            {
                "attempted": False,
                "decision": "trigger_not_met",
                "outcome": "trigger_not_met",
                "legacy_missing_field": True,
            },
        )

    def test_unknown_clicked_hypothesis_decision_fails_loudly(self):
        report = {
            "candidate_arbitration": {
                "candidates": {
                    "original_otsu": {
                        "adjacency_verification": {
                            "clicked_hypothesis_arbitration": {
                                "attempted": True,
                                "decision": "future_unknown_decision",
                            }
                        }
                    }
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "unknown clicked hypothesis"):
            summarize_clicked_hypothesis_arbitration(report)

    @patch(
        "tools.run_adjacency_safety_scan.git_provenance",
        return_value={"git_commit": "abc123", "git_dirty": False},
    )
    @patch(
        "tools.run_adjacency_safety_scan.build_pitch_reference_map",
        return_value=object(),
    )
    @patch("tools.run_adjacency_safety_scan.interactive_pipeline.run_interactive_case")
    def test_scan_aggregates_one_hypothesis_outcome_per_unique_coordinate(
        self,
        run_case,
        _build_pitch_map,
        _provenance,
    ):
        run_case.return_value = SimpleNamespace(
            report={
                "algorithm_revision": "clicked_hypothesis_v1",
                "interactive_result": {
                    "success": False,
                    "failure_reasons": [
                        "immediate_neighbors_not_verified"
                    ],
                    "left": None,
                    "clicked": None,
                    "right": None,
                    "stripe_spacing_px": None,
                },
                "candidate_arbitration": {
                    "candidates": {
                        "original_otsu": {
                            "adjacency_verification": {
                                "clicked_hypothesis_arbitration": {
                                    "attempted": True,
                                    "decision": "unique_verified_adopted",
                                }
                            }
                        }
                    }
                },
            }
        )

        report = run_scan(np.zeros((1, 1), dtype=np.uint8))

        self.assertEqual(report["unique_coordinate_count_per_brightness"], 605)
        self.assertEqual(report["total_unique_evaluation_count"], 2420)
        self.assertEqual(run_case.call_count, 2420)
        for row in report["brightness_results"]:
            counts = row["clicked_hypothesis_arbitration_counts"]
            self.assertEqual(row["point_count"], 605)
            self.assertEqual(counts["outcome_coordinate_count"], 605)
            self.assertEqual(counts["attempted"], 605)
            self.assertEqual(counts["adopted"], 605)
            self.assertEqual(counts["no_verified"], 0)
            self.assertEqual(counts["ambiguous"], 0)
            self.assertEqual(counts["trigger_not_met"], 0)
            self.assertEqual(counts["legacy_missing_field"], 0)

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
