"""Tests for read-only adjacency diagnostic helpers."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.export_candidate_diagnostics import (  # noqa: E402
    _candidate_loss_stage,
    brighten_with_diagnostics,
    diagnose_loss_stage,
    ensure_project_module_origins,
    morphology_statistics,
)
from tools.compare_candidate_diagnostics import (  # noqa: E402
    compare_reports,
)


class CandidateDiagnosticTests(unittest.TestCase):
    def test_brightness_diagnostics_report_actual_clipping(self):
        image = np.array([[100, 110], [200, 255]], dtype=np.uint8)

        brightened, ratio = brighten_with_diagnostics(image, 150)

        self.assertEqual(ratio, 0.75)
        np.testing.assert_array_equal(
            brightened,
            np.array([[250, 255], [255, 255]], dtype=np.uint8),
        )

    def test_morphology_statistics_compare_threshold_and_close(self):
        stages = SimpleNamespace(
            threshold_binary=np.array(
                [[0, 255], [0, 0]],
                dtype=np.uint8,
            ),
            vertical_close=np.array(
                [[0, 255], [0, 255]],
                dtype=np.uint8,
            ),
            close_delta=np.array(
                [[0, 0], [0, 255]],
                dtype=np.uint8,
            ),
            black_mask=np.array(
                [[255, 0], [255, 0]],
                dtype=np.uint8,
            ),
        )

        diagnostics = morphology_statistics(stages)

        self.assertEqual(diagnostics["pixel_count"], 4)
        self.assertEqual(
            diagnostics["threshold_white_pixel_count"],
            1,
        )
        self.assertEqual(
            diagnostics["post_close_white_pixel_count"],
            2,
        )
        self.assertEqual(
            diagnostics["threshold_to_close_changed_pixel_count"],
            1,
        )
        self.assertEqual(
            diagnostics["post_close_black_mask_pixel_ratio"],
            0.5,
        )

    def test_loss_stage_follows_generation_to_verification_order(self):
        def row(
            candidate_id,
            selected=False,
            verified=False,
            eligible=False,
            generated=False,
        ):
            return {
                "candidate_id": candidate_id,
                "selection_matches_ground_truth": selected,
                "adjacency_verification": (
                    {"status": "verified" if verified else "rejected"}
                    if selected
                    else None
                ),
                "eligible_correct_track_set_exists": eligible,
                "correct_track_set_exists": generated,
            }

        scenarios = (
            (
                [row("verified", selected=True, verified=True)],
                "verified_candidate_exists",
            ),
            (
                [row("rejected", selected=True)],
                "adjacency_verification",
            ),
            (
                [row("combination", eligible=True, generated=True)],
                "candidate_combination",
            ),
            (
                [row("eligibility", generated=True)],
                "track_eligibility",
            ),
            (
                [row("preprocessing")],
                "preprocessing_or_track_generation",
            ),
        )

        for rows, expected_stage in scenarios:
            with self.subTest(expected_stage=expected_stage):
                self.assertEqual(
                    diagnose_loss_stage(rows)["stage"],
                    expected_stage,
                )
        self.assertEqual(
            diagnose_loss_stage(
                [row("legacy", selected=True)],
                formal_success=True,
            )["stage"],
            "not_applicable_formal_success",
        )

    def test_candidate_loss_stage_is_role_aware(self):
        row = {
            "selection_matches_ground_truth": False,
            "expected_track_evidence": {
                label: {"matching_track_ids": [index]}
                for index, label in enumerate(
                    ("left", "clicked", "right"),
                    start=1,
                )
            },
            "selected_role_matches_ground_truth": {
                "left": True,
                "clicked": False,
                "right": True,
            },
            "role_eligibility": {
                "clicked_pixel_is_black": True,
                "roles": {
                    "left": [
                        {
                            "stable_selection_eligible": False,
                            "meets_outer_recovery_min_support": True,
                        }
                    ],
                    "right": [
                        {
                            "stable_selection_eligible": False,
                            "meets_outer_recovery_min_support": True,
                        }
                    ],
                },
            },
        }

        stage = _candidate_loss_stage(row)

        self.assertEqual(stage["stage"], "clicked_track_association")
        self.assertTrue(stage["clicked_pixel_is_black"])

    def test_candidate_loss_stage_separates_threshold_from_close(self):
        row = {
            "selection_matches_ground_truth": False,
            "expected_track_evidence": {
                "left": {
                    "matching_track_ids": [],
                    "preprocessing_column_evidence": {
                        "available": True,
                        "center_column": {
                            "threshold_black_row_count": 0,
                            "post_close_black_row_count": 0,
                        },
                    },
                    "post_close_run_evidence": {
                        "accepted_run_count": 0,
                    },
                }
            },
            "selected_role_matches_ground_truth": {"left": False},
            "role_eligibility": {
                "clicked_pixel_is_black": False,
                "roles": {"left": []},
            },
        }

        stage = _candidate_loss_stage(row)

        self.assertEqual(stage["stage"], "threshold_representation")
        self.assertEqual(stage["label_stages"], {"left": "threshold_representation"})

    def test_module_origin_guard_rejects_cached_other_worktree(self):
        cached_module = SimpleNamespace(
            __file__="/private/tmp/other/src/interactive_pipeline.py"
        )

        with patch.dict(
            sys.modules,
            {"interactive_pipeline": cached_module},
        ):
            with self.assertRaisesRegex(RuntimeError, "fresh process"):
                ensure_project_module_origins(PROJECT_ROOT)

    def test_comparison_identifies_retained_correct_rejected_candidate(self):
        def candidate(adjacency):
            return {
                "candidate_id": "original_otsu",
                "selection_matches_ground_truth": True,
                "selected_tracks": {
                    "left": {"center_x_global_at_click_y": 10.0},
                    "clicked": {"center_x_global_at_click_y": 20.0},
                    "right": {"center_x_global_at_click_y": 30.0},
                },
                "adjacency_verification": adjacency,
                "neighbor_consistency": {
                    "checked": True,
                    "passed": False,
                },
                "quality_success": True,
                "quality_hard_invalid_reasons": [],
                "grayscale_topology": {"status": "Consistent"},
            }

        baseline_case = {
            "case_id": "case",
            "image_path": "image.bmp",
            "click": {"x": 20, "y": 30},
            "ground_truth": {},
            "formal_result": {
                "success": True,
                "geometry": "original",
                "threshold_method": "otsu",
            },
            "loss_stage": {"stage": "legacy"},
            "adjacency_arbitration": None,
            "candidates": [candidate(None)],
        }
        current_case = {
            "case_id": "case",
            "image_path": "image.bmp",
            "click": {"x": 20, "y": 30},
            "ground_truth": {},
            "formal_result": {
                "success": False,
                "geometry": None,
                "threshold_method": None,
            },
            "loss_stage": {"stage": "adjacency_verification"},
            "adjacency_arbitration": {
                "formal_winner_before_gate": {
                    "geometry": "original",
                    "threshold_method": "otsu",
                },
                "debug_winner": {
                    "geometry": "original",
                    "threshold_method": "otsu",
                },
            },
            "candidates": [
                candidate(
                    {
                        "status": "rejected",
                        "reason": (
                            "selected_stripes_not_immediate_neighbors"
                        ),
                    }
                )
            ],
        }
        baseline = {
            "git_commit": "baseline",
            "git_dirty": False,
            "suite": "s10",
            "brightness_offset": 0,
            "center_tolerance_px": 3.0,
            "cases": [baseline_case],
        }
        current = {
            "git_commit": "current",
            "git_dirty": False,
            "suite": "s10",
            "brightness_offset": 0,
            "center_tolerance_px": 3.0,
            "cases": [current_case],
        }

        comparison = compare_reports(baseline, current)

        self.assertEqual(comparison["correct_to_safe_failure_count"], 1)
        self.assertEqual(
            comparison["cases"][0]["interpretation"],
            (
                "correct_candidate_retained_but_rejected_by_"
                "adjacency_verification"
            ),
        )
        with self.assertRaisesRegex(ValueError, "brightness_offset"):
            compare_reports(
                baseline,
                {**current, "brightness_offset": 50},
            )


if __name__ == "__main__":
    unittest.main()
