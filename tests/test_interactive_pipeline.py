"""Unit tests for point-centered ROI and rotation shadow behavior."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np


SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import InteractiveConfig, ProcessingConfig  # noqa: E402
from interactive_pipeline import (  # noqa: E402
    _apply_adjacency_safety_gate,
    _apply_neighbor_consistency_guard,
    _apply_width_aware_adjacency_evidence,
    _build_adjacency_verification,
    _build_shadow_arbitration,
    _clicked_hypothesis_hard_failure,
    _effective_adaptive_block_size,
    _formal_result_failure,
    _recover_weak_neighbors,
    _resolve_alternate_clicked_hypothesis,
    _select_threshold_candidate,
    calculate_interactive_roi_bounds,
    estimate_rotation_shadow,
    run_interactive_case,
    validate_interactive_config,
)
from interactive_analysis import (  # noqa: E402
    InteractiveStripeSelection,
    select_interactive_tracks,
)
from stripe_analysis import analyze_adjacent_stripes  # noqa: E402


class InteractivePipelineTests(unittest.TestCase):
    def hypothesis_track(
        self,
        track_id,
        center_x,
        *,
        width=10.0,
        support=1.0,
        crossing_valid_count=0,
        rejection_reasons=(),
        valid_rows=(0, 119),
    ):
        return SimpleNamespace(
            track_id=track_id,
            center_x_roi=float(center_x),
            median_width_px=float(width),
            valid_row_ratio=float(support),
            crossing_valid_count=crossing_valid_count,
            rejection_reasons=list(rejection_reasons),
            valid_runs=[
                SimpleNamespace(y_roi=y_roi) for y_roi in valid_rows
            ],
        )

    def clicked_hypothesis_candidate(
        self,
        *,
        geometry="original",
        threshold_method="otsu",
        adjacency_status="rejected",
        adjacency_reason="selected_stripes_not_immediate_neighbors",
        clicked_center=110.0,
        clicked_width=30.0,
        extra_tracks=(),
    ):
        left = self.hypothesis_track(1, 80)
        clicked = self.hypothesis_track(
            2,
            clicked_center,
            width=clicked_width,
            crossing_valid_count=100,
        )
        right = self.hypothesis_track(3, 120)
        stable_context = (
            self.hypothesis_track(4, 40),
            self.hypothesis_track(5, 60),
            self.hypothesis_track(6, 140),
        )
        selection = InteractiveStripeSelection(
            click_classification="black_stripe",
            clicked_track=clicked,
            left_track=left,
            right_track=right,
            success=True,
            failure_reasons=(),
            warning_flags=(),
        )
        return SimpleNamespace(
            geometry=geometry,
            threshold_method=threshold_method,
            selection=selection,
            analysis=SimpleNamespace(
                tracks=[
                    *stable_context,
                    left,
                    clicked,
                    right,
                    *extra_tracks,
                ],
                x_ref_roi=100,
                roi_shape=(120, 200),
            ),
            adjacency_verification={
                "status": adjacency_status,
                "reason": adjacency_reason,
            },
            grayscale_topology={
                "strong_same_basin_conflict": False,
                "strong_merged_basin_conflict": False,
            },
            inverse_rotation_matrix=np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
        )

    def resolve_hypothesis_report(self, candidate):
        with patch(
            "interactive_pipeline._attach_clicked_hypothesis_report",
            side_effect=lambda _candidate, report: report,
        ):
            return _resolve_alternate_clicked_hypothesis(
                candidate,
                np.full((120, 200), 128, dtype=np.uint8),
                100,
                100,
                60,
                ProcessingConfig(),
                InteractiveConfig(),
                None,
            )

    def weak_neighbor_case(self, weak_center_x: int, continuous_dark=False):
        height, width = 120, 300
        mask = np.zeros((height, width), dtype=np.uint8)
        image = np.full((height, width), 220, dtype=np.uint8)
        for center_x in (60, 140, 180):
            mask[:, center_x - 5 : center_x + 6] = 255
            image[:, center_x - 5 : center_x + 6] = 20
        for y0, y1 in ((15, 30), (85, 100)):
            mask[y0:y1, weak_center_x - 5 : weak_center_x + 6] = 255
        image[:, weak_center_x - 5 : weak_center_x + 6] = 20
        if continuous_dark:
            image[:, weak_center_x - 5 : 146] = 20
        processing = ProcessingConfig(stripe_search_radius_px=120)
        analysis = analyze_adjacent_stripes(
            mask,
            140,
            140,
            0,
            processing,
        )
        selection = select_interactive_tracks(
            mask,
            140,
            60,
            analysis,
            processing,
            image_gray_roi=image,
        )
        return image, analysis, selection

    def clicked_support_case(self, support_ratio: float):
        height, width = 200, 300
        mask = np.zeros((height, width), dtype=np.uint8)
        image = np.full((height, width), 220, dtype=np.uint8)
        for center_x in (60, 100, 180, 220):
            mask[:, center_x - 2 : center_x + 3] = 255
        for center_x in (60, 100, 140, 180, 220):
            image[:, center_x - 2 : center_x + 3] = 20
        support_rows = round(height * support_ratio)
        mask[:support_rows, 138:143] = 255
        processing = ProcessingConfig(stripe_search_radius_px=120)
        analysis = analyze_adjacent_stripes(
            mask,
            141,
            141,
            0,
            processing,
        )
        selection = select_interactive_tracks(
            mask,
            141,
            100,
            analysis,
            processing,
            image_gray_roi=image,
        )
        guarded, details = _apply_neighbor_consistency_guard(
            selection,
            analysis,
            processing,
            InteractiveConfig(),
        )
        verification = _build_adjacency_verification(
            guarded,
            details,
            {"success": selection.success},
        )
        return selection, verification

    def test_adaptive_block_size_must_be_odd_and_at_least_three(self):
        for invalid_block_size in (1, 2, 30):
            with self.subTest(block_size=invalid_block_size):
                with self.assertRaises(ValueError):
                    validate_interactive_config(
                        InteractiveConfig(
                            adaptive_threshold_block_size=(
                                invalid_block_size
                            )
                        )
                    )

    def test_adaptive_block_size_reduces_for_small_roi(self):
        block_size, warnings = _effective_adaptive_block_size(
            (20, 10),
            InteractiveConfig(adaptive_threshold_block_size=31),
        )

        self.assertEqual(block_size, 9)
        self.assertIn(
            "adaptive_block_size_reduced_for_roi",
            warnings,
        )

    def test_adaptive_is_disabled_when_roi_is_too_small(self):
        block_size, warnings = _effective_adaptive_block_size(
            (3, 30),
            InteractiveConfig(),
        )

        self.assertIsNone(block_size)
        self.assertIn("adaptive_disabled_roi_too_small", warnings)

    def test_shadow_topology_ratios_must_be_between_zero_and_one(self):
        for field_name in (
            "topology_shadow_strong_same_basin_support_ratio",
            "topology_shadow_strong_merged_basin_support_ratio",
        ):
            for invalid_ratio in (-0.1, 1.1):
                with self.subTest(
                    field=field_name,
                    ratio=invalid_ratio,
                ):
                    with self.assertRaises(ValueError):
                        validate_interactive_config(
                            InteractiveConfig(
                                **{field_name: invalid_ratio}
                            )
                        )

    def test_merged_conflict_uses_safe_same_geometry_replacement(self):
        def candidate(geometry, method, merged=False):
            return SimpleNamespace(
                geometry=geometry,
                threshold_method=method,
                grayscale_topology={
                    "strong_same_basin_conflict": False,
                    "strong_merged_basin_conflict": merged,
                },
                quality={"success": True},
                selection=SimpleNamespace(success=True),
            )

        current = candidate("original", "otsu", merged=True)
        original_safe = candidate("original", "adaptive")
        rotated_higher_quality = candidate("rotated", "adaptive")
        original_candidates = [current, original_safe]
        rotated_candidates = [rotated_higher_quality]

        def threshold_winner(candidates):
            if candidates is original_candidates:
                return original_safe, "only_available_threshold_method", [
                    "otsu"
                ]
            return (
                rotated_higher_quality,
                "only_available_threshold_method",
                [],
            )

        with patch(
            "interactive_pipeline._shadow_threshold_winner",
            side_effect=threshold_winner,
        ), patch(
            "interactive_pipeline._select_geometry_candidate",
            return_value=(
                rotated_higher_quality,
                "better_combined_pitch_status",
            ),
        ), patch(
            "interactive_pipeline._safe_merged_basin_replacement",
            side_effect=lambda _current, replacement, *_configs: (
                replacement is original_safe
            ),
        ):
            report, formal, debug = _build_shadow_arbitration(
                original_candidates,
                rotated_candidates,
                current,
                True,
                ProcessingConfig(),
                InteractiveConfig(),
            )

        self.assertIs(formal, original_safe)
        self.assertIs(debug, original_safe)
        self.assertEqual(
            report["hypothetical_winner"],
            {"geometry": "rotated", "threshold_method": "adaptive"},
        )
        self.assertEqual(
            report["final_winner"],
            {"geometry": "original", "threshold_method": "adaptive"},
        )
        self.assertEqual(
            report["geometry_selection_reason"],
            "merged_conflict_safe_same_geometry_replacement",
        )

    def test_shadow_reports_topology_quality_ordering(self):
        def candidate(method, topology_status):
            return SimpleNamespace(
                geometry="original",
                threshold_method=method,
                grayscale_topology={
                    "status": topology_status,
                    "strong_same_basin_conflict": False,
                    "strong_merged_basin_conflict": False,
                },
                quality={"success": True},
                selection=SimpleNamespace(success=True),
            )

        support_winner = candidate("adaptive", "Unable to verify")
        topology_winner = candidate("otsu", "Consistent")

        with patch(
            "interactive_pipeline._shadow_threshold_winner",
            return_value=(
                topology_winner,
                "better_grayscale_topology_status",
                [],
            ),
        ):
            report, formal, debug = _build_shadow_arbitration(
                [support_winner, topology_winner],
                [],
                support_winner,
                True,
                ProcessingConfig(),
                InteractiveConfig(),
            )

        self.assertIs(formal, topology_winner)
        self.assertIs(debug, topology_winner)
        self.assertTrue(report["topology_ordering_applied"])
        self.assertFalse(report["rejection_applied"])
        self.assertTrue(report["would_change_formal_result"])
        self.assertEqual(
            report["final_winner"],
            {"geometry": "original", "threshold_method": "otsu"},
        )

    def test_neighbor_recovery_ratios_must_be_between_zero_and_one(self):
        for field_name in (
            "outer_neighbor_recovery_min_support_ratio",
            "neighbor_recovery_max_pitch_error_ratio",
        ):
            for invalid_ratio in (-0.1, 1.1):
                with self.subTest(
                    field=field_name,
                    ratio=invalid_ratio,
                ):
                    with self.assertRaises(ValueError):
                        validate_interactive_config(
                            InteractiveConfig(
                                **{field_name: invalid_ratio}
                            )
                        )

    def test_clicked_hypothesis_center_distance_must_be_positive(self):
        for invalid_distance in (0.0, -1.0):
            with self.subTest(distance=invalid_distance):
                with self.assertRaises(ValueError):
                    validate_interactive_config(
                        InteractiveConfig(
                            clicked_hypothesis_max_center_distance_px=(
                                invalid_distance
                            )
                        )
                    )

    def test_pitch_and_grayscale_can_recover_a_weak_neighbor(self):
        image, analysis, selection = self.weak_neighbor_case(100)

        recovered, report = _recover_weak_neighbors(
            selection,
            analysis,
            image,
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            {"baseline_pitch_px": 40.0},
            InteractiveConfig(),
        )

        self.assertTrue(report["applied"])
        self.assertEqual(recovered.left_track.center_x_roi, 100.0)
        self.assertIn(
            "weak_neighbor_recovered",
            recovered.warning_flags,
        )

    def test_clicked_track_support_is_monotonic_across_stable_threshold(self):
        for support_ratio in (
            0.215,
            0.249,
            0.25,
            0.28,
            0.35,
            0.499,
            0.5,
        ):
            with self.subTest(support_ratio=support_ratio):
                selection, verification = self.clicked_support_case(
                    support_ratio
                )

                self.assertTrue(selection.success)
                self.assertEqual(
                    selection.clicked_track.center_x_roi,
                    140.0,
                )
                self.assertEqual(
                    verification["status"],
                    "verified",
                )

    def test_clicked_hypothesis_trigger_is_narrowly_scoped(self):
        cases = (
            (
                {"geometry": "rotated"},
                "geometry_not_original",
            ),
            (
                {"threshold_method": "adaptive"},
                "threshold_method_not_otsu",
            ),
            (
                {
                    "adjacency_status": "verified",
                    "adjacency_reason": "immediate_neighbors_verified",
                },
                "original_adjacency_not_rejected",
            ),
            (
                {"clicked_center": 103.0},
                "original_clicked_not_materially_offset",
            ),
            (
                {"clicked_width": 18.0},
                "original_clicked_not_abnormally_wide",
            ),
        )
        for overrides, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                report = self.resolve_hypothesis_report(
                    self.clicked_hypothesis_candidate(**overrides)
                )

                self.assertFalse(report["attempted"])
                self.assertEqual(
                    report["decision"],
                    "trigger_not_met",
                )
                self.assertEqual(
                    report["trigger_reason"],
                    expected_reason,
                )

    def test_clicked_hypothesis_reports_insufficient_weak_evidence(self):
        report = self.resolve_hypothesis_report(
            self.clicked_hypothesis_candidate()
        )

        self.assertTrue(report["attempted"])
        self.assertEqual(
            report["decision"],
            "no_eligible_alternate",
        )
        self.assertEqual(report["verified_hypothesis_count"], 0)
        self.assertIsNone(report["adopted_track_ids"])

    def test_offset_weak_track_cannot_steal_clicked_identity(self):
        offset_weak_track = self.hypothesis_track(
            20,
            104,
            support=0.2,
            crossing_valid_count=20,
            rejection_reasons=("insufficient_row_support",),
        )

        report = self.resolve_hypothesis_report(
            self.clicked_hypothesis_candidate(
                extra_tracks=(offset_weak_track,)
            )
        )

        self.assertTrue(report["attempted"])
        self.assertEqual(
            report["decision"],
            "no_eligible_alternate",
        )
        self.assertEqual(report["hypotheses"], [])

    def test_conflicting_verified_clicked_hypotheses_remain_ambiguous(self):
        weak_left = self.hypothesis_track(
            20,
            99,
            support=0.2,
            crossing_valid_count=20,
            rejection_reasons=("insufficient_row_support",),
        )
        weak_right = self.hypothesis_track(
            21,
            101,
            support=0.2,
            crossing_valid_count=20,
            rejection_reasons=("insufficient_row_support",),
        )
        candidate = self.clicked_hypothesis_candidate(
            extra_tracks=(weak_left, weak_right)
        )

        def verified_evaluation(selection, *_args):
            return {
                "selection": selection,
                "neighbor_consistency": {
                    "checked": True,
                    "passed": True,
                },
                "pitch_guard": {
                    "status": "Normal",
                    "interval_pitch_ratios": [1.0, 1.0],
                },
                "topology": SimpleNamespace(
                    report={
                        "status": "Consistent",
                        "strong_same_basin_conflict": False,
                        "strong_merged_basin_conflict": False,
                    },
                    debug_image=np.zeros((1, 1), dtype=np.uint8),
                ),
                "quality": {
                    "success": True,
                    "hard_invalid_reasons": [],
                },
                "adjacency_verification": {
                    "status": "verified",
                    "reason": "immediate_neighbors_verified",
                },
                "pitch_tolerance_passed": True,
                "verified_for_adoption": True,
                "rejection_reason": None,
            }

        with patch(
            "interactive_pipeline._evaluate_alternate_clicked_selection",
            side_effect=verified_evaluation,
        ), patch(
            "interactive_pipeline._attach_clicked_hypothesis_report",
            side_effect=lambda _candidate, report: report,
        ):
            report = _resolve_alternate_clicked_hypothesis(
                candidate,
                np.full((120, 200), 128, dtype=np.uint8),
                100,
                100,
                60,
                ProcessingConfig(),
                InteractiveConfig(),
                None,
            )

        self.assertTrue(report["attempted"])
        self.assertEqual(report["verified_hypothesis_count"], 2)
        self.assertEqual(
            report["decision"],
            "ambiguous_verified_clicked_hypotheses",
        )
        self.assertEqual(
            report["hard_failure_reason"],
            "ambiguous_verified_clicked_hypotheses",
        )
        self.assertIsNone(report["adopted_track_ids"])

    def width_aware_case(
        self,
        *,
        include_intermediate=False,
        topology_status="Consistent",
        clicked_width=21.0,
        clicked_center=103.0,
    ):
        left = self.hypothesis_track(1, 84, width=9)
        clicked = self.hypothesis_track(
            2,
            clicked_center,
            width=clicked_width,
            crossing_valid_count=120,
        )
        right = self.hypothesis_track(3, 128, width=9)
        tracks = [
            self.hypothesis_track(10, 40, width=9),
            self.hypothesis_track(11, 55, width=9),
            left,
            clicked,
            right,
            self.hypothesis_track(12, 143, width=9),
        ]
        if include_intermediate:
            tracks.append(
                self.hypothesis_track(
                    20,
                    116,
                    width=9,
                    support=0.1,
                    rejection_reasons=("insufficient_row_support",),
                )
            )
        selection = InteractiveStripeSelection(
            click_classification="black_stripe",
            clicked_track=clicked,
            left_track=left,
            right_track=right,
            success=True,
            failure_reasons=(),
            warning_flags=(
                "selected_stripes_not_immediate_neighbors",
            ),
        )
        neighbor = {
            "checked": True,
            "passed": False,
            "local_pitch_px": 14.5,
            "selected_span_px": 25.0,
            "span_pitch_ratio": 1.724,
            "comparison_basis": "center_distance",
            "width_aware_evidence": None,
            "max_span_pitch_ratio": 1.5,
            "pitch_track_count": 5,
            "reason": "selected_span_exceeds_immediate_neighbor_limit",
        }
        topology = {
            "status": topology_status,
            "strong_same_basin_conflict": False,
            "strong_merged_basin_conflict": False,
            "evaluated_intervals": [
                {
                    "left_track_id": 1,
                    "right_track_id": 2,
                    "status": topology_status,
                    "strong_same_basin_conflict": False,
                    "informative_row_count": 120,
                    "separator_support_ratio": 1.0,
                    "vertical_band_coverage": {
                        "covered_band_count": 3,
                    },
                },
                {
                    "left_track_id": 2,
                    "right_track_id": 3,
                    "status": topology_status,
                    "strong_same_basin_conflict": False,
                    "informative_row_count": 120,
                    "separator_support_ratio": 1.0,
                    "vertical_band_coverage": {
                        "covered_band_count": 3,
                    },
                },
            ],
        }
        return (
            selection,
            SimpleNamespace(tracks=tracks, x_ref_roi=100),
            neighbor,
            topology,
        )

    def test_wide_clicked_edge_gaps_can_verify_adjacency(self):
        selection, analysis, neighbor, topology = self.width_aware_case()

        updated_selection, updated = (
            _apply_width_aware_adjacency_evidence(
                selection,
                analysis,
                neighbor,
                topology,
                ProcessingConfig(),
                InteractiveConfig(),
            )
        )

        self.assertTrue(updated["passed"])
        self.assertEqual(
            updated["comparison_basis"],
            "width_normalized_edge_gap",
        )
        self.assertAlmostEqual(updated["span_pitch_ratio"], 1.31)
        evidence = updated["width_aware_evidence"]
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["typical_width"], 9.0)
        self.assertEqual(evidence["regular_max"], 18.75)
        self.assertEqual(evidence["intermediate_track_ids"], [])
        self.assertNotIn(
            "selected_stripes_not_immediate_neighbors",
            updated_selection.warning_flags,
        )

    def test_width_aware_adjacency_rejects_intermediate_track(self):
        selection, analysis, neighbor, topology = self.width_aware_case(
            include_intermediate=True,
        )

        _selection, updated = _apply_width_aware_adjacency_evidence(
            selection,
            analysis,
            neighbor,
            topology,
            ProcessingConfig(),
            InteractiveConfig(),
        )

        self.assertFalse(updated["passed"])
        evidence = updated["width_aware_evidence"]
        self.assertEqual(
            evidence["reason"],
            "intermediate_track_evidence_present",
        )
        self.assertEqual(evidence["intermediate_track_ids"], [20])

    def test_width_aware_adjacency_requires_wide_track_and_topology(self):
        cases = (
            (
                {"clicked_width": 9.0},
                "clicked_track_not_abnormally_wide",
            ),
            (
                {"topology_status": "Contradictory"},
                "topology_not_consistent",
            ),
            (
                {"clicked_center": 119.0},
                "clicked_track_not_centered_on_click",
            ),
        )
        for overrides, expected_reason in cases:
            with self.subTest(reason=expected_reason):
                selection, analysis, neighbor, topology = (
                    self.width_aware_case(**overrides)
                )

                _selection, updated = (
                    _apply_width_aware_adjacency_evidence(
                        selection,
                        analysis,
                        neighbor,
                        topology,
                        ProcessingConfig(),
                        InteractiveConfig(),
                    )
                )

                self.assertFalse(updated["passed"])
                self.assertEqual(
                    updated["width_aware_evidence"]["reason"],
                    expected_reason,
                )

    def test_verified_candidate_beats_higher_support_unverified_or_rejected(self):
        def candidate(method, status, support):
            return SimpleNamespace(
                threshold_method=method,
                quality={
                    "success": True,
                    "adjacency_verification_status": status,
                    "combined_pitch_status": "Normal",
                    "grayscale_topology_status": "Consistent",
                    "minimum_valid_row_ratio": support,
                    "minimum_retention_ratio": support,
                    "maximum_center_mad_px": 0.0,
                    "maximum_normalized_width_mad": 0.0,
                    "click_association_distance_px": 0.0,
                    "suspected_fragment_count": 0,
                    "low_support_track_count": 0,
                },
            )

        for unsafe_status in ("unverified", "rejected"):
            with self.subTest(unsafe_status=unsafe_status):
                verified = candidate("otsu", "verified", 0.55)
                unsafe = candidate("adaptive", unsafe_status, 1.0)

                winner, reason = _select_threshold_candidate(
                    [verified, unsafe]
                )

                self.assertIs(winner, verified)
                self.assertEqual(
                    reason,
                    "better_adjacency_verification",
                )

    def test_half_pitch_or_same_basin_weak_track_is_not_recovered(self):
        for weak_center_x, continuous_dark in (
            (120, False),
            (100, True),
        ):
            with self.subTest(
                weak_center_x=weak_center_x,
                continuous_dark=continuous_dark,
            ):
                image, analysis, selection = self.weak_neighbor_case(
                    weak_center_x,
                    continuous_dark,
                )
                recovered, report = _recover_weak_neighbors(
                    selection,
                    analysis,
                    image,
                    np.array(
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        dtype=np.float64,
                    ),
                    {"baseline_pitch_px": 40.0},
                    InteractiveConfig(),
                )

                self.assertFalse(report["applied"])
                self.assertEqual(
                    recovered.left_track.center_x_roi,
                    60.0,
                )

    def test_weak_neighbor_is_not_recovered_without_pitch(self):
        image, analysis, selection = self.weak_neighbor_case(100)

        recovered, report = _recover_weak_neighbors(
            selection,
            analysis,
            image,
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            {"baseline_pitch_px": None},
            InteractiveConfig(),
        )

        self.assertFalse(report["applied"])
        self.assertEqual(report["reason"], "no_reliable_pitch_baseline")
        self.assertEqual(recovered.left_track.center_x_roi, 60.0)

    def test_neighbor_guard_marks_skipped_stripes_for_rejection(self):
        mask = np.zeros((100, 300), dtype=np.uint8)
        for center_x in (50, 130, 250):
            mask[:, center_x - 5 : center_x + 6] = 255
        for center_x in (170, 210):
            mask[:40, center_x - 5 : center_x + 6] = 255
        processing_config = ProcessingConfig(stripe_search_radius_px=150)
        analysis = analyze_adjacent_stripes(
            mask,
            133,
            133,
            0,
            processing_config,
        )
        selection = select_interactive_tracks(
            mask,
            133,
            50,
            analysis,
            processing_config,
        )

        guarded, details = _apply_neighbor_consistency_guard(
            selection,
            analysis,
            processing_config,
            InteractiveConfig(),
        )

        self.assertTrue(selection.success)
        self.assertTrue(guarded.success)
        self.assertIsNotNone(guarded.left_track)
        self.assertIsNotNone(guarded.right_track)
        self.assertIn(
            "selected_stripes_not_immediate_neighbors",
            guarded.warning_flags,
        )
        self.assertTrue(details["checked"])
        self.assertFalse(details["passed"])
        self.assertGreater(
            details["span_pitch_ratio"],
            InteractiveConfig().neighbor_max_span_pitch_ratio,
        )
        verification = _build_adjacency_verification(
            guarded,
            details,
            {"success": True},
        )
        self.assertEqual(verification["status"], "rejected")
        self.assertEqual(
            verification["reason"],
            "selected_stripes_not_immediate_neighbors",
        )

    def test_neighbor_guard_warns_when_pitch_cannot_be_verified(self):
        mask = np.zeros((100, 320), dtype=np.uint8)
        mask[:, 45:56] = 255
        mask[:, 125:136] = 255
        mask[:, 205:276] = 255
        processing_config = ProcessingConfig(stripe_search_radius_px=160)
        analysis = analyze_adjacent_stripes(
            mask,
            133,
            133,
            0,
            processing_config,
        )
        selection = select_interactive_tracks(
            mask,
            133,
            50,
            analysis,
            processing_config,
        )

        guarded, details = _apply_neighbor_consistency_guard(
            selection,
            analysis,
            processing_config,
            InteractiveConfig(),
        )

        self.assertTrue(selection.success)
        self.assertFalse(details["checked"])
        self.assertTrue(guarded.success)
        self.assertIn(
            "neighbor_consistency_not_verifiable",
            guarded.warning_flags,
        )
        verification = _build_adjacency_verification(
            guarded,
            details,
            {"success": True},
        )
        self.assertEqual(verification["status"], "unverified")

    def test_neighbor_guard_accepts_one_pitch_on_each_side(self):
        mask = np.zeros((100, 300), dtype=np.uint8)
        for center_x in (50, 130, 210):
            mask[:, center_x - 5 : center_x + 6] = 255
        processing_config = ProcessingConfig(stripe_search_radius_px=150)
        analysis = analyze_adjacent_stripes(
            mask,
            133,
            133,
            0,
            processing_config,
        )
        selection = select_interactive_tracks(
            mask,
            133,
            50,
            analysis,
            processing_config,
        )

        guarded, details = _apply_neighbor_consistency_guard(
            selection,
            analysis,
            processing_config,
            InteractiveConfig(),
        )

        self.assertTrue(guarded.success)
        self.assertTrue(details["passed"])
        self.assertEqual(details["span_pitch_ratio"], 1.0)
        verification = _build_adjacency_verification(
            guarded,
            details,
            {"success": True},
        )
        self.assertEqual(verification["status"], "verified")

    def test_adjacency_gate_clears_unverified_formal_candidate(self):
        unsafe_selection = InteractiveStripeSelection(
            click_classification="black_stripe",
            clicked_track=object(),
            left_track=object(),
            right_track=object(),
            success=True,
            failure_reasons=(),
            warning_flags=(),
        )
        candidate = SimpleNamespace(
            geometry="original",
            threshold_method="otsu",
            adjacency_verification={"status": "unverified"},
            selection=unsafe_selection,
        )

        report, formal, debug = _apply_adjacency_safety_gate(
            candidate,
            candidate,
            [candidate],
            None,
        )

        self.assertIsNone(formal)
        self.assertIs(debug, candidate)
        self.assertTrue(report["gate_applied"])
        self.assertEqual(
            report["failure_reason"],
            "immediate_neighbors_not_verified",
        )
        failure = _formal_result_failure(
            candidate,
            report["failure_reason"],
        )
        self.assertFalse(failure.success)
        self.assertIsNone(failure.left_track)
        self.assertIsNone(failure.clicked_track)
        self.assertIsNone(failure.right_track)
        self.assertEqual(
            failure.failure_reasons,
            ("immediate_neighbors_not_verified",),
        )

    def test_adjacency_gate_keeps_verified_formal_candidate(self):
        candidate = SimpleNamespace(
            geometry="original",
            threshold_method="adaptive",
            adjacency_verification={"status": "verified"},
        )

        report, formal, debug = _apply_adjacency_safety_gate(
            candidate,
            candidate,
            [candidate],
            None,
        )

        self.assertIs(formal, candidate)
        self.assertIs(debug, candidate)
        self.assertFalse(report["gate_applied"])
        self.assertIsNone(report["failure_reason"])

    def test_adjacency_gate_always_supplies_a_failure_reason(self):
        debug_candidate = SimpleNamespace(
            geometry="original",
            threshold_method="otsu",
            adjacency_verification={"status": "unverified"},
        )

        report, formal, _debug = _apply_adjacency_safety_gate(
            None,
            debug_candidate,
            [debug_candidate],
            None,
        )

        self.assertIsNone(formal)
        self.assertEqual(
            report["failure_reason"],
            "immediate_neighbors_not_verified",
        )

    def test_ambiguous_verified_clicked_hypotheses_are_a_safe_failure(self):
        ambiguous_selection = InteractiveStripeSelection(
            click_classification="black_stripe",
            clicked_track=object(),
            left_track=object(),
            right_track=object(),
            success=True,
            failure_reasons=(),
            warning_flags=(),
        )
        candidate = SimpleNamespace(
            geometry="original",
            threshold_method="otsu",
            adjacency_verification={
                "status": "unverified",
                "reason": "ambiguous_verified_clicked_hypotheses",
                "clicked_hypothesis_arbitration": {
                    "attempted": True,
                    "decision": (
                        "ambiguous_verified_clicked_hypotheses"
                    ),
                },
            },
            selection=ambiguous_selection,
        )

        hard_failure = _clicked_hypothesis_hard_failure([candidate])
        report, formal, debug = _apply_adjacency_safety_gate(
            candidate,
            candidate,
            [candidate],
            None,
            hard_failure=hard_failure,
        )

        self.assertIsNone(formal)
        self.assertIs(debug, candidate)
        self.assertTrue(report["gate_applied"])
        self.assertTrue(report["hard_failure_applied"])
        self.assertEqual(
            report["failure_reason"],
            "ambiguous_verified_clicked_hypotheses",
        )
        failure = _formal_result_failure(
            candidate,
            report["failure_reason"],
        )
        self.assertFalse(failure.success)
        self.assertIsNone(failure.left_track)
        self.assertIsNone(failure.clicked_track)
        self.assertIsNone(failure.right_track)

    def test_roi_is_fixed_size_away_from_edges(self):
        bounds = calculate_interactive_roi_bounds(
            (2000, 2000),
            1000,
            900,
            InteractiveConfig(),
        )
        self.assertEqual(bounds.width_roi, 500)
        self.assertEqual(bounds.height_roi, 200)
        self.assertEqual((bounds.x0_global, bounds.y0_global), (750, 800))

    def test_roi_clips_without_moving_the_click(self):
        bounds = calculate_interactive_roi_bounds(
            (2000, 2000),
            40,
            30,
            InteractiveConfig(),
        )
        self.assertEqual((bounds.x0_global, bounds.y0_global), (0, 0))
        self.assertEqual((bounds.x1_global, bounds.y1_global), (290, 130))

    def test_rotation_shadow_finds_known_small_tilt(self):
        vertical = np.zeros((300, 300), dtype=np.uint8)
        for center_x in range(30, 280, 25):
            vertical[:, center_x - 4 : center_x + 5] = 255
        tilted = cv2.warpAffine(
            vertical,
            cv2.getRotationMatrix2D((149.5, 149.5), 4.0, 1.0),
            (300, 300),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )

        estimate = estimate_rotation_shadow(tilted, InteractiveConfig())

        self.assertAlmostEqual(estimate.best_angle_deg, -4.0, delta=0.5)

    def test_vertical_pipeline_keeps_original_detection_space(self):
        image = np.full((600, 600), 255, dtype=np.uint8)
        for center_x in (210, 300, 390):
            image[:, center_x - 5 : center_x + 6] = 0
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_interactive_case(
                image,
                303,
                300,
                Path(temporary_directory),
                ProcessingConfig(),
                InteractiveConfig(),
                image_name="synthetic.png",
            )

            self.assertEqual(result.selection.click_classification, "black_stripe")
            self.assertTrue(result.selection.success)
            self.assertFalse(
                result.report["rotation_shadow"]["applied_to_detection"]
            )
            self.assertEqual(
                result.report["interactive_result"]["stripe_spacing_px"],
                180.0,
            )
            self.assertTrue(
                (Path(temporary_directory) / "interactive_results.json").is_file()
            )
            arbitration = result.report["candidate_arbitration"]
            self.assertEqual(
                set(arbitration["candidates"]),
                {
                    "original_otsu",
                    "original_adaptive",
                    "rotated_otsu",
                    "rotated_adaptive",
                },
            )
            self.assertIn(
                result.report["interactive_result"]["threshold_method"],
                ("otsu", "adaptive"),
            )
            self.assertTrue(result.report["rotation_shadow"]["guard_reasons"])
            self.assertIn("unrotated_black_mask.png", result.debug_images)
            self.assertIn(
                "rotation_candidate_black_mask.png",
                result.debug_images,
            )
            self.assertIn(
                "original_otsu_grayscale_topology.png",
                result.debug_images,
            )
            self.assertTrue(
                result.report["shadow_arbitration"]["enforced"]
            )
            self.assertEqual(
                result.report["shadow_arbitration"]["mode"],
                "enforced",
            )

    def test_tilted_pipeline_rotates_before_detection(self):
        vertical = np.full((600, 600), 255, dtype=np.uint8)
        for center_x in (210, 300, 390):
            vertical[:, center_x - 5 : center_x + 6] = 0
        tilted = cv2.warpAffine(
            vertical,
            cv2.getRotationMatrix2D((299.5, 299.5), 4.0, 1.0),
            (600, 600),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_interactive_case(
                tilted,
                300,
                300,
                Path(temporary_directory),
                ProcessingConfig(),
                InteractiveConfig(),
                image_name="tilted.png",
            )

            rotation = result.report["rotation_shadow"]
            self.assertTrue(rotation["applied_to_detection"])
            self.assertAlmostEqual(rotation["applied_angle_deg"], -4.0, delta=0.5)
            self.assertEqual(rotation["detection_space"], "rotated")
            self.assertTrue(result.selection.success)
            left_line = result.report["interactive_result"]["left"][
                "line_endpoints_global"
            ]
            self.assertNotEqual(left_line[0][0], left_line[1][0])

    def test_brightened_saturated_image_keeps_same_neighbors(self):
        image = np.full((600, 600), 120, dtype=np.uint8)
        for center_x in (210, 300, 390):
            image[:, center_x - 5 : center_x + 6] = 0
        brightened = np.clip(
            image.astype(np.int16) + 150,
            0,
            255,
        ).astype(np.uint8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            original = run_interactive_case(
                image,
                303,
                300,
                Path(temporary_directory) / "original",
                ProcessingConfig(),
                InteractiveConfig(),
                image_name="original.png",
            )
            saturated = run_interactive_case(
                brightened,
                303,
                300,
                Path(temporary_directory) / "brightened",
                ProcessingConfig(),
                InteractiveConfig(),
                image_name="brightened.png",
            )

        self.assertTrue(original.selection.success)
        self.assertTrue(saturated.selection.success)
        self.assertEqual(
            original.report["interactive_result"]["left"][
                "center_x_global"
            ],
            saturated.report["interactive_result"]["left"][
                "center_x_global"
            ],
        )
        self.assertEqual(
            original.report["interactive_result"]["right"][
                "center_x_global"
            ],
            saturated.report["interactive_result"]["right"][
                "center_x_global"
            ],
        )

    def test_clipped_roi_does_not_apply_rotation(self):
        vertical = np.full((600, 600), 255, dtype=np.uint8)
        for center_x in (30, 120, 210):
            vertical[:, center_x - 5 : center_x + 6] = 0
        tilted = cv2.warpAffine(
            vertical,
            cv2.getRotationMatrix2D((299.5, 299.5), 4.0, 1.0),
            (600, 600),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_interactive_case(
                tilted,
                30,
                300,
                Path(temporary_directory),
                ProcessingConfig(),
                InteractiveConfig(),
                image_name="clipped.png",
            )

            rotation = result.report["rotation_shadow"]
            self.assertFalse(rotation["applied_to_detection"])
            self.assertIn(
                "rotation_not_applied_to_clipped_roi",
                rotation["guard_reasons"],
            )


if __name__ == "__main__":
    unittest.main()
