"""Unit tests for point-centered ROI and rotation shadow behavior."""

from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np


SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import InteractiveConfig, ProcessingConfig  # noqa: E402
from interactive_pipeline import (  # noqa: E402
    _apply_neighbor_consistency_guard,
    _effective_adaptive_block_size,
    _recover_weak_neighbors,
    calculate_interactive_roi_bounds,
    estimate_rotation_shadow,
    run_interactive_case,
    validate_interactive_config,
)
from interactive_analysis import select_interactive_tracks  # noqa: E402
from stripe_analysis import analyze_adjacent_stripes  # noqa: E402


class InteractivePipelineTests(unittest.TestCase):
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
        for invalid_ratio in (-0.1, 1.1):
            with self.subTest(ratio=invalid_ratio):
                with self.assertRaises(ValueError):
                    validate_interactive_config(
                        InteractiveConfig(
                            topology_shadow_strong_same_basin_support_ratio=(
                                invalid_ratio
                            )
                        )
                    )

    def test_neighbor_recovery_ratios_must_be_between_zero_and_one(self):
        for field_name in (
            "neighbor_recovery_min_support_ratio",
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

    def test_neighbor_guard_warns_without_clearing_skipped_stripes(self):
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
