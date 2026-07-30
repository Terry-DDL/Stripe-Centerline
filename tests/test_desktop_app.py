"""Tests for desktop UI helpers that do not require opening a window."""

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.desktop_app import (  # noqa: E402
    AnalysisCompletion,
    DesktopSelectionState,
    basin_table_rows,
    build_output_dir,
    calculate_display_size,
    completion_matches_active_request,
    create_failure_result_overlay,
    create_result_roi_crop,
    create_magnifier_display,
    cross_image_experiment_enabled,
    desktop_debug_enabled,
    extract_centered_region,
    get_or_build_pitch_reference,
    magnifier_canvas_position,
    map_display_point_to_source,
    mouse_wheel_scroll_pixels,
    resize_for_display,
    result_performance_text,
    result_metric_values,
    reliable_tk_runtime,
    signed_16_bit,
    pitch_safety_note,
    pitch_safety_text,
    parse_desktop_arguments,
    track_table_rows,
    unpack_touchpad_scroll_delta,
    version_numbers,
    zoom_bounds,
)


class DesktopAppHelperTests(unittest.TestCase):
    def test_release_debug_switch_is_off_by_default(self):
        self.assertFalse(desktop_debug_enabled(False, ""))
        self.assertFalse(desktop_debug_enabled(False, "0"))

    def test_release_debug_switch_accepts_cli_or_environment(self):
        self.assertTrue(desktop_debug_enabled(True, ""))
        self.assertTrue(desktop_debug_enabled(False, "true"))
        self.assertTrue(parse_desktop_arguments(["--debug"]).debug)

    def test_cross_image_experiment_is_default_off_and_explicit_only(self):
        self.assertFalse(cross_image_experiment_enabled(""))
        self.assertFalse(cross_image_experiment_enabled("0"))
        self.assertTrue(cross_image_experiment_enabled("1"))
        self.assertTrue(cross_image_experiment_enabled("true"))

    def test_old_macos_tk_is_rejected(self):
        self.assertFalse(reliable_tk_runtime("8.6.12", "darwin"))
        self.assertTrue(reliable_tk_runtime("8.6.13", "darwin"))
        self.assertTrue(reliable_tk_runtime("9.0.4", "darwin"))

    def test_tk_check_does_not_restrict_other_platforms(self):
        self.assertTrue(reliable_tk_runtime("8.6.12", "linux"))

    def test_version_numbers_accept_patchlevel_text(self):
        self.assertEqual(version_numbers("8.6.13"), (8, 6, 13))

    def test_signed_16_bit_handles_positive_and_negative_values(self):
        self.assertEqual(signed_16_bit(7), 7)
        self.assertEqual(signed_16_bit(0xFFF9), -7)

    def test_touchpad_scroll_delta_unpacks_both_axes(self):
        packed = ((3 & 0xFFFF) << 16) | (-7 & 0xFFFF)

        self.assertEqual(unpack_touchpad_scroll_delta(packed), (3, -7))

    def test_touchpad_scroll_delta_accepts_signed_32_bit_value(self):
        packed = ((-3 & 0xFFFF) << 16) | (7 & 0xFFFF)
        signed_packed = packed - 0x100000000

        self.assertEqual(
            unpack_touchpad_scroll_delta(signed_packed),
            (-3, 7),
        )

    def test_mouse_wheel_delta_maps_to_pixel_movement(self):
        self.assertEqual(mouse_wheel_scroll_pixels(120), -54)
        self.assertEqual(mouse_wheel_scroll_pixels(-120), 54)
        self.assertEqual(mouse_wheel_scroll_pixels(1), -54)

    def test_display_size_preserves_aspect_ratio(self):
        self.assertEqual(
            calculate_display_size((2056, 2452), 1100),
            (1100, 922),
        )

    def test_resize_scales_map_back_to_source_dimensions(self):
        image = np.zeros((2056, 2452), dtype=np.uint8)

        display, scale_x, scale_y = resize_for_display(image, 1100)

        self.assertEqual(display.shape, (922, 1100))
        self.assertAlmostEqual(display.shape[1] * scale_x, 2452)
        self.assertAlmostEqual(display.shape[0] * scale_y, 2056)

    def test_display_click_maps_to_clipped_source_coordinate(self):
        mapped = map_display_point_to_source(
            1100,
            922,
            2452 / 1100,
            2056 / 922,
            (2056, 2452),
        )

        self.assertEqual(mapped, (2451, 2055))

    def test_centered_region_keeps_fixed_size_at_top_left_edge(self):
        image = np.arange(25, dtype=np.uint8).reshape(5, 5)

        region = extract_centered_region(image, (0, 0), 4, 4)

        self.assertEqual(region.shape, (4, 4))
        self.assertEqual(region[0, 0], image[0, 0])
        self.assertEqual(region[2, 2], image[0, 0])
        self.assertEqual(region[-1, -1], image[1, 1])

    def test_centered_region_keeps_fixed_size_at_bottom_right_edge(self):
        image = np.arange(25, dtype=np.uint8).reshape(5, 5)

        region = extract_centered_region(image, (4, 4), 4, 4)

        self.assertEqual(region.shape, (4, 4))
        self.assertEqual(region[2, 2], image[4, 4])
        self.assertEqual(region[-1, -1], image[4, 4])

    def test_centered_region_keeps_point_centered_at_every_edge(self):
        image = np.arange(25, dtype=np.uint8).reshape(5, 5)
        points = ((0, 2), (4, 2), (2, 0), (2, 4), (4, 0), (0, 4))

        for point in points:
            with self.subTest(point=point):
                region = extract_centered_region(image, point, 4, 4)
                self.assertEqual(region.shape, (4, 4))
                self.assertEqual(region[2, 2], image[point[1], point[0]])

    def test_magnifier_display_is_four_times_source_size(self):
        image = np.zeros((100, 100), dtype=np.uint8)

        display = create_magnifier_display(image, (50, 50), 80, 60, 4.0)

        self.assertEqual(display.shape, (240, 320, 3))
        self.assertTupleEqual(tuple(display[120, 160]), (255, 0, 0))

    def test_magnifier_defaults_to_pointer_bottom_right(self):
        position = magnifier_canvas_position(
            100,
            100,
            320,
            240,
            1100,
            900,
            18,
        )

        self.assertEqual(position, (118, 118))

    def test_magnifier_flips_at_canvas_bottom_right(self):
        position = magnifier_canvas_position(
            1000,
            800,
            320,
            240,
            1100,
            900,
            18,
        )

        self.assertEqual(position, (662, 542))

    def test_magnifier_flips_each_axis_independently(self):
        right_edge = magnifier_canvas_position(
            1000, 100, 320, 240, 1100, 900, 18
        )
        bottom_edge = magnifier_canvas_position(
            100, 800, 320, 240, 1100, 900, 18
        )

        self.assertEqual(right_edge, (662, 118))
        self.assertEqual(bottom_edge, (118, 542))

    def test_zoom_bounds_clip_at_image_edge(self):
        self.assertEqual(
            zoom_bounds((2056, 2452), (50, 40)),
            (0, 0, 170, 140),
        )

    def test_output_directory_keeps_existing_layout(self):
        output = build_output_dir(
            "Sample 1.bmp",
            b"abc",
            (1374, 1631),
            Path("/tmp/results"),
        )

        self.assertEqual(
            output,
            Path(
                "/tmp/results/Sample_1_ba7816bf/point_1374_1631"
            ),
        )

    def test_result_roi_crop_uses_global_bounds(self):
        overlay = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)
        result = type(
            "Result",
            (),
            {
                "bounds_global": type(
                    "Bounds",
                    (),
                    {
                        "x0_global": 2,
                        "x1_global": 7,
                        "y0_global": 1,
                        "y1_global": 5,
                    },
                )(),
                "debug_images": {
                    "original_interactive_result.png": overlay
                },
            },
        )()

        cropped = create_result_roi_crop(result)

        np.testing.assert_array_equal(cropped, overlay[1:5, 2:7])

    def test_failed_result_metrics_hide_all_candidate_measurements(self):
        values = result_metric_values(
            {"x_global": 662, "y_global": 1065},
            {
                "success": False,
                "left": {"distance_to_click_px": 14.5},
                "right": {"distance_to_click_px": 50.0},
                "stripe_spacing_px": 64.5,
                "pitch_guard": {"status": "Suspicious"},
            },
        )

        self.assertEqual(
            values,
            (("Reference point", "(662, 1065)", ""),),
        )

    def test_stage3_metrics_show_only_frozen_pitch_evidence(self):
        values = result_metric_values(
            {"x_global": 270, "y_global": 100},
            {
                "result_source": "stage3_1",
                "success": True,
                "left": {"distance_to_click_px": 30.0},
                "right": {"distance_to_click_px": 30.0},
                "stripe_spacing_px": 60.0,
                "pitch_evidence": {
                    "confidence": "medium",
                    "success_eligible": False,
                    "diagnostic_pitch_px": 59.0,
                    "usable_pitch_px": None,
                    "harmonic_ambiguity": {"detected": False},
                },
            },
        )

        self.assertEqual("Raw pitch evidence", values[-1][0])
        self.assertEqual("Medium · diagnostics only", values[-1][1])
        self.assertIn("59", values[-1][2])

    def test_performance_row_is_present_for_success_and_unavailable(self):
        for success in (True, False):
            with self.subTest(success=success):
                self.assertEqual(
                    "Analysis: 55 ms · Total: 144 ms",
                    result_performance_text(
                        {"success": success},
                        55.168,
                        143.865,
                    ),
                )

    def test_performance_row_safely_formats_missing_or_invalid_values(self):
        for analysis_ms, total_ms in (
            (None, None),
            (float("nan"), float("inf")),
            (-1.0, True),
        ):
            with self.subTest(
                analysis_ms=analysis_ms,
                total_ms=total_ms,
            ):
                self.assertEqual(
                    "Analysis: — · Total: —",
                    result_performance_text(
                        {"success": False},
                        analysis_ms,
                        total_ms,
                    ),
                )

    def test_rapid_click_completion_requires_click_and_run_id(self):
        completion = AnalysisCompletion(
            analysis_key=("image-a", (482, 704)),
            result=object(),
            error=None,
            run_id="request-old",
            click_started_ns=1,
            output_dir=Path("/tmp/request-old"),
            performance_metadata={},
            stage3_algorithm_ms=55.0,
        )

        self.assertFalse(
            completion_matches_active_request(
                completion,
                ("image-a", (482, 704)),
                "request-new",
            )
        )
        self.assertFalse(
            completion_matches_active_request(
                completion,
                ("image-a", (661, 921)),
                "request-old",
            )
        )
        self.assertTrue(
            completion_matches_active_request(
                completion,
                ("image-a", (482, 704)),
                "request-old",
            )
        )

    def test_failed_result_overlay_has_roi_and_click_but_no_centerlines(self):
        image = np.full((30, 40), 128, dtype=np.uint8)
        result = type(
            "Result",
            (),
            {
                "bounds_global": type(
                    "Bounds",
                    (),
                    {
                        "x0_global": 5,
                        "x1_global": 35,
                        "y0_global": 4,
                        "y1_global": 26,
                    },
                )(),
                "report": {
                    "click": {"x_global": 20, "y_global": 15}
                },
            },
        )()

        overlay = create_failure_result_overlay(image, result)

        self.assertTrue(np.any(np.all(overlay == (0, 255, 0), axis=2)))
        self.assertTrue(np.any(np.all(overlay == (0, 0, 255), axis=2)))
        self.assertFalse(np.any(np.all(overlay == (255, 0, 0), axis=2)))
        self.assertFalse(np.any(np.all(overlay == (0, 255, 255), axis=2)))

    def test_new_image_clears_selection_and_result(self):
        state = DesktopSelectionState(
            image_identity="old.bmp:1234",
            zoom_center=(100, 200),
            click_point=(101, 201),
            result=object(),
        )

        changed = state.select_image("new.bmp:5678")

        self.assertTrue(changed)
        self.assertEqual(state.image_identity, "new.bmp:5678")
        self.assertIsNone(state.zoom_center)
        self.assertIsNone(state.click_point)
        self.assertIsNone(state.result)

    def test_same_image_keeps_selection(self):
        state = DesktopSelectionState(
            image_identity="same.bmp:1234",
            zoom_center=(100, 200),
            click_point=(101, 201),
        )

        changed = state.select_image("same.bmp:1234")

        self.assertFalse(changed)
        self.assertEqual(state.zoom_center, (100, 200))
        self.assertEqual(state.click_point, (101, 201))

    def test_direct_reference_selection_sets_both_points_and_clears_result(self):
        state = DesktopSelectionState(
            zoom_center=(100, 200),
            click_point=(101, 201),
            result=object(),
        )

        state.select_reference_point((300, 400))

        self.assertEqual(state.zoom_center, (300, 400))
        self.assertEqual(state.click_point, (300, 400))
        self.assertIsNone(state.result)

    def test_track_table_rows_keep_success_and_failure_cells(self):
        rows = track_table_rows(
            {
                "left": {
                    "center_x_global": 100.5,
                    "distance_to_click_px": 20.5,
                    "valid_row_ratio": 1.0,
                    "retention_ratio": 0.9,
                },
                "right": None,
            }
        )

        self.assertEqual(rows[0]["side"], "left")
        self.assertEqual(rows[0]["center_x_global"], 100.5)
        self.assertEqual(rows[1]["side"], "right")
        self.assertEqual(rows[1]["center_x_global"], "")

    def test_basin_table_rows_do_not_claim_legacy_track_support(self):
        rows = basin_table_rows(
            {
                "left": {
                    "center_x_global": 240.0,
                    "distance_to_click_px": 30.0,
                    "basin_width_px": 40.0,
                    "evidence_status": "verified",
                },
                "right": None,
            }
        )

        self.assertEqual("verified", rows[0]["evidence_status"])
        self.assertEqual(40.0, rows[0]["basin_width_px"])
        self.assertEqual("", rows[1]["center_x_global"])

    def test_pitch_reference_is_built_once_per_image(self):
        calls = []

        def builder(image, processing_config, interactive_config):
            calls.append(image)
            return object()

        cache = {}
        image = np.zeros((10, 10), dtype=np.uint8)
        first = get_or_build_pitch_reference(
            cache,
            "same-image",
            image,
            builder=builder,
        )
        second = get_or_build_pitch_reference(
            cache,
            "same-image",
            image,
            builder=builder,
        )

        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)

    def test_pitch_safety_text_keeps_status_and_ratios_compact(self):
        guard = {
            "status": "Suspicious",
            "baseline_pitch_px": 36.0,
            "observed_intervals_px": [21.0],
            "interval_pitch_ratios": [0.583],
            "normal_interval_ratio_min": 0.67,
            "normal_interval_ratio_max": 1.5,
        }

        status, details = pitch_safety_text(guard)
        note = pitch_safety_note(guard)

        self.assertEqual(status, "Suspicious")
        self.assertEqual(
            details,
            "Average pitch: 36 px · Interval / average: 21 px (0.583×)",
        )
        self.assertIn("normal range is 0.67–1.5×", note)


if __name__ == "__main__":
    unittest.main()
