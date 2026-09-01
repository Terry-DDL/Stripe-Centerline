"""Directed tests for the frozen Stage 3.1 desktop production contract."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tests.test_basin_shadow_integration import (
    available_shadow_result,
    basin,
    separator_candidate,
    unavailable_shadow_result,
)
from tools import desktop_performance as performance
from tools.desktop_app import calculate_interactive_roi_bounds
from tools.lightweight_profile import ProfileSession, activate, stage
from tools import stage3_desktop_runtime as runtime
from tools.windows_benchmark_runner import bundled_image_path
from config import INTERACTIVE_CONFIG


def bounds():
    return SimpleNamespace(
        x0_global=20,
        y0_global=10,
        x1_global=480,
        y1_global=190,
    )


class Stage3DesktopRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixed_image = cv2.imread(
            str(bundled_image_path()),
            cv2.IMREAD_GRAYSCALE,
        )
        if cls.fixed_image is None:
            raise RuntimeError("fixed benchmark image could not be decoded")

    def _fixed_result(self, click_x: int, click_y: int):
        fixed_bounds = calculate_interactive_roi_bounds(
            self.fixed_image.shape,
            click_x,
            click_y,
            INTERACTIVE_CONFIG,
        )
        reference = {"x": click_x, "y": click_y}
        stage3 = runtime.run_frozen_stage3(
            self.fixed_image,
            reference,
            fixed_bounds,
        )
        return runtime.build_desktop_result(
            self.fixed_image,
            "fixed-benchmark.bmp",
            reference,
            fixed_bounds,
            stage3,
            Path("/tmp/not-written"),
        )

    def test_lightweight_profile_separates_nested_self_time(self):
        session = ProfileSession()
        with activate(session):
            with stage("outer", "geometry_validation"):
                with stage("inner", "preprocessing"):
                    pass

        report = session.report()
        self.assertEqual(1, report["call_counts"]["outer"])
        self.assertEqual(1, report["call_counts"]["inner"])
        self.assertIn("geometry_validation", report["categories_ms"])
        self.assertIn("preprocessing", report["categories_ms"])

    def _partial_stage3_result(self, failing_side: str) -> dict:
        stage3 = available_shadow_result()
        stage3.update(
            success=False,
            status="unavailable",
            unavailable_reason="basin_structure_not_verified",
            final_hypothesis=None,
        )
        candidates = stage3["debug"]["separator_result"]["candidates"]
        candidates.insert(0, separator_candidate("C00", [160.0] * 3))
        for candidate in candidates:
            candidate.update(support_fraction=1.0, mean_step_px=0.0)
        failing_id = "C00" if failing_side == "left" else "C03"
        next(item for item in candidates if item["candidate_id"] == failing_id)[
            "support_fraction"
        ] = 0.25
        stage3["debug"]["separator_result"]["arbitration_debug"] = {
            "role_hypotheses": [
                {
                    "type": "clicked_basin_roles",
                    "selection_candidate_ids": {
                        "left_adjacent": "C00",
                        "left_clicked_boundary": "C01",
                        "right_clicked_boundary": "C02",
                        "right_adjacent": "C03",
                    },
                }
            ]
        }
        stage3["debug"]["basin_graph"]["basin_candidates"] = [
            basin("B_C00_C01", "C00", "C01", 160.0, 200.0),
            basin("B_C01_C02", "C01", "C02", 200.0, 240.0),
            basin("B_C02_C03", "C02", "C03", 240.0, 320.0),
        ]
        for item in stage3["debug"]["basin_graph"]["basin_candidates"]:
            item["verified"] = True
        return stage3

    def test_unreliable_rotation_cannot_diverge_from_reported_basins(self):
        stage3 = available_shadow_result()
        weak_rotation = SimpleNamespace(
            best_angle_deg=5.0,
            zero_angle_score=0.10,
            best_score=0.105,
            peak_separation=0.001,
        )
        stages = SimpleNamespace(
            vertical_close=np.zeros((180, 460), dtype=np.uint8),
            threshold_warning_flags=(),
        )
        with (
            patch.object(runtime, "_preprocess_roi", return_value=stages),
            patch.object(
                runtime,
                "estimate_rotation_shadow",
                return_value=weak_rotation,
            ),
        ):
            angle = runtime._estimate_formal_line_angle_deg(  # noqa: SLF001
                np.zeros((200, 500), dtype=np.uint8),
                bounds(),
                stage3,
            )

        self.assertEqual(0.0, angle)

    def test_final_lines_follow_existing_angle_through_reference_centers(self):
        image = np.zeros((200, 500), dtype=np.uint8)
        stage3 = available_shadow_result()
        with patch.object(
            runtime,
            "_estimate_formal_line_angle_deg",
            return_value=5.0,
        ):
            result = runtime.build_desktop_result(
                image,
                "synthetic.bmp",
                {"x": 270, "y": 100},
                bounds(),
                stage3,
                Path("/tmp/not-written"),
            )

        interactive = result.report["interactive_result"]
        self.assertEqual(240.0, interactive["left"]["center_x_global"])
        self.assertEqual(300.0, interactive["right"]["center_x_global"])
        self.assertEqual(30.0, interactive["left"]["distance_to_click_px"])
        self.assertEqual(30.0, interactive["right"]["distance_to_click_px"])
        self.assertEqual(60.0, interactive["stripe_spacing_px"])

        overlay = result.debug_images[runtime.FORMAL_OVERLAY_FILENAME]
        self.assertTrue(
            np.any(
                np.all(
                    overlay[100, 239:242] == (255, 0, 0),
                    axis=1,
                )
            )
        )
        self.assertTrue(
            np.any(
                np.all(
                    overlay[100, 299:302] == (0, 255, 255),
                    axis=1,
                )
            )
        )
        self.assertTrue(
            np.any(
                np.all(
                    overlay[10, 247:250] == (255, 0, 0),
                    axis=1,
                )
            )
        )
        self.assertTrue(
            np.any(
                np.all(
                    overlay[189, 231:234] == (255, 0, 0),
                    axis=1,
                )
            )
        )

    def test_zero_angle_keeps_final_lines_vertical(self):
        image = np.zeros((200, 500), dtype=np.uint8)
        stage3 = available_shadow_result()
        with patch.object(
            runtime,
            "_estimate_formal_line_angle_deg",
            return_value=0.0,
        ):
            result = runtime.build_desktop_result(
                image,
                "synthetic.bmp",
                {"x": 270, "y": 100},
                bounds(),
                stage3,
                Path("/tmp/not-written"),
            )

        overlay = result.debug_images[runtime.FORMAL_OVERLAY_FILENAME]
        self.assertTrue(
            np.any(
                np.all(
                    overlay[10, 239:242] == (255, 0, 0),
                    axis=1,
                )
            )
        )
        self.assertTrue(
            np.any(
                np.all(
                    overlay[189, 239:242] == (255, 0, 0),
                    axis=1,
                )
            )
        )

    def test_success_uses_one_atomic_stage3_geometry(self):
        image = np.zeros((200, 500), dtype=np.uint8)
        stage3 = available_shadow_result()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            result = runtime.build_desktop_result(
                image,
                "synthetic.bmp",
                {"x": 270, "y": 100},
                bounds(),
                stage3,
                output_dir,
            )
            runtime.persist_formal_result(result)

            interactive = result.report["interactive_result"]
            self.assertTrue(interactive["success"])
            self.assertTrue(interactive["bilateral_success"])
            self.assertEqual(240.0, interactive["left"]["center_x_global"])
            self.assertEqual(300.0, interactive["right"]["center_x_global"])
            self.assertEqual(30.0, interactive["left"]["distance_to_click_px"])
            self.assertEqual(30.0, interactive["right"]["distance_to_click_px"])
            self.assertEqual(
                "B_C01_C02",
                interactive["left"]["basin_id"],
            )
            self.assertEqual(
                "B_C02_C03",
                interactive["right"]["basin_id"],
            )
            self.assertEqual(
                ["C01", "C02", "C03"],
                interactive["geometry"]["separator_sequence"],
            )
            self.assertTrue(interactive["geometry"]["atomic"])
            saved = json.loads(
                (output_dir / runtime.FORMAL_REPORT_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(runtime.RESULT_SOURCE, saved["result_source"])
            overlay = cv2.imread(
                str(output_dir / runtime.FORMAL_OVERLAY_FILENAME)
            )
            self.assertIsNotNone(overlay)
            self.assertGreater(
                np.count_nonzero(np.all(overlay == (255, 0, 0), axis=2)),
                0,
            )
            self.assertGreater(
                np.count_nonzero(np.all(overlay == (0, 255, 255), axis=2)),
                0,
            )

    def test_partial_left_result_keeps_left_and_marks_right_unavailable(self):
        result = runtime.build_desktop_result(
            np.zeros((200, 500), dtype=np.uint8),
            "synthetic.bmp",
            {"x": 230, "y": 100},
            bounds(),
            self._partial_stage3_result("right"),
            Path("/tmp/not-written"),
        )

        interactive = result.report["interactive_result"]
        self.assertTrue(interactive["success"])
        self.assertFalse(interactive["bilateral_success"])
        self.assertIsNotNone(interactive["left"])
        self.assertIsNone(interactive["right"])
        self.assertEqual(
            "unavailable",
            interactive["side_status"]["right"]["status"],
        )
        self.assertIsNone(interactive["stripe_spacing_px"])
        overlay = result.debug_images[runtime.FORMAL_OVERLAY_FILENAME]
        self.assertTrue(np.any(np.all(overlay == (255, 0, 0), axis=2)))
        self.assertFalse(np.any(np.all(overlay == (0, 255, 255), axis=2)))

    def test_partial_right_result_keeps_right_and_marks_left_unavailable(self):
        result = runtime.build_desktop_result(
            np.zeros((200, 500), dtype=np.uint8),
            "synthetic.bmp",
            {"x": 250, "y": 100},
            bounds(),
            self._partial_stage3_result("left"),
            Path("/tmp/not-written"),
        )

        interactive = result.report["interactive_result"]
        self.assertTrue(interactive["success"])
        self.assertFalse(interactive["bilateral_success"])
        self.assertIsNone(interactive["left"])
        self.assertIsNotNone(interactive["right"])
        self.assertEqual(
            "unavailable",
            interactive["side_status"]["left"]["status"],
        )
        self.assertIsNone(interactive["stripe_spacing_px"])
        overlay = result.debug_images[runtime.FORMAL_OVERLAY_FILENAME]
        self.assertFalse(np.any(np.all(overlay == (255, 0, 0), axis=2)))
        self.assertTrue(np.any(np.all(overlay == (0, 255, 255), axis=2)))

    def test_left_image_edge_keeps_only_verified_right_side(self):
        result = self._fixed_result(5, 427)
        interactive = result.report["interactive_result"]

        self.assertTrue(interactive["success"])
        self.assertFalse(interactive["bilateral_success"])
        self.assertIsNone(interactive["left"])
        self.assertEqual("B_C01_C02", interactive["right"]["basin_id"])
        self.assertEqual(20.0, interactive["right"]["center_x_global"])
        self.assertEqual(15.0, interactive["right"]["distance_to_click_px"])
        self.assertIsNone(interactive["stripe_spacing_px"])

    def test_right_image_edge_keeps_only_verified_left_side(self):
        result = self._fixed_result(2446, 427)
        interactive = result.report["interactive_result"]

        self.assertTrue(interactive["success"])
        self.assertFalse(interactive["bilateral_success"])
        self.assertEqual("B_C15_C16", interactive["left"]["basin_id"])
        self.assertEqual(2437.0, interactive["left"]["center_x_global"])
        self.assertEqual(9.0, interactive["left"]["distance_to_click_px"])
        self.assertIsNone(interactive["right"])
        self.assertIsNone(interactive["stripe_spacing_px"])

    def test_edge_side_failing_center_darkness_remains_unavailable(self):
        result = self._fixed_result(20, 1500)
        interactive = result.report["interactive_result"]

        self.assertFalse(interactive["success"])
        self.assertIsNone(interactive["left"])
        self.assertIsNone(interactive["right"])
        self.assertEqual(
            "output_basin_center_not_dark",
            interactive["side_status"]["right"]["reason"],
        )
        self.assertIsNone(interactive["stripe_spacing_px"])

    def test_regular_bilateral_result_is_unchanged(self):
        result = self._fixed_result(80, 427)
        interactive = result.report["interactive_result"]

        self.assertTrue(interactive["success"])
        self.assertTrue(interactive["bilateral_success"])
        self.assertEqual("B_C04_C05", interactive["left"]["basin_id"])
        self.assertEqual("B_C06_C07", interactive["right"]["basin_id"])
        self.assertAlmostEqual(
            29.764705882352942,
            interactive["stripe_spacing_px"],
        )

    def test_result_sequence_has_no_stale_side_or_spacing(self):
        stage3_results = [
            available_shadow_result(),
            self._partial_stage3_result("right"),
            self._partial_stage3_result("left"),
            unavailable_shadow_result(),
        ]
        expected = [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, False),
        ]

        for stage3, (has_left, has_right, has_spacing) in zip(
            stage3_results,
            expected,
        ):
            result = runtime.build_desktop_result(
                np.zeros((200, 500), dtype=np.uint8),
                "synthetic.bmp",
                {"x": 270, "y": 100},
                bounds(),
                stage3,
                Path("/tmp/not-written"),
            )
            interactive = result.report["interactive_result"]
            self.assertEqual(has_left, interactive["left"] is not None)
            self.assertEqual(has_right, interactive["right"] is not None)
            self.assertEqual(
                has_spacing,
                interactive["stripe_spacing_px"] is not None,
            )
            overlay = result.debug_images[runtime.FORMAL_OVERLAY_FILENAME]
            self.assertEqual(
                has_left,
                bool(np.any(np.all(overlay == (255, 0, 0), axis=2))),
            )
            self.assertEqual(
                has_right,
                bool(np.any(np.all(overlay == (0, 255, 255), axis=2))),
            )

    def test_unavailable_never_exposes_geometry_or_distance(self):
        image = np.zeros((200, 500), dtype=np.uint8)
        result = runtime.build_desktop_result(
            image,
            "synthetic.bmp",
            {"x": 270, "y": 100},
            bounds(),
            unavailable_shadow_result(),
            Path("/tmp/not-written"),
        )

        interactive = result.report["interactive_result"]
        self.assertFalse(interactive["success"])
        self.assertFalse(interactive["bilateral_success"])
        self.assertIsNone(interactive["left"])
        self.assertIsNone(interactive["right"])
        self.assertEqual(
            "unavailable",
            interactive["side_status"]["left"]["status"],
        )
        self.assertEqual(
            "unavailable",
            interactive["side_status"]["right"]["status"],
        )
        self.assertIsNone(interactive["stripe_spacing_px"])
        self.assertIsNone(interactive["geometry"])
        self.assertIn(runtime.UNAVAILABLE_REASON, interactive["failure_reasons"])
        overlay = result.debug_images[runtime.FORMAL_OVERLAY_FILENAME]
        self.assertFalse(np.any(np.all(overlay == (255, 0, 0), axis=2)))
        self.assertFalse(np.any(np.all(overlay == (0, 255, 255), axis=2)))

    def test_report_and_overlay_can_be_persisted_separately(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            result = runtime.build_desktop_result(
                np.zeros((200, 500), dtype=np.uint8),
                "synthetic.bmp",
                {"x": 270, "y": 100},
                bounds(),
                available_shadow_result(),
                output_dir,
            )

            runtime.persist_formal_report(result)

            report_path = output_dir / runtime.FORMAL_REPORT_FILENAME
            overlay_path = output_dir / runtime.FORMAL_OVERLAY_FILENAME
            self.assertTrue(report_path.is_file())
            self.assertFalse(overlay_path.exists())

            runtime.persist_formal_overlay(
                result,
                temporary_token="test-run",
            )

            self.assertTrue(overlay_path.is_file())
            saved_overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(
                result.debug_images[runtime.FORMAL_OVERLAY_FILENAME],
                saved_overlay,
            )
            self.assertEqual([], list(output_dir.glob("*.tmp.png")))

    def test_non_atomic_success_is_converted_to_safe_unavailable(self):
        image = np.zeros((200, 500), dtype=np.uint8)
        stage3 = available_shadow_result()
        stage3["final_hypothesis"]["atomic"] = False
        result = runtime.build_desktop_result(
            image,
            "synthetic.bmp",
            {"x": 270, "y": 100},
            bounds(),
            stage3,
            Path("/tmp/not-written"),
        )

        interactive = result.report["interactive_result"]
        self.assertFalse(interactive["success"])
        self.assertEqual(
            "stage3_atomic_result_contract_invalid",
            interactive["unavailable_reason"],
        )
        self.assertIsNone(interactive["left"])
        self.assertIsNone(interactive["right"])

    def test_timing_updates_merge_and_summarize_median_and_max(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for run_id, offset in (("a", 0.0), ("b", 10.0)):
                output_dir = root / run_id
                metadata = {
                    "image_name": "synthetic.bmp",
                    "reference_global": {"x": 10, "y": 20},
                    "result_source": runtime.RESULT_SOURCE,
                }
                performance.update_timing(
                    root / "logs",
                    output_dir,
                    run_id,
                    metadata,
                    {
                        "stage3_algorithm_ms": 20.0 + offset,
                        "result_write_ms": 2.0 + offset,
                    },
                )
                summary = performance.update_timing(
                    root / "logs",
                    output_dir,
                    run_id,
                    metadata,
                    {
                        "legacy_algorithm_ms": 40.0 + offset,
                        "ui_draw_ms": 5.0 + offset,
                        "click_to_display_ms": 30.0 + offset,
                    },
                )

            self.assertEqual(2, summary["complete_run_count"])
            self.assertEqual(
                25.0,
                summary["phases"]["stage3_algorithm_ms"]["median_ms"],
            )
            self.assertEqual(
                40.0,
                summary["phases"]["click_to_display_ms"]["max_ms"],
            )
            saved = json.loads(
                (
                    root
                    / "logs"
                    / performance.SUMMARY_FILENAME
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(summary, saved)


if __name__ == "__main__":
    unittest.main()
