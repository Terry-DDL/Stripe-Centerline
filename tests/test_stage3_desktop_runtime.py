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
    unavailable_shadow_result,
)
from tools import desktop_performance as performance
from tools import stage3_desktop_runtime as runtime


def bounds():
    return SimpleNamespace(
        x0_global=20,
        y0_global=10,
        x1_global=480,
        y1_global=190,
    )


class Stage3DesktopRuntimeTests(unittest.TestCase):
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
        self.assertIsNone(interactive["left"])
        self.assertIsNone(interactive["right"])
        self.assertIsNone(interactive["stripe_spacing_px"])
        self.assertIsNone(interactive["geometry"])
        self.assertIn(runtime.UNAVAILABLE_REASON, interactive["failure_reasons"])
        overlay = result.debug_images[runtime.FORMAL_OVERLAY_FILENAME]
        self.assertFalse(np.any(np.all(overlay == (255, 0, 0), axis=2)))
        self.assertFalse(np.any(np.all(overlay == (0, 255, 255), axis=2)))

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
