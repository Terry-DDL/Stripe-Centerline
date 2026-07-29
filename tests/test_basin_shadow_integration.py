"""Tests for the desktop Stage 3.1 shadow-only integration."""

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools import basin_shadow_integration as shadow
from tools import desktop_app


def production_result(
    click: tuple[int, int] = (270, 100),
) -> SimpleNamespace:
    bounds = SimpleNamespace(
        x0_global=20,
        y0_global=10,
        x1_global=480,
        y1_global=190,
    )
    return SimpleNamespace(
        click_x_global=click[0],
        click_y_global=click[1],
        bounds_global=bounds,
        report={
            "algorithm_revision": "formal_v1",
            "image": {"shape": [200, 500]},
            "interactive_result": {
                "success": False,
                "failure_reasons": ["formal_safe_failure"],
                "left": None,
                "right": None,
                "stripe_spacing_px": None,
            },
        },
    )


def separator_candidate(
    candidate_id: str,
    x_by_band_roi: list[float],
    accepted: bool = True,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "accepted": accepted,
        "rejection_reason": None if accepted else "weak_path",
        "band_centers_y_roi": [20.0, 90.0, 160.0],
        "x_by_band_roi": x_by_band_roi,
        "x_at_reference_roi": x_by_band_roi[1],
    }


def basin(
    basin_id: str,
    left_separator_id: str,
    right_separator_id: str,
    left_x: float,
    right_x: float,
) -> dict:
    return {
        "basin_id": basin_id,
        "left_separator_id": left_separator_id,
        "right_separator_id": right_separator_id,
        "band_centers_y_roi": [20.0, 90.0, 160.0],
        "left_x_by_band_roi": [left_x] * 3,
        "right_x_by_band_roi": [right_x] * 3,
        "center_x_by_band_roi": [(left_x + right_x) / 2.0] * 3,
        "center_x_at_reference_roi": (left_x + right_x) / 2.0,
        "width_at_reference_px": right_x - left_x,
    }


def unavailable_shadow_result() -> dict:
    return {
        "algorithm_revision": "stage3_1",
        "configuration_checksum": "joint-checksum",
        "success": False,
        "status": "unavailable",
        "unavailable_reason": "basin_structure_not_verified",
        "final_hypothesis": None,
        "debug": {
            "separator_result": {
                "reference_y_roi": 90.0,
                "candidates": [
                    separator_candidate("C01", [200.0] * 3),
                    separator_candidate("C02", [240.0] * 3),
                    separator_candidate(
                        "C03",
                        [320.0] * 3,
                        accepted=False,
                    ),
                ],
            },
            "reference_relation": {
                "status": "unavailable",
                "relation": None,
                "unavailable_reason": "basin_structure_not_verified",
            },
            "basin_graph": {"path_order_conflicts": []},
            "raw_pitch_result": {
                "algorithm_revision": "raw_pitch_v3",
                "configuration_checksum": "pitch-checksum",
                "confidence": "unavailable",
                "success_eligible": False,
                "diagnostic_pitch_px": None,
                "usable_pitch_px": None,
                "harmonic_ambiguity": {
                    "detected": False,
                    "reason": None,
                },
            },
        },
    }


def available_shadow_result() -> dict:
    result = unavailable_shadow_result()
    result.update(
        {
            "success": True,
            "status": "available",
            "unavailable_reason": None,
            "final_hypothesis": {
                "reference_relation": "on_separator",
                "clicked_separator_id": "C02",
                "separator_sequence": ["C01", "C02", "C03"],
                "basin_ids": {
                    "left": "B_C01_C02",
                    "right": "B_C02_C03",
                },
                "basin_sequence": ["B_C01_C02", "B_C02_C03"],
                "separator_ids": None,
                "shared_boundary": {
                    "separator_id": "C02",
                    "left_basin_id": "B_C01_C02",
                    "right_basin_id": "B_C02_C03",
                },
                "geometry": {
                    "reference_relation": "on_separator",
                    "basins": {
                        "left": {
                            "center_x_at_reference_roi": 220.0,
                            "center_x_at_reference_global": 240.0,
                        },
                        "right": {
                            "center_x_at_reference_roi": 280.0,
                            "center_x_at_reference_global": 300.0,
                        },
                    },
                },
                "pitch_evidence": {
                    "algorithm_revision": "raw_pitch_v3",
                    "configuration_checksum": "pitch-checksum",
                    "confidence": "high",
                    "success_eligible": True,
                    "diagnostic_pitch_px": 60.0,
                    "usable_pitch_px": 60.0,
                    "harmonic_ambiguity": {
                        "detected": False,
                        "reason": None,
                    },
                    "joint_decision": "high_pitch_supports_geometry",
                    "geometry_spacing_ratios": [1.0],
                },
                "atomic": True,
            },
        }
    )
    result["debug"]["reference_relation"] = {
        "status": "unique",
        "relation": "on_separator",
        "unavailable_reason": None,
    }
    result["debug"]["separator_result"]["candidates"][-1] = (
        separator_candidate("C03", [320.0] * 3)
    )
    result["debug"]["basin_graph"]["verified_basins"] = [
        basin("B_C01_C02", "C01", "C02", 200.0, 240.0),
        basin("B_C02_C03", "C02", "C03", 240.0, 320.0),
    ]
    return result


class BasinShadowIntegrationTests(unittest.TestCase):
    def test_same_image_reference_and_formal_roi_are_passed_to_shadow(self):
        image = np.arange(200 * 500, dtype=np.uint8).reshape(200, 500)
        formal = production_result()
        formal_before = copy.deepcopy(formal.report)
        captured = {}

        def runner(
            received_image,
            reference_global,
            roi_bounds_global,
            direction,
            config,
        ):
            captured.update(
                {
                    "same_object": received_image is image,
                    "reference": reference_global,
                    "bounds": roi_bounds_global,
                    "direction": direction,
                    "config": config,
                }
            )
            return unavailable_shadow_result()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            record = shadow.run_shadow_and_log(
                image,
                "Sample 2.bmp",
                formal,
                root / "point",
                shadow.ShadowLogConfig(log_root=root / "logs"),
                shadow_runner=runner,
            )

            self.assertTrue(captured["same_object"])
            self.assertEqual({"x": 270, "y": 100}, captured["reference"])
            self.assertEqual(
                {"x0": 20, "y0": 10, "x1": 480, "y1": 190},
                captured["bounds"],
            )
            self.assertEqual("vertical", captured["direction"])
            self.assertIs(captured["config"], shadow.joint.DEFAULT_CONFIG)
            self.assertEqual("completed", record["integration_status"])
            self.assertTrue(record["formal_report_unchanged"])
            self.assertTrue(record["image_unchanged"])
            self.assertEqual(formal_before, formal.report)
            self.assertTrue(
                (root / "point" / shadow.PER_CLICK_FILENAME).exists()
            )
            self.assertTrue(
                (root / "logs" / shadow.JSONL_FILENAME).exists()
            )
            self.assertTrue(
                (root / "logs" / shadow.CSV_FILENAME).exists()
            )
            acceptance = record["shadow"]["acceptance_geometry"]
            self.assertEqual(
                "unavailable_candidates",
                acceptance["mode"],
            )
            self.assertEqual(
                ["C01", "C02", "C03"],
                [
                    path["separator_id"]
                    for path in acceptance["separator_paths"]
                ],
            )
            self.assertIsNone(acceptance["final_geometry"])
            self.assertEqual(
                "basin_structure_not_verified",
                acceptance["rejection_reason"],
            )
            self.assertEqual(
                "written",
                record["shadow_overlay"]["status"],
            )
            self.assertTrue(
                (root / "point" / shadow.SHADOW_OVERLAY_FILENAME).exists()
            )

    def test_available_shadow_log_contains_atomic_geometry_and_pitch(self):
        image = np.zeros((200, 500), dtype=np.uint8)
        formal = production_result()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            record = shadow.run_shadow_and_log(
                image,
                "synthetic.bmp",
                formal,
                root / "point",
                shadow.ShadowLogConfig(log_root=root / "logs"),
                shadow_runner=lambda *args: available_shadow_result(),
            )
            geometry = record["shadow"]["geometry"]
            self.assertTrue(record["shadow"]["success"])
            self.assertEqual(240.0, geometry["left_basin_center_x_global"])
            self.assertEqual(300.0, geometry["right_basin_center_x_global"])
            self.assertEqual(30.0, geometry["left_distance_to_reference_px"])
            self.assertEqual(30.0, geometry["right_distance_to_reference_px"])
            self.assertEqual(
                ["C01", "C02", "C03"],
                geometry["separator_sequence"],
            )
            self.assertTrue(geometry["atomic"])
            self.assertEqual(
                "pitch-checksum",
                record["shadow"]["pitch_provenance"][
                    "configuration_checksum"
                ],
            )
            acceptance = record["shadow"]["acceptance_geometry"]
            self.assertEqual("final_hypothesis", acceptance["mode"])
            self.assertEqual(
                ["C01", "C02", "C03"],
                [
                    path["separator_id"]
                    for path in acceptance["separator_paths"]
                ],
            )
            self.assertEqual(
                [20.0, 90.0, 160.0],
                acceptance["separator_paths"][0][
                    "band_centers_y_roi"
                ],
            )
            self.assertEqual(
                ["B_C01_C02", "B_C02_C03"],
                [item["basin_id"] for item in acceptance["basins"]],
            )
            self.assertEqual(
                [200.0, 200.0, 200.0],
                acceptance["basins"][0][
                    "left_boundary_x_by_band_roi"
                ],
            )
            self.assertEqual(
                240.0,
                acceptance["final_geometry"]["left_center_x_global"],
            )
            self.assertEqual(
                30.0,
                acceptance["final_geometry"]["left_distance_px"],
            )
            overlay_path = (
                root / "point" / shadow.SHADOW_OVERLAY_FILENAME
            )
            overlay = cv2.imread(str(overlay_path))
            self.assertIsNotNone(overlay)
            self.assertEqual((432, 920, 3), overlay.shape)
            self.assertGreater(
                np.count_nonzero(
                    (overlay[:, :, 2] > 220)
                    & (overlay[:, :, 1] < 80)
                    & (overlay[:, :, 0] < 80)
                ),
                0,
            )
            saved = json.loads(
                (
                    root / "point" / shadow.PER_CLICK_FILENAME
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                acceptance,
                saved["shadow"]["acceptance_geometry"],
            )
            self.assertEqual(
                "written",
                saved["shadow_overlay"]["status"],
            )

            with (root / "logs" / shadow.CSV_FILENAME).open(
                encoding="utf-8",
                newline="",
            ) as input_file:
                rows = list(csv.DictReader(input_file))
            self.assertEqual(1, len(rows))
            self.assertEqual("240.0", rows[0]["left_center_x_global"])
            self.assertEqual("completed", rows[0]["integration_status"])

    def test_shadow_error_is_logged_without_changing_formal_result(self):
        image = np.zeros((200, 500), dtype=np.uint8)
        formal = production_result()
        formal_before = copy.deepcopy(formal.report)

        def failing_runner(*args):
            raise RuntimeError("shadow failed")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            record = shadow.run_shadow_and_log(
                image,
                "synthetic.bmp",
                formal,
                root / "point",
                shadow.ShadowLogConfig(log_root=root / "logs"),
                shadow_runner=failing_runner,
            )
            saved = json.loads(
                (
                    root / "point" / shadow.PER_CLICK_FILENAME
                ).read_text(encoding="utf-8")
            )
        self.assertEqual("shadow_error", record["integration_status"])
        self.assertEqual("RuntimeError", saved["integration_error"]["type"])
        self.assertEqual(formal_before, formal.report)

    def test_overlay_error_is_logged_without_changing_shadow_or_formal_result(
        self,
    ):
        image = np.zeros((200, 500), dtype=np.uint8)
        formal = production_result()
        formal_before = copy.deepcopy(formal.report)
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch.object(
                shadow,
                "_atomic_write_png",
                side_effect=OSError("overlay unavailable"),
            ),
        ):
            root = Path(temporary_directory)
            record = shadow.run_shadow_and_log(
                image,
                "synthetic.bmp",
                formal,
                root / "point",
                shadow.ShadowLogConfig(log_root=root / "logs"),
                shadow_runner=lambda *args: available_shadow_result(),
            )
            saved = json.loads(
                (
                    root / "point" / shadow.PER_CLICK_FILENAME
                ).read_text(encoding="utf-8")
            )

        self.assertEqual("completed", record["integration_status"])
        self.assertTrue(record["shadow"]["success"])
        self.assertEqual(
            "artifact_error",
            record["shadow_overlay"]["status"],
        )
        self.assertEqual(
            "OSError",
            saved["shadow_overlay"]["error"]["type"],
        )
        self.assertEqual(formal_before, formal.report)

    def test_desktop_worker_never_falls_back_when_legacy_logging_raises(self):
        application = desktop_app.StripeDesktopApp.__new__(
            desktop_app.StripeDesktopApp
        )
        application.pitch_reference_cache = {}
        application.pitch_reference_lock = __import__("threading").Lock()
        application.analysis_queue = __import__("queue").Queue()
        image = np.zeros((200, 500), dtype=np.uint8)
        legacy = production_result()
        logged = __import__("threading").Event()

        def failing_log(*args, **kwargs):
            logged.set()
            raise RuntimeError("log unavailable")

        with (
            patch.object(
                desktop_app,
                "get_or_build_pitch_reference",
                return_value=object(),
            ),
            patch.object(
                desktop_app,
                "run_interactive_case",
                return_value=legacy,
            ),
            patch.object(
                desktop_app,
                "run_frozen_stage3",
                return_value=unavailable_shadow_result(),
            ),
            patch.object(
                desktop_app,
                "persist_formal_result",
            ),
            patch.object(
                desktop_app,
                "run_shadow_and_log",
                side_effect=failing_log,
            ),
            patch.object(
                desktop_app,
                "update_timing",
            ),
        ):
            application._run_analysis_worker(
                ("image-id", (270, 100)),
                image,
                "synthetic.bmp",
                270,
                100,
                Path("/tmp/shadow-integration-test"),
            )
            self.assertTrue(logged.wait(timeout=2.0))

        completion = application.analysis_queue.get_nowait()
        self.assertIsNone(completion.error)
        self.assertIsNot(legacy, completion.result)
        self.assertEqual(
            desktop_app.STAGE3_RESULT_SOURCE,
            completion.result.report["result_source"],
        )
        self.assertFalse(
            completion.result.report["interactive_result"]["success"]
        )

    def test_legacy_detector_failure_cannot_replace_stage3_result(self):
        application = desktop_app.StripeDesktopApp.__new__(
            desktop_app.StripeDesktopApp
        )
        application.pitch_reference_cache = {}
        application.pitch_reference_lock = __import__("threading").Lock()
        application.analysis_queue = __import__("queue").Queue()
        image = np.zeros((200, 500), dtype=np.uint8)
        legacy_attempted = __import__("threading").Event()

        def failing_legacy(*args, **kwargs):
            legacy_attempted.set()
            raise RuntimeError("legacy unavailable")

        with (
            patch.object(
                desktop_app,
                "get_or_build_pitch_reference",
                return_value=object(),
            ),
            patch.object(
                desktop_app,
                "run_interactive_case",
                side_effect=failing_legacy,
            ),
            patch.object(
                desktop_app,
                "run_frozen_stage3",
                return_value=available_shadow_result(),
            ),
            patch.object(desktop_app, "persist_formal_result"),
            patch.object(desktop_app, "update_timing"),
        ):
            application._run_analysis_worker(
                ("image-id", (270, 100)),
                image,
                "synthetic.bmp",
                270,
                100,
                Path("/tmp/stage3-no-fallback-test"),
            )
            self.assertTrue(legacy_attempted.wait(timeout=2.0))

        completion = application.analysis_queue.get_nowait()
        self.assertIsNone(completion.error)
        self.assertTrue(
            completion.result.report["interactive_result"]["success"]
        )
        self.assertEqual(
            desktop_app.STAGE3_RESULT_SOURCE,
            completion.result.report["result_source"],
        )

    def test_shadow_source_does_not_read_heldout_or_formal_detectors(self):
        source = "\n".join(
            (
                Path(shadow.__file__).read_text(encoding="utf-8"),
                Path(shadow.artifacts.__file__).read_text(
                    encoding="utf-8"
                ),
            )
        )
        for forbidden in (
            "heldout.json",
            "interactive_pipeline",
            "interactive_analysis",
            "run_interactive_case",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
