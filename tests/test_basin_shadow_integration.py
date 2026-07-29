"""Tests for the desktop Stage 3.1 shadow-only integration."""

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

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


def unavailable_shadow_result() -> dict:
    return {
        "algorithm_revision": "stage3_1",
        "configuration_checksum": "joint-checksum",
        "success": False,
        "status": "unavailable",
        "unavailable_reason": "basin_structure_not_verified",
        "final_hypothesis": None,
        "debug": {
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

    def test_desktop_worker_keeps_formal_result_when_shadow_wrapper_raises(self):
        application = desktop_app.StripeDesktopApp.__new__(
            desktop_app.StripeDesktopApp
        )
        application.pitch_reference_cache = {}
        application.analysis_queue = __import__("queue").Queue()
        image = np.zeros((200, 500), dtype=np.uint8)
        formal = production_result()

        with (
            patch.object(
                desktop_app,
                "get_or_build_pitch_reference",
                return_value=object(),
            ),
            patch.object(
                desktop_app,
                "run_interactive_case",
                return_value=formal,
            ),
            patch.object(
                desktop_app,
                "run_shadow_and_log",
                side_effect=RuntimeError("log unavailable"),
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

        _key, queued_result, error, _pitch_map = (
            application.analysis_queue.get_nowait()
        )
        self.assertIs(formal, queued_result)
        self.assertIsNone(error)

    def test_shadow_source_does_not_read_heldout_or_formal_detectors(self):
        source = Path(shadow.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "heldout.json",
            "interactive_pipeline",
            "interactive_analysis",
            "run_interactive_case",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
