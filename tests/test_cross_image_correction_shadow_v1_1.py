"""Directed tests for the default-off cross-image v1.1 shadow bridge."""

import copy
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
from tools import cross_image_correction_shadow_v1_1 as shadow
from tools import desktop_app


def bounds():
    return SimpleNamespace(
        x0_global=20,
        y0_global=10,
        x1_global=480,
        y1_global=190,
    )


class CrossImageCorrectionShadowTests(unittest.TestCase):
    def test_shadow_persists_full_result_refinement_and_overlay(self):
        image = np.zeros((200, 500), dtype=np.uint8)
        experimental = available_shadow_result()
        formal = unavailable_shadow_result()
        formal_before = copy.deepcopy(formal)
        refinement = {
            "attempted": True,
            "identity_before": {"success": True},
            "identity_after": {"success": True},
            "basins": {"left": {}, "right": {}},
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            record = shadow.run_cross_image_experiment_shadow(
                image,
                "synthetic.bmp",
                {"x": 270, "y": 100},
                bounds(),
                output_dir,
                "run-1",
                formal,
                runner=lambda *_args: experimental,
                refiner=lambda *_args: refinement,
                overlay_builder=lambda *_args: np.full(
                    (40, 80, 3),
                    127,
                    dtype=np.uint8,
                ),
            )
            saved = json.loads(
                (output_dir / shadow.RESULT_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            overlay = cv2.imread(
                str(output_dir / shadow.OVERLAY_FILENAME)
            )

        self.assertTrue(record["shadow"]["success"])
        self.assertEqual(experimental, record["shadow"]["result"])
        self.assertEqual(refinement, record["shadow"]["center_refinement"])
        self.assertEqual("run-1", saved["run_id"])
        self.assertEqual(
            experimental["final_hypothesis"]["basin_ids"],
            saved["shadow"]["result"]["final_hypothesis"]["basin_ids"],
        )
        self.assertFalse(
            saved["contract"]["formal_result_used_as_inference_input"]
        )
        self.assertFalse(saved["contract"]["legacy_fallback_allowed"])
        self.assertEqual((40, 80, 3), overlay.shape)
        self.assertEqual(formal_before, formal)

    def test_formal_success_cannot_restore_unavailable_experiment(self):
        image = np.zeros((200, 500), dtype=np.uint8)
        formal = available_shadow_result()
        experimental = unavailable_shadow_result()

        with tempfile.TemporaryDirectory() as temporary_directory:
            record = shadow.run_cross_image_experiment_shadow(
                image,
                "synthetic.bmp",
                {"x": 270, "y": 100},
                bounds(),
                Path(temporary_directory),
                "run-2",
                formal,
                runner=lambda *_args: experimental,
                refiner=lambda *_args: {"attempted": False},
                overlay_builder=lambda *_args: np.zeros(
                    (20, 20, 3),
                    dtype=np.uint8,
                ),
            )

        self.assertFalse(record["shadow"]["success"])
        self.assertTrue(record["formal_stage3_comparison"]["success"])
        self.assertIsNone(
            record["shadow"]["result"]["final_hypothesis"]
        )

    def test_desktop_schedules_shadow_after_formal_completion(self):
        application = desktop_app.StripeDesktopApp.__new__(
            desktop_app.StripeDesktopApp
        )
        application.analysis_queue = __import__("queue").Queue()
        application.debug_enabled = False
        image = np.zeros((200, 500), dtype=np.uint8)
        scheduled = {}

        class FakeTimer:
            def __init__(self, interval, function, args):
                scheduled["interval"] = interval
                scheduled["function"] = function
                scheduled["args"] = args
                scheduled["queue_size_at_schedule"] = (
                    application.analysis_queue.qsize()
                )
                self.daemon = False

            def start(self):
                scheduled["started"] = True
                scheduled["daemon"] = self.daemon

        with (
            patch.object(
                desktop_app,
                "cross_image_experiment_enabled",
                return_value=True,
            ),
            patch.object(desktop_app.threading, "Timer", FakeTimer),
            patch.object(
                desktop_app,
                "run_frozen_stage3",
                return_value=available_shadow_result(),
            ),
            patch.object(desktop_app, "persist_formal_result"),
        ):
            application._run_analysis_worker(
                ("image-id", (270, 100)),
                image,
                "synthetic.bmp",
                270,
                100,
                Path("/tmp/cross-image-shadow-schedule-test"),
                "run-3",
                1,
            )

        completion = application.analysis_queue.get_nowait()
        self.assertIsNone(completion.error)
        self.assertEqual(1, scheduled["queue_size_at_schedule"])
        self.assertEqual(
            desktop_app.CROSS_IMAGE_EXPERIMENT_START_DELAY_SECONDS,
            scheduled["interval"],
        )
        self.assertTrue(scheduled["started"])
        self.assertTrue(scheduled["daemon"])
        self.assertIs(
            application._run_cross_image_experiment_worker.__func__,
            scheduled["function"].__func__,
        )

    def test_desktop_default_does_not_schedule_or_import_shadow(self):
        application = desktop_app.StripeDesktopApp.__new__(
            desktop_app.StripeDesktopApp
        )
        application.analysis_queue = __import__("queue").Queue()
        application.debug_enabled = False
        image = np.zeros((200, 500), dtype=np.uint8)

        with (
            patch.object(
                desktop_app,
                "cross_image_experiment_enabled",
                return_value=False,
            ),
            patch.object(desktop_app.threading, "Timer") as timer,
            patch.object(
                desktop_app,
                "run_frozen_stage3",
                return_value=available_shadow_result(),
            ),
            patch.object(desktop_app, "persist_formal_result"),
        ):
            application._run_analysis_worker(
                ("image-id", (270, 100)),
                image,
                "synthetic.bmp",
                270,
                100,
                Path("/tmp/cross-image-shadow-default-off-test"),
                "run-default-off",
                1,
            )

        completion = application.analysis_queue.get_nowait()
        self.assertIsNone(completion.error)
        timer.assert_not_called()

    def test_shadow_exception_never_escapes_into_desktop_worker(self):
        application = desktop_app.StripeDesktopApp.__new__(
            desktop_app.StripeDesktopApp
        )
        image = np.zeros((20, 20), dtype=np.uint8)

        with patch.object(
            shadow,
            "run_cross_image_experiment_shadow",
            side_effect=RuntimeError("isolated failure"),
        ):
            application._run_cross_image_experiment_worker(
                image,
                "synthetic.bmp",
                10,
                10,
                bounds(),
                Path("/tmp/cross-image-shadow-error-test"),
                available_shadow_result(),
                "run-4",
            )

    def test_bridge_source_has_no_heldout_or_legacy_detector_access(self):
        source = Path(shadow.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "heldout",
            "run_interactive_case",
            "interactive_analysis",
            "legacy_output",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
