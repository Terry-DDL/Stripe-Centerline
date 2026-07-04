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
    calculate_interactive_roi_bounds,
    estimate_rotation_shadow,
    run_interactive_case,
)


class InteractivePipelineTests(unittest.TestCase):
    def test_roi_is_fixed_size_away_from_edges(self):
        bounds = calculate_interactive_roi_bounds(
            (2000, 2000),
            1000,
            900,
            InteractiveConfig(),
        )
        self.assertEqual(bounds.width_roi, 500)
        self.assertEqual(bounds.height_roi, 500)
        self.assertEqual((bounds.x0_global, bounds.y0_global), (750, 650))

    def test_roi_clips_without_moving_the_click(self):
        bounds = calculate_interactive_roi_bounds(
            (2000, 2000),
            40,
            30,
            InteractiveConfig(),
        )
        self.assertEqual((bounds.x0_global, bounds.y0_global), (0, 0))
        self.assertEqual((bounds.x1_global, bounds.y1_global), (290, 280))

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

    def test_pipeline_saves_debug_and_keeps_rotation_shadow_only(self):
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
            self.assertEqual(len(result.debug_images), 13)


if __name__ == "__main__":
    unittest.main()
