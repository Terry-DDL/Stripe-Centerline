"""Small tests for image display coordinate helpers."""

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.interactive_app import (  # noqa: E402
    display_image_and_scale,
    safe_stem,
)


class InteractiveAppHelperTests(unittest.TestCase):
    def test_display_scales_map_back_to_original_dimensions(self):
        image = np.zeros((2056, 2452), dtype=np.uint8)

        display, scale_x, scale_y = display_image_and_scale(image)

        self.assertEqual(display.shape, (922, 1100, 3))
        self.assertAlmostEqual(display.shape[1] * scale_x, 2452)
        self.assertAlmostEqual(display.shape[0] * scale_y, 2056)

    def test_safe_stem_removes_spaces_and_path_punctuation(self):
        self.assertEqual(safe_stem("Sample 1.bmp"), "Sample_1")


if __name__ == "__main__":
    unittest.main()
