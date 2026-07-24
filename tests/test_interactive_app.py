"""Small tests for image display coordinate helpers."""

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.interactive_app import (  # noqa: E402
    ACTIVE_IMAGE_STATE_KEY,
    CLICK_POINT_STATE_KEY,
    FULL_IMAGE_REVISION_KEY,
    RESULT_STATE_KEY,
    ZOOM_CENTER_STATE_KEY,
    ZOOM_IMAGE_REVISION_KEY,
    create_zoom_display,
    display_image_and_scale,
    map_display_point_to_source,
    next_widget_revision,
    prepare_image_state,
    safe_stem,
    zoom_bounds,
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

    def test_new_image_clears_old_interaction_and_revises_widgets(self):
        state = {
            ACTIVE_IMAGE_STATE_KEY: "old.bmp:1234",
            ZOOM_CENTER_STATE_KEY: (100, 200),
            CLICK_POINT_STATE_KEY: (101, 201),
            RESULT_STATE_KEY: object(),
            FULL_IMAGE_REVISION_KEY: 3,
            ZOOM_IMAGE_REVISION_KEY: 7,
        }

        changed = prepare_image_state(state, "new.bmp:5678")

        self.assertTrue(changed)
        self.assertEqual(state[ACTIVE_IMAGE_STATE_KEY], "new.bmp:5678")
        self.assertNotIn(ZOOM_CENTER_STATE_KEY, state)
        self.assertNotIn(CLICK_POINT_STATE_KEY, state)
        self.assertNotIn(RESULT_STATE_KEY, state)
        self.assertEqual(state[FULL_IMAGE_REVISION_KEY], 4)
        self.assertEqual(state[ZOOM_IMAGE_REVISION_KEY], 8)

    def test_same_image_keeps_current_selection(self):
        state = {
            ACTIVE_IMAGE_STATE_KEY: "same.bmp:1234",
            CLICK_POINT_STATE_KEY: (101, 201),
        }

        changed = prepare_image_state(state, "same.bmp:1234")

        self.assertFalse(changed)
        self.assertEqual(state[CLICK_POINT_STATE_KEY], (101, 201))
        self.assertEqual(state[FULL_IMAGE_REVISION_KEY], 0)
        self.assertEqual(state[ZOOM_IMAGE_REVISION_KEY], 0)

    def test_widget_revision_always_advances(self):
        state = {}

        self.assertEqual(next_widget_revision(state, "revision"), 1)
        self.assertEqual(next_widget_revision(state, "revision"), 2)

    def test_zoom_bounds_and_click_map_to_source_coordinates(self):
        image = np.zeros((2056, 2452), dtype=np.uint8)

        self.assertEqual(
            zoom_bounds(image.shape, (1000, 900)),
            (880, 800, 1120, 1000),
        )
        zoom, x0, y0, scale_x, scale_y = create_zoom_display(
            image,
            (1000, 900),
            None,
        )
        mapped = map_display_point_to_source(
            375,
            330,
            scale_x,
            scale_y,
            image.shape,
            x0,
            y0,
        )

        self.assertEqual(zoom.shape, (600, 720, 3))
        self.assertEqual(mapped, (1005, 910))


if __name__ == "__main__":
    unittest.main()
