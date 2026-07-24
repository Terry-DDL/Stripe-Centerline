"""Unit tests for click-aware stripe selection."""

from pathlib import Path
import sys
import unittest

import numpy as np


SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import ProcessingConfig  # noqa: E402
from interactive_analysis import select_interactive_tracks  # noqa: E402
from stripe_analysis import analyze_adjacent_stripes  # noqa: E402


class InteractiveSelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = ProcessingConfig()

    def make_mask(self, stripe_centers=(70, 150, 220), height=100, width=300):
        mask = np.zeros((height, width), dtype=np.uint8)
        for center_x in stripe_centers:
            mask[:, center_x - 5 : center_x + 6] = 255
        return mask

    def analyze(self, mask, click_x):
        return analyze_adjacent_stripes(
            mask,
            click_x,
            click_x,
            0,
            self.config,
        )

    def test_white_click_uses_legacy_left_and_right(self):
        mask = self.make_mask()
        analysis = self.analyze(mask, 185)

        selection = select_interactive_tracks(
            mask, 185, 50, analysis, self.config
        )

        self.assertEqual(selection.click_classification, "white_region")
        self.assertEqual(selection.left_track.center_x_roi, 150.0)
        self.assertEqual(selection.right_track.center_x_roi, 220.0)
        self.assertTrue(selection.success)

    def test_black_click_uses_neighbors_when_click_is_offset(self):
        mask = self.make_mask()
        analysis = self.analyze(mask, 153)

        selection = select_interactive_tracks(
            mask, 153, 50, analysis, self.config
        )

        self.assertEqual(selection.click_classification, "black_stripe")
        self.assertEqual(selection.clicked_track.center_x_roi, 150.0)
        self.assertEqual(selection.left_track.center_x_roi, 70.0)
        self.assertEqual(selection.right_track.center_x_roi, 220.0)
        self.assertTrue(selection.success)

    def test_black_click_at_track_center_accepts_on_reference_track(self):
        mask = self.make_mask()
        analysis = self.analyze(mask, 150)

        selection = select_interactive_tracks(
            mask, 150, 50, analysis, self.config
        )

        self.assertEqual(selection.clicked_track.center_x_roi, 150.0)
        self.assertEqual(
            selection.clicked_track.rejection_reasons,
            ["center_on_reference"],
        )
        self.assertEqual(selection.left_track.center_x_roi, 70.0)
        self.assertEqual(selection.right_track.center_x_roi, 220.0)

    def test_unstable_black_region_is_reported_as_ambiguous(self):
        mask = self.make_mask(stripe_centers=(70, 220))
        mask[50, 148:154] = 255
        analysis = self.analyze(mask, 150)

        selection = select_interactive_tracks(
            mask, 150, 50, analysis, self.config
        )

        self.assertFalse(selection.success)
        self.assertEqual(
            selection.click_classification,
            "ambiguous_black_region",
        )
        self.assertEqual(
            selection.failure_reasons,
            ("clicked_black_region_not_stable_track",),
        )

    def test_missing_outer_neighbor_is_an_explicit_failure(self):
        mask = self.make_mask(stripe_centers=(150, 220))
        analysis = self.analyze(mask, 147)

        selection = select_interactive_tracks(
            mask, 147, 50, analysis, self.config
        )

        self.assertFalse(selection.success)
        self.assertIn("no_valid_left_stripe", selection.failure_reasons)


if __name__ == "__main__":
    unittest.main()
