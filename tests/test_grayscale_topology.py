"""Tests for the shadow-only same-dark-basin evidence."""

from pathlib import Path
import sys
import unittest

import numpy as np


SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import InteractiveConfig, ProcessingConfig  # noqa: E402
from grayscale_topology import evaluate_grayscale_topology  # noqa: E402
from interactive_analysis import select_interactive_tracks  # noqa: E402
from stripe_analysis import analyze_adjacent_stripes  # noqa: E402


IDENTITY = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


class GrayscaleTopologyTests(unittest.TestCase):
    def selection_from_mask(
        self,
        mask,
        click_x: int,
        click_y: int,
        processing_config: ProcessingConfig | None = None,
    ):
        config = processing_config or ProcessingConfig(
            stripe_search_radius_px=100
        )
        analysis = analyze_adjacent_stripes(
            mask,
            click_x,
            click_x,
            0,
            config,
        )
        return select_interactive_tracks(
            mask,
            click_x,
            click_y,
            analysis,
            config,
        )

    def test_split_tracks_inside_one_wide_dark_region_are_contradictory(self):
        image = np.full((90, 240), 220, dtype=np.uint8)
        image[:, 85:165] = 20
        mask = np.zeros_like(image)
        mask[:, 100:111] = 255
        mask[:, 140:151] = 255
        selection = self.selection_from_mask(mask, 125, 45)

        evaluation = evaluate_grayscale_topology(
            image,
            selection,
            IDENTITY,
            InteractiveConfig(),
        )

        self.assertEqual(evaluation.report["status"], "Contradictory")
        self.assertTrue(
            evaluation.report["strong_same_basin_conflict"]
        )
        self.assertEqual(
            evaluation.report["reason"],
            "same_dark_basin_split",
        )

    def test_real_narrow_separator_is_not_a_same_basin_conflict(self):
        image = np.full((90, 240), 220, dtype=np.uint8)
        image[:, 90:112] = 20
        image[:, 116:140] = 20
        mask = np.zeros_like(image)
        mask[:, 90:112] = 255
        mask[:, 116:140] = 255
        selection = self.selection_from_mask(mask, 114, 45)

        evaluation = evaluate_grayscale_topology(
            image,
            selection,
            IDENTITY,
            InteractiveConfig(),
        )

        self.assertEqual(evaluation.report["status"], "Consistent")
        self.assertFalse(
            evaluation.report["strong_same_basin_conflict"]
        )

    def test_weak_separator_becomes_unverifiable_instead_of_contradictory(self):
        image = np.full((90, 240), 220, dtype=np.uint8)
        image[:, 85:165] = 20
        image[:, 123:127] = 40
        mask = np.zeros_like(image)
        mask[:, 100:111] = 255
        mask[:, 140:151] = 255
        selection = self.selection_from_mask(mask, 125, 45)

        evaluation = evaluate_grayscale_topology(
            image,
            selection,
            IDENTITY,
            InteractiveConfig(),
        )

        self.assertEqual(
            evaluation.report["status"],
            "Unable to verify",
        )
        self.assertFalse(
            evaluation.report["strong_same_basin_conflict"]
        )

    def test_rows_concentrated_in_one_vertical_region_are_unverifiable(self):
        image = np.full((90, 240), 220, dtype=np.uint8)
        image[:, 85:165] = 20
        mask = np.zeros_like(image)
        mask[:12, 100:111] = 255
        mask[:12, 140:151] = 255
        processing = ProcessingConfig(
            stripe_search_radius_px=100,
            min_stripe_support_ratio=0.05,
        )
        selection = self.selection_from_mask(mask, 125, 5, processing)

        evaluation = evaluate_grayscale_topology(
            image,
            selection,
            IDENTITY,
            InteractiveConfig(),
        )

        self.assertEqual(
            evaluation.report["status"],
            "Unable to verify",
        )
        self.assertFalse(
            evaluation.report["strong_same_basin_conflict"]
        )

    def test_rotated_candidate_samples_inverse_mapped_original_pixels(self):
        image = np.full((90, 260), 220, dtype=np.uint8)
        image[:, 100:170] = 20
        mask = np.zeros_like(image)
        mask[:, 70:81] = 255
        mask[:, 115:126] = 255
        selection = self.selection_from_mask(mask, 95, 45)
        detection_to_original = np.array(
            [[1.0, 0.0, 40.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )

        evaluation = evaluate_grayscale_topology(
            image,
            selection,
            detection_to_original,
            InteractiveConfig(),
        )

        self.assertEqual(evaluation.report["status"], "Contradictory")
        self.assertTrue(
            evaluation.report["strong_same_basin_conflict"]
        )

    def test_two_dark_basins_merged_into_one_track_are_contradictory(self):
        image = np.full((90, 240), 220, dtype=np.uint8)
        image[:, 90:101] = 20
        image[:, 120:131] = 20
        image[:, 145:156] = 20
        image[:, 164:175] = 20
        mask = np.zeros_like(image)
        mask[:, 90:101] = 255
        mask[:, 120:131] = 255
        mask[:, 145:175] = 255
        selection = self.selection_from_mask(mask, 125, 45)

        evaluation = evaluate_grayscale_topology(
            image,
            selection,
            IDENTITY,
            InteractiveConfig(),
        )

        self.assertEqual(evaluation.report["status"], "Contradictory")
        self.assertFalse(
            evaluation.report["strong_same_basin_conflict"]
        )
        self.assertTrue(
            evaluation.report["strong_merged_basin_conflict"]
        )
        self.assertEqual(
            evaluation.report["reason"],
            "merged_dark_basins",
        )
        right = next(
            track
            for track in evaluation.report["evaluated_tracks"]
            if track["label"] == "right"
        )
        self.assertEqual(right["merged_basin_support_ratio"], 1.0)

    def test_one_legitimate_wide_dark_basin_is_not_a_merge_conflict(self):
        image = np.full((90, 240), 220, dtype=np.uint8)
        image[:, 90:101] = 20
        image[:, 120:131] = 20
        image[:, 145:175] = 20
        mask = np.zeros_like(image)
        mask[:, 90:101] = 255
        mask[:, 120:131] = 255
        mask[:, 145:175] = 255
        selection = self.selection_from_mask(mask, 125, 45)

        evaluation = evaluate_grayscale_topology(
            image,
            selection,
            IDENTITY,
            InteractiveConfig(),
        )

        self.assertEqual(evaluation.report["status"], "Consistent")
        self.assertFalse(
            evaluation.report["strong_merged_basin_conflict"]
        )

    def test_local_bright_defect_does_not_create_a_merge_conflict(self):
        image = np.full((90, 240), 220, dtype=np.uint8)
        image[:, 90:101] = 20
        image[:, 120:131] = 20
        image[:, 145:175] = 20
        image[:15, 157:163] = 180
        mask = np.zeros_like(image)
        mask[:, 90:101] = 255
        mask[:, 120:131] = 255
        mask[:, 145:175] = 255
        selection = self.selection_from_mask(mask, 125, 45)

        evaluation = evaluate_grayscale_topology(
            image,
            selection,
            IDENTITY,
            InteractiveConfig(),
        )

        self.assertFalse(
            evaluation.report["strong_merged_basin_conflict"]
        )

    def test_merged_track_uses_inverse_mapped_original_grayscale(self):
        image = np.full((90, 260), 220, dtype=np.uint8)
        image[:, 90:101] = 20
        image[:, 120:131] = 20
        image[:, 145:156] = 20
        image[:, 164:175] = 20
        mask = np.zeros_like(image)
        mask[:, 50:61] = 255
        mask[:, 80:91] = 255
        mask[:, 105:135] = 255
        selection = self.selection_from_mask(mask, 85, 45)
        detection_to_original = np.array(
            [[1.0, 0.0, 40.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )

        evaluation = evaluate_grayscale_topology(
            image,
            selection,
            detection_to_original,
            InteractiveConfig(),
        )

        self.assertTrue(
            evaluation.report["strong_merged_basin_conflict"]
        )

    def test_out_of_bounds_mapped_rows_do_not_repeat_border_pixels(self):
        image = np.full((90, 240), 220, dtype=np.uint8)
        image[:, 90:101] = 20
        image[:, 120:131] = 20
        image[:, 145:156] = 20
        image[:, 164:175] = 20
        mask = np.zeros_like(image)
        mask[:, 90:101] = 255
        mask[:, 120:131] = 255
        mask[:, 145:175] = 255
        selection = self.selection_from_mask(mask, 125, 45)
        detection_to_original = np.array(
            [[1.0, 0.0, 100.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )

        evaluation = evaluate_grayscale_topology(
            image,
            selection,
            detection_to_original,
            InteractiveConfig(),
        )

        self.assertFalse(
            evaluation.report["strong_merged_basin_conflict"]
        )


if __name__ == "__main__":
    unittest.main()
