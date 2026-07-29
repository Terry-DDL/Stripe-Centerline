"""Small synthetic tests for the offline column-basin prototype."""

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.prototype_column_basins import (  # noqa: E402
    band_white_counts,
    basin_context,
    dark_basins_from_separators,
    merge_short_separator_gaps,
    separator_columns,
)


class ColumnBasinPrototypeTests(unittest.TestCase):
    def test_separator_requires_three_agreeing_bands(self):
        image = np.zeros((10, 6), dtype=np.uint8)
        for band_index in (0, 1, 3):
            image[band_index * 2, 2] = 235
        for band_index in (0, 1):
            image[band_index * 2, 4] = 235

        counts = band_white_counts(image, 220)
        separators = separator_columns(counts, min_count=1)

        self.assertTrue(separators[2])
        self.assertFalse(separators[4])

    def test_short_gap_merging_controls_dark_basin_split(self):
        separators = np.array(
            [False, False, True, True, False, True, True, False],
            dtype=bool,
        )

        merged = merge_short_separator_gaps(separators, max_gap_px=1)
        basins = dark_basins_from_separators(merged)

        self.assertTrue(merged[4])
        self.assertEqual(
            [(basin.x0, basin.x1) for basin in basins],
            [(0, 1), (7, 7)],
        )

    def test_context_handles_dark_basin_and_white_separator_clicks(self):
        basins = dark_basins_from_separators(
            np.array(
                [False, False, True, False, False, True, False],
                dtype=bool,
            )
        )

        inside = basin_context(basins, 3)
        separator = basin_context(basins, 2)

        self.assertEqual(inside["click_basin_id"], 1)
        self.assertEqual(inside["left_neighbor_basin_id"], 0)
        self.assertEqual(inside["right_neighbor_basin_id"], 2)
        self.assertIsNone(separator["click_basin_id"])
        self.assertEqual(separator["left_neighbor_basin_id"], 0)
        self.assertEqual(separator["right_neighbor_basin_id"], 1)


if __name__ == "__main__":
    unittest.main()
