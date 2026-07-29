import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools import separator_path_prototype as prototype


def synthetic_broken_separators(
    separator_x: tuple[int, ...] = (145, 205, 295, 355),
    height: int = 240,
    width: int = 500,
    half_width: int = 2,
) -> np.ndarray:
    image = np.full((height, width), 24, dtype=np.float64)
    rng = np.random.default_rng(17)
    image += rng.normal(0, 3, image.shape)
    for separator_index, base_x in enumerate(separator_x):
        for y in range(height):
            if (35 + separator_index * 19) <= y < (
                57 + separator_index * 19
            ):
                continue
            drift = int(round(3.0 * np.sin(y / 75.0 + separator_index)))
            x = base_x + drift
            image[
                y,
                max(0, x - half_width) : min(width, x + half_width + 1),
            ] = 225
    return np.clip(image, 0, 255).astype(np.uint8)


class SeparatorPathPrototypeTests(unittest.TestCase):
    def test_broken_tilted_paths_are_selected_without_crossing(self):
        image = synthetic_broken_separators()
        result = prototype.detect_separator_paths(
            image,
            {"x": 250, "y": 120},
            {"x0": 0, "y0": 0, "x1": 500, "y1": 240},
        )
        self.assertEqual("available", result["status"])
        self.assertFalse(result["crossing"])
        self.assertEqual(set(prototype.ROLE_ORDER), set(result["selection"]))
        selected_x = [
            result["selection"][role]["x_at_reference_roi"]
            for role in prototype.ROLE_ORDER
        ]
        self.assertTrue(
            all(left < right for left, right in zip(selected_x, selected_x[1:]))
        )
        expected = (145, 205, 295, 355)
        for predicted, target in zip(selected_x, expected):
            self.assertLess(abs(predicted - target), 5.0)

    def test_reference_on_separator_is_safe_unavailable(self):
        image = synthetic_broken_separators()
        result = prototype.detect_separator_paths(
            image,
            {"x": 205, "y": 120},
            {"x0": 0, "y0": 0, "x1": 500, "y1": 240},
        )
        self.assertEqual("unavailable", result["status"])
        self.assertEqual(
            "reference_on_separator",
            result["unavailable_reason"],
        )
        self.assertEqual({}, result["selection"])

    def test_one_wide_bright_ridge_is_not_split_into_two_paths(self):
        image = synthetic_broken_separators(half_width=11)
        result = prototype.detect_separator_paths(
            image,
            {"x": 250, "y": 120},
            {"x0": 0, "y0": 0, "x1": 500, "y1": 240},
        )
        accepted = [
            candidate
            for candidate in result["candidates"]
            if candidate["accepted"]
        ]
        self.assertEqual(4, len(accepted))
        self.assertEqual("available", result["status"])

    def test_uniform_raw_roi_is_unavailable(self):
        image = np.full((240, 500), 80, dtype=np.uint8)
        result = prototype.detect_separator_paths(
            image,
            {"x": 250, "y": 120},
            {"x0": 0, "y0": 0, "x1": 500, "y1": 240},
        )
        self.assertEqual("unavailable", result["status"])
        self.assertEqual(
            "no_accepted_separator_paths",
            result["unavailable_reason"],
        )

    def test_development_loader_rejects_other_split(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "other.json"
            path.write_text(
                json.dumps(
                    {
                        "dataset_version": prototype.DATASET_VERSION,
                        "split": "evaluation",
                        "annotations": {"H001": {}},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                prototype.load_development_document(path)

    def test_source_does_not_import_production_or_old_detection(self):
        source = Path(prototype.__file__).read_text(encoding="utf-8")
        forbidden = (
            "interactive_pipeline",
            "interactive_analysis",
            "pitch_reference",
            "prototype_column_basins",
            "cv2.threshold",
            "cv2.adaptiveThreshold",
            "cv2.morphologyEx",
            "heldout.json",
        )
        for fragment in forbidden:
            self.assertNotIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
