import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tools import raw_local_pitch_prototype as prototype


def synthetic_vertical_stripes(
    pitch: float,
    height: int = 200,
    width: int = 500,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.arange(width, dtype=np.float64)
    phase = np.mod(x, pitch)
    white = np.exp(
        -0.5 * np.square((phase - pitch * 0.25) / (pitch * 0.10))
    )
    base = 25.0 + 190.0 * white
    image = np.tile(base, (height, 1))
    image += np.linspace(-12, 18, height)[:, None]
    image += rng.normal(0, 4, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


class RawLocalPitchPrototypeTests(unittest.TestCase):
    def test_synthetic_raw_gray_pitch(self):
        image = synthetic_vertical_stripes(24.0)
        result = prototype.estimate_raw_local_pitch(
            image,
            {"x": 250, "y": 100},
            {"x0": 0, "y0": 0, "x1": 500, "y1": 200},
        )
        self.assertIsNotNone(result["pitch_px"])
        self.assertLess(abs(result["pitch_px"] - 24.0) / 24.0, 0.05)
        self.assertIn(result["confidence"], {"high", "medium"})

    def test_low_contrast_is_unavailable(self):
        image = np.full((200, 500), 80, dtype=np.uint8)
        result = prototype.estimate_raw_local_pitch(
            image,
            {"x": 250, "y": 100},
            {"x0": 0, "y0": 0, "x1": 500, "y1": 200},
        )
        self.assertIsNone(result["pitch_px"])
        self.assertEqual("unavailable", result["confidence"])
        self.assertEqual(
            "insufficient_periodic_subwindows",
            result["unavailable_reason"],
        )

    def test_horizontal_direction_transposes_pitch_axis(self):
        image = synthetic_vertical_stripes(
            20.0,
            height=500,
            width=200,
        ).T
        result = prototype.estimate_raw_local_pitch(
            image,
            {"x": 250, "y": 100},
            {"x0": 0, "y0": 0, "x1": 500, "y1": 200},
            direction="horizontal",
        )
        self.assertIsNotNone(result["pitch_px"])
        self.assertLess(abs(result["pitch_px"] - 20.0) / 20.0, 0.05)

    def test_configuration_checksum_is_stable_and_parameter_sensitive(self):
        checksum = prototype.configuration_checksum(
            prototype.DEFAULT_CONFIG
        )
        self.assertEqual(64, len(checksum))
        changed = prototype.RawLocalPitchConfig(
            profile_percentile=71.0
        )
        self.assertNotEqual(
            checksum,
            prototype.configuration_checksum(changed),
        )

    def test_frozen_configuration_matches_code_and_checksum(self):
        frozen_path = (
            prototype.PROJECT_ROOT
            / "configs"
            / "raw_local_pitch_stage1_v1.json"
        )
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        self.assertEqual(
            prototype.ALGORITHM_REVISION,
            frozen["algorithm_revision"],
        )
        self.assertEqual(
            prototype.canonical_configuration(
                prototype.DEFAULT_CONFIG
            )["parameters"],
            frozen["parameters"],
        )
        self.assertEqual(
            prototype.configuration_checksum(prototype.DEFAULT_CONFIG),
            frozen["configuration_checksum"],
        )
        self.assertTrue(
            frozen["development_acceptance"]["acceptance_passed"]
        )
        self.assertFalse(frozen["heldout_evaluated"])

    def test_development_loader_refuses_non_development_document(self):
        original_path = prototype.DEVELOPMENT_PATH
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "not-development.json"
            path.write_text(
                json.dumps(
                    {
                        "dataset_version": prototype.DATASET_VERSION,
                        "split": "held-out",
                        "access_policy": (
                            "evaluation_only_after_parameter_freeze"
                        ),
                        "annotations": {},
                    }
                ),
                encoding="utf-8",
            )
            prototype.DEVELOPMENT_PATH = path
            try:
                with self.assertRaises(ValueError):
                    prototype.evaluate_development()
            finally:
                prototype.DEVELOPMENT_PATH = original_path

    def test_source_has_no_production_detector_imports(self):
        source = Path(prototype.__file__).read_text(encoding="utf-8")
        forbidden = (
            "interactive_pipeline",
            "interactive_analysis",
            "pitch_reference",
            "prototype_column_basins",
            "cv2.threshold",
            "cv2.morphologyEx",
            "adaptiveThreshold",
        )
        for fragment in forbidden:
            self.assertNotIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
