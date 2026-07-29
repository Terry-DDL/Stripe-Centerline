import json
import os
from pathlib import Path
import unittest

import numpy as np

from tools import raw_local_pitch_prototype as v2
from tools import raw_local_pitch_prototype_v3 as prototype


def synthetic_vertical_stripes(
    pitch: float,
    height: int = 200,
    width: int = 500,
    seed: int = 11,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.arange(width, dtype=np.float64)
    phase = np.mod(x, pitch)
    white = np.exp(
        -0.5 * np.square((phase - pitch * 0.25) / (pitch * 0.10))
    )
    image = np.tile(25.0 + 190.0 * white, (height, 1))
    image += np.linspace(-10, 15, height)[:, None]
    image += rng.normal(0, 4, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def hypothesis(
    identifier: str,
    period: float,
    score: float,
    spectral: float,
) -> dict:
    return {
        "hypothesis_id": identifier,
        "period_px": period,
        "supporting_band_count": 5,
        "autocorrelation_band_count": 5,
        "spectral_band_count": 5,
        "qualified_autocorrelation_band_count": 5,
        "qualified_spectral_band_count": 5,
        "cross_source_band_count": 5,
        "relative_mad": 0.01,
        "median_autocorrelation": 0.65,
        "median_spectral_support": spectral,
        "aggregate_score": score,
        "band_support": [],
    }


class RawLocalPitchPrototypeV3Tests(unittest.TestCase):
    def test_synthetic_pitch_is_high_and_atomic(self):
        result = prototype.estimate_raw_local_pitch_v3(
            synthetic_vertical_stripes(24.0),
            {"x": 250, "y": 100},
            {"x0": 0, "y0": 0, "x1": 500, "y1": 200},
        )
        self.assertEqual("high", result["confidence"])
        self.assertTrue(result["success_eligible"])
        self.assertLess(
            abs(result["usable_pitch_px"] - 24.0) / 24.0,
            0.05,
        )
        selected = next(
            hypothesis
            for hypothesis in result["hypotheses"]
            if (
                hypothesis["hypothesis_id"]
                == result["arbitration"]["selected_hypothesis_id"]
            )
        )
        self.assertEqual(
            selected["period_px"],
            result["diagnostic_pitch_px"],
        )
        self.assertEqual(
            selected["period_px"],
            result["usable_pitch_px"],
        )

    def test_low_contrast_is_unavailable(self):
        result = prototype.estimate_raw_local_pitch_v3(
            np.full((200, 500), 80, dtype=np.uint8),
            {"x": 250, "y": 100},
            {"x0": 0, "y0": 0, "x1": 500, "y1": 200},
        )
        self.assertEqual("unavailable", result["confidence"])
        self.assertFalse(result["success_eligible"])
        self.assertIsNone(result["usable_pitch_px"])
        self.assertEqual(
            "no_cross_band_hypothesis",
            result["unavailable_reason"],
        )

    def test_spectral_candidate_survives_without_acf_candidate(self):
        band_results = []
        for band_index in range(5):
            band_results.append(
                {
                    "band_index": band_index,
                    "candidates": [
                        {
                            "source": "spectrum",
                            "period_px": 38.0 + 0.05 * band_index,
                            "autocorrelation": 0.02,
                            "spectral_support": 0.90,
                        },
                        {
                            "source": "autocorrelation",
                            "period_px": 26.0 + 0.04 * band_index,
                            "autocorrelation": 0.30,
                            "spectral_support": 0.20,
                        },
                    ],
                }
            )
        hypotheses = prototype.build_unified_hypotheses(
            band_results,
            prototype.DEFAULT_CONFIG,
        )
        correct = min(
            hypotheses,
            key=lambda candidate: abs(candidate["period_px"] - 38.0),
        )
        self.assertLess(abs(correct["period_px"] - 38.0), 0.2)
        self.assertEqual(5, correct["spectral_band_count"])
        self.assertEqual(0, correct["autocorrelation_band_count"])
        self.assertGreater(correct["aggregate_score"], 0.70)

    def test_weak_candidates_are_retained_before_aggregation(self):
        profile = np.sin(np.arange(500) * 2.0 * np.pi / 24.0)
        profile += 0.15 * np.sin(
            np.arange(500) * 2.0 * np.pi / 37.0
        )
        profile = 120.0 + 40.0 * profile
        band = prototype._band_candidate_evidence(
            profile,
            0,
            (0, 40),
            prototype.DEFAULT_CONFIG,
        )
        self.assertTrue(band["evidence_available"])
        self.assertTrue(
            any(
                candidate["source"] == "spectrum"
                and candidate["spectral_support"]
                < prototype.DEFAULT_CONFIG.aggregate_spectral_evidence_floor
                for candidate in band["candidates"]
            )
        )

    def test_close_nonharmonic_hypotheses_fail_safely(self):
        selected = hypothesis("H01", 26.0, 0.80, 1.0)
        competitor = hypothesis("H02", 38.0, 0.56, 0.35)
        result = prototype._arbitrate_hypotheses(
            [selected, competitor],
            prototype.DEFAULT_CONFIG,
        )
        self.assertEqual(
            "ambiguous_competing_hypotheses",
            result["rejection_reason"],
        )
        self.assertEqual(1, len(result["close_nonharmonic_competitors"]))

    def test_bidirectional_double_and_triple_conflicts_fail_safely(self):
        cases = (
            (20.0, 40.0, "selected_is_smaller", 2),
            (60.0, 20.0, "selected_is_larger", 3),
        )
        for selected_period, competitor_period, direction, multiple in cases:
            with self.subTest(
                selected_period=selected_period,
                competitor_period=competitor_period,
            ):
                selected = hypothesis(
                    "H01",
                    selected_period,
                    0.80,
                    0.50,
                )
                competitor = hypothesis(
                    "H02",
                    competitor_period,
                    0.75,
                    0.48,
                )
                result = prototype._arbitrate_hypotheses(
                    [selected, competitor],
                    prototype.DEFAULT_CONFIG,
                )
                self.assertEqual(
                    "harmonic_ambiguous",
                    result["rejection_reason"],
                )
                conflict = result[
                    "unresolved_harmonic_competitors"
                ][0]
                self.assertEqual(direction, conflict["direction"])
                self.assertEqual(multiple, conflict["nearest_multiple"])

    def test_medium_or_low_can_never_be_success_eligible(self):
        config = prototype.RawLocalPitchV3Config(
            high_aggregate_score_floor=1.1
        )
        result = prototype.estimate_raw_local_pitch_v3(
            synthetic_vertical_stripes(24.0),
            {"x": 250, "y": 100},
            {"x0": 0, "y0": 0, "x1": 500, "y1": 200},
            config=config,
        )
        self.assertIn(result["confidence"], {"medium", "low"})
        self.assertFalse(result["success_eligible"])
        self.assertIsNone(result["usable_pitch_px"])

    def test_p026_remains_harmonically_safe(self):
        document = json.loads(
            v2.P026_VALIDATION_PATH.read_text(encoding="utf-8")
        )
        annotation = document["annotations"]["P026"]
        image_path = Path(
            os.environ.get("STRIPE_TEST_IMAGES_DIR", str(v2.IMAGES_DIR))
        ) / annotation["image_name"]
        if not image_path.exists():
            self.skipTest("P026 source image is not present in this worktree")
        image = v2.load_grayscale_image(image_path)
        result = prototype.estimate_raw_local_pitch_v3(
            image,
            annotation["reference_global"],
            annotation["roi_bounds_global"],
        )
        self.assertFalse(result["success_eligible"])
        self.assertIsNone(result["usable_pitch_px"])
        self.assertIn(
            result["unavailable_reason"],
            {
                "harmonic_ambiguous",
                "ambiguous_competing_hypotheses",
            },
        )

    def test_configuration_checksum_is_stable(self):
        checksum = prototype.configuration_checksum(
            prototype.DEFAULT_CONFIG
        )
        self.assertEqual(64, len(checksum))
        changed = prototype.RawLocalPitchV3Config(
            profile_percentile=71.0
        )
        self.assertNotEqual(
            checksum,
            prototype.configuration_checksum(changed),
        )

    def test_frozen_configuration_matches_code(self):
        path = (
            Path(prototype.__file__).resolve().parent.parent
            / "configs"
            / "raw_local_pitch_stage1_v3.json"
        )
        frozen = json.loads(path.read_text(encoding="utf-8"))
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
            frozen["internal_acceptance"]["acceptance_passed"]
        )
        self.assertFalse(frozen["independent_evaluation"])

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
