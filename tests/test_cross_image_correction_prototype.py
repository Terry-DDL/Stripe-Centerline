import copy
import unittest

import numpy as np

from tools import cross_image_correction_prototype as prototype


def candidate(seed_x: int, path_x: float) -> dict:
    return {
        "candidate_id": "C01",
        "seed_x_roi": seed_x,
        "seed_response": 2.0,
        "band_centers_y_roi": [10.0, 30.0, 50.0],
        "x_by_band_roi": [path_x, path_x, path_x],
        "response_by_band": [2.0, 2.0, 2.0],
        "supported_band_indices": [0, 1, 2],
        "support_fraction": 1.0,
        "top_response": 2.0,
        "mean_step_px": 0.0,
        "path_score": 2.45,
        "peak_width_px": 20,
        "accepted": True,
        "rejection_reason": None,
        "suppressed_candidates": [],
    }


def evidence(with_dark_valley: bool = False) -> dict:
    profiles = np.full((3, 100), 10.0, dtype=np.float64)
    profiles[:, 30:50] = 230.0
    if with_dark_valley:
        profiles[:, 37:43] = 10.0
    responses = np.zeros_like(profiles)
    responses[profiles > 100] = 2.0
    return {
        "profiles": profiles,
        "responses": responses,
    }


class PlateauCenterizationTests(unittest.TestCase):
    def test_broad_paired_edges_become_one_center_path(self):
        first = candidate(30, 30.0)
        second = candidate(49, 49.0)
        frozen = copy.deepcopy(first)
        frozen["suppressed_candidates"] = [
            {
                "seed_x_roi": 49,
                "path_score": second["path_score"],
                "reason": "same_physical_bright_ridge",
            }
        ]

        transformed, mappings = (
            prototype.centerize_frozen_candidates(
                [frozen],
                [first, second],
                evidence(),
                local_pitch_px=50.0,
            )
        )

        self.assertTrue(mappings[0]["adopted"])
        self.assertAlmostEqual(
            np.median(transformed[0]["x_by_band_roi"]),
            39.5,
            delta=0.75,
        )
        self.assertEqual(transformed[0]["candidate_id"], "C01")

    def test_dark_valley_prevents_edge_merge(self):
        first = candidate(30, 30.0)
        second = candidate(49, 49.0)
        frozen = copy.deepcopy(first)
        frozen["suppressed_candidates"] = [
            {
                "seed_x_roi": 49,
                "path_score": second["path_score"],
                "reason": "same_physical_bright_ridge",
            }
        ]

        transformed, mappings = (
            prototype.centerize_frozen_candidates(
                [frozen],
                [first, second],
                evidence(with_dark_valley=True),
                local_pitch_px=50.0,
            )
        )

        self.assertFalse(mappings[0]["adopted"])
        self.assertEqual(
            transformed[0]["x_by_band_roi"],
            frozen["x_by_band_roi"],
        )

    def test_narrow_duplicate_keeps_frozen_path_exactly(self):
        first = candidate(30, 30.0)
        second = candidate(32, 32.0)
        frozen = copy.deepcopy(first)
        frozen["suppressed_candidates"] = [
            {
                "seed_x_roi": 32,
                "path_score": second["path_score"],
                "reason": "same_physical_bright_ridge",
            }
        ]

        transformed, mappings = (
            prototype.centerize_frozen_candidates(
                [frozen],
                [first, second],
                evidence(),
                local_pitch_px=50.0,
            )
        )

        self.assertFalse(mappings[0]["adopted"])
        self.assertEqual(
            transformed[0]["x_by_band_roi"],
            frozen["x_by_band_roi"],
        )


class BasinRefinementTests(unittest.TestCase):
    def test_refined_path_stays_inside_original_basin(self):
        profiles = []
        for _ in range(12):
            profile = np.full(80, 220.0, dtype=np.float64)
            profile[20:60] = (
                20.0
                + np.abs(np.arange(40, dtype=np.float64) - 23.0)
            )
            profiles.append(profile.tolist())
        basin = {
            "basin_id": "B_C01_C02",
            "left_separator_id": "C01",
            "right_separator_id": "C02",
            "left_x_by_band_roi": [20.0] * 12,
            "right_x_by_band_roi": [60.0] * 12,
            "center_x_by_band_roi": [40.0] * 12,
        }

        result = prototype._refine_one_basin(
            basin,
            profiles,
            prototype.DEFAULT_CORRECTION_CONFIG,
        )

        self.assertTrue(result["adopted"])
        self.assertTrue(
            all(
                20.0 < center < 60.0
                for center in result[
                    "bandwise_center_path_x_roi"
                ]
            )
        )
        self.assertLess(
            result["refined_minimum_error_median_px"],
            result["baseline_minimum_error_median_px"],
        )

    def test_inconsistent_bands_keep_arithmetic_center(self):
        profiles = []
        for band_index in range(12):
            profile = np.full(80, 220.0, dtype=np.float64)
            minimum = 22 if band_index % 2 == 0 else 57
            profile[minimum] = 0.0
            profiles.append(profile.tolist())
        basin = {
            "basin_id": "B_C01_C02",
            "left_separator_id": "C01",
            "right_separator_id": "C02",
            "left_x_by_band_roi": [20.0] * 12,
            "right_x_by_band_roi": [60.0] * 12,
            "center_x_by_band_roi": [40.0] * 12,
        }

        result = prototype._refine_one_basin(
            basin,
            profiles,
            prototype.DEFAULT_CORRECTION_CONFIG,
        )

        self.assertFalse(result["adopted"])
        self.assertEqual(
            result["bandwise_center_path_x_roi"],
            [40.0] * 12,
        )


class EvaluationIsolationTests(unittest.TestCase):
    def test_empty_candidate_scale_falls_back_to_roi_width(self):
        self.assertEqual(
            prototype._local_pitch_scale(
                {"diagnostic_pitch_px": None},
                [],
                reference_y_roi=10.0,
                roi_width_px=500,
            ),
            500.0,
        )

    def test_manifest_forbids_separator_heldout(self):
        manifest = prototype.load_evaluation_manifest()
        self.assertFalse(
            manifest["access_policy"]["separator_heldout_allowed"]
        )
        self.assertEqual(len(manifest["sample2_issues"]), 10)
        self.assertEqual(len(manifest["sample2_random"]), 17)

    def test_all_new_scale_parameters_are_dimensionless_ratios(self):
        parameter_names = vars(
            prototype.DEFAULT_CORRECTION_CONFIG
        ).keys()
        self.assertTrue(all(not name.endswith("_px") for name in parameter_names))


if __name__ == "__main__":
    unittest.main()
