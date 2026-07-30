import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from tools import cross_image_correction_prototype_v1_1 as prototype


class CrossImageCorrectionV11Tests(unittest.TestCase):
    def synthetic_plateau(self):
        band_count = 12
        width = 100
        profiles = np.zeros((band_count, width), dtype=np.float64)
        profiles[:, 30:50] = 255.0
        responses = np.zeros((band_count, width), dtype=np.float64)
        responses[:, 30] = 1.5
        responses[:, 49] = 1.4
        evidence = {
            "profiles": profiles,
            "responses": responses,
        }
        left = {
            "raw_candidate_id": "R01",
            "seed_x_roi": 30,
            "x_by_band_roi": [30.0] * band_count,
            "response_by_band": [1.5] * band_count,
            "support_fraction": 1.0,
        }
        right = {
            "raw_candidate_id": "R02",
            "seed_x_roi": 49,
            "x_by_band_roi": [49.0] * band_count,
            "response_by_band": [1.4] * band_count,
            "support_fraction": 1.0,
        }
        frozen = {
            "candidate_id": "C01",
            "seed_x_roi": 30,
            "x_by_band_roi": [30.0] * band_count,
            "response_by_band": [1.5] * band_count,
            "supported_band_indices": list(range(band_count)),
            "support_fraction": 1.0,
            "top_response": 1.5,
            "mean_step_px": 0.0,
            "path_score": 1.95,
            "peak_width_px": 1,
            "accepted": True,
            "rejection_reason": None,
            "band_centers_y_roi": [
                float(index * 10 + 5) for index in range(band_count)
            ],
        }
        group_evidence = {
            "eligible": True,
            "reason": (
                "paired_edges_enclose_one_continuous_bright_plateau"
            ),
            "band_agreement_fraction": 1.0,
            "median_span_pitch_ratio": 0.38,
            "bands": [
                {
                    "same_plateau": True,
                    "span_px": 19.0,
                    "span_pitch_ratio": 0.38,
                }
                for _ in range(band_count)
            ],
        }
        return frozen, [left, right], evidence, group_evidence

    def test_plateau_support_comes_from_source_edges(self):
        frozen, paths, evidence, group = self.synthetic_plateau()
        candidate = prototype._centerized_candidate_v1_1(
            frozen,
            paths,
            evidence,
            50.0,
            group,
        )
        center = int(round(np.median(candidate["x_by_band_roi"])))
        self.assertEqual(float(evidence["responses"][0, center]), 0.0)
        self.assertEqual(candidate["support_fraction"], 1.0)
        self.assertTrue(candidate["accepted"])
        support = candidate["plateau_centerization"][
            "plateau_support"
        ]
        self.assertEqual(
            support["source_edge_ids"],
            {"left": "R01", "right": "R02"},
        )
        self.assertEqual(support["paired_band_fraction"], 1.0)
        self.assertEqual(support["plateau_coverage_fraction"], 1.0)
        self.assertEqual(support["rejection_reasons"], [])

    def test_width_aware_reference_core_is_definite(self):
        frozen, paths, evidence, group = self.synthetic_plateau()
        candidate = prototype._centerized_candidate_v1_1(
            frozen,
            paths,
            evidence,
            50.0,
            group,
        )
        relation = prototype._plateau_reference_classification(
            candidate,
            40.0,
        )
        self.assertEqual(
            relation["classification"],
            "definite_on_separator",
        )
        self.assertEqual(relation["core_band_fraction"], 1.0)

    def test_width_aware_reference_edge_is_ambiguous(self):
        frozen, paths, evidence, group = self.synthetic_plateau()
        candidate = prototype._centerized_candidate_v1_1(
            frozen,
            paths,
            evidence,
            50.0,
            group,
        )
        geometry = candidate["plateau_centerization"]
        for band in geometry["band_geometry"]:
            band["core_left_x_roi"] = 33.0
            band["core_right_x_roi"] = 46.0
        relation = prototype._plateau_reference_classification(
            candidate,
            31.0,
        )
        self.assertEqual(relation["classification"], "ambiguous")
        self.assertEqual(
            relation["reason"],
            "reference_in_plateau_edge_uncertainty",
        )

    def test_reference_outside_plateau_uses_frozen_fallback(self):
        frozen, paths, evidence, group = self.synthetic_plateau()
        candidate = prototype._centerized_candidate_v1_1(
            frozen,
            paths,
            evidence,
            50.0,
            group,
        )
        relation = prototype._plateau_reference_classification(
            candidate,
            70.0,
        )
        self.assertEqual(
            relation["classification"],
            "outside_plateau",
        )

    def test_narrow_path_has_no_width_aware_override(self):
        relation = prototype._plateau_reference_classification(
            {
                "candidate_id": "C01",
                "plateau_centerization": {"adopted": False},
            },
            20.0,
        )
        self.assertEqual(relation["classification"], "not_plateau")

    def test_preregistered_manifest_is_fixed_grid(self):
        manifest_path = (
            prototype.SMOKE_MANIFEST_PATH
        )
        manifest = json.loads(manifest_path.read_text())
        self.assertTrue(manifest["registered_before_v1_1_execution"])
        self.assertFalse(
            manifest["selection_method"]["parameter_selection_allowed"]
        )
        self.assertEqual(len(manifest["images"]), 5)
        self.assertTrue(
            all(len(item["points"]) == 6 for item in manifest["images"])
        )
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "6b9f51385e12ab58920c5a440be2c18712dc7f4261c8b30236a2c564c4f029ff",
        )

    def test_freeze_forbids_heldout_and_runtime_gt(self):
        freeze = json.loads(prototype.FREEZE_PATH.read_text())
        self.assertFalse(freeze["separator_heldout_allowed"])
        self.assertFalse(freeze["runtime_gt_allowed"])
        source = Path(prototype.__file__).read_text()
        self.assertNotIn("separator_path_gt_v1/heldout", source)

    def test_configuration_checksum_is_deterministic(self):
        self.assertEqual(
            prototype.configuration_checksum(),
            prototype.configuration_checksum(),
        )


if __name__ == "__main__":
    unittest.main()
