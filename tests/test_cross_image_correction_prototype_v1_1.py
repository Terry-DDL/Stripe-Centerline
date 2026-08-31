import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools import cross_image_correction_prototype_v1_1 as prototype


class CrossImageCorrectionV11Tests(unittest.TestCase):
    def test_reused_pitch_result_is_deep_copied(self):
        frozen = {
            "diagnostic_pitch_px": 40.0,
            "nested": {"values": [1, 2, 3]},
        }
        with patch.object(
            prototype.raw_pitch,
            "estimate_raw_local_pitch_v3",
        ) as estimate:
            reused = prototype._reuse_or_estimate_pitch_result(
                frozen,
                np.zeros((4, 4), dtype=np.uint8),
                {"x": 2, "y": 2},
                {"x0": 0, "y0": 0, "x1": 4, "y1": 4},
                "vertical",
            )

        estimate.assert_not_called()
        self.assertEqual(frozen, reused)
        self.assertIsNot(frozen, reused)
        self.assertIsNot(frozen["nested"], reused["nested"])
        reused["nested"]["values"].append(4)
        self.assertEqual([1, 2, 3], frozen["nested"]["values"])

    def test_basin_graph_reuse_key_uses_only_graph_inputs(self):
        first = {
            "reference_y_roi": 10.0,
            "candidates": [
                {
                    "candidate_id": "C01",
                    "accepted": True,
                    "band_centers_y_roi": [5.0, 15.0],
                    "x_by_band_roi": [20.0, 21.0],
                    "plateau_centerization": {"adopted": False},
                }
            ],
        }
        second = copy.deepcopy(first)
        second["candidates"][0]["plateau_centerization"] = {
            "adopted": False,
            "diagnostic": "different metadata",
        }

        self.assertTrue(prototype._basin_graph_inputs_match(first, second))

        second["candidates"][0]["x_by_band_roi"][1] = 22.0
        self.assertFalse(prototype._basin_graph_inputs_match(first, second))

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

    def semantic_fixture(self, reference_x):
        band_count = 12
        profiles = np.zeros((band_count, 100), dtype=np.float64)
        profiles[:, 48:53] = 255.0
        evidence = {"profiles": profiles}

        def path(candidate_id, x):
            return {
                "candidate_id": candidate_id,
                "accepted": True,
                "x_by_band_roi": [float(x)] * band_count,
            }

        candidates = [
            path("C01", 30),
            path("C02", 50),
            path("C03", 70),
        ]
        return prototype._reference_grayscale_semantic_classification(
            candidates[1],
            candidates,
            reference_x,
            evidence,
            local_pitch_px=20.0,
        )

    def test_reference_gray_semantic_resolves_inside_basin(self):
        result = self.semantic_fixture(56.0)
        self.assertEqual(result["classification"], "inside_basin")
        self.assertEqual(result["inside_basin_band_fraction"], 1.0)

    def test_reference_gray_semantic_preserves_on_separator(self):
        result = self.semantic_fixture(50.0)
        self.assertEqual(result["classification"], "on_separator")
        self.assertEqual(result["on_separator_band_fraction"], 1.0)

    def test_reference_gray_semantic_keeps_inconsistent_bands_ambiguous(
        self,
    ):
        band_count = 12
        profiles = np.zeros((band_count, 100), dtype=np.float64)
        profiles[:6, 48:57] = 255.0
        profiles[6:, 48:53] = 255.0
        evidence = {"profiles": profiles}

        def path(candidate_id, x):
            return {
                "candidate_id": candidate_id,
                "accepted": True,
                "x_by_band_roi": [float(x)] * band_count,
            }

        candidates = [path("C01", 30), path("C02", 50), path("C03", 70)]
        result = prototype._reference_grayscale_semantic_classification(
            candidates[1],
            candidates,
            56.0,
            evidence,
            local_pitch_px=20.0,
        )
        self.assertEqual(result["classification"], "ambiguous")

    def test_intermediate_dark_profile_uses_existing_contrast_gate(self):
        profile = np.full(100, 20.0, dtype=np.float64)
        profile[31:41] = np.linspace(20.0, 200.0, 10)
        profile[41:51] = np.linspace(200.0, 10.0, 10)
        profile[51:61] = np.linspace(10.0, 200.0, 10)
        profile[61:70] = np.linspace(200.0, 20.0, 9)
        result = prototype._intermediate_dark_profile_candidate(
            profile, 30.0, 70.0
        )
        self.assertIsNotNone(result)
        self.assertTrue(result["supported"])
        self.assertGreaterEqual(
            result["normalized_contrast"],
            prototype.separator.DEFAULT_CONFIG.minimum_dark_basin_contrast,
        )

    def test_local_valleys_keep_weak_but_stable_ridge_separate(self):
        profile = np.full(100, 20.0, dtype=np.float64)
        profile[30:71] = np.interp(
            np.arange(30, 71),
            [30, 40, 50, 60, 70],
            [200.0, 10.0, 22.0, 10.0, 200.0],
        )
        valleys = prototype._raw_dark_valleys_in_interval(
            profile, 30.0, 70.0
        )
        self.assertEqual(
            [item["minimum_x_roi"] for item in valleys],
            [40.0, 60.0],
        )

    @unittest.skipUnless(
        (
            prototype.PROJECT_ROOT
            / "images"
            / "Stripe_10_e0_t221236602_v8p56736_retry.bmp"
        ).is_file(),
        "Stripe 10 offline acceptance image is not installed",
    )
    def test_three_merged_clicked_basins_return_direct_neighbors(self):
        image = cv2.imread(
            str(
                prototype.PROJECT_ROOT
                / "images"
                / "Stripe_10_e0_t221236602_v8p56736_retry.bmp"
            ),
            cv2.IMREAD_GRAYSCALE,
        )
        cases = {
            (1910, 1144): (1892.0, 1922.0),
            (651, 825): (642.0, 668.7),
            (747, 335): (732.2, 759.0),
        }
        for (x, y), expected in cases.items():
            case = {"reference_global": {"x": x, "y": y}}
            bounds = prototype.v1._case_roi(case, image)  # noqa: SLF001
            result = prototype.run_joint_case_v1_1(
                image, case["reference_global"], bounds
            )
            self.assertTrue(result["success"])
            hypothesis = result["final_hypothesis"]
            self.assertTrue(hypothesis["atomic"])
            geometry = hypothesis["geometry"]["basins"]
            actual = tuple(
                geometry[side].get(
                    "center_x_at_reference_global",
                    bounds["x0"]
                    + geometry[side]["center_x_at_reference_roi"],
                )
                for side in ("left", "right")
            )
            for value, target in zip(actual, expected):
                self.assertAlmostEqual(value, target, delta=2.0)

    @unittest.skipUnless(
        (prototype.PROJECT_ROOT / "images" / "Sample 1.bmp").is_file(),
        "Sample 1 offline regression image is not installed",
    )
    def test_local_contrast_recovers_broken_white_paths_near_click(self):
        image = cv2.imread(
            str(prototype.PROJECT_ROOT / "images" / "Sample 1.bmp"),
            cv2.IMREAD_GRAYSCALE,
        )
        reference = {"x": 336, "y": 811}
        bounds = prototype.v1._case_roi(  # noqa: SLF001
            {"reference_global": reference}, image
        )
        result = prototype.run_joint_case_v1_1(
            image, reference, bounds
        )

        self.assertTrue(result["success"])
        recovery = result["debug"]["local_dark_valley_recovery"]
        self.assertEqual(
            recovery["mode"], "separator_sequence_local_contrast"
        )
        self.assertEqual(
            recovery["source_failure"],
            "dark_basin_sequence_not_verified",
        )
        self.assertTrue(
            all(item["supported"] for item in recovery["local_evidence"])
        )
        self.assertEqual(
            result["final_hypothesis"]["separator_sequence"],
            ["C16", "C17", "C18", "C19"],
        )
        basins = result["final_hypothesis"]["geometry"]["basins"]
        self.assertAlmostEqual(
            basins["left"]["center_x_at_reference_roi"], 234.5
        )
        self.assertAlmostEqual(
            basins["right"]["center_x_at_reference_roi"], 264.0
        )

        separator_result = result["debug"]["separator_result"]
        graph = result["debug"]["basin_graph"]
        low_pitch = {
            **result["debug"]["raw_pitch_result"],
            "confidence": "low",
            "success_eligible": False,
        }
        rejected_pitch = (
            prototype._recover_contrast_limited_separator_sequence(
                image,
                bounds,
                "vertical",
                separator_result,
                graph,
                low_pitch,
            )
        )
        self.assertFalse(rejected_pitch["success"])
        self.assertEqual(
            rejected_pitch["reason"],
            "high_pitch_does_not_support_recovery",
        )

        no_local_contrast = image.copy()
        reference_y_roi = int(separator_result["reference_y_roi"])
        y_global = bounds["y0"] + reference_y_roi
        half_window = int(
            round(prototype.separator.DEFAULT_CONFIG.match_tolerance_px)
        )
        no_local_contrast[
            y_global - half_window : y_global + half_window + 1,
            bounds["x0"] : bounds["x1"],
        ] = 0
        rejected_local = (
            prototype._recover_contrast_limited_separator_sequence(
                no_local_contrast,
                bounds,
                "vertical",
                separator_result,
                graph,
                result["debug"]["raw_pitch_result"],
            )
        )
        self.assertFalse(rejected_local["success"])
        self.assertEqual(
            rejected_local["reason"],
            "local_reference_contrast_not_verified",
        )

    def test_local_contrast_does_not_bypass_structural_failure(self):
        result = prototype._recover_local_dark_valley_sequence(
            np.zeros((20, 20), dtype=np.uint8),
            {"x0": 0, "y0": 0, "x1": 20, "y1": 20},
            "vertical",
            {
                "status": "unavailable",
                "unavailable_reason": "basin_structure_not_verified",
                "arbitration_debug": {
                    "role_hypotheses": [
                        {
                            "type": "clicked_basin_roles",
                            "crossing": False,
                            "rejection_reasons": [
                                "role_path_geometry_unstable"
                            ],
                        }
                    ]
                },
            },
            {"ordered_separator_ids": [], "basin_candidates": []},
            None,
            {},
        )
        self.assertFalse(result["success"])
        self.assertFalse(result["triggered"])
        self.assertEqual(result["reason"], "not_contrast_coverage_only")

    def test_runtime_recovery_has_no_image_or_coordinate_special_case(self):
        source = Path(prototype.__file__).read_text()
        for forbidden in (
            "1910",
            "1144",
            "651",
            "825",
            "747",
            "335",
            "336",
            "811",
            "Stripe_10_e0_t221236602_v8p56736_retry.bmp",
            "Sample 1.bmp",
        ):
            self.assertNotIn(forbidden, source)

    @unittest.skipUnless(
        (prototype.PROJECT_ROOT / "images" / "Sample 2.bmp").is_file(),
        "Sample 2 offline regression image is not installed",
    )
    def test_sample2_issue_and_random_identities_do_not_change(self):
        image = cv2.imread(
            str(prototype.PROJECT_ROOT / "images" / "Sample 2.bmp"),
            cv2.IMREAD_GRAYSCALE,
        )
        manifest = prototype.v1.load_evaluation_manifest()
        for group_name in ("sample2_issues", "sample2_random"):
            for case in manifest[group_name]:
                full_case = {**case, "image_name": "Sample 2.bmp"}
                bounds = prototype.v1._case_roi(  # noqa: SLF001
                    full_case, image
                )
                result = prototype.run_joint_case_v1_1(
                    image, case["reference_global"], bounds
                )
                expected = case["expected"]
                self.assertEqual(
                    result["success"],
                    expected["success"],
                    case["sample_id"],
                )
                if expected["success"]:
                    hypothesis = result["final_hypothesis"]
                    self.assertEqual(
                        hypothesis["basin_ids"],
                        expected["basin_ids"],
                        case["sample_id"],
                    )
                    self.assertEqual(
                        hypothesis["separator_sequence"],
                        expected["separator_sequence"],
                        case["sample_id"],
                    )
                    expected_geometry = expected.get("geometry")
                    if expected_geometry is not None:
                        geometry = hypothesis["geometry"]
                        self.assertAlmostEqual(
                            geometry["left_distance_px"],
                            expected_geometry["left_distance_px"],
                            msg=case["sample_id"],
                        )
                        self.assertAlmostEqual(
                            geometry["right_distance_px"],
                            expected_geometry["right_distance_px"],
                            msg=case["sample_id"],
                        )
                        for side, center in expected_geometry[
                            "basin_centers_roi"
                        ].items():
                            self.assertAlmostEqual(
                                geometry["basins"][side][
                                    "center_x_at_reference_roi"
                                ],
                                center,
                                msg=f"{case['sample_id']}:{side}",
                            )
                    expected_reported = expected.get(
                        "reported_geometry"
                    )
                    if expected_reported is not None:
                        geometry = hypothesis["geometry"]
                        basins = geometry["basins"]
                        self.assertEqual(
                            "left" in basins,
                            expected_reported["left_available"],
                            case["sample_id"],
                        )
                        self.assertEqual(
                            "right" in basins,
                            expected_reported["right_available"],
                            case["sample_id"],
                        )
                        reference_x_roi = (
                            case["reference_global"]["x"]
                            - bounds["x0"]
                        )
                        left_center = basins["left"][
                            "center_x_at_reference_roi"
                        ]
                        right_center = basins["right"][
                            "center_x_at_reference_roi"
                        ]
                        self.assertAlmostEqual(
                            reference_x_roi - left_center,
                            expected_reported[
                                "left_distance_to_click_px"
                            ],
                            msg=case["sample_id"],
                        )
                        self.assertAlmostEqual(
                            right_center - reference_x_roi,
                            expected_reported[
                                "right_distance_to_click_px"
                            ],
                            msg=case["sample_id"],
                        )
                        self.assertAlmostEqual(
                            right_center - left_center,
                            expected_reported["stripe_spacing_px"],
                            msg=case["sample_id"],
                        )
                else:
                    self.assertEqual(
                        result["unavailable_reason"],
                        expected["unavailable_reason"],
                        case["sample_id"],
                    )

    def test_intermediate_dark_veto_downgrades_without_replacement(self):
        hypothesis = {
            "hypothesis_id": "JH01",
            "reference_relation": "inside_basin",
        }
        relation = {
            "status": "unique",
            "unavailable_reason": None,
            "hypotheses": [hypothesis],
        }
        separator_result = {
            "reference_x_roi": 50.0,
            "candidates": [],
        }
        pitch_result = {
            "algorithm_revision": "pitch-v1",
            "configuration_checksum": "pitch-config",
        }
        evidence = {
            "applied": True,
            "reference_relation": "inside_basin",
            "sides": [
                {
                    "side": "right",
                    "verified": True,
                    "dark_basin_x_global": 123.0,
                    "band_support": 1.0,
                    "row_support": 0.99,
                    "median_normalized_contrast": 0.4,
                }
            ],
        }
        frozen_separator_result = {"source": "already_computed"}
        with (
            patch.object(
                prototype,
                "FROZEN_RUN_JOINT_CASE",
                return_value={
                    "success": True,
                    "debug": {
                        "separator_result": frozen_separator_result,
                        "raw_pitch_result": pitch_result,
                    },
                },
            ),
            patch.object(
                prototype,
                "detect_centerized_separator_paths_v1_1",
                return_value=separator_result,
            ) as detect_centerized,
            patch.object(
                prototype.joint,
                "build_ordered_basin_graph",
                return_value={"verified_basins": []},
            ),
            patch.object(
                prototype.joint,
                "resolve_reference_relation",
                return_value=relation,
            ),
            patch.object(
                prototype.raw_pitch,
                "estimate_raw_local_pitch_v3",
                return_value=pitch_result,
            ) as estimate_pitch,
            patch.object(
                prototype.joint,
                "_basin_geometry",
                return_value={"basins": {}},
            ),
            patch.object(
                prototype.joint,
                "_pitch_evidence_for_geometry",
                return_value={"rejects_geometry": False},
            ),
            patch.object(
                prototype,
                "_intermediate_dark_basin_evidence",
                return_value=evidence,
            ),
        ):
            result = prototype.run_joint_case_v1_1(
                np.zeros((20, 20), dtype=np.uint8),
                {"x": 10, "y": 10},
                {"x0": 0, "y0": 0, "x1": 20, "y1": 20},
            )

        self.assertFalse(result["success"])
        self.assertIs(
            detect_centerized.call_args.kwargs[
                "frozen_separator_result"
            ],
            frozen_separator_result,
        )
        self.assertIs(
            detect_centerized.call_args.kwargs["frozen_pitch_result"],
            pitch_result,
        )
        estimate_pitch.assert_not_called()
        self.assertEqual(result["debug"]["raw_pitch_result"], pitch_result)
        self.assertIsNot(result["debug"]["raw_pitch_result"], pitch_result)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(
            result["unavailable_reason"],
            "intermediate_dark_basin_present",
        )
        self.assertIsNone(result["final_hypothesis"])
        veto = result["debug"]["intermediate_dark_basin_veto"]
        self.assertEqual(veto["triggered_sides"], evidence["sides"])

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

    def test_release_unavailable_is_audit_only_after_all_checks(self):
        separator_result = {
            "reference_x_roi": 50.0,
            "reference_y_roi": 10.0,
            "candidates": [],
        }
        relation_hypothesis = {
            "hypothesis_id": "JH01",
            "reference_relation": "inside_basin",
        }
        relation = {
            "status": "unique",
            "unavailable_reason": None,
            "hypotheses": [relation_hypothesis],
        }
        pitch_result = {
            "algorithm_revision": "pitch-v1",
            "configuration_checksum": "pitch-config",
        }
        pitch_evidence = {"rejects_geometry": False}
        with (
            patch.object(
                prototype,
                "FROZEN_RUN_JOINT_CASE",
                return_value={
                    "success": False,
                    "unavailable_reason": "legacy_failure",
                },
            ),
            patch.object(
                prototype,
                "detect_centerized_separator_paths_v1_1",
                return_value=separator_result,
            ),
            patch.object(
                prototype.joint,
                "build_ordered_basin_graph",
                return_value={"verified_basins": []},
            ),
            patch.object(
                prototype.joint,
                "resolve_reference_relation",
                return_value=relation,
            ),
            patch.object(
                prototype.raw_pitch,
                "estimate_raw_local_pitch_v3",
                return_value=pitch_result,
            ),
            patch.object(
                prototype.joint,
                "_basin_geometry",
                return_value={"basins": {}},
            ),
            patch.object(
                prototype.joint,
                "_pitch_evidence_for_geometry",
                return_value=pitch_evidence,
            ),
            patch.object(
                prototype.joint,
                "validate_output_basin_center_darkness",
                return_value={
                    "success": True,
                    "reason": "output_basin_centers_dark",
                    "basin_audits": [],
                },
            ),
        ):
            result = prototype.run_joint_case_v1_1(
                np.zeros((20, 20), dtype=np.uint8),
                {"x": 10, "y": 10},
                {"x0": 0, "y0": 0, "x1": 20, "y1": 20},
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "available")
        self.assertIsNone(result["unavailable_reason"])
        audit = result["debug"]["release_unavailable_audit"]
        self.assertFalse(audit["hard_veto_applied"])
        self.assertEqual(
            audit["release_unavailable_reason"],
            "legacy_failure",
        )
        self.assertEqual(
            audit["accepted_hypothesis"],
            result["final_hypothesis"],
        )


if __name__ == "__main__":
    unittest.main()
