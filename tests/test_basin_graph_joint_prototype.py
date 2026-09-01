import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from tools import basin_graph_joint_prototype as joint
from tools import separator_path_prototype as separator


def synthetic_stripes(
    pitch: int = 40,
    height: int = 240,
    width: int = 500,
) -> np.ndarray:
    image = np.full((height, width), 22, dtype=np.float64)
    rng = np.random.default_rng(23)
    image += rng.normal(0, 2, image.shape)
    for base_x in range(10, width, pitch):
        for y in range(height):
            drift = int(round(2.0 * np.sin(y / 80.0)))
            x = base_x + drift
            image[y, max(0, x - 2) : min(width, x + 3)] = 225
    return np.clip(image, 0, 255).astype(np.uint8)


class BasinGraphJointPrototypeTests(unittest.TestCase):
    # Stripe 01 and Stripe 09 are retained as historical development data, but
    # neither image is in the current product/release scope.  Keep the sample
    # identities explicit so they cannot silently become release-gating cases.
    NON_RELEASE_GATING_DEVELOPMENT_CASES = {
        "D014": "Stripe_01_date20250601_t113239765.bmp",
        "D019": "Stripe_09_e0_t200229235_v3p02565_do.bmp",
    }

    @staticmethod
    def run_development_sample(sample_id: str) -> dict:
        annotation = separator.load_development_document()[
            "annotations"
        ][sample_id]
        image = separator.load_grayscale_image(
            separator.IMAGES_DIR / annotation["image_name"]
        )
        return joint.run_joint_case(
            image,
            annotation["reference_global"],
            annotation["roi_bounds_global"],
            annotation["direction"],
        )

    def test_frozen_dependency_checksums(self):
        checks = joint.verify_frozen_dependencies()
        self.assertEqual(
            joint.FROZEN_SEPARATOR_CONFIGURATION_CHECKSUM,
            checks["separator_configuration_checksum"],
        )
        self.assertEqual(
            joint.FROZEN_RAW_PITCH_CONFIGURATION_CHECKSUM,
            checks["raw_pitch_configuration_checksum"],
        )

    def test_basin_center_darkness_rejects_internal_bright_ridge(self):
        left = {"x_by_band_roi": [2.0, 2.0, 2.0]}
        right = {"x_by_band_roi": [8.0, 8.0, 8.0]}
        dark_center = np.full((3, 12), 20.0)
        bright_center = dark_center.copy()
        for profiles in (dark_center, bright_center):
            profiles[:, 2] = 200.0
            profiles[:, 8] = 200.0
        bright_center[:, 4:7] = 120.0

        dark = joint._basin_center_darkness_evidence(  # noqa: SLF001
            left,
            right,
            {"profiles": dark_center},
        )
        bright = joint._basin_center_darkness_evidence(  # noqa: SLF001
            left,
            right,
            {"profiles": bright_center},
        )

        self.assertTrue(
            all(
                value
                <= joint.DEFAULT_CONFIG.local_reference_maximum_center_brightness_excess
                for value in dark[
                    "normalized_center_brightness_excess_by_band"
                ]
            )
        )
        self.assertTrue(
            all(
                value
                > joint.DEFAULT_CONFIG.local_reference_maximum_center_brightness_excess
                for value in bright[
                    "normalized_center_brightness_excess_by_band"
                ]
            )
        )

    def test_basin_center_darkness_reuses_only_exact_percentile_statistics(self):
        left = {
            "candidate_id": "C01",
            "x_by_band_roi": [2.0, 2.0, 2.0],
        }
        right = {
            "candidate_id": "C02",
            "x_by_band_roi": [8.0, 8.0, 8.0],
        }
        profiles = np.full((3, 12), 20.0)
        profiles[:, 2] = 200.0
        profiles[:, 8] = 200.0
        evidence = {"profiles": profiles}
        _dark, statistics = separator._dark_basin_evidence(  # noqa: SLF001
            left,
            right,
            evidence,
            separator.DEFAULT_CONFIG,
            capture_percentile_statistics=True,
        )
        expected = joint._basin_center_darkness_evidence(  # noqa: SLF001
            left,
            right,
            evidence,
        )

        original_percentile = np.percentile
        with mock.patch.object(
            joint.np,
            "percentile",
            wraps=original_percentile,
        ) as percentile:
            actual = joint._basin_center_darkness_evidence(  # noqa: SLF001
                left,
                right,
                evidence,
                statistics,
            )
        self.assertEqual(expected, actual)
        self.assertEqual(0, percentile.call_count)
        self.assertIsInstance(statistics.bands, tuple)
        with self.assertRaises(AttributeError):
            statistics.left_candidate_id = "changed"

        copied_evidence = {"profiles": profiles.copy()}
        with mock.patch.object(
            joint.np,
            "percentile",
            wraps=original_percentile,
        ) as percentile:
            fallback = joint._basin_center_darkness_evidence(  # noqa: SLF001
                left,
                right,
                copied_evidence,
                statistics,
            )
        self.assertEqual(expected, fallback)
        self.assertEqual(9, percentile.call_count)

    def test_output_center_darkness_checks_only_reported_basins(self):
        def basin(basin_id: str, values: list[float]) -> dict:
            return {
                "basin_id": basin_id,
                "band_centers_y_roi": [10.0, 20.0, 30.0],
                "center_darkness_evidence": {
                    "normalized_center_brightness_excess_by_band": values,
                },
            }

        graph = {
            "basin_candidates": [
                basin("left", [0.01, 0.02, 0.01]),
                basin("clicked", [0.40, 0.40, 0.40]),
                basin("right", [0.02, 0.08, 0.09]),
            ]
        }
        hypothesis = {
            "basin_ids": {
                "left": "left",
                "clicked": "clicked",
                "right": "right",
            }
        }

        result = joint.validate_output_basin_center_darkness(
            hypothesis,
            graph,
            20.0,
        )

        self.assertFalse(result["success"])
        self.assertEqual("output_basin_center_not_dark", result["reason"])
        self.assertEqual(
            ["left", "right"],
            [audit["role"] for audit in result["basin_audits"]],
        )

    def test_success_is_one_atomic_continuous_hypothesis(self):
        image = synthetic_stripes()
        result = joint.run_joint_case(
            image,
            {"x": 270, "y": 120},
            {"x0": 0, "y0": 0, "x1": 500, "y1": 240},
        )
        self.assertTrue(result["success"])
        hypothesis = result["final_hypothesis"]
        self.assertTrue(hypothesis["atomic"])
        self.assertEqual("inside_basin", hypothesis["reference_relation"])
        self.assertEqual(4, len(hypothesis["separator_sequence"]))
        self.assertEqual(3, len(hypothesis["basin_sequence"]))
        ordered = result["debug"]["basin_graph"][
            "ordered_separator_ids"
        ]
        indexes = [
            ordered.index(item) for item in hypothesis["separator_sequence"]
        ]
        self.assertEqual(
            list(range(indexes[0], indexes[0] + 4)),
            indexes,
        )
        self.assertEqual(
            joint.FROZEN_RAW_PITCH_CONFIGURATION_CHECKSUM,
            hypothesis["pitch_evidence"]["configuration_checksum"],
        )
        graph = result["debug"]["basin_graph"]
        basin_by_id = {
            basin["basin_id"]: basin
            for basin in graph["verified_basins"]
        }
        for index, role in enumerate(("left", "clicked", "right")):
            basin = basin_by_id[hypothesis["basin_ids"][role]]
            self.assertEqual(
                hypothesis["separator_sequence"][index],
                basin["left_separator_id"],
            )
            self.assertEqual(
                hypothesis["separator_sequence"][index + 1],
                basin["right_separator_id"],
            )
            self.assertEqual(
                hypothesis["basin_ids"][role],
                hypothesis["geometry"]["basins"][role]["basin_id"],
            )
        separator_x = hypothesis["geometry"][
            "separator_x_at_reference_roi"
        ]
        reference_x = result["debug"]["separator_result"][
            "reference_x_roi"
        ]
        self.assertAlmostEqual(
            reference_x - separator_x["left_clicked_boundary"],
            hypothesis["geometry"]["left_distance_px"],
        )
        self.assertAlmostEqual(
            separator_x["right_clicked_boundary"] - reference_x,
            hypothesis["geometry"]["right_distance_px"],
        )

    def test_high_pitch_conflict_is_hard_failure(self):
        image = synthetic_stripes(pitch=40)
        original = joint.raw_pitch.estimate_raw_local_pitch_v3

        def conflicting_pitch(*args, **kwargs):
            result = original(*args, **kwargs)
            return {
                **result,
                "confidence": "high",
                "success_eligible": True,
                "usable_pitch_px": 10.0,
                "diagnostic_pitch_px": 10.0,
                "harmonic_ambiguity": {
                    "detected": False,
                    "reason": None,
                },
                "unavailable_reason": None,
            }

        joint.raw_pitch.estimate_raw_local_pitch_v3 = conflicting_pitch
        try:
            result = joint.run_joint_case(
                image,
                {"x": 270, "y": 120},
                {"x0": 0, "y0": 0, "x1": 500, "y1": 240},
            )
        finally:
            joint.raw_pitch.estimate_raw_local_pitch_v3 = original
        self.assertFalse(result["success"])
        self.assertEqual(
            "high_pitch_geometry_conflict",
            result["unavailable_reason"],
        )
        self.assertIsNone(result["final_hypothesis"])

    def test_unavailable_pitch_neither_creates_nor_rejects_geometry(self):
        image = synthetic_stripes()
        original = joint.raw_pitch.estimate_raw_local_pitch_v3

        def unavailable_pitch(*args, **kwargs):
            result = original(*args, **kwargs)
            return {
                **result,
                "pitch_px": None,
                "diagnostic_pitch_px": None,
                "usable_pitch_px": None,
                "confidence": "unavailable",
                "success_eligible": False,
                "harmonic_ambiguity": {
                    "detected": False,
                    "reason": None,
                },
                "unavailable_reason": "no_cross_band_hypothesis",
            }

        joint.raw_pitch.estimate_raw_local_pitch_v3 = unavailable_pitch
        try:
            result = joint.run_joint_case(
                image,
                {"x": 270, "y": 120},
                {"x0": 0, "y0": 0, "x1": 500, "y1": 240},
            )
        finally:
            joint.raw_pitch.estimate_raw_local_pitch_v3 = original
        self.assertTrue(result["success"])
        self.assertEqual(
            "unavailable_no_effect",
            result["final_hypothesis"]["pitch_evidence"][
                "joint_decision"
            ],
        )

    def test_medium_pitch_conflict_remains_diagnostics_only(self):
        image = synthetic_stripes()
        original = joint.raw_pitch.estimate_raw_local_pitch_v3

        def medium_pitch(*args, **kwargs):
            result = original(*args, **kwargs)
            return {
                **result,
                "confidence": "medium",
                "success_eligible": False,
                "usable_pitch_px": None,
                "diagnostic_pitch_px": 10.0,
                "harmonic_ambiguity": {
                    "detected": False,
                    "reason": None,
                },
                "unavailable_reason": None,
            }

        joint.raw_pitch.estimate_raw_local_pitch_v3 = medium_pitch
        try:
            result = joint.run_joint_case(
                image,
                {"x": 270, "y": 120},
                {"x0": 0, "y0": 0, "x1": 500, "y1": 240},
            )
        finally:
            joint.raw_pitch.estimate_raw_local_pitch_v3 = original
        self.assertTrue(result["success"])
        evidence = result["final_hypothesis"]["pitch_evidence"]
        self.assertEqual("diagnostics_only", evidence["joint_decision"])
        self.assertFalse(evidence["rejects_geometry"])
        self.assertIsNone(evidence["usable_pitch_px"])

    def test_unique_reference_separator_is_shared_atomic_boundary(self):
        for sample_id in ("D007", "D008"):
            with self.subTest(sample_id=sample_id):
                result = self.run_development_sample(sample_id)
                self.assertTrue(result["success"])
                hypothesis = result["final_hypothesis"]
                self.assertEqual(
                    "on_separator",
                    hypothesis["reference_relation"],
                )
                self.assertTrue(hypothesis["atomic"])
                self.assertEqual(
                    hypothesis["clicked_separator_id"],
                    hypothesis["shared_boundary"]["separator_id"],
                )
                self.assertEqual(
                    hypothesis["basin_ids"]["left"],
                    hypothesis["shared_boundary"]["left_basin_id"],
                )
                self.assertEqual(
                    hypothesis["basin_ids"]["right"],
                    hypothesis["shared_boundary"]["right_basin_id"],
                )
                self.assertEqual(3, len(hypothesis["separator_sequence"]))
                self.assertEqual(2, len(hypothesis["basin_sequence"]))
                geometry = hypothesis["geometry"]
                self.assertLess(
                    geometry["basins"]["left"][
                        "center_x_at_reference_roi"
                    ],
                    geometry["reference_separator_x_roi"],
                )
                self.assertGreater(
                    geometry["basins"]["right"][
                        "center_x_at_reference_roi"
                    ],
                    geometry["reference_separator_x_roi"],
                )
                self.assertGreater(
                    geometry[
                        "reference_to_left_basin_center_distance_px"
                    ],
                    0,
                )
                self.assertGreater(
                    geometry[
                        "reference_to_right_basin_center_distance_px"
                    ],
                    0,
                )
                self.assertGreater(
                    geometry[
                        "shared_boundary_to_left_basin_center_distance_px"
                    ],
                    0,
                )
                self.assertGreater(
                    geometry[
                        "shared_boundary_to_right_basin_center_distance_px"
                    ],
                    0,
                )

    def test_reference_separator_does_not_bypass_basin_conflicts(self):
        expected_reasons = {
            "D012": "separator_adjacent_dark_basins_not_verified",
        }
        for sample_id, expected_reason in expected_reasons.items():
            with self.subTest(sample_id=sample_id):
                result = self.run_development_sample(sample_id)
                self.assertFalse(result["success"])
                self.assertIsNone(result["final_hypothesis"])
                self.assertEqual(
                    expected_reason,
                    result["unavailable_reason"],
                )

    def test_non_release_gating_legacy_cases_remain_explicit(self):
        annotations = separator.load_development_document()["annotations"]
        self.assertEqual(
            self.NON_RELEASE_GATING_DEVELOPMENT_CASES,
            {
                sample_id: annotations[sample_id]["image_name"]
                for sample_id in self.NON_RELEASE_GATING_DEVELOPMENT_CASES
            },
        )

    def test_distinct_verified_reference_interpretation_is_ambiguous(self):
        result = self.run_development_sample("D007")
        separator_result = copy.deepcopy(
            result["debug"]["separator_result"]
        )
        separator_result["arbitration_debug"][
            "competing_explanations"
        ].append(
            {
                "type": "clicked_basin_roles",
                "hypothesis_id": "H99",
                "verified": True,
                "selection_candidate_ids": {
                    "left_clicked_boundary": "C01",
                    "right_clicked_boundary": "C02",
                },
            }
        )
        with mock.patch.object(
            joint,
            "_local_reference_adjacency_validation",
            return_value={
                "success": False,
                "reason": "local_evidence_not_verified",
                "basins": [],
            },
        ):
            relation = joint._separator_adjacency_relation(
                separator_result,
                result["debug"]["basin_graph"],
            )
        self.assertEqual("unavailable", relation["status"])
        self.assertEqual(
            "multiple_reasonable_reference_hypotheses",
            relation["unavailable_reason"],
        )

    def test_local_adjacency_runs_before_reference_competition_return(self):
        result = self.run_development_sample("D007")
        separator_result = copy.deepcopy(
            result["debug"]["separator_result"]
        )
        separator_result["arbitration_debug"][
            "competing_explanations"
        ].append(
            {
                "type": "clicked_basin_roles",
                "hypothesis_id": "H99",
                "verified": True,
                "selection_candidate_ids": {
                    "left_clicked_boundary": "C01",
                    "right_clicked_boundary": "C02",
                },
            }
        )
        graph = copy.deepcopy(result["debug"]["basin_graph"])
        relation = joint._separator_adjacency_relation(
            separator_result,
            graph,
            result["debug"].get("raw_pitch_result"),
        )
        self.assertEqual("unique", relation["status"])
        self.assertEqual("on_separator", relation["relation"])
        local = relation["hypotheses"][0][
            "local_reference_adjacency_validation"
        ]
        self.assertTrue(local["success"])
        self.assertEqual(
            "click_local_reference_adjacency_verified",
            local["reason"],
        )

    def test_multiple_reference_separator_matches_are_unavailable(self):
        result = self.run_development_sample("D007")
        separator_result = copy.deepcopy(
            result["debug"]["separator_result"]
        )
        separator_result["arbitration_debug"][
            "competing_explanations"
        ].append(
            {
                "type": "reference_on_separator",
                "candidate_id": "C05",
                "distance_px": 2.0,
            }
        )
        relation = joint._separator_adjacency_relation(
            separator_result,
            result["debug"]["basin_graph"],
        )
        self.assertEqual("unavailable", relation["status"])
        self.assertEqual(
            "reference_relationship_not_unique",
            relation["unavailable_reason"],
        )

    def test_high_pitch_can_reject_reference_separator_geometry(self):
        annotation = separator.load_development_document()[
            "annotations"
        ]["D017"]
        image = separator.load_grayscale_image(
            separator.IMAGES_DIR / annotation["image_name"]
        )
        original = joint.raw_pitch.estimate_raw_local_pitch_v3

        def conflicting_pitch(*args, **kwargs):
            result = original(*args, **kwargs)
            return {
                **result,
                "confidence": "high",
                "success_eligible": True,
                "usable_pitch_px": 10.0,
                "diagnostic_pitch_px": 10.0,
                "harmonic_ambiguity": {
                    "detected": False,
                    "reason": None,
                },
                "unavailable_reason": None,
            }

        joint.raw_pitch.estimate_raw_local_pitch_v3 = conflicting_pitch
        try:
            result = joint.run_joint_case(
                image,
                annotation["reference_global"],
                annotation["roi_bounds_global"],
                annotation["direction"],
            )
        finally:
            joint.raw_pitch.estimate_raw_local_pitch_v3 = original
        self.assertFalse(result["success"])
        self.assertEqual(
            "high_pitch_geometry_conflict",
            result["unavailable_reason"],
        )

    def test_development_safety_contract(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = joint.evaluate_development(
                output_dir=Path(temporary_directory)
            )
        metrics = report["metrics"]
        wrong_success_ids = {
            sample["sample_id"]
            for sample in report["samples"]
            if sample["classification"] == "wrong_success"
        }
        self.assertEqual(
            set(self.NON_RELEASE_GATING_DEVELOPMENT_CASES),
            wrong_success_ids,
        )
        self.assertEqual(
            [],
            [
                sample_id
                for sample_id in wrong_success_ids
                if sample_id
                not in self.NON_RELEASE_GATING_DEVELOPMENT_CASES
            ],
        )
        # false_split is the historical alias of wrong_success in this
        # evaluator, so it contains the same two out-of-scope samples.
        self.assertEqual(
            len(self.NON_RELEASE_GATING_DEVELOPMENT_CASES),
            metrics["false_split"],
        )
        self.assertEqual(
            sorted(self.NON_RELEASE_GATING_DEVELOPMENT_CASES),
            metrics["ambiguous_formally_available"],
        )
        self.assertEqual([], metrics["unavailable_formally_available"])
        self.assertEqual(
            [],
            metrics["crossing_or_order_conflict_formally_available"],
        )
        self.assertEqual(66, metrics["candidate_path_recall_count"])
        self.assertEqual(
            separator.FROZEN_DEVELOPMENT_CANDIDATE_OUTPUT_CHECKSUM,
            metrics["separator_candidate_output_checksum"],
        )
        self.assertTrue(
            metrics["separator_candidate_generation_unchanged"]
        )
        self.assertEqual(17, metrics["correct_success"])
        self.assertEqual(
            9 - len(self.NON_RELEASE_GATING_DEVELOPMENT_CASES),
            metrics["safe_failure"],
        )
        self.assertEqual(
            ["D007", "D008", "D017", "D022"],
            metrics["reference_on_separator_correct_success"],
        )
        self.assertTrue(metrics["stage3_inside_geometry_unchanged"])
        self.assertEqual(
            ["D007", "D008", "D017", "D022"],
            [
                item["sample_id"]
                for item in metrics[
                    "stage3_to_stage3_1_changed_samples"
                ]
                if item["sample_id"]
                not in self.NON_RELEASE_GATING_DEVELOPMENT_CASES
            ],
        )

    def test_source_has_no_tracks_or_production_detector(self):
        source = Path(joint.__file__).read_text(encoding="utf-8")
        forbidden = (
            "interactive_pipeline",
            "interactive_analysis",
            "desktop_app",
            "track_id",
            "heldout.json",
            "run_interactive_case",
        )
        for fragment in forbidden:
            self.assertNotIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
