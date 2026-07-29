from pathlib import Path
import tempfile
import unittest

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

    def test_development_safety_contract(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = joint.evaluate_development(
                output_dir=Path(temporary_directory)
            )
        metrics = report["metrics"]
        self.assertEqual(0, metrics["wrong_success"])
        self.assertEqual(0, metrics["false_split"])
        self.assertEqual([], metrics["ambiguous_formally_available"])
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
