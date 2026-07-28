"""Tests for explainable lexicographic candidate arbitration."""

from pathlib import Path
import sys
import unittest


SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from candidate_quality import candidate_quality_key  # noqa: E402


def quality(**overrides):
    values = {
        "success": True,
        "adjacency_verification_status": "verified",
        "combined_pitch_status": "Normal",
        "grayscale_topology_status": "Consistent",
        "minimum_valid_row_ratio": 0.8,
        "minimum_retention_ratio": 0.9,
        "maximum_center_mad_px": 1.0,
        "maximum_normalized_width_mad": 0.1,
        "click_association_distance_px": 2.0,
        "suspected_fragment_count": 0,
        "low_support_track_count": 0,
    }
    values.update(overrides)
    return values


class CandidateQualityTests(unittest.TestCase):
    def test_verified_adjacency_precedes_pitch_and_support(self):
        verified = quality(
            combined_pitch_status="Suspicious",
            minimum_valid_row_ratio=0.55,
        )
        rejected = quality(
            adjacency_verification_status="rejected",
            combined_pitch_status="Normal",
            minimum_valid_row_ratio=1.0,
        )

        self.assertGreater(
            candidate_quality_key(verified),
            candidate_quality_key(rejected),
        )

    def test_pitch_status_precedes_track_support(self):
        normal = quality(minimum_valid_row_ratio=0.55)
        suspicious = quality(
            combined_pitch_status="Suspicious",
            minimum_valid_row_ratio=1.0,
        )

        self.assertGreater(
            candidate_quality_key(normal),
            candidate_quality_key(suspicious),
        )

    def test_grayscale_topology_precedes_track_support(self):
        consistent = quality(minimum_valid_row_ratio=0.55)
        unverifiable = quality(
            grayscale_topology_status="Unable to verify",
            minimum_valid_row_ratio=1.0,
        )

        self.assertGreater(
            candidate_quality_key(consistent),
            candidate_quality_key(unverifiable),
        )
        self.assertLess(
            candidate_quality_key(
                consistent,
                include_topology=False,
            ),
            candidate_quality_key(
                unverifiable,
                include_topology=False,
            ),
        )

    def test_support_precedes_center_mad(self):
        supported = quality(
            minimum_valid_row_ratio=0.95,
            maximum_center_mad_px=3.0,
        )
        smooth_but_weak = quality(
            minimum_valid_row_ratio=0.55,
            maximum_center_mad_px=0.0,
        )

        self.assertGreater(
            candidate_quality_key(supported),
            candidate_quality_key(smooth_but_weak),
        )

    def test_otsu_wins_only_after_complete_quality_tie(self):
        tied = quality()

        self.assertGreater(
            candidate_quality_key(tied, "otsu"),
            candidate_quality_key(tied, "adaptive"),
        )


if __name__ == "__main__":
    unittest.main()
