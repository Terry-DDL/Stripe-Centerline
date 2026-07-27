"""Tests for reviewed topology shadow trial metrics."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from evaluate_topology_shadow_trial import evaluate_trial  # noqa: E402


def candidate(status: str, strong: bool) -> dict:
    return {
        "grayscale_topology": {
            "status": status,
            "strong_same_basin_conflict": strong,
        }
    }


def trial_case(
    case_id: str,
    current: str,
    hypothetical: str,
    would_change: bool,
) -> dict:
    return {
        "id": case_id,
        "original_candidates": {
            "otsu": candidate("Consistent", False),
            "adaptive": candidate("Contradictory", True),
        },
        "shadow_arbitration": {
            "current_winner": {
                "geometry": "original",
                "threshold_method": current,
            },
            "hypothetical_winner": {
                "geometry": "original",
                "threshold_method": hypothetical,
            },
            "hypothetical_success": True,
            "would_change_formal_result": would_change,
        },
    }


class TopologyShadowTrialTests(unittest.TestCase):
    def test_reports_safe_detection_but_unsafe_fallback_separately(self):
        manifest = {
            "known_error_cases": [
                trial_case("known", "adaptive", "otsu", True)
            ],
            "pending_review_cases": [
                trial_case("pending", "adaptive", "otsu", True)
            ],
        }
        decisions = {
            "decisions": {
                "known": {"choice": "otsu"},
                "pending": {"choice": "neither"},
            }
        }

        report = evaluate_trial(manifest, decisions)

        self.assertTrue(report["shadow_detector_gate_passed"])
        self.assertFalse(report["rejection_activation_gate_passed"])
        self.assertEqual(
            report["strong_conflict_on_correct_candidate_count"],
            0,
        )
        self.assertEqual(
            report["counterfactual_outcome_counts"],
            {"fixes_error": 1, "remains_error": 1},
        )

    def test_rejects_missing_review_decisions(self):
        manifest = {
            "known_error_cases": [
                trial_case("known", "adaptive", "otsu", True)
            ],
            "pending_review_cases": [],
        }

        with self.assertRaises(ValueError):
            evaluate_trial(manifest, {"decisions": {}})


if __name__ == "__main__":
    unittest.main()
