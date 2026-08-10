import importlib.util
import json
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools/run_staged_targeted_regression.py"
OUTPUT_DIR = (
    REPO_ROOT
    / "outputs/batch_anomaly_scan_20260807_full/manual_review/staged_implementation"
)

SPEC = importlib.util.spec_from_file_location("run_staged_targeted_regression", MODULE_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class StagedTargetedRegressionTests(unittest.TestCase):
    def load(self, name):
        return json.loads((OUTPUT_DIR / name).read_text(encoding="utf-8"))

    def test_snapshots_contain_only_the_three_targeted_cohorts(self):
        final = self.load("stage3_pitch_guided_missing_separator.json")
        self.assertEqual(final["case_count"], 207)
        self.assertEqual(final["wrong_detection_loaded"], 0)
        self.assertFalse(final["full_sweep_rerun"])
        labels = {case["human_label"] for case in final["cases"]}
        self.assertEqual(labels, RUNNER.IN_SCOPE_LABELS)

    def test_incremental_stage_results_are_regression_free(self):
        expected = {
            "stage1_incremental.json": (10, 10),
            "stage2_incremental.json": (5, 15),
            "stage3_incremental.json": (3, 18),
        }
        for filename, (incremental, cumulative) in expected.items():
            with self.subTest(filename=filename):
                comparison = self.load(filename)
                self.assertEqual(
                    comparison["incremental_should_detect_recovered"], incremental
                )
                self.assertEqual(
                    comparison["cumulative_should_detect_recovered"], cumulative
                )
                self.assertEqual(comparison["valid_unavailable_regression_count"], 0)
                self.assertEqual(comparison["correct_detection_regression_count"], 0)

    def test_final_correct_detection_identity_and_geometry_match_baseline(self):
        baseline = self.load("stage0_baseline.json")
        final = self.load("stage3_pitch_guided_missing_separator.json")
        comparison = RUNNER.compare_snapshots(baseline, final)
        self.assertEqual(comparison["valid_unavailable_regression_count"], 0)
        self.assertEqual(comparison["correct_detection_regression_count"], 0)
        self.assertEqual(comparison["correct_detection_geometry_change_count"], 0)
        self.assertEqual(comparison["cumulative_should_detect_recovered"], 18)
        self.assertEqual(comparison["cumulative_non_stripe01_09_recovered"], 18)
        self.assertEqual(comparison["cumulative_stripe01_09_recovered"], 0)


if __name__ == "__main__":
    unittest.main()
