"""Tests for the Windows EXE versus Python benchmark comparison."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools import compare_windows_benchmarks as comparison


def _payload(frozen: bool, analysis_ms: float) -> dict:
    system = {
        "frozen": frozen,
        "python_version": "3.11.9",
        "python_implementation": "CPython",
        "numpy_version": "2.4.6",
        "opencv_version": "5.0.0",
        "opencv_num_threads": 8,
        "opencv_use_optimized": True,
        "platform": "Windows-10",
        "machine": "AMD64",
        "processor_identifier": "test CPU",
        "processor_architecture": "AMD64",
        "cpu_count": 8,
        "opencv_build_summary": ["Parallel framework: Concurrency"],
        "opencv_build_information": "test OpenCV build",
        "benchmark_build": {"source_revision": "abc123"},
    }
    return {
        "benchmark_revision": "windows_fixed_point_benchmark_v1",
        "detector_result_source": "cross_image_correction_v1_1",
        "profile_stages": ["analysis_total"],
        "fixed_case": {
            "image": {"sha256": "fixed"},
            "reference_point": {"x": 775, "y": 427},
            "roi": {"x0": 525, "y0": 327, "x1": 1025, "y1": 527},
            "run_count": 5,
            "same_process": True,
            "image_loaded_once": True,
            "image_load_once_ms": 1.0 if frozen else 2.0,
        },
        "system": system,
        "runs": [{"detection_state_sha256": "same"}],
        "results_consistent": True,
        "warm_summary": {"analysis_total": {"median_ms": analysis_ms}},
    }


class CompareWindowsBenchmarksTests(unittest.TestCase):
    def test_matching_runs_are_comparable_and_ratio_is_reported(self):
        result = comparison.compare_results(
            _payload(True, 600.0),
            _payload(False, 300.0),
        )

        self.assertTrue(result["comparable"])
        self.assertEqual(
            2.0,
            result["warm_median_timings_ms"]["analysis_total"][
                "exe_over_python_ratio"
            ],
        )

    def test_detection_mismatch_blocks_direct_comparison(self):
        exe = _payload(True, 600.0)
        source = copy.deepcopy(_payload(False, 300.0))
        source["runs"][0]["detection_state_sha256"] = "different"

        result = comparison.compare_results(exe, source)

        self.assertFalse(result["comparable"])
        self.assertFalse(result["identity_checks"]["detection_state"])

    def test_runtime_mismatch_blocks_direct_comparison(self):
        exe = _payload(True, 600.0)
        source = copy.deepcopy(_payload(False, 300.0))
        source["system"]["opencv_num_threads"] = 1

        result = comparison.compare_results(exe, source)

        self.assertFalse(result["comparable"])
        self.assertFalse(result["environment_match"])

    def test_main_writes_json_and_text_summaries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            exe_path = root / "exe.json"
            source_path = root / "source.json"
            output_dir = root / "summary"
            exe_path.write_text(json.dumps(_payload(True, 600.0)))
            source_path.write_text(json.dumps(_payload(False, 300.0)))

            exit_code = comparison.main(
                [
                    "--exe",
                    str(exe_path),
                    "--python-source",
                    str(source_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(0, exit_code)
            self.assertTrue(
                (output_dir / comparison.SUMMARY_JSON_FILENAME).is_file()
            )
            self.assertTrue(
                (output_dir / comparison.SUMMARY_TEXT_FILENAME).is_file()
            )


if __name__ == "__main__":
    unittest.main()
