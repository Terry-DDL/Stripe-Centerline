"""Tests for the isolated fixed-point Windows benchmark runner."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from tools import windows_benchmark_runner as benchmark


class WindowsBenchmarkRunnerTests(unittest.TestCase):
    def test_fixed_asset_is_exact_mac_benchmark_image(self):
        image, info = benchmark.load_fixed_image(
            benchmark.bundled_image_path()
        )

        self.assertEqual(
            (benchmark.IMAGE_HEIGHT, benchmark.IMAGE_WIDTH),
            image.shape,
        )
        self.assertEqual(benchmark.IMAGE_SHA256, info["sha256"])

    def test_warm_summary_uses_runs_two_through_five(self):
        runs = []
        for run_number in range(1, 6):
            runs.append(
                {
                    "run_number": run_number,
                    "run_kind": "cold" if run_number == 1 else "warm",
                    "timings_ms": {
                        stage: float(run_number)
                        for stage in benchmark.PROFILE_STAGES
                    },
                }
            )

        summary = benchmark.summarize_warm_runs(runs)

        self.assertEqual(3.5, summary["analysis_total"]["median_ms"])
        self.assertEqual(2.0, summary["analysis_total"]["min_ms"])
        self.assertEqual(5.0, summary["analysis_total"]["max_ms"])

    def test_full_benchmark_is_saved_in_visible_results_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "benchmark_results"
            exit_code = benchmark.main(
                [
                    "--no-dialog",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(0, exit_code)
            self.assertTrue(
                (output_dir / benchmark.RESULTS_FILENAME).is_file()
            )
            self.assertTrue(
                (output_dir / benchmark.SUMMARY_FILENAME).is_file()
            )
            payload = json.loads(
                (output_dir / benchmark.RESULTS_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(5, len(payload["runs"]))
            self.assertEqual("cold", payload["runs"][0]["run_kind"])
            self.assertTrue(
                all(
                    run["run_kind"] == "warm"
                    for run in payload["runs"][1:]
                )
            )
            self.assertTrue(payload["results_consistent"])
            required_stages = set(benchmark.PROFILE_STAGES)
            for run in payload["runs"]:
                self.assertEqual(
                    required_stages,
                    set(run["timings_ms"]),
                )
                self.assertTrue(run["stage_profile"]["events"])

    def test_failure_writes_json_and_error_log(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "benchmark_results"
            with patch.object(
                benchmark,
                "run_benchmark",
                side_effect=RuntimeError("synthetic benchmark failure"),
            ):
                exit_code = benchmark.main(
                    [
                        "--no-dialog",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(1, exit_code)
            self.assertTrue(
                (output_dir / benchmark.ERROR_FILENAME).is_file()
            )
            payload = json.loads(
                (output_dir / benchmark.RESULTS_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("error", payload["status"])
            self.assertIn("synthetic benchmark failure", payload["error"])


if __name__ == "__main__":
    unittest.main()
