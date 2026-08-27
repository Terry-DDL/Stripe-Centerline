"""Compare fixed-point Windows EXE and Python benchmark result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUMMARY_JSON_FILENAME = "ab_summary.json"
SUMMARY_TEXT_FILENAME = "ab_summary.txt"
IDENTITY_FIELDS = (
    "benchmark_revision",
    "detector_result_source",
    "profile_stages",
)
ENVIRONMENT_FIELDS = (
    "python_version",
    "python_implementation",
    "numpy_version",
    "opencv_version",
    "opencv_num_threads",
    "opencv_use_optimized",
    "platform",
    "machine",
    "processor_identifier",
    "processor_architecture",
    "cpu_count",
    "opencv_build_summary",
    "benchmark_build",
)


def _read_result(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("status") == "error":
        raise ValueError(f"benchmark failed: {path}: {payload.get('error')}")
    return payload


def compare_results(exe: dict, source: dict) -> dict:
    identity_checks = {
        field: exe.get(field) == source.get(field)
        for field in IDENTITY_FIELDS
    }
    fixed_case_fields = (
        "image",
        "reference_point",
        "roi",
        "run_count",
        "same_process",
        "image_loaded_once",
    )
    identity_checks["fixed_case"] = all(
        exe["fixed_case"].get(field) == source["fixed_case"].get(field)
        for field in fixed_case_fields
    )
    identity_checks["benchmark_build"] = (
        exe["system"].get("benchmark_build") is not None
        and exe["system"].get("benchmark_build")
        == source["system"].get("benchmark_build")
    )
    identity_checks["frozen_modes"] = (
        exe["system"].get("frozen") is True
        and source["system"].get("frozen") is False
    )
    identity_checks["detection_state"] = (
        exe["runs"][0]["detection_state_sha256"]
        == source["runs"][0]["detection_state_sha256"]
        and exe.get("results_consistent") is True
        and source.get("results_consistent") is True
    )

    environment = {}
    for field in ENVIRONMENT_FIELDS:
        exe_value = exe["system"].get(field)
        source_value = source["system"].get(field)
        environment[field] = {
            "exe": exe_value,
            "python_source": source_value,
            "match": exe_value == source_value,
        }
    environment["opencv_build_information"] = {
        "exe": "recorded in frozen_exe/benchmark_results.json",
        "python_source": "recorded in python_source/benchmark_results.json",
        "match": (
            exe["system"].get("opencv_build_information")
            == source["system"].get("opencv_build_information")
        ),
    }
    environment_match = all(
        values["match"] for values in environment.values()
    )

    timings = {}
    for stage in exe["profile_stages"]:
        exe_ms = float(exe["warm_summary"][stage]["median_ms"])
        source_ms = float(source["warm_summary"][stage]["median_ms"])
        timings[stage] = {
            "exe_median_ms": round(exe_ms, 3),
            "python_source_median_ms": round(source_ms, 3),
            "exe_over_python_ratio": (
                round(exe_ms / source_ms, 3) if source_ms else None
            ),
        }

    return {
        "comparable": all(identity_checks.values()) and environment_match,
        "identity_checks": identity_checks,
        "environment_match": environment_match,
        "environment": environment,
        "warm_median_timings_ms": timings,
    }


def build_summary_text(comparison: dict) -> str:
    lines = [
        "Stripe Centerline Windows EXE vs Python A/B",
        "",
        f"Directly comparable: {comparison['comparable']}",
        f"Runtime environment match: {comparison['environment_match']}",
        "",
        "Identity checks:",
    ]
    for field, passed in comparison["identity_checks"].items():
        lines.append(f"  {field}: {'PASS' if passed else 'FAIL'}")

    lines.extend(["", "Warm medians (runs 2-5):"])
    for stage, values in comparison["warm_median_timings_ms"].items():
        ratio = values["exe_over_python_ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.3f}x"
        lines.append(
            f"  {stage}: EXE {values['exe_median_ms']:.3f} ms | "
            f"Python {values['python_source_median_ms']:.3f} ms | "
            f"EXE/Python {ratio_text}"
        )

    lines.extend(["", "Environment:"])
    for field, values in comparison["environment"].items():
        status = "MATCH" if values["match"] else "DIFF"
        lines.append(
            f"  {field}: {status} | EXE={values['exe']} | "
            f"Python={values['python_source']}"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--python-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)

    comparison = compare_results(
        _read_result(arguments.exe),
        _read_result(arguments.python_source),
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / SUMMARY_JSON_FILENAME).write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (arguments.output_dir / SUMMARY_TEXT_FILENAME).write_text(
        build_summary_text(comparison),
        encoding="utf-8",
    )
    return 0 if comparison["comparable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
