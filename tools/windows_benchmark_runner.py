"""One-click Windows benchmark for the fixed Stripe 10 desktop case.

This is a diagnostic runner only.  It calls the frozen desktop detector and
adapter without changing detector configuration, search logic, or results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import sys
import time
import traceback

import cv2
import numpy as np


PROJECT_ROOT = (
    Path(sys._MEIPASS)
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
    else Path(__file__).resolve().parent.parent
)
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from config import INTERACTIVE_CONFIG  # noqa: E402
from interactive_pipeline import calculate_interactive_roi_bounds  # noqa: E402
from tools.lightweight_profile import (  # noqa: E402
    ProfileSession,
    activate as activate_profile,
    stage as profile_stage,
)
from tools.stage3_desktop_runtime import (  # noqa: E402
    RESULT_SOURCE,
    build_desktop_result,
    persist_formal_result,
    run_frozen_stage3,
)


BENCHMARK_REVISION = "windows_fixed_point_benchmark_v1"
IMAGE_FILENAME = "Stripe_10_e0_t221236602_v8p56736_retry.bmp"
IMAGE_SHA256 = (
    "0c7588f5bf1f4571a0b43528e5713873ec33fda03b892d9771bdbf9120ea7646"
)
IMAGE_WIDTH = 2452
IMAGE_HEIGHT = 2056
REFERENCE_POINT = (775, 427)
EXPECTED_ROI = (525, 327, 1025, 527)
RUN_COUNT = 5
RESULTS_DIRECTORY_NAME = "benchmark_results"
RESULTS_FILENAME = "benchmark_results.json"
SUMMARY_FILENAME = "benchmark_summary.txt"
ERROR_FILENAME = "benchmark_error.log"
BUILD_INFO_FILENAME = "benchmark_build_info.json"

PROFILE_STAGES = (
    "image_roi_preparation",
    "preprocessing",
    "candidate_detection_and_path_tracking",
    "path_track_basin_search",
    "pitch_estimation",
    "geometry_validation",
    "analysis_total",
    "result_construction",
    "overlay_visualization",
    "result_persistence",
    "complete_total",
)

SUMMARY_LABELS = {
    "image_roi_preparation": "image/ROI preparation",
    "preprocessing": "preprocessing",
    "candidate_detection_and_path_tracking": "candidate/path",
    "path_track_basin_search": "basin/path",
    "pitch_estimation": "pitch",
    "geometry_validation": "geometry",
    "analysis_total": "Analysis",
    "result_construction": "result construction",
    "overlay_visualization": "overlay",
    "result_persistence": "persistence",
    "complete_total": "total",
}


def bundled_image_path() -> Path:
    """Return the fixed image inside source or the PyInstaller bundle."""

    return PROJECT_ROOT / "benchmark_assets" / IMAGE_FILENAME


def default_results_directory() -> Path:
    """Keep results beside the executable in a plainly visible folder."""

    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path.cwd()
    return base / RESULTS_DIRECTORY_NAME


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_fixed_image(image_path: Path) -> tuple[np.ndarray, dict]:
    """Load and validate the exact image used by the Mac benchmark."""

    image_bytes = image_path.read_bytes()
    digest = hashlib.sha256(image_bytes).hexdigest()
    if digest != IMAGE_SHA256:
        raise ValueError(
            f"fixed image SHA-256 mismatch: expected {IMAGE_SHA256}, got {digest}"
        )
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image_gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise ValueError("OpenCV could not decode the fixed benchmark image")
    if image_gray.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(
            "fixed image dimensions mismatch: "
            f"expected {(IMAGE_HEIGHT, IMAGE_WIDTH)}, got {image_gray.shape}"
        )
    return image_gray, {
        "filename": IMAGE_FILENAME,
        "sha256": digest,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
    }


def _bounds_tuple(bounds) -> tuple[int, int, int, int]:
    return (
        int(bounds.x0_global),
        int(bounds.y0_global),
        int(bounds.x1_global),
        int(bounds.y1_global),
    )


def _side_state(interactive: dict, side: str) -> dict:
    value = interactive.get(side)
    side_status = interactive.get("side_status", {}).get(side, {})
    if value is None:
        return {
            "available": False,
            "status": side_status.get("status", "unavailable"),
            "reason": side_status.get("reason"),
            "basin_id": None,
            "left_separator_id": None,
            "right_separator_id": None,
            "distance_to_click_px": None,
        }
    return {
        "available": True,
        "status": side_status.get("status", "available"),
        "reason": side_status.get("reason"),
        "basin_id": value.get("basin_id"),
        "left_separator_id": value.get("left_separator_id"),
        "right_separator_id": value.get("right_separator_id"),
        "distance_to_click_px": value.get("distance_to_click_px"),
    }


def detection_state(result, stage3_result: dict) -> dict:
    """Return stable result fields without embedding profiling timestamps."""

    interactive = result.report["interactive_result"]
    hypothesis = stage3_result.get("final_hypothesis") or {}
    return {
        "success": bool(interactive.get("success")),
        "bilateral_success": bool(interactive.get("bilateral_success")),
        "status": interactive.get("status"),
        "unavailable_reason": interactive.get("unavailable_reason"),
        "left": _side_state(interactive, "left"),
        "right": _side_state(interactive, "right"),
        "basin_ids": hypothesis.get("basin_ids"),
        "separator_ids": hypothesis.get("separator_ids"),
        "separator_sequence": hypothesis.get("separator_sequence"),
        "clicked_separator_id": hypothesis.get("clicked_separator_id"),
        "result_source": result.report.get("result_source"),
    }


def _result_signature(state: dict) -> str:
    encoded = json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_once(
    image_gray: np.ndarray,
    formal_output_dir: Path,
    run_number: int,
) -> dict:
    """Run one unchanged formal analysis and collect its complete profile."""

    reference_global = {"x": REFERENCE_POINT[0], "y": REFERENCE_POINT[1]}
    profile_session = ProfileSession()
    complete_started = time.perf_counter()
    with activate_profile(profile_session):
        with profile_stage(
            "benchmark_image_roi_preparation",
            "image_roi_preparation",
        ):
            bounds = calculate_interactive_roi_bounds(
                image_gray.shape,
                REFERENCE_POINT[0],
                REFERENCE_POINT[1],
                INTERACTIVE_CONFIG,
            )
            actual_roi = _bounds_tuple(bounds)
            if actual_roi != EXPECTED_ROI:
                raise ValueError(
                    f"ROI mismatch: expected {EXPECTED_ROI}, got {actual_roi}"
                )

        analysis_started = time.perf_counter()
        stage3_result = run_frozen_stage3(
            image_gray,
            reference_global,
            bounds,
        )
        analysis_ms = (time.perf_counter() - analysis_started) * 1000.0

        result = build_desktop_result(
            image_gray,
            IMAGE_FILENAME,
            reference_global,
            bounds,
            stage3_result,
            formal_output_dir,
        )
        with profile_stage(
            "benchmark_result_persistence",
            "result_persistence",
        ):
            persist_formal_result(result)

    complete_ms = (time.perf_counter() - complete_started) * 1000.0
    stage_profile = profile_session.report()
    stage_profile["existing_timing_ms"] = {
        "analysis_total_ms": round(analysis_ms, 3),
        "complete_total_ms": round(complete_ms, 3),
    }
    categories = dict(stage_profile["categories_ms"])
    timings = {
        stage: round(float(categories.get(stage, 0.0)), 3)
        for stage in PROFILE_STAGES
        if stage not in {"analysis_total", "complete_total"}
    }
    timings["analysis_total"] = round(analysis_ms, 3)
    timings["complete_total"] = round(complete_ms, 3)
    timings = {stage: timings[stage] for stage in PROFILE_STAGES}
    state = detection_state(result, stage3_result)
    return {
        "run_number": run_number,
        "run_kind": "cold" if run_number == 1 else "warm",
        "timings_ms": timings,
        "stage_profile": stage_profile,
        "detection_state": state,
        "detection_state_sha256": _result_signature(state),
    }


def summarize_warm_runs(runs: list[dict]) -> dict:
    """Return median/min/max for every required stage after warm-up."""

    warm_runs = [run for run in runs if run["run_kind"] == "warm"]
    expected_warm_runs = len(runs) - 1
    if len(warm_runs) != expected_warm_runs:
        raise ValueError(f"expected {expected_warm_runs} warm runs")
    summary = {}
    for stage in PROFILE_STAGES:
        values = [float(run["timings_ms"][stage]) for run in warm_runs]
        summary[stage] = {
            "median_ms": round(float(median(values)), 3),
            "min_ms": round(min(values), 3),
            "max_ms": round(max(values), 3),
        }
    return summary


def system_information() -> dict:
    """Collect dependency-free platform details plus bundled runtime versions."""

    uname = platform.uname()
    opencv_build_information = cv2.getBuildInformation()
    return {
        "platform": platform.platform(),
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": platform.processor(),
        "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "processor_architecture": os.environ.get(
            "PROCESSOR_ARCHITECTURE",
            "",
        ),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "executable": str(Path(sys.executable).resolve()),
        "frozen": bool(getattr(sys, "frozen", False)),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "opencv_num_threads": int(cv2.getNumThreads()),
        "opencv_use_optimized": bool(cv2.useOptimized()),
        "opencv_build_summary": _opencv_build_summary(
            opencv_build_information
        ),
        "opencv_build_information": opencv_build_information,
        "benchmark_build": _benchmark_build_information(),
    }


def _opencv_build_summary(build_information: str) -> list[str]:
    """Extract CPU/SIMD and parallel-runtime lines for quick comparison."""

    lines = build_information.splitlines()
    selected = []
    in_cpu_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("CPU/HW features:"):
            in_cpu_section = True
        elif in_cpu_section and not stripped:
            in_cpu_section = False
        if in_cpu_section or any(
            label in stripped
            for label in (
                "Parallel framework:",
                "Intel IPP:",
                "Intel IPP IW:",
                "OpenCL:",
            )
        ):
            selected.append(stripped)
    return selected


def _benchmark_build_information() -> dict | None:
    """Read the artifact revision shared by the EXE and source bundle."""

    path = PROJECT_ROOT / BUILD_INFO_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary_text(payload: dict) -> str:
    system = payload["system"]
    lines = [
        "Stripe Centerline Windows fixed-point benchmark",
        "",
        f"Mode: {'frozen EXE' if system['frozen'] else 'Python source'}",
        f"Python: {system['python_version']}",
        f"NumPy: {system['numpy_version']}",
        f"OpenCV: {system['opencv_version']}",
        f"OpenCV threads: {system['opencv_num_threads']}",
        f"OpenCV optimized: {system['opencv_use_optimized']}",
        f"Platform: {system['platform']}",
        f"CPU: {system['processor_identifier'] or system['processor']}",
        "",
    ]
    for run in payload["runs"]:
        run_label = (
            "warm-up" if run["run_kind"] == "cold" else "measured"
        )
        lines.append(f"Run {run['run_number']} ({run_label})")
        for stage in PROFILE_STAGES:
            label = SUMMARY_LABELS[stage]
            value = run["timings_ms"][stage]
            lines.append(f"  {label}: {value:.3f} ms")
    lines.extend(
        [
            "",
            "Warm Analysis median: "
            f"{payload['warm_summary']['analysis_total']['median_ms']:.3f} ms",
            "",
        ]
    )
    for stage in PROFILE_STAGES:
        if stage == "analysis_total":
            continue
        label = SUMMARY_LABELS[stage]
        value = payload["warm_summary"][stage]["median_ms"]
        lines.append(f"{label} median: {value:.3f} ms")
    lines.extend(
        [
            "",
            f"Detection states consistent: {payload['results_consistent']}",
            "",
            f"Results saved to: {RESULTS_FILENAME}",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(
    image_path: Path,
    results_dir: Path,
    run_count: int = RUN_COUNT,
) -> dict:
    """Execute fixed runs in one process and persist both reports."""

    if run_count < 2:
        raise ValueError("run count must include warm-up and measured runs")

    results_dir.mkdir(parents=True, exist_ok=True)
    image_load_started = time.perf_counter()
    image_gray, image_info = load_fixed_image(image_path)
    image_load_ms = round(
        (time.perf_counter() - image_load_started) * 1000.0,
        3,
    )
    formal_output_dir = results_dir / "formal_output"
    runs = [
        run_once(image_gray, formal_output_dir, run_number)
        for run_number in range(1, run_count + 1)
    ]
    signatures = [run["detection_state_sha256"] for run in runs]
    payload = {
        "benchmark_revision": BENCHMARK_REVISION,
        "detector_result_source": RESULT_SOURCE,
        "system": system_information(),
        "fixed_case": {
            "image": image_info,
            "image_load_once_ms": image_load_ms,
            "reference_point": {
                "x": REFERENCE_POINT[0],
                "y": REFERENCE_POINT[1],
            },
            "roi": {
                "x0": EXPECTED_ROI[0],
                "y0": EXPECTED_ROI[1],
                "x1": EXPECTED_ROI[2],
                "y1": EXPECTED_ROI[3],
            },
            "run_count": run_count,
            "same_process": True,
            "image_loaded_once": True,
        },
        "profile_stages": list(PROFILE_STAGES),
        "runs": runs,
        "cold_run": runs[0],
        "warm_summary": summarize_warm_runs(runs),
        "results_consistent": len(set(signatures)) == 1,
    }
    _write_json(results_dir / RESULTS_FILENAME, payload)
    (results_dir / SUMMARY_FILENAME).write_text(
        build_summary_text(payload),
        encoding="utf-8",
    )
    return payload


def _show_message(title: str, message: str, error: bool = False) -> None:
    """Show completion for a windowed exe without introducing a full UI."""

    try:
        from tkinter import messagebox

        if error:
            messagebox.showerror(title, message)
        else:
            messagebox.showinfo(title, message)
    except Exception:
        pass


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-dialog", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--run-count",
        type=int,
        default=RUN_COUNT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    results_dir = arguments.output_dir or default_results_directory()
    try:
        payload = run_benchmark(
            bundled_image_path(),
            results_dir,
            arguments.run_count,
        )
    except Exception as error:
        results_dir.mkdir(parents=True, exist_ok=True)
        error_text = traceback.format_exc()
        (results_dir / ERROR_FILENAME).write_text(error_text, encoding="utf-8")
        _write_json(
            results_dir / RESULTS_FILENAME,
            {
                "benchmark_revision": BENCHMARK_REVISION,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": error_text,
                "system": system_information(),
            },
        )
        if not arguments.no_dialog:
            _show_message(
                "Benchmark failed",
                f"Benchmark failed. Error details were saved to:\n{results_dir}",
                error=True,
            )
        return 1

    if not arguments.no_dialog:
        _show_message(
            "Benchmark complete",
            f"{arguments.run_count} benchmark runs completed. "
            "Results were saved to:\n"
            f"{results_dir}\n\n"
            f"Results consistent: {payload['results_consistent']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
