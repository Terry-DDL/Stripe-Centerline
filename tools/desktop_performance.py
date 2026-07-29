"""Small thread-safe performance log for desktop click analysis."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from statistics import median
import threading


PERFORMANCE_REVISION = "desktop_stage3_switch_timing_v1"
DEFAULT_LOG_ROOT = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "stage3_production_v1"
    / "desktop_performance"
)
PER_CLICK_FILENAME = "desktop_performance_timing.json"
RUNS_FILENAME = "performance_timings.json"
SUMMARY_FILENAME = "performance_summary.json"
TIMING_FIELDS = (
    "legacy_algorithm_ms",
    "stage3_algorithm_ms",
    "result_write_ms",
    "ui_draw_ms",
    "click_to_display_ms",
)
_LOCK = threading.Lock()


def elapsed_ms(start_ns: int, end_ns: int) -> float:
    """Convert one monotonic nanosecond interval into milliseconds."""

    return round((end_ns - start_ns) / 1_000_000.0, 3)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def summarize_runs(runs: list[dict]) -> dict:
    """Return independent sample count, median, and max for every phase."""

    phases = {}
    for field in TIMING_FIELDS:
        values = [
            float(run[field])
            for run in runs
            if run.get(field) is not None
        ]
        phases[field] = {
            "sample_count": len(values),
            "median_ms": round(float(median(values)), 3) if values else None,
            "max_ms": round(max(values), 3) if values else None,
        }
    complete_count = sum(
        all(run.get(field) is not None for field in TIMING_FIELDS)
        for run in runs
    )
    return {
        "performance_revision": PERFORMANCE_REVISION,
        "run_count": len(runs),
        "complete_run_count": complete_count,
        "phases": phases,
    }


def update_timing(
    log_root: Path,
    output_dir: Path,
    run_id: str,
    metadata: dict,
    measurements: dict,
) -> dict:
    """Merge one worker/UI timing update and refresh cumulative summaries."""

    unknown = set(measurements) - set(TIMING_FIELDS) - {
        "stage3_error",
        "legacy_diagnostic_error",
        "result_write_error",
    }
    if unknown:
        raise ValueError(f"unknown timing fields: {sorted(unknown)}")
    runs_path = log_root / RUNS_FILENAME
    summary_path = log_root / SUMMARY_FILENAME
    with _LOCK:
        if runs_path.exists():
            payload = json.loads(runs_path.read_text(encoding="utf-8"))
        else:
            payload = {
                "performance_revision": PERFORMANCE_REVISION,
                "runs": [],
            }
        runs = payload["runs"]
        record = next(
            (item for item in runs if item["run_id"] == run_id),
            None,
        )
        if record is None:
            record = {
                "run_id": run_id,
                "timestamp": datetime.now().astimezone().isoformat(),
                **metadata,
                **{field: None for field in TIMING_FIELDS},
                "stage3_error": None,
                "legacy_diagnostic_error": None,
                "result_write_error": None,
            }
            runs.append(record)
        record.update(measurements)
        record["updated_at"] = datetime.now().astimezone().isoformat()
        summary = summarize_runs(runs)
        _atomic_write_json(runs_path, payload)
        _atomic_write_json(summary_path, summary)
        _atomic_write_json(
            output_dir / PER_CLICK_FILENAME,
            {
                "performance_revision": PERFORMANCE_REVISION,
                "timing": record,
                "cumulative_summary": summary,
            },
        )
    return summary
