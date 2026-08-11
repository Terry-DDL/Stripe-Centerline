#!/usr/bin/env python3
"""Evaluate the Stage A decision-ordering change on frozen audit surfaces."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.evaluate_full_staged_regression import (  # noqa: E402
    _geometry_snapshot,
    _key,
    _load_raw_results,
    _material_geometry_changes,
    _read_csv,
    _write_csv,
    replay_exact_baseline,
)


OLD_ROOT = PROJECT_ROOT / "outputs" / "batch_anomaly_scan_20260807_full"
RANDOM_ROOT = PROJECT_ROOT / "outputs" / "rc_random_validation_1254815_seed_20260810"
OUTPUT_ROOT = RANDOM_ROOT / "stage_a_reference_competition_regression"

SURFACES = {
    "old_2800": OLD_ROOT / "rr058_orientation_fix_full_regression" / "current_sweep_results.csv",
    "random_1200": RANDOM_ROOT / "sweep_results.csv",
}

REVIEW_SOURCES = (
    OLD_ROOT / "manual_review" / "review.csv",
    OLD_ROOT / "new_successes_manual_review_75" / "review.csv",
    RANDOM_ROOT / "manual_review" / "review.csv",
    RANDOM_ROOT
    / "rr058_orientation_diagnosis"
    / "manual_review_rotation_geometry_8"
    / "manual_review"
    / "review.csv",
)


def _review_labels() -> dict[tuple[str, int, int], str]:
    labels: dict[tuple[str, int, int], str] = {}
    for path in REVIEW_SOURCES:
        for row in _read_csv(path):
            label = row.get("human_label", "").strip()
            if not label:
                continue
            key = _key(row)
            previous = labels.get(key)
            if previous and previous != label:
                raise ValueError(f"conflicting labels for {key}: {previous}, {label}")
            labels[key] = label
    return labels


def _surface_diff(
    name: str,
    baseline: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    before_by_key = {_key(row): row for row in baseline}
    after_by_key = {_key(row): row for row in current}
    if set(before_by_key) != set(after_by_key):
        raise ValueError(f"{name}: coordinate sets differ")
    success_before = {
        key for key, row in before_by_key.items() if row["current_result"] == "success"
    }
    raw_before = _load_raw_results(baseline, success_before)
    raw_after = _load_raw_results(current, success_before)
    transitions: list[dict[str, Any]] = []
    geometry_changes: list[dict[str, Any]] = []
    for key, before in before_by_key.items():
        after = after_by_key[key]
        before_status = before["current_result"]
        after_status = after["current_result"]
        if before_status != after_status:
            transitions.append(
                {
                    "surface": name,
                    "image": key[0],
                    "x": key[1],
                    "y": key[2],
                    "before": before_status,
                    "after": after_status,
                    "before_reason": before.get("primary_failure_reason", ""),
                    "after_reason": after.get("primary_failure_reason", ""),
                }
            )
        if before_status == after_status == "success":
            before_geometry = _geometry_snapshot(raw_before[key])
            after_geometry = _geometry_snapshot(raw_after[key])
            changes = _material_geometry_changes(before_geometry, after_geometry)
            if changes:
                geometry_changes.append(
                    {
                        "surface": name,
                        "image": key[0],
                        "x": key[1],
                        "y": key[2],
                        "changes": ";".join(changes),
                        "before_geometry": json.dumps(before_geometry, sort_keys=True),
                        "after_geometry": json.dumps(after_geometry, sort_keys=True),
                    }
                )
    return {
        "points": len(baseline),
        "before_success": len(success_before),
        "after_success": sum(
            row["current_result"] == "success" for row in current
        ),
        "status_transition_count": len(transitions),
        "success_identity_or_geometry_change_count": len(geometry_changes),
        "transitions": transitions,
        "geometry_changes": geometry_changes,
    }, transitions + geometry_changes


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    labels = _review_labels()
    label_counts = Counter(labels.values())
    expected = {
        "should_detect": 84,
        "valid_unavailable": 39,
        "correct_detection": 376,
        "wrong_detection": 12,
        "invalid_scene": 72,
    }
    if any(label_counts[name] != count for name, count in expected.items()):
        raise ValueError(f"unexpected merged labels: {dict(label_counts)}")

    surface_results: dict[str, Any] = {}
    combined_baseline: dict[tuple[str, int, int], dict[str, Any]] = {}
    combined_current: dict[tuple[str, int, int], dict[str, Any]] = {}
    all_changes: list[dict[str, Any]] = []
    for name, path in SURFACES.items():
        baseline = _read_csv(path)
        output_dir = OUTPUT_ROOT / name
        current = replay_exact_baseline(baseline, output_dir, workers=4)
        result, changes = _surface_diff(name, baseline, current)
        surface_results[name] = result
        all_changes.extend(changes)
        combined_baseline.update({_key(row): row for row in baseline})
        combined_current.update({_key(row): row for row in current})

    random_review = _read_csv(RANDOM_ROOT / "manual_review" / "review.csv")
    random_should = {
        _key(row): row for row in random_review if row["human_label"] == "should_detect"
    }
    if len(random_should) != 39:
        raise ValueError("expected 39 random should-detect cases")
    recovered = [
        row["review_id"]
        for key, row in random_should.items()
        if combined_baseline[key]["current_result"] == "unavailable"
        and combined_current[key]["current_result"] == "success"
    ]

    correct_keys = {key for key, label in labels.items() if label == "correct_detection"}
    valid_keys = {key for key, label in labels.items() if label == "valid_unavailable"}
    correct_regressions = []
    valid_releases = []
    for key in correct_keys:
        before = combined_baseline[key]
        after = combined_current[key]
        if before["current_result"] != after["current_result"]:
            correct_regressions.append({"key": key, "change": "status"})
    geometry_changed_keys = {
        (row["image"], int(row["x"]), int(row["y"]))
        for result in surface_results.values()
        for row in result["geometry_changes"]
    }
    correct_regressions.extend(
        {"key": key, "change": "identity_or_geometry"}
        for key in sorted(correct_keys & geometry_changed_keys)
    )
    for key in valid_keys:
        if (
            combined_baseline[key]["current_result"] == "unavailable"
            and combined_current[key]["current_result"] == "success"
        ):
            valid_releases.append(key)

    summary = {
        "schema_version": 1,
        "stage": "local_adjacency_before_reference_competition_return",
        "formal_pipeline": "tools.stage3_desktop_runtime.run_frozen_stage3",
        "surface_points": sum(item["points"] for item in surface_results.values()),
        "surfaces": surface_results,
        "targeted": {
            "random_should_detect_count": 39,
            "random_should_detect_recovered_count": len(recovered),
            "random_should_detect_recovered_ids": recovered,
            "confirmed_correct_count": len(correct_keys),
            "confirmed_correct_regression_count": len(correct_regressions),
            "confirmed_correct_regressions": correct_regressions,
            "valid_unavailable_count": len(valid_keys),
            "valid_unavailable_released_count": len(valid_releases),
            "valid_unavailable_released": valid_releases,
        },
        "clean": not correct_regressions and not valid_releases,
    }
    (OUTPUT_ROOT / "evaluation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(
        OUTPUT_ROOT / "surface_changes.csv",
        [
            "surface", "image", "x", "y", "before", "after",
            "before_reason", "after_reason", "changes", "before_geometry",
            "after_geometry",
        ],
        all_changes,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
