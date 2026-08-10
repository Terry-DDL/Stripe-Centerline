#!/usr/bin/env python3
"""Run and compare staged regression snapshots for reviewed audit points.

Only ``should_detect``, ``valid_unavailable``, and ``correct_detection`` rows
are loaded.  The seven ``wrong_detection`` rows are intentionally excluded.
This is a targeted 207-point runner; it never starts the full sweep.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from config import INTERACTIVE_CONFIG  # noqa: E402
from interactive_pipeline import calculate_interactive_roi_bounds  # noqa: E402
from tools import stage3_desktop_runtime as formal_runtime  # noqa: E402
from tools import basin_graph_joint_prototype as joint  # noqa: E402
from tools import cross_image_correction_prototype_v1_1 as cross_image  # noqa: E402


IN_SCOPE_LABELS = {
    "should_detect",
    "valid_unavailable",
    "correct_detection",
}
CENTER_TOLERANCE_PX = 1.0
SPACING_TOLERANCE_PX = 1.0


def _read_review(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("human_label") in IN_SCOPE_LABELS
        ]
    counts = Counter(row["human_label"] for row in rows)
    expected = {
        "should_detect": 45,
        "valid_unavailable": 39,
        "correct_detection": 123,
    }
    if counts != expected:
        raise ValueError(f"expected targeted cohorts {expected}, got {dict(counts)}")
    if len(rows) != 207:
        raise ValueError(f"expected 207 targeted rows, got {len(rows)}")
    return rows


def _geometry_snapshot(result: dict[str, Any]) -> dict[str, Any] | None:
    hypothesis = result.get("final_hypothesis")
    if not result.get("success") or not hypothesis:
        return None
    geometry = hypothesis.get("geometry", {})
    basins = geometry.get("basins", {})
    left = basins.get("left", {})
    right = basins.get("right", {})
    left_center = left.get("center_x_at_reference_global")
    right_center = right.get("center_x_at_reference_global")
    if left_center is None or right_center is None:
        return None
    return {
        "reference_relation": hypothesis.get("reference_relation"),
        "basin_ids": hypothesis.get("basin_ids"),
        "separator_sequence": hypothesis.get("separator_sequence"),
        "clicked_separator_id": hypothesis.get("clicked_separator_id"),
        "left_center": float(left_center),
        "right_center": float(right_center),
        "spacing": float(right_center) - float(left_center),
    }


def _case_snapshot(row: dict[str, str], result: dict[str, Any]) -> dict[str, Any]:
    debug = result.get("debug", {}) or {}
    separator = debug.get("separator_result", {}) or {}
    relation = debug.get("reference_relation", {}) or {}
    recovery = debug.get("local_dark_valley_recovery", {}) or {}
    return {
        "review_id": row["review_id"],
        "human_label": row["human_label"],
        "image": row["image"],
        "x": int(row["x"]),
        "y": int(row["y"]),
        "is_stripe01_09": row["image"].startswith(("Stripe_01_", "Stripe_09_")),
        "success": bool(result.get("success")),
        "unavailable_reason": result.get("unavailable_reason") or "",
        "geometry": _geometry_snapshot(result),
        "diagnostics": {
            "separator_reason": separator.get("unavailable_reason") or "",
            "reference_relation_status": relation.get("status") or "",
            "reference_relation_reason": relation.get("unavailable_reason") or "",
            "recovery_triggered": bool(recovery.get("triggered")),
            "recovery_success": bool(recovery.get("success")),
            "recovery_reason": recovery.get("reason") or "",
            "staged_release": debug.get("staged_release", {}),
        },
    }


def run_snapshot(
    review_path: Path,
    sweep_path: Path,
    stage_name: str,
    max_stage: int,
) -> dict[str, Any]:
    cross_image.ENABLE_REFERENCE_SEMANTIC_TIEBREAK = max_stage >= 1
    joint.ENABLE_LOCAL_REFERENCE_ADJACENCY_VALIDATION = max_stage >= 2
    cross_image.ENABLE_PITCH_GUIDED_MISSING_SEPARATOR_COMPLETION = max_stage >= 3
    rows = _read_review(review_path)
    with sweep_path.open("r", encoding="utf-8-sig", newline="") as handle:
        image_paths = {
            row["image"]: row["image_path"]
            for row in csv.DictReader(handle)
        }
    image_cache: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        image_path = image_paths[row["image"]]
        image = image_cache.get(image_path)
        if image is None:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"unable to read {image_path}")
            image_cache[image_path] = image
        x, y = int(row["x"]), int(row["y"])
        bounds = calculate_interactive_roi_bounds(
            image.shape,
            x,
            y,
            INTERACTIVE_CONFIG,
        )
        try:
            result = formal_runtime.run_frozen_stage3(
                image,
                {"x": x, "y": y},
                bounds,
            )
        except Exception as error:
            result = {
                "success": False,
                "unavailable_reason": f"formal_pipeline_exception:{type(error).__name__}",
                "debug": {"exception_message": str(error)},
            }
        cases.append(_case_snapshot(row, result))
        if index % 25 == 0 or index == len(rows):
            print(f"{stage_name}: {index}/{len(rows)}")
    return {
        "schema_version": 1,
        "stage": stage_name,
        "max_stage": max_stage,
        "formal_pipeline": "tools.stage3_desktop_runtime.run_frozen_stage3",
        "full_sweep_rerun": False,
        "wrong_detection_loaded": 0,
        "case_count": len(cases),
        "cases": cases,
    }


def _material_geometry_changes(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    if before is None or after is None:
        return [] if before is after else ["geometry_presence"]
    changes: list[str] = []
    if before.get("reference_relation") != after.get("reference_relation"):
        changes.append("reference_relation")
    if before.get("basin_ids") != after.get("basin_ids"):
        changes.append("basin_identity")
    if before.get("separator_sequence") != after.get("separator_sequence"):
        changes.append("separator_roles")
    if before.get("clicked_separator_id") != after.get("clicked_separator_id"):
        changes.append("clicked_separator_identity")
    for key in ("left_center", "right_center"):
        if abs(float(before[key]) - float(after[key])) > CENTER_TOLERANCE_PX:
            changes.append(key)
    if abs(float(before["spacing"]) - float(after["spacing"])) > SPACING_TOLERANCE_PX:
        changes.append("spacing")
    return changes


def compare_snapshots(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    previous_by_id = {case["review_id"]: case for case in previous["cases"]}
    current_by_id = {case["review_id"]: case for case in current["cases"]}
    if set(previous_by_id) != set(current_by_id):
        raise ValueError("snapshot review_id sets differ")

    newly_recovered = []
    newly_lost_should_detect = []
    valid_regressions = []
    correct_regressions = []
    correct_geometry_changes = []
    for review_id, before in previous_by_id.items():
        after = current_by_id[review_id]
        label = before["human_label"]
        if label == "should_detect":
            if not before["success"] and after["success"]:
                newly_recovered.append(review_id)
            elif before["success"] and not after["success"]:
                newly_lost_should_detect.append(review_id)
        elif label == "valid_unavailable":
            if not before["success"] and after["success"]:
                valid_regressions.append(review_id)
        elif label == "correct_detection":
            if before["success"] and not after["success"]:
                correct_regressions.append({"review_id": review_id, "change": "became_unavailable"})
            elif before["success"] and after["success"]:
                changes = _material_geometry_changes(before["geometry"], after["geometry"])
                if changes:
                    item = {"review_id": review_id, "changes": changes}
                    correct_regressions.append(item)
                    correct_geometry_changes.append(item)

    current_should_success = [
        case for case in current["cases"]
        if case["human_label"] == "should_detect" and case["success"]
    ]
    return {
        "previous_stage": previous["stage"],
        "current_stage": current["stage"],
        "incremental_should_detect_recovered": len(newly_recovered),
        "incremental_should_detect_recovered_ids": newly_recovered,
        "incremental_should_detect_lost": len(newly_lost_should_detect),
        "incremental_should_detect_lost_ids": newly_lost_should_detect,
        "cumulative_should_detect_recovered": len(current_should_success),
        "cumulative_non_stripe01_09_recovered": sum(
            not case["is_stripe01_09"] for case in current_should_success
        ),
        "cumulative_stripe01_09_recovered": sum(
            case["is_stripe01_09"] for case in current_should_success
        ),
        "valid_unavailable_regression_count": len(valid_regressions),
        "valid_unavailable_regression_ids": valid_regressions,
        "correct_detection_regression_count": len(correct_regressions),
        "correct_detection_regressions": correct_regressions,
        "correct_detection_geometry_change_count": len(correct_geometry_changes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument(
        "--review",
        type=Path,
        default=PROJECT_ROOT / "outputs/batch_anomaly_scan_20260807_full/manual_review/review.csv",
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        default=PROJECT_ROOT / "outputs/batch_anomaly_scan_20260807_full/sweep_results.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    parser.add_argument("--max-stage", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument(
        "--existing-snapshot",
        type=Path,
        help="Compare an already written snapshot without rerunning detection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.existing_snapshot:
        snapshot = json.loads(args.existing_snapshot.read_text(encoding="utf-8"))
    else:
        snapshot = run_snapshot(
            args.review.resolve(),
            args.sweep.resolve(),
            args.stage,
            args.max_stage,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    if args.compare_to:
        previous = json.loads(args.compare_to.read_text(encoding="utf-8"))
        comparison = compare_snapshots(previous, snapshot)
        output = args.comparison_output or args.output.with_name(
            f"{args.output.stem}_comparison.json"
        )
        output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
