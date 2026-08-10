#!/usr/bin/env python3
"""Replay the frozen 2,800-point sweep and compare baseline with staged v1.1.

This is an evaluation-only runner.  It reuses the exact image/x/y rows in the
archived sweep and calls the formal desktop runtime without changing detector
configuration, thresholds, or decisions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import batch_anomaly_scan as batch  # noqa: E402
from tools.run_staged_targeted_regression import (  # noqa: E402
    _geometry_snapshot,
    _material_geometry_changes,
)


DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "batch_anomaly_scan_20260807_full"
FORMAL_FILES = (
    PROJECT_ROOT / "tools" / "stage3_desktop_runtime.py",
    PROJECT_ROOT / "tools" / "cross_image_correction_prototype_v1_1.py",
    PROJECT_ROOT / "tools" / "basin_graph_joint_prototype.py",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _key(row: dict[str, Any]) -> tuple[str, int, int]:
    return row["image"], int(row["x"]), int(row["y"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _formal_hashes() -> dict[str, str]:
    return {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in FORMAL_FILES}


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def replay_exact_baseline(
    baseline: list[dict[str, str]], output_dir: Path, workers: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    spacing: dict[tuple[str, str], int] = {}
    for row in baseline:
        group = (row["image"], row["image_path"])
        grouped[group].append((int(row["x"]), int(row["y"])))
        spacing[group] = int(row["grid_spacing_px"])

    diagnostics_dir = output_dir / "diagnostics"
    tasks = []
    for index, ((image, image_path), points) in enumerate(sorted(grouped.items()), 1):
        tasks.append(
            {
                "image_id": f"F{index:02d}_{Path(image).stem}",
                "image_path": image_path,
                "points": points,
                "spacing_px": spacing[(image, image_path)],
                "diagnostics_path": str(diagnostics_dir / f"F{index:02d}_{Path(image).stem}.jsonl.gz"),
            }
        )

    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        futures = {pool.submit(batch.scan_one_image, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            task = futures[future]
            image_records = future.result()
            records.extend(image_records)
            completed += 1
            print(
                f"full staged replay: {completed}/{len(tasks)} images; "
                f"{task['image_id']} ({len(image_records)} points)",
                flush=True,
            )
    records.sort(key=lambda row: (row["image"].lower(), int(row["y"]), int(row["x"])))
    if {_key(row) for row in records} != {_key(row) for row in baseline}:
        raise RuntimeError("current replay coordinate set differs from baseline")
    if len(records) != len(baseline):
        raise RuntimeError("current replay row count differs from baseline")
    _write_csv(output_dir / "current_sweep_results.csv", batch.SWEEP_FIELDS, records)
    return records


def _load_raw_results(
    rows: list[dict[str, Any]], wanted: set[tuple[str, int, int]]
) -> dict[tuple[str, int, int], dict[str, Any]]:
    paths = sorted({Path(row["raw_diagnostics_path"]) for row in rows if _key(row) in wanted})
    results: dict[tuple[str, int, int], dict[str, Any]] = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                raw = json.loads(line)
                key = (raw["image"], int(raw["x"]), int(raw["y"]))
                if key in wanted:
                    results[key] = raw["stage3_result"]
    missing = wanted - set(results)
    if missing:
        raise RuntimeError(f"missing {len(missing)} raw diagnostic records")
    return results


def _status_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    success = sum(str(row["current_result"]).lower() == "success" for row in rows)
    suspicious = sum(_bool(row["suspicious_success"]) for row in rows)
    return {
        "success": success,
        "unavailable": len(rows) - success,
        "suspicious_success": suspicious,
    }


def evaluate(
    baseline: list[dict[str, str]],
    current: list[dict[str, Any]],
    review: list[dict[str, str]],
    hashes_before: dict[str, str],
    hashes_after: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_by_key = {_key(row): row for row in baseline}
    current_by_key = {_key(row): row for row in current}
    review_by_key = {_key(row): row for row in review}

    original_success_keys = {
        key for key, row in baseline_by_key.items() if row["current_result"] == "success"
    }
    baseline_raw = _load_raw_results(baseline, original_success_keys)
    current_raw = _load_raw_results(current, original_success_keys)

    new_successes: list[dict[str, Any]] = []
    lost_successes: list[dict[str, Any]] = []
    original_success_changes: list[dict[str, Any]] = []
    for key, before in baseline_by_key.items():
        after = current_by_key[key]
        before_success = before["current_result"] == "success"
        after_success = after["current_result"] == "success"
        if not before_success and after_success:
            review_row = review_by_key.get(key, {})
            new_successes.append(
                {
                    "image": key[0],
                    "x": key[1],
                    "y": key[2],
                    "original_failure_reason": before["primary_failure_reason"],
                    "current_suspicious_success": _bool(after["suspicious_success"]),
                    "current_why_suspicious": after["why_suspicious"],
                    "human_label": review_row.get("human_label", ""),
                    "review_id": review_row.get("review_id", ""),
                    "left_center": after["left_center"],
                    "right_center": after["right_center"],
                    "reference_relationship": after["reference_relationship"],
                }
            )
        elif before_success and not after_success:
            lost_successes.append(
                {
                    "image": key[0], "x": key[1], "y": key[2],
                    "new_failure_reason": after["primary_failure_reason"],
                }
            )
        elif before_success and after_success:
            before_geometry = _geometry_snapshot(baseline_raw[key])
            after_geometry = _geometry_snapshot(current_raw[key])
            changes = _material_geometry_changes(before_geometry, after_geometry)
            if changes:
                original_success_changes.append(
                    {
                        "image": key[0], "x": key[1], "y": key[2],
                        "changes": ";".join(changes),
                        "baseline_geometry": json.dumps(before_geometry, ensure_ascii=False, sort_keys=True),
                        "current_geometry": json.dumps(after_geometry, ensure_ascii=False, sort_keys=True),
                    }
                )

    review_changes = Counter()
    review_details = []
    for row in review:
        label = row["human_label"]
        before = baseline_by_key[_key(row)]
        after = current_by_key[_key(row)]
        before_status, after_status = before["current_result"], after["current_result"]
        if label == "wrong_detection":
            review_changes["wrong_detection_excluded"] += 1
            continue
        if before_status != after_status:
            review_changes[f"{label}:{before_status}_to_{after_status}"] += 1
            review_details.append(
                {"review_id": row["review_id"], "human_label": label,
                 "image": row["image"], "x": row["x"], "y": row["y"],
                 "baseline_result": before_status, "current_result": after_status}
            )

    new_suspicious = [
        row for row in new_successes if row["current_suspicious_success"]
    ]
    existing_became_suspicious = []
    for key in original_success_keys:
        before, after = baseline_by_key[key], current_by_key[key]
        if after["current_result"] == "success" and not _bool(before["suspicious_success"]) and _bool(after["suspicious_success"]):
            existing_became_suspicious.append(
                {"image": key[0], "x": key[1], "y": key[2], "why": after["why_suspicious"]}
            )

    by_image = Counter(row["image"] for row in new_successes)
    by_reason = Counter(row["original_failure_reason"] for row in new_successes)
    by_label = Counter(row["human_label"] or "not_in_review" for row in new_successes)
    suspicious_reasons = Counter()
    for row in new_suspicious:
        suspicious_reasons.update(filter(None, row["current_why_suspicious"].split(";")))

    risk = {
        "valid_unavailable_became_success": by_label["valid_unavailable"],
        "invalid_scene_became_success": by_label["invalid_scene"],
        "correct_detection_became_unavailable": sum(
            item["human_label"] == "correct_detection" and item["current_result"] == "unavailable"
            for item in review_details
        ),
        "original_success_lost": len(lost_successes),
        "original_success_identity_or_geometry_changed": len(original_success_changes),
        "original_normal_success_became_suspicious": len(existing_became_suspicious),
        "new_successes_marked_suspicious": len(new_suspicious),
    }
    clean_for_commit = all(
        risk[key] == 0
        for key in (
            "valid_unavailable_became_success",
            "invalid_scene_became_success",
            "original_success_lost",
            "original_success_identity_or_geometry_changed",
            "original_normal_success_became_suspicious",
        )
    )
    result = {
        "schema_version": 1,
        "formal_pipeline": "tools.stage3_desktop_runtime.run_frozen_stage3",
        "exact_baseline_coordinate_replay": True,
        "full_sweep_points": len(baseline),
        "images": len({row["image"] for row in baseline}),
        "wrong_detection_excluded_from_review_risk": 7,
        "candidate4_implemented": False,
        "formal_file_hashes_unchanged_during_evaluation": hashes_before == hashes_after,
        "formal_file_hashes_before": hashes_before,
        "formal_file_hashes_after": hashes_after,
        "overall": {
            "baseline": _status_summary(baseline),
            "current": _status_summary(current),
        },
        "transitions": {
            "unavailable_to_success": len(new_successes),
            "success_to_unavailable": len(lost_successes),
        },
        "new_successes": {
            "count": len(new_successes),
            "by_image": dict(by_image.most_common()),
            "by_original_failure_reason": dict(by_reason.most_common()),
            "by_human_label": dict(by_label.most_common()),
            "suspicious_count": len(new_suspicious),
            "suspicious_reason_signals": dict(suspicious_reasons.most_common()),
        },
        "manual_review_250": {
            "baseline_labels": dict(Counter(row["human_label"] for row in review)),
            "status_changes_excluding_wrong_detection": dict(review_changes),
            "should_detect_recovered": by_label["should_detect"],
            "valid_unavailable_became_success": by_label["valid_unavailable"],
            "invalid_scene_became_success": by_label["invalid_scene"],
            "wrong_detection_excluded": 7,
        },
        "original_success": {
            "count": len(original_success_keys),
            "became_unavailable": len(lost_successes),
            "identity_or_geometry_changes": len(original_success_changes),
        },
        "risk_surface": risk,
        "clean_for_commit": clean_for_commit,
        "recommend_commit": clean_for_commit,
        "recommend_larger_release_candidate": clean_for_commit,
    }
    return result, new_successes, original_success_changes


def _render_report(result: dict[str, Any]) -> str:
    overall = result["overall"]
    base, current = overall["baseline"], overall["current"]
    new = result["new_successes"]
    review = result["manual_review_250"]
    risk = result["risk_surface"]
    lines = [
        "# Full 2,800-point staged regression evaluation", "",
        "## Overall", "",
        f"- Baseline: success {base['success']}, unavailable {base['unavailable']}, suspicious success {base['suspicious_success']}.",
        f"- Current: success {current['success']}, unavailable {current['unavailable']}, suspicious success {current['suspicious_success']}.",
        f"- Delta: success {current['success'] - base['success']:+d}, unavailable {current['unavailable'] - base['unavailable']:+d}, suspicious success {current['suspicious_success'] - base['suspicious_success']:+d}.",
        f"- Transitions: unavailable→success {result['transitions']['unavailable_to_success']}; success→unavailable {result['transitions']['success_to_unavailable']}.",
        "", "## Manual review set", "",
        f"- Should-detect recovered: {review['should_detect_recovered']} / 45.",
        f"- Valid-unavailable released: {review['valid_unavailable_became_success']} / 39.",
        f"- Invalid-scene released: {review['invalid_scene_became_success']} / 36.",
        "- Seven wrong-detection rows were excluded from review risk and blocker logic.",
        "", "## New successes", "",
        f"- Total: {new['count']}; marked suspicious by existing diagnostics: {new['suspicious_count']}.",
        "- By image:",
    ]
    lines.extend(f"  - `{name}`: {count}" for name, count in new["by_image"].items())
    lines.append("- By original failure reason:")
    lines.extend(f"  - `{name}`: {count}" for name, count in new["by_original_failure_reason"].items())
    lines.extend([
        "", "## Original success stability", "",
        f"- Original successes: {result['original_success']['count']}.",
        f"- Became unavailable: {result['original_success']['became_unavailable']}.",
        f"- Material identity/geometry changes (>1 px or role/identity change): {result['original_success']['identity_or_geometry_changes']}.",
        "", "## Risk surface", "",
    ])
    lines.extend(f"- `{name}`: {count}" for name, count in risk.items())
    lines.extend([
        "", "## Recommendation", "",
        f"- Formal detector hashes unchanged during evaluation: {result['formal_file_hashes_unchanged_during_evaluation']}.",
        f"- Recommend commit: {result['recommend_commit']}.",
        f"- Recommend larger release candidate: {result['recommend_larger_release_candidate']}.",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_ROOT / "sweep_results.csv")
    parser.add_argument("--review", type=Path, default=DEFAULT_ROOT / "manual_review" / "review.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "full_staged_regression")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = _read_csv(args.baseline.resolve())
    review = _read_csv(args.review.resolve())
    if len(baseline) != 2800 or len({_key(row) for row in baseline}) != 2800:
        raise ValueError("expected exactly 2,800 unique baseline points")
    if len(review) != 250:
        raise ValueError("expected 250 manual review rows")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes_before = _formal_hashes()
    current = replay_exact_baseline(baseline, output_dir, args.workers)
    hashes_after = _formal_hashes()
    result, new_successes, success_changes = evaluate(
        baseline, current, review, hashes_before, hashes_after
    )
    (output_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(
        output_dir / "new_successes.csv",
        ["image", "x", "y", "original_failure_reason", "current_suspicious_success",
         "current_why_suspicious", "human_label", "review_id", "left_center",
         "right_center", "reference_relationship"],
        new_successes,
    )
    _write_csv(
        output_dir / "original_success_changes.csv",
        ["image", "x", "y", "changes", "baseline_geometry", "current_geometry"],
        success_changes,
    )
    (output_dir / "full_staged_regression_report.md").write_text(
        _render_report(result), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
