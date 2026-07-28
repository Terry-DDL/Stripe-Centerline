"""Compare baseline and current read-only candidate diagnostic exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _identity_to_candidate_id(identity: dict | None) -> str | None:
    if identity is None:
        return None
    return f"{identity['geometry']}_{identity['threshold_method']}"


def _formal_candidate_id(case: dict) -> str | None:
    formal = case["formal_result"]
    if not formal["success"]:
        return None
    geometry = formal.get("geometry")
    threshold_method = formal.get("threshold_method")
    if geometry is None or threshold_method is None:
        return None
    return f"{geometry}_{threshold_method}"


def _candidate_map(case: dict) -> dict[str, dict]:
    return {
        candidate["candidate_id"]: candidate
        for candidate in case["candidates"]
    }


def _is_safe_failure(formal_result: dict) -> bool:
    return (
        not formal_result["success"]
        and all(
            formal_result.get(field) is None
            for field in (
                "left",
                "clicked",
                "right",
                "stripe_spacing_px",
            )
        )
    )


def _selected_centers(candidate: dict | None) -> dict | None:
    if candidate is None:
        return None
    return {
        label: (
            None
            if track is None
            else track["center_x_global_at_click_y"]
        )
        for label, track in candidate["selected_tracks"].items()
    }


def _adjacency_summary(candidate: dict | None) -> dict | None:
    if candidate is None:
        return None
    verification = candidate["adjacency_verification"]
    return {
        "status": (
            None if verification is None else verification["status"]
        ),
        "reason": (
            None if verification is None else verification["reason"]
        ),
        "neighbor_consistency": candidate["neighbor_consistency"],
        "quality_success": candidate["quality_success"],
        "quality_hard_invalid_reasons": candidate[
            "quality_hard_invalid_reasons"
        ],
        "topology_status": candidate["grayscale_topology"]["status"],
        "selection_matches_ground_truth": candidate[
            "selection_matches_ground_truth"
        ],
        "selected_centers_global": _selected_centers(candidate),
    }


def compare_case(baseline_case: dict, current_case: dict) -> dict:
    """Compare one case at the candidate and final-result layers."""

    baseline_candidates = _candidate_map(baseline_case)
    current_candidates = _candidate_map(current_case)
    baseline_winner_id = _formal_candidate_id(baseline_case)
    adjacency_arbitration = current_case.get("adjacency_arbitration")
    current_pre_gate_id = (
        None
        if adjacency_arbitration is None
        else _identity_to_candidate_id(
            adjacency_arbitration["formal_winner_before_gate"]
        )
    )
    current_debug_id = (
        None
        if adjacency_arbitration is None
        else _identity_to_candidate_id(
            adjacency_arbitration["debug_winner"]
        )
    )
    current_final_id = _formal_candidate_id(current_case)

    baseline_correct = sorted(
        candidate_id
        for candidate_id, candidate in baseline_candidates.items()
        if candidate["selection_matches_ground_truth"]
    )
    current_correct = sorted(
        candidate_id
        for candidate_id, candidate in current_candidates.items()
        if candidate["selection_matches_ground_truth"]
    )
    retained_correct = sorted(
        set(baseline_correct).intersection(current_correct)
    )
    baseline_winner_is_correct = baseline_winner_id in baseline_correct
    current_safe_failure = _is_safe_failure(
        current_case["formal_result"]
    )
    pre_gate_candidate = current_candidates.get(current_pre_gate_id)
    pre_gate_verification = (
        None
        if pre_gate_candidate is None
        else pre_gate_candidate["adjacency_verification"]
    )
    if (
        retained_correct
        and pre_gate_candidate is not None
        and pre_gate_candidate["selection_matches_ground_truth"]
        and pre_gate_verification is not None
        and pre_gate_verification["status"] != "verified"
    ):
        interpretation = (
            "correct_candidate_retained_but_rejected_by_adjacency_verification"
        )
    elif not retained_correct:
        interpretation = "correct_candidate_not_retained"
    elif current_final_id is not None:
        interpretation = "correct_candidate_remains_formally_available"
    else:
        interpretation = "formal_failure_requires_manual_review"

    return {
        "case_id": current_case["case_id"],
        "ground_truth": current_case["ground_truth"],
        "baseline_formal_success": baseline_case[
            "formal_result"
        ]["success"],
        "current_formal_success": current_case[
            "formal_result"
        ]["success"],
        "baseline_winner_is_correct": baseline_winner_is_correct,
        "current_is_safe_failure": current_safe_failure,
        "baseline_winner": baseline_winner_id,
        "current_pre_gate_winner": current_pre_gate_id,
        "current_debug_winner": current_debug_id,
        "current_final_winner": current_final_id,
        "baseline_correct_candidate_ids": baseline_correct,
        "current_correct_candidate_ids": current_correct,
        "same_correct_candidate_ids": retained_correct,
        "baseline_winner_evidence": _adjacency_summary(
            baseline_candidates.get(baseline_winner_id)
        ),
        "current_pre_gate_evidence": _adjacency_summary(
            pre_gate_candidate
        ),
        "current_loss_stage": current_case["loss_stage"],
        "interpretation": interpretation,
        "candidate_comparison": {
            candidate_id: {
                "baseline": _adjacency_summary(
                    baseline_candidates.get(candidate_id)
                ),
                "current": _adjacency_summary(
                    current_candidates.get(candidate_id)
                ),
            }
            for candidate_id in sorted(
                set(baseline_candidates).union(current_candidates)
            )
        },
    }


def compare_reports(baseline: dict, current: dict) -> dict:
    """Return a durable comparison for cases present in both exports."""

    for field in ("suite", "brightness_offset", "center_tolerance_px"):
        if baseline.get(field) != current.get(field):
            raise ValueError(
                f"baseline and current {field} do not match"
            )

    baseline_cases = {
        case["case_id"]: case for case in baseline["cases"]
    }
    current_cases = {
        case["case_id"]: case for case in current["cases"]
    }
    if set(baseline_cases) != set(current_cases):
        raise ValueError("baseline and current case ids do not match")
    for case_id in baseline_cases:
        baseline_case = baseline_cases[case_id]
        current_case = current_cases[case_id]
        for field in ("image_path", "click", "ground_truth"):
            if baseline_case[field] != current_case[field]:
                raise ValueError(
                    f"{case_id} baseline and current {field} do not match"
                )
    comparisons = [
        compare_case(baseline_cases[case_id], current_cases[case_id])
        for case_id in sorted(current_cases)
    ]
    return {
        "schema_version": 1,
        "diagnostics_schema_revision": "adjacency_diagnostics_v1",
        "baseline": {
            "git_commit": baseline["git_commit"],
            "git_dirty": baseline["git_dirty"],
        },
        "current": {
            "git_commit": current["git_commit"],
            "git_dirty": current["git_dirty"],
        },
        "case_count": len(comparisons),
        "correct_to_safe_failure_count": sum(
            row["baseline_formal_success"]
            and row["baseline_winner_is_correct"]
            and row["current_is_safe_failure"]
            for row in comparisons
        ),
        "cases": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))
    report = compare_reports(baseline, current)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case_count": report["case_count"],
                "correct_to_safe_failure_count": report[
                    "correct_to_safe_failure_count"
                ],
                "interpretations": {
                    case["case_id"]: case["interpretation"]
                    for case in report["cases"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
