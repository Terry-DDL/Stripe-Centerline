"""Evaluate reviewed method choices against topology shadow evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "topology_shadow_trial"
    / "trial_manifest.json"
)
DEFAULT_DECISIONS = (
    PROJECT_ROOT
    / "tests"
    / "data"
    / "topology_shadow_trial_review_decisions.json"
)


def _correct_methods(choice: str) -> set[str]:
    choices = {
        "both": {"otsu", "adaptive"},
        "otsu": {"otsu"},
        "adaptive": {"adaptive"},
        "neither": set(),
    }
    if choice not in choices:
        raise ValueError(f"unsupported review choice: {choice}")
    return choices[choice]


def _method_from_identity(identity: dict | None) -> str | None:
    if identity is None:
        return None
    return identity.get("threshold_method")


def evaluate_trial(manifest: dict, decisions_document: dict) -> dict:
    """Return shadow precision and counterfactual outcome counts."""

    cases = (
        manifest["known_error_cases"]
        + manifest["pending_review_cases"]
    )
    decisions = decisions_document.get("decisions", {})
    case_ids = {case["id"] for case in cases}
    decision_ids = set(decisions)
    if decision_ids != case_ids:
        missing = sorted(case_ids - decision_ids)
        extra = sorted(decision_ids - case_ids)
        raise ValueError(
            f"review decision ids do not match cases; "
            f"missing={missing}, extra={extra}"
        )

    status_counts = Counter()
    correct_status_counts = Counter()
    strong_conflicts = []
    correct_strong_conflicts = []
    known_error_conflicts = []
    counterfactual_rows = []
    correct_candidate_count = 0

    known_error_ids = {
        case["id"] for case in manifest["known_error_cases"]
    }
    for case in cases:
        case_id = case["id"]
        correct_methods = _correct_methods(
            decisions[case_id]["choice"]
        )
        for method, candidate in case["original_candidates"].items():
            topology = candidate["grayscale_topology"]
            status_counts[(method, topology["status"])] += 1
            is_correct = method in correct_methods
            if is_correct:
                correct_candidate_count += 1
                correct_status_counts[topology["status"]] += 1
            strong_topology_conflict = bool(
                topology["strong_same_basin_conflict"]
                or topology.get(
                    "strong_merged_basin_conflict",
                    False,
                )
            )
            if strong_topology_conflict:
                row = {
                    "case_id": case_id,
                    "method": method,
                    "reviewed_correct": is_correct,
                    "status": topology["status"],
                }
                strong_conflicts.append(row)
                if is_correct:
                    correct_strong_conflicts.append(row)
                if case_id in known_error_ids:
                    known_error_conflicts.append(case_id)

        shadow = case["shadow_arbitration"]
        if not shadow["would_change_formal_result"]:
            continue
        current_method = _method_from_identity(
            shadow["current_winner"]
        )
        hypothetical_method = _method_from_identity(
            shadow["hypothetical_winner"]
        )
        current_correct = current_method in correct_methods
        hypothetical_correct = (
            hypothetical_method in correct_methods
        )
        if not current_correct and hypothetical_correct:
            outcome = "fixes_error"
        elif current_correct and not hypothetical_correct:
            outcome = "introduces_regression"
        elif current_correct and hypothetical_correct:
            outcome = "changes_between_correct_methods"
        else:
            outcome = "remains_error"
        counterfactual_rows.append(
            {
                "case_id": case_id,
                "current_method": current_method,
                "hypothetical_method": hypothetical_method,
                "outcome": outcome,
                "hypothetical_success": shadow[
                    "hypothetical_success"
                ],
            }
        )

    strong_conflict_count = len(strong_conflicts)
    incorrect_strong_conflict_count = (
        strong_conflict_count - len(correct_strong_conflicts)
    )
    counterfactual_counts = Counter(
        row["outcome"] for row in counterfactual_rows
    )
    known_captured = len(set(known_error_conflicts))
    detector_gate_passed = (
        known_captured == len(known_error_ids)
        and not correct_strong_conflicts
    )
    activation_gate_passed = (
        detector_gate_passed
        and counterfactual_counts["introduces_regression"] == 0
        and counterfactual_counts["remains_error"] == 0
    )
    return {
        "mode": "shadow",
        "reviewed_case_count": len(cases),
        "correct_candidate_count": correct_candidate_count,
        "candidate_status_counts": {
            f"{method}:{status}": count
            for (method, status), count in sorted(status_counts.items())
        },
        "correct_candidate_status_counts": dict(
            sorted(correct_status_counts.items())
        ),
        "strong_conflict_count": strong_conflict_count,
        "strong_conflict_on_correct_candidate_count": len(
            correct_strong_conflicts
        ),
        "strong_conflict_precision": (
            None
            if strong_conflict_count == 0
            else round(
                incorrect_strong_conflict_count
                / strong_conflict_count,
                6,
            )
        ),
        "strong_conflicts": strong_conflicts,
        "known_error_capture": {
            "captured": known_captured,
            "total": len(known_error_ids),
            "rate": round(
                known_captured / max(len(known_error_ids), 1),
                6,
            ),
        },
        "counterfactual_change_count": len(counterfactual_rows),
        "counterfactual_outcome_counts": dict(
            sorted(counterfactual_counts.items())
        ),
        "counterfactual_rows": counterfactual_rows,
        "shadow_detector_gate_passed": detector_gate_passed,
        "rejection_activation_gate_passed": activation_gate_passed,
        "activation_blockers": (
            []
            if activation_gate_passed
            else [
                "A rejected Adaptive candidate would still fall back "
                "to an incorrect Otsu candidate in one or more cases."
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "topology_shadow_trial"
            / "review_report.json"
        ),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    report = evaluate_trial(manifest, decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
