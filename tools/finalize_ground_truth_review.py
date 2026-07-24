"""Turn reviewed A/O/B/N/X decisions into fixed ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ground_truth_evaluation import validate_ground_truth_document  # noqa: E402


def _mean_present(*values):
    finite_values = [float(value) for value in values if value is not None]
    if not finite_values:
        return None
    return round(sum(finite_values) / len(finite_values), 3)


def _truth_from_candidate(candidate: dict, notes: str) -> dict:
    successful = bool(candidate["success"])
    return {
        "review_status": "confirmed",
        "valid_pair": successful,
        "click_classification": candidate.get("classification"),
        "clicked_center_x": (
            candidate.get("clicked_center_x") if successful else None
        ),
        "left_center_x": (
            candidate.get("left_center_x") if successful else None
        ),
        "right_center_x": (
            candidate.get("right_center_x") if successful else None
        ),
        "notes": notes,
    }


def _truth_from_both(case: dict, notes: str) -> dict:
    otsu = case["candidates"]["otsu"]
    adaptive = case["candidates"]["adaptive"]
    if not otsu["success"] and not adaptive["success"]:
        return {
            "review_status": "confirmed",
            "valid_pair": False,
            "click_classification": None,
            "clicked_center_x": None,
            "left_center_x": None,
            "right_center_x": None,
            "notes": notes,
        }
    if not otsu["success"] or not adaptive["success"]:
        raise ValueError(
            f"{case['id']}: 'both' requires both methods to agree on success"
        )
    classifications = {
        otsu.get("classification"),
        adaptive.get("classification"),
    }
    classification = (
        classifications.pop() if len(classifications) == 1 else None
    )
    return {
        "review_status": "confirmed",
        "valid_pair": True,
        "click_classification": classification,
        "clicked_center_x": _mean_present(
            otsu.get("clicked_center_x"),
            adaptive.get("clicked_center_x"),
        ),
        "left_center_x": _mean_present(
            otsu.get("left_center_x"),
            adaptive.get("left_center_x"),
        ),
        "right_center_x": _mean_present(
            otsu.get("right_center_x"),
            adaptive.get("right_center_x"),
        ),
        "notes": notes,
    }


def _truth_from_override(decision: dict) -> dict:
    truth = {
        "review_status": "confirmed",
        "valid_pair": bool(decision["valid_pair"]),
        "click_classification": decision.get("click_classification"),
        "clicked_center_x": decision.get("clicked_center_x"),
        "left_center_x": decision.get("left_center_x"),
        "right_center_x": decision.get("right_center_x"),
        "notes": decision.get("notes", ""),
    }
    for optional_field in (
        "expected_partial_left_center_x",
        "expected_partial_right_center_x",
    ):
        if optional_field in decision:
            truth[optional_field] = decision[optional_field]
    return truth


def finalize_ground_truth(review: dict, decisions_document: dict) -> dict:
    """Apply every human decision and return a compact confirmed document."""

    decisions = decisions_document.get("decisions")
    if not isinstance(decisions, dict):
        raise ValueError("decisions document must contain a decisions object")
    review_cases = {case["id"]: case for case in review["cases"]}
    if set(decisions) != set(review_cases):
        missing = sorted(set(review_cases) - set(decisions))
        extra = sorted(set(decisions) - set(review_cases))
        raise ValueError(f"decision ids differ; missing={missing}, extra={extra}")

    confirmed_cases = []
    for case in review["cases"]:
        decision = decisions[case["id"]]
        choice = decision["choice"]
        notes = decision.get("notes", "")
        if choice == "both":
            truth = _truth_from_both(case, notes)
        elif choice in ("otsu", "adaptive"):
            truth = _truth_from_candidate(
                case["candidates"][choice],
                notes,
            )
        elif choice == "none":
            truth = {
                "review_status": "confirmed",
                "valid_pair": False,
                "click_classification": decision.get(
                    "click_classification"
                ),
                "clicked_center_x": None,
                "left_center_x": None,
                "right_center_x": None,
                "notes": notes,
            }
        elif choice == "override":
            truth = _truth_from_override(decision)
        else:
            raise ValueError(f"{case['id']}: unsupported choice {choice}")
        confirmed_cases.append(
            {
                "id": case["id"],
                "stripe": case["stripe"],
                "category": case["category"],
                "screenshot_point": case["screenshot_point"],
                "image_path": case["image_path"],
                "click": case["click"],
                "review_image": case["review_image"],
                "reviewed_choice": choice,
                "ground_truth": truth,
            }
        )

    document = {
        "schema_version": review["schema_version"],
        "review_status": "confirmed",
        "center_tolerance_px": review["center_tolerance_px"],
        "case_count": len(confirmed_cases),
        "reviewer_note": decisions_document.get("reviewer_note", ""),
        "cases": confirmed_cases,
    }
    validate_ground_truth_document(document, require_confirmed=True)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "ground_truth_review"
            / "review_cases.json"
        ),
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=(
            PROJECT_ROOT
            / "tests"
            / "data"
            / "stripe_09_10_review_decisions.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "tests"
            / "data"
            / "stripe_09_10_ground_truth.json"
        ),
    )
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    document = finalize_ground_truth(review, decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {document['case_count']} confirmed cases to {args.output}")


if __name__ == "__main__":
    main()
