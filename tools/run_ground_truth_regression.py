"""Run the local interactive pipeline against confirmed review cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import CONFIG, INTERACTIVE_CONFIG  # noqa: E402
from ground_truth_evaluation import (  # noqa: E402
    evaluate_ground_truth_predictions,
)
from interactive_pipeline import run_interactive_case  # noqa: E402
from pitch_reference import build_pitch_reference_map  # noqa: E402


def _candidate_shadow_rows(
    case: dict,
    report: dict,
    tolerance: float,
) -> list[dict]:
    """Return candidate topology rows with truth matching when available."""

    truth = case["ground_truth"]
    roi_x0 = report["roi"]["x0_global"]
    rows = []
    for candidate_id, candidate in report["candidate_arbitration"][
        "candidates"
    ].items():
        if not candidate.get("evaluated"):
            continue
        centers = candidate["selected_track_centers_roi"]
        left_roi = centers["left"]
        right_roi = centers["right"]
        left_global = (
            None if left_roi is None else roi_x0 + left_roi
        )
        right_global = (
            None if right_roi is None else roi_x0 + right_roi
        )
        matches_truth = False
        if (
            truth["valid_pair"]
            and candidate["quality_success"]
            and left_global is not None
            and right_global is not None
        ):
            matches_truth = (
                abs(left_global - truth["left_center_x"]) <= tolerance
                and abs(right_global - truth["right_center_x"])
                <= tolerance
            )
        topology = candidate["grayscale_topology"]
        rows.append(
            {
                "case_id": case["id"],
                "candidate_id": candidate_id,
                "quality_success": candidate["quality_success"],
                "matches_ground_truth": matches_truth,
                "status": topology["status"],
                "strong_same_basin_conflict": topology[
                    "strong_same_basin_conflict"
                ],
                "left_center_x_global": left_global,
                "right_center_x_global": right_global,
            }
        )
    return rows


def _shadow_summary(rows: list[dict], reports: list[dict]) -> dict:
    correct_rows = [row for row in rows if row["matches_ground_truth"]]
    false_conflicts = [
        row
        for row in correct_rows
        if row["strong_same_basin_conflict"]
    ]
    statuses = {}
    for row in correct_rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    changed = [
        report["image_name"] + ":"
        + str(report["click"]["x_global"])
        + ","
        + str(report["click"]["y_global"])
        for report in reports
        if report["shadow_arbitration"]["would_change_formal_result"]
    ]
    return {
        "mode": "shadow",
        "formal_results_changed": False,
        "correct_candidate_count": len(correct_rows),
        "correct_candidate_status_counts": dict(sorted(statuses.items())),
        "correct_candidate_strong_conflict_count": len(false_conflicts),
        "correct_candidate_strong_conflicts": false_conflicts,
        "counterfactual_change_count": len(changed),
        "counterfactual_change_cases": changed,
    }


def run_regression(truth: dict, output_dir: Path) -> dict:
    """Evaluate all cases while reusing one pitch map per source image."""

    images = {}
    pitch_maps = {}
    predictions = {}
    shadow_rows = []
    reports = []
    for case in truth["cases"]:
        relative_image_path = case["image_path"]
        if relative_image_path not in images:
            image = cv2.imread(
                str(PROJECT_ROOT / relative_image_path),
                cv2.IMREAD_GRAYSCALE,
            )
            if image is None:
                raise FileNotFoundError(relative_image_path)
            images[relative_image_path] = image
            pitch_maps[relative_image_path] = build_pitch_reference_map(
                image,
                CONFIG,
                INTERACTIVE_CONFIG,
            )
        result = run_interactive_case(
            images[relative_image_path],
            case["click"]["x"],
            case["click"]["y"],
            output_dir / "cases" / case["id"],
            CONFIG,
            INTERACTIVE_CONFIG,
            image_name=relative_image_path,
            pitch_reference_map=pitch_maps[relative_image_path],
        )
        interactive = result.report["interactive_result"]
        reports.append(result.report)
        shadow_rows.extend(
            _candidate_shadow_rows(
                case,
                result.report,
                float(truth["center_tolerance_px"]),
            )
        )
        rotation = result.report["rotation_shadow"]
        left = interactive["left"]
        right = interactive["right"]
        predictions[case["id"]] = {
            "success": interactive["success"],
            "click_classification": result.report["click"][
                "classification"
            ],
            "clicked_center_x": (
                None
                if interactive.get("clicked") is None
                else interactive["clicked"]["center_x_global"]
            ),
            "left_center_x": (
                None if left is None else left["center_x_global"]
            ),
            "right_center_x": (
                None if right is None else right["center_x_global"]
            ),
            "threshold_method": interactive.get(
                "threshold_method",
                "otsu",
            ),
            "geometry": interactive.get(
                "geometry",
                rotation["detection_space"],
            ),
            "pitch_status": interactive["pitch_guard"]["status"],
        }
    metrics = evaluate_ground_truth_predictions(truth, predictions)
    return {
        "metrics": metrics,
        "shadow_metrics": _shadow_summary(shadow_rows, reports),
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--truth",
        type=Path,
        default=(
            PROJECT_ROOT
            / "tests"
            / "data"
            / "stripe_09_10_ground_truth.json"
        ),
    )
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    output_dir = (
        PROJECT_ROOT / "outputs" / "ground_truth_regression" / args.label
    )
    result = run_regression(truth, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "regression_report.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["metrics"], indent=2))
    print(json.dumps(result["shadow_metrics"], indent=2))


if __name__ == "__main__":
    main()
