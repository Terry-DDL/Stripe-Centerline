"""Run the phase-one Sample 2 adjacency safety acceptance scan."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import CONFIG, INTERACTIVE_CONFIG  # noqa: E402
import interactive_pipeline  # noqa: E402
from pitch_reference import build_pitch_reference_map  # noqa: E402


ANCHORS = (
    {
        "id": "sample2_662_1046",
        "click": (662, 1046),
        "left_center_x": 623.5,
        "right_center_x": 701.0,
    },
    {
        "id": "sample2_662_1065",
        "click": (662, 1065),
        "left_center_x": 623.5,
        "right_center_x": 701.0,
    },
    {
        "id": "sample2_701_996",
        "click": (701, 996),
        "left_center_x": 662.5,
        "right_center_x": 738.5,
    },
)
BRIGHTNESS_SHIFTS = (0, 50, 100, 150)
CENTER_TOLERANCE_PX = 3.0
CLICKED_HYPOTHESIS_DECISION_OUTCOMES = {
    "trigger_not_met": "trigger_not_met",
    "no_eligible_alternate": "no_verified",
    "no_verified_alternate": "no_verified",
    "unique_verified_adopted": "adopted",
    "ambiguous_verified_clicked_hypotheses": "ambiguous",
}


def classify_formal_result(
    interactive_result: dict,
    expected_left: float,
    expected_right: float,
    tolerance: float,
) -> str:
    """Classify one formal result without treating failures as successes."""

    if interactive_result["success"]:
        left = interactive_result.get("left")
        right = interactive_result.get("right")
        if left is not None and right is not None:
            left_error = abs(
                float(left["center_x_global"]) - expected_left
            )
            right_error = abs(
                float(right["center_x_global"]) - expected_right
            )
            if left_error <= tolerance and right_error <= tolerance:
                return "correct_success"
        return "wrong_success"

    hidden_fields = (
        interactive_result.get("left"),
        interactive_result.get("clicked"),
        interactive_result.get("right"),
        interactive_result.get("stripe_spacing_px"),
    )
    if all(value is None for value in hidden_fields):
        return "safe_failure"
    return "contract_violation"


def summarize_clicked_hypothesis_arbitration(report: dict) -> dict:
    """Normalize the original-Otsu clicked-hypothesis decision for scans."""

    arbitration = (
        report.get("candidate_arbitration", {})
        .get("candidates", {})
        .get("original_otsu", {})
        .get("adjacency_verification", {})
        .get("clicked_hypothesis_arbitration")
    )
    if not isinstance(arbitration, dict):
        return {
            "attempted": False,
            "decision": "trigger_not_met",
            "outcome": "trigger_not_met",
            "legacy_missing_field": True,
        }

    decision = arbitration.get("decision")
    if decision not in CLICKED_HYPOTHESIS_DECISION_OUTCOMES:
        raise ValueError(
            "unknown clicked hypothesis arbitration decision: "
            f"{decision!r}"
        )
    attempted = arbitration.get("attempted")
    if not isinstance(attempted, bool):
        raise ValueError(
            "clicked hypothesis arbitration attempted must be boolean"
        )
    return {
        "attempted": attempted,
        "decision": decision,
        "outcome": CLICKED_HYPOTHESIS_DECISION_OUTCOMES[decision],
        "legacy_missing_field": False,
    }


def _brighten(image_gray: np.ndarray, shift: int) -> np.ndarray:
    return np.clip(
        image_gray.astype(np.int16) + shift,
        0,
        255,
    ).astype(np.uint8)


def clipped_pixel_ratio(image_gray: np.ndarray, shift: int) -> float:
    """Return the fraction of source pixels clipped by a brightness shift."""

    if shift < 0:
        raise ValueError("brightness shift must not be negative")
    clipped = image_gray.astype(np.int16) + shift > 255
    return float(np.count_nonzero(clipped) / clipped.size)


def git_provenance(project_root: Path = PROJECT_ROOT) -> dict:
    """Return the exact Git revision and dirty state used for a scan."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "git_commit": commit,
        "git_dirty": bool(status.strip()),
    }


def build_coordinate_plan(
    anchors: tuple[dict, ...] = ANCHORS,
    y_step: int = 1,
) -> tuple[dict, ...]:
    """Build one unique coordinate list and retain anchor membership."""

    if y_step <= 0:
        raise ValueError("y_step must be greater than zero")

    coordinates = {}
    for anchor in anchors:
        click_x, click_y = anchor["click"]
        expected_pair = (
            float(anchor["left_center_x"]),
            float(anchor["right_center_x"]),
        )
        for x_global in range(click_x - 2, click_x + 3):
            for y_global in range(
                click_y - 25,
                click_y + 26,
                y_step,
            ):
                coordinate = (x_global, y_global)
                point = coordinates.get(coordinate)
                if point is None:
                    coordinates[coordinate] = {
                        "x_global": x_global,
                        "y_global": y_global,
                        "anchor_ids": [anchor["id"]],
                        "expected_left_center_x": expected_pair[0],
                        "expected_right_center_x": expected_pair[1],
                    }
                    continue

                existing_pair = (
                    point["expected_left_center_x"],
                    point["expected_right_center_x"],
                )
                if existing_pair != expected_pair:
                    raise ValueError(
                        "conflicting ground truth for coordinate "
                        f"{coordinate}: {existing_pair} vs {expected_pair}"
                    )
                point["anchor_ids"].append(anchor["id"])
    return tuple(coordinates.values())


def _point_summary(
    result,
    expected_left: float,
    expected_right: float,
    tolerance: float,
) -> dict:
    report = result.report
    interactive = report["interactive_result"]
    classification = classify_formal_result(
        interactive,
        expected_left,
        expected_right,
        tolerance,
    )
    clicked_hypothesis = summarize_clicked_hypothesis_arbitration(report)
    return {
        "classification": classification,
        "failure_reasons": interactive["failure_reasons"],
        "left_center_x": (
            None
            if interactive.get("left") is None
            else interactive["left"]["center_x_global"]
        ),
        "right_center_x": (
            None
            if interactive.get("right") is None
            else interactive["right"]["center_x_global"]
        ),
        "algorithm_revision": report["algorithm_revision"],
        "clicked_hypothesis_arbitration": clicked_hypothesis,
    }


def run_scan(
    image_gray: np.ndarray,
    y_step: int = 1,
    tolerance: float = CENTER_TOLERANCE_PX,
) -> dict:
    """Run all anchors and brightness variants with one map per image."""

    if y_step <= 0:
        raise ValueError("y_step must be greater than zero")

    coordinate_plan = build_coordinate_plan(y_step=y_step)
    brightness_results = []
    details = []
    revisions = set()
    for brightness_shift in BRIGHTNESS_SHIFTS:
        image_variant = _brighten(image_gray, brightness_shift)
        clipping_ratio = clipped_pixel_ratio(
            image_gray,
            brightness_shift,
        )
        pitch_map = build_pitch_reference_map(
            image_variant,
            CONFIG,
            INTERACTIVE_CONFIG,
        )
        aggregate = Counter()
        failure_reasons = Counter()
        clicked_hypothesis_counts = Counter()
        anchor_counts = {
            anchor["id"]: Counter() for anchor in ANCHORS
        }
        anchor_reasons = {
            anchor["id"]: Counter() for anchor in ANCHORS
        }
        for coordinate in coordinate_plan:
            x_global = coordinate["x_global"]
            y_global = coordinate["y_global"]
            result = interactive_pipeline.run_interactive_case(
                image_variant,
                x_global,
                y_global,
                PROJECT_ROOT / "outputs" / "_scan_scratch",
                CONFIG,
                INTERACTIVE_CONFIG,
                image_name="Sample 2.bmp",
                pitch_reference_map=pitch_map,
            )
            point = _point_summary(
                result,
                coordinate["expected_left_center_x"],
                coordinate["expected_right_center_x"],
                tolerance,
            )
            classification = point["classification"]
            aggregate[classification] += 1
            revisions.add(point["algorithm_revision"])
            clicked_hypothesis = point["clicked_hypothesis_arbitration"]
            clicked_hypothesis_counts["attempted"] += int(
                clicked_hypothesis["attempted"]
            )
            clicked_hypothesis_counts[
                clicked_hypothesis["outcome"]
            ] += 1
            clicked_hypothesis_counts["legacy_missing_field"] += int(
                clicked_hypothesis["legacy_missing_field"]
            )
            reason = None
            if classification == "safe_failure":
                reason = (
                    "|".join(point["failure_reasons"])
                    or "unspecified_failure"
                )
                failure_reasons[reason] += 1
            for anchor_id in coordinate["anchor_ids"]:
                anchor_counts[anchor_id][classification] += 1
                if reason is not None:
                    anchor_reasons[anchor_id][reason] += 1
            details.append(
                {
                    "brightness_offset": brightness_shift,
                    "brightness_shift": brightness_shift,
                    "clipped_pixel_ratio": clipping_ratio,
                    **coordinate,
                    **point,
                }
            )

        regions = {}
        for anchor in ANCHORS:
            counts = anchor_counts[anchor["id"]]
            reasons = anchor_reasons[anchor["id"]]
            regions[anchor["id"]] = {
                "point_count": sum(counts.values()),
                "correct_success": counts["correct_success"],
                "safe_failure": counts["safe_failure"],
                "wrong_success": counts["wrong_success"],
                "contract_violation": counts[
                    "contract_violation"
                ],
                "failure_reasons": dict(sorted(reasons.items())),
            }

        brightness_results.append(
            {
                "brightness_offset": brightness_shift,
                "brightness_shift": brightness_shift,
                "point_count": sum(aggregate.values()),
                "unique_point_count": len(coordinate_plan),
                "anchor_membership_count": sum(
                    region["point_count"] for region in regions.values()
                ),
                "clipped_pixel_ratio": clipping_ratio,
                "correct_success": aggregate["correct_success"],
                "safe_failure": aggregate["safe_failure"],
                "wrong_success": aggregate["wrong_success"],
                "contract_violation": aggregate["contract_violation"],
                "failure_reasons": dict(sorted(failure_reasons.items())),
                "clicked_hypothesis_arbitration_counts": {
                    "attempted": clicked_hypothesis_counts["attempted"],
                    "adopted": clicked_hypothesis_counts["adopted"],
                    "no_verified": clicked_hypothesis_counts["no_verified"],
                    "ambiguous": clicked_hypothesis_counts["ambiguous"],
                    "trigger_not_met": clicked_hypothesis_counts[
                        "trigger_not_met"
                    ],
                    "legacy_missing_field": clicked_hypothesis_counts[
                        "legacy_missing_field"
                    ],
                    "outcome_coordinate_count": sum(
                        clicked_hypothesis_counts[outcome]
                        for outcome in (
                            "adopted",
                            "no_verified",
                            "ambiguous",
                            "trigger_not_met",
                        )
                    ),
                    "missing_field_policy": (
                        "legacy_safe_counted_as_trigger_not_met"
                    ),
                },
                "regions": regions,
            }
        )

    provenance = git_provenance()
    return {
        **provenance,
        "algorithm_revisions": sorted(revisions),
        "image_name": "Sample 2.bmp",
        "center_tolerance_px": tolerance,
        "x_offsets": [-2, -1, 0, 1, 2],
        "y_offset_min": -25,
        "y_offset_max": 25,
        "y_step": y_step,
        "anchors": ANCHORS,
        "region_counts_are_non_additive": True,
        "unique_coordinate_count_per_brightness": len(coordinate_plan),
        "total_unique_evaluation_count": (
            len(coordinate_plan) * len(BRIGHTNESS_SHIFTS)
        ),
        "brightness_results": brightness_results,
        "details": details,
    }


def _write_outputs(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scan_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "scan_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        fieldnames = (
            "brightness_offset",
            "brightness_shift",
            "point_count",
            "unique_point_count",
            "anchor_membership_count",
            "clipped_pixel_ratio",
            "correct_success",
            "safe_failure",
            "wrong_success",
            "contract_violation",
            "failure_reasons",
            "clicked_hypothesis_arbitration_counts",
        )
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["brightness_results"]:
            writer.writerow(
                {
                    field: (
                        json.dumps(row[field], sort_keys=True)
                        if field
                        in (
                            "failure_reasons",
                            "clicked_hypothesis_arbitration_counts",
                        )
                        else row[field]
                    )
                    for field in fieldnames
                }
            )


def _empty_debug_image(*_args, **_kwargs) -> np.ndarray:
    """Skip expensive presentation-only images during a numeric scan."""

    return np.zeros((1, 1), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=PROJECT_ROOT / "images" / "Sample 2.bmp",
    )
    parser.add_argument("--label", default="clicked_hypothesis_v1")
    parser.add_argument("--y-step", type=int, default=1)
    args = parser.parse_args()

    image_gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(args.image)
    with patch.object(
        interactive_pipeline,
        "save_debug_image",
    ), patch.object(
        interactive_pipeline,
        "_save_report",
    ), patch.object(
        interactive_pipeline,
        "_create_original_overlay",
        side_effect=_empty_debug_image,
    ), patch.object(
        interactive_pipeline,
        "_create_interactive_roi_result",
        side_effect=_empty_debug_image,
    ), patch.object(
        interactive_pipeline,
        "_create_interactive_candidates_debug",
        side_effect=_empty_debug_image,
    ), patch.object(
        interactive_pipeline,
        "_create_interactive_votes_debug",
        side_effect=_empty_debug_image,
    ), patch.object(
        interactive_pipeline,
        "create_pitch_reference_debug",
        side_effect=_empty_debug_image,
    ):
        report = run_scan(image_gray, y_step=args.y_step)
    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "adjacency_safety_regression"
        / args.label
    )
    _write_outputs(report, output_dir)
    print(json.dumps(report["brightness_results"], indent=2))

    if any(
        row["wrong_success"] or row["contract_violation"]
        for row in report["brightness_results"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
