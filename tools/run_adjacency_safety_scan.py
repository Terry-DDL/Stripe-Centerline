"""Run the phase-one Sample 2 adjacency safety acceptance scan."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
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


def _brighten(image_gray: np.ndarray, shift: int) -> np.ndarray:
    return np.clip(
        image_gray.astype(np.int16) + shift,
        0,
        255,
    ).astype(np.uint8)


def _point_summary(
    result,
    anchor: dict,
    tolerance: float,
) -> dict:
    report = result.report
    interactive = report["interactive_result"]
    classification = classify_formal_result(
        interactive,
        anchor["left_center_x"],
        anchor["right_center_x"],
        tolerance,
    )
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
    }


def run_scan(
    image_gray: np.ndarray,
    y_step: int = 1,
    tolerance: float = CENTER_TOLERANCE_PX,
) -> dict:
    """Run all anchors and brightness variants with one map per image."""

    if y_step <= 0:
        raise ValueError("y_step must be greater than zero")

    brightness_results = []
    details = []
    revisions = set()
    for brightness_shift in BRIGHTNESS_SHIFTS:
        image_variant = _brighten(image_gray, brightness_shift)
        pitch_map = build_pitch_reference_map(
            image_variant,
            CONFIG,
            INTERACTIVE_CONFIG,
        )
        cached_points = {}
        aggregate = Counter()
        failure_reasons = Counter()
        regions = {}
        for anchor in ANCHORS:
            anchor_counts = Counter()
            anchor_reasons = Counter()
            click_x, click_y = anchor["click"]
            for x_global in range(click_x - 2, click_x + 3):
                for y_global in range(
                    click_y - 25,
                    click_y + 26,
                    y_step,
                ):
                    cache_key = (x_global, y_global)
                    point = cached_points.get(cache_key)
                    if point is None:
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
                            anchor,
                            tolerance,
                        )
                        cached_points[cache_key] = point
                    classification = point["classification"]
                    anchor_counts[classification] += 1
                    aggregate[classification] += 1
                    revisions.add(point["algorithm_revision"])
                    if classification == "safe_failure":
                        reason = (
                            "|".join(point["failure_reasons"])
                            or "unspecified_failure"
                        )
                        anchor_reasons[reason] += 1
                        failure_reasons[reason] += 1
                    details.append(
                        {
                            "brightness_shift": brightness_shift,
                            "anchor_id": anchor["id"],
                            "x_global": x_global,
                            "y_global": y_global,
                            **point,
                        }
                    )
            regions[anchor["id"]] = {
                "point_count": sum(anchor_counts.values()),
                "correct_success": anchor_counts["correct_success"],
                "safe_failure": anchor_counts["safe_failure"],
                "wrong_success": anchor_counts["wrong_success"],
                "contract_violation": anchor_counts[
                    "contract_violation"
                ],
                "failure_reasons": dict(sorted(anchor_reasons.items())),
            }

        brightness_results.append(
            {
                "brightness_shift": brightness_shift,
                "point_count": sum(aggregate.values()),
                "unique_point_count": len(cached_points),
                "correct_success": aggregate["correct_success"],
                "safe_failure": aggregate["safe_failure"],
                "wrong_success": aggregate["wrong_success"],
                "contract_violation": aggregate["contract_violation"],
                "failure_reasons": dict(sorted(failure_reasons.items())),
                "regions": regions,
            }
        )

    return {
        "algorithm_revisions": sorted(revisions),
        "image_name": "Sample 2.bmp",
        "center_tolerance_px": tolerance,
        "x_offsets": [-2, -1, 0, 1, 2],
        "y_offset_min": -25,
        "y_offset_max": 25,
        "y_step": y_step,
        "anchors": ANCHORS,
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
            "brightness_shift",
            "point_count",
            "unique_point_count",
            "correct_success",
            "safe_failure",
            "wrong_success",
            "contract_violation",
            "failure_reasons",
        )
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["brightness_results"]:
            writer.writerow(
                {
                    field: (
                        json.dumps(row[field], sort_keys=True)
                        if field == "failure_reasons"
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
    parser.add_argument("--label", default="adjacency_gate_v1")
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
