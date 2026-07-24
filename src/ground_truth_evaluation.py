"""Validation and metrics for manually reviewed nearest-neighbor cases."""

from collections import Counter
import math

import numpy as np


GROUND_TRUTH_SCHEMA_VERSION = 1


def validate_ground_truth_document(
    document: dict,
    require_confirmed: bool = False,
) -> None:
    """Validate the review document without treating suggestions as truth."""

    if document.get("schema_version") != GROUND_TRUTH_SCHEMA_VERSION:
        raise ValueError("unsupported ground-truth schema_version")
    tolerance = document.get("center_tolerance_px")
    if not isinstance(tolerance, (int, float)) or tolerance <= 0:
        raise ValueError("center_tolerance_px must be greater than 0")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("ground-truth document must contain cases")

    seen_ids = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("every case must have a non-empty id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate ground-truth case id: {case_id}")
        seen_ids.add(case_id)
        click = case.get("click")
        if not isinstance(click, dict):
            raise ValueError(f"{case_id}: click must be an object")
        for coordinate in ("x", "y"):
            if not isinstance(click.get(coordinate), int):
                raise ValueError(f"{case_id}: click {coordinate} must be int")
        truth = case.get("ground_truth")
        if not isinstance(truth, dict):
            raise ValueError(f"{case_id}: ground_truth must be an object")
        review_status = truth.get("review_status")
        if review_status not in ("pending", "confirmed"):
            raise ValueError(f"{case_id}: invalid review_status")
        if require_confirmed and review_status != "confirmed":
            raise ValueError(f"{case_id}: ground truth is not confirmed")
        if review_status != "confirmed":
            continue
        valid_pair = truth.get("valid_pair")
        if not isinstance(valid_pair, bool):
            raise ValueError(f"{case_id}: valid_pair must be boolean")
        if valid_pair:
            for field in ("left_center_x", "right_center_x"):
                value = truth.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{case_id}: {field} must be finite")
            if truth["left_center_x"] >= truth["right_center_x"]:
                raise ValueError(
                    f"{case_id}: left_center_x must be below right_center_x"
                )


def _percentile(values: list[float], percentile: float):
    if not values:
        return None
    return round(float(np.percentile(values, percentile)), 3)


def evaluate_ground_truth_predictions(
    document: dict,
    predictions: dict[str, dict],
) -> dict:
    """Return safety-focused metrics for confirmed cases and predictions."""

    validate_ground_truth_document(document, require_confirmed=True)
    tolerance = float(document["center_tolerance_px"])
    cases = document["cases"]
    valid_case_count = sum(
        case["ground_truth"]["valid_pair"] for case in cases
    )
    success_count = 0
    correct_count = 0
    false_success_count = 0
    suspicious_count = 0
    classification_case_count = 0
    classification_correct_count = 0
    clicked_center_errors = []
    left_errors = []
    right_errors = []
    threshold_methods = Counter()
    geometries = Counter()
    rows = []

    for case in cases:
        case_id = case["id"]
        if case_id not in predictions:
            raise ValueError(f"missing prediction for case: {case_id}")
        prediction = predictions[case_id]
        truth = case["ground_truth"]
        success = bool(prediction.get("success"))
        success_count += success
        if success:
            threshold_methods[prediction.get("threshold_method", "unknown")] += 1
            geometries[prediction.get("geometry", "unknown")] += 1
            suspicious_count += (
                prediction.get("pitch_status") == "Suspicious"
            )

        left_error = None
        right_error = None
        expected_classification = truth.get("click_classification")
        classification_correct = None
        if expected_classification is not None:
            classification_case_count += 1
            classification_correct = (
                prediction.get("click_classification")
                == expected_classification
            )
            classification_correct_count += classification_correct
        clicked_center_error = None
        expected_clicked_center = truth.get("clicked_center_x")
        predicted_clicked_center = prediction.get("clicked_center_x")
        if (
            expected_clicked_center is not None
            and isinstance(predicted_clicked_center, (int, float))
        ):
            clicked_center_error = abs(
                float(predicted_clicked_center)
                - float(expected_clicked_center)
            )
            clicked_center_errors.append(clicked_center_error)
        correct = False
        if success and truth["valid_pair"]:
            left = prediction.get("left_center_x")
            right = prediction.get("right_center_x")
            if isinstance(left, (int, float)) and isinstance(
                right, (int, float)
            ):
                left_error = abs(float(left) - truth["left_center_x"])
                right_error = abs(float(right) - truth["right_center_x"])
                left_errors.append(left_error)
                right_errors.append(right_error)
                correct = (
                    left_error <= tolerance and right_error <= tolerance
                )
        correct_count += correct
        false_success = success and (
            not truth["valid_pair"] or not correct
        )
        false_success_count += false_success
        rows.append(
            {
                "id": case_id,
                "success": success,
                "correct_nearest_neighbors": correct,
                "false_success": false_success,
                "left_error_px": (
                    None if left_error is None else round(left_error, 3)
                ),
                "right_error_px": (
                    None if right_error is None else round(right_error, 3)
                ),
                "click_classification_correct": classification_correct,
                "clicked_center_error_px": (
                    None
                    if clicked_center_error is None
                    else round(clicked_center_error, 3)
                ),
                "threshold_method": prediction.get("threshold_method"),
                "geometry": prediction.get("geometry"),
                "pitch_status": prediction.get("pitch_status"),
            }
        )

    total = len(cases)
    threshold_method_rates = {
        method: round(count / max(success_count, 1), 6)
        for method, count in sorted(threshold_methods.items())
    }
    geometry_rates = {
        geometry: round(count / max(success_count, 1), 6)
        for geometry, count in sorted(geometries.items())
    }
    screenshot_wrong_success_count = sum(
        row["false_success"]
        for row, case in zip(rows, cases)
        if case.get("screenshot_point")
    )
    correct_rate = correct_count / max(valid_case_count, 1)
    false_success_rate = false_success_count / total
    left_mae = (
        None if not left_errors else float(np.mean(left_errors))
    )
    right_mae = (
        None if not right_errors else float(np.mean(right_errors))
    )
    gates_passed = (
        false_success_rate <= 0.05
        and correct_rate >= 0.90
        and left_mae is not None
        and left_mae <= 3.0
        and right_mae is not None
        and right_mae <= 3.0
        and screenshot_wrong_success_count == 0
    )
    return {
        "case_count": total,
        "valid_pair_case_count": valid_case_count,
        "detection_success_rate": round(success_count / total, 6),
        "correct_nearest_neighbor_rate": round(correct_rate, 6),
        "false_success_rate": round(false_success_rate, 6),
        "error_among_success_rate": round(
            false_success_count / max(success_count, 1),
            6,
        ),
        "suspicious_among_success_rate": round(
            suspicious_count / max(success_count, 1),
            6,
        ),
        "click_classification_accuracy": (
            None
            if classification_case_count == 0
            else round(
                classification_correct_count / classification_case_count,
                6,
            )
        ),
        "clicked_center_error_px": {
            "mae": (
                None
                if not clicked_center_errors
                else round(float(np.mean(clicked_center_errors)), 3)
            ),
            "max": (
                None
                if not clicked_center_errors
                else round(max(clicked_center_errors), 3)
            ),
            "p95": _percentile(clicked_center_errors, 95),
        },
        "left_center_error_px": {
            "mae": (
                None
                if not left_errors
                else round(float(np.mean(left_errors)), 3)
            ),
            "max": None if not left_errors else round(max(left_errors), 3),
            "p95": _percentile(left_errors, 95),
        },
        "right_center_error_px": {
            "mae": (
                None
                if not right_errors
                else round(float(np.mean(right_errors)), 3)
            ),
            "max": None if not right_errors else round(max(right_errors), 3),
            "p95": _percentile(right_errors, 95),
        },
        "threshold_method_counts": dict(sorted(threshold_methods.items())),
        "threshold_method_rates": threshold_method_rates,
        "geometry_counts": dict(sorted(geometries.items())),
        "geometry_rates": geometry_rates,
        "screenshot_wrong_success_count": screenshot_wrong_success_count,
        "quality_gate": {
            "false_success_rate_at_most": 0.05,
            "correct_nearest_neighbor_rate_at_least": 0.90,
            "left_mae_at_most_px": 3.0,
            "right_mae_at_most_px": 3.0,
            "screenshot_wrong_success_count_at_most": 0,
            "passed": gates_passed,
        },
        "rows": rows,
    }
