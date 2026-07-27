"""Original-grayscale evidence for high-confidence topology mistakes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from config import InteractiveConfig
from interactive_analysis import InteractiveStripeSelection
from stripe_analysis import StripeTrack


@dataclass(frozen=True)
class GrayscaleTopologyEvaluation:
    """JSON-safe evidence plus a hidden debug image."""

    report: dict
    debug_image: np.ndarray


def _empty_report(reason: str) -> dict:
    return {
        "mode": "shadow",
        "status": "Unable to verify",
        "strong_same_basin_conflict": False,
        "strong_merged_basin_conflict": False,
        "evaluated_intervals": [],
        "evaluated_tracks": [],
        "informative_row_count": 0,
        "separator_support_ratio": None,
        "same_basin_support_ratio": None,
        "vertical_band_coverage": {
            "informative_rows": [0, 0, 0],
            "same_basin_rows": [0, 0, 0],
            "covered_band_count": 0,
            "vertical_span_ratio": 0.0,
        },
        "reason": reason,
    }


def _selected_intervals(
    selection: InteractiveStripeSelection,
) -> list[tuple[str, StripeTrack, StripeTrack]]:
    left = selection.left_track
    right = selection.right_track
    if left is None or right is None:
        return []
    if (
        selection.click_classification == "black_stripe"
        and selection.clicked_track is not None
    ):
        return [
            ("left_to_clicked", left, selection.clicked_track),
            ("clicked_to_right", selection.clicked_track, right),
        ]
    return [("left_to_right", left, right)]


def _selected_tracks(
    selection: InteractiveStripeSelection,
) -> list[tuple[str, StripeTrack]]:
    tracks = []
    if selection.left_track is not None:
        tracks.append(("left", selection.left_track))
    if selection.right_track is not None:
        tracks.append(("right", selection.right_track))
    return tracks


def _map_point(
    inverse_rotation_matrix: np.ndarray,
    x_detection: float,
    y_detection: float,
) -> tuple[float, float]:
    return (
        float(
            inverse_rotation_matrix[0, 0] * x_detection
            + inverse_rotation_matrix[0, 1] * y_detection
            + inverse_rotation_matrix[0, 2]
        ),
        float(
            inverse_rotation_matrix[1, 0] * x_detection
            + inverse_rotation_matrix[1, 1] * y_detection
            + inverse_rotation_matrix[1, 2]
        ),
    )


def _sample_line_nearest(
    image_gray_roi: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
) -> np.ndarray:
    height, width = image_gray_roi.shape
    sample_count = max(
        2,
        int(
            math.ceil(
                max(abs(end[0] - start[0]), abs(end[1] - start[1]))
            )
        )
        + 1,
    )
    x_values = np.rint(
        np.linspace(start[0], end[0], sample_count)
    ).astype(np.int32)
    y_values = np.rint(
        np.linspace(start[1], end[1], sample_count)
    ).astype(np.int32)
    x_values = np.clip(x_values, 0, width - 1)
    y_values = np.clip(y_values, 0, height - 1)
    return image_gray_roi[y_values, x_values].astype(np.float64)


def _core_level(
    image_gray_roi: np.ndarray,
    run,
    inverse_rotation_matrix: np.ndarray,
) -> float:
    half_width = max(1.0, min(3.0, run.width_px / 4.0))
    values = []
    for offset in np.linspace(-half_width, half_width, 5):
        point = _map_point(
            inverse_rotation_matrix,
            run.center_x_roi + float(offset),
            run.y_roi,
        )
        values.append(
            _sample_line_nearest(image_gray_roi, point, point)[0]
        )
    return float(np.median(values))


def _rolling_mean_peak(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if values.size < 3:
        return float(np.max(values))
    return float(
        np.max(np.convolve(values, np.ones(3) / 3.0, mode="valid"))
    )


def _row_evidence(
    image_gray_roi: np.ndarray,
    left_run,
    right_run,
    inverse_rotation_matrix: np.ndarray,
    config: InteractiveConfig,
) -> dict | None:
    left_point = _map_point(
        inverse_rotation_matrix,
        left_run.center_x_roi,
        left_run.y_roi,
    )
    right_point = _map_point(
        inverse_rotation_matrix,
        right_run.center_x_roi,
        right_run.y_roi,
    )
    vector_x = right_point[0] - left_point[0]
    vector_y = right_point[1] - left_point[1]
    interval_length = math.hypot(vector_x, vector_y)
    if interval_length < 3.0:
        return None

    unit_x = vector_x / interval_length
    unit_y = vector_y / interval_length
    extension = interval_length * 0.75
    context_start = (
        left_point[0] - unit_x * extension,
        left_point[1] - unit_y * extension,
    )
    context_end = (
        right_point[0] + unit_x * extension,
        right_point[1] + unit_y * extension,
    )
    context = _sample_line_nearest(
        image_gray_roi,
        context_start,
        context_end,
    )
    low_level = float(np.percentile(context, 10))
    high_level = float(np.percentile(context, 90))
    dynamic_range = high_level - low_level
    if dynamic_range < config.topology_shadow_min_dynamic_range:
        return None

    left_core = _core_level(
        image_gray_roi,
        left_run,
        inverse_rotation_matrix,
    )
    right_core = _core_level(
        image_gray_roi,
        right_run,
        inverse_rotation_matrix,
    )
    darker_reference = max(left_core, right_core)
    normalized_left = (left_core - low_level) / dynamic_range
    normalized_right = (right_core - low_level) / dynamic_range
    if normalized_left > 0.5 or normalized_right > 0.5:
        return None

    interval = _sample_line_nearest(
        image_gray_roi,
        left_point,
        right_point,
    )
    edge_trim = max(
        1,
        min(
            interval.size // 4,
            int(
                round(
                    min(left_run.width_px, right_run.width_px) / 4.0
                )
            ),
        ),
    )
    interior = interval[edge_trim:-edge_trim]
    if interior.size == 0:
        return None
    separator_level = _rolling_mean_peak(interior)
    prominence = separator_level - darker_reference
    normalized_prominence = prominence / dynamic_range

    core_samples = np.concatenate(
        (
            interval[: max(1, edge_trim)],
            interval[-max(1, edge_trim) :],
        )
    )
    core_median = float(np.median(core_samples))
    core_mad = float(np.median(np.abs(core_samples - core_median)))
    noise_scale = min(
        max(core_mad * 1.4826, 1.0),
        dynamic_range * 0.25,
    )
    separator_threshold = max(
        config.topology_shadow_noise_mad_multiplier * noise_scale,
        config.topology_shadow_separator_prominence_ratio
        * dynamic_range,
    )
    same_basin_threshold = max(
        noise_scale,
        config.topology_shadow_same_basin_prominence_ratio
        * dynamic_range,
    )
    supports_separator = prominence >= separator_threshold
    supports_same_basin = prominence <= same_basin_threshold
    midpoint_y = (left_point[1] + right_point[1]) / 2.0
    return {
        "left_point": left_point,
        "right_point": right_point,
        "midpoint_y": midpoint_y,
        "dynamic_range": dynamic_range,
        "normalized_prominence": normalized_prominence,
        "supports_separator": supports_separator,
        "supports_same_basin": supports_same_basin,
    }


def _merged_basin_row_evidence(
    image_gray_roi: np.ndarray,
    run,
    inverse_rotation_matrix: np.ndarray,
    config: InteractiveConfig,
) -> dict | None:
    """Return strong evidence that one binary run contains two dark basins."""

    if run.width_px < 7:
        return None
    run_start = _map_point(
        inverse_rotation_matrix,
        float(run.x0_roi),
        float(run.y_roi),
    )
    run_end = _map_point(
        inverse_rotation_matrix,
        float(run.x1_roi - 1),
        float(run.y_roi),
    )
    run_length = math.hypot(
        run_end[0] - run_start[0],
        run_end[1] - run_start[1],
    )
    if run_length < 6.0:
        return None

    unit_x = (run_end[0] - run_start[0]) / run_length
    unit_y = (run_end[1] - run_start[1]) / run_length
    extension = run_length * 0.5
    context_start = (
        run_start[0] - unit_x * extension,
        run_start[1] - unit_y * extension,
    )
    context_end = (
        run_end[0] + unit_x * extension,
        run_end[1] + unit_y * extension,
    )
    height, width = image_gray_roi.shape
    if any(
        not (0.0 <= point[0] < width and 0.0 <= point[1] < height)
        for point in (run_start, run_end, context_start, context_end)
    ):
        return None
    context = _sample_line_nearest(
        image_gray_roi,
        context_start,
        context_end,
    )
    low_level = float(np.percentile(context, 10))
    high_level = float(np.percentile(context, 90))
    dynamic_range = high_level - low_level
    if dynamic_range < config.topology_shadow_min_dynamic_range:
        return None

    values = _sample_line_nearest(
        image_gray_roi,
        run_start,
        run_end,
    )
    if values.size < 7:
        return None
    smoothed = np.convolve(
        values,
        np.ones(3, dtype=np.float64) / 3.0,
        mode="same",
    )
    minimum_side_width = max(2, int(round(values.size * 0.18)))
    search_start = minimum_side_width
    search_end = values.size - minimum_side_width
    if search_end <= search_start:
        return None
    ridge_index = int(
        search_start
        + np.argmax(smoothed[search_start:search_end])
    )
    left_values = smoothed[1 : ridge_index - 1]
    right_values = smoothed[ridge_index + 2 : -1]
    if left_values.size < 2 or right_values.size < 2:
        return None

    left_dark = float(np.percentile(left_values, 20))
    right_dark = float(np.percentile(right_values, 20))
    darker_lobe_reference = max(left_dark, right_dark)
    ridge_level = float(smoothed[ridge_index])
    center_index = values.size // 2
    center_start = max(0, center_index - 1)
    center_end = min(values.size, center_index + 2)
    center_level = float(np.mean(values[center_start:center_end]))

    dark_samples = np.concatenate(
        (
            left_values[left_values <= np.percentile(left_values, 35)],
            right_values[
                right_values <= np.percentile(right_values, 35)
            ],
        )
    )
    dark_median = float(np.median(dark_samples))
    dark_mad = float(
        np.median(np.abs(dark_samples - dark_median))
    )
    noise_scale = min(
        max(dark_mad * 1.4826, 1.0),
        dynamic_range * 0.25,
    )
    prominence_threshold = max(
        config.topology_shadow_noise_mad_multiplier * noise_scale,
        config.topology_shadow_separator_prominence_ratio
        * dynamic_range,
    )
    ridge_prominence = ridge_level - darker_lobe_reference
    center_prominence = center_level - darker_lobe_reference
    normalized_left = (left_dark - low_level) / dynamic_range
    normalized_right = (right_dark - low_level) / dynamic_range
    supports_merged_basin = bool(
        normalized_left <= 0.5
        and normalized_right <= 0.5
        and ridge_prominence >= prominence_threshold
        and center_prominence >= prominence_threshold
    )
    ridge_fraction = ridge_index / max(values.size - 1, 1)
    ridge_point = (
        run_start[0] + (run_end[0] - run_start[0]) * ridge_fraction,
        run_start[1] + (run_end[1] - run_start[1]) * ridge_fraction,
    )
    return {
        "left_point": run_start,
        "right_point": run_end,
        "ridge_point": ridge_point,
        "midpoint_y": (run_start[1] + run_end[1]) / 2.0,
        "dynamic_range": dynamic_range,
        "normalized_prominence": (
            ridge_prominence / dynamic_range
        ),
        "normalized_center_prominence": (
            center_prominence / dynamic_range
        ),
        "supports_merged_basin": supports_merged_basin,
    }


def _vertical_coverage(
    rows: list[dict],
    roi_height: int,
) -> dict:
    informative_counts = [0, 0, 0]
    same_basin_counts = [0, 0, 0]
    y_values = []
    for row in rows:
        y_value = min(max(row["midpoint_y"], 0.0), roi_height - 1.0)
        band = min(2, int((y_value / max(roi_height, 1)) * 3))
        informative_counts[band] += 1
        same_basin_counts[band] += int(row["supports_same_basin"])
        y_values.append(y_value)
    span_ratio = 0.0
    if len(y_values) >= 2:
        span_ratio = (max(y_values) - min(y_values)) / max(
            roi_height - 1,
            1,
        )
    return {
        "informative_rows": informative_counts,
        "same_basin_rows": same_basin_counts,
        "covered_band_count": sum(count > 0 for count in informative_counts),
        "vertical_span_ratio": round(float(span_ratio), 6),
    }


def _vertical_merged_coverage(
    rows: list[dict],
    roi_height: int,
) -> dict:
    informative_counts = [0, 0, 0]
    merged_counts = [0, 0, 0]
    y_values = []
    for row in rows:
        y_value = min(max(row["midpoint_y"], 0.0), roi_height - 1.0)
        band = min(2, int((y_value / max(roi_height, 1)) * 3))
        informative_counts[band] += 1
        merged_counts[band] += int(row["supports_merged_basin"])
        y_values.append(y_value)
    span_ratio = 0.0
    if len(y_values) >= 2:
        span_ratio = (max(y_values) - min(y_values)) / max(
            roi_height - 1,
            1,
        )
    return {
        "informative_rows": informative_counts,
        "merged_basin_rows": merged_counts,
        "covered_band_count": sum(count > 0 for count in informative_counts),
        "vertical_span_ratio": round(float(span_ratio), 6),
    }


def _evaluate_selected_track(
    label: str,
    track: StripeTrack,
    comparison_widths: list[float],
    image_gray_original_roi: np.ndarray,
    inverse_rotation_matrix: np.ndarray,
    config: InteractiveConfig,
) -> tuple[dict, list[dict]]:
    typical_width = (
        None
        if not comparison_widths
        else float(np.median(comparison_widths))
    )
    width_ratio = (
        None
        if typical_width is None or typical_width <= 0.0
        else track.median_width_px / typical_width
    )
    row_evidence = []
    for run in track.valid_runs:
        evidence = _merged_basin_row_evidence(
            image_gray_original_roi,
            run,
            inverse_rotation_matrix,
            config,
        )
        if evidence is not None:
            row_evidence.append(evidence)

    informative_count = len(row_evidence)
    merged_count = sum(
        row["supports_merged_basin"] for row in row_evidence
    )
    merged_ratio = (
        None
        if informative_count == 0
        else merged_count / informative_count
    )
    coverage = _vertical_merged_coverage(
        row_evidence,
        image_gray_original_roi.shape[0],
    )
    enough_rows = (
        informative_count
        >= config.topology_shadow_min_informative_rows
    )
    broad_coverage = (
        coverage["covered_band_count"] == 3
        and coverage["vertical_span_ratio"]
        >= config.topology_shadow_min_vertical_span_ratio
    )
    strong_conflict = bool(
        enough_rows
        and broad_coverage
        and merged_ratio is not None
        and merged_ratio
        >= config.topology_shadow_strong_merged_basin_support_ratio
        and all(
            count > 0 for count in coverage["merged_basin_rows"]
        )
    )
    if strong_conflict:
        status = "Contradictory"
        reason = "merged_dark_basins"
    else:
        status = "Unable to verify"
        reason = "no_strong_merged_basin_evidence"
    prominence_values = [
        row["normalized_prominence"] for row in row_evidence
    ]
    center_prominence_values = [
        row["normalized_center_prominence"] for row in row_evidence
    ]
    report = {
        "label": label,
        "track_id": track.track_id,
        "status": status,
        "strong_merged_basin_conflict": strong_conflict,
        "median_width_px": round(track.median_width_px, 6),
        "comparison_width_px": (
            None if typical_width is None else round(typical_width, 6)
        ),
        "width_ratio": (
            None if width_ratio is None else round(width_ratio, 6)
        ),
        "informative_row_count": informative_count,
        "merged_basin_support_ratio": (
            None
            if merged_ratio is None
            else round(float(merged_ratio), 6)
        ),
        "normalized_prominence": (
            None
            if not prominence_values
            else round(float(np.median(prominence_values)), 6)
        ),
        "normalized_center_prominence": (
            None
            if not center_prominence_values
            else round(
                float(np.median(center_prominence_values)),
                6,
            )
        ),
        "vertical_band_coverage": coverage,
        "reason": reason,
    }
    return report, row_evidence


def _evaluate_interval(
    label: str,
    left_track: StripeTrack,
    right_track: StripeTrack,
    image_gray_original_roi: np.ndarray,
    inverse_rotation_matrix: np.ndarray,
    config: InteractiveConfig,
) -> tuple[dict, list[dict]]:
    left_runs = {run.y_roi: run for run in left_track.valid_runs}
    right_runs = {run.y_roi: run for run in right_track.valid_runs}
    row_evidence = []
    for y_roi in sorted(set(left_runs) & set(right_runs)):
        evidence = _row_evidence(
            image_gray_original_roi,
            left_runs[y_roi],
            right_runs[y_roi],
            inverse_rotation_matrix,
            config,
        )
        if evidence is not None:
            row_evidence.append(evidence)

    informative_count = len(row_evidence)
    separator_count = sum(
        row["supports_separator"] for row in row_evidence
    )
    same_basin_count = sum(
        row["supports_same_basin"] for row in row_evidence
    )
    separator_ratio = (
        None
        if informative_count == 0
        else separator_count / informative_count
    )
    same_basin_ratio = (
        None
        if informative_count == 0
        else same_basin_count / informative_count
    )
    coverage = _vertical_coverage(
        row_evidence,
        image_gray_original_roi.shape[0],
    )
    enough_rows = (
        informative_count
        >= config.topology_shadow_min_informative_rows
    )
    broad_coverage = (
        coverage["covered_band_count"] == 3
        and coverage["vertical_span_ratio"]
        >= config.topology_shadow_min_vertical_span_ratio
    )
    strong_conflict = bool(
        enough_rows
        and broad_coverage
        and same_basin_ratio is not None
        and same_basin_ratio
        >= config.topology_shadow_strong_same_basin_support_ratio
        and all(count > 0 for count in coverage["same_basin_rows"])
    )
    consistent = bool(
        enough_rows
        and broad_coverage
        and separator_ratio is not None
        and separator_ratio
        >= config.topology_shadow_min_separator_support_ratio
    )
    if strong_conflict:
        status = "Contradictory"
        reason = "same_dark_basin_split"
    elif consistent:
        status = "Consistent"
        reason = "bright_separator_supported"
    else:
        status = "Unable to verify"
        reason = "insufficient_or_mixed_grayscale_evidence"
    prominence_values = [
        row["normalized_prominence"] for row in row_evidence
    ]
    report = {
        "label": label,
        "left_track_id": left_track.track_id,
        "right_track_id": right_track.track_id,
        "status": status,
        "strong_same_basin_conflict": strong_conflict,
        "informative_row_count": informative_count,
        "separator_support_ratio": (
            None
            if separator_ratio is None
            else round(float(separator_ratio), 6)
        ),
        "same_basin_support_ratio": (
            None
            if same_basin_ratio is None
            else round(float(same_basin_ratio), 6)
        ),
        "normalized_prominence": (
            None
            if not prominence_values
            else round(float(np.median(prominence_values)), 6)
        ),
        "vertical_band_coverage": coverage,
        "reason": reason,
    }
    return report, row_evidence


def _debug_image(
    image_gray_original_roi: np.ndarray,
    report: dict,
    debug_rows: list[dict],
) -> np.ndarray:
    debug = cv2.cvtColor(image_gray_original_roi, cv2.COLOR_GRAY2BGR)
    if debug_rows:
        sample_step = max(1, (len(debug_rows) + 49) // 50)
        for row in debug_rows[::sample_step]:
            if row.get("supports_merged_basin"):
                color = (0, 165, 255)
            elif row.get("supports_same_basin"):
                color = (0, 0, 255)
            elif row.get("supports_separator"):
                color = (0, 180, 0)
            else:
                color = (128, 128, 128)
            start = tuple(int(round(value)) for value in row["left_point"])
            end = tuple(int(round(value)) for value in row["right_point"])
            cv2.line(debug, start, end, color, 1)
            if row.get("supports_merged_basin") and "ridge_point" in row:
                ridge = tuple(
                    int(round(value)) for value in row["ridge_point"]
                )
                cv2.circle(debug, ridge, 2, color, -1)
    cv2.rectangle(debug, (0, 0), (debug.shape[1] - 1, 28), (0, 0, 0), -1)
    cv2.putText(
        debug,
        f"topology shadow: {report['status']}",
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return debug


def evaluate_grayscale_topology(
    image_gray_original_roi: np.ndarray,
    selection: InteractiveStripeSelection,
    inverse_rotation_matrix: np.ndarray,
    config: InteractiveConfig,
) -> GrayscaleTopologyEvaluation:
    """Evaluate strong split and merge conflicts in the original grayscale."""

    if image_gray_original_roi.ndim != 2:
        raise ValueError(
            "image_gray_original_roi must be a single-channel image"
        )
    if inverse_rotation_matrix.shape != (2, 3):
        raise ValueError("inverse_rotation_matrix must have shape (2, 3)")
    if not selection.success:
        report = _empty_report("selection_not_successful")
        return GrayscaleTopologyEvaluation(
            report,
            _debug_image(image_gray_original_roi, report, []),
        )
    intervals = _selected_intervals(selection)
    if not intervals:
        report = _empty_report("selection_has_no_evaluable_intervals")
        return GrayscaleTopologyEvaluation(
            report,
            _debug_image(image_gray_original_roi, report, []),
        )

    interval_reports = []
    all_debug_rows = []
    for label, left_track, right_track in intervals:
        interval_report, debug_rows = _evaluate_interval(
            label,
            left_track,
            right_track,
            image_gray_original_roi,
            inverse_rotation_matrix,
            config,
        )
        interval_reports.append(interval_report)
        all_debug_rows.extend(debug_rows)

    selected_tracks = _selected_tracks(selection)
    track_reports = []
    for label, track in selected_tracks:
        comparison_widths = [
            other_track.median_width_px
            for _other_label, other_track in selected_tracks
            if other_track is not track
        ]
        track_report, debug_rows = _evaluate_selected_track(
            label,
            track,
            comparison_widths,
            image_gray_original_roi,
            inverse_rotation_matrix,
            config,
        )
        track_reports.append(track_report)
        all_debug_rows.extend(debug_rows)

    strong_same_basin_conflict = any(
        interval["strong_same_basin_conflict"]
        for interval in interval_reports
    )
    strong_merged_basin_conflict = any(
        track["strong_merged_basin_conflict"]
        for track in track_reports
    )
    all_consistent = all(
        interval["status"] == "Consistent"
        for interval in interval_reports
    )
    if strong_same_basin_conflict:
        status = "Contradictory"
        reason = "same_dark_basin_split"
    elif strong_merged_basin_conflict:
        status = "Contradictory"
        reason = "merged_dark_basins"
    elif all_consistent:
        status = "Consistent"
        reason = "all_expected_separators_supported"
    else:
        status = "Unable to verify"
        reason = "one_or_more_intervals_unverifiable"

    informative_count = sum(
        interval["informative_row_count"]
        for interval in interval_reports
    )
    separator_rows = sum(
        round(
            interval["separator_support_ratio"]
            * interval["informative_row_count"]
        )
        for interval in interval_reports
        if interval["separator_support_ratio"] is not None
    )
    same_basin_rows = sum(
        round(
            interval["same_basin_support_ratio"]
            * interval["informative_row_count"]
        )
        for interval in interval_reports
        if interval["same_basin_support_ratio"] is not None
    )
    band_informative = [
        sum(
            interval["vertical_band_coverage"]["informative_rows"][index]
            for interval in interval_reports
        )
        for index in range(3)
    ]
    band_same_basin = [
        sum(
            interval["vertical_band_coverage"]["same_basin_rows"][index]
            for interval in interval_reports
        )
        for index in range(3)
    ]
    report = {
        "mode": "shadow",
        "status": status,
        "strong_same_basin_conflict": strong_same_basin_conflict,
        "strong_merged_basin_conflict": (
            strong_merged_basin_conflict
        ),
        "evaluated_intervals": interval_reports,
        "evaluated_tracks": track_reports,
        "informative_row_count": informative_count,
        "separator_support_ratio": (
            None
            if informative_count == 0
            else round(separator_rows / informative_count, 6)
        ),
        "same_basin_support_ratio": (
            None
            if informative_count == 0
            else round(same_basin_rows / informative_count, 6)
        ),
        "vertical_band_coverage": {
            "informative_rows": band_informative,
            "same_basin_rows": band_same_basin,
            "covered_band_count": sum(
                count > 0 for count in band_informative
            ),
            "vertical_span_ratio": round(
                max(
                    (
                        interval["vertical_band_coverage"][
                            "vertical_span_ratio"
                        ]
                        for interval in interval_reports
                    ),
                    default=0.0,
                ),
                6,
            ),
        },
        "reason": reason,
    }
    return GrayscaleTopologyEvaluation(
        report,
        _debug_image(image_gray_original_roi, report, all_debug_rows),
    )
