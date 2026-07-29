"""Audit where the fixed column-separator prototype loses a bright ridge.

This is an offline diagnostic only. It reuses the production ROI and rotation
decisions, but it does not change the detector or feed results back into it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import CONFIG, INTERACTIVE_CONFIG  # noqa: E402
from interactive_pipeline import (  # noqa: E402
    _rotation_matrix_for_detection,
    calculate_interactive_roi_bounds,
    run_interactive_case,
)
from pitch_reference import build_pitch_reference_map  # noqa: E402
from prototype_column_basins import (  # noqa: E402
    BAND_COUNT,
    MAX_SEPARATOR_GAP_PX,
    MIN_BAND_AGREEMENT,
    _current_tracks,
    _prototype_roi,
    analyze_parameter_choice,
    band_white_counts,
    basin_index_at_x,
    merge_short_separator_gaps,
    separator_columns,
)


CURRENT_WHITE_THRESHOLD = 220
CURRENT_MIN_COUNT = 2
DIAGNOSTIC_THRESHOLDS = (180, 200, 220, 235)
ISSUE_POINTS = ((1086, 525), (661, 921), (482, 704))
CONTROL_POINTS = ((662, 1046), (1684, 1615), (396, 1506))

COLORS = (
    (35, 80, 230),
    (35, 170, 70),
    (210, 120, 20),
    (180, 45, 170),
    (40, 180, 200),
)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive runs of true values."""

    result = []
    x = 0
    while x < len(mask):
        if not mask[x]:
            x += 1
            continue
        start = x
        while x + 1 < len(mask) and mask[x + 1]:
            x += 1
        result.append((start, x))
        x += 1
    return result


def _strongest_internal_ridge(
    roi: np.ndarray,
    basin_x0: int,
    basin_x1: int,
) -> tuple[int, int, dict]:
    """Find the strongest raw bright ridge away from basin boundaries.

    The locator is diagnostic, not a detector parameter. It ranks the inner
    60% of the basin first by >=180 pixel coverage and then by the p99 profile.
    The returned range is a compact half-height extent around that peak.
    """

    basin_width = basin_x1 - basin_x0 + 1
    inset = max(2, int(round(basin_width * 0.20)))
    search_x0 = min(basin_x1, basin_x0 + inset)
    search_x1 = max(search_x0, basin_x1 - inset)

    count_180 = np.count_nonzero(roi >= 180, axis=0).astype(float)
    p99 = np.percentile(roi, 99, axis=0)
    kernel = np.ones(3, dtype=float) / 3.0
    smooth_count = np.convolve(count_180, kernel, mode="same")
    smooth_p99 = np.convolve(p99, kernel, mode="same")
    local = np.arange(search_x0, search_x1 + 1)
    best_count = np.max(smooth_count[local])
    count_ties = local[np.isclose(smooth_count[local], best_count)]
    best_p99 = np.max(smooth_p99[count_ties])
    brightness_ties = count_ties[
        np.isclose(smooth_p99[count_ties], best_p99)
    ]
    basin_center = (basin_x0 + basin_x1) / 2.0
    peak_x = int(
        brightness_ties[
            np.argmin(np.abs(brightness_ties - basin_center))
        ]
    )

    local_profile = smooth_p99[local]
    baseline = float(np.median(local_profile))
    peak_value = float(smooth_p99[peak_x])
    half_height = baseline + 0.5 * max(0.0, peak_value - baseline)
    ridge_x0 = peak_x
    ridge_x1 = peak_x
    if peak_value <= baseline:
        return ridge_x0, ridge_x1, {
            "search_x0_roi": search_x0,
            "search_x1_roi": search_x1,
            "peak_x_roi": peak_x,
            "peak_count_ge_180": int(count_180[peak_x]),
            "peak_p99": float(p99[peak_x]),
            "range_method": (
                "inner_60pct; rank smoothed >=180 coverage then p99; "
                "flat profile uses basin-center tie break"
            ),
        }
    while (
        ridge_x0 > search_x0
        and peak_x - ridge_x0 < 4
        and smooth_p99[ridge_x0 - 1] >= half_height
    ):
        ridge_x0 -= 1
    while (
        ridge_x1 < search_x1
        and ridge_x1 - peak_x < 4
        and smooth_p99[ridge_x1 + 1] >= half_height
    ):
        ridge_x1 += 1

    return ridge_x0, ridge_x1, {
        "search_x0_roi": search_x0,
        "search_x1_roi": search_x1,
        "peak_x_roi": peak_x,
        "peak_count_ge_180": int(count_180[peak_x]),
        "peak_p99": float(p99[peak_x]),
        "range_method": (
            "inner_60pct; rank smoothed >=180 coverage then p99; "
            "expand to p99 half-height, max 9px"
        ),
    }


def _map_roi_x_to_global(
    roi_x: float,
    click_y_roi: int,
    bounds,
    identity: dict,
    report: dict,
) -> float:
    """Map an ROI x at click-row height back into source coordinates."""

    if identity["geometry"] != "rotated":
        return float(bounds.x0_global + roi_x)
    angle = float(report["rotation_shadow"]["best_angle_deg"])
    matrix = _rotation_matrix_for_detection(
        int(round(report["reference_point_roi"]["x"])),
        int(round(report["reference_point_roi"]["y"])),
        angle,
    )
    inverse = cv2.invertAffineTransform(matrix)
    source_x, _ = inverse @ np.array(
        [roi_x, float(click_y_roi), 1.0],
        dtype=float,
    )
    return float(bounds.x0_global + source_x)


def _band_audit(
    roi: np.ndarray,
    ridge_x0: int,
    ridge_x1: int,
    current_counts: np.ndarray,
) -> list[dict]:
    rows_by_band = np.array_split(np.arange(roi.shape[0]), BAND_COUNT)
    result = []
    for band_index, rows in enumerate(rows_by_band):
        ridge_pixels = roi[rows, ridge_x0 : ridge_x1 + 1]
        threshold_counts = {
            str(threshold): int(
                np.count_nonzero(ridge_pixels >= threshold)
            )
            for threshold in DIAGNOSTIC_THRESHOLDS
        }
        max_per_column = {}
        for threshold in DIAGNOSTIC_THRESHOLDS:
            per_column = np.count_nonzero(
                roi[rows, :] >= threshold,
                axis=0,
            )
            max_per_column[str(threshold)] = int(
                np.max(per_column[ridge_x0 : ridge_x1 + 1])
            )

        pass_mask = (
            current_counts[band_index, :] >= CURRENT_MIN_COUNT
        )
        pass_columns = np.flatnonzero(
            pass_mask[ridge_x0 : ridge_x1 + 1]
        ) + ridge_x0
        max_current_count = int(
            np.max(
                current_counts[
                    band_index,
                    ridge_x0 : ridge_x1 + 1,
                ]
            )
        )
        passed = len(pass_columns) > 0
        reason = (
            f"pass: {len(pass_columns)} ridge column(s) have "
            f">220 count >= {CURRENT_MIN_COUNT}"
            if passed
            else (
                f"fail: maximum >220 count is {max_current_count}, "
                f"below {CURRENT_MIN_COUNT}"
            )
        )
        result.append(
            {
                "band": band_index + 1,
                "row_range": [int(rows[0]), int(rows[-1])],
                "counts_ge_threshold_in_ridge": threshold_counts,
                "max_count_ge_threshold_per_column": max_per_column,
                "current_rule_passed_in_ridge": passed,
                "current_rule_pass_columns_roi": [
                    int(x) for x in pass_columns
                ],
                "current_rule_reason": reason,
            }
        )
    return result


def _classify_loss(
    roi: np.ndarray,
    ridge_x0: int,
    ridge_x1: int,
    pre_band_mask: np.ndarray,
    consensus_mask: np.ndarray,
    final_mask: np.ndarray,
    prototype_final_mask: np.ndarray,
) -> tuple[str, str]:
    ridge = slice(ridge_x0, ridge_x1 + 1)
    expected_consensus = (
        np.count_nonzero(pre_band_mask, axis=0)
        >= MIN_BAND_AGREEMENT
    )
    if (
        not np.array_equal(consensus_mask, expected_consensus)
        or not np.array_equal(final_mask, prototype_final_mask)
    ):
        return (
            "implementation_bug",
            "recomputed mask does not match the current rule/prototype",
        )
    if np.any(consensus_mask[ridge]) and not np.any(final_mask[ridge]):
        return (
            "postprocessing_erased_separator",
            "3/5 consensus existed, but cleanup removed it",
        )

    max_agreement = int(
        np.max(
            np.count_nonzero(
                pre_band_mask[:, ridge],
                axis=0,
            )
        )
    )
    if max_agreement in (1, 2):
        return (
            "band_consensus_too_strict",
            f"at most {max_agreement}/5 bands pass at one ridge column",
        )

    lower_threshold_has_consensus = False
    for threshold in (180, 200):
        counts = band_white_counts(roi, threshold - 1)
        lower_consensus = separator_columns(
            counts,
            CURRENT_MIN_COUNT,
            MIN_BAND_AGREEMENT,
        )
        if np.any(lower_consensus[ridge]):
            lower_threshold_has_consensus = True
            break
    raw_max = int(np.max(roi[:, ridge]))
    detail = (
        "stable 3/5 evidence appears below 220"
        if lower_threshold_has_consensus
        else (
            f"strongest internal ridge never reaches current evidence "
            f"requirement (raw max={raw_max})"
        )
    )
    return "threshold_too_strict", detail


def _draw_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.45,
    color: tuple[int, int, int] = (30, 30, 30),
) -> None:
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _plot_profiles(
    profiles: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    ridge_x0: int,
    ridge_x1: int,
    title: str,
    width: int = 1100,
    height: int = 220,
) -> np.ndarray:
    panel = np.full((height, width, 3), 250, dtype=np.uint8)
    left, right, top, bottom = 58, width - 18, 28, height - 28
    cv2.rectangle(panel, (left, top), (right, bottom), (190, 190, 190), 1)
    _draw_text(panel, title, 8, 18, 0.43)
    source_width = len(profiles[0][1])

    for value in (0, 180, 200, 220, 235, 255):
        y = int(round(bottom - (bottom - top) * value / 255.0))
        cv2.line(panel, (left, y), (right, y), (225, 225, 225), 1)
        _draw_text(panel, str(value), 8, y + 4, 0.34, (90, 90, 90))

    def map_x(x: float) -> int:
        return int(round(left + (right - left) * x / (source_width - 1)))

    shade_x0 = map_x(ridge_x0)
    shade_x1 = map_x(ridge_x1)
    cv2.rectangle(
        panel,
        (shade_x0, top),
        (max(shade_x0 + 1, shade_x1), bottom),
        (235, 220, 250),
        -1,
    )
    for label, values, color in profiles:
        points = np.column_stack(
            (
                np.linspace(left, right, source_width),
                bottom - (bottom - top) * values / 255.0,
            )
        ).astype(np.int32)
        cv2.polylines(panel, [points], False, color, 1, cv2.LINE_AA)
    legend_x = 68
    for label, _, color in profiles:
        cv2.line(panel, (legend_x, 40), (legend_x + 18, 40), color, 2)
        _draw_text(panel, label, legend_x + 22, 44, 0.34)
        legend_x += max(110, len(label) * 7 + 35)
    return panel


def _mask_panel(
    pre_band_mask: np.ndarray,
    consensus_mask: np.ndarray,
    final_mask: np.ndarray,
    ridge_x0: int,
    ridge_x1: int,
    width: int = 1100,
) -> np.ndarray:
    labels = [
        "pre B1",
        "pre B2",
        "pre B3",
        "pre B4",
        "pre B5",
        "3/5 consensus",
        "after 2px gap merge",
    ]
    masks = [
        *[pre_band_mask[index] for index in range(BAND_COUNT)],
        consensus_mask,
        final_mask,
    ]
    row_height = 24
    left = 130
    panel = np.full(
        (34 + row_height * len(masks), width, 3),
        248,
        dtype=np.uint8,
    )
    _draw_text(
        panel,
        "separator masks (white = separator; magenta = audited ridge)",
        8,
        19,
        0.43,
    )
    source_width = len(masks[0])
    target_width = width - left - 18
    ridge_left = left + int(round(target_width * ridge_x0 / source_width))
    ridge_right = left + int(
        round(target_width * (ridge_x1 + 1) / source_width)
    )
    for index, (label, mask) in enumerate(zip(labels, masks)):
        y0 = 30 + index * row_height
        y1 = y0 + row_height - 5
        strip = np.where(mask, 255, 25).astype(np.uint8)[None, :]
        strip = cv2.resize(
            strip,
            (target_width, y1 - y0 + 1),
            interpolation=cv2.INTER_NEAREST,
        )
        panel[y0 : y1 + 1, left : left + target_width] = cv2.cvtColor(
            strip,
            cv2.COLOR_GRAY2BGR,
        )
        cv2.rectangle(
            panel,
            (ridge_left, y0),
            (max(ridge_left + 1, ridge_right), y1),
            (210, 40, 210),
            1,
        )
        _draw_text(panel, label, 8, y1 - 3, 0.34)
    return panel


def _render_audit(
    roi: np.ndarray,
    point: tuple[int, int],
    group: str,
    click_x_roi: int,
    click_y_roi: int,
    basin,
    tracks: list[dict],
    ridge_x0: int,
    ridge_x1: int,
    raw_stats: dict,
    band_rows: list[dict],
    pre_band_mask: np.ndarray,
    consensus_mask: np.ndarray,
    final_mask: np.ndarray,
    category: str,
    loss_detail: str,
) -> np.ndarray:
    width = 1100
    overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(
        overlay,
        (basin.x0, 0),
        (basin.x1, roi.shape[0] - 1),
        (255, 170, 0),
        1,
    )
    cv2.rectangle(
        overlay,
        (ridge_x0, 0),
        (ridge_x1, roi.shape[0] - 1),
        (210, 30, 210),
        2,
    )
    cv2.drawMarker(
        overlay,
        (click_x_roi, click_y_roi),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        13,
        2,
    )
    for track in tracks:
        center = int(round(track["center_x_roi"]))
        cv2.line(
            overlay,
            (center, 0),
            (center, roi.shape[0] - 1),
            (0, 210, 255),
            1,
        )
    overlay = cv2.resize(
        overlay,
        (1000, 400),
        interpolation=cv2.INTER_NEAREST,
    )
    header = np.full((92, width, 3), 250, dtype=np.uint8)
    x_global, y_global = point
    _draw_text(
        header,
        f"{group} ({x_global},{y_global})  category={category}",
        10,
        23,
        0.55,
    )
    _draw_text(
        header,
        (
            f"click basin ROI [{basin.x0},{basin.x1}]  "
            f"ridge ROI [{ridge_x0},{ridge_x1}]  {loss_detail}"
        ),
        10,
        48,
        0.43,
    )
    _draw_text(
        header,
        (
            "ridge raw: "
            f"max={raw_stats['max']:.0f} p99={raw_stats['p99']:.1f} "
            f"p95={raw_stats['p95']:.1f} "
            f"median={raw_stats['median']:.1f}"
        ),
        10,
        72,
        0.43,
    )
    image_panel = np.full((420, width, 3), 248, dtype=np.uint8)
    image_panel[10:410, 50:1050] = overlay

    full_profiles = [
        ("max", np.max(roi, axis=0).astype(float), (20, 20, 220)),
        ("p99", np.percentile(roi, 99, axis=0), (20, 150, 20)),
        ("p95", np.percentile(roi, 95, axis=0), (220, 100, 20)),
        ("median", np.median(roi, axis=0), (150, 40, 150)),
    ]
    rows_by_band = np.array_split(np.arange(roi.shape[0]), BAND_COUNT)
    band_profiles = []
    for index, rows in enumerate(rows_by_band):
        sorted_band = np.sort(roi[rows, :], axis=0)
        second_brightest = sorted_band[-CURRENT_MIN_COUNT, :].astype(
            float
        )
        band_profiles.append(
            (f"B{index + 1} 2nd-brightest", second_brightest, COLORS[index])
        )

    full_plot = _plot_profiles(
        full_profiles,
        ridge_x0,
        ridge_x1,
        "horizontal raw-intensity profiles",
        width,
    )
    band_plot = _plot_profiles(
        band_profiles,
        ridge_x0,
        ridge_x1,
        "five band profiles; 2nd-brightest >220 means that band passes",
        width,
    )
    masks = _mask_panel(
        pre_band_mask,
        consensus_mask,
        final_mask,
        ridge_x0,
        ridge_x1,
        width,
    )
    detail_height = 26 * len(band_rows) + 14
    details = np.full((detail_height, width, 3), 250, dtype=np.uint8)
    for index, band in enumerate(band_rows):
        counts = band["counts_ge_threshold_in_ridge"]
        line = (
            f"B{band['band']} rows {band['row_range'][0]}-"
            f"{band['row_range'][1]}  ridge counts >= "
            f"180/200/220/235: {counts['180']}/{counts['200']}/"
            f"{counts['220']}/{counts['235']}  "
            f"{band['current_rule_reason']}"
        )
        _draw_text(details, line, 8, 20 + index * 26, 0.36)
    return np.vstack(
        (header, image_panel, full_plot, band_plot, masks, details)
    )


def _audit_point(
    image_gray: np.ndarray,
    pitch_map: dict,
    group: str,
    point: tuple[int, int],
    scratch: Path,
    images_dir: Path,
) -> dict:
    x_global, y_global = point
    result = run_interactive_case(
        image_gray,
        x_global,
        y_global,
        scratch / f"{group}_{x_global}_{y_global}",
        CONFIG,
        INTERACTIVE_CONFIG,
        image_name="Sample 2.bmp",
        pitch_reference_map=pitch_map,
    )
    report = result.report
    identity, tracks = _current_tracks(report)
    roi, click_x_roi, click_y_roi = _prototype_roi(
        image_gray,
        report,
        x_global,
        y_global,
        identity,
    )
    parameter_result = analyze_parameter_choice(
        roi,
        click_x_roi,
        tracks,
        CURRENT_WHITE_THRESHOLD,
        CURRENT_MIN_COUNT,
    )
    basins = parameter_result["basins"]
    basin_id = basin_index_at_x(basins, click_x_roi)
    if basin_id is None:
        raise RuntimeError(
            f"click ({x_global},{y_global}) lies on a separator column"
        )
    basin = basins[basin_id]
    ridge_x0, ridge_x1, ridge_locator = _strongest_internal_ridge(
        roi,
        basin.x0,
        basin.x1,
    )

    current_counts = band_white_counts(roi, CURRENT_WHITE_THRESHOLD)
    pre_band_mask = current_counts >= CURRENT_MIN_COUNT
    consensus_mask = separator_columns(
        current_counts,
        CURRENT_MIN_COUNT,
        MIN_BAND_AGREEMENT,
    )
    final_mask = merge_short_separator_gaps(
        consensus_mask,
        MAX_SEPARATOR_GAP_PX,
    )
    band_rows = _band_audit(
        roi,
        ridge_x0,
        ridge_x1,
        current_counts,
    )
    category, loss_detail = _classify_loss(
        roi,
        ridge_x0,
        ridge_x1,
        pre_band_mask,
        consensus_mask,
        final_mask,
        parameter_result["separator_columns"],
    )

    ridge_pixels = roi[:, ridge_x0 : ridge_x1 + 1]
    raw_stats = {
        "max": float(np.max(ridge_pixels)),
        "p99": float(np.percentile(ridge_pixels, 99)),
        "p95": float(np.percentile(ridge_pixels, 95)),
        "median": float(np.median(ridge_pixels)),
    }
    bounds = calculate_interactive_roi_bounds(
        image_gray.shape,
        x_global,
        y_global,
        INTERACTIVE_CONFIG,
    )
    ridge_global_x0 = _map_roi_x_to_global(
        ridge_x0,
        click_y_roi,
        bounds,
        identity,
        report,
    )
    ridge_global_x1 = _map_roi_x_to_global(
        ridge_x1,
        click_y_roi,
        bounds,
        identity,
        report,
    )
    max_agreement_before = int(
        np.max(
            np.count_nonzero(
                pre_band_mask[:, ridge_x0 : ridge_x1 + 1],
                axis=0,
            )
        )
    )
    row = {
        "id": f"{group}_{x_global}_{y_global}",
        "group": group,
        "x_global": x_global,
        "y_global": y_global,
        "candidate_source": (
            f"{identity['geometry']}/{identity['threshold_method']}"
        ),
        "click_basin_id": basin_id,
        "click_basin_x0_roi": basin.x0,
        "click_basin_x1_roi": basin.x1,
        "ridge_x0_roi": ridge_x0,
        "ridge_x1_roi": ridge_x1,
        "ridge_x0_global": ridge_global_x0,
        "ridge_x1_global": ridge_global_x1,
        "raw_stats": raw_stats,
        "bands": band_rows,
        "max_agreeing_bands_before_consensus": max_agreement_before,
        "pre_consensus_runs_by_band_roi": [
            _runs(pre_band_mask[index]) for index in range(BAND_COUNT)
        ],
        "consensus_runs_roi": _runs(consensus_mask),
        "final_runs_roi": _runs(final_mask),
        "consensus_present_in_ridge": bool(
            np.any(consensus_mask[ridge_x0 : ridge_x1 + 1])
        ),
        "final_separator_present_in_ridge": bool(
            np.any(final_mask[ridge_x0 : ridge_x1 + 1])
        ),
        "disappears_at": (
            "implementation"
            if category == "implementation_bug"
            else (
                "postprocessing"
                if category == "postprocessing_erased_separator"
                else (
                    "band_consensus"
                    if category == "band_consensus_too_strict"
                    else "raw_threshold"
                )
            )
        ),
        "category": category,
        "loss_detail": loss_detail,
        "ridge_locator": ridge_locator,
        "current_rule": {
            "pixel_operator": ">",
            "white_threshold": CURRENT_WHITE_THRESHOLD,
            "minimum_count_per_band": CURRENT_MIN_COUNT,
            "minimum_band_agreement": MIN_BAND_AGREEMENT,
            "maximum_separator_gap_px": MAX_SEPARATOR_GAP_PX,
        },
    }

    figure = _render_audit(
        roi,
        point,
        group,
        click_x_roi,
        click_y_roi,
        basin,
        tracks,
        ridge_x0,
        ridge_x1,
        raw_stats,
        band_rows,
        pre_band_mask,
        consensus_mask,
        final_mask,
        category,
        loss_detail,
    )
    image_path = images_dir / f"{row['id']}.png"
    if not cv2.imwrite(str(image_path), figure):
        raise OSError(f"could not write audit figure: {image_path}")
    row["figure"] = str(image_path.relative_to(PROJECT_ROOT))
    return row


def run_audit(image_gray: np.ndarray, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    pitch_map = build_pitch_reference_map(
        image_gray,
        CONFIG,
        INTERACTIVE_CONFIG,
    )
    cases = [
        ("issue", point) for point in ISSUE_POINTS
    ] + [
        ("control", point) for point in CONTROL_POINTS
    ]
    rows = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        scratch = Path(temporary_directory)
        for group, point in cases:
            rows.append(
                _audit_point(
                    image_gray,
                    pitch_map,
                    group,
                    point,
                    scratch,
                    images_dir,
                )
            )

    report = {
        "audit": "separator_failure_audit_v1",
        "production_integration": False,
        "image_name": "Sample 2.bmp",
        "fixed_current_rule": {
            "pixel_operator": ">",
            "white_threshold": CURRENT_WHITE_THRESHOLD,
            "minimum_count_per_band": CURRENT_MIN_COUNT,
            "minimum_band_agreement": MIN_BAND_AGREEMENT,
            "maximum_separator_gap_px": MAX_SEPARATOR_GAP_PX,
        },
        "diagnostic_count_thresholds": list(DIAGNOSTIC_THRESHOLDS),
        "rows": rows,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "id": row["id"],
                "group": row["group"],
                "candidate_source": row["candidate_source"],
                "click_basin_roi": (
                    f"{row['click_basin_x0_roi']}-"
                    f"{row['click_basin_x1_roi']}"
                ),
                "ridge_roi": (
                    f"{row['ridge_x0_roi']}-{row['ridge_x1_roi']}"
                ),
                "ridge_global": (
                    f"{row['ridge_x0_global']:.2f}-"
                    f"{row['ridge_x1_global']:.2f}"
                ),
                "max": row["raw_stats"]["max"],
                "p99": row["raw_stats"]["p99"],
                "p95": row["raw_stats"]["p95"],
                "median": row["raw_stats"]["median"],
                "max_agreeing_bands": (
                    row["max_agreeing_bands_before_consensus"]
                ),
                "consensus_in_ridge": row[
                    "consensus_present_in_ridge"
                ],
                "final_in_ridge": row[
                    "final_separator_present_in_ridge"
                ],
                "disappears_at": row["disappears_at"],
                "category": row["category"],
                "loss_detail": row["loss_detail"],
                "band_details": json.dumps(
                    row["bands"],
                    separators=(",", ":"),
                ),
                "figure": row["figure"],
            }
        )
    with (output_dir / "audit.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(csv_rows[0]),
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    return report


def _print_summary(report: dict) -> None:
    print(
        "point | group | ridge global | raw max/p99/p95/median | "
        "max bands | disappears | category"
    )
    for row in report["rows"]:
        stats = row["raw_stats"]
        print(
            f"({row['x_global']},{row['y_global']}) | "
            f"{row['group']} | "
            f"{row['ridge_x0_global']:.2f}-"
            f"{row['ridge_x1_global']:.2f} | "
            f"{stats['max']:.0f}/{stats['p99']:.1f}/"
            f"{stats['p95']:.1f}/{stats['median']:.1f} | "
            f"{row['max_agreeing_bands_before_consensus']}/5 | "
            f"{row['disappears_at']} | {row['category']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=PROJECT_ROOT / "images" / "Sample 2.bmp",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "separator_failure_audit",
    )
    args = parser.parse_args()
    image_gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(args.image)
    report = run_audit(image_gray, args.output_dir)
    _print_summary(report)


if __name__ == "__main__":
    main()
