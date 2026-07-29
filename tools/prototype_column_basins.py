"""Offline prototype for column-wise white separators and dark basins.

This script does not change the production detector. It reuses the existing
point-centered ROI and rotation decisions, then evaluates a small fixed grid
of raw-grayscale separator parameters.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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
from image_processing import crop_roi_global  # noqa: E402
from interactive_pipeline import (  # noqa: E402
    _rotate_grayscale_for_detection,
    _rotation_matrix_for_detection,
    calculate_interactive_roi_bounds,
    run_interactive_case,
)
from pitch_reference import build_pitch_reference_map  # noqa: E402


WHITE_THRESHOLDS = (200, 220, 235)
MIN_COUNTS = (1, 2, 3)
BAND_COUNT = 5
MIN_BAND_AGREEMENT = 3
MAX_SEPARATOR_GAP_PX = 2

ISSUE_POINTS = (
    (661, 921),
    (482, 704),
    (1044, 581),
    (1086, 525),
    (1044, 378),
    (507, 686),
    (150, 1829),
    (1143, 1239),
    (275, 1507),
    (546, 721),
)

CONTROL_POINTS = (
    (662, 1046),
    (1684, 1615),
    (396, 1506),
)


@dataclass(frozen=True)
class DarkBasin:
    """Inclusive horizontal extent of one non-separator region."""

    basin_id: int
    x0: int
    x1: int

    @property
    def center(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1


def band_white_counts(
    image_gray_roi: np.ndarray,
    white_threshold: int,
    band_count: int = BAND_COUNT,
) -> np.ndarray:
    """Count near-white pixels per x column inside each vertical band."""

    if image_gray_roi.ndim != 2:
        raise ValueError("image_gray_roi must be single-channel")
    if band_count <= 0 or band_count > image_gray_roi.shape[0]:
        raise ValueError("band_count must fit inside the ROI height")

    rows = np.array_split(np.arange(image_gray_roi.shape[0]), band_count)
    return np.vstack(
        [
            np.count_nonzero(
                image_gray_roi[band_rows, :] > white_threshold,
                axis=0,
            )
            for band_rows in rows
        ]
    )


def separator_columns(
    counts_by_band: np.ndarray,
    min_count: int,
    min_band_agreement: int = MIN_BAND_AGREEMENT,
) -> np.ndarray:
    """Return columns with enough near-white evidence in enough bands."""

    if counts_by_band.ndim != 2:
        raise ValueError("counts_by_band must have shape bands x columns")
    if min_count <= 0:
        raise ValueError("min_count must be positive")
    if not 1 <= min_band_agreement <= counts_by_band.shape[0]:
        raise ValueError("min_band_agreement must fit the band count")
    agreeing_bands = np.count_nonzero(
        counts_by_band >= min_count,
        axis=0,
    )
    return agreeing_bands >= min_band_agreement


def merge_short_separator_gaps(
    is_separator: np.ndarray,
    max_gap_px: int = MAX_SEPARATOR_GAP_PX,
) -> np.ndarray:
    """Join white-column runs separated by only a small dark gap."""

    if is_separator.ndim != 1:
        raise ValueError("is_separator must be one-dimensional")
    if max_gap_px < 0:
        raise ValueError("max_gap_px must not be negative")

    merged = is_separator.astype(bool, copy=True)
    x = 0
    while x < len(merged):
        if merged[x]:
            x += 1
            continue
        gap_start = x
        while x < len(merged) and not merged[x]:
            x += 1
        gap_end = x - 1
        bounded = gap_start > 0 and x < len(merged)
        if bounded and gap_end - gap_start + 1 <= max_gap_px:
            merged[gap_start:x] = True
    return merged


def dark_basins_from_separators(
    is_separator: np.ndarray,
) -> list[DarkBasin]:
    """Convert consecutive non-separator columns into dark basins."""

    if is_separator.ndim != 1:
        raise ValueError("is_separator must be one-dimensional")
    basins = []
    x = 0
    while x < len(is_separator):
        if is_separator[x]:
            x += 1
            continue
        x0 = x
        while x < len(is_separator) and not is_separator[x]:
            x += 1
        basins.append(DarkBasin(len(basins), x0, x - 1))
    return basins


def basin_index_at_x(
    basins: list[DarkBasin],
    x_roi: float,
) -> int | None:
    """Return the basin containing x, if x is not a separator column."""

    for basin in basins:
        if basin.x0 <= x_roi <= basin.x1:
            return basin.basin_id
    return None


def basin_context(
    basins: list[DarkBasin],
    click_x_roi: int,
) -> dict:
    """Describe the click basin or the basins bracketing a white gap."""

    click_basin_id = basin_index_at_x(basins, click_x_roi)
    if click_basin_id is not None:
        left_id = click_basin_id - 1 if click_basin_id > 0 else None
        right_id = (
            click_basin_id + 1
            if click_basin_id + 1 < len(basins)
            else None
        )
        mode = "inside_dark_basin"
    else:
        left = [basin for basin in basins if basin.x1 < click_x_roi]
        right = [basin for basin in basins if basin.x0 > click_x_roi]
        left_id = left[-1].basin_id if left else None
        right_id = right[0].basin_id if right else None
        mode = "on_white_separator"
    return {
        "mode": mode,
        "click_basin_id": click_basin_id,
        "left_neighbor_basin_id": left_id,
        "right_neighbor_basin_id": right_id,
        "complete": left_id is not None and right_id is not None,
    }


def _candidate_identity(report: dict) -> dict:
    final_winner = report["adjacency_arbitration"]["final_winner"]
    if final_winner is not None:
        return final_winner
    current_winner = report["shadow_arbitration"].get("current_winner")
    if current_winner is not None:
        return current_winner
    return report["adjacency_arbitration"]["debug_winner"]


def _current_tracks(report: dict) -> tuple[dict, list[dict]]:
    identity = _candidate_identity(report)
    candidate_id = (
        f"{identity['geometry']}_{identity['threshold_method']}"
    )
    candidate = report["candidate_arbitration"]["candidates"][
        candidate_id
    ]
    centers = candidate["selected_track_centers_roi"]
    metrics = candidate["quality"].get("track_metrics", {})
    tracks = []
    for label in ("left", "clicked", "right"):
        center = centers.get(label)
        track_metrics = metrics.get(label)
        if center is None or track_metrics is None:
            continue
        tracks.append(
            {
                "label": label,
                "track_id": track_metrics["track_id"],
                "center_x_roi": float(center),
                "median_width_px": float(
                    track_metrics["median_width_px"]
                ),
                "valid_row_ratio": float(
                    track_metrics["valid_row_ratio"]
                ),
            }
        )
    return identity, tracks


def _basin_overlap_count(
    basins: list[DarkBasin],
    x0: float,
    x1: float,
) -> int:
    return sum(
        not (basin.x1 < x0 or basin.x0 > x1) for basin in basins
    )


def analyze_parameter_choice(
    image_gray_roi: np.ndarray,
    click_x_roi: int,
    tracks: list[dict],
    white_threshold: int,
    min_count: int,
) -> dict:
    """Evaluate one separator parameter pair for one ROI."""

    counts = band_white_counts(image_gray_roi, white_threshold)
    raw_separator = separator_columns(counts, min_count)
    merged_separator = merge_short_separator_gaps(raw_separator)
    basins = dark_basins_from_separators(merged_separator)
    context = basin_context(basins, click_x_roi)

    track_basin_ids = {
        track["label"]: basin_index_at_x(
            basins,
            track["center_x_roi"],
        )
        for track in tracks
    }
    present_basin_ids = [
        basin_id
        for basin_id in track_basin_ids.values()
        if basin_id is not None
    ]
    same_selected_basin = (
        len(present_basin_ids) != len(set(present_basin_ids))
    )

    ordered_tracks = sorted(
        tracks,
        key=lambda track: track["center_x_roi"],
    )
    intermediate_basin_ids = set()
    for left_track, right_track in zip(
        ordered_tracks,
        ordered_tracks[1:],
    ):
        left_id = track_basin_ids[left_track["label"]]
        right_id = track_basin_ids[right_track["label"]]
        if left_id is None or right_id is None:
            continue
        for basin_id in range(left_id + 1, right_id):
            intermediate_basin_ids.add(basin_id)

    wide_track_labels = []
    for track in tracks:
        half_width = track["median_width_px"] / 2.0
        overlap_count = _basin_overlap_count(
            basins,
            track["center_x_roi"] - half_width,
            track["center_x_roi"] + half_width,
        )
        if overlap_count > 1:
            wide_track_labels.append(track["label"])

    return {
        "white_threshold": white_threshold,
        "min_count": min_count,
        "counts_by_band": counts,
        "separator_columns": merged_separator,
        "basins": basins,
        "context": context,
        "track_basin_ids": track_basin_ids,
        "same_selected_basin": same_selected_basin,
        "intermediate_basin_ids": sorted(intermediate_basin_ids),
        "wide_track_labels": wide_track_labels,
        "anomaly_detected": bool(
            same_selected_basin
            or intermediate_basin_ids
            or wide_track_labels
        ),
    }


def _score_parameter(
    rows: list[dict],
    white_threshold: int,
    min_count: int,
) -> tuple:
    issue_covered = sum(
        row["group"] == "issue"
        and (
            (
                row["formal_success"]
                and row["parameter_result"]["anomaly_detected"]
            )
            or (
                not row["formal_success"]
                and row["parameter_result"]["context"]["complete"]
            )
        )
        for row in rows
    )
    controls_clean = sum(
        row["group"] == "control"
        and row["parameter_result"]["context"]["complete"]
        and not row["parameter_result"]["anomaly_detected"]
        for row in rows
    )
    return (
        issue_covered + controls_clean,
        issue_covered,
        controls_clean,
        -abs(white_threshold - 220),
        -abs(min_count - 2),
    )


def _basin_text(
    basins: list[DarkBasin],
    basin_id: int | None,
) -> str:
    if basin_id is None:
        return ""
    basin = basins[basin_id]
    return f"B{basin_id}[{basin.x0}-{basin.x1}]"


def _render_comparison(
    image_gray_roi: np.ndarray,
    click_x_roi: int,
    click_y_roi: int,
    tracks: list[dict],
    identity: dict,
    parameter_result: dict,
    title: str,
) -> np.ndarray:
    height, width = image_gray_roi.shape
    overlay = cv2.cvtColor(image_gray_roi, cv2.COLOR_GRAY2BGR)
    basins = parameter_result["basins"]
    context = parameter_result["context"]

    for basin in basins:
        cv2.rectangle(
            overlay,
            (basin.x0, 0),
            (basin.x1, height - 1),
            (80, 80, 80),
            1,
        )
    basin_colors = (
        (context["left_neighbor_basin_id"], (255, 120, 0)),
        (context["click_basin_id"], (0, 0, 255)),
        (context["right_neighbor_basin_id"], (0, 220, 220)),
    )
    for basin_id, color in basin_colors:
        if basin_id is None:
            continue
        basin = basins[basin_id]
        cv2.rectangle(
            overlay,
            (basin.x0, 1),
            (basin.x1, height - 2),
            color,
            2,
        )

    separator_x = np.flatnonzero(
        parameter_result["separator_columns"]
    )
    overlay[0:5, separator_x] = (255, 255, 255)
    for boundary in np.linspace(0, height, BAND_COUNT + 1)[1:-1]:
        y = min(height - 1, int(round(boundary)))
        cv2.line(overlay, (0, y), (width - 1, y), (0, 120, 0), 1)

    track_colors = {
        "left": (255, 0, 0),
        "clicked": (0, 140, 255),
        "right": (0, 255, 255),
    }
    for track in tracks:
        x = int(round(track["center_x_roi"]))
        half_width = int(round(track["median_width_px"] / 2.0))
        color = track_colors[track["label"]]
        cv2.line(overlay, (x, 0), (x, height - 1), color, 2)
        cv2.line(
            overlay,
            (max(0, x - half_width), 8),
            (min(width - 1, x + half_width), 8),
            color,
            2,
        )
    cv2.drawMarker(
        overlay,
        (click_x_roi, click_y_roi),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        15,
        2,
    )

    panel_height = 150
    panel = np.full((panel_height, width, 3), 245, dtype=np.uint8)
    counts = parameter_result["counts_by_band"]
    min_count = parameter_result["min_count"]
    strip_height = 12
    for band_index in range(BAND_COUNT):
        qualified = counts[band_index] >= min_count
        y0 = 8 + band_index * strip_height
        panel[y0 : y0 + 8, qualified] = (235, 235, 235)
        panel[y0 : y0 + 8, ~qualified] = (45, 45, 45)

    lines = [
        title,
        (
            f"{identity['geometry']}/{identity['threshold_method']}  "
            f"gray>{parameter_result['white_threshold']}  "
            f"count>={min_count}  bands>=3/5  "
            f"basins={len(basins)}"
        ),
        (
            f"click={context['mode']} "
            f"{_basin_text(basins, context['click_basin_id'])}  "
            f"L={_basin_text(basins, context['left_neighbor_basin_id'])}  "
            f"R={_basin_text(basins, context['right_neighbor_basin_id'])}"
        ),
        (
            f"track basins={parameter_result['track_basin_ids']}  "
            f"same={parameter_result['same_selected_basin']}  "
            f"middle={parameter_result['intermediate_basin_ids']}  "
            f"wide={parameter_result['wide_track_labels']}"
        ),
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (8, 82 + index * 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return np.vstack((overlay, panel))


def _prototype_roi(
    image_gray: np.ndarray,
    report: dict,
    click_x_global: int,
    click_y_global: int,
    identity: dict,
) -> tuple[np.ndarray, int, int]:
    bounds = calculate_interactive_roi_bounds(
        image_gray.shape,
        click_x_global,
        click_y_global,
        INTERACTIVE_CONFIG,
    )
    roi = crop_roi_global(image_gray, bounds)
    click_x_roi = click_x_global - bounds.x0_global
    click_y_roi = click_y_global - bounds.y0_global
    if identity["geometry"] == "rotated":
        angle = report["rotation_shadow"]["best_angle_deg"]
        matrix = _rotation_matrix_for_detection(
            click_x_roi,
            click_y_roi,
            angle,
        )
        roi = _rotate_grayscale_for_detection(roi, matrix)
    return roi, click_x_roi, click_y_roi


def run_prototype(
    image_gray: np.ndarray,
    output_dir: Path,
) -> dict:
    """Run the bounded experiment and write one image per point."""

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    pitch_map = build_pitch_reference_map(
        image_gray,
        CONFIG,
        INTERACTIVE_CONFIG,
    )

    cases = [
        ("issue", x, y) for x, y in ISSUE_POINTS
    ] + [
        ("control", x, y) for x, y in CONTROL_POINTS
    ]
    point_runs = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        scratch = Path(temporary_directory)
        for group, x_global, y_global in cases:
            result = run_interactive_case(
                image_gray,
                x_global,
                y_global,
                scratch / f"{x_global}_{y_global}",
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
            parameter_results = {}
            for white_threshold in WHITE_THRESHOLDS:
                for min_count in MIN_COUNTS:
                    key = f"{white_threshold}_{min_count}"
                    parameter_results[key] = analyze_parameter_choice(
                        roi,
                        click_x_roi,
                        tracks,
                        white_threshold,
                        min_count,
                    )
            point_runs.append(
                {
                    "id": f"{group}_{x_global}_{y_global}",
                    "group": group,
                    "x_global": x_global,
                    "y_global": y_global,
                    "formal_success": bool(
                        report["interactive_result"]["success"]
                    ),
                    "formal_failure_reasons": report[
                        "interactive_result"
                    ]["failure_reasons"],
                    "identity": identity,
                    "tracks": tracks,
                    "roi": roi,
                    "click_x_roi": click_x_roi,
                    "click_y_roi": click_y_roi,
                    "rotation_applied": identity["geometry"] == "rotated",
                    "parameter_results": parameter_results,
                }
            )

    parameter_scores = []
    for white_threshold in WHITE_THRESHOLDS:
        for min_count in MIN_COUNTS:
            key = f"{white_threshold}_{min_count}"
            rows = [
                {
                    **point,
                    "parameter_result": point["parameter_results"][key],
                }
                for point in point_runs
            ]
            score = _score_parameter(rows, white_threshold, min_count)
            parameter_scores.append(
                {
                    "white_threshold": white_threshold,
                    "min_count": min_count,
                    "total_covered": score[0],
                    "issue_covered": score[1],
                    "controls_clean": score[2],
                    "_rank": score,
                }
            )
    best = max(parameter_scores, key=lambda item: item["_rank"])
    best_key = f"{best['white_threshold']}_{best['min_count']}"

    summary_rows = []
    for point in point_runs:
        chosen = point["parameter_results"][best_key]
        context = chosen["context"]
        if point["group"] == "issue":
            covered = (
                chosen["anomaly_detected"]
                if point["formal_success"]
                else context["complete"]
            )
        else:
            covered = (
                context["complete"] and not chosen["anomaly_detected"]
            )
        row = {
            "id": point["id"],
            "group": point["group"],
            "x_global": point["x_global"],
            "y_global": point["y_global"],
            "formal_success": point["formal_success"],
            "candidate_source": (
                f"{point['identity']['geometry']}/"
                f"{point['identity']['threshold_method']}"
            ),
            "white_threshold": best["white_threshold"],
            "min_count": best["min_count"],
            "basin_count": len(chosen["basins"]),
            "click_mode": context["mode"],
            "click_basin": _basin_text(
                chosen["basins"],
                context["click_basin_id"],
            ),
            "left_neighbor_basin": _basin_text(
                chosen["basins"],
                context["left_neighbor_basin_id"],
            ),
            "right_neighbor_basin": _basin_text(
                chosen["basins"],
                context["right_neighbor_basin_id"],
            ),
            "context_complete": context["complete"],
            "track_basin_ids": json.dumps(
                chosen["track_basin_ids"],
                sort_keys=True,
            ),
            "same_selected_basin": chosen["same_selected_basin"],
            "intermediate_basin_ids": json.dumps(
                chosen["intermediate_basin_ids"]
            ),
            "wide_track_labels": json.dumps(
                chosen["wide_track_labels"]
            ),
            "diagnostic_covered": bool(covered),
        }
        summary_rows.append(row)
        comparison = _render_comparison(
            point["roi"],
            point["click_x_roi"],
            point["click_y_roi"],
            point["tracks"],
            point["identity"],
            chosen,
            point["id"],
        )
        image_path = images_dir / f"{point['id']}.png"
        if not cv2.imwrite(str(image_path), comparison):
            raise OSError(f"could not save prototype image: {image_path}")

    for item in parameter_scores:
        item.pop("_rank")
    report = {
        "prototype": "column_white_separator_dark_basins_v1",
        "production_integration": False,
        "image_name": "Sample 2.bmp",
        "parameter_grid": {
            "white_thresholds": list(WHITE_THRESHOLDS),
            "min_counts": list(MIN_COUNTS),
            "band_count": BAND_COUNT,
            "minimum_band_agreement": MIN_BAND_AGREEMENT,
            "maximum_separator_gap_px": MAX_SEPARATOR_GAP_PX,
        },
        "best_parameter": {
            "white_threshold": best["white_threshold"],
            "min_count": best["min_count"],
        },
        "parameter_scores": parameter_scores,
        "summary_rows": summary_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(summary_rows[0]),
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    return report


def _print_summary(report: dict) -> None:
    best = report["best_parameter"]
    print(
        "best parameters: "
        f"white_threshold={best['white_threshold']}, "
        f"min_count={best['min_count']}, bands>=3/5"
    )
    print(
        "point | formal | source | same basin | middle basin | "
        "wide track | click context | covered"
    )
    for row in report["summary_rows"]:
        print(
            f"{row['x_global']},{row['y_global']} | "
            f"{'success' if row['formal_success'] else 'failure'} | "
            f"{row['candidate_source']} | "
            f"{row['same_selected_basin']} | "
            f"{row['intermediate_basin_ids']} | "
            f"{row['wide_track_labels']} | "
            f"{row['context_complete']} | "
            f"{row['diagnostic_covered']}"
        )
    print("parameter coverage:")
    for score in sorted(
        report["parameter_scores"],
        key=lambda item: (
            item["total_covered"],
            item["issue_covered"],
            item["controls_clean"],
        ),
        reverse=True,
    ):
        print(
            f"  {score['white_threshold']}/{score['min_count']}: "
            f"total={score['total_covered']}/13, "
            f"issues={score['issue_covered']}/10, "
            f"controls={score['controls_clean']}/3"
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
        default=PROJECT_ROOT / "outputs" / "column_basins_prototype",
    )
    args = parser.parse_args()
    image_gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(args.image)
    report = run_prototype(image_gray, args.output_dir)
    _print_summary(report)


if __name__ == "__main__":
    main()
