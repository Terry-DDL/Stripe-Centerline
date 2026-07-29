"""Offline comparison of 2/5 versus 3/5 white-band consensus.

All other separator parameters remain fixed. Results are diagnostic only and
are never passed back into the production detector.
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
from interactive_pipeline import run_interactive_case  # noqa: E402
from pitch_reference import build_pitch_reference_map  # noqa: E402
from audit_separator_failures import (  # noqa: E402
    _band_audit,
    _plot_profiles,
    _strongest_internal_ridge,
)
from prototype_column_basins import (  # noqa: E402
    BAND_COUNT,
    CONTROL_POINTS,
    ISSUE_POINTS,
    DarkBasin,
    _basin_overlap_count,
    _current_tracks,
    _prototype_roi,
    band_white_counts,
    basin_context,
    basin_index_at_x,
    dark_basins_from_separators,
    merge_short_separator_gaps,
    separator_columns,
)


WHITE_THRESHOLD = 220
MIN_COUNT = 2
MAX_GAP = 2
AGREEMENTS = (2, 3)
TARGET_POINT = (1086, 525)

TRACK_COLORS = {
    "left": (255, 0, 0),
    "clicked": (0, 140, 255),
    "right": (0, 255, 255),
}
BAND_COLORS = (
    (35, 80, 230),
    (35, 170, 70),
    (210, 120, 20),
    (180, 45, 170),
    (40, 180, 200),
)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    result = []
    x = 0
    while x < len(mask):
        if not mask[x]:
            x += 1
            continue
        x0 = x
        while x + 1 < len(mask) and mask[x + 1]:
            x += 1
        result.append((x0, x))
        x += 1
    return result


def _analyze(
    roi: np.ndarray,
    click_x_roi: int,
    tracks: list[dict],
    agreement: int,
) -> dict:
    counts = band_white_counts(roi, WHITE_THRESHOLD)
    consensus = separator_columns(counts, MIN_COUNT, agreement)
    final = merge_short_separator_gaps(consensus, MAX_GAP)
    basins = dark_basins_from_separators(final)
    context = basin_context(basins, click_x_roi)
    track_basin_ids = {
        track["label"]: basin_index_at_x(
            basins,
            track["center_x_roi"],
        )
        for track in tracks
    }
    present_ids = [
        basin_id
        for basin_id in track_basin_ids.values()
        if basin_id is not None
    ]
    same_selected_basin = len(present_ids) != len(set(present_ids))

    intermediate_ids = set()
    ordered_tracks = sorted(
        tracks,
        key=lambda track: track["center_x_roi"],
    )
    for left_track, right_track in zip(
        ordered_tracks,
        ordered_tracks[1:],
    ):
        left_id = track_basin_ids[left_track["label"]]
        right_id = track_basin_ids[right_track["label"]]
        if left_id is None or right_id is None:
            continue
        intermediate_ids.update(range(left_id + 1, right_id))

    wide_labels = []
    for track in tracks:
        half_width = track["median_width_px"] / 2.0
        if (
            _basin_overlap_count(
                basins,
                track["center_x_roi"] - half_width,
                track["center_x_roi"] + half_width,
            )
            > 1
        ):
            wide_labels.append(track["label"])
    return {
        "agreement": agreement,
        "counts": counts,
        "consensus_mask": consensus,
        "final_mask": final,
        "basins": basins,
        "context": context,
        "track_basin_ids": track_basin_ids,
        "same_selected_basin": same_selected_basin,
        "intermediate_basin_ids": sorted(intermediate_ids),
        "wide_track_labels": wide_labels,
        "anomaly_detected": bool(
            same_selected_basin or intermediate_ids or wide_labels
        ),
    }


def _relevant_basins(result_3: dict) -> list[DarkBasin]:
    context = result_3["context"]
    ids = [
        context["left_neighbor_basin_id"],
        context["click_basin_id"],
        context["right_neighbor_basin_id"],
    ]
    return [
        result_3["basins"][basin_id]
        for basin_id in ids
        if basin_id is not None
    ]


def _new_isolated_runs(
    mask_2: np.ndarray,
    mask_3: np.ndarray,
) -> list[tuple[int, int]]:
    """Return new 2/5 runs that do more than widen a 3/5 separator."""

    result = []
    for x0, x1 in _runs(mask_2):
        near_existing = mask_3[
            max(0, x0 - 1) : min(len(mask_3), x1 + 2)
        ]
        if not np.any(near_existing):
            result.append((x0, x1))
    return result


def _overlaps_any(
    run: tuple[int, int],
    basins: list[DarkBasin],
) -> bool:
    x0, x1 = run
    return any(
        not (x1 < basin.x0 or x0 > basin.x1) for basin in basins
    )


def _put_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.42,
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


def _overlay(
    roi: np.ndarray,
    result: dict,
    tracks: list[dict],
    click_x_roi: int,
    click_y_roi: int,
    new_relevant_runs: list[tuple[int, int]],
    target_basin: DarkBasin | None,
    target_ridge: tuple[int, int] | None,
) -> np.ndarray:
    overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    for basin in result["basins"]:
        cv2.rectangle(
            overlay,
            (basin.x0, 0),
            (basin.x1, roi.shape[0] - 1),
            (85, 85, 85),
            1,
        )
    for run_x0, run_x1 in new_relevant_runs:
        cv2.rectangle(
            overlay,
            (run_x0, 0),
            (run_x1, roi.shape[0] - 1),
            (220, 30, 220),
            2,
        )
    if target_basin is not None:
        cv2.rectangle(
            overlay,
            (target_basin.x0, 1),
            (target_basin.x1, roi.shape[0] - 2),
            (255, 120, 0),
            2,
        )
    if target_ridge is not None:
        cv2.rectangle(
            overlay,
            (target_ridge[0], 0),
            (target_ridge[1], roi.shape[0] - 1),
            (210, 30, 210),
            2,
        )
    for track in tracks:
        x = int(round(track["center_x_roi"]))
        color = TRACK_COLORS[track["label"]]
        cv2.line(overlay, (x, 0), (x, roi.shape[0] - 1), color, 1)
    cv2.drawMarker(
        overlay,
        (click_x_roi, click_y_roi),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        13,
        2,
    )
    separator_x = np.flatnonzero(result["final_mask"])
    overlay[0:4, separator_x] = (255, 255, 255)
    return overlay


def _mask_rows(
    counts: np.ndarray,
    mask_2: np.ndarray,
    mask_3: np.ndarray,
    new_runs: list[tuple[int, int]],
    width: int = 1100,
) -> np.ndarray:
    labels = [f"B{index + 1}" for index in range(BAND_COUNT)]
    masks = [counts[index] >= MIN_COUNT for index in range(BAND_COUNT)]
    labels.extend(("2/5 final", "3/5 final"))
    masks.extend((mask_2, mask_3))
    left = 92
    target_width = width - left - 18
    row_height = 22
    panel = np.full(
        (26 + row_height * len(masks), width, 3),
        248,
        dtype=np.uint8,
    )
    _put_text(
        panel,
        "pre-consensus band masks and final separator masks",
        8,
        18,
    )
    for index, (label, mask) in enumerate(zip(labels, masks)):
        y0 = 25 + index * row_height
        y1 = y0 + 16
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
        _put_text(panel, label, 8, y1 - 3, 0.35)
        for run_x0, run_x1 in new_runs:
            x0 = left + int(target_width * run_x0 / len(mask))
            x1 = left + int(
                target_width * (run_x1 + 1) / len(mask)
            )
            cv2.rectangle(
                panel,
                (x0, y0),
                (max(x0 + 1, x1), y1),
                (210, 30, 210),
                1,
            )
    return panel


def _render(
    point_row: dict,
    roi: np.ndarray,
    tracks: list[dict],
    result_2: dict,
    result_3: dict,
    click_x_roi: int,
    click_y_roi: int,
    target_basin: DarkBasin | None,
    target_ridge: tuple[int, int] | None,
) -> np.ndarray:
    width = 1100
    header = np.full((78, width, 3), 250, dtype=np.uint8)
    _put_text(
        header,
        (
            f"{point_row['group']} ({point_row['x_global']},"
            f"{point_row['y_global']})  "
            f"{point_row['candidate_source']}  fixed >220/count>=2/gap=2"
        ),
        8,
        22,
        0.52,
    )
    _put_text(
        header,
        (
            f"new isolated 2/5 runs ROI="
            f"{point_row['new_isolated_runs_roi']}  relevant="
            f"{point_row['new_relevant_runs_roi']}  "
            f"class={point_row['split_classification']}"
        ),
        8,
        48,
    )
    _put_text(
        header,
        "magenta=new relevant cut; cyan=target left-neighbor basin",
        8,
        69,
        0.36,
    )

    new_runs = [
        tuple(run) for run in point_row["new_relevant_runs_roi"]
    ]
    overlay_3 = _overlay(
        roi,
        result_3,
        tracks,
        click_x_roi,
        click_y_roi,
        new_runs,
        target_basin,
        target_ridge,
    )
    overlay_2 = _overlay(
        roi,
        result_2,
        tracks,
        click_x_roi,
        click_y_roi,
        new_runs,
        target_basin,
        target_ridge,
    )
    overlay_3 = cv2.resize(
        overlay_3,
        (500, 200),
        interpolation=cv2.INTER_NEAREST,
    )
    overlay_2 = cv2.resize(
        overlay_2,
        (500, 200),
        interpolation=cv2.INTER_NEAREST,
    )
    comparisons = np.full((250, width, 3), 248, dtype=np.uint8)
    comparisons[35:235, 30:530] = overlay_3
    comparisons[35:235, 570:1070] = overlay_2
    _put_text(comparisons, "3/5 bands", 30, 22, 0.48)
    _put_text(comparisons, "2/5 bands", 570, 22, 0.48)

    masks = _mask_rows(
        result_3["counts"],
        result_2["final_mask"],
        result_3["final_mask"],
        new_runs,
        width,
    )

    if target_ridge is None:
        plot_x0 = click_x_roi
        plot_x1 = click_x_roi
    else:
        plot_x0, plot_x1 = target_ridge
    rows_by_band = np.array_split(
        np.arange(roi.shape[0]),
        BAND_COUNT,
    )
    profiles = []
    for index, rows in enumerate(rows_by_band):
        sorted_band = np.sort(roi[rows, :], axis=0)
        profiles.append(
            (
                f"B{index + 1}",
                sorted_band[-MIN_COUNT, :].astype(float),
                BAND_COLORS[index],
            )
        )
    profiles_panel = _plot_profiles(
        profiles,
        plot_x0,
        plot_x1,
        "five band 2nd-brightest profiles; >220 passes a band",
        width,
        200,
    )
    return np.vstack((header, comparisons, profiles_panel, masks))


def _target_left_basin_audit(
    roi: np.ndarray,
    click_x_roi: int,
    x_global: int,
    result_3: dict,
) -> tuple[dict, DarkBasin, tuple[int, int]]:
    left_basin_id = result_3["context"]["left_neighbor_basin_id"]
    if left_basin_id is None:
        raise RuntimeError("target point has no left-neighbor basin")
    left_basin = result_3["basins"][left_basin_id]
    ridge_x0, ridge_x1, locator = _strongest_internal_ridge(
        roi,
        left_basin.x0,
        left_basin.x1,
    )
    counts = result_3["counts"]
    bands = _band_audit(
        roi,
        ridge_x0,
        ridge_x1,
        counts,
    )
    ridge_pixels = roi[:, ridge_x0 : ridge_x1 + 1]
    pre_mask = counts >= MIN_COUNT
    max_agreement = int(
        np.max(
            np.count_nonzero(
                pre_mask[:, ridge_x0 : ridge_x1 + 1],
                axis=0,
            )
        )
    )
    separator_2 = separator_columns(counts, MIN_COUNT, 2)
    separator_3 = separator_columns(counts, MIN_COUNT, 3)
    final_2 = merge_short_separator_gaps(separator_2, MAX_GAP)
    final_3 = merge_short_separator_gaps(separator_3, MAX_GAP)
    origin_global_x = x_global - click_x_roi
    audit = {
        "point": [x_global, TARGET_POINT[1]],
        "target": "current_3_of_5_left_neighbor_basin",
        "basin_id": left_basin_id,
        "basin_x_roi": [left_basin.x0, left_basin.x1],
        "basin_x_global": [
            origin_global_x + left_basin.x0,
            origin_global_x + left_basin.x1,
        ],
        "ridge_x_roi": [ridge_x0, ridge_x1],
        "ridge_x_global": [
            origin_global_x + ridge_x0,
            origin_global_x + ridge_x1,
        ],
        "raw_stats": {
            "max": float(np.max(ridge_pixels)),
            "p99": float(np.percentile(ridge_pixels, 99)),
            "p95": float(np.percentile(ridge_pixels, 95)),
            "median": float(np.median(ridge_pixels)),
        },
        "bands": bands,
        "max_agreeing_bands": max_agreement,
        "separator_present_before_consensus_by_band": [
            bool(
                np.any(
                    pre_mask[
                        index,
                        ridge_x0 : ridge_x1 + 1,
                    ]
                )
            )
            for index in range(BAND_COUNT)
        ],
        "separator_present_2_of_5_before_gap": bool(
            np.any(separator_2[ridge_x0 : ridge_x1 + 1])
        ),
        "separator_present_3_of_5_before_gap": bool(
            np.any(separator_3[ridge_x0 : ridge_x1 + 1])
        ),
        "separator_present_2_of_5_after_gap": bool(
            np.any(final_2[ridge_x0 : ridge_x1 + 1])
        ),
        "separator_present_3_of_5_after_gap": bool(
            np.any(final_3[ridge_x0 : ridge_x1 + 1])
        ),
        "disappears_at": "band_consensus",
        "classification": "band_consensus_too_strict",
        "ridge_locator": locator,
    }
    return audit, left_basin, (ridge_x0, ridge_x1)


def run_comparison(image_gray: np.ndarray, output_dir: Path) -> dict:
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
    target_audit = None
    with tempfile.TemporaryDirectory() as temporary_directory:
        scratch = Path(temporary_directory)
        for group, point in cases:
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
            result_2 = _analyze(roi, click_x_roi, tracks, 2)
            result_3 = _analyze(roi, click_x_roi, tracks, 3)
            all_new = _new_isolated_runs(
                result_2["final_mask"],
                result_3["final_mask"],
            )
            relevant_basins = _relevant_basins(result_3)
            relevant_new = [
                run
                for run in all_new
                if _overlaps_any(run, relevant_basins)
            ]
            if group == "issue" and relevant_new:
                split_classification = "new_correct_split"
            elif group == "control" and relevant_new:
                split_classification = "new_incorrect_split"
            else:
                split_classification = "no_relevant_new_split"

            origin_global_x = x_global - click_x_roi
            row = {
                "id": f"{group}_{x_global}_{y_global}",
                "group": group,
                "x_global": x_global,
                "y_global": y_global,
                "candidate_source": (
                    f"{identity['geometry']}/"
                    f"{identity['threshold_method']}"
                ),
                "basin_count_3_of_5": len(result_3["basins"]),
                "basin_count_2_of_5": len(result_2["basins"]),
                "new_isolated_runs_roi": [
                    list(run) for run in all_new
                ],
                "new_isolated_runs_global": [
                    [
                        origin_global_x + run[0],
                        origin_global_x + run[1],
                    ]
                    for run in all_new
                ],
                "new_relevant_runs_roi": [
                    list(run) for run in relevant_new
                ],
                "new_relevant_runs_global": [
                    [
                        origin_global_x + run[0],
                        origin_global_x + run[1],
                    ]
                    for run in relevant_new
                ],
                "split_classification": split_classification,
            }
            target_basin = None
            target_ridge = None
            if point == TARGET_POINT:
                (
                    target_audit,
                    target_basin,
                    target_ridge,
                ) = _target_left_basin_audit(
                    roi,
                    click_x_roi,
                    x_global,
                    result_3,
                )
            image_path = images_dir / f"{row['id']}.png"
            figure = _render(
                row,
                roi,
                tracks,
                result_2,
                result_3,
                click_x_roi,
                click_y_roi,
                target_basin,
                target_ridge,
            )
            if not cv2.imwrite(str(image_path), figure):
                raise OSError(f"could not write {image_path}")
            row["figure"] = str(image_path.relative_to(PROJECT_ROOT))
            rows.append(row)

    if target_audit is None:
        raise RuntimeError("target point was not audited")
    summary = {
        "issue_points_with_new_correct_split": sum(
            row["split_classification"] == "new_correct_split"
            for row in rows
        ),
        "new_correct_split_runs": sum(
            len(row["new_relevant_runs_roi"])
            for row in rows
            if row["group"] == "issue"
        ),
        "control_points_with_new_incorrect_split": sum(
            row["split_classification"] == "new_incorrect_split"
            for row in rows
        ),
        "new_incorrect_split_runs": sum(
            len(row["new_relevant_runs_roi"])
            for row in rows
            if row["group"] == "control"
        ),
        "new_out_of_context_runs": sum(
            len(row["new_isolated_runs_roi"])
            - len(row["new_relevant_runs_roi"])
            for row in rows
        ),
    }
    report = {
        "comparison": "separator_band_consensus_2_vs_3",
        "production_integration": False,
        "fixed_parameters": {
            "pixel_operator": ">",
            "white_threshold": WHITE_THRESHOLD,
            "minimum_count_per_band": MIN_COUNT,
            "maximum_gap_px": MAX_GAP,
            "agreements_compared": list(AGREEMENTS),
        },
        "target_left_neighbor_audit": target_audit,
        "summary": summary,
        "rows": rows,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "new_isolated_runs_roi",
                        "new_isolated_runs_global",
                        "new_relevant_runs_roi",
                        "new_relevant_runs_global",
                    }
                },
                "new_isolated_runs_roi": json.dumps(
                    row["new_isolated_runs_roi"]
                ),
                "new_isolated_runs_global": json.dumps(
                    row["new_isolated_runs_global"]
                ),
                "new_relevant_runs_roi": json.dumps(
                    row["new_relevant_runs_roi"]
                ),
                "new_relevant_runs_global": json.dumps(
                    row["new_relevant_runs_global"]
                ),
            }
        )
    with (output_dir / "comparison.csv").open(
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
    target = report["target_left_neighbor_audit"]
    print(
        "target (1086,525) left basin: "
        f"ROI={target['basin_x_roi']} "
        f"global={target['basin_x_global']}"
    )
    print(
        "target ridge: "
        f"ROI={target['ridge_x_roi']} "
        f"global={target['ridge_x_global']} "
        f"stats={target['raw_stats']} "
        f"bands={target['max_agreeing_bands']}/5 "
        f"stage={target['disappears_at']}"
    )
    print(
        "point | 3/5 basins -> 2/5 basins | new relevant global | class"
    )
    for row in report["rows"]:
        print(
            f"({row['x_global']},{row['y_global']}) | "
            f"{row['basin_count_3_of_5']} -> "
            f"{row['basin_count_2_of_5']} | "
            f"{row['new_relevant_runs_global']} | "
            f"{row['split_classification']}"
        )
    print("summary:", json.dumps(report["summary"], sort_keys=True))


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
        default=(
            PROJECT_ROOT
            / "outputs"
            / "separator_band_consensus_2_vs_3"
        ),
    )
    args = parser.parse_args()
    image_gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(args.image)
    report = run_comparison(image_gray, args.output_dir)
    _print_summary(report)


if __name__ == "__main__":
    main()
