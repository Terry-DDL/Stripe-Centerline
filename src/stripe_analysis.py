"""Explainable row-run voting for adjacent vertical black stripes."""

from dataclasses import dataclass, field
import json
from pathlib import Path

import cv2
import numpy as np

from config import ProcessingConfig


@dataclass
class BlackRun:
    """One white interval in black_mask, expressed in ROI coordinates."""

    y_roi: int
    x0_roi: int
    x1_roi: int
    center_x_roi: float
    width_px: int
    accepted: bool
    rejection_reason: str | None = None


@dataclass
class StripeTrack:
    """A vertically stable group of row runs near one x position."""

    track_id: int
    seed_x_roi: int
    initial_center_median: float
    initial_width_median: float
    width_tolerance_px: float
    center_x_roi: float
    median_width_px: float
    assigned_row_count: int
    valid_row_count: int
    valid_row_ratio: float
    retention_ratio: float
    track_stage_rejection_counts: dict[str, int]
    crossing_assigned_count: int
    crossing_valid_count: int
    side: str
    eligible: bool
    rejection_reasons: list[str]
    assigned_runs: list[BlackRun] = field(repr=False)
    valid_runs: list[BlackRun] = field(repr=False)
    rejected_runs: list[tuple[BlackRun, str]] = field(repr=False)
    selected: bool = False
    selection_reason: str = "not_evaluated"


@dataclass
class AdjacentStripeAnalysis:
    """All intermediate data needed for debug images and JSON output."""

    x_ref_roi: int
    x_ref_global: int
    roi_x0_global: int
    roi_shape: tuple[int, int]
    all_runs: list[BlackRun]
    run_filter_counts: dict[str, int]
    raw_votes: np.ndarray
    smoothed_votes: np.ndarray
    tracks: list[StripeTrack]
    left_track: StripeTrack | None
    right_track: StripeTrack | None
    success: bool


def _find_row_runs(mask_row) -> list[tuple[int, int]]:
    """Return half-open white intervals for one binary-mask row."""

    runs = []
    width_roi = mask_row.shape[0]
    x_roi = 0
    while x_roi < width_roi:
        if mask_row[x_roi] == 0:
            x_roi += 1
            continue

        x0_roi = x_roi
        while x_roi < width_roi and mask_row[x_roi] != 0:
            x_roi += 1
        runs.append((x0_roi, x_roi))
    return runs


def extract_and_filter_black_runs(
    black_mask_roi,
    x_ref_roi: int,
    config: ProcessingConfig,
) -> tuple[list[BlackRun], dict[str, int]]:
    """Extract every row run and attach one clear filter decision."""

    height_roi, width_roi = black_mask_roi.shape[:2]
    all_runs = []
    counts = {
        "total": 0,
        "accepted": 0,
        "touches_roi_border": 0,
        "too_narrow": 0,
        "too_wide": 0,
        "outside_search_radius": 0,
    }

    for y_roi in range(height_roi):
        for x0_roi, x1_roi in _find_row_runs(black_mask_roi[y_roi]):
            width_px = x1_roi - x0_roi
            center_x_roi = (x0_roi + x1_roi - 1) / 2.0
            reason = None

            if config.reject_border_touching_runs and (
                x0_roi == 0 or x1_roi == width_roi
            ):
                reason = "touches_roi_border"
            elif width_px < config.min_run_width_px:
                reason = "too_narrow"
            elif width_px > config.max_run_width_px:
                reason = "too_wide"
            elif abs(center_x_roi - x_ref_roi) > config.stripe_search_radius_px:
                reason = "outside_search_radius"

            accepted = reason is None
            counts["total"] += 1
            counts["accepted" if accepted else reason] += 1
            all_runs.append(
                BlackRun(
                    y_roi=y_roi,
                    x0_roi=x0_roi,
                    x1_roi=x1_roi,
                    center_x_roi=center_x_roi,
                    width_px=width_px,
                    accepted=accepted,
                    rejection_reason=reason,
                )
            )

    return all_runs, counts


def _candidate_seeds(
    raw_votes: np.ndarray,
    smoothed_votes: np.ndarray,
    x_ref_roi: int,
    config: ProcessingConfig,
) -> list[int]:
    """Pick separated vote peaks, strongest and closest to reference first."""

    x_min = max(0, x_ref_roi - config.stripe_search_radius_px)
    x_max = min(len(smoothed_votes) - 1, x_ref_roi + config.stripe_search_radius_px)
    ranked_x = sorted(
        (x_roi for x_roi in range(x_min, x_max + 1) if raw_votes[x_roi] > 0),
        key=lambda x_roi: (-int(smoothed_votes[x_roi]), abs(x_roi - x_ref_roi)),
    )

    seeds = []
    suppression_radius = config.center_cluster_tolerance_px * 2
    for x_roi in ranked_x:
        if smoothed_votes[x_roi] <= 0:
            break
        if all(abs(x_roi - seed_x_roi) > suppression_radius for seed_x_roi in seeds):
            seeds.append(x_roi)
    return seeds


def _build_track(
    seed_x_roi: int,
    accepted_runs: list[BlackRun],
    height_roi: int,
    x_ref_roi: int,
    config: ProcessingConfig,
) -> StripeTrack | None:
    """Assign at most one nearby run per row, then remove width outliers."""

    nearest_by_row = {}
    for run in accepted_runs:
        center_distance = abs(run.center_x_roi - seed_x_roi)
        if center_distance > config.center_cluster_tolerance_px:
            continue
        previous = nearest_by_row.get(run.y_roi)
        if previous is None or center_distance < abs(previous.center_x_roi - seed_x_roi):
            nearest_by_row[run.y_roi] = run

    assigned_runs = list(nearest_by_row.values())
    if not assigned_runs:
        return None

    initial_centers = np.array([run.center_x_roi for run in assigned_runs])
    initial_widths = np.array([run.width_px for run in assigned_runs])
    center_median = float(np.median(initial_centers))
    width_median = float(np.median(initial_widths))
    width_tolerance = max(
        config.min_width_tolerance_px,
        width_median * config.max_width_deviation_ratio,
    )

    valid_runs = []
    rejected_runs = []
    track_stage_rejection_counts = {
        "center_only": 0,
        "width_only": 0,
        "center_and_width": 0,
    }
    for run in assigned_runs:
        center_rejected = (
            abs(run.center_x_roi - center_median)
            > config.center_cluster_tolerance_px
        )
        width_rejected = abs(run.width_px - width_median) > width_tolerance

        if center_rejected and width_rejected:
            reason = "center_and_width"
        elif center_rejected:
            reason = "center_only"
        elif width_rejected:
            reason = "width_only"
        else:
            valid_runs.append(run)
            continue

        track_stage_rejection_counts[reason] += 1
        rejected_runs.append((run, reason))

    if not valid_runs:
        return None

    center_x_roi = float(np.median([run.center_x_roi for run in valid_runs]))
    median_width_px = float(np.median([run.width_px for run in valid_runs]))
    assigned_row_count = len(assigned_runs)
    valid_row_count = len(valid_runs)
    valid_row_ratio = valid_row_count / height_roi
    retention_ratio = valid_row_count / assigned_row_count
    crossing_assigned_count = sum(
        run.x0_roi <= x_ref_roi < run.x1_roi for run in assigned_runs
    )
    crossing_valid_count = sum(
        run.x0_roi <= x_ref_roi < run.x1_roi for run in valid_runs
    )

    if center_x_roi < x_ref_roi:
        side = "left"
    elif center_x_roi > x_ref_roi:
        side = "right"
    else:
        side = "on_reference"

    rejection_reasons = []
    if valid_row_ratio < config.min_stripe_support_ratio:
        rejection_reasons.append("insufficient_row_support")
    if side == "on_reference":
        rejection_reasons.append("center_on_reference")

    return StripeTrack(
        track_id=-1,
        seed_x_roi=seed_x_roi,
        initial_center_median=center_median,
        initial_width_median=width_median,
        width_tolerance_px=width_tolerance,
        center_x_roi=center_x_roi,
        median_width_px=median_width_px,
        assigned_row_count=assigned_row_count,
        valid_row_count=valid_row_count,
        valid_row_ratio=valid_row_ratio,
        retention_ratio=retention_ratio,
        track_stage_rejection_counts=track_stage_rejection_counts,
        crossing_assigned_count=crossing_assigned_count,
        crossing_valid_count=crossing_valid_count,
        side=side,
        eligible=not rejection_reasons,
        rejection_reasons=rejection_reasons,
        assigned_runs=assigned_runs,
        valid_runs=valid_runs,
        rejected_runs=rejected_runs,
    )


def analyze_adjacent_stripes(
    black_mask_roi,
    x_ref_roi: int,
    x_ref_global: int,
    roi_x0_global: int,
    config: ProcessingConfig,
) -> AdjacentStripeAnalysis:
    """Find the nearest stable black-stripe track on each side of reference."""

    if black_mask_roi.ndim != 2:
        raise ValueError("black_mask_roi must be a single-channel image")

    height_roi, width_roi = black_mask_roi.shape
    if not 0 <= x_ref_roi < width_roi:
        raise ValueError("x_ref_roi is outside black_mask_roi")

    all_runs, run_filter_counts = extract_and_filter_black_runs(
        black_mask_roi, x_ref_roi, config
    )
    accepted_runs = [run for run in all_runs if run.accepted]

    raw_votes = np.zeros(width_roi, dtype=np.int32)
    for run in accepted_runs:
        vote_x_roi = int(run.center_x_roi + 0.5)
        raw_votes[vote_x_roi] += 1

    vote_window = np.ones(config.center_cluster_tolerance_px * 2 + 1, dtype=np.int32)
    smoothed_votes = np.convolve(raw_votes, vote_window, mode="same")

    tracks = []
    for seed_x_roi in _candidate_seeds(raw_votes, smoothed_votes, x_ref_roi, config):
        track = _build_track(
            seed_x_roi,
            accepted_runs,
            height_roi,
            x_ref_roi,
            config,
        )
        if track is None:
            continue
        if any(
            abs(track.center_x_roi - existing.center_x_roi)
            <= config.center_cluster_tolerance_px
            for existing in tracks
        ):
            continue
        tracks.append(track)

    tracks.sort(key=lambda track: track.center_x_roi)
    for track_id, track in enumerate(tracks):
        track.track_id = track_id

    eligible_left = [track for track in tracks if track.eligible and track.side == "left"]
    eligible_right = [track for track in tracks if track.eligible and track.side == "right"]
    left_track = max(eligible_left, key=lambda track: track.center_x_roi, default=None)
    right_track = min(eligible_right, key=lambda track: track.center_x_roi, default=None)

    if left_track is not None:
        left_track.selected = True
        left_track.selection_reason = "nearest_eligible_left"
    if right_track is not None:
        right_track.selected = True
        right_track.selection_reason = "nearest_eligible_right"

    for track in tracks:
        if track.selected:
            continue
        if track.eligible:
            track.selection_reason = "eligible_but_farther_from_reference"
        else:
            track.selection_reason = "not_eligible"

    return AdjacentStripeAnalysis(
        x_ref_roi=x_ref_roi,
        x_ref_global=x_ref_global,
        roi_x0_global=roi_x0_global,
        roi_shape=(height_roi, width_roi),
        all_runs=all_runs,
        run_filter_counts=run_filter_counts,
        raw_votes=raw_votes,
        smoothed_votes=smoothed_votes,
        tracks=tracks,
        left_track=left_track,
        right_track=right_track,
        success=left_track is not None and right_track is not None,
    )


def _track_result(track: StripeTrack | None, analysis: AdjacentStripeAnalysis):
    """Convert a selected track into JSON-safe ROI/global measurements."""

    if track is None:
        return None
    center_x_global = analysis.roi_x0_global + track.center_x_roi
    return {
        "track_id": track.track_id,
        "center_x_roi": round(track.center_x_roi, 3),
        "center_x_global": round(center_x_global, 3),
        "distance_px": round(abs(track.center_x_roi - analysis.x_ref_roi), 3),
        "median_width_px": round(track.median_width_px, 3),
        "valid_row_count": track.valid_row_count,
        "valid_row_ratio": round(track.valid_row_ratio, 6),
    }


def analysis_to_dict(
    analysis: AdjacentStripeAnalysis, config: ProcessingConfig
) -> dict:
    """Build the compact candidate, rejection, and final-result report."""

    candidates = []
    for track in analysis.tracks:
        candidates.append(
            {
                "track_id": track.track_id,
                "seed_x_roi": track.seed_x_roi,
                "seed_raw_vote": int(analysis.raw_votes[track.seed_x_roi]),
                "seed_window_vote": int(
                    analysis.smoothed_votes[track.seed_x_roi]
                ),
                "initial_center_median": round(track.initial_center_median, 3),
                "initial_width_median": round(track.initial_width_median, 3),
                "width_tolerance_px": round(track.width_tolerance_px, 3),
                "center_x_roi": round(track.center_x_roi, 3),
                "median_width_px": round(track.median_width_px, 3),
                "assigned_row_count": track.assigned_row_count,
                "valid_row_count": track.valid_row_count,
                "valid_row_ratio": round(track.valid_row_ratio, 6),
                "retention_ratio": round(track.retention_ratio, 6),
                "track_stage_rejection_counts": (
                    track.track_stage_rejection_counts
                ),
                "crossing_assigned_count": track.crossing_assigned_count,
                "crossing_valid_count": track.crossing_valid_count,
                "side": track.side,
                "eligible": track.eligible,
                "selected": track.selected,
                "selection_reason": track.selection_reason,
                "rejection_reasons": track.rejection_reasons,
            }
        )

    failure_reasons = []
    if analysis.left_track is None:
        failure_reasons.append("no_valid_left_stripe")
    if analysis.right_track is None:
        failure_reasons.append("no_valid_right_stripe")

    return {
        "algorithm": "row_run_center_voting",
        "parameters": {
            "stripe_search_radius_px": config.stripe_search_radius_px,
            "min_run_width_px": config.min_run_width_px,
            "max_run_width_px": config.max_run_width_px,
            "center_cluster_tolerance_px": config.center_cluster_tolerance_px,
            "max_width_deviation_ratio": config.max_width_deviation_ratio,
            "min_width_tolerance_px": config.min_width_tolerance_px,
            "min_stripe_support_ratio": config.min_stripe_support_ratio,
            "reject_border_touching_runs": config.reject_border_touching_runs,
        },
        "reference": {
            "x_ref_roi": analysis.x_ref_roi,
            "x_ref_global": analysis.x_ref_global,
            "roi_x0_global": analysis.roi_x0_global,
        },
        "run_filter_summary": analysis.run_filter_counts,
        "candidates": candidates,
        "result": {
            "success": analysis.success,
            "failure_reasons": failure_reasons,
            "left": _track_result(analysis.left_track, analysis),
            "right": _track_result(analysis.right_track, analysis),
        },
    }


def save_analysis_json(
    output_path: Path,
    analysis: AdjacentStripeAnalysis,
    config: ProcessingConfig,
) -> None:
    """Save candidate decisions and final measurements as readable JSON."""

    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(analysis_to_dict(analysis, config), output_file, indent=2)
    except OSError as error:
        raise OSError(f"Could not save stripe results: {output_path}") from error


def create_black_run_candidates_debug(
    black_mask_roi,
    analysis: AdjacentStripeAnalysis,
    config: ProcessingConfig,
):
    """Show prefilter votes and selected-track second-stage decisions."""

    debug_image = cv2.cvtColor(black_mask_roi, cv2.COLOR_GRAY2BGR)
    debug_image = cv2.convertScaleAbs(debug_image, alpha=0.25)

    for run in analysis.all_runs:
        color = (0, 150, 0) if run.accepted else (0, 0, 100)
        x_center = int(run.center_x_roi + 0.5)
        debug_image[run.y_roi, x_center] = color

    for track in (analysis.left_track, analysis.right_track):
        if track is None:
            continue
        for run, _reason in track.rejected_runs:
            cv2.line(
                debug_image,
                (run.x0_roi, run.y_roi),
                (run.x1_roi - 1, run.y_roi),
                (0, 140, 255),
                1,
            )

    for track, color in (
        (analysis.left_track, (255, 0, 0)),
        (analysis.right_track, (0, 255, 255)),
    ):
        if track is None:
            continue
        for run in track.valid_runs:
            cv2.line(
                debug_image,
                (run.x0_roi, run.y_roi),
                (run.x1_roi - 1, run.y_roi),
                color,
                1,
            )

    x_min = max(0, analysis.x_ref_roi - config.stripe_search_radius_px)
    x_max = min(analysis.roi_shape[1] - 1, analysis.x_ref_roi + config.stripe_search_radius_px)
    cv2.line(debug_image, (x_min, 0), (x_min, analysis.roi_shape[0] - 1), (0, 255, 255), 1)
    cv2.line(debug_image, (x_max, 0), (x_max, analysis.roi_shape[0] - 1), (0, 255, 255), 1)
    cv2.line(
        debug_image,
        (analysis.x_ref_roi, 0),
        (analysis.x_ref_roi, analysis.roi_shape[0] - 1),
        (255, 0, 255),
        1,
    )
    cv2.rectangle(
        debug_image,
        (0, 0),
        (analysis.roi_shape[1] - 1, 82),
        (0, 0, 0),
        -1,
    )
    legend = (
        ("green point: prefilter accepted (one vote)", (0, 150, 0)),
        ("dark red point: prefilter rejected", (0, 0, 180)),
        ("orange run: selected-track rejected", (0, 140, 255)),
        ("blue/yellow run: selected valid L/R", (220, 220, 220)),
    )
    for index, (text, color) in enumerate(legend):
        cv2.putText(
            debug_image,
            text,
            (8, 17 + index * 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return debug_image


def create_stripe_center_votes_debug(analysis: AdjacentStripeAnalysis):
    """Draw the smoothed x-vote curve and candidate track centers."""

    chart_height = 320
    chart_width = analysis.roi_shape[1]
    margin_top = 25
    margin_bottom = 35
    baseline_y = chart_height - margin_bottom
    plot_height = baseline_y - margin_top
    chart = np.zeros((chart_height, chart_width, 3), dtype=np.uint8)

    max_vote = int(analysis.smoothed_votes.max())
    if max_vote > 0:
        points = []
        for x_roi, vote in enumerate(analysis.smoothed_votes):
            y_chart = baseline_y - round((int(vote) / max_vote) * plot_height)
            points.append((x_roi, y_chart))
        cv2.polylines(chart, [np.array(points, dtype=np.int32)], False, (0, 220, 0), 1)

    cv2.line(chart, (0, baseline_y), (chart_width - 1, baseline_y), (100, 100, 100), 1)
    cv2.line(
        chart,
        (analysis.x_ref_roi, margin_top),
        (analysis.x_ref_roi, baseline_y),
        (0, 0, 255),
        1,
    )

    for track in analysis.tracks:
        x_track = int(track.center_x_roi + 0.5)
        if track.selected and track.side == "left":
            color = (255, 0, 0)
        elif track.selected and track.side == "right":
            color = (0, 255, 255)
        elif track.eligible:
            color = (255, 255, 255)
        else:
            color = (80, 80, 80)
        cv2.circle(chart, (x_track, margin_top + 8), 4, color, -1)

    cv2.putText(
        chart,
        "center votes",
        (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    text_y = baseline_y + 15
    for label, track, color in (
        ("L", analysis.left_track, (255, 0, 0)),
        ("R", analysis.right_track, (0, 255, 255)),
    ):
        if track is None:
            text = f"{label}: no valid track"
        else:
            text = (
                f"{label} seed={track.seed_x_roi} "
                f"rows={track.assigned_row_count}->{track.valid_row_count} "
                f"keep={track.retention_ratio:.3f}"
            )
        cv2.putText(
            chart,
            text,
            (8, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
        text_y += 16
    return chart


def create_adjacent_stripes_result_debug(
    image_gray_roi,
    analysis: AdjacentStripeAnalysis,
):
    """Overlay valid run centers and the final three vertical lines."""

    result_image = cv2.cvtColor(image_gray_roi, cv2.COLOR_GRAY2BGR)
    height_roi = image_gray_roi.shape[0]
    cv2.line(
        result_image,
        (analysis.x_ref_roi, 0),
        (analysis.x_ref_roi, height_roi - 1),
        (0, 0, 255),
        2,
    )

    for track, color in (
        (analysis.left_track, (255, 0, 0)),
        (analysis.right_track, (0, 255, 255)),
    ):
        if track is None:
            continue
        for run in track.valid_runs:
            x_run = int(run.center_x_roi + 0.5)
            cv2.circle(result_image, (x_run, run.y_roi), 1, color, -1)
        x_track = int(track.center_x_roi + 0.5)
        cv2.line(result_image, (x_track, 0), (x_track, height_roi - 1), color, 2)

    status = "SUCCESS" if analysis.success else "FAILED"
    cv2.putText(
        result_image,
        status,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0) if analysis.success else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    text_y = 50
    for label, track, color in (
        ("L", analysis.left_track, (255, 0, 0)),
        ("R", analysis.right_track, (0, 255, 255)),
    ):
        if track is None:
            text = f"{label}: no valid stripe"
        else:
            distance_px = abs(track.center_x_roi - analysis.x_ref_roi)
            text = (
                f"{label}: x={track.center_x_roi:.1f} "
                f"d={distance_px:.1f} "
                f"rows={track.assigned_row_count}->{track.valid_row_count}"
            )
        cv2.putText(
            result_image,
            text,
            (10, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        text_y += 22
    return result_image
