"""Explainable quality fields and ordering for detection candidates."""

from __future__ import annotations

import math

import numpy as np

from config import ProcessingConfig
from interactive_analysis import InteractiveStripeSelection
from stripe_analysis import AdjacentStripeAnalysis, StripeTrack


PITCH_RANK = {
    "Not applicable": -1,
    "Suspicious": 0,
    "Unable to verify": 1,
    "Normal": 2,
}

ADJACENCY_RANK = {
    "rejected": 0,
    "unverified": 1,
    "verified": 2,
}

TOPOLOGY_RANK = {
    "Contradictory": 0,
    "Not considered": 1,
    "Unable to verify": 1,
    "Consistent": 2,
}


def _track_metrics(track: StripeTrack | None) -> dict | None:
    if track is None:
        return None
    centers = np.array(
        [run.center_x_roi for run in track.valid_runs],
        dtype=np.float64,
    )
    widths = np.array(
        [run.width_px for run in track.valid_runs],
        dtype=np.float64,
    )
    center_mad = float(
        np.median(np.abs(centers - np.median(centers)))
    )
    width_median = float(np.median(widths))
    width_mad = float(np.median(np.abs(widths - width_median)))
    return {
        "track_id": track.track_id,
        "center_x_roi": round(track.center_x_roi, 3),
        "valid_row_ratio": round(track.valid_row_ratio, 6),
        "retention_ratio": round(track.retention_ratio, 6),
        "center_mad_px": round(center_mad, 6),
        "median_width_px": round(track.median_width_px, 6),
        "normalized_width_mad": round(
            width_mad / max(width_median, 1e-12),
            6,
        ),
        "crossing_valid_count": track.crossing_valid_count,
        "crossing_valid_ratio": round(
            track.crossing_valid_count
            / max(track.valid_row_count, 1),
            6,
        ),
    }


def combined_pitch_status(
    selection: InteractiveStripeSelection,
    local_neighbor_check: dict,
    pitch_guard: dict,
) -> tuple[str, list[str]]:
    """Combine local and whole-image pitch without making either absolute."""

    reasons = []
    if not selection.success:
        return "Not applicable", ["selection_not_successful"]
    if local_neighbor_check.get("passed") is False:
        reasons.append("local_neighbor_spacing_suspicious")
    elif not local_neighbor_check.get("checked"):
        reasons.append("local_neighbor_spacing_unverifiable")
    global_status = pitch_guard.get("status", "Unable to verify")
    if global_status == "Suspicious":
        reasons.append("whole_image_pitch_suspicious")
    elif global_status in ("Unable to verify", "Not applicable"):
        reasons.append("whole_image_pitch_unverifiable")
    if "local_neighbor_spacing_suspicious" in reasons or (
        "whole_image_pitch_suspicious" in reasons
    ):
        return "Suspicious", reasons
    if reasons:
        return "Unable to verify", reasons
    return "Normal", reasons


def _selected_tracks(
    selection: InteractiveStripeSelection,
) -> list[tuple[str, StripeTrack | None]]:
    tracks = [
        ("left", selection.left_track),
        ("right", selection.right_track),
    ]
    if selection.click_classification == "black_stripe":
        tracks.insert(1, ("clicked", selection.clicked_track))
    return tracks


def _hard_invalid_reasons(
    selection: InteractiveStripeSelection,
    click_x_roi: int,
) -> list[str]:
    reasons = []
    if not selection.success:
        reasons.extend(selection.failure_reasons)
        return reasons
    left = selection.left_track
    right = selection.right_track
    if left is None or right is None:
        return ["missing_required_track"]
    if not left.center_x_roi < click_x_roi < right.center_x_roi:
        reasons.append("selected_tracks_do_not_straddle_click")
    if left.crossing_valid_count or right.crossing_valid_count:
        reasons.append("outer_track_crosses_click_column")
    if selection.click_classification == "black_stripe":
        clicked = selection.clicked_track
        if clicked is None:
            reasons.append("missing_clicked_track")
        else:
            if not (
                left.center_x_roi
                < clicked.center_x_roi
                < right.center_x_roi
            ):
                reasons.append("clicked_track_not_between_neighbors")
            if clicked.crossing_valid_count == 0:
                reasons.append("clicked_track_does_not_cross_click_column")
    elif selection.clicked_track is not None:
        reasons.append("white_click_has_clicked_track")
    return reasons


def build_candidate_quality(
    selection: InteractiveStripeSelection,
    analysis: AdjacentStripeAnalysis,
    local_neighbor_check: dict,
    pitch_guard: dict,
    grayscale_topology: dict,
    click_x_roi: int,
    processing_config: ProcessingConfig,
) -> dict:
    """Build all visible quality fields used by lexicographic arbitration."""

    hard_invalid_reasons = _hard_invalid_reasons(selection, click_x_roi)
    track_metrics = {
        label: _track_metrics(track)
        for label, track in _selected_tracks(selection)
    }
    present_metrics = [
        metrics for metrics in track_metrics.values() if metrics is not None
    ]
    minimum_valid_row_ratio = min(
        (
            metrics["valid_row_ratio"]
            for metrics in present_metrics
        ),
        default=0.0,
    )
    minimum_retention_ratio = min(
        (
            metrics["retention_ratio"]
            for metrics in present_metrics
        ),
        default=0.0,
    )
    maximum_center_mad_px = max(
        (metrics["center_mad_px"] for metrics in present_metrics),
        default=math.inf,
    )
    maximum_normalized_width_mad = max(
        (
            metrics["normalized_width_mad"]
            for metrics in present_metrics
        ),
        default=math.inf,
    )
    if (
        selection.click_classification == "black_stripe"
        and selection.clicked_track is not None
    ):
        click_association_distance = abs(
            selection.clicked_track.center_x_roi - click_x_roi
        )
    elif (
        selection.left_track is not None
        and selection.right_track is not None
    ):
        midpoint = (
            selection.left_track.center_x_roi
            + selection.right_track.center_x_roi
        ) / 2.0
        click_association_distance = abs(midpoint - click_x_roi)
    else:
        click_association_distance = math.inf

    low_support_count = sum(
        track.valid_row_ratio < processing_config.min_stripe_support_ratio
        for track in analysis.tracks
    )
    stable_widths = [
        track.median_width_px
        for track in analysis.tracks
        if track.valid_row_ratio
        >= processing_config.min_stripe_support_ratio
    ]
    typical_width = (
        float(np.median(stable_widths)) if stable_widths else None
    )
    fragment_count = low_support_count
    if typical_width is not None:
        fragment_count += sum(
            track.valid_row_ratio
            >= processing_config.min_stripe_support_ratio
            and track.median_width_px < typical_width * 0.5
            for track in analysis.tracks
        )

    pitch_status, pitch_reasons = combined_pitch_status(
        selection,
        local_neighbor_check,
        pitch_guard,
    )
    successful = selection.success and not hard_invalid_reasons
    return {
        "success": successful,
        "hard_valid": not hard_invalid_reasons,
        "hard_invalid_reasons": hard_invalid_reasons,
        "adjacency_verification_status": "unverified",
        "combined_pitch_status": pitch_status,
        "combined_pitch_reasons": pitch_reasons,
        "grayscale_topology_status": grayscale_topology.get(
            "status",
            "Unable to verify",
        ),
        "minimum_valid_row_ratio": round(minimum_valid_row_ratio, 6),
        "minimum_retention_ratio": round(minimum_retention_ratio, 6),
        "maximum_center_mad_px": (
            None
            if not math.isfinite(maximum_center_mad_px)
            else round(maximum_center_mad_px, 6)
        ),
        "maximum_normalized_width_mad": (
            None
            if not math.isfinite(maximum_normalized_width_mad)
            else round(maximum_normalized_width_mad, 6)
        ),
        "click_association_distance_px": (
            None
            if not math.isfinite(click_association_distance)
            else round(click_association_distance, 6)
        ),
        "candidate_track_count": len(analysis.tracks),
        "low_support_track_count": low_support_count,
        "suspected_fragment_count": fragment_count,
        "track_metrics": track_metrics,
    }


def candidate_quality_key(
    quality: dict,
    threshold_method: str | None = None,
    include_topology: bool = True,
) -> tuple:
    """Return the documented quality sequence without a hidden score."""

    center_mad = quality["maximum_center_mad_px"]
    width_mad = quality["maximum_normalized_width_mad"]
    click_distance = quality["click_association_distance_px"]
    key_parts = [
        ADJACENCY_RANK[quality["adjacency_verification_status"]],
        PITCH_RANK[quality["combined_pitch_status"]],
    ]
    if include_topology:
        key_parts.append(
            TOPOLOGY_RANK[quality["grayscale_topology_status"]]
        )
    key_parts.extend(
        (
            bool(quality["success"]),
            quality["minimum_valid_row_ratio"],
            quality["minimum_retention_ratio"],
            -math.inf if center_mad is None else -center_mad,
            -math.inf if width_mad is None else -width_mad,
            -math.inf if click_distance is None else -click_distance,
            -quality["suspected_fragment_count"],
            -quality["low_support_track_count"],
        )
    )
    key = tuple(key_parts)
    if threshold_method is None:
        return key
    return key + (threshold_method == "otsu",)


def quality_difference_reason(
    winner_quality: dict,
    loser_quality: dict,
    include_topology: bool = True,
) -> str:
    """Name the first documented quality field that decided a comparison."""

    fields = (
        ("adjacency_verification_status", True),
        ("combined_pitch_status", True),
        ("grayscale_topology_status", True),
        ("success", True),
        ("minimum_valid_row_ratio", True),
        ("minimum_retention_ratio", True),
        ("maximum_center_mad_px", False),
        ("maximum_normalized_width_mad", False),
        ("click_association_distance_px", False),
        ("suspected_fragment_count", False),
        ("low_support_track_count", False),
    )
    for field, higher_is_better in fields:
        if field == "grayscale_topology_status" and not include_topology:
            continue
        winner = winner_quality[field]
        loser = loser_quality[field]
        if winner == loser:
            continue
        if field == "adjacency_verification_status":
            return "better_adjacency_verification"
        if field == "grayscale_topology_status":
            return "better_grayscale_topology_status"
        if higher_is_better:
            return f"better_{field}"
        return f"lower_{field}"
    return "quality_equal"
