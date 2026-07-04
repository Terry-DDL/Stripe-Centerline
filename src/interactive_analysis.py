"""Click-aware selection built on existing stable stripe tracks."""

from dataclasses import dataclass

from config import ProcessingConfig
from stripe_analysis import AdjacentStripeAnalysis, StripeTrack


@dataclass(frozen=True)
class InteractiveStripeSelection:
    """Final interactive selection without mutating the legacy analysis."""

    click_classification: str
    clicked_track: StripeTrack | None
    left_track: StripeTrack | None
    right_track: StripeTrack | None
    success: bool
    failure_reasons: tuple[str, ...]
    warning_flags: tuple[str, ...]


def _stable_for_interactive_selection(track: StripeTrack) -> bool:
    """Accept supported tracks, including a track exactly on reference."""

    return track.rejection_reasons in ([], ["center_on_reference"])


def _missing_side_reasons(
    left_track: StripeTrack | None,
    right_track: StripeTrack | None,
) -> list[str]:
    reasons = []
    if left_track is None:
        reasons.append("no_valid_left_stripe")
    if right_track is None:
        reasons.append("no_valid_right_stripe")
    return reasons


def select_interactive_tracks(
    black_mask_roi,
    click_x_roi: int,
    click_y_roi: int,
    analysis: AdjacentStripeAnalysis,
    config: ProcessingConfig,
) -> InteractiveStripeSelection:
    """Select adjacent tracks while excluding a stably clicked black stripe."""

    height_roi, width_roi = black_mask_roi.shape[:2]
    if not 0 <= click_x_roi < width_roi:
        raise ValueError("click_x_roi is outside black_mask_roi")
    if not 0 <= click_y_roi < height_roi:
        raise ValueError("click_y_roi is outside black_mask_roi")

    clicked_pixel_is_black = bool(black_mask_roi[click_y_roi, click_x_roi])
    if not clicked_pixel_is_black:
        left_track = analysis.left_track
        right_track = analysis.right_track
        failure_reasons = _missing_side_reasons(left_track, right_track)
        return InteractiveStripeSelection(
            click_classification="white_region",
            clicked_track=None,
            left_track=left_track,
            right_track=right_track,
            success=not failure_reasons,
            failure_reasons=tuple(failure_reasons),
            warning_flags=tuple(failure_reasons),
        )

    stable_tracks = [
        track
        for track in analysis.tracks
        if track.valid_row_ratio >= config.min_stripe_support_ratio
        and _stable_for_interactive_selection(track)
    ]
    matching_tracks = [
        track
        for track in stable_tracks
        if abs(click_x_roi - track.center_x_roi) <= track.median_width_px / 2.0
    ]
    clicked_track = min(
        matching_tracks,
        key=lambda track: abs(click_x_roi - track.center_x_roi),
        default=None,
    )
    if clicked_track is None:
        reason = "clicked_black_region_not_stable_track"
        return InteractiveStripeSelection(
            click_classification="ambiguous_black_region",
            clicked_track=None,
            left_track=None,
            right_track=None,
            success=False,
            failure_reasons=(reason,),
            warning_flags=(reason,),
        )

    left_track = max(
        (
            track
            for track in stable_tracks
            if track.track_id != clicked_track.track_id
            and track.center_x_roi < clicked_track.center_x_roi
        ),
        key=lambda track: track.center_x_roi,
        default=None,
    )
    right_track = min(
        (
            track
            for track in stable_tracks
            if track.track_id != clicked_track.track_id
            and track.center_x_roi > clicked_track.center_x_roi
        ),
        key=lambda track: track.center_x_roi,
        default=None,
    )
    failure_reasons = _missing_side_reasons(left_track, right_track)
    return InteractiveStripeSelection(
        click_classification="black_stripe",
        clicked_track=clicked_track,
        left_track=left_track,
        right_track=right_track,
        success=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
        warning_flags=tuple(failure_reasons),
    )
