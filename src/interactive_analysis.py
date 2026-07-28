"""Click-aware selection built on existing stable stripe tracks."""

from dataclasses import dataclass

import numpy as np

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


def grayscale_supports_dark_click(
    image_gray_roi,
    click_x_roi: int,
    click_y_roi: int,
    config: ProcessingConfig,
) -> bool:
    """Return whether raw grayscale supports a dark click location."""

    if image_gray_roi is None:
        return False
    height, width = image_gray_roi.shape[:2]
    half_height = config.clicked_track_contrast_half_height_px
    half_width = config.clicked_track_contrast_half_width_px
    y0 = max(0, click_y_roi - half_height)
    y1 = min(height, click_y_roi + half_height + 1)
    left0 = max(0, click_x_roi - half_width)
    left1 = max(0, click_x_roi - 2)
    right0 = min(width, click_x_roi + 3)
    right1 = min(width, click_x_roi + half_width + 1)
    center0 = max(0, click_x_roi - 1)
    center1 = min(width, click_x_roi + 2)
    if left1 <= left0 or right1 <= right0 or center1 <= center0:
        return False
    center_level = float(np.median(image_gray_roi[y0:y1, center0:center1]))
    left_level = float(np.median(image_gray_roi[y0:y1, left0:left1]))
    right_level = float(np.median(image_gray_roi[y0:y1, right0:right1]))
    local_contrast = min(left_level, right_level) - center_level
    if local_contrast >= config.clicked_track_min_local_contrast:
        return True

    context_half_width = config.clicked_track_context_half_width_px
    context0 = max(0, click_x_roi - context_half_width)
    context1 = min(width, click_x_roi + context_half_width + 1)
    context = image_gray_roi[y0:y1, context0:context1]
    if context.size == 0:
        return False
    low_level = float(
        np.percentile(
            context,
            config.clicked_track_context_low_percentile,
        )
    )
    high_level = float(
        np.percentile(
            context,
            config.clicked_track_context_high_percentile,
        )
    )
    dynamic_range = high_level - low_level
    if dynamic_range < config.clicked_track_min_local_contrast:
        return False
    normalized_center = (center_level - low_level) / dynamic_range
    return (
        normalized_center
        <= config.clicked_track_dark_max_normalized_level
    )


def _has_local_dark_dip(
    image_gray_roi,
    click_x_roi: int,
    click_y_roi: int,
    config: ProcessingConfig,
) -> bool:
    """Keep the original narrow dark-dip classification behavior."""

    if image_gray_roi is None:
        return False
    height, width = image_gray_roi.shape[:2]
    half_height = config.clicked_track_contrast_half_height_px
    half_width = config.clicked_track_contrast_half_width_px
    y0 = max(0, click_y_roi - half_height)
    y1 = min(height, click_y_roi + half_height + 1)
    left0 = max(0, click_x_roi - half_width)
    left1 = max(0, click_x_roi - 2)
    right0 = min(width, click_x_roi + 3)
    right1 = min(width, click_x_roi + half_width + 1)
    center0 = max(0, click_x_roi - 1)
    center1 = min(width, click_x_roi + 2)
    if left1 <= left0 or right1 <= right0 or center1 <= center0:
        return False
    center_level = float(np.median(image_gray_roi[y0:y1, center0:center1]))
    left_level = float(np.median(image_gray_roi[y0:y1, left0:left1]))
    right_level = float(np.median(image_gray_roi[y0:y1, right0:right1]))
    local_contrast = min(left_level, right_level) - center_level
    return local_contrast >= config.clicked_track_min_local_contrast


def _faint_clicked_track(
    click_x_roi: int,
    analysis: AdjacentStripeAnalysis,
    config: ProcessingConfig,
) -> StripeTrack | None:
    """Find the low-support track associated with a verified dark dip."""

    weak_matches = [
        track
        for track in analysis.tracks
        if track.valid_row_ratio
        >= config.min_clicked_track_support_ratio
        and track.rejection_reasons == ["insufficient_row_support"]
        and abs(click_x_roi - track.center_x_roi)
        <= max(track.median_width_px / 2.0, 2.0)
    ]
    return min(
        weak_matches,
        key=lambda track: abs(click_x_roi - track.center_x_roi),
        default=None,
    )


def select_interactive_tracks(
    black_mask_roi,
    click_x_roi: int,
    click_y_roi: int,
    analysis: AdjacentStripeAnalysis,
    config: ProcessingConfig,
    image_gray_roi=None,
) -> InteractiveStripeSelection:
    """Select adjacent tracks while excluding a stably clicked black stripe."""

    height_roi, width_roi = black_mask_roi.shape[:2]
    if not 0 <= click_x_roi < width_roi:
        raise ValueError("click_x_roi is outside black_mask_roi")
    if not 0 <= click_y_roi < height_roi:
        raise ValueError("click_y_roi is outside black_mask_roi")

    clicked_pixel_is_black = bool(black_mask_roi[click_y_roi, click_x_roi])
    faint_clicked_track = None
    grayscale_has_dark_dip = False
    if not clicked_pixel_is_black:
        grayscale_has_dark_dip = _has_local_dark_dip(
            image_gray_roi,
            click_x_roi,
            click_y_roi,
            config,
        )
        if grayscale_has_dark_dip:
            faint_clicked_track = _faint_clicked_track(
                click_x_roi,
                analysis,
                config,
            )
        clicked_pixel_is_black = grayscale_has_dark_dip
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
    clicked_track = faint_clicked_track
    if clicked_track is None:
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

    left_neighbor = max(
        (
            track
            for track in stable_tracks
            if track.center_x_roi < clicked_track.center_x_roi
        ),
        key=lambda track: track.center_x_roi,
        default=None,
    )
    right_neighbor = min(
        (
            track
            for track in stable_tracks
            if track.center_x_roi > clicked_track.center_x_roi
        ),
        key=lambda track: track.center_x_roi,
        default=None,
    )
    # The clicked black track is only the reference track.  The final pair
    # must be its nearest stable neighbors, one on each side.
    left_track = left_neighbor
    right_track = right_neighbor
    failure_reasons = _missing_side_reasons(left_track, right_track)
    warning_flags = list(failure_reasons)
    if faint_clicked_track is not None:
        warning_flags.append("faint_clicked_track_low_support")
    return InteractiveStripeSelection(
        click_classification="black_stripe",
        clicked_track=clicked_track,
        left_track=left_track,
        right_track=right_track,
        success=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
        warning_flags=tuple(warning_flags),
    )
