"""Interactive point-centered pipeline built around the legacy detector."""

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

import cv2
import numpy as np

from candidate_quality import (
    build_candidate_quality,
    candidate_quality_key,
    quality_difference_reason,
)
from config import InteractiveConfig, ProcessingConfig
from grayscale_topology import evaluate_grayscale_topology
from image_processing import (
    RoiBoundsGlobal,
    apply_vertical_close_roi,
    create_black_mask_roi,
    create_close_delta_roi,
    create_otsu_binary_roi,
    crop_roi_global,
    gaussian_blur_roi,
    save_debug_image,
)
from interactive_analysis import (
    InteractiveStripeSelection,
    select_interactive_tracks,
)
from pitch_reference import (
    PitchReferenceMap,
    build_pitch_reference_map,
    create_pitch_reference_debug,
    evaluate_pitch_guard,
    validate_pitch_reference_config,
)
from stripe_analysis import (
    AdjacentStripeAnalysis,
    StripeTrack,
    analysis_to_dict,
    analyze_adjacent_stripes,
)

ALGORITHM_REVISION = "adjacency_gate_v1_clicked_hypothesis_v1"


@dataclass(frozen=True)
class RotationShadowResult:
    """Explainable angle estimate used by guarded rotation detection."""

    best_angle_deg: float
    zero_angle_score: float
    best_score: float
    peak_separation: float
    angles_deg: np.ndarray
    scores: np.ndarray
    preview_image: np.ndarray
    score_chart: np.ndarray


@dataclass(frozen=True)
class RoiProcessingStages:
    """Preprocessing images for one unrotated or corrected ROI."""

    image_gray: np.ndarray
    threshold_method: str
    threshold_binary: np.ndarray
    threshold_warning_flags: tuple[str, ...]
    vertical_close: np.ndarray
    close_delta: np.ndarray
    black_mask: np.ndarray

    @property
    def otsu_binary(self) -> np.ndarray:
        """Keep the old debug-image attribute available to callers."""

        return self.threshold_binary


@dataclass(frozen=True)
class DetectionCandidate:
    """One geometry + threshold result before layered arbitration."""

    geometry: str
    threshold_method: str
    stages: RoiProcessingStages
    analysis: AdjacentStripeAnalysis
    selection: InteractiveStripeSelection
    neighbor_consistency: dict
    adjacency_verification: dict
    pitch_guard: dict
    neighbor_recovery: dict
    grayscale_topology: dict
    grayscale_topology_debug: np.ndarray
    quality: dict
    rotation_matrix: np.ndarray
    inverse_rotation_matrix: np.ndarray


@dataclass(frozen=True)
class InteractivePipelineResult:
    """Data and debug images produced for one clicked point."""

    click_x_global: int
    click_y_global: int
    click_x_roi: int
    click_y_roi: int
    bounds_global: RoiBoundsGlobal
    legacy_analysis: AdjacentStripeAnalysis
    active_analysis: AdjacentStripeAnalysis
    selection: InteractiveStripeSelection
    rotation_shadow: RotationShadowResult
    rotation_applied: bool
    report: dict
    debug_images: dict[str, np.ndarray]
    output_dir: Path
    pitch_reference_map: PitchReferenceMap


def validate_interactive_config(config: InteractiveConfig) -> None:
    """Validate settings that belong only to the interactive workflow."""

    validate_pitch_reference_config(config)
    if config.roi_half_width_px <= 0:
        raise ValueError("roi_half_width_px must be greater than 0")
    if config.roi_half_height_px <= 0:
        raise ValueError("roi_half_height_px must be greater than 0")
    if config.selection_zoom_half_width_px <= 0:
        raise ValueError(
            "selection_zoom_half_width_px must be greater than 0"
        )
    if config.selection_zoom_half_height_px <= 0:
        raise ValueError(
            "selection_zoom_half_height_px must be greater than 0"
        )
    if config.selection_zoom_scale <= 1.0:
        raise ValueError("selection_zoom_scale must be greater than 1")
    if config.rotation_min_angle_deg > config.rotation_max_angle_deg:
        raise ValueError("rotation_min_angle_deg must not exceed maximum")
    if config.rotation_angle_step_deg <= 0.0:
        raise ValueError("rotation_angle_step_deg must be greater than 0")
    if not (
        config.rotation_min_angle_deg <= 0.0
        <= config.rotation_max_angle_deg
    ):
        raise ValueError("rotation shadow angle range must include zero")
    if config.rotation_min_abs_angle_deg < 0.0:
        raise ValueError("rotation_min_abs_angle_deg must not be negative")
    if config.rotation_min_relative_score_gain < 0.0:
        raise ValueError(
            "rotation_min_relative_score_gain must not be negative"
        )
    if config.rotation_min_peak_separation < 0.0:
        raise ValueError("rotation_min_peak_separation must not be negative")
    if (
        not isinstance(config.adaptive_threshold_block_size, int)
        or isinstance(config.adaptive_threshold_block_size, bool)
    ):
        raise ValueError(
            "adaptive_threshold_block_size must be an integer"
        )
    if config.adaptive_threshold_block_size < 3:
        raise ValueError(
            "adaptive_threshold_block_size must be at least 3"
        )
    if config.adaptive_threshold_block_size % 2 == 0:
        raise ValueError(
            "adaptive_threshold_block_size must be odd"
        )
    if not isinstance(config.adaptive_threshold_enabled, bool):
        raise ValueError("adaptive_threshold_enabled must be boolean")
    if not isinstance(config.adaptive_threshold_c, (int, float)) or (
        not math.isfinite(config.adaptive_threshold_c)
    ):
        raise ValueError("adaptive_threshold_c must be finite")
    if config.neighbor_max_span_pitch_ratio <= 1.0:
        raise ValueError(
            "neighbor_max_span_pitch_ratio must be greater than 1"
        )
    if config.neighbor_min_pitch_track_count < 3:
        raise ValueError(
            "neighbor_min_pitch_track_count must be at least 3"
        )
    if not isinstance(
        config.enable_grayscale_topology_rejection,
        bool,
    ):
        raise ValueError(
            "enable_grayscale_topology_rejection must be boolean"
        )
    for field_name in (
        "outer_neighbor_recovery_min_support_ratio",
        "neighbor_recovery_max_pitch_error_ratio",
    ):
        value = getattr(config, field_name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name} must be between 0 and 1")
    if config.clicked_hypothesis_max_center_distance_px <= 0.0:
        raise ValueError(
            "clicked_hypothesis_max_center_distance_px "
            "must be greater than 0"
        )
    if config.topology_shadow_min_informative_rows < 1:
        raise ValueError(
            "topology_shadow_min_informative_rows must be at least 1"
        )
    for field_name in (
        "topology_shadow_min_separator_support_ratio",
        "topology_shadow_strong_same_basin_support_ratio",
        "topology_shadow_strong_merged_basin_support_ratio",
        "topology_shadow_min_vertical_span_ratio",
        "topology_shadow_separator_prominence_ratio",
        "topology_shadow_same_basin_prominence_ratio",
    ):
        value = getattr(config, field_name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name} must be between 0 and 1")
    if config.topology_shadow_min_dynamic_range <= 0.0:
        raise ValueError(
            "topology_shadow_min_dynamic_range must be greater than 0"
        )
    if config.topology_shadow_noise_mad_multiplier <= 0.0:
        raise ValueError(
            "topology_shadow_noise_mad_multiplier must be greater than 0"
        )
    if config.display_max_width_px <= 0:
        raise ValueError("display_max_width_px must be greater than 0")


def calculate_interactive_roi_bounds(
    image_shape: tuple[int, ...],
    click_x_global: int,
    click_y_global: int,
    config: InteractiveConfig,
) -> RoiBoundsGlobal:
    """Return a fixed-pixel ROI centered on the click and clipped to image."""

    height_global, width_global = image_shape[:2]
    if not isinstance(click_x_global, int) or isinstance(click_x_global, bool):
        raise ValueError("click_x_global must be an integer")
    if not isinstance(click_y_global, int) or isinstance(click_y_global, bool):
        raise ValueError("click_y_global must be an integer")
    if not 0 <= click_x_global < width_global:
        raise ValueError("click_x_global is outside the image")
    if not 0 <= click_y_global < height_global:
        raise ValueError("click_y_global is outside the image")

    return RoiBoundsGlobal(
        x0_global=max(0, click_x_global - config.roi_half_width_px),
        x1_global=min(
            width_global, click_x_global + config.roi_half_width_px
        ),
        y0_global=max(0, click_y_global - config.roi_half_height_px),
        y1_global=min(
            height_global, click_y_global + config.roi_half_height_px
        ),
    )


def _effective_adaptive_block_size(
    roi_shape: tuple[int, ...],
    config: InteractiveConfig,
) -> tuple[int | None, tuple[str, ...]]:
    """Return a legal adaptive block size for this ROI."""

    short_side = min(roi_shape[:2])
    requested = config.adaptive_threshold_block_size
    if short_side <= 3:
        return None, ("adaptive_disabled_roi_too_small",)
    if requested < short_side:
        return requested, ()
    effective = short_side - 1
    if effective % 2 == 0:
        effective -= 1
    if effective < 3:
        return None, ("adaptive_disabled_roi_too_small",)
    return effective, ("adaptive_block_size_reduced_for_roi",)


def _preprocess_roi(
    image_gray_roi,
    config: ProcessingConfig,
    interactive_config: InteractiveConfig,
    threshold_method: str,
) -> RoiProcessingStages:
    """Run one configured threshold method and shared morphology."""

    blurred_roi = gaussian_blur_roi(image_gray_roi, config)
    warning_flags = ()
    if threshold_method == "otsu":
        threshold_binary_roi = create_otsu_binary_roi(blurred_roi)
    elif threshold_method == "adaptive":
        block_size, warning_flags = _effective_adaptive_block_size(
            image_gray_roi.shape,
            interactive_config,
        )
        if block_size is None:
            raise ValueError("adaptive threshold is unavailable for this ROI")
        threshold_binary_roi = cv2.adaptiveThreshold(
            blurred_roi,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            interactive_config.adaptive_threshold_c,
        )
    else:
        raise ValueError(f"unsupported threshold method: {threshold_method}")
    vertical_close_roi = apply_vertical_close_roi(
        threshold_binary_roi,
        config,
    )
    close_delta_roi = create_close_delta_roi(
        threshold_binary_roi, vertical_close_roi
    )
    black_mask_roi = create_black_mask_roi(vertical_close_roi)
    return RoiProcessingStages(
        image_gray=image_gray_roi,
        threshold_method=threshold_method,
        threshold_binary=threshold_binary_roi,
        threshold_warning_flags=warning_flags,
        vertical_close=vertical_close_roi,
        close_delta=close_delta_roi,
        black_mask=black_mask_roi,
    )


def _rotate_same_canvas(image, angle_deg: float):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(
        ((width - 1) / 2.0, (height - 1) / 2.0),
        angle_deg,
        1.0,
    )
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _common_rotation_crop(image, max_abs_angle_deg: float):
    """Remove a conservative border shared by every tested rotation."""

    height, width = image.shape[:2]
    angle_rad = math.radians(max_abs_angle_deg)
    margin_x = math.ceil(
        width * (1.0 - math.cos(angle_rad)) / 2.0
        + height * math.sin(angle_rad) / 2.0
    )
    margin_y = math.ceil(
        height * (1.0 - math.cos(angle_rad)) / 2.0
        + width * math.sin(angle_rad) / 2.0
    )
    margin_x = min(margin_x, max(0, width // 2 - 2))
    margin_y = min(margin_y, max(0, height // 2 - 2))
    return image[
        margin_y : height - margin_y,
        margin_x : width - margin_x,
    ]


def _rotation_score(rotated_binary, max_abs_angle_deg: float) -> float:
    common = _common_rotation_crop(rotated_binary, max_abs_angle_deg)
    if common.size == 0 or common.shape[1] < 2:
        return 0.0
    column_profile = np.mean(common.astype(np.float32) / 255.0, axis=0)
    return float(np.std(column_profile))


def _create_rotation_score_chart(
    angles_deg: np.ndarray,
    scores: np.ndarray,
    best_index: int,
) -> np.ndarray:
    chart_height = 320
    chart_width = 720
    left = 55
    right = 18
    top = 30
    bottom = 45
    chart = np.zeros((chart_height, chart_width, 3), dtype=np.uint8)
    cv2.line(
        chart,
        (left, chart_height - bottom),
        (chart_width - right, chart_height - bottom),
        (120, 120, 120),
        1,
    )
    score_min = float(scores.min())
    score_range = max(float(scores.max()) - score_min, 1e-12)
    points = []
    for index, score in enumerate(scores):
        x_chart = left + round(
            index * (chart_width - left - right) / max(len(scores) - 1, 1)
        )
        y_chart = chart_height - bottom - round(
            (float(score) - score_min)
            / score_range
            * (chart_height - top - bottom)
        )
        points.append((x_chart, y_chart))
    cv2.polylines(
        chart,
        [np.array(points, dtype=np.int32)],
        False,
        (0, 220, 0),
        1,
    )
    best_point = points[best_index]
    cv2.circle(chart, best_point, 5, (0, 255, 255), -1)
    zero_index = int(np.argmin(np.abs(angles_deg)))
    cv2.circle(chart, points[zero_index], 4, (0, 0, 255), -1)
    cv2.putText(
        chart,
        "Rotation score: yellow=best, red=0 deg",
        (12, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        chart,
        f"{angles_deg[0]:.2f} deg",
        (left - 25, chart_height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        chart,
        f"{angles_deg[-1]:.2f} deg",
        (chart_width - 85, chart_height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return chart


def estimate_rotation_shadow(
    binary_roi,
    config: InteractiveConfig,
) -> RotationShadowResult:
    """Estimate a correction angle for guarded rotation detection."""

    angles_deg = np.arange(
        config.rotation_min_angle_deg,
        config.rotation_max_angle_deg
        + config.rotation_angle_step_deg / 2.0,
        config.rotation_angle_step_deg,
        dtype=np.float64,
    )
    max_abs_angle = max(abs(float(angles_deg[0])), abs(float(angles_deg[-1])))
    rotated_images = []
    scores = []
    for angle_deg in angles_deg:
        rotated = _rotate_same_canvas(binary_roi, float(angle_deg))
        rotated_images.append(rotated)
        scores.append(_rotation_score(rotated, max_abs_angle))
    scores_array = np.array(scores, dtype=np.float64)
    best_index = int(np.argmax(scores_array))
    best_score = float(scores_array[best_index])
    zero_index = int(np.argmin(np.abs(angles_deg)))
    other_scores = [
        float(score)
        for index, score in enumerate(scores_array)
        if abs(index - best_index) > 1
    ]
    second_score = max(other_scores, default=float(scores_array[zero_index]))
    peak_separation = (
        max(0.0, best_score - second_score) / best_score
        if best_score > 0.0
        else 0.0
    )
    return RotationShadowResult(
        best_angle_deg=float(angles_deg[best_index]),
        zero_angle_score=float(scores_array[zero_index]),
        best_score=best_score,
        peak_separation=peak_separation,
        angles_deg=angles_deg,
        scores=scores_array,
        preview_image=rotated_images[best_index],
        score_chart=_create_rotation_score_chart(
            angles_deg, scores_array, best_index
        ),
    )


def _relative_rotation_score_gain(rotation: RotationShadowResult) -> float:
    if rotation.zero_angle_score <= 0.0:
        return 0.0
    return max(
        0.0,
        (rotation.best_score - rotation.zero_angle_score)
        / rotation.zero_angle_score,
    )


def _rotation_guard_reasons(
    rotation: RotationShadowResult,
    roi_warning_flags: list[str],
    config: InteractiveConfig,
) -> list[str]:
    """Explain why an estimated angle should not alter detection."""

    reasons = []
    if not config.rotation_apply_enabled:
        reasons.append("rotation_application_disabled")
    if abs(rotation.best_angle_deg) < config.rotation_min_abs_angle_deg:
        reasons.append("rotation_angle_below_minimum")
    if (
        _relative_rotation_score_gain(rotation)
        < config.rotation_min_relative_score_gain
    ):
        reasons.append("rotation_score_gain_below_minimum")
    if rotation.peak_separation < config.rotation_min_peak_separation:
        reasons.append("rotation_peak_separation_below_minimum")
    if np.isclose(
        rotation.best_angle_deg,
        config.rotation_min_angle_deg,
    ) or np.isclose(
        rotation.best_angle_deg,
        config.rotation_max_angle_deg,
    ):
        reasons.append("rotation_best_angle_at_search_boundary")
    if any(flag.startswith("roi_clipped_") for flag in roi_warning_flags):
        reasons.append("rotation_not_applied_to_clipped_roi")
    return reasons


def _neighbor_consistency_check(
    selection: InteractiveStripeSelection,
    analysis: AdjacentStripeAnalysis,
    processing_config: ProcessingConfig,
    interactive_config: InteractiveConfig,
) -> dict:
    """Check that every selected neighbor is only one stripe pitch away."""

    details = {
        "checked": False,
        "passed": None,
        "local_pitch_px": None,
        "selected_span_px": None,
        "span_pitch_ratio": None,
        "max_span_pitch_ratio": (
            interactive_config.neighbor_max_span_pitch_ratio
        ),
        "pitch_track_count": 0,
        "reason": None,
    }
    if not selection.success:
        details["reason"] = "selection_not_successful"
        return details
    if (
        selection.left_track is None
        or selection.right_track is None
    ):
        details["reason"] = "missing_selected_side"
        return details

    stable_tracks = [
        track
        for track in analysis.tracks
        if track.valid_row_ratio
        >= processing_config.min_stripe_support_ratio
        and track.rejection_reasons in ([], ["center_on_reference"])
    ]
    if len(stable_tracks) < interactive_config.neighbor_min_pitch_track_count:
        details["reason"] = "insufficient_tracks_for_pitch_check"
        return details

    width_statistics = _regular_width_statistics(
        stable_tracks,
        processing_config,
    )
    min_normal_width = width_statistics["regular_min"]
    max_normal_width = width_statistics["regular_max"]
    pitch_tracks = [
        track
        for track in analysis.tracks
        if min_normal_width
        <= track.median_width_px
        <= max_normal_width
    ]
    if len(pitch_tracks) < interactive_config.neighbor_min_pitch_track_count:
        details["reason"] = "insufficient_regular_width_tracks"
        return details

    centers = sorted(track.center_x_roi for track in pitch_tracks)
    gaps = [
        right - left
        for left, right in zip(centers, centers[1:])
        if right - left > processing_config.center_cluster_tolerance_px
    ]
    if len(gaps) < interactive_config.neighbor_min_pitch_track_count - 1:
        details["reason"] = "insufficient_pitch_gaps"
        return details

    local_pitch = float(np.median(gaps))
    if (
        selection.click_classification == "black_stripe"
        and selection.clicked_track is not None
    ):
        intervals = [
            (
                selection.left_track.center_x_roi,
                selection.clicked_track.center_x_roi,
            ),
            (
                selection.clicked_track.center_x_roi,
                selection.right_track.center_x_roi,
            ),
        ]
    else:
        intervals = [
            (
                selection.left_track.center_x_roi,
                selection.right_track.center_x_roi,
            )
        ]

    interval_spans = [right - left for left, right in intervals]
    span_pitch_ratios = [
        span / max(local_pitch, 1e-12) for span in interval_spans
    ]
    selected_span = max(interval_spans)
    span_pitch_ratio = max(span_pitch_ratios)
    passed = all(
        ratio <= interactive_config.neighbor_max_span_pitch_ratio
        for ratio in span_pitch_ratios
    )
    if not passed:
        reason = "selected_span_exceeds_immediate_neighbor_limit"
    else:
        reason = None
    details.update(
        {
            "checked": True,
            "passed": passed,
            "local_pitch_px": round(local_pitch, 3),
            "selected_span_px": round(selected_span, 3),
            "span_pitch_ratio": round(span_pitch_ratio, 3),
            "pitch_track_count": len(pitch_tracks),
            "reason": reason,
        }
    )
    return details


def _regular_width_statistics(
    stable_tracks: list[StripeTrack],
    processing_config: ProcessingConfig,
) -> dict | None:
    """Return the shared regular-width range for a stable-track set."""

    if not stable_tracks:
        return None
    stable_widths = [track.median_width_px for track in stable_tracks]
    typical_width = float(np.median(stable_widths))
    return {
        "typical_width": typical_width,
        "regular_min": typical_width * 0.5,
        "regular_max": (
            typical_width
            * (1.0 + processing_config.max_width_deviation_ratio)
            + processing_config.min_width_tolerance_px
        ),
        "stable_track_count": len(stable_tracks),
    }


def _apply_neighbor_consistency_guard(
    selection: InteractiveStripeSelection,
    analysis: AdjacentStripeAnalysis,
    processing_config: ProcessingConfig,
    interactive_config: InteractiveConfig,
) -> tuple[InteractiveStripeSelection, dict]:
    """Attach a neighbor warning without discarding detected centerlines."""

    details = _neighbor_consistency_check(
        selection,
        analysis,
        processing_config,
        interactive_config,
    )
    warning_reason = None
    if details["passed"] is False:
        warning_reason = "selected_stripes_not_immediate_neighbors"
    elif (
        selection.success
        and not details["checked"]
    ):
        warning_reason = "neighbor_consistency_not_verifiable"

    if warning_reason is None:
        return selection, details

    return (
        InteractiveStripeSelection(
            click_classification=selection.click_classification,
            clicked_track=selection.clicked_track,
            left_track=selection.left_track,
            right_track=selection.right_track,
            success=selection.success,
            failure_reasons=selection.failure_reasons,
            warning_flags=selection.warning_flags + (warning_reason,),
        ),
        details,
    )


def _build_adjacency_verification(
    selection: InteractiveStripeSelection,
    neighbor_consistency: dict,
    quality: dict,
) -> dict:
    """Describe whether an existing candidate proves immediate adjacency."""

    if neighbor_consistency.get("passed") is False:
        status = "rejected"
        reason = "selected_stripes_not_immediate_neighbors"
    elif not quality["success"]:
        status = "unverified"
        reason = "candidate_not_formally_valid"
    elif neighbor_consistency.get("passed") is True:
        status = "verified"
        reason = "immediate_neighbors_verified"
    else:
        status = "unverified"
        reason = (
            neighbor_consistency.get("reason")
            or "neighbor_consistency_not_verifiable"
        )

    return {
        "status": status,
        "reason": reason,
        "neighbor_check": {
            "checked": bool(neighbor_consistency.get("checked")),
            "passed": neighbor_consistency.get("passed"),
            "reason": neighbor_consistency.get("reason"),
            "local_pitch_px": neighbor_consistency.get("local_pitch_px"),
            "selected_span_px": neighbor_consistency.get(
                "selected_span_px"
            ),
            "span_pitch_ratio": neighbor_consistency.get(
                "span_pitch_ratio"
            ),
            "max_span_pitch_ratio": neighbor_consistency.get(
                "max_span_pitch_ratio"
            ),
        },
        "arbitration_fields": {},
    }


def _track_triplet_ids(selection: InteractiveStripeSelection) -> dict:
    """Return the selected track ids without exposing geometry."""

    return {
        "left": (
            None
            if selection.left_track is None
            else selection.left_track.track_id
        ),
        "clicked": (
            None
            if selection.clicked_track is None
            else selection.clicked_track.track_id
        ),
        "right": (
            None
            if selection.right_track is None
            else selection.right_track.track_id
        ),
    }


def _hypothesis_track_summary(track: StripeTrack) -> dict:
    """Return compact evidence for one alternate-hypothesis role."""

    return {
        "track_id": track.track_id,
        "center_x_roi": round(track.center_x_roi, 3),
        "median_width_px": round(track.median_width_px, 3),
        "valid_row_ratio": round(track.valid_row_ratio, 6),
        "crossing_valid_count": track.crossing_valid_count,
        "rejection_reasons": list(track.rejection_reasons),
    }


def _vertical_track_distribution(
    track: StripeTrack,
    roi_height: int,
    interactive_config: InteractiveConfig,
) -> dict:
    """Describe whether a weak track is vertically distributed evidence."""

    rows = sorted({run.y_roi for run in track.valid_runs})
    if not rows:
        return {
            "passed": False,
            "covered_band_count": 0,
            "vertical_span_ratio": 0.0,
        }
    covered_bands = {
        min(2, int((y_roi / max(roi_height, 1)) * 3))
        for y_roi in rows
    }
    vertical_span_ratio = (
        (rows[-1] - rows[0]) / max(roi_height - 1, 1)
    )
    return {
        "passed": (
            len(covered_bands) >= 2
            and vertical_span_ratio
            >= interactive_config.topology_shadow_min_vertical_span_ratio
        ),
        "covered_band_count": len(covered_bands),
        "vertical_span_ratio": round(vertical_span_ratio, 6),
    }


def _track_is_regular_width(
    track: StripeTrack,
    width_statistics: dict,
) -> bool:
    return (
        width_statistics["regular_min"]
        <= track.median_width_px
        <= width_statistics["regular_max"]
    )


def _outer_track_allowed_for_clicked_hypothesis(
    track: StripeTrack,
    width_statistics: dict,
    processing_config: ProcessingConfig,
    interactive_config: InteractiveConfig,
    roi_height: int,
) -> tuple[bool, dict]:
    """Apply the alternate-only floor without changing global eligibility."""

    vertical_distribution = _vertical_track_distribution(
        track,
        roi_height,
        interactive_config,
    )
    rejection_allowed = track.rejection_reasons in (
        [],
        ["insufficient_row_support"],
    )
    weak_distribution_allowed = (
        track.valid_row_ratio >= processing_config.min_stripe_support_ratio
        or vertical_distribution["passed"]
    )
    allowed = (
        track.valid_row_ratio
        >= processing_config.min_clicked_track_support_ratio
        and rejection_allowed
        and track.crossing_valid_count == 0
        and _track_is_regular_width(track, width_statistics)
        and weak_distribution_allowed
    )
    return allowed, vertical_distribution


def _tracks_are_geometrically_separate(
    left_track: StripeTrack,
    right_track: StripeTrack,
) -> bool:
    center_gap = right_track.center_x_roi - left_track.center_x_roi
    minimum_gap = (
        left_track.median_width_px + right_track.median_width_px
    ) / 2.0
    return center_gap >= minimum_gap


def _evaluate_alternate_clicked_selection(
    selection: InteractiveStripeSelection,
    analysis: AdjacentStripeAnalysis,
    image_gray_original_roi,
    click_x_roi: int,
    click_x_global: int,
    click_y_global: int,
    processing_config: ProcessingConfig,
    interactive_config: InteractiveConfig,
    pitch_reference_map: PitchReferenceMap,
    inverse_rotation_matrix: np.ndarray,
) -> dict:
    """Evaluate one complete alternate with the unchanged formal guards."""

    guarded_selection, neighbor_consistency = (
        _apply_neighbor_consistency_guard(
            selection,
            analysis,
            processing_config,
            interactive_config,
        )
    )
    pitch_guard = evaluate_pitch_guard(
        guarded_selection,
        click_x_global,
        click_y_global,
        pitch_reference_map,
        interactive_config,
    )
    topology = evaluate_grayscale_topology(
        image_gray_original_roi,
        guarded_selection,
        inverse_rotation_matrix,
        interactive_config,
    )
    quality = build_candidate_quality(
        guarded_selection,
        analysis,
        neighbor_consistency,
        pitch_guard,
        (
            topology.report
            if interactive_config.enable_grayscale_topology_rejection
            else {"status": "Not considered"}
        ),
        click_x_roi,
        processing_config,
    )
    adjacency_verification = _build_adjacency_verification(
        guarded_selection,
        neighbor_consistency,
        quality,
    )
    quality["adjacency_verification_status"] = (
        adjacency_verification["status"]
    )
    adjacency_verification["arbitration_fields"] = {
        "adjacency_verification_status": adjacency_verification["status"],
        "combined_pitch_status": quality["combined_pitch_status"],
        "grayscale_topology_status": quality[
            "grayscale_topology_status"
        ],
        "quality_success": quality["success"],
        "minimum_valid_row_ratio": quality["minimum_valid_row_ratio"],
        "minimum_retention_ratio": quality["minimum_retention_ratio"],
    }
    pitch_ratios = pitch_guard.get("interval_pitch_ratios", [])
    pitch_tolerance_passed = (
        len(pitch_ratios) == 2
        and all(
            abs(ratio - 1.0)
            <= interactive_config.neighbor_recovery_max_pitch_error_ratio
            for ratio in pitch_ratios
        )
    )
    strong_topology_conflict = bool(
        topology.report.get("strong_same_basin_conflict")
        or topology.report.get("strong_merged_basin_conflict")
    )
    verified_for_adoption = (
        quality["success"]
        and adjacency_verification["status"] == "verified"
        and pitch_tolerance_passed
        and not strong_topology_conflict
    )
    if not quality["success"]:
        rejection_reason = "candidate_not_formally_valid"
    elif adjacency_verification["status"] != "verified":
        rejection_reason = adjacency_verification["reason"]
    elif not pitch_tolerance_passed:
        rejection_reason = "pitch_not_within_recovery_tolerance"
    elif strong_topology_conflict:
        rejection_reason = "strong_grayscale_topology_conflict"
    else:
        rejection_reason = None
    return {
        "selection": guarded_selection,
        "neighbor_consistency": neighbor_consistency,
        "pitch_guard": pitch_guard,
        "topology": topology,
        "quality": quality,
        "adjacency_verification": adjacency_verification,
        "pitch_tolerance_passed": pitch_tolerance_passed,
        "verified_for_adoption": verified_for_adoption,
        "rejection_reason": rejection_reason,
    }


def _clicked_hypothesis_report(
    candidate: DetectionCandidate,
    width_statistics: dict | None,
) -> dict:
    selection = candidate.selection
    clicked = selection.clicked_track
    return {
        "attempted": False,
        "decision": "trigger_not_met",
        "trigger_reason": None,
        "typical_width": (
            None
            if width_statistics is None
            else round(width_statistics["typical_width"], 6)
        ),
        "regular_min": (
            None
            if width_statistics is None
            else round(width_statistics["regular_min"], 6)
        ),
        "regular_max": (
            None
            if width_statistics is None
            else round(width_statistics["regular_max"], 6)
        ),
        "regular_width_stable_track_count": (
            0
            if width_statistics is None
            else width_statistics["stable_track_count"]
        ),
        "max_clicked_center_distance_px": None,
        "original_track_ids": _track_triplet_ids(selection),
        "original_clicked_distance_px": (
            None
            if clicked is None
            else round(
                abs(clicked.center_x_roi - candidate.analysis.x_ref_roi),
                6,
            )
        ),
        "original_clicked_width_px": (
            None if clicked is None else round(clicked.median_width_px, 6)
        ),
        "hypotheses": [],
        "verified_hypothesis_count": 0,
        "adopted_track_ids": None,
        "hard_failure_reason": None,
    }


def _attach_clicked_hypothesis_report(
    candidate: DetectionCandidate,
    report: dict,
) -> DetectionCandidate:
    adjacency_verification = {
        **candidate.adjacency_verification,
        "clicked_hypothesis_arbitration": report,
    }
    return replace(
        candidate,
        adjacency_verification=adjacency_verification,
    )


def _resolve_alternate_clicked_hypothesis(
    candidate: DetectionCandidate,
    image_gray_original_roi,
    click_x_roi: int,
    click_x_global: int,
    click_y_global: int,
    processing_config: ProcessingConfig,
    interactive_config: InteractiveConfig,
    pitch_reference_map: PitchReferenceMap,
) -> DetectionCandidate:
    """Recover only a unique, fully verified alternate clicked triplet."""

    original_clicked = candidate.selection.clicked_track
    stable_tracks_without_clicked = [
        track
        for track in candidate.analysis.tracks
        if track is not original_clicked
        and track.valid_row_ratio
        >= processing_config.min_stripe_support_ratio
        and track.rejection_reasons in ([], ["center_on_reference"])
    ]
    width_statistics = _regular_width_statistics(
        stable_tracks_without_clicked,
        processing_config,
    )
    report = _clicked_hypothesis_report(candidate, width_statistics)
    report["max_clicked_center_distance_px"] = (
        interactive_config.clicked_hypothesis_max_center_distance_px
    )

    def trigger_not_met(reason: str) -> DetectionCandidate:
        report["trigger_reason"] = reason
        return _attach_clicked_hypothesis_report(candidate, report)

    if candidate.geometry != "original":
        return trigger_not_met("geometry_not_original")
    if candidate.threshold_method != "otsu":
        return trigger_not_met("threshold_method_not_otsu")
    if candidate.adjacency_verification["status"] != "rejected":
        return trigger_not_met("original_adjacency_not_rejected")
    if (
        candidate.adjacency_verification["reason"]
        != "selected_stripes_not_immediate_neighbors"
    ):
        return trigger_not_met("original_rejection_reason_not_supported")
    if (
        not candidate.selection.success
        or candidate.selection.click_classification != "black_stripe"
        or original_clicked is None
        or candidate.selection.left_track is None
        or candidate.selection.right_track is None
    ):
        return trigger_not_met("original_selection_not_complete_black_triplet")
    if (
        candidate.grayscale_topology.get("strong_same_basin_conflict")
        or candidate.grayscale_topology.get("strong_merged_basin_conflict")
    ):
        return trigger_not_met("original_has_strong_topology_conflict")
    if (
        width_statistics is None
        or width_statistics["stable_track_count"]
        < interactive_config.neighbor_min_pitch_track_count
    ):
        return trigger_not_met("insufficient_regular_width_reference_tracks")

    click_distance = abs(
        original_clicked.center_x_roi - candidate.analysis.x_ref_roi
    )
    if click_distance <= processing_config.center_cluster_tolerance_px:
        return trigger_not_met("original_clicked_not_materially_offset")
    if original_clicked.median_width_px <= width_statistics["regular_max"]:
        return trigger_not_met("original_clicked_not_abnormally_wide")

    report["attempted"] = True
    report["trigger_reason"] = "wide_offset_clicked_track_rejected"
    roi_height = candidate.analysis.roi_shape[0]
    weak_clicked_tracks = []
    for track in candidate.analysis.tracks:
        weak_distance = abs(
            track.center_x_roi - candidate.analysis.x_ref_roi
        )
        vertical_distribution = _vertical_track_distribution(
            track,
            roi_height,
            interactive_config,
        )
        width_merge_limit = (
            track.median_width_px
            * (1.0 + processing_config.max_width_deviation_ratio)
            + processing_config.min_width_tolerance_px
        )
        if (
            track.rejection_reasons == ["insufficient_row_support"]
            and track.valid_row_ratio
            >= processing_config.min_clicked_track_support_ratio
            and weak_distance
            <= interactive_config.clicked_hypothesis_max_center_distance_px
            and click_distance - weak_distance
            >= processing_config.center_cluster_tolerance_px
            and track.crossing_valid_count > 0
            and weak_distance <= track.median_width_px / 2.0
            and _track_is_regular_width(track, width_statistics)
            and original_clicked.median_width_px > width_merge_limit
            and vertical_distribution["passed"]
        ):
            weak_clicked_tracks.append((track, vertical_distribution))

    evaluated_hypotheses = []
    for weak_clicked, clicked_vertical_distribution in weak_clicked_tracks:
        left_candidates = []
        right_candidates = []
        vertical_by_track_id = {
            weak_clicked.track_id: clicked_vertical_distribution,
        }
        for track in candidate.analysis.tracks:
            allowed, vertical_distribution = (
                _outer_track_allowed_for_clicked_hypothesis(
                    track,
                    width_statistics,
                    processing_config,
                    interactive_config,
                    roi_height,
                )
            )
            if not allowed:
                continue
            vertical_by_track_id[track.track_id] = vertical_distribution
            if track.center_x_roi < weak_clicked.center_x_roi:
                left_candidates.append(track)
            elif track.center_x_roi > weak_clicked.center_x_roi:
                right_candidates.append(track)

        left_track = max(
            left_candidates,
            key=lambda track: track.center_x_roi,
            default=None,
        )
        right_track = min(
            right_candidates,
            key=lambda track: track.center_x_roi,
            default=None,
        )
        hypothesis_report = {
            "track_ids": {
                "left": (
                    None if left_track is None else left_track.track_id
                ),
                "clicked": weak_clicked.track_id,
                "right": (
                    None if right_track is None else right_track.track_id
                ),
            },
            "tracks": {
                "left": (
                    None
                    if left_track is None
                    else _hypothesis_track_summary(left_track)
                ),
                "clicked": _hypothesis_track_summary(weak_clicked),
                "right": (
                    None
                    if right_track is None
                    else _hypothesis_track_summary(right_track)
                ),
            },
            "center_improvement_px": round(
                click_distance
                - abs(
                    weak_clicked.center_x_roi
                    - candidate.analysis.x_ref_roi
                ),
                6,
            ),
            "width_merge_check": {
                "passed": True,
                "original_clicked_width_px": round(
                    original_clicked.median_width_px,
                    6,
                ),
                "alternate_clicked_width_px": round(
                    weak_clicked.median_width_px,
                    6,
                ),
            },
            "vertical_distribution": {},
            "neighbor_consistency": None,
            "pitch_status": None,
            "pitch_interval_ratios": [],
            "pitch_tolerance_passed": False,
            "topology_status": None,
            "strong_topology_conflict": None,
            "quality_success": False,
            "quality_hard_invalid_reasons": [],
            "adjacency_status": None,
            "rejection_reason": None,
        }
        if left_track is None or right_track is None:
            hypothesis_report["rejection_reason"] = (
                "missing_nearest_allowed_outer_track"
            )
            report["hypotheses"].append(hypothesis_report)
            continue
        hypothesis_report["vertical_distribution"] = {
            "left": vertical_by_track_id[left_track.track_id],
            "clicked": clicked_vertical_distribution,
            "right": vertical_by_track_id[right_track.track_id],
        }
        if (
            max(
                left_track.valid_row_ratio,
                right_track.valid_row_ratio,
            )
            < interactive_config.outer_neighbor_recovery_min_support_ratio
        ):
            hypothesis_report["rejection_reason"] = (
                "no_outer_reaches_existing_recovery_support"
            )
            report["hypotheses"].append(hypothesis_report)
            continue
        if not (
            _tracks_are_geometrically_separate(
                left_track,
                weak_clicked,
            )
            and _tracks_are_geometrically_separate(
                weak_clicked,
                right_track,
            )
        ):
            hypothesis_report["rejection_reason"] = (
                "alternate_tracks_not_geometrically_separate"
            )
            report["hypotheses"].append(hypothesis_report)
            continue

        alternate_selection = InteractiveStripeSelection(
            click_classification="black_stripe",
            clicked_track=weak_clicked,
            left_track=left_track,
            right_track=right_track,
            success=True,
            failure_reasons=(),
            warning_flags=("alternate_clicked_hypothesis_adopted",),
        )
        evaluation = _evaluate_alternate_clicked_selection(
            alternate_selection,
            candidate.analysis,
            image_gray_original_roi,
            click_x_roi,
            click_x_global,
            click_y_global,
            processing_config,
            interactive_config,
            pitch_reference_map,
            candidate.inverse_rotation_matrix,
        )
        topology_report = evaluation["topology"].report
        hypothesis_report.update(
            {
                "neighbor_consistency": evaluation[
                    "neighbor_consistency"
                ],
                "pitch_status": evaluation["pitch_guard"]["status"],
                "pitch_interval_ratios": evaluation["pitch_guard"].get(
                    "interval_pitch_ratios",
                    [],
                ),
                "pitch_tolerance_passed": evaluation[
                    "pitch_tolerance_passed"
                ],
                "topology_status": topology_report["status"],
                "strong_topology_conflict": bool(
                    topology_report.get("strong_same_basin_conflict")
                    or topology_report.get(
                        "strong_merged_basin_conflict"
                    )
                ),
                "quality_success": evaluation["quality"]["success"],
                "quality_hard_invalid_reasons": evaluation["quality"][
                    "hard_invalid_reasons"
                ],
                "adjacency_status": evaluation[
                    "adjacency_verification"
                ]["status"],
                "rejection_reason": evaluation["rejection_reason"],
            }
        )
        report["hypotheses"].append(hypothesis_report)
        evaluated_hypotheses.append(
            (
                (
                    left_track.track_id,
                    weak_clicked.track_id,
                    right_track.track_id,
                ),
                evaluation,
            )
        )

    verified_by_triplet = {}
    for triplet, evaluation in evaluated_hypotheses:
        if evaluation["verified_for_adoption"]:
            verified_by_triplet.setdefault(triplet, evaluation)
    report["verified_hypothesis_count"] = len(verified_by_triplet)

    if len(verified_by_triplet) > 1:
        report["decision"] = "ambiguous_verified_clicked_hypotheses"
        report["hard_failure_reason"] = (
            "ambiguous_verified_clicked_hypotheses"
        )
        return _attach_clicked_hypothesis_report(candidate, report)
    if not verified_by_triplet:
        report["decision"] = (
            "no_eligible_alternate"
            if not evaluated_hypotheses
            else "no_verified_alternate"
        )
        return _attach_clicked_hypothesis_report(candidate, report)

    triplet, evaluation = next(iter(verified_by_triplet.items()))
    report["decision"] = "unique_verified_adopted"
    report["adopted_track_ids"] = {
        "left": triplet[0],
        "clicked": triplet[1],
        "right": triplet[2],
    }
    adjacency_verification = {
        **evaluation["adjacency_verification"],
        "clicked_hypothesis_arbitration": report,
    }
    return replace(
        candidate,
        selection=evaluation["selection"],
        neighbor_consistency=evaluation["neighbor_consistency"],
        adjacency_verification=adjacency_verification,
        pitch_guard=evaluation["pitch_guard"],
        neighbor_recovery={
            "applied": False,
            "recoveries": [],
            "reason": "alternate_clicked_hypothesis_selected",
        },
        grayscale_topology=evaluation["topology"].report,
        grayscale_topology_debug=evaluation["topology"].debug_image,
        quality=evaluation["quality"],
    )


def _recover_weak_neighbors(
    selection: InteractiveStripeSelection,
    analysis: AdjacentStripeAnalysis,
    image_gray_original_roi,
    inverse_rotation_matrix: np.ndarray,
    pitch_guard: dict,
    interactive_config: InteractiveConfig,
) -> tuple[InteractiveStripeSelection, dict]:
    """Recover a pitch-matching neighbor rejected only for low support."""

    report = {
        "applied": False,
        "recoveries": [],
        "reason": "no_qualifying_weak_neighbor",
    }
    if (
        not selection.success
        or selection.click_classification != "black_stripe"
        or selection.clicked_track is None
    ):
        report["reason"] = "not_applicable_to_selection"
        return selection, report

    baseline_pitch = pitch_guard.get("baseline_pitch_px")
    if baseline_pitch is None or baseline_pitch <= 0.0:
        report["reason"] = "no_reliable_pitch_baseline"
        return selection, report

    recovered = selection
    for side, interval_label in (
        ("left", "left_to_clicked"),
        ("right", "clicked_to_right"),
    ):
        current_neighbor = getattr(recovered, f"{side}_track")
        clicked_track = recovered.clicked_track
        if current_neighbor is None or clicked_track is None:
            continue

        x_min, x_max = sorted(
            (
                current_neighbor.center_x_roi,
                clicked_track.center_x_roi,
            )
        )
        current_pitch_error = abs(
            abs(
                current_neighbor.center_x_roi
                - clicked_track.center_x_roi
            )
            / baseline_pitch
            - 1.0
        )
        qualifying = []
        for track in analysis.tracks:
            if not x_min < track.center_x_roi < x_max:
                continue
            if track.rejection_reasons != ["insufficient_row_support"]:
                continue
            if (
                track.valid_row_ratio
                < interactive_config.outer_neighbor_recovery_min_support_ratio
            ):
                continue

            interval_pitch_ratio = (
                abs(track.center_x_roi - clicked_track.center_x_roi)
                / baseline_pitch
            )
            pitch_error = abs(interval_pitch_ratio - 1.0)
            if (
                pitch_error
                > interactive_config.neighbor_recovery_max_pitch_error_ratio
                or pitch_error >= current_pitch_error
            ):
                continue

            tentative = replace(
                recovered,
                **{f"{side}_track": track},
            )
            topology = evaluate_grayscale_topology(
                image_gray_original_roi,
                tentative,
                inverse_rotation_matrix,
                interactive_config,
            ).report
            interval = next(
                (
                    item
                    for item in topology["evaluated_intervals"]
                    if item["label"] == interval_label
                ),
                None,
            )
            if interval is None:
                continue
            vertical_coverage = interval["vertical_band_coverage"]
            separator_support = interval["separator_support_ratio"]
            if (
                interval["strong_same_basin_conflict"]
                or interval["informative_row_count"]
                < interactive_config.topology_shadow_min_informative_rows
                or separator_support is None
                or separator_support
                < interactive_config.topology_shadow_min_separator_support_ratio
                or vertical_coverage["covered_band_count"] < 2
            ):
                continue
            qualifying.append(
                (
                    (
                        pitch_error,
                        -track.valid_row_ratio,
                        abs(
                            track.center_x_roi
                            - clicked_track.center_x_roi
                        ),
                    ),
                    track,
                    interval_pitch_ratio,
                    interval,
                )
            )

        if not qualifying:
            continue
        _key, track, interval_pitch_ratio, interval = min(
            qualifying,
            key=lambda item: item[0],
        )
        recovered = replace(
            recovered,
            **{f"{side}_track": track},
        )
        report["recoveries"].append(
            {
                "side": side,
                "original_center_x_roi": round(
                    current_neighbor.center_x_roi,
                    3,
                ),
                "recovered_center_x_roi": round(
                    track.center_x_roi,
                    3,
                ),
                "valid_row_ratio": round(
                    track.valid_row_ratio,
                    6,
                ),
                "interval_pitch_ratio": round(
                    interval_pitch_ratio,
                    6,
                ),
                "informative_row_count": interval[
                    "informative_row_count"
                ],
                "separator_support_ratio": separator_support,
                "vertical_band_count": interval[
                    "vertical_band_coverage"
                ]["covered_band_count"],
            }
        )

    if not report["recoveries"]:
        return selection, report
    warning_flags = recovered.warning_flags
    if "weak_neighbor_recovered" not in warning_flags:
        warning_flags += ("weak_neighbor_recovered",)
    recovered = replace(recovered, warning_flags=warning_flags)
    report["applied"] = True
    report["reason"] = "pitch_and_grayscale_supported"
    return recovered, report


def _rotation_matrix_for_detection(
    click_x_roi: int,
    click_y_roi: int,
    angle_deg: float,
) -> np.ndarray:
    """Create the original-ROI to corrected-ROI affine transform."""

    return cv2.getRotationMatrix2D(
        (float(click_x_roi), float(click_y_roi)),
        angle_deg,
        1.0,
    )


def _rotate_grayscale_for_detection(
    image_gray_roi,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """Rotate grayscale data before running unchanged preprocessing."""

    height, width = image_gray_roi.shape[:2]
    return cv2.warpAffine(
        image_gray_roi,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _identity_affine_matrix() -> np.ndarray:
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )


def _candidate_threshold_methods(
    image_gray_roi,
    config: InteractiveConfig,
) -> tuple[str, ...]:
    methods = ["otsu"]
    block_size, _warnings = _effective_adaptive_block_size(
        image_gray_roi.shape,
        config,
    )
    if config.adaptive_threshold_enabled and block_size is not None:
        methods.append("adaptive")
    return tuple(methods)


def _build_detection_candidate(
    image_gray_detection,
    image_gray_original_roi,
    geometry: str,
    threshold_method: str,
    click_x_roi: int,
    click_y_roi: int,
    click_x_global: int,
    click_y_global: int,
    bounds: RoiBoundsGlobal,
    processing_config: ProcessingConfig,
    interactive_config: InteractiveConfig,
    pitch_reference_map: PitchReferenceMap,
    rotation_matrix: np.ndarray,
    inverse_rotation_matrix: np.ndarray,
) -> DetectionCandidate:
    stages = _preprocess_roi(
        image_gray_detection,
        processing_config,
        interactive_config,
        threshold_method,
    )
    analysis = analyze_adjacent_stripes(
        stages.black_mask,
        click_x_roi,
        click_x_global,
        bounds.x0_global,
        processing_config,
    )
    raw_selection = select_interactive_tracks(
        stages.black_mask,
        click_x_roi,
        click_y_roi,
        analysis,
        processing_config,
        image_gray_roi=stages.image_gray,
    )
    initial_selection, _initial_neighbor_consistency = (
        _apply_neighbor_consistency_guard(
            raw_selection,
            analysis,
            processing_config,
            interactive_config,
        )
    )
    initial_pitch_guard = evaluate_pitch_guard(
        initial_selection,
        click_x_global,
        click_y_global,
        pitch_reference_map,
        interactive_config,
    )
    recovered_selection, neighbor_recovery = _recover_weak_neighbors(
        raw_selection,
        analysis,
        image_gray_original_roi,
        inverse_rotation_matrix,
        initial_pitch_guard,
        interactive_config,
    )
    selection, neighbor_consistency = _apply_neighbor_consistency_guard(
        recovered_selection,
        analysis,
        processing_config,
        interactive_config,
    )
    pitch_guard = evaluate_pitch_guard(
        selection,
        click_x_global,
        click_y_global,
        pitch_reference_map,
        interactive_config,
    )
    topology = evaluate_grayscale_topology(
        image_gray_original_roi,
        selection,
        inverse_rotation_matrix,
        interactive_config,
    )
    quality = build_candidate_quality(
        selection,
        analysis,
        neighbor_consistency,
        pitch_guard,
        (
            topology.report
            if interactive_config.enable_grayscale_topology_rejection
            else {"status": "Not considered"}
        ),
        click_x_roi,
        processing_config,
    )
    adjacency_verification = _build_adjacency_verification(
        selection,
        neighbor_consistency,
        quality,
    )
    quality["adjacency_verification_status"] = (
        adjacency_verification["status"]
    )
    adjacency_verification["arbitration_fields"] = {
        "adjacency_verification_status": adjacency_verification["status"],
        "combined_pitch_status": quality["combined_pitch_status"],
        "grayscale_topology_status": quality[
            "grayscale_topology_status"
        ],
        "quality_success": quality["success"],
        "minimum_valid_row_ratio": quality["minimum_valid_row_ratio"],
        "minimum_retention_ratio": quality["minimum_retention_ratio"],
    }
    candidate = DetectionCandidate(
        geometry=geometry,
        threshold_method=threshold_method,
        stages=stages,
        analysis=analysis,
        selection=selection,
        neighbor_consistency=neighbor_consistency,
        adjacency_verification=adjacency_verification,
        pitch_guard=pitch_guard,
        neighbor_recovery=neighbor_recovery,
        grayscale_topology=topology.report,
        grayscale_topology_debug=topology.debug_image,
        quality=quality,
        rotation_matrix=rotation_matrix,
        inverse_rotation_matrix=inverse_rotation_matrix,
    )
    return _resolve_alternate_clicked_hypothesis(
        candidate,
        image_gray_original_roi,
        click_x_roi,
        click_x_global,
        click_y_global,
        processing_config,
        interactive_config,
        pitch_reference_map,
    )


def _build_geometry_candidates(
    image_gray_detection,
    image_gray_original_roi,
    geometry: str,
    click_x_roi: int,
    click_y_roi: int,
    click_x_global: int,
    click_y_global: int,
    bounds: RoiBoundsGlobal,
    processing_config: ProcessingConfig,
    interactive_config: InteractiveConfig,
    pitch_reference_map: PitchReferenceMap,
    rotation_matrix: np.ndarray,
    inverse_rotation_matrix: np.ndarray,
) -> list[DetectionCandidate]:
    return [
        _build_detection_candidate(
            image_gray_detection,
            image_gray_original_roi,
            geometry,
            threshold_method,
            click_x_roi,
            click_y_roi,
            click_x_global,
            click_y_global,
            bounds,
            processing_config,
            interactive_config,
            pitch_reference_map,
            rotation_matrix,
            inverse_rotation_matrix,
        )
        for threshold_method in _candidate_threshold_methods(
            image_gray_detection,
            interactive_config,
        )
    ]


def _select_threshold_candidate(
    candidates: list[DetectionCandidate],
    include_topology: bool = True,
) -> tuple[DetectionCandidate, str]:
    if not candidates:
        raise ValueError("at least one threshold candidate is required")
    winner = max(
        candidates,
        key=lambda candidate: candidate_quality_key(
            candidate.quality,
            candidate.threshold_method,
            include_topology,
        ),
    )
    if len(candidates) == 1:
        return winner, "only_available_threshold_method"
    loser = next(candidate for candidate in candidates if candidate is not winner)
    reason = quality_difference_reason(
        winner.quality,
        loser.quality,
        include_topology,
    )
    if reason == "quality_equal":
        reason = "quality_equal_otsu_tie_break"
    return winner, reason


def _select_geometry_candidate(
    original: DetectionCandidate,
    rotated: DetectionCandidate | None,
    include_topology: bool = True,
) -> tuple[DetectionCandidate, str]:
    if rotated is None:
        return original, "rotation_not_eligible"
    original_key = candidate_quality_key(
        original.quality,
        include_topology=include_topology,
    )
    rotated_key = candidate_quality_key(
        rotated.quality,
        include_topology=include_topology,
    )
    if rotated_key > original_key:
        return (
            rotated,
            quality_difference_reason(
                rotated.quality,
                original.quality,
                include_topology,
            ),
        )
    if rotated_key == original_key:
        return original, "quality_equal_original_tie_break"
    return (
        original,
        quality_difference_reason(
            original.quality,
            rotated.quality,
            include_topology,
        ),
    )


def _candidate_identity(candidate: DetectionCandidate | None) -> dict | None:
    if candidate is None:
        return None
    return {
        "geometry": candidate.geometry,
        "threshold_method": candidate.threshold_method,
    }


def _strong_topology_conflict_reason(
    candidate: DetectionCandidate,
) -> str | None:
    topology = candidate.grayscale_topology
    if topology["strong_same_basin_conflict"]:
        return "same_dark_basin_split"
    if topology.get("strong_merged_basin_conflict", False):
        return "merged_dark_basins"
    return None


def _shadow_threshold_winner(
    candidates: list[DetectionCandidate],
) -> tuple[DetectionCandidate | None, str, list[str]]:
    """Select the best candidate after removing strong topology conflicts."""

    rejected = [
        candidate.threshold_method
        for candidate in candidates
        if _strong_topology_conflict_reason(candidate) is not None
    ]
    remaining = [
        candidate
        for candidate in candidates
        if _strong_topology_conflict_reason(candidate) is None
    ]
    if not remaining:
        return None, "all_candidates_would_be_rejected", rejected
    winner, reason = _select_threshold_candidate(remaining)
    return winner, reason, rejected


def _safe_topology_replacement(
    candidate: DetectionCandidate | None,
) -> bool:
    """Return whether a rejected winner has a safe formal replacement."""

    if candidate is None or not candidate.quality["success"]:
        return False
    if _strong_topology_conflict_reason(candidate) is not None:
        return False
    if candidate.quality["combined_pitch_status"] == "Suspicious":
        return False
    return (
        candidate.grayscale_topology["status"] == "Consistent"
        or candidate.neighbor_recovery["applied"]
    )


def _selection_track_by_label(
    selection: InteractiveStripeSelection,
    label: str,
) -> StripeTrack | None:
    if label == "left":
        return selection.left_track
    if label == "clicked":
        return selection.clicked_track
    if label == "right":
        return selection.right_track
    raise ValueError(f"unknown selection track label: {label}")


def _replacement_track_is_contained(
    merged_track: StripeTrack,
    replacement_track: StripeTrack,
    roi_height: int,
    interactive_config: InteractiveConfig,
) -> bool:
    merged_runs = {
        run.y_roi: run for run in merged_track.valid_runs
    }
    replacement_runs = {
        run.y_roi: run for run in replacement_track.valid_runs
    }
    common_rows = sorted(set(merged_runs) & set(replacement_runs))
    if (
        len(common_rows)
        < interactive_config.topology_shadow_min_informative_rows
    ):
        return False
    contained_rows = [
        y_roi
        for y_roi in common_rows
        if (
            merged_runs[y_roi].x0_roi
            <= replacement_runs[y_roi].center_x_roi
            < merged_runs[y_roi].x1_roi
        )
    ]
    if (
        len(contained_rows) / len(common_rows)
        < interactive_config.topology_shadow_strong_merged_basin_support_ratio
    ):
        return False
    bands = {
        min(2, int((y_roi / max(roi_height, 1)) * 3))
        for y_roi in contained_rows
    }
    span_ratio = (
        (max(contained_rows) - min(contained_rows))
        / max(roi_height - 1, 1)
    )
    return (
        len(bands) == 3
        and span_ratio
        >= interactive_config.topology_shadow_min_vertical_span_ratio
    )


def _safe_merged_basin_replacement(
    current: DetectionCandidate,
    candidate: DetectionCandidate | None,
    processing_config: ProcessingConfig,
    interactive_config: InteractiveConfig,
) -> bool:
    """Require a tightly matched, one-pitch replacement for a merged run."""

    if candidate is None or not candidate.quality["success"]:
        return False
    if candidate.geometry != current.geometry:
        return False
    if (
        candidate.selection.click_classification
        != current.selection.click_classification
    ):
        return False
    if current.selection.click_classification != "black_stripe":
        return False
    if _strong_topology_conflict_reason(candidate) is not None:
        return False
    if candidate.quality["combined_pitch_status"] != "Normal":
        return False
    if candidate.grayscale_topology["status"] != "Consistent":
        return False

    candidate_ratios = candidate.pitch_guard.get(
        "interval_pitch_ratios",
        [],
    )
    current_ratios = current.pitch_guard.get(
        "interval_pitch_ratios",
        [],
    )
    if not candidate_ratios or not current_ratios:
        return False
    maximum_pitch_error = (
        interactive_config.neighbor_recovery_max_pitch_error_ratio
    )
    candidate_error = max(abs(ratio - 1.0) for ratio in candidate_ratios)
    current_error = max(abs(ratio - 1.0) for ratio in current_ratios)
    if candidate_error > maximum_pitch_error:
        return False
    if candidate_error >= current_error:
        return False

    merged_labels = {
        track_report["label"]
        for track_report in current.grayscale_topology.get(
            "evaluated_tracks",
            [],
        )
        if track_report.get("strong_merged_basin_conflict")
    }
    if not merged_labels:
        return False
    baseline_pitch = candidate.pitch_guard.get("baseline_pitch_px")
    if baseline_pitch is None or baseline_pitch <= 0.0:
        return False
    alignment_tolerance = min(
        processing_config.center_cluster_tolerance_px,
        baseline_pitch * maximum_pitch_error,
    )
    current_clicked = current.selection.clicked_track
    candidate_clicked = candidate.selection.clicked_track
    if current_clicked is None or candidate_clicked is None:
        return False
    for label in ("left", "clicked", "right"):
        current_track = _selection_track_by_label(
            current.selection,
            label,
        )
        candidate_track = _selection_track_by_label(
            candidate.selection,
            label,
        )
        if current_track is None and candidate_track is None:
            continue
        if current_track is None or candidate_track is None:
            return False
        center_difference = abs(
            candidate_track.center_x_roi - current_track.center_x_roi
        )
        if label in merged_labels:
            if not _replacement_track_is_contained(
                current_track,
                candidate_track,
                current.analysis.roi_shape[0],
                interactive_config,
            ):
                return False
            current_distance = abs(
                current_track.center_x_roi
                - current_clicked.center_x_roi
            )
            candidate_distance = abs(
                candidate_track.center_x_roi
                - candidate_clicked.center_x_roi
            )
            if candidate_distance >= current_distance:
                return False
            if label == "left" and not (
                current_track.center_x_roi
                < candidate_track.center_x_roi
                < candidate_clicked.center_x_roi
            ):
                return False
            if label == "right" and not (
                candidate_clicked.center_x_roi
                < candidate_track.center_x_roi
                < current_track.center_x_roi
            ):
                return False
        elif center_difference > alignment_tolerance:
            return False
    return True


def _build_shadow_arbitration(
    original_candidates: list[DetectionCandidate],
    rotated_candidates: list[DetectionCandidate],
    current_winner: DetectionCandidate,
    enforce_rejection: bool,
    processing_config: ProcessingConfig,
    interactive_config: InteractiveConfig,
) -> tuple[dict, DetectionCandidate | None, DetectionCandidate]:
    """Build the compatible report and return the formal/debug winners."""

    original, original_reason, original_rejected = (
        _shadow_threshold_winner(original_candidates)
    )
    rotated, rotated_reason, rotated_rejected = (
        _shadow_threshold_winner(rotated_candidates)
        if rotated_candidates
        else (None, "rotation_not_evaluated", [])
    )
    geometry_reason = None
    if original is None and rotated is None:
        hypothetical_winner = None
        geometry_reason = "no_candidate_after_shadow_rejection"
    elif original is None:
        hypothetical_winner = rotated
        geometry_reason = "original_candidates_would_be_rejected"
    elif rotated is None:
        hypothetical_winner = original
        geometry_reason = "rotated_candidate_unavailable"
    else:
        hypothetical_winner, geometry_reason = (
            _select_geometry_candidate(original, rotated)
        )

    hypothetical_selection = (
        None
        if hypothetical_winner is None
        else _selection_for_output(hypothetical_winner)
    )
    current_conflict_reason = _strong_topology_conflict_reason(
        current_winner
    )
    current_has_conflict = current_conflict_reason is not None
    formal_winner = current_winner
    debug_winner = current_winner
    formal_failure_reason = None
    final_geometry_reason = geometry_reason
    topology_ordering_applied = False
    if enforce_rejection and current_has_conflict:
        if current_conflict_reason == "merged_dark_basins":
            same_geometry_replacement = (
                original
                if current_winner.geometry == "original"
                else rotated
            )
            safe_replacement = _safe_merged_basin_replacement(
                current_winner,
                same_geometry_replacement,
                processing_config,
                interactive_config,
            )
            replacement_candidate = same_geometry_replacement
        else:
            safe_replacement = _safe_topology_replacement(
                hypothetical_winner
            )
            replacement_candidate = hypothetical_winner
        if safe_replacement:
            formal_winner = replacement_candidate
            debug_winner = replacement_candidate
            if current_conflict_reason == "merged_dark_basins":
                final_geometry_reason = (
                    "merged_conflict_safe_same_geometry_replacement"
                )
        else:
            formal_winner = None
            debug_winner = hypothetical_winner or current_winner
            if current_conflict_reason == "merged_dark_basins":
                final_geometry_reason = (
                    "merged_conflict_no_safe_same_geometry_replacement"
                )
                formal_failure_reason = (
                    "strong_merged_basin_conflict_no_safe_alternative"
                )
            else:
                formal_failure_reason = (
                    "strong_same_basin_conflict_no_safe_alternative"
                )
    elif (
        enforce_rejection
        and hypothetical_winner is not None
        and hypothetical_winner is not current_winner
    ):
        formal_winner = hypothetical_winner
        debug_winner = hypothetical_winner
        topology_ordering_applied = True
        final_geometry_reason = "topology_quality_ordering"

    would_change = (
        formal_winner is None or formal_winner is not current_winner
        if enforce_rejection
        else (
            hypothetical_winner is None
            or hypothetical_winner is not current_winner
        )
    )
    report = {
        "mode": "enforced" if enforce_rejection else "shadow",
        "enforced": enforce_rejection,
        "current_winner": _candidate_identity(current_winner),
        "current_winner_conflict_reason": current_conflict_reason,
        "rejected_candidates_if_enabled": [
            f"original_{method}" for method in original_rejected
        ]
        + [f"rotated_{method}" for method in rotated_rejected],
        "threshold_selection_reasons": {
            "original": original_reason,
            "rotated": rotated_reason,
        },
        "geometry_selection_reason": final_geometry_reason,
        "hypothetical_geometry_selection_reason": geometry_reason,
        "hypothetical_winner": _candidate_identity(
            hypothetical_winner
        ),
        "hypothetical_success": (
            False
            if hypothetical_selection is None
            else hypothetical_selection.success
        ),
        "rejection_applied": (
            enforce_rejection and current_has_conflict
        ),
        "topology_ordering_applied": topology_ordering_applied,
        "final_winner": _candidate_identity(formal_winner),
        "formal_failure_reason": formal_failure_reason,
        "would_change_formal_result": would_change,
    }
    return report, formal_winner, debug_winner


def _clicked_hypothesis_hard_failure(
    candidates: list[DetectionCandidate],
) -> dict | None:
    """Return an ambiguity that no later candidate may bypass."""

    sources = []
    source_arbitration = None
    for candidate in candidates:
        arbitration = candidate.adjacency_verification.get(
            "clicked_hypothesis_arbitration"
        )
        if (
            arbitration is None
            or arbitration.get("decision")
            != "ambiguous_verified_clicked_hypotheses"
        ):
            continue
        sources.append(_candidate_identity(candidate))
        if source_arbitration is None:
            source_arbitration = arbitration
    if not sources:
        return None
    return {
        "reason": "ambiguous_verified_clicked_hypotheses",
        "source_candidates": sources,
        "clicked_hypothesis_arbitration": source_arbitration,
    }


def _apply_adjacency_safety_gate(
    formal_candidate: DetectionCandidate | None,
    debug_candidate: DetectionCandidate,
    candidates: list[DetectionCandidate],
    prior_failure_reason: str | None,
    hard_failure: dict | None = None,
) -> tuple[dict, DetectionCandidate | None, DetectionCandidate]:
    """Allow only a locally verified candidate into the formal result."""

    failure_reason = prior_failure_reason
    gated_candidate = formal_candidate
    gate_applied = False
    if hard_failure is not None:
        gated_candidate = None
        failure_reason = hard_failure["reason"]
        gate_applied = True
    elif (
        gated_candidate is not None
        and gated_candidate.adjacency_verification["status"] != "verified"
    ):
        gated_candidate = None
        failure_reason = "immediate_neighbors_not_verified"
        gate_applied = True
    elif gated_candidate is None and failure_reason is None:
        failure_reason = "immediate_neighbors_not_verified"

    report = {
        "required_status": "verified",
        "formal_winner_before_gate": _candidate_identity(formal_candidate),
        "debug_winner": _candidate_identity(debug_candidate),
        "debug_winner_status": debug_candidate.adjacency_verification[
            "status"
        ],
        "candidate_statuses": {
            (
                f"{candidate.geometry}_{candidate.threshold_method}"
            ): candidate.adjacency_verification["status"]
            for candidate in candidates
        },
        "gate_applied": gate_applied,
        "final_winner": _candidate_identity(gated_candidate),
        "failure_reason": failure_reason,
        "hard_failure_applied": hard_failure is not None,
        "hard_failure": hard_failure,
    }
    return report, gated_candidate, debug_candidate


def _formal_result_failure(
    candidate: DetectionCandidate,
    reason: str,
) -> InteractiveStripeSelection:
    """Return a safe formal failure while retaining the candidate for debug."""

    selection = candidate.selection
    return InteractiveStripeSelection(
        click_classification=selection.click_classification,
        clicked_track=None,
        left_track=None,
        right_track=None,
        success=False,
        failure_reasons=selection.failure_reasons + (reason,),
        warning_flags=selection.warning_flags + (reason,),
    )


def _safe_failure_neighbor_consistency(
    reason: str,
    interactive_config: InteractiveConfig,
) -> dict:
    """Return a geometry-free formal neighbor report."""

    return {
        "checked": False,
        "passed": None,
        "local_pitch_px": None,
        "selected_span_px": None,
        "span_pitch_ratio": None,
        "max_span_pitch_ratio": (
            interactive_config.neighbor_max_span_pitch_ratio
        ),
        "pitch_track_count": 0,
        "reason": reason,
    }


def _selection_for_output(
    candidate: DetectionCandidate,
) -> InteractiveStripeSelection:
    """Convert a hard-invalid raw selection into an explicit failure."""

    if candidate.quality["success"] or not candidate.selection.success:
        return candidate.selection
    reasons = tuple(candidate.quality["hard_invalid_reasons"])
    return InteractiveStripeSelection(
        click_classification=candidate.selection.click_classification,
        clicked_track=candidate.selection.clicked_track,
        left_track=None,
        right_track=None,
        success=False,
        failure_reasons=candidate.selection.failure_reasons + reasons,
        warning_flags=candidate.selection.warning_flags + reasons,
    )


def _candidate_summary(candidate: DetectionCandidate) -> dict:
    selection = candidate.selection
    return {
        "evaluated": True,
        "geometry": candidate.geometry,
        "threshold_method": candidate.threshold_method,
        "raw_success": selection.success,
        "quality_success": candidate.quality["success"],
        "click_classification": selection.click_classification,
        "failure_reasons": list(selection.failure_reasons),
        "warning_flags": list(selection.warning_flags),
        "selected_track_centers_roi": {
            "left": (
                None
                if selection.left_track is None
                else round(selection.left_track.center_x_roi, 3)
            ),
            "clicked": (
                None
                if selection.clicked_track is None
                else round(selection.clicked_track.center_x_roi, 3)
            ),
            "right": (
                None
                if selection.right_track is None
                else round(selection.right_track.center_x_roi, 3)
            ),
        },
        "neighbor_consistency": candidate.neighbor_consistency,
        "adjacency_verification": candidate.adjacency_verification,
        "pitch_guard": candidate.pitch_guard,
        "neighbor_recovery": candidate.neighbor_recovery,
        "grayscale_topology": candidate.grayscale_topology,
        "quality": candidate.quality,
        "threshold_warning_flags": list(
            candidate.stages.threshold_warning_flags
        ),
    }


def _sample_track_centers(image, track: StripeTrack, color) -> None:
    sample_step = max(1, (len(track.valid_runs) + 49) // 50)
    for run in track.valid_runs[::sample_step]:
        image[run.y_roi, int(run.center_x_roi + 0.5)] = color


def _create_interactive_roi_result(
    image_gray_roi,
    click_x_roi: int,
    click_y_roi: int,
    selection: InteractiveStripeSelection,
) -> np.ndarray:
    result = cv2.cvtColor(image_gray_roi, cv2.COLOR_GRAY2BGR)
    height_roi = image_gray_roi.shape[0]
    if selection.clicked_track is not None:
        _sample_track_centers(result, selection.clicked_track, (0, 140, 255))
    for track, point_color in (
        (selection.left_track, (150, 80, 80)),
        (selection.right_track, (40, 150, 150)),
    ):
        if track is not None:
            _sample_track_centers(result, track, point_color)
    for track, color in (
        (selection.left_track, (255, 0, 0)),
        (selection.right_track, (0, 255, 255)),
    ):
        if track is not None:
            x_track = int(track.center_x_roi + 0.5)
            cv2.line(result, (x_track, 0), (x_track, height_roi - 1), color, 1)
    cv2.drawMarker(
        result,
        (click_x_roi, click_y_roi),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        15,
        2,
    )
    status = "SUCCESS" if selection.success else "FAILED"
    cv2.rectangle(result, (0, 0), (result.shape[1] - 1, 72), (0, 0, 0), -1)
    cv2.putText(
        result,
        f"{status} click={selection.click_classification}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0) if selection.success else (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    clicked_id = (
        "none" if selection.clicked_track is None else selection.clicked_track.track_id
    )
    cv2.putText(
        result,
        f"clicked track={clicked_id} (orange dots, excluded)",
        (8, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 170, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        "final centerlines: L blue, R yellow; click: red cross",
        (8, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return result


def _create_interactive_candidates_debug(
    black_mask_roi,
    analysis: AdjacentStripeAnalysis,
    selection: InteractiveStripeSelection,
    click_x_roi: int,
    click_y_roi: int,
) -> np.ndarray:
    debug = cv2.cvtColor(black_mask_roi, cv2.COLOR_GRAY2BGR)
    debug = cv2.convertScaleAbs(debug, alpha=0.25)
    for run in analysis.all_runs:
        color = (0, 150, 0) if run.accepted else (0, 0, 100)
        debug[run.y_roi, int(run.center_x_roi + 0.5)] = color
    for track, color in (
        (selection.clicked_track, (0, 140, 255)),
        (selection.left_track, (255, 0, 0)),
        (selection.right_track, (0, 255, 255)),
    ):
        if track is None:
            continue
        for run in track.valid_runs:
            cv2.line(
                debug,
                (run.x0_roi, run.y_roi),
                (run.x1_roi - 1, run.y_roi),
                color,
                1,
            )
    cv2.drawMarker(
        debug,
        (click_x_roi, click_y_roi),
        (255, 0, 255),
        cv2.MARKER_CROSS,
        13,
        1,
    )
    return debug


def _create_interactive_votes_debug(
    analysis: AdjacentStripeAnalysis,
    selection: InteractiveStripeSelection,
) -> np.ndarray:
    chart_height = 320
    chart_width = analysis.roi_shape[1]
    top = 35
    bottom = 40
    baseline_y = chart_height - bottom
    plot_height = baseline_y - top
    chart = np.zeros((chart_height, chart_width, 3), dtype=np.uint8)
    max_vote = int(analysis.smoothed_votes.max())
    if max_vote > 0:
        points = []
        for x_roi, vote in enumerate(analysis.smoothed_votes):
            y_chart = baseline_y - round(int(vote) / max_vote * plot_height)
            points.append((x_roi, y_chart))
        cv2.polylines(
            chart,
            [np.array(points, dtype=np.int32)],
            False,
            (0, 200, 0),
            1,
        )
    cv2.line(chart, (0, baseline_y), (chart_width - 1, baseline_y), (100, 100, 100), 1)
    selected_ids = {
        track.track_id
        for track in (selection.left_track, selection.right_track)
        if track is not None
    }
    for track in analysis.tracks:
        if track.track_id in selected_ids:
            color = (255, 0, 0) if track is selection.left_track else (0, 255, 255)
        elif (
            selection.clicked_track is not None
            and track.track_id == selection.clicked_track.track_id
        ):
            color = (0, 140, 255)
        elif _stable_for_display(track):
            color = (220, 220, 220)
        else:
            color = (80, 80, 80)
        cv2.circle(chart, (int(track.center_x_roi + 0.5), top + 7), 4, color, -1)
    cv2.line(
        chart,
        (analysis.x_ref_roi, top),
        (analysis.x_ref_roi, baseline_y),
        (255, 0, 255),
        1,
    )
    cv2.putText(
        chart,
        "Interactive votes: click magenta; clicked orange; final L blue/R yellow",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return chart


def _stable_for_display(track: StripeTrack) -> bool:
    return track.rejection_reasons in ([], ["center_on_reference"])


def _detection_point_to_global(
    x_detection: float,
    y_detection: float,
    inverse_rotation_matrix: np.ndarray,
    bounds: RoiBoundsGlobal,
) -> tuple[float, float]:
    point = np.array(
        [[[x_detection, y_detection]]],
        dtype=np.float64,
    )
    x_roi, y_roi = cv2.transform(point, inverse_rotation_matrix)[0, 0]
    return (
        float(bounds.x0_global + x_roi),
        float(bounds.y0_global + y_roi),
    )


def _track_line_endpoints_global(
    track: StripeTrack,
    roi_height: int,
    inverse_rotation_matrix: np.ndarray,
    bounds: RoiBoundsGlobal,
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        _detection_point_to_global(
            track.center_x_roi,
            0.0,
            inverse_rotation_matrix,
            bounds,
        ),
        _detection_point_to_global(
            track.center_x_roi,
            float(roi_height - 1),
            inverse_rotation_matrix,
            bounds,
        ),
    )


def _create_original_overlay(
    image_gray,
    bounds: RoiBoundsGlobal,
    click_x_global: int,
    click_y_global: int,
    selection: InteractiveStripeSelection,
    inverse_rotation_matrix: np.ndarray,
) -> np.ndarray:
    overlay = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(
        overlay,
        (bounds.x0_global, bounds.y0_global),
        (bounds.x1_global - 1, bounds.y1_global - 1),
        (0, 255, 0),
        2,
    )
    for track, color in (
        (selection.left_track, (255, 0, 0)),
        (selection.right_track, (0, 255, 255)),
    ):
        if track is None:
            continue
        point0, point1 = _track_line_endpoints_global(
            track,
            bounds.height_roi,
            inverse_rotation_matrix,
            bounds,
        )
        line_start = tuple(round(value) for value in point0)
        line_end = tuple(round(value) for value in point1)
        visible, line_start, line_end = cv2.clipLine(
            (
                bounds.x0_global,
                bounds.y0_global,
                bounds.width_roi,
                bounds.height_roi,
            ),
            line_start,
            line_end,
        )
        if not visible:
            continue
        cv2.line(
            overlay,
            line_start,
            line_end,
            color,
            2,
        )
    cv2.drawMarker(
        overlay,
        (click_x_global, click_y_global),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        21,
        2,
    )
    cv2.circle(overlay, (click_x_global, click_y_global), 7, (0, 0, 255), 1)
    return overlay


def _track_report(
    track: StripeTrack | None,
    bounds: RoiBoundsGlobal,
    click_x_roi: int,
    click_y_roi: int,
    inverse_rotation_matrix: np.ndarray,
):
    if track is None:
        return None
    point0, point1 = _track_line_endpoints_global(
        track,
        bounds.height_roi,
        inverse_rotation_matrix,
        bounds,
    )
    click_y_global = float(bounds.y0_global + click_y_roi)
    line_dy = point1[1] - point0[1]
    if abs(line_dy) < 1e-12:
        center_x_global = point0[0]
    else:
        interpolation = (click_y_global - point0[1]) / line_dy
        center_x_global = point0[0] + interpolation * (
            point1[0] - point0[0]
        )
    return {
        "track_id": track.track_id,
        "center_x_roi": round(track.center_x_roi, 3),
        "center_x_global": round(center_x_global, 3),
        "center_y_global": round(click_y_global, 3),
        "line_endpoints_global": [
            [round(point0[0], 3), round(point0[1], 3)],
            [round(point1[0], 3), round(point1[1], 3)],
        ],
        "distance_to_click_px": round(
            abs(track.center_x_roi - click_x_roi), 3
        ),
        "median_width_px": round(track.median_width_px, 3),
        "assigned_row_count": track.assigned_row_count,
        "valid_row_count": track.valid_row_count,
        "valid_row_ratio": round(track.valid_row_ratio, 6),
        "retention_ratio": round(track.retention_ratio, 6),
    }


def _roi_warning_flags(
    image_shape: tuple[int, ...],
    click_x_global: int,
    click_y_global: int,
    config: InteractiveConfig,
) -> list[str]:
    height, width = image_shape[:2]
    flags = []
    if click_x_global - config.roi_half_width_px < 0:
        flags.append("roi_clipped_left")
    if click_x_global + config.roi_half_width_px > width:
        flags.append("roi_clipped_right")
    if click_y_global - config.roi_half_height_px < 0:
        flags.append("roi_clipped_top")
    if click_y_global + config.roi_half_height_px > height:
        flags.append("roi_clipped_bottom")
    return flags


def _build_report(
    image_name: str,
    image_shape: tuple[int, ...],
    click_x_global: int,
    click_y_global: int,
    click_x_roi: int,
    click_y_roi: int,
    bounds: RoiBoundsGlobal,
    selection: InteractiveStripeSelection,
    neighbor_consistency: dict,
    baseline_neighbor_consistency: dict,
    rotated_neighbor_consistency: dict | None,
    baseline_analysis: AdjacentStripeAnalysis,
    rotated_analysis: AdjacentStripeAnalysis | None,
    rotation: RotationShadowResult,
    rotation_applied: bool,
    rotation_guard_reasons: list[str],
    rotation_fallback_reason: str | None,
    rotation_matrix: np.ndarray,
    inverse_rotation_matrix: np.ndarray,
    processing_config: ProcessingConfig,
    interactive_config: InteractiveConfig,
    pitch_guard: dict,
    pitch_reference_map: PitchReferenceMap,
    selected_candidate: DetectionCandidate,
    candidate_summaries: dict,
    threshold_selection_reasons: dict,
    geometry_selection_reason: str,
    shadow_arbitration: dict,
    adjacency_arbitration: dict,
) -> dict:
    final_candidate_identity = adjacency_arbitration["final_winner"]
    hard_failure = adjacency_arbitration.get("hard_failure")
    if hard_failure is None:
        formal_adjacency_verification = (
            selected_candidate.adjacency_verification
        )
    else:
        formal_adjacency_verification = {
            "status": "rejected",
            "reason": hard_failure["reason"],
            "neighbor_check": {
                "checked": False,
                "passed": None,
                "reason": hard_failure["reason"],
                "local_pitch_px": None,
                "selected_span_px": None,
                "span_pitch_ratio": None,
                "max_span_pitch_ratio": (
                    interactive_config.neighbor_max_span_pitch_ratio
                ),
            },
            "arbitration_fields": {},
            "clicked_hypothesis_arbitration": hard_failure[
                "clicked_hypothesis_arbitration"
            ],
        }
    left = _track_report(
        selection.left_track,
        bounds,
        click_x_roi,
        click_y_roi,
        inverse_rotation_matrix,
    )
    right = _track_report(
        selection.right_track,
        bounds,
        click_x_roi,
        click_y_roi,
        inverse_rotation_matrix,
    )
    clicked = _track_report(
        selection.clicked_track,
        bounds,
        click_x_roi,
        click_y_roi,
        inverse_rotation_matrix,
    )
    stripe_spacing = None
    if selection.left_track is not None and selection.right_track is not None:
        stripe_spacing = round(
            selection.right_track.center_x_roi
            - selection.left_track.center_x_roi,
            3,
        )
    warning_flags = list(selection.warning_flags)
    for flag in _roi_warning_flags(
        image_shape,
        click_x_global,
        click_y_global,
        interactive_config,
    ):
        if flag not in warning_flags:
            warning_flags.append(flag)
    if rotation_fallback_reason is not None:
        warning_flags.append(rotation_fallback_reason)
    if pitch_guard["status"] == "Suspicious":
        warning_flags.append("whole_image_pitch_suspicious")
    elif pitch_guard["status"] == "Unable to verify":
        warning_flags.append("whole_image_pitch_unverifiable")
    rotated_report = None
    if rotated_analysis is not None:
        rotated_report = analysis_to_dict(rotated_analysis, processing_config)
        rotated_report["coordinate_space"] = (
            "rotated_roi; use interactive_result for mapped global geometry"
        )
    return {
        "algorithm": (
            "interactive_layered_threshold_rotation_row_run_center_voting"
        ),
        "algorithm_revision": ALGORITHM_REVISION,
        "image_name": image_name,
        "click": {
            "x_global": click_x_global,
            "y_global": click_y_global,
            "x_roi": click_x_roi,
            "y_roi": click_y_roi,
            "classification": selection.click_classification,
        },
        "roi": {
            "x0_global": bounds.x0_global,
            "x1_global": bounds.x1_global,
            "y0_global": bounds.y0_global,
            "y1_global": bounds.y1_global,
            "width_px": bounds.width_roi,
            "height_px": bounds.height_roi,
        },
        "rotation_shadow": {
            "applied_to_detection": rotation_applied,
            "detection_space": selected_candidate.geometry,
            "best_angle_deg": round(rotation.best_angle_deg, 3),
            "applied_angle_deg": (
                round(rotation.best_angle_deg, 3) if rotation_applied else 0.0
            ),
            "zero_angle_score": round(rotation.zero_angle_score, 8),
            "best_score": round(rotation.best_score, 8),
            "relative_score_gain": round(
                _relative_rotation_score_gain(rotation), 8
            ),
            "peak_separation": round(rotation.peak_separation, 8),
            "guard_eligible": not rotation_guard_reasons,
            "guard_reasons": rotation_guard_reasons,
            "fallback_reason": rotation_fallback_reason,
            "baseline_neighbor_consistency": baseline_neighbor_consistency,
            "candidate_neighbor_consistency": rotated_neighbor_consistency,
            "matrix_original_to_detection": rotation_matrix.tolist(),
            "matrix_detection_to_original": (
                inverse_rotation_matrix.tolist()
            ),
        },
        "interactive_result": {
            "success": selection.success,
            "failure_reasons": list(selection.failure_reasons),
            "warning_flags": warning_flags,
            "threshold_method": (
                None
                if final_candidate_identity is None
                else final_candidate_identity["threshold_method"]
            ),
            "geometry": (
                None
                if final_candidate_identity is None
                else final_candidate_identity["geometry"]
            ),
            "combined_pitch_status": (
                "Not applicable"
                if final_candidate_identity is None
                else selected_candidate.quality["combined_pitch_status"]
            ),
            "clicked_track_id": (
                None
                if selection.clicked_track is None
                else selection.clicked_track.track_id
            ),
            "clicked": clicked,
            "distance_definition": "horizontal_in_detection_space",
            "neighbor_consistency": neighbor_consistency,
            "adjacency_verification": formal_adjacency_verification,
            "pitch_guard": pitch_guard,
            "stripe_spacing_px": stripe_spacing,
            "left": left,
            "right": right,
        },
        "candidate_arbitration": {
            "candidates": candidate_summaries,
            "threshold_selection_reasons": threshold_selection_reasons,
            "geometry_selection_reason": geometry_selection_reason,
            "final_threshold_method": (
                None
                if final_candidate_identity is None
                else final_candidate_identity["threshold_method"]
            ),
            "final_geometry": (
                None
                if final_candidate_identity is None
                else final_candidate_identity["geometry"]
            ),
        },
        "shadow_arbitration": shadow_arbitration,
        "adjacency_arbitration": adjacency_arbitration,
        "pitch_reference": {
            "tile_count": len(pitch_reference_map.tiles),
            "valid_tile_count": pitch_reference_map.valid_tile_count,
            "tile_width_px": pitch_reference_map.tile_width_px,
            "tile_height_px": pitch_reference_map.tile_height_px,
            "stride_x_px": pitch_reference_map.stride_x_px,
            "stride_y_px": pitch_reference_map.stride_y_px,
        },
        "legacy_analysis": analysis_to_dict(
            baseline_analysis, processing_config
        ),
        "rotated_analysis": rotated_report,
    }


def _save_report(output_path: Path, report: dict) -> None:
    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(report, output_file, indent=2)
    except OSError as error:
        raise OSError(f"Could not save interactive report: {output_path}") from error


def run_interactive_case(
    image_gray,
    click_x_global: int,
    click_y_global: int,
    output_dir: Path,
    processing_config: ProcessingConfig,
    interactive_config: InteractiveConfig,
    image_name: str = "interactive_image",
    pitch_reference_map: PitchReferenceMap | None = None,
) -> InteractivePipelineResult:
    """Run one clicked-point case and save explainable debug outputs."""

    validate_interactive_config(interactive_config)
    if image_gray.ndim != 2:
        raise ValueError("image_gray must be a single-channel image")
    if pitch_reference_map is None:
        pitch_reference_map = build_pitch_reference_map(
            image_gray,
            processing_config,
            interactive_config,
        )
    elif pitch_reference_map.image_shape != image_gray.shape[:2]:
        raise ValueError(
            "pitch_reference_map image shape does not match image_gray"
        )
    bounds = calculate_interactive_roi_bounds(
        image_gray.shape,
        click_x_global,
        click_y_global,
        interactive_config,
    )
    image_gray_roi = crop_roi_global(image_gray, bounds)
    click_x_roi = click_x_global - bounds.x0_global
    click_y_roi = click_y_global - bounds.y0_global
    identity_matrix = _identity_affine_matrix()
    original_candidates = _build_geometry_candidates(
        image_gray_roi,
        image_gray_roi,
        "original",
        click_x_roi,
        click_y_roi,
        click_x_global,
        click_y_global,
        bounds,
        processing_config,
        interactive_config,
        pitch_reference_map,
        identity_matrix,
        identity_matrix,
    )
    original_best, original_threshold_reason = (
        _select_threshold_candidate(
            original_candidates,
            include_topology=False,
        )
    )
    original_by_method = {
        candidate.threshold_method: candidate
        for candidate in original_candidates
    }
    original_otsu = original_by_method["otsu"]
    baseline_stages = original_otsu.stages
    legacy_analysis = original_otsu.analysis
    baseline_neighbor_consistency = (
        original_best.neighbor_consistency
    )
    rotation_shadow = estimate_rotation_shadow(
        baseline_stages.vertical_close,
        interactive_config,
    )
    roi_warning_flags = _roi_warning_flags(
        image_gray.shape,
        click_x_global,
        click_y_global,
        interactive_config,
    )
    rotation_guard_reasons = _rotation_guard_reasons(
        rotation_shadow,
        roi_warning_flags,
        interactive_config,
    )
    rotated_candidates = []
    rotated_best = None
    rotated_threshold_reason = "rotation_not_eligible"
    rotation_fallback_reason = None
    rotated_analysis = None
    rotated_neighbor_consistency = None
    rotation_candidate_stages = baseline_stages

    if not rotation_guard_reasons:
        candidate_matrix = _rotation_matrix_for_detection(
            click_x_roi,
            click_y_roi,
            rotation_shadow.best_angle_deg,
        )
        rotated_gray_roi = _rotate_grayscale_for_detection(
            image_gray_roi,
            candidate_matrix,
        )
        candidate_inverse_matrix = cv2.invertAffineTransform(
            candidate_matrix
        )
        rotated_candidates = _build_geometry_candidates(
            rotated_gray_roi,
            image_gray_roi,
            "rotated",
            click_x_roi,
            click_y_roi,
            click_x_global,
            click_y_global,
            bounds,
            processing_config,
            interactive_config,
            pitch_reference_map,
            candidate_matrix,
            candidate_inverse_matrix,
        )
        rotated_best, rotated_threshold_reason = (
            _select_threshold_candidate(
                rotated_candidates,
                include_topology=False,
            )
        )
        rotation_candidate_stages = rotated_best.stages
        rotated_analysis = rotated_best.analysis
        rotated_neighbor_consistency = rotated_best.neighbor_consistency

    current_candidate, current_geometry_selection_reason = (
        _select_geometry_candidate(
            original_best,
            rotated_best,
            include_topology=False,
        )
    )
    (
        shadow_arbitration,
        topology_formal_candidate,
        selected_candidate,
    ) = _build_shadow_arbitration(
        original_candidates,
        rotated_candidates,
        current_candidate,
        interactive_config.enable_grayscale_topology_rejection,
        processing_config,
        interactive_config,
    )
    all_candidates = original_candidates + rotated_candidates
    clicked_hypothesis_hard_failure = (
        _clicked_hypothesis_hard_failure(all_candidates)
    )
    (
        adjacency_arbitration,
        formal_candidate,
        selected_candidate,
    ) = _apply_adjacency_safety_gate(
        topology_formal_candidate,
        selected_candidate,
        all_candidates,
        shadow_arbitration["formal_failure_reason"],
        hard_failure=clicked_hypothesis_hard_failure,
    )
    if (
        shadow_arbitration["rejection_applied"]
        or shadow_arbitration["topology_ordering_applied"]
    ):
        threshold_selection_reasons = shadow_arbitration[
            "threshold_selection_reasons"
        ]
        geometry_selection_reason = shadow_arbitration[
            "geometry_selection_reason"
        ]
    else:
        threshold_selection_reasons = {
            "original": original_threshold_reason,
            "rotated": rotated_threshold_reason,
        }
        geometry_selection_reason = current_geometry_selection_reason

    debug_candidate = selected_candidate
    result_candidate = formal_candidate or debug_candidate
    rotation_applied = result_candidate.geometry == "rotated"
    if rotated_best is not None and not rotation_applied:
        rotation_fallback_reason = (
            f"rotation_candidate_not_selected_{geometry_selection_reason}"
        )
    active_stages = result_candidate.stages
    active_analysis = result_candidate.analysis
    if formal_candidate is None:
        selection = _formal_result_failure(
            debug_candidate,
            adjacency_arbitration["failure_reason"],
        )
    else:
        selection = _selection_for_output(formal_candidate)
    if adjacency_arbitration["hard_failure_applied"]:
        neighbor_consistency = _safe_failure_neighbor_consistency(
            adjacency_arbitration["failure_reason"],
            interactive_config,
        )
    else:
        neighbor_consistency = result_candidate.neighbor_consistency
    rotation_matrix = result_candidate.rotation_matrix
    inverse_rotation_matrix = result_candidate.inverse_rotation_matrix
    if formal_candidate is None:
        pitch_guard = evaluate_pitch_guard(
            selection,
            click_x_global,
            click_y_global,
            pitch_reference_map,
            interactive_config,
        )
    else:
        pitch_guard = formal_candidate.pitch_guard
    candidate_summaries = {}
    for geometry, candidates in (
        ("original", original_candidates),
        ("rotated", rotated_candidates),
    ):
        by_method = {
            candidate.threshold_method: candidate
            for candidate in candidates
        }
        for threshold_method in ("otsu", "adaptive"):
            key = f"{geometry}_{threshold_method}"
            candidate = by_method.get(threshold_method)
            if candidate is None:
                reason = (
                    "rotation_not_eligible"
                    if geometry == "rotated" and rotation_guard_reasons
                    else "threshold_method_unavailable"
                )
                candidate_summaries[key] = {
                    "evaluated": False,
                    "geometry": geometry,
                    "threshold_method": threshold_method,
                    "reason": reason,
                }
            else:
                candidate_summaries[key] = _candidate_summary(candidate)
    original_overlay = _create_original_overlay(
        image_gray,
        bounds,
        click_x_global,
        click_y_global,
        selection,
        inverse_rotation_matrix,
    )
    interactive_result = _create_interactive_roi_result(
        active_stages.image_gray,
        click_x_roi,
        click_y_roi,
        selection,
    )
    candidates_debug = _create_interactive_candidates_debug(
        debug_candidate.stages.black_mask,
        debug_candidate.analysis,
        debug_candidate.selection,
        click_x_roi,
        click_y_roi,
    )
    votes_debug = _create_interactive_votes_debug(
        debug_candidate.analysis,
        debug_candidate.selection,
    )
    debug_candidate_result = _create_interactive_roi_result(
        debug_candidate.stages.image_gray,
        click_x_roi,
        click_y_roi,
        debug_candidate.selection,
    )
    pitch_reference_debug = create_pitch_reference_debug(
        image_gray,
        pitch_reference_map,
    )
    report = _build_report(
        image_name,
        image_gray.shape,
        click_x_global,
        click_y_global,
        click_x_roi,
        click_y_roi,
        bounds,
        selection,
        neighbor_consistency,
        baseline_neighbor_consistency,
        rotated_neighbor_consistency,
        legacy_analysis,
        rotated_analysis,
        rotation_shadow,
        rotation_applied,
        rotation_guard_reasons,
        rotation_fallback_reason,
        rotation_matrix,
        inverse_rotation_matrix,
        processing_config,
        interactive_config,
        pitch_guard,
        pitch_reference_map,
        result_candidate,
        candidate_summaries,
        threshold_selection_reasons,
        geometry_selection_reason,
        shadow_arbitration,
        adjacency_arbitration,
    )
    debug_images = {
        "original_interactive_result.png": original_overlay,
        "roi_debug.png": original_overlay,
        "roi_crop.png": image_gray_roi,
        "roi_gray.png": active_stages.image_gray,
        "unrotated_roi_gray.png": baseline_stages.image_gray,
        "rotation_candidate_roi_gray.png": (
            rotation_candidate_stages.image_gray
        ),
        "otsu_binary.png": active_stages.otsu_binary,
        "threshold_binary.png": active_stages.threshold_binary,
        "vertical_close.png": active_stages.vertical_close,
        "close_delta.png": active_stages.close_delta,
        "black_mask.png": active_stages.black_mask,
        "unrotated_black_mask.png": baseline_stages.black_mask,
        "rotation_candidate_black_mask.png": (
            rotation_candidate_stages.black_mask
        ),
        "black_run_candidates.png": candidates_debug,
        "stripe_center_votes.png": votes_debug,
        "interactive_result.png": interactive_result,
        "debug_candidate_interactive_result.png": debug_candidate_result,
        "rotation_score.png": rotation_shadow.score_chart,
        "rotation_preview.png": rotation_shadow.preview_image,
        "pitch_reference_map.png": pitch_reference_debug,
    }
    for candidate in original_candidates + rotated_candidates:
        prefix = f"{candidate.geometry}_{candidate.threshold_method}"
        debug_images[f"{prefix}_binary.png"] = (
            candidate.stages.threshold_binary
        )
        debug_images[f"{prefix}_black_mask.png"] = (
            candidate.stages.black_mask
        )
        debug_images[f"{prefix}_grayscale_topology.png"] = (
            candidate.grayscale_topology_debug
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, image in debug_images.items():
        save_debug_image(output_dir / filename, image)
    _save_report(output_dir / "interactive_results.json", report)
    _save_report(
        output_dir / "pitch_reference_map.json",
        pitch_reference_map.to_dict(),
    )
    return InteractivePipelineResult(
        click_x_global=click_x_global,
        click_y_global=click_y_global,
        click_x_roi=click_x_roi,
        click_y_roi=click_y_roi,
        bounds_global=bounds,
        legacy_analysis=legacy_analysis,
        active_analysis=active_analysis,
        selection=selection,
        rotation_shadow=rotation_shadow,
        rotation_applied=rotation_applied,
        report=report,
        debug_images=debug_images,
        output_dir=output_dir,
        pitch_reference_map=pitch_reference_map,
    )
