"""Interactive point-centered pipeline built around the legacy detector."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np

from config import InteractiveConfig, ProcessingConfig
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
from stripe_analysis import (
    AdjacentStripeAnalysis,
    StripeTrack,
    analysis_to_dict,
    analyze_adjacent_stripes,
)


@dataclass(frozen=True)
class RotationShadowResult:
    """Explainable angle estimate that does not affect detection yet."""

    best_angle_deg: float
    zero_angle_score: float
    best_score: float
    peak_separation: float
    angles_deg: np.ndarray
    scores: np.ndarray
    preview_image: np.ndarray
    score_chart: np.ndarray


@dataclass(frozen=True)
class InteractivePipelineResult:
    """Data and debug images produced for one clicked point."""

    click_x_global: int
    click_y_global: int
    click_x_roi: int
    click_y_roi: int
    bounds_global: RoiBoundsGlobal
    legacy_analysis: AdjacentStripeAnalysis
    selection: InteractiveStripeSelection
    rotation_shadow: RotationShadowResult
    report: dict
    debug_images: dict[str, np.ndarray]
    output_dir: Path


def validate_interactive_config(config: InteractiveConfig) -> None:
    """Validate settings that belong only to the interactive workflow."""

    if config.roi_half_width_px <= 0:
        raise ValueError("roi_half_width_px must be greater than 0")
    if config.roi_half_height_px <= 0:
        raise ValueError("roi_half_height_px must be greater than 0")
    if config.rotation_min_angle_deg > config.rotation_max_angle_deg:
        raise ValueError("rotation_min_angle_deg must not exceed maximum")
    if config.rotation_angle_step_deg <= 0.0:
        raise ValueError("rotation_angle_step_deg must be greater than 0")
    if not (
        config.rotation_min_angle_deg <= 0.0
        <= config.rotation_max_angle_deg
    ):
        raise ValueError("rotation shadow angle range must include zero")
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
        "Rotation shadow: yellow=best, red=0 deg (not applied)",
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
    """Estimate a correction angle without changing the detector input."""

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
        if (
            selection.clicked_track is not None
            and track.track_id == selection.clicked_track.track_id
        ):
            color = (0, 140, 255)
        elif track.track_id in selected_ids:
            color = (255, 0, 0) if track is selection.left_track else (0, 255, 255)
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
        "Interactive votes: click magenta; excluded orange; final L blue/R yellow",
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


def _create_original_overlay(
    image_gray,
    bounds: RoiBoundsGlobal,
    click_x_global: int,
    click_y_global: int,
    selection: InteractiveStripeSelection,
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
        x_global = int(bounds.x0_global + track.center_x_roi + 0.5)
        cv2.line(
            overlay,
            (x_global, bounds.y0_global),
            (x_global, bounds.y1_global - 1),
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
):
    if track is None:
        return None
    return {
        "track_id": track.track_id,
        "center_x_roi": round(track.center_x_roi, 3),
        "center_x_global": round(bounds.x0_global + track.center_x_roi, 3),
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
    analysis: AdjacentStripeAnalysis,
    rotation: RotationShadowResult,
    processing_config: ProcessingConfig,
    interactive_config: InteractiveConfig,
) -> dict:
    left = _track_report(selection.left_track, bounds, click_x_roi)
    right = _track_report(selection.right_track, bounds, click_x_roi)
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
    return {
        "algorithm": "interactive_row_run_center_voting",
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
            "applied_to_detection": False,
            "best_angle_deg": round(rotation.best_angle_deg, 3),
            "zero_angle_score": round(rotation.zero_angle_score, 8),
            "best_score": round(rotation.best_score, 8),
            "peak_separation": round(rotation.peak_separation, 8),
        },
        "interactive_result": {
            "success": selection.success,
            "failure_reasons": list(selection.failure_reasons),
            "warning_flags": warning_flags,
            "clicked_track_id": (
                None
                if selection.clicked_track is None
                else selection.clicked_track.track_id
            ),
            "stripe_spacing_px": stripe_spacing,
            "left": left,
            "right": right,
        },
        "legacy_analysis": analysis_to_dict(analysis, processing_config),
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
) -> InteractivePipelineResult:
    """Run one clicked-point case and save explainable debug outputs."""

    validate_interactive_config(interactive_config)
    if image_gray.ndim != 2:
        raise ValueError("image_gray must be a single-channel image")
    bounds = calculate_interactive_roi_bounds(
        image_gray.shape,
        click_x_global,
        click_y_global,
        interactive_config,
    )
    image_gray_roi = crop_roi_global(image_gray, bounds)
    blurred_roi = gaussian_blur_roi(image_gray_roi, processing_config)
    otsu_binary_roi = create_otsu_binary_roi(blurred_roi)
    vertical_close_roi = apply_vertical_close_roi(
        otsu_binary_roi, processing_config
    )
    close_delta_roi = create_close_delta_roi(
        otsu_binary_roi, vertical_close_roi
    )
    black_mask_roi = create_black_mask_roi(vertical_close_roi)
    click_x_roi = click_x_global - bounds.x0_global
    click_y_roi = click_y_global - bounds.y0_global
    legacy_analysis = analyze_adjacent_stripes(
        black_mask_roi,
        click_x_roi,
        click_x_global,
        bounds.x0_global,
        processing_config,
    )
    selection = select_interactive_tracks(
        black_mask_roi,
        click_x_roi,
        click_y_roi,
        legacy_analysis,
        processing_config,
    )
    rotation_shadow = estimate_rotation_shadow(
        vertical_close_roi,
        interactive_config,
    )
    original_overlay = _create_original_overlay(
        image_gray,
        bounds,
        click_x_global,
        click_y_global,
        selection,
    )
    interactive_result = _create_interactive_roi_result(
        image_gray_roi,
        click_x_roi,
        click_y_roi,
        selection,
    )
    candidates_debug = _create_interactive_candidates_debug(
        black_mask_roi,
        legacy_analysis,
        selection,
        click_x_roi,
        click_y_roi,
    )
    votes_debug = _create_interactive_votes_debug(
        legacy_analysis,
        selection,
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
        legacy_analysis,
        rotation_shadow,
        processing_config,
        interactive_config,
    )
    debug_images = {
        "original_interactive_result.png": original_overlay,
        "roi_debug.png": original_overlay,
        "roi_crop.png": image_gray_roi,
        "roi_gray.png": image_gray_roi,
        "otsu_binary.png": otsu_binary_roi,
        "vertical_close.png": vertical_close_roi,
        "close_delta.png": close_delta_roi,
        "black_mask.png": black_mask_roi,
        "black_run_candidates.png": candidates_debug,
        "stripe_center_votes.png": votes_debug,
        "interactive_result.png": interactive_result,
        "rotation_score.png": rotation_shadow.score_chart,
        "rotation_preview.png": rotation_shadow.preview_image,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, image in debug_images.items():
        save_debug_image(output_dir / filename, image)
    _save_report(output_dir / "interactive_results.json", report)
    return InteractivePipelineResult(
        click_x_global=click_x_global,
        click_y_global=click_y_global,
        click_x_roi=click_x_roi,
        click_y_roi=click_y_roi,
        bounds_global=bounds,
        legacy_analysis=legacy_analysis,
        selection=selection,
        rotation_shadow=rotation_shadow,
        report=report,
        debug_images=debug_images,
        output_dir=output_dir,
    )
