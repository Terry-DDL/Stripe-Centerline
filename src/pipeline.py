"""Run one complete reference-line analysis and save its debug outputs."""

from dataclasses import dataclass
from pathlib import Path

from config import ProcessingConfig
from image_processing import (
    RoiBoundsGlobal,
    apply_vertical_close_roi,
    calculate_reference_point_global,
    calculate_roi_bounds_global,
    create_black_mask_roi,
    create_close_delta_roi,
    create_original_with_centerline,
    create_otsu_binary_roi,
    create_roi_debug_image,
    crop_roi_global,
    gaussian_blur_roi,
    save_debug_image,
)
from stripe_analysis import (
    AdjacentStripeAnalysis,
    analyze_adjacent_stripes,
    create_adjacent_stripes_result_debug,
    create_black_run_candidates_debug,
    create_stripe_center_votes_debug,
    save_analysis_json,
)


@dataclass(frozen=True)
class PipelineResult:
    """Reference and ROI data returned by one complete pipeline run."""

    x_ref_global: int
    y_ref_global: int
    bounds_global: RoiBoundsGlobal
    stripe_analysis: AdjacentStripeAnalysis


def run_reference_case(
    image_gray,
    x_ref_global: int,
    output_dir: Path,
    config: ProcessingConfig,
) -> PipelineResult:
    """Run the existing Stage 1-3 pipeline for one global reference x."""

    _height_global, width_global = image_gray.shape[:2]
    if not isinstance(x_ref_global, int):
        raise ValueError("x_ref_global must be an integer")
    if not 0 <= x_ref_global < width_global:
        raise ValueError(
            f"x_ref_global must be between 0 and {width_global - 1}"
        )

    _default_x_ref, y_ref_global = calculate_reference_point_global(
        image_gray.shape, config
    )
    bounds_global = calculate_roi_bounds_global(
        image_gray.shape, x_ref_global, y_ref_global, config
    )

    image_with_centerline = create_original_with_centerline(
        image_gray, x_ref_global, config
    )
    roi_debug = create_roi_debug_image(
        image_with_centerline, bounds_global, config
    )
    image_gray_roi = crop_roi_global(image_gray, bounds_global)
    image_blurred_roi = gaussian_blur_roi(image_gray_roi, config)
    otsu_binary_roi = create_otsu_binary_roi(image_blurred_roi)
    vertical_close_roi = apply_vertical_close_roi(otsu_binary_roi, config)
    close_delta_roi = create_close_delta_roi(
        otsu_binary_roi, vertical_close_roi
    )
    black_mask_roi = create_black_mask_roi(vertical_close_roi)
    x_ref_roi = x_ref_global - bounds_global.x0_global
    stripe_analysis = analyze_adjacent_stripes(
        black_mask_roi,
        x_ref_roi,
        x_ref_global,
        bounds_global.x0_global,
        config,
    )
    black_run_candidates = create_black_run_candidates_debug(
        black_mask_roi, stripe_analysis, config
    )
    stripe_center_votes = create_stripe_center_votes_debug(stripe_analysis)
    adjacent_stripes_result = create_adjacent_stripes_result_debug(
        image_gray_roi, stripe_analysis
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_debug_image(
        output_dir / config.original_with_centerline_filename,
        image_with_centerline,
    )
    save_debug_image(output_dir / config.roi_debug_filename, roi_debug)
    save_debug_image(output_dir / config.roi_crop_filename, image_gray_roi)
    save_debug_image(output_dir / config.roi_gray_filename, image_gray_roi)
    save_debug_image(output_dir / config.otsu_binary_filename, otsu_binary_roi)
    save_debug_image(
        output_dir / config.vertical_close_filename, vertical_close_roi
    )
    save_debug_image(output_dir / config.close_delta_filename, close_delta_roi)
    save_debug_image(output_dir / config.black_mask_filename, black_mask_roi)
    save_debug_image(
        output_dir / config.black_run_candidates_filename,
        black_run_candidates,
    )
    save_debug_image(
        output_dir / config.stripe_center_votes_filename,
        stripe_center_votes,
    )
    save_debug_image(
        output_dir / config.adjacent_stripes_result_filename,
        adjacent_stripes_result,
    )
    save_analysis_json(
        output_dir / config.stripe_results_filename,
        stripe_analysis,
        config,
    )

    return PipelineResult(
        x_ref_global=x_ref_global,
        y_ref_global=y_ref_global,
        bounds_global=bounds_global,
        stripe_analysis=stripe_analysis,
    )
