"""Run the ROI selection and preprocessing debug-image pipeline."""

from config import CONFIG
from image_processing import (
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
    load_grayscale_image,
    save_debug_image,
    validate_config,
)
from stripe_analysis import (
    analysis_to_dict,
    analyze_adjacent_stripes,
    create_adjacent_stripes_result_debug,
    create_black_run_candidates_debug,
    create_stripe_center_votes_debug,
    save_analysis_json,
)


def main() -> int:
    """Create and save ROI selection and preprocessing debug images."""

    try:
        validate_config(CONFIG)
        image_gray = load_grayscale_image(CONFIG.image_path)

        x_ref_global, y_ref_global = calculate_reference_point_global(
            image_gray.shape, CONFIG
        )
        bounds_global = calculate_roi_bounds_global(
            image_gray.shape, x_ref_global, y_ref_global, CONFIG
        )

        image_with_centerline = create_original_with_centerline(
            image_gray, x_ref_global, CONFIG
        )
        roi_debug = create_roi_debug_image(
            image_with_centerline, bounds_global, CONFIG
        )
        image_gray_roi = crop_roi_global(image_gray, bounds_global)
        image_blurred_roi = gaussian_blur_roi(image_gray_roi, CONFIG)
        otsu_binary_roi = create_otsu_binary_roi(image_blurred_roi)
        vertical_close_roi = apply_vertical_close_roi(otsu_binary_roi, CONFIG)
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
            CONFIG,
        )
        black_run_candidates = create_black_run_candidates_debug(
            black_mask_roi, stripe_analysis, CONFIG
        )
        stripe_center_votes = create_stripe_center_votes_debug(stripe_analysis)
        adjacent_stripes_result = create_adjacent_stripes_result_debug(
            image_gray_roi, stripe_analysis
        )

        CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
        save_debug_image(
            CONFIG.original_with_centerline_path, image_with_centerline
        )
        save_debug_image(CONFIG.roi_debug_path, roi_debug)
        save_debug_image(CONFIG.roi_crop_path, image_gray_roi)
        save_debug_image(CONFIG.roi_gray_path, image_gray_roi)
        save_debug_image(CONFIG.otsu_binary_path, otsu_binary_roi)
        save_debug_image(CONFIG.vertical_close_path, vertical_close_roi)
        save_debug_image(CONFIG.close_delta_path, close_delta_roi)
        save_debug_image(CONFIG.black_mask_path, black_mask_roi)
        save_debug_image(CONFIG.black_run_candidates_path, black_run_candidates)
        save_debug_image(CONFIG.stripe_center_votes_path, stripe_center_votes)
        save_debug_image(
            CONFIG.adjacent_stripes_result_path, adjacent_stripes_result
        )
        save_analysis_json(CONFIG.stripe_results_path, stripe_analysis, CONFIG)

        print(f"image shape: {image_gray.shape}")
        print(f"x_ref_global: {x_ref_global}")
        print(f"y_ref_global: {y_ref_global}")
        print(
            "ROI boundary: "
            f"x0={bounds_global.x0_global}, "
            f"x1={bounds_global.x1_global}, "
            f"y0={bounds_global.y0_global}, "
            f"y1={bounds_global.y1_global}"
        )
        print(
            "ROI width and height: "
            f"{bounds_global.width_roi} x {bounds_global.height_roi}"
        )
        result = analysis_to_dict(stripe_analysis, CONFIG)["result"]
        print(f"stripe detection success: {result['success']}")
        for side in ("left", "right"):
            side_result = result[side]
            if side_result is None:
                print(f"{side}: no valid stripe")
                continue
            print(
                f"{side}: "
                f"center_x_roi={side_result['center_x_roi']}, "
                f"center_x_global={side_result['center_x_global']}, "
                f"distance_px={side_result['distance_px']}, "
                f"valid_row_ratio={side_result['valid_row_ratio']}"
            )
        return 0 if stripe_analysis.success else 1
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
