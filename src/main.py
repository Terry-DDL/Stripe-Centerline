"""Run one configured image through the complete debug pipeline."""

from config import CONFIG
from image_processing import (
    calculate_reference_point_global,
    load_grayscale_image,
    validate_config,
)
from pipeline import run_reference_case
from stripe_analysis import analysis_to_dict


def main() -> int:
    """Run and report the configured single-image analysis."""

    try:
        validate_config(CONFIG)
        image_gray = load_grayscale_image(CONFIG.image_path)

        x_ref_global, _y_ref_global = calculate_reference_point_global(
            image_gray.shape, CONFIG
        )
        pipeline_result = run_reference_case(
            image_gray,
            x_ref_global,
            CONFIG.output_dir,
            CONFIG,
        )
        bounds_global = pipeline_result.bounds_global
        stripe_analysis = pipeline_result.stripe_analysis

        print(f"image shape: {image_gray.shape}")
        print(f"x_ref_global: {x_ref_global}")
        print(f"y_ref_global: {pipeline_result.y_ref_global}")
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
