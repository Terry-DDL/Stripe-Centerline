"""Run the first reference-line and ROI debug-image pipeline."""

from config import CONFIG
from image_processing import (
    calculate_reference_point_global,
    calculate_roi_bounds_global,
    create_original_with_centerline,
    create_roi_debug_image,
    crop_roi_global,
    load_grayscale_image,
    save_debug_image,
    validate_config,
)


def main() -> int:
    """Create and save the three requested debug images."""

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
        image_roi = crop_roi_global(image_gray, bounds_global)

        CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
        save_debug_image(
            CONFIG.original_with_centerline_path, image_with_centerline
        )
        save_debug_image(CONFIG.roi_debug_path, roi_debug)
        save_debug_image(CONFIG.roi_crop_path, image_roi)

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
        return 0
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
