"""Small OpenCV helpers for reference-line and ROI debug images."""

from dataclasses import dataclass
from pathlib import Path

import cv2

from config import ProcessingConfig


@dataclass(frozen=True)
class RoiBoundsGlobal:
    """ROI boundaries in global image coordinates; x1/y1 are exclusive."""

    x0_global: int
    x1_global: int
    y0_global: int
    y1_global: int

    @property
    def width_roi(self) -> int:
        return self.x1_global - self.x0_global

    @property
    def height_roi(self) -> int:
        return self.y1_global - self.y0_global


def validate_config(config: ProcessingConfig) -> None:
    """Raise a clear error when a tunable ratio or drawing value is invalid."""

    if not 0.0 <= config.ref_x_ratio <= 1.0:
        raise ValueError("ref_x_ratio must be between 0.0 and 1.0")
    if not 0.0 <= config.ref_y_ratio <= 1.0:
        raise ValueError("ref_y_ratio must be between 0.0 and 1.0")
    if not 0.0 < config.roi_half_width_ratio <= 1.0:
        raise ValueError("roi_half_width_ratio must be greater than 0.0 and at most 1.0")
    if not 0.0 < config.roi_half_height_ratio <= 1.0:
        raise ValueError("roi_half_height_ratio must be greater than 0.0 and at most 1.0")
    if len(config.blur_kernel) != 2:
        raise ValueError("blur_kernel must contain width and height")

    blur_kernel_width, blur_kernel_height = config.blur_kernel
    if blur_kernel_width <= 0 or blur_kernel_width % 2 == 0:
        raise ValueError("blur_kernel width must be a positive odd number")
    if blur_kernel_height <= 0 or blur_kernel_height % 2 == 0:
        raise ValueError("blur_kernel height must be a positive odd number")
    if config.close_kernel_width <= 0 or config.close_kernel_width % 2 == 0:
        raise ValueError("close_kernel_width must be a positive odd number")
    if config.close_kernel_height <= 0 or config.close_kernel_height % 2 == 0:
        raise ValueError("close_kernel_height must be a positive odd number")
    if config.stripe_search_radius_px <= 0:
        raise ValueError("stripe_search_radius_px must be greater than 0")
    if config.min_run_width_px <= 0:
        raise ValueError("min_run_width_px must be greater than 0")
    if config.max_run_width_px < config.min_run_width_px:
        raise ValueError("max_run_width_px must be at least min_run_width_px")
    if config.center_cluster_tolerance_px <= 0:
        raise ValueError("center_cluster_tolerance_px must be greater than 0")
    if config.max_width_deviation_ratio < 0.0:
        raise ValueError("max_width_deviation_ratio must not be negative")
    if config.min_width_tolerance_px < 0:
        raise ValueError("min_width_tolerance_px must not be negative")
    if not 0.0 < config.min_stripe_support_ratio <= 1.0:
        raise ValueError("min_stripe_support_ratio must be between 0.0 and 1.0")
    if config.line_thickness <= 0:
        raise ValueError("line_thickness must be greater than 0")


def load_grayscale_image(image_path: Path):
    """Load an image as grayscale and fail clearly if OpenCV cannot read it."""

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise ValueError(f"OpenCV could not read the input image: {image_path}")
    if image_gray.size == 0:
        raise ValueError(f"Input image is empty: {image_path}")
    return image_gray


def calculate_reference_point_global(
    image_shape: tuple[int, ...], config: ProcessingConfig
) -> tuple[int, int]:
    """Calculate the configurable reference point in global coordinates."""

    height_global, width_global = image_shape[:2]
    x_ref_global = round(width_global * config.ref_x_ratio)
    y_ref_global = round(height_global * config.ref_y_ratio)

    x_ref_global = min(max(x_ref_global, 0), width_global - 1)
    y_ref_global = min(max(y_ref_global, 0), height_global - 1)
    return x_ref_global, y_ref_global


def calculate_roi_bounds_global(
    image_shape: tuple[int, ...],
    x_ref_global: int,
    y_ref_global: int,
    config: ProcessingConfig,
) -> RoiBoundsGlobal:
    """Calculate an image-clipped ROI using exclusive upper boundaries."""

    height_global, width_global = image_shape[:2]
    half_width_roi = round(width_global * config.roi_half_width_ratio)
    half_height_roi = round(height_global * config.roi_half_height_ratio)

    bounds_global = RoiBoundsGlobal(
        x0_global=max(0, x_ref_global - half_width_roi),
        x1_global=min(width_global, x_ref_global + half_width_roi),
        y0_global=max(0, y_ref_global - half_height_roi),
        y1_global=min(height_global, y_ref_global + half_height_roi),
    )
    if bounds_global.width_roi <= 0 or bounds_global.height_roi <= 0:
        raise ValueError("Calculated ROI is empty")
    return bounds_global


def create_original_with_centerline(
    image_gray, x_ref_global: int, config: ProcessingConfig
):
    """Return a color copy of the image with the vertical reference line."""

    image_with_centerline = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    height_global = image_gray.shape[0]
    cv2.line(
        image_with_centerline,
        (x_ref_global, 0),
        (x_ref_global, height_global - 1),
        config.reference_line_color_bgr,
        config.line_thickness,
    )
    return image_with_centerline


def create_roi_debug_image(
    image_with_centerline,
    bounds_global: RoiBoundsGlobal,
    config: ProcessingConfig,
):
    """Return the full image with the global ROI boundary marked."""

    roi_debug = image_with_centerline.copy()
    cv2.rectangle(
        roi_debug,
        (bounds_global.x0_global, bounds_global.y0_global),
        (bounds_global.x1_global - 1, bounds_global.y1_global - 1),
        config.roi_rectangle_color_bgr,
        config.line_thickness,
    )
    return roi_debug


def crop_roi_global(image_gray, bounds_global: RoiBoundsGlobal):
    """Crop the grayscale ROI using global, half-open boundaries."""

    image_roi = image_gray[
        bounds_global.y0_global : bounds_global.y1_global,
        bounds_global.x0_global : bounds_global.x1_global,
    ].copy()
    if image_roi.size == 0:
        raise ValueError("Cropped ROI is empty")
    return image_roi


def gaussian_blur_roi(image_gray_roi, config: ProcessingConfig):
    """Smooth small grayscale noise before Otsu thresholding."""

    return cv2.GaussianBlur(image_gray_roi, config.blur_kernel, 0)


def create_otsu_binary_roi(image_blurred_roi):
    """Create a binary mask where bright stripe pixels are white."""

    _, otsu_binary_roi = cv2.threshold(
        image_blurred_roi,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return otsu_binary_roi


def apply_vertical_close_roi(otsu_binary_roi, config: ProcessingConfig):
    """Connect short vertical gaps in white stripes."""

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (config.close_kernel_width, config.close_kernel_height),
    )
    return cv2.morphologyEx(otsu_binary_roi, cv2.MORPH_CLOSE, close_kernel)


def create_close_delta_roi(otsu_binary_roi, vertical_close_roi):
    """Keep only white pixels newly filled by the close operation."""

    return cv2.bitwise_and(
        vertical_close_roi,
        cv2.bitwise_not(otsu_binary_roi),
    )


def create_black_mask_roi(vertical_close_roi):
    """Create the black-region mask by inverting the closed white mask."""

    return cv2.bitwise_not(vertical_close_roi)


def save_debug_image(output_path: Path, image) -> None:
    """Save one debug image and report an explicit write failure."""

    try:
        saved = cv2.imwrite(str(output_path), image)
    except cv2.error as error:
        raise OSError(f"Could not save output image: {output_path}") from error

    if not saved:
        raise OSError(f"Could not save output image: {output_path}")
