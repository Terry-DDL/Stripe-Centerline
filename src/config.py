"""Configuration for ROI selection and preprocessing debug images."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ProcessingConfig:
    """Tunable paths, ROI ratios, and debug drawing settings."""

    image_path: Path = PROJECT_ROOT / "images" / "Sample 1.bmp"
    output_dir: Path = PROJECT_ROOT / "outputs"
    original_with_centerline_filename: str = "original_with_centerline.png"
    roi_debug_filename: str = "roi_debug.png"
    roi_crop_filename: str = "roi_crop.png"
    roi_gray_filename: str = "roi_gray.png"
    otsu_binary_filename: str = "otsu_binary.png"
    vertical_close_filename: str = "vertical_close.png"
    close_delta_filename: str = "close_delta.png"
    black_mask_filename: str = "black_mask.png"
    black_run_candidates_filename: str = "black_run_candidates.png"
    stripe_center_votes_filename: str = "stripe_center_votes.png"
    adjacent_stripes_result_filename: str = "adjacent_stripes_result.png"
    stripe_results_filename: str = "stripe_results.json"

    ref_x_ratio: float = 0.5
    ref_y_ratio: float = 0.5
    roi_half_width_ratio: float = 0.15
    roi_half_height_ratio: float = 0.25

    blur_kernel: tuple[int, int] = (5, 5)
    close_kernel_width: int = 3
    close_kernel_height: int = 25

    stripe_search_radius_px: int = 120
    min_run_width_px: int = 3
    max_run_width_px: int = 120
    center_cluster_tolerance_px: int = 5
    max_width_deviation_ratio: float = 0.75
    min_width_tolerance_px: int = 3
    min_stripe_support_ratio: float = 0.5
    reject_border_touching_runs: bool = True

    reference_line_color_bgr: tuple[int, int, int] = (0, 0, 255)
    roi_rectangle_color_bgr: tuple[int, int, int] = (0, 255, 0)
    line_thickness: int = 2

    @property
    def original_with_centerline_path(self) -> Path:
        return self.output_dir / self.original_with_centerline_filename

    @property
    def roi_debug_path(self) -> Path:
        return self.output_dir / self.roi_debug_filename

    @property
    def roi_crop_path(self) -> Path:
        return self.output_dir / self.roi_crop_filename

    @property
    def roi_gray_path(self) -> Path:
        return self.output_dir / self.roi_gray_filename

    @property
    def otsu_binary_path(self) -> Path:
        return self.output_dir / self.otsu_binary_filename

    @property
    def vertical_close_path(self) -> Path:
        return self.output_dir / self.vertical_close_filename

    @property
    def close_delta_path(self) -> Path:
        return self.output_dir / self.close_delta_filename

    @property
    def black_mask_path(self) -> Path:
        return self.output_dir / self.black_mask_filename

    @property
    def black_run_candidates_path(self) -> Path:
        return self.output_dir / self.black_run_candidates_filename

    @property
    def stripe_center_votes_path(self) -> Path:
        return self.output_dir / self.stripe_center_votes_filename

    @property
    def adjacent_stripes_result_path(self) -> Path:
        return self.output_dir / self.adjacent_stripes_result_filename

    @property
    def stripe_results_path(self) -> Path:
        return self.output_dir / self.stripe_results_filename


CONFIG = ProcessingConfig()
