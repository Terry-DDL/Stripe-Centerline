"""Configuration for ROI selection and preprocessing debug images."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ProcessingConfig:
    """Tunable paths, ROI ratios, and debug drawing settings."""

    image_path: Path = PROJECT_ROOT / "images" / "Sample 1.bmp"
    sweep_image_paths: tuple[Path, ...] = (
        PROJECT_ROOT / "images" / "Sample 1.bmp",
        PROJECT_ROOT / "images" / "Sample 2.bmp",
        PROJECT_ROOT / "images" / "Stripe_01_date20250601_t113239765.bmp",
        PROJECT_ROOT / "images" / "Stripe_02_e0_t105204227_v-41p8_do.bmp",
        PROJECT_ROOT / "images" / "Stripe_03_e0_t105348300_v-41p8_do.bmp",
        PROJECT_ROOT / "images" / "Stripe_04_e0_t105403334_v7p8651_do.bmp",
        PROJECT_ROOT / "images" / "Stripe_05_e0_t145409670_v0p10745_do.bmp",
        PROJECT_ROOT / "images" / "Stripe_06_e0_t160727286_v0p9056_do.bmp",
        PROJECT_ROOT / "images" / "Stripe_07_e0_t160816815_v12p0066_do.bmp",
        PROJECT_ROOT / "images" / "Stripe_08_e0_t160911003_v13p2083_do.bmp",
        PROJECT_ROOT / "images" / "Stripe_09_e0_t200229235_v3p02565_do.bmp",
        PROJECT_ROOT / "images" / "Stripe_10_e0_t221236602_v8p56736_retry.bmp",
        PROJECT_ROOT / "images" / "Stripe_11_e0_t225623761_v5p9823_do.bmp",
        PROJECT_ROOT / "images" / "Stripe_12_e1_t220320710_v7p3386_do.bmp",
    )
    sweep_ref_x_globals: tuple[int, ...] = (
        850,
        900,
        950,
        1000,
        1050,
        1100,
        1150,
    )
    output_root_dir: Path = PROJECT_ROOT / "outputs"
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
    def output_dir(self) -> Path:
        """Return the image-specific output directory."""

        output_folder_name = self.image_path.stem.replace(" ", "_")
        return self.output_root_dir / output_folder_name

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


@dataclass(frozen=True)
class InteractiveConfig:
    """Tunable settings used only by interactive point analysis."""

    output_root_dir: Path = PROJECT_ROOT / "outputs" / "interactive"
    roi_half_width_px: int = 250
    roi_half_height_px: int = 250
    rotation_min_angle_deg: float = -10.0
    rotation_max_angle_deg: float = 10.0
    rotation_angle_step_deg: float = 0.25
    display_max_width_px: int = 1100


CONFIG = ProcessingConfig()
INTERACTIVE_CONFIG = InteractiveConfig()
