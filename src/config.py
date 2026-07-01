"""Configuration for the first ROI debug-image version."""

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

    ref_x_ratio: float = 0.5
    ref_y_ratio: float = 0.5
    roi_half_width_ratio: float = 0.15
    roi_half_height_ratio: float = 0.25

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


CONFIG = ProcessingConfig()
