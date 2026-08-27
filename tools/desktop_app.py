"""Tkinter desktop UI for interactive single-image stripe analysis."""

import argparse
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import uuid

import cv2
import numpy as np
from PIL import Image, ImageTk


PROJECT_ROOT = (
    Path(sys._MEIPASS)
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
    else Path(__file__).resolve().parent.parent
)
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from config import CONFIG, INTERACTIVE_CONFIG  # noqa: E402
from interactive_pipeline import (  # noqa: E402
    calculate_interactive_roi_bounds,
    run_interactive_case,
)
from pitch_reference import build_pitch_reference_map  # noqa: E402
from tools.basin_shadow_integration import (  # noqa: E402
    ShadowLogConfig,
    run_shadow_and_log,
)
from tools.desktop_performance import (  # noqa: E402
    DEFAULT_LOG_ROOT as PERFORMANCE_LOG_ROOT,
    elapsed_ms,
    update_timing,
)
from tools.stage3_desktop_runtime import (  # noqa: E402
    RESULT_SOURCE as STAGE3_RESULT_SOURCE,
    build_desktop_result,
    persist_formal_result,
    run_frozen_stage3,
)


SUPPORTED_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg"}
MINIMUM_RELIABLE_MACOS_TK = (8, 6, 13)
MOUSE_WHEEL_SCROLL_PIXELS = 54
DEBUG_ENVIRONMENT_VARIABLE = "STRIPE_CENTERLINE_DEBUG"
CROSS_IMAGE_EXPERIMENT_ENVIRONMENT_VARIABLE = (
    "STRIPE_CENTERLINE_CROSS_IMAGE_EXPERIMENT"
)
DEBUG_TRUE_VALUES = {"1", "true", "yes", "on"}
CROSS_IMAGE_EXPERIMENT_START_DELAY_SECONDS = 0.25


@dataclass
class DesktopSelectionState:
    """Selection data that must be cleared when the image changes."""

    image_identity: str | None = None
    zoom_center: tuple[int, int] | None = None
    click_point: tuple[int, int] | None = None
    result: object | None = None

    def select_image(self, image_identity: str) -> bool:
        """Reset stale interaction state when a different image is loaded."""

        if self.image_identity == image_identity:
            return False
        self.image_identity = image_identity
        self.zoom_center = None
        self.click_point = None
        self.result = None
        return True

    def select_reference_point(self, point: tuple[int, int]) -> None:
        """Select a final point and clear the result from any earlier point."""

        self.zoom_center = point
        self.click_point = point
        self.result = None


@dataclass(frozen=True)
class AnalysisCompletion:
    """One formal Stage 3.1 worker result returned to the Tk thread."""

    analysis_key: tuple
    result: object | None
    error: Exception | None
    run_id: str
    click_started_ns: int
    output_dir: Path
    performance_metadata: dict
    stage3_algorithm_ms: float | None


def desktop_debug_enabled(
    cli_debug: bool = False,
    environment_value: str | None = None,
) -> bool:
    """Enable diagnostic work only through an explicit release switch."""

    if cli_debug:
        return True
    if environment_value is None:
        environment_value = os.environ.get(DEBUG_ENVIRONMENT_VARIABLE, "")
    return environment_value.strip().lower() in DEBUG_TRUE_VALUES


def cross_image_experiment_enabled(
    environment_value: str | None = None,
) -> bool:
    """Enable the isolated v1.1 shadow only through its explicit switch."""

    if environment_value is None:
        environment_value = os.environ.get(
            CROSS_IMAGE_EXPERIMENT_ENVIRONMENT_VARIABLE,
            "",
        )
    return environment_value.strip().lower() in DEBUG_TRUE_VALUES


def parse_desktop_arguments(argv=None):
    """Parse release-only switches without changing detector parameters."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable legacy comparison and diagnostic disk outputs",
    )
    parser.add_argument(
        "--startup-smoke-test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def project_image_paths() -> list[Path]:
    """Return supported project images in stable filename order."""

    image_dir = PROJECT_ROOT / "images"
    if not image_dir.is_dir():
        return []
    return sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: path.name.lower(),
    )


def desktop_output_root() -> Path:
    """Use a writable user-data location inside the packaged application."""

    if getattr(sys, "frozen", False):
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "StripeCenterline"
            / "outputs"
        )
    return INTERACTIVE_CONFIG.output_root_dir


def desktop_performance_log_root() -> Path:
    """Keep optional packaged diagnostics beside packaged formal outputs."""

    if getattr(sys, "frozen", False):
        return desktop_output_root() / "diagnostics" / "performance"
    return PERFORMANCE_LOG_ROOT


def decode_grayscale_image(image_bytes: bytes):
    """Decode image bytes as one grayscale OpenCV array."""

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image_gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image_gray is None or image_gray.size == 0:
        raise ValueError("OpenCV could not decode the selected image")
    return image_gray


def safe_stem(filename: str) -> str:
    """Return a short filesystem-safe name for an interactive output."""

    stem = Path(filename).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return safe or "image"


def image_hash(image_bytes: bytes) -> str:
    """Return the short content hash used by interactive output folders."""

    return hashlib.sha256(image_bytes).hexdigest()[:8]


def get_or_build_pitch_reference(
    cache: dict,
    image_identity: str,
    image_gray,
    processing_config=CONFIG,
    interactive_config=INTERACTIVE_CONFIG,
    builder=build_pitch_reference_map,
):
    """Return the cached whole-image pitch map or build it once."""

    pitch_map = cache.get(image_identity)
    if pitch_map is None:
        pitch_map = builder(
            image_gray,
            processing_config,
            interactive_config,
        )
        cache[image_identity] = pitch_map
    return pitch_map


def version_numbers(version: str) -> tuple[int, ...]:
    """Return numeric version parts such as 8.6.12 -> (8, 6, 12)."""

    parts = []
    for part in version.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def reliable_tk_runtime(
    patchlevel: str,
    platform_name: str = sys.platform,
) -> bool:
    """Reject the Tk version known to lose mouse events on modern macOS."""

    if platform_name != "darwin":
        return True
    return version_numbers(patchlevel) >= MINIMUM_RELIABLE_MACOS_TK


def signed_16_bit(value: int) -> int:
    """Interpret the low 16 bits of one integer as a signed value."""

    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def unpack_touchpad_scroll_delta(packed_delta: int) -> tuple[int, int]:
    """Return the horizontal and vertical deltas packed by Tk 9."""

    packed_delta &= 0xFFFFFFFF
    return (
        signed_16_bit(packed_delta >> 16),
        signed_16_bit(packed_delta),
    )


def mouse_wheel_scroll_pixels(delta: int) -> int:
    """Convert Tk wheel deltas into a short, predictable pixel movement."""

    if delta == 0:
        return 0
    notches = delta / 120 if abs(delta) >= 120 else delta
    notches = max(-8, min(8, notches))
    return round(-notches * MOUSE_WHEEL_SCROLL_PIXELS)


def build_output_dir(
    image_name: str,
    image_bytes: bytes,
    click_point: tuple[int, int],
    output_root: Path | None = None,
) -> Path:
    """Build the unchanged interactive output directory for one click."""

    if output_root is None:
        output_root = desktop_output_root()
    click_x, click_y = click_point
    return (
        output_root
        / f"{safe_stem(image_name)}_{image_hash(image_bytes)}"
        / f"point_{click_x}_{click_y}"
    )


def calculate_display_size(
    image_shape: tuple[int, ...],
    max_width: int,
    max_height: int | None = None,
) -> tuple[int, int]:
    """Return an aspect-preserving display size that never enlarges."""

    height, width = image_shape[:2]
    scale = min(1.0, max_width / width)
    if max_height is not None:
        scale = min(scale, max_height / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize_for_display(
    image,
    max_width: int,
    max_height: int | None = None,
    nearest: bool = False,
):
    """Resize one display copy and return exact source coordinate scales."""

    display_width, display_height = calculate_display_size(
        image.shape,
        max_width,
        max_height,
    )
    height, width = image.shape[:2]
    if (display_width, display_height) == (width, height):
        display = image.copy()
    else:
        interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
        display = cv2.resize(
            image,
            (display_width, display_height),
            interpolation=interpolation,
        )
    return display, width / display_width, height / display_height


def map_display_point_to_source(
    x_display: float,
    y_display: float,
    scale_x_to_source: float,
    scale_y_to_source: float,
    image_shape: tuple[int, ...],
    x0_source: int = 0,
    y0_source: int = 0,
) -> tuple[int, int]:
    """Map one desktop-image click back to a clipped source coordinate."""

    height, width = image_shape[:2]
    x_source = x0_source + round(x_display * scale_x_to_source)
    y_source = y0_source + round(y_display * scale_y_to_source)
    return (
        min(width - 1, max(0, x_source)),
        min(height - 1, max(0, y_source)),
    )


def extract_centered_region(
    image,
    center_point: tuple[int, int],
    width: int,
    height: int,
):
    """Extract a fixed-size region, replicating pixels beyond image edges."""

    if width < 1 or height < 1:
        raise ValueError("region width and height must be positive")

    image_height, image_width = image.shape[:2]
    center_x, center_y = center_point
    x0 = center_x - width // 2
    y0 = center_y - height // 2
    x1 = x0 + width
    y1 = y0 + height

    clipped_x0 = max(0, x0)
    clipped_y0 = max(0, y0)
    clipped_x1 = min(image_width, x1)
    clipped_y1 = min(image_height, y1)
    region = image[clipped_y0:clipped_y1, clipped_x0:clipped_x1]
    return cv2.copyMakeBorder(
        region,
        clipped_y0 - y0,
        y1 - clipped_y1,
        clipped_x0 - x0,
        x1 - clipped_x1,
        cv2.BORDER_REPLICATE,
    )


def create_magnifier_display(
    image_gray,
    center_point: tuple[int, int],
    source_width: int,
    source_height: int,
    scale: float,
):
    """Create an RGB magnifier image centered on one source coordinate."""

    region = extract_centered_region(
        image_gray,
        center_point,
        source_width,
        source_height,
    )
    display_width = max(1, round(source_width * scale))
    display_height = max(1, round(source_height * scale))
    display = cv2.cvtColor(region, cv2.COLOR_GRAY2RGB)
    display = cv2.resize(
        display,
        (display_width, display_height),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.drawMarker(
        display,
        (display_width // 2, display_height // 2),
        (255, 0, 0),
        cv2.MARKER_CROSS,
        17,
        1,
    )
    return display


def magnifier_canvas_position(
    pointer_x: int,
    pointer_y: int,
    magnifier_width: int,
    magnifier_height: int,
    canvas_width: int,
    canvas_height: int,
    offset: int,
) -> tuple[int, int]:
    """Place a magnifier beside the pointer and keep it inside the canvas."""

    x = pointer_x + offset
    y = pointer_y + offset
    if x + magnifier_width > canvas_width:
        x = pointer_x - offset - magnifier_width
    if y + magnifier_height > canvas_height:
        y = pointer_y - offset - magnifier_height
    return (
        max(0, min(x, max(0, canvas_width - magnifier_width))),
        max(0, min(y, max(0, canvas_height - magnifier_height))),
    )


def zoom_bounds(
    image_shape: tuple[int, ...],
    center_point: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return the clipped source rectangle for precise point selection."""

    height, width = image_shape[:2]
    center_x, center_y = center_point
    return (
        max(0, center_x - INTERACTIVE_CONFIG.selection_zoom_half_width_px),
        max(0, center_y - INTERACTIVE_CONFIG.selection_zoom_half_height_px),
        min(width, center_x + INTERACTIVE_CONFIG.selection_zoom_half_width_px),
        min(height, center_y + INTERACTIVE_CONFIG.selection_zoom_half_height_px),
    )


def create_zoom_display(
    image_gray,
    center_point: tuple[int, int],
    click_point: tuple[int, int] | None,
):
    """Create the enlarged local RGB image and its coordinate mapping."""

    x0, y0, x1, y1 = zoom_bounds(image_gray.shape, center_point)
    zoom_rgb = cv2.cvtColor(image_gray[y0:y1, x0:x1], cv2.COLOR_GRAY2RGB)
    if click_point is not None:
        click_x, click_y = click_point
        if x0 <= click_x < x1 and y0 <= click_y < y1:
            cv2.drawMarker(
                zoom_rgb,
                (click_x - x0, click_y - y0),
                (255, 0, 0),
                cv2.MARKER_CROSS,
                11,
                1,
            )

    source_height, source_width = zoom_rgb.shape[:2]
    display_width = max(
        1,
        round(source_width * INTERACTIVE_CONFIG.selection_zoom_scale),
    )
    display_height = max(
        1,
        round(source_height * INTERACTIVE_CONFIG.selection_zoom_scale),
    )
    enlarged = cv2.resize(
        zoom_rgb,
        (display_width, display_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return (
        enlarged,
        x0,
        y0,
        source_width / display_width,
        source_height / display_height,
    )


def draw_click_preview(
    image_gray,
    click_point: tuple[int, int] | None,
    zoom_center: tuple[int, int] | None,
):
    """Draw the zoom area, stored click, and ROI on the full-image copy."""

    preview = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    if zoom_center is not None:
        zoom_x0, zoom_y0, zoom_x1, zoom_y1 = zoom_bounds(
            image_gray.shape,
            zoom_center,
        )
        cv2.rectangle(
            preview,
            (zoom_x0, zoom_y0),
            (zoom_x1 - 1, zoom_y1 - 1),
            (255, 180, 0),
            2,
        )
    if click_point is None:
        return preview

    click_x, click_y = click_point
    height, width = image_gray.shape[:2]
    x0 = max(0, click_x - INTERACTIVE_CONFIG.roi_half_width_px)
    x1 = min(width, click_x + INTERACTIVE_CONFIG.roi_half_width_px)
    y0 = max(0, click_y - INTERACTIVE_CONFIG.roi_half_height_px)
    y1 = min(height, click_y + INTERACTIVE_CONFIG.roi_half_height_px)
    cv2.rectangle(preview, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
    cv2.drawMarker(
        preview,
        (click_x, click_y),
        (255, 0, 0),
        cv2.MARKER_CROSS,
        21,
        2,
    )
    return preview


def track_table_rows(result_report: dict) -> list[dict]:
    """Return compact left/right measurements for the result table."""

    rows = []
    for side in ("left", "right"):
        side_result = result_report.get(side)
        if side_result is None:
            rows.append(
                {
                    "side": side,
                    "center_x_global": "",
                    "distance_to_click_px": "",
                    "median_width_px": "",
                    "valid_row_ratio": "",
                    "retention_ratio": "",
                }
            )
        else:
            rows.append(
                {
                    "side": side,
                    "center_x_global": side_result["center_x_global"],
                    "distance_to_click_px": side_result[
                        "distance_to_click_px"
                    ],
                    "median_width_px": side_result.get(
                        "median_width_px",
                        "",
                    ),
                    "valid_row_ratio": side_result["valid_row_ratio"],
                    "retention_ratio": side_result["retention_ratio"],
                }
            )
    return rows


def basin_table_rows(result_report: dict) -> list[dict]:
    """Return Stage 3.1 basin geometry without legacy track evidence."""

    rows = []
    for side in ("left", "right"):
        side_result = result_report.get(side)
        if side_result is None:
            rows.append(
                {
                    "side": side,
                    "center_x_global": "",
                    "distance_to_click_px": "",
                    "basin_width_px": "",
                    "evidence_status": "unavailable",
                }
            )
        else:
            rows.append(
                {
                    "side": side,
                    "center_x_global": side_result["center_x_global"],
                    "distance_to_click_px": side_result[
                        "distance_to_click_px"
                    ],
                    "basin_width_px": side_result["basin_width_px"],
                    "evidence_status": side_result["evidence_status"],
                }
            )
    return rows


def raw_pitch_evidence_text(pitch: dict) -> tuple[str, str]:
    """Describe frozen raw pitch without promoting diagnostics to success."""

    ambiguity = pitch.get("harmonic_ambiguity") or {}
    confidence = pitch.get("confidence", "unavailable")
    usable = pitch.get("usable_pitch_px")
    diagnostic = pitch.get("diagnostic_pitch_px")
    if (
        confidence == "high"
        and pitch.get("success_eligible")
        and not ambiguity.get("detected")
        and usable is not None
    ):
        return "High-confidence", f"Usable local pitch: {usable:g} px"
    if ambiguity.get("detected"):
        return "Harmonic ambiguous", "Diagnostics only"
    if diagnostic is not None:
        return (
            f"{confidence.capitalize()} · diagnostics only",
            f"Diagnostic pitch: {diagnostic:g} px",
        )
    return "Unavailable", "No formal pitch evidence"


def result_performance_text(
    interactive_result: dict,
    stage3_ms: float | None,
    total_ms: float | None,
) -> str:
    """Format one timing row for either formal result status."""

    if "success" not in interactive_result:
        raise ValueError("interactive_result must include success")

    def format_ms(value: float | None) -> str:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            return "—"
        return f"{value:.0f} ms"

    return (
        f"Analysis: {format_ms(stage3_ms)}"
        f" · Total: {format_ms(total_ms)}"
    )


def completion_matches_active_request(
    completion: AnalysisCompletion,
    current_key: tuple,
    active_run_id: str | None,
) -> bool:
    """Require both click identity and request identity to match."""

    return bool(
        active_run_id is not None
        and completion.run_id == active_run_id
        and completion.analysis_key == current_key
    )


def result_metric_values(click: dict, interactive_result: dict) -> tuple:
    """Return only measurements that are safe for the formal result."""

    values = [
        (
            "Reference point",
            f"({click['x_global']}, {click['y_global']})",
            "",
        )
    ]
    if not interactive_result["success"]:
        return tuple(values)

    def distance_text(side_result) -> str:
        if side_result is None:
            return "unavailable"
        return f"{side_result['distance_to_click_px']:g} px"

    spacing = interactive_result["stripe_spacing_px"]
    if interactive_result.get("result_source") == STAGE3_RESULT_SOURCE:
        evidence_title = "Raw pitch evidence"
        pitch_status, pitch_details = raw_pitch_evidence_text(
            interactive_result["pitch_evidence"]
        )
    else:
        evidence_title = "Pitch safety"
        pitch_status, pitch_details = pitch_safety_text(
            interactive_result["pitch_guard"]
        )
    values.extend(
        (
            (
                "Left distance",
                distance_text(interactive_result.get("left")),
                "",
            ),
            (
                "Right distance",
                distance_text(interactive_result.get("right")),
                "",
            ),
            (
                "Centerline spacing",
                "unavailable" if spacing is None else f"{spacing:g} px",
                "",
            ),
            (evidence_title, pitch_status, pitch_details),
        )
    )
    return tuple(values)


def pitch_safety_text(pitch_guard: dict) -> tuple[str, str]:
    """Return compact reader-facing pitch status and measurement details."""

    status = pitch_guard.get("status", "Not applicable")
    baseline = pitch_guard.get("baseline_pitch_px")
    intervals = pitch_guard.get("observed_intervals_px", [])
    ratios = pitch_guard.get("interval_pitch_ratios", [])
    if baseline is None:
        if status == "Unable to verify":
            return status, "No reliable local baseline"
        return status, "Local detection did not produce intervals"
    measurements = ", ".join(
        f"{interval:g} px ({ratio:g}×)"
        for interval, ratio in zip(intervals, ratios)
    )
    return (
        status,
        f"Average pitch: {baseline:g} px · Interval / average: {measurements}",
    )


def pitch_safety_note(pitch_guard: dict) -> str | None:
    """Explain only pitch states that need the reader's attention."""

    status, details = pitch_safety_text(pitch_guard)
    if status == "Suspicious":
        minimum = pitch_guard["normal_interval_ratio_min"]
        maximum = pitch_guard["normal_interval_ratio_max"]
        return (
            f"Pitch safety: suspicious — {details}; "
            f"normal range is {minimum:g}–{maximum:g}×."
        )
    if status == "Unable to verify":
        return "Pitch safety: unable to verify — no reliable map near this point."
    return None


def debug_image_to_rgb(image):
    """Convert OpenCV debug data to RGB for Pillow."""

    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def array_to_photo(
    image,
    master,
    max_width: int,
    max_height: int | None = None,
    nearest: bool = False,
):
    """Create a Tk image plus source/display scales from a NumPy image."""

    display, scale_x, scale_y = resize_for_display(
        image,
        max_width,
        max_height,
        nearest=nearest,
    )
    photo = ImageTk.PhotoImage(Image.fromarray(display), master=master)
    return photo, scale_x, scale_y


def create_failure_result_overlay(image_gray, result):
    """Draw a failed formal result without any candidate centerlines."""

    overlay = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    bounds = result.bounds_global
    click = result.report["click"]
    cv2.rectangle(
        overlay,
        (bounds.x0_global, bounds.y0_global),
        (bounds.x1_global - 1, bounds.y1_global - 1),
        (0, 255, 0),
        2,
    )
    cv2.drawMarker(
        overlay,
        (click["x_global"], click["y_global"]),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        21,
        2,
    )
    return overlay


def create_result_roi_crop(result, overlay=None):
    """Crop the final ROI overlay from the full original-image result."""

    bounds = result.bounds_global
    if overlay is None:
        overlay = result.debug_images["original_interactive_result.png"]
    return overlay[
        bounds.y0_global : bounds.y1_global,
        bounds.x0_global : bounds.x1_global,
    ].copy()


class ScrollableFrame(ttk.Frame):
    """A vertically scrollable frame used by the main and result windows."""

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(
            self,
            highlightthickness=0,
            yscrollincrement=1,
        )
        scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview,
        )
        self.content = ttk.Frame(self.canvas, padding=(18, 12))
        self.window_id = self.canvas.create_window(
            (0, 0),
            window=self.content,
            anchor="nw",
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.content.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_content)
        self.winfo_toplevel().bind(
            "<MouseWheel>",
            self._scroll_mouse_wheel,
            add="+",
        )
        try:
            self.winfo_toplevel().bind(
                "<TouchpadScroll>",
                self._scroll_touchpad,
                add="+",
            )
        except tk.TclError:
            # Tk 8 does not know the Tk 9 precision-scroll event.
            pass

    def _update_scroll_region(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_content(self, event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _scroll_mouse_wheel(self, event) -> None:
        if event.delta == 0 or not self._event_is_inside(event):
            return
        self.canvas.yview_scroll(
            mouse_wheel_scroll_pixels(event.delta),
            "units",
        )
        return "break"

    def _scroll_touchpad(self, event) -> None:
        if not self._event_is_inside(event):
            return
        _horizontal_delta, vertical_delta = unpack_touchpad_scroll_delta(
            event.delta
        )
        if vertical_delta == 0:
            return
        self.canvas.yview_scroll(-vertical_delta, "units")
        return "break"

    def _event_is_inside(self, event) -> bool:
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            if widget is self:
                return True
            widget = widget.master
        return False


class StripeDesktopApp:
    """Desktop controller for image selection, clicking, and result display."""

    def __init__(
        self,
        root: tk.Tk,
        debug_enabled: bool = False,
    ):
        self.root = root
        self.debug_enabled = bool(debug_enabled)
        self.state = DesktopSelectionState()
        self.image_gray = None
        self.image_bytes = b""
        self.image_name = ""
        self.full_photo = None
        self.zoom_photo = None
        self.magnifier_photo = None
        self.full_scale_x = 1.0
        self.full_scale_y = 1.0
        self.zoom_x0 = 0
        self.zoom_y0 = 0
        self.zoom_scale_x = 1.0
        self.zoom_scale_y = 1.0
        self.magnifier_pointer = None
        self.magnifier_after_id = None
        self.magnifier_image_id = None
        self.magnifier_border_id = None
        self.result_frame = None
        self.result_photos = []
        self.result_timing_label = None
        self.project_paths = project_image_paths()
        self.analysis_queue = queue.Queue()
        self.analysis_running = False
        self.active_analysis_run_id = None
        self.pitch_reference_cache = {}
        self.pitch_reference_lock = threading.Lock()

        self._configure_window()
        self._build_controls()
        self._build_content()
        self._bind_shortcuts()
        if self.project_paths:
            self.root.after(0, lambda: self.load_image(self.project_paths[0]))

    def _configure_window(self) -> None:
        self.root.title("Interactive Stripe Analysis")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=("", 22, "bold"))
        style.configure("Section.TLabel", font=("", 15, "bold"))
        style.configure("MetricValue.TLabel", font=("", 20, "bold"))
        style.configure(
            "Primary.TButton",
            font=("", 12, "bold"),
            padding=(16, 9),
        )
        style.configure(
            "Secondary.TButton",
            padding=(12, 7),
        )

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Command-o>", lambda _event: self.open_image())
        self.root.bind(
            "<Command-Return>",
            lambda _event: self.analyze(),
        )
        self.root.bind("<Escape>", lambda _event: self._show_selection_view())

    def _build_controls(self) -> None:
        controls = ttk.Frame(self.root, padding=(18, 14))
        self.controls = controls
        controls.pack(fill="x")
        ttk.Label(
            controls,
            text="Interactive Stripe Analysis",
            style="Title.TLabel",
        ).grid(row=0, column=0, columnspan=7, sticky="w", pady=(0, 8))

        ttk.Label(controls, text="Project image:").grid(
            row=1,
            column=0,
            sticky="w",
        )
        self.project_name = tk.StringVar()
        self.project_combo = ttk.Combobox(
            controls,
            textvariable=self.project_name,
            values=[path.name for path in self.project_paths],
            state="readonly" if self.project_paths else "disabled",
            width=52,
        )
        self.project_combo.grid(row=1, column=1, sticky="ew", padx=(8, 8))
        self.project_combo.bind("<<ComboboxSelected>>", self._load_project_image)
        if self.project_paths:
            self.project_combo.current(0)

        self.load_selected_button = ttk.Button(
            controls,
            text="Load selected",
            command=self._load_project_image,
            style="Secondary.TButton",
        )
        self.load_selected_button.grid(row=1, column=2, padx=(0, 8))
        self.open_image_button = ttk.Button(
            controls,
            text="Open image...",
            command=self.open_image,
            style="Secondary.TButton",
        )
        self.open_image_button.grid(row=1, column=3, padx=(0, 12))

        self.image_info = ttk.Label(controls, text="No image loaded")
        self.image_info.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(10, 0),
        )
        self.selected_point_label = ttk.Label(
            controls,
            text="Selected reference point: unavailable",
        )
        self.selected_point_label.grid(
            row=2,
            column=2,
            columnspan=2,
            sticky="w",
            padx=(10, 14),
            pady=(10, 0),
        )
        self.analyze_button = ttk.Button(
            controls,
            text="Analyze selected point",
            command=self.analyze,
            state="disabled",
            style="Primary.TButton",
        )
        self.analyze_button.grid(
            row=2,
            column=4,
            sticky="w",
            pady=(10, 0),
        )
        self.status_label = ttk.Label(controls, text="")
        self.status_label.grid(
            row=2,
            column=5,
            columnspan=2,
            sticky="w",
            padx=(14, 0),
            pady=(10, 0),
        )
        controls.columnconfigure(1, weight=1)

    def _build_content(self) -> None:
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill="both", expand=True)
        content = ttk.Frame(self.main_container, padding=(18, 8, 18, 14))
        self.selection_frame = content
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(2, weight=1)

        ttk.Label(
            content,
            text="1. Select the reference point",
            style="Section.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            content,
            text=(
                "Move over the image for a live 4× magnifier, "
                "then click to select."
            ),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 10))
        self.full_image_canvas = tk.Canvas(
            content,
            highlightthickness=1,
            highlightbackground="#777777",
            background="#202020",
            cursor="crosshair",
        )
        self.full_image_canvas.grid(
            row=2,
            column=0,
            sticky="nw",
            padx=(0, 18),
        )
        self.full_image_canvas.bind("<Motion>", self._queue_magnifier)
        self.full_image_canvas.bind("<Leave>", self._hide_magnifier)
        self.full_image_canvas.bind("<Button-1>", self._select_zoom_center)

        self.zoom_frame = ttk.Frame(content)
        self.zoom_frame.grid(row=2, column=1, sticky="nw")
        ttk.Label(
            self.zoom_frame,
            text="Optional: refine the reference point",
            style="Section.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            self.zoom_frame,
            text="The local image is enlarged 3×. Click it to fine-tune the point.",
        ).pack(anchor="w", pady=(2, 10))
        self.zoom_image_canvas = tk.Canvas(
            self.zoom_frame,
            highlightthickness=1,
            highlightbackground="#777777",
            background="#202020",
            cursor="crosshair",
        )
        self.zoom_image_canvas.pack(anchor="w")
        self.zoom_image_canvas.bind("<Button-1>", self._select_precise_point)
        self.zoom_frame.grid_remove()

    def _load_project_image(self, _event=None) -> None:
        selected_name = self.project_name.get()
        for path in self.project_paths:
            if path.name == selected_name:
                self.load_image(path)
                return

    def open_image(self) -> None:
        path_value = filedialog.askopenfilename(
            parent=self.root,
            title="Open stripe image",
            filetypes=(
                ("Supported images", "*.bmp *.png *.jpg *.jpeg"),
                ("All files", "*.*"),
            ),
        )
        if path_value:
            self.load_image(Path(path_value))

    def load_image(self, path: Path) -> None:
        self._hide_magnifier()
        try:
            image_bytes = path.read_bytes()
            image_gray = decode_grayscale_image(image_bytes)
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not load image", str(error), parent=self.root)
            return

        self.image_bytes = image_bytes
        self.image_gray = image_gray
        self.image_name = path.name
        identity = f"{path.name}:{image_hash(image_bytes)}"
        image_changed = self.state.select_image(identity)
        if image_changed:
            self._close_result_window()

        height, width = image_gray.shape[:2]
        self.image_info.configure(
            text=f"Loaded: {path.name} ({width} × {height} px)"
        )
        if self.state.click_point is None:
            self.selected_point_label.configure(
                text="Selected reference point: unavailable"
            )
        else:
            self.selected_point_label.configure(
                text=f"Selected reference point: {self.state.click_point}"
            )
        self._update_analyze_button()
        self.status_label.configure(text="Image loaded")
        self._render_full_image()
        if self.state.zoom_center is None:
            self.zoom_frame.grid_remove()
        else:
            self._render_zoom_image()
            self.zoom_frame.grid()
        self._show_selection_view()

    def _render_full_image(self) -> None:
        if self.image_gray is None:
            return
        self._hide_magnifier()
        preview = draw_click_preview(
            self.image_gray,
            self.state.click_point,
            self.state.zoom_center,
        )
        self.full_photo, self.full_scale_x, self.full_scale_y = array_to_photo(
            preview,
            self.root,
            INTERACTIVE_CONFIG.desktop_full_image_max_width_px,
            INTERACTIVE_CONFIG.desktop_full_image_max_height_px,
        )
        self.full_image_canvas.configure(
            width=self.full_photo.width(),
            height=self.full_photo.height(),
        )
        self.full_image_canvas.delete("all")
        self.full_image_canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self.full_photo,
        )

    def _queue_magnifier(self, event) -> None:
        if self.image_gray is None or self.full_photo is None:
            return
        if not (
            0 <= event.x < self.full_photo.width()
            and 0 <= event.y < self.full_photo.height()
        ):
            self._hide_magnifier()
            return
        self.magnifier_pointer = (event.x, event.y)
        if self.magnifier_after_id is None:
            self.magnifier_after_id = self.root.after(
                INTERACTIVE_CONFIG.magnifier_refresh_ms,
                self._render_magnifier,
            )

    def _render_magnifier(self) -> None:
        self.magnifier_after_id = None
        if (
            self.image_gray is None
            or self.full_photo is None
            or self.magnifier_pointer is None
        ):
            return

        pointer_x, pointer_y = self.magnifier_pointer
        source_point = map_display_point_to_source(
            pointer_x,
            pointer_y,
            self.full_scale_x,
            self.full_scale_y,
            self.image_gray.shape,
        )
        display = create_magnifier_display(
            self.image_gray,
            source_point,
            INTERACTIVE_CONFIG.magnifier_source_width_px,
            INTERACTIVE_CONFIG.magnifier_source_height_px,
            INTERACTIVE_CONFIG.magnifier_scale,
        )
        display, _scale_x, _scale_y = resize_for_display(
            display,
            self.full_photo.width(),
            self.full_photo.height(),
            nearest=True,
        )
        self.magnifier_photo = ImageTk.PhotoImage(
            Image.fromarray(display),
            master=self.root,
        )
        magnifier_x, magnifier_y = magnifier_canvas_position(
            pointer_x,
            pointer_y,
            self.magnifier_photo.width(),
            self.magnifier_photo.height(),
            self.full_photo.width(),
            self.full_photo.height(),
            INTERACTIVE_CONFIG.magnifier_offset_px,
        )
        if self.magnifier_image_id is None:
            self.magnifier_image_id = self.full_image_canvas.create_image(
                magnifier_x,
                magnifier_y,
                anchor="nw",
                image=self.magnifier_photo,
                tags=("magnifier",),
            )
            self.magnifier_border_id = self.full_image_canvas.create_rectangle(
                magnifier_x,
                magnifier_y,
                magnifier_x + self.magnifier_photo.width(),
                magnifier_y + self.magnifier_photo.height(),
                outline="#ffb000",
                width=2,
                tags=("magnifier",),
            )
        else:
            self.full_image_canvas.itemconfigure(
                self.magnifier_image_id,
                image=self.magnifier_photo,
            )
            self.full_image_canvas.coords(
                self.magnifier_image_id,
                magnifier_x,
                magnifier_y,
            )
            self.full_image_canvas.coords(
                self.magnifier_border_id,
                magnifier_x,
                magnifier_y,
                magnifier_x + self.magnifier_photo.width(),
                magnifier_y + self.magnifier_photo.height(),
            )
        self.full_image_canvas.tag_raise("magnifier")

    def _hide_magnifier(self, _event=None) -> None:
        if self.magnifier_after_id is not None:
            try:
                self.root.after_cancel(self.magnifier_after_id)
            except tk.TclError:
                pass
        self.magnifier_after_id = None
        self.magnifier_pointer = None
        self.full_image_canvas.delete("magnifier")
        self.magnifier_image_id = None
        self.magnifier_border_id = None
        self.magnifier_photo = None

    def _select_zoom_center(self, event) -> None:
        if self.image_gray is None:
            return
        point = map_display_point_to_source(
            event.x,
            event.y,
            self.full_scale_x,
            self.full_scale_y,
            self.image_gray.shape,
        )
        self.state.select_reference_point(point)
        self._close_result_window()
        self.selected_point_label.configure(
            text=f"Selected reference point: {self.state.click_point}"
        )
        self._update_analyze_button()
        if self.analysis_running:
            status = "Reference point selected; current analysis still running"
        else:
            status = "Reference point selected; refine below if needed"
        self.status_label.configure(text=status)
        self._render_full_image()
        self._render_zoom_image()
        self.zoom_frame.grid()

    def _render_zoom_image(self) -> None:
        if self.image_gray is None or self.state.zoom_center is None:
            return
        (
            zoom_display,
            self.zoom_x0,
            self.zoom_y0,
            self.zoom_scale_x,
            self.zoom_scale_y,
        ) = create_zoom_display(
            self.image_gray,
            self.state.zoom_center,
            self.state.click_point,
        )
        zoom_display, resize_scale_x, resize_scale_y = resize_for_display(
            zoom_display,
            INTERACTIVE_CONFIG.desktop_zoom_max_width_px,
            INTERACTIVE_CONFIG.desktop_zoom_max_height_px,
            nearest=True,
        )
        self.zoom_scale_x *= resize_scale_x
        self.zoom_scale_y *= resize_scale_y
        self.zoom_photo = ImageTk.PhotoImage(
            Image.fromarray(zoom_display),
            master=self.root,
        )
        self.zoom_image_canvas.configure(
            width=self.zoom_photo.width(),
            height=self.zoom_photo.height(),
        )
        self.zoom_image_canvas.delete("all")
        self.zoom_image_canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self.zoom_photo,
        )

    def _select_precise_point(self, event) -> None:
        if self.image_gray is None or self.state.zoom_center is None:
            return
        self.state.click_point = map_display_point_to_source(
            event.x,
            event.y,
            self.zoom_scale_x,
            self.zoom_scale_y,
            self.image_gray.shape,
            self.zoom_x0,
            self.zoom_y0,
        )
        self.state.result = None
        self._close_result_window()
        self.selected_point_label.configure(
            text=f"Selected reference point: {self.state.click_point}"
        )
        self._update_analyze_button()
        if self.analysis_running:
            status = "Reference point selected; current analysis still running"
        else:
            status = "Reference point selected"
        self.status_label.configure(text=status)
        self._render_full_image()
        self._render_zoom_image()

    def _update_analyze_button(self) -> None:
        enabled = (
            self.state.click_point is not None
            and not self.analysis_running
        )
        self.analyze_button.configure(
            state="normal" if enabled else "disabled"
        )

    def analyze(self) -> None:
        if (
            self.analysis_running
            or self.image_gray is None
            or self.state.click_point is None
        ):
            return
        click_started_ns = time.perf_counter_ns()
        run_id = uuid.uuid4().hex
        self.active_analysis_run_id = run_id
        output_dir = build_output_dir(
            self.image_name,
            self.image_bytes,
            self.state.click_point,
        )
        analysis_key = (
            self.state.image_identity,
            self.state.click_point,
        )
        image_gray = self.image_gray
        image_name = self.image_name
        click_x, click_y = self.state.click_point
        self.analysis_running = True
        self._update_analyze_button()
        self.status_label.configure(text="Analyzing…")
        if (
            self.result_timing_label is not None
            and self.result_timing_label.winfo_exists()
        ):
            self.result_timing_label.configure(text="Analyzing…")
        self.root.update_idletasks()
        worker = threading.Thread(
            target=self._run_analysis_worker,
            args=(
                analysis_key,
                image_gray,
                image_name,
                click_x,
                click_y,
                output_dir,
                run_id,
                click_started_ns,
            ),
            daemon=True,
        )
        worker.start()
        self.root.after(50, self._poll_analysis)

    def _run_analysis_worker(
        self,
        analysis_key,
        image_gray,
        image_name: str,
        click_x: int,
        click_y: int,
        output_dir: Path,
        run_id: str | None = None,
        click_started_ns: int | None = None,
    ) -> None:
        run_id = run_id or uuid.uuid4().hex
        click_started_ns = (
            click_started_ns
            if click_started_ns is not None
            else time.perf_counter_ns()
        )
        metadata = {
            "image_name": image_name,
            "reference_global": {"x": click_x, "y": click_y},
            "result_source": STAGE3_RESULT_SOURCE,
        }
        stage3_result = None
        bounds = None
        stage3_ms = None
        result_write_ms = None
        phase = "roi"
        try:
            bounds = calculate_interactive_roi_bounds(
                image_gray.shape,
                click_x,
                click_y,
                INTERACTIVE_CONFIG,
            )
            reference_global = {"x": click_x, "y": click_y}
            phase = "stage3"
            stage3_started_ns = time.perf_counter_ns()
            try:
                stage3_result = run_frozen_stage3(
                    image_gray,
                    reference_global,
                    bounds,
                )
            finally:
                stage3_ms = elapsed_ms(
                    stage3_started_ns,
                    time.perf_counter_ns(),
                )
            phase = "adapter"
            result = build_desktop_result(
                image_gray,
                image_name,
                reference_global,
                bounds,
                stage3_result,
                output_dir,
            )
            phase = "result_write"
            write_started_ns = time.perf_counter_ns()
            try:
                persist_formal_result(result)
            finally:
                result_write_ms = elapsed_ms(
                    write_started_ns,
                    time.perf_counter_ns(),
                )
            error = None
        except Exception as caught_error:
            result = None
            error = caught_error
        measurements = {
            "stage3_algorithm_ms": stage3_ms,
            "result_write_ms": result_write_ms,
        }
        if error is not None and phase in {"stage3", "adapter"}:
            measurements["stage3_error"] = (
                f"{type(error).__name__}: {error}"
            )
        if error is not None and phase == "result_write":
            measurements["result_write_error"] = (
                f"{type(error).__name__}: {error}"
            )
        self._record_performance(
            output_dir,
            run_id,
            metadata,
            measurements,
        )
        self.analysis_queue.put(
            AnalysisCompletion(
                analysis_key=analysis_key,
                result=result,
                error=error,
                run_id=run_id,
                click_started_ns=click_started_ns,
                output_dir=output_dir,
                performance_metadata=metadata,
                stage3_algorithm_ms=stage3_ms,
            )
        )
        if (
            cross_image_experiment_enabled()
            and bounds is not None
        ):
            experiment_worker = threading.Timer(
                CROSS_IMAGE_EXPERIMENT_START_DELAY_SECONDS,
                self._run_cross_image_experiment_worker,
                args=(
                    image_gray,
                    image_name,
                    click_x,
                    click_y,
                    bounds,
                    output_dir,
                    stage3_result,
                    run_id,
                ),
            )
            experiment_worker.daemon = True
            experiment_worker.start()
        if (
            self.debug_enabled
            and result is not None
            and stage3_result is not None
        ):
            diagnostic_worker = threading.Thread(
                target=self._run_legacy_diagnostic_worker,
                args=(
                    analysis_key[0],
                    image_gray,
                    image_name,
                    click_x,
                    click_y,
                    output_dir,
                    stage3_result,
                    run_id,
                    metadata,
                ),
                daemon=True,
            )
            diagnostic_worker.start()

    def _run_cross_image_experiment_worker(
        self,
        image_gray,
        image_name: str,
        click_x: int,
        click_y: int,
        bounds,
        output_dir: Path,
        stage3_result: dict | None,
        run_id: str,
    ) -> None:
        """Run the default-off v1.1 shadow outside the formal result path."""

        try:
            from tools.cross_image_correction_shadow_v1_1 import (
                run_cross_image_experiment_shadow,
            )

            run_cross_image_experiment_shadow(
                image_gray,
                image_name,
                {"x": click_x, "y": click_y},
                bounds,
                output_dir / "cross_image_correction_v1_1_shadow",
                run_id,
                stage3_result,
            )
        except Exception as experiment_error:
            print(
                "Cross-image correction v1.1 shadow failed: "
                f"{type(experiment_error).__name__}: "
                f"{experiment_error}",
                file=sys.stderr,
            )

    def _run_legacy_diagnostic_worker(
        self,
        image_identity: str,
        image_gray,
        image_name: str,
        click_x: int,
        click_y: int,
        output_dir: Path,
        stage3_result: dict,
        run_id: str,
        metadata: dict,
    ) -> None:
        """Run the former detector for diagnostics without touching the UI."""

        legacy_started_ns = time.perf_counter_ns()
        legacy_result = None
        legacy_error = None
        legacy_output_dir = output_dir / "legacy_diagnostic"
        try:
            lock = getattr(self, "pitch_reference_lock", None)
            if lock is None:
                pitch_reference_map = get_or_build_pitch_reference(
                    self.pitch_reference_cache,
                    image_identity,
                    image_gray,
                )
            else:
                with lock:
                    pitch_reference_map = get_or_build_pitch_reference(
                        self.pitch_reference_cache,
                        image_identity,
                        image_gray,
                    )
            legacy_result = run_interactive_case(
                image_gray,
                click_x,
                click_y,
                legacy_output_dir,
                CONFIG,
                INTERACTIVE_CONFIG,
                image_name=image_name,
                pitch_reference_map=pitch_reference_map,
            )
        except Exception as caught_error:
            legacy_error = caught_error
        legacy_ms = elapsed_ms(
            legacy_started_ns,
            time.perf_counter_ns(),
        )
        if legacy_result is not None:
            try:
                run_shadow_and_log(
                    image_gray,
                    image_name,
                    legacy_result,
                    legacy_output_dir,
                    log_config=ShadowLogConfig(
                        log_root=(
                            desktop_output_root()
                            / "diagnostics"
                            / "legacy_comparisons"
                        )
                    ),
                    shadow_runner=lambda *_args: stage3_result,
                )
            except Exception as comparison_error:
                print(
                    "Legacy diagnostic comparison logging failed: "
                    f"{type(comparison_error).__name__}: "
                    f"{comparison_error}",
                    file=sys.stderr,
                )
        measurements = {"legacy_algorithm_ms": legacy_ms}
        if legacy_error is not None:
            measurements["legacy_diagnostic_error"] = (
                f"{type(legacy_error).__name__}: {legacy_error}"
            )
            print(
                "Legacy diagnostic failed: "
                f"{measurements['legacy_diagnostic_error']}",
                file=sys.stderr,
            )
        self._record_performance(
            output_dir,
            run_id,
            metadata,
            measurements,
        )

    def _record_performance(
        self,
        output_dir: Path,
        run_id: str,
        metadata: dict,
        measurements: dict,
    ) -> None:
        """Keep instrumentation failures outside the detection contract."""

        if not getattr(self, "debug_enabled", False):
            return

        try:
            update_timing(
                desktop_performance_log_root(),
                output_dir,
                run_id,
                metadata,
                measurements,
            )
        except Exception as timing_error:
            print(
                "Performance timing write failed: "
                f"{type(timing_error).__name__}: {timing_error}",
                file=sys.stderr,
            )

    def _poll_analysis(self) -> None:
        try:
            completion = self.analysis_queue.get_nowait()
        except queue.Empty:
            if self.analysis_running:
                self.root.after(50, self._poll_analysis)
            return

        current_key = (
            self.state.image_identity,
            self.state.click_point,
        )
        same_run = (
            completion.run_id == self.active_analysis_run_id
        )
        if not completion_matches_active_request(
            completion,
            current_key,
            self.active_analysis_run_id,
        ):
            if same_run:
                self.analysis_running = False
                self.active_analysis_run_id = None
                self._update_analyze_button()
                self.status_label.configure(
                    text=(
                        "Previous analysis ignored after "
                        "selection changed"
                    )
                )
            elif self.analysis_running:
                self.root.after(50, self._poll_analysis)
            return

        self.analysis_running = False
        self.active_analysis_run_id = None
        self._update_analyze_button()
        if completion.error is not None:
            self.status_label.configure(text="Analysis failed")
            messagebox.showerror(
                "Analysis failed",
                str(completion.error),
                parent=self.root,
            )
            return

        self.state.result = completion.result
        self.status_label.configure(text="Analysis complete")
        ui_started_ns = time.perf_counter_ns()
        self._show_result(
            completion.result,
            completion.stage3_algorithm_ms,
        )
        self.root.update_idletasks()
        displayed_ns = time.perf_counter_ns()
        total_ms = elapsed_ms(
            completion.click_started_ns,
            displayed_ns,
        )
        ui_draw_ms = elapsed_ms(ui_started_ns, displayed_ns)
        if (
            self.result_timing_label is not None
            and self.result_timing_label.winfo_exists()
        ):
            interactive_result = completion.result.report[
                "interactive_result"
            ]
            self.result_timing_label.configure(
                text=result_performance_text(
                    interactive_result,
                    completion.stage3_algorithm_ms,
                    total_ms,
                )
            )
            self.root.update_idletasks()
        self._record_performance(
            completion.output_dir,
            completion.run_id,
            completion.performance_metadata,
            {
                "ui_draw_ms": ui_draw_ms,
                "click_to_display_ms": total_ms,
            },
        )

    def _show_selection_view(self) -> None:
        """Return to point selection inside the same application window."""

        self._hide_magnifier()
        if self.result_frame is not None and self.result_frame.winfo_exists():
            self.result_frame.destroy()
        self.result_frame = None
        self.result_photos = []
        self.result_timing_label = None
        if not self.controls.winfo_manager():
            self.controls.pack(fill="x", before=self.main_container)
        if not self.selection_frame.winfo_manager():
            self.selection_frame.pack(fill="both", expand=True)
        self.root.title("Interactive Stripe Analysis")

    def _close_result_window(self) -> None:
        """Compatibility wrapper for callers that change the current point."""

        self._show_selection_view()

    def _show_result(
        self,
        result,
        stage3_ms: float | None = None,
    ) -> None:
        """Replace the selection view with one compact final-result page."""

        self._hide_magnifier()
        if self.result_frame is not None and self.result_frame.winfo_exists():
            self.result_frame.destroy()
        self.controls.pack_forget()
        self.selection_frame.pack_forget()
        self.result_photos = []

        content = ttk.Frame(self.main_container, padding=(18, 14))
        self.result_frame = content
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(3, weight=1)
        self.root.title(f"Stripe Analysis Result — {self.image_name}")

        report = result.report
        interactive_result = report["interactive_result"]
        click = report["click"]
        success = interactive_result["success"]
        bilateral_success = interactive_result.get(
            "bilateral_success",
            bool(
                interactive_result.get("left") is not None
                and interactive_result.get("right") is not None
            ),
        )
        available_sides = [
            side
            for side in ("left", "right")
            if interactive_result.get(side) is not None
        ]
        stage3_formal = (
            report.get("result_source") == STAGE3_RESULT_SOURCE
        )

        header = ttk.Frame(content)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text=(
                "Detection successful"
                if bilateral_success
                else (
                    f"Partial detection — {available_sides[0].capitalize()} available"
                    if success
                    else (
                        "Unable to determine reliably"
                        if stage3_formal
                        else "Detection failed"
                    )
                )
            ),
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            header,
            text="Select another point",
            command=self._show_selection_view,
            style="Secondary.TButton",
        ).grid(row=0, column=1, sticky="e")
        ttk.Label(
            header,
            text=self.image_name,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self.result_timing_label = ttk.Label(
            header,
            text=result_performance_text(
                interactive_result,
                stage3_ms,
                None,
            ),
        )
        self.result_timing_label.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(3, 0),
        )

        pitch_guard = interactive_result.get("pitch_guard", {})
        values = result_metric_values(click, interactive_result)
        metrics = ttk.Frame(content)
        metrics.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        for column, (title, value, details) in enumerate(values):
            metrics.columnconfigure(column, weight=1)
            metric = ttk.LabelFrame(metrics, text=title, padding=(10, 6))
            metric.grid(
                row=0,
                column=column,
                sticky="nsew",
                padx=(0, 8 if column < len(values) - 1 else 0),
            )
            ttk.Label(
                metric,
                text=value,
                style="MetricValue.TLabel",
            ).pack(anchor="w")
            if details:
                ttk.Label(
                    metric,
                    text=details,
                    wraplength=220,
                    justify="left",
                ).pack(anchor="w", pady=(2, 0))

        warning_flags = [
            flag
            for flag in interactive_result.get("warning_flags", [])
            if not flag.startswith("whole_image_pitch_")
        ]
        notes = []
        if success and not bilateral_success:
            unavailable_side = (
                "right" if available_sides == ["left"] else "left"
            )
            side_details = interactive_result.get("side_status", {}).get(
                unavailable_side, {}
            )
            side_reason = side_details.get("reason")
            notes.append(
                f"{unavailable_side.capitalize()} side unavailable."
                + (
                    " Diagnostic reason: " + side_reason.replace("_", " ")
                    if side_reason
                    else ""
                )
            )
        if not success:
            failure_reasons = interactive_result["failure_reasons"]
            if stage3_formal:
                internal_reason = interactive_result.get(
                    "unavailable_reason"
                )
                notes.append(
                    "Unable to determine reliably."
                    + (
                        " Diagnostic reason: "
                        + internal_reason.replace("_", " ")
                        if internal_reason
                        else ""
                    )
                )
            else:
                notes.append(
                    "Detection failed: "
                    + (
                        " | ".join(failure_reasons)
                        if failure_reasons
                        else "immediate neighbors were not verified"
                    )
                )
        if warning_flags:
            notes.append("Detection note: " + " | ".join(warning_flags))
        pitch_note = (
            pitch_safety_note(pitch_guard)
            if success and not stage3_formal
            else None
        )
        if pitch_note is not None:
            notes.append(pitch_note)
        if notes:
            warning = tk.Label(
                content,
                text="\n".join(notes),
                background="#fff3cd",
                foreground="#7a5700",
                anchor="w",
                justify="left",
                padx=10,
                pady=6,
            )
            warning.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        body = ttk.Frame(content)
        body.grid(row=3, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        original_panel = ttk.Frame(body)
        original_panel.grid(row=0, column=0, sticky="nw", padx=(0, 16))
        ttk.Label(
            original_panel,
            text="Original image with ROI",
            style="Section.TLabel",
        ).pack(anchor="w", pady=(0, 6))
        ttk.Label(
            original_panel,
            text=(
                "Green: ROI · Red: reference · Blue/Yellow: centerlines"
                if success
                else "Green: ROI · Red: reference · No centerlines reported"
            ),
        ).pack(anchor="w", pady=(0, 7))
        original_image = (
            result.debug_images.get("original_interactive_result.png")
            if success
            else create_failure_result_overlay(self.image_gray, result)
        )
        if original_image is not None:
            original_photo, _scale_x, _scale_y = array_to_photo(
                debug_image_to_rgb(original_image),
                self.root,
                INTERACTIVE_CONFIG.desktop_result_full_max_width_px,
                INTERACTIVE_CONFIG.desktop_result_full_max_height_px,
            )
            self.result_photos.append(original_photo)
            ttk.Label(
                original_panel,
                image=original_photo,
            ).pack(anchor="w")

        roi_panel = ttk.Frame(body)
        roi_panel.grid(row=0, column=1, sticky="nw")
        ttk.Label(
            roi_panel,
            text="Enlarged ROI",
            style="Section.TLabel",
        ).pack(anchor="w", pady=(0, 6))
        roi_photo, _scale_x, _scale_y = array_to_photo(
            debug_image_to_rgb(
                create_result_roi_crop(result, overlay=original_image)
            ),
            self.root,
            INTERACTIVE_CONFIG.desktop_result_roi_max_width_px,
            INTERACTIVE_CONFIG.desktop_result_roi_max_height_px,
            nearest=True,
        )
        self.result_photos.append(roi_photo)
        ttk.Label(roi_panel, image=roi_photo).pack(anchor="w", pady=(0, 10))
        if success:
            self._add_track_table(roi_panel, interactive_result)
        roi = report["roi"]
        ttk.Label(
            roi_panel,
            text=f"ROI: {roi['width_px']} × {roi['height_px']} px",
        ).pack(anchor="w", pady=(6, 0))

    def _add_track_table(self, parent, interactive_result: dict) -> None:
        if (
            interactive_result.get("result_source")
            == STAGE3_RESULT_SOURCE
        ):
            columns = (
                "side",
                "center_x_global",
                "distance_to_click_px",
                "basin_width_px",
                "evidence_status",
            )
            headings = (
                "Side",
                "Center X",
                "Distance",
                "Basin width",
                "Evidence",
            )
            rows = basin_table_rows(interactive_result)
        else:
            columns = (
                "side",
                "center_x_global",
                "distance_to_click_px",
                "median_width_px",
                "valid_row_ratio",
            )
            headings = (
                "Side",
                "Center X",
                "Distance",
                "Width",
                "Row support",
            )
            rows = track_table_rows(interactive_result)
        table = ttk.Treeview(
            parent,
            columns=columns,
            show="headings",
            height=2,
        )
        widths = (65, 100, 105, 90, 110)
        for column, heading, width in zip(columns, headings, widths):
            table.heading(column, text=heading)
            table.column(column, width=width, anchor="center")
        for row in rows:
            table.insert("", "end", values=[row[column] for column in columns])
        table.pack(fill="x")


def main(argv=None) -> None:
    """Launch the local desktop application."""

    arguments = parse_desktop_arguments(argv)
    debug_enabled = desktop_debug_enabled(arguments.debug)
    root = tk.Tk()
    patchlevel = str(root.tk.call("info", "patchlevel"))
    if not reliable_tk_runtime(patchlevel):
        message = (
            f"Tk {patchlevel} can randomly ignore clicks on modern macOS.\n\n"
            "Start the desktop app with the reliable environment:\n"
            ".venv-desktop/bin/python tools/desktop_app.py"
        )
        print(message, file=sys.stderr)
        messagebox.showerror(
            "Unsupported Tk version",
            message,
            parent=root,
        )
        root.destroy()
        return
    application = StripeDesktopApp(
        root,
        debug_enabled=debug_enabled,
    )
    if arguments.startup_smoke_test:
        root.update()
        print(
            "STARTUP_SMOKE_OK "
            f"image_loaded={application.image_gray is not None} "
            f"debug={application.debug_enabled}"
        )
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
