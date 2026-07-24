"""Streamlit UI for interactive single-image stripe analysis."""

import hashlib
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import streamlit as st

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ModuleNotFoundError:
    streamlit_image_coordinates = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import CONFIG, INTERACTIVE_CONFIG  # noqa: E402
from interactive_pipeline import run_interactive_case  # noqa: E402


SUPPORTED_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg"}
ACTIVE_IMAGE_STATE_KEY = "active_image_identity"
ZOOM_CENTER_STATE_KEY = "selection_zoom_center"
CLICK_POINT_STATE_KEY = "selection_click_point"
RESULT_STATE_KEY = "selection_result"
FULL_IMAGE_REVISION_KEY = "full_image_widget_revision"
ZOOM_IMAGE_REVISION_KEY = "zoom_image_widget_revision"


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


def decode_grayscale_image(image_bytes: bytes):
    """Decode uploaded or project image bytes as one grayscale array."""

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


def next_widget_revision(state, key: str) -> int:
    """Advance one component revision so an old click cannot be reused."""

    revision = int(state.get(key, 0)) + 1
    state[key] = revision
    return revision


def prepare_image_state(state, image_identity: str) -> bool:
    """Clear interaction state when the selected image really changes."""

    state.setdefault(FULL_IMAGE_REVISION_KEY, 0)
    state.setdefault(ZOOM_IMAGE_REVISION_KEY, 0)
    if state.get(ACTIVE_IMAGE_STATE_KEY) == image_identity:
        return False
    state[ACTIVE_IMAGE_STATE_KEY] = image_identity
    state.pop(ZOOM_CENTER_STATE_KEY, None)
    state.pop(CLICK_POINT_STATE_KEY, None)
    state.pop(RESULT_STATE_KEY, None)
    next_widget_revision(state, FULL_IMAGE_REVISION_KEY)
    next_widget_revision(state, ZOOM_IMAGE_REVISION_KEY)
    return True


def display_image_and_scale(image_gray):
    """Resize the browser copy and return exact display-to-source scales."""

    height, width = image_gray.shape[:2]
    scale = min(1.0, INTERACTIVE_CONFIG.display_max_width_px / width)
    display_width = max(1, round(width * scale))
    display_height = max(1, round(height * scale))
    if scale < 1.0:
        display_gray = cv2.resize(
            image_gray,
            (display_width, display_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        display_gray = image_gray.copy()
    return (
        cv2.cvtColor(display_gray, cv2.COLOR_GRAY2RGB),
        width / display_width,
        height / display_height,
    )


def zoom_bounds(
    image_shape: tuple[int, ...],
    center_point: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return a clipped source-image rectangle for precise point selection."""

    height, width = image_shape[:2]
    center_x, center_y = center_point
    return (
        max(0, center_x - INTERACTIVE_CONFIG.selection_zoom_half_width_px),
        max(0, center_y - INTERACTIVE_CONFIG.selection_zoom_half_height_px),
        min(width, center_x + INTERACTIVE_CONFIG.selection_zoom_half_width_px),
        min(height, center_y + INTERACTIVE_CONFIG.selection_zoom_half_height_px),
    )


def map_display_point_to_source(
    x_display: float,
    y_display: float,
    scale_x_to_source: float,
    scale_y_to_source: float,
    image_shape: tuple[int, ...],
    x0_source: int = 0,
    y0_source: int = 0,
) -> tuple[int, int]:
    """Map one browser-image click back to a clipped source coordinate."""

    height, width = image_shape[:2]
    x_source = x0_source + round(x_display * scale_x_to_source)
    y_source = y0_source + round(y_display * scale_y_to_source)
    return (
        min(width - 1, max(0, x_source)),
        min(height - 1, max(0, y_source)),
    )


def create_zoom_display(
    image_gray,
    center_point: tuple[int, int],
    click_point: tuple[int, int] | None,
):
    """Create an enlarged local image and its display-to-source mapping."""

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
    zoom_center: tuple[int, int] | None = None,
):
    """Draw the zoom area, stored click, and ROI on the display copy."""

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


def load_selected_image() -> tuple[str, bytes] | None:
    """Render source controls and return the selected filename and bytes."""

    source = st.sidebar.radio("Image source", ("Project images", "Upload"))
    if source == "Project images":
        image_paths = project_image_paths()
        if not image_paths:
            st.warning("No supported files found under images/.")
            return None
        selected_name = st.sidebar.selectbox(
            "Image",
            [path.name for path in image_paths],
        )
        selected_path = PROJECT_ROOT / "images" / selected_name
        try:
            return selected_name, selected_path.read_bytes()
        except OSError as error:
            st.error(f"Could not read {selected_name}: {error}")
            return None

    uploaded = st.sidebar.file_uploader(
        "Upload BMP, PNG, or JPG",
        type=["bmp", "png", "jpg", "jpeg"],
    )
    if uploaded is None:
        st.info("Upload an image to begin.")
        return None
    return uploaded.name, uploaded.getvalue()


def show_track_table(result_report: dict) -> None:
    """Display compact left/right measurements from the interactive report."""

    rows = []
    for side in ("left", "right"):
        side_result = result_report.get(side)
        if side_result is None:
            rows.append(
                {
                    "side": side,
                    "center_x_global": "",
                    "distance_to_click_px": "",
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
                    "valid_row_ratio": side_result["valid_row_ratio"],
                    "retention_ratio": side_result["retention_ratio"],
                }
            )
    st.dataframe(rows, hide_index=True, width="stretch")


def bgr_to_rgb(image):
    """Convert saved OpenCV color debug data for Streamlit display."""

    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def show_analysis_result(result) -> None:
    """Render measurements and debug images from one completed analysis."""

    report = result.report
    interactive_result = report["interactive_result"]
    click = report["click"]
    rotation = report["rotation_shadow"]
    status_column, click_column, spacing_column = st.columns(3)
    status_column.metric("Success", str(interactive_result["success"]))
    click_column.metric(
        "Reference point",
        f"({click['x_global']}, {click['y_global']})",
    )
    spacing_column.metric(
        "Stripe spacing (px)",
        interactive_result["stripe_spacing_px"]
        if interactive_result["stripe_spacing_px"] is not None
        else "unavailable",
    )
    st.write(f"click classification: `{click['classification']}`")
    st.write(f"clicked track: `{interactive_result['clicked_track_id']}`")
    warning_flags = interactive_result["warning_flags"]
    if warning_flags:
        st.warning("warning_flags: " + " | ".join(warning_flags))
    show_track_table(interactive_result)

    st.subheader("Final result on original image")
    st.image(
        bgr_to_rgb(result.debug_images["original_interactive_result.png"]),
        width="stretch",
    )
    st.subheader("ROI result")
    st.image(
        bgr_to_rgb(result.debug_images["interactive_result.png"]),
        width="stretch",
    )
    candidates_column, votes_column = st.columns(2)
    with candidates_column:
        st.subheader("Black run candidates")
        st.image(
            bgr_to_rgb(result.debug_images["black_run_candidates.png"]),
            width="stretch",
        )
    with votes_column:
        st.subheader("Stripe center votes")
        st.image(
            bgr_to_rgb(result.debug_images["stripe_center_votes.png"]),
            width="stretch",
        )

    st.subheader("Guarded rotation detection")
    applied_column, angle_column, space_column = st.columns(3)
    applied_column.metric(
        "Applied to detection",
        str(rotation["applied_to_detection"]),
    )
    angle_column.metric(
        "Estimated / applied angle",
        f"{rotation['best_angle_deg']}° / {rotation['applied_angle_deg']}°",
    )
    space_column.metric("Detection space", rotation["detection_space"])
    st.write(
        "rotation score gain: "
        f"`{rotation['relative_score_gain']}`; "
        f"peak separation: `{rotation['peak_separation']}`"
    )
    if rotation["guard_reasons"]:
        st.caption("Not applied: " + " | ".join(rotation["guard_reasons"]))
    if rotation["fallback_reason"] is not None:
        st.warning(rotation["fallback_reason"])
    rotation_chart, rotation_preview = st.columns(2)
    with rotation_chart:
        st.image(
            bgr_to_rgb(result.debug_images["rotation_score.png"]),
            width="stretch",
        )
    with rotation_preview:
        st.image(
            result.debug_images["rotation_preview.png"],
            caption="Best-angle binary preview",
            width="stretch",
        )
    st.caption(f"Debug output: {result.output_dir}")


def main() -> None:
    """Render the interactive single-image analysis application."""

    st.set_page_config(page_title="Interactive Stripe Analysis", layout="wide")
    st.title("Interactive Stripe Analysis")
    st.caption(
        "Select an image, click one reference point, then analyze the local ROI."
    )
    if streamlit_image_coordinates is None:
        st.error(
            "Missing dependency: install requirements.txt before running this app."
        )
        return

    selected = load_selected_image()
    if selected is None:
        return
    image_name, image_bytes = selected
    image_hash = hashlib.sha256(image_bytes).hexdigest()[:8]
    image_identity = f"{image_name}:{image_hash}"
    prepare_image_state(st.session_state, image_identity)
    try:
        image_gray = decode_grayscale_image(image_bytes)
    except ValueError as error:
        st.error(str(error))
        return

    image_height, image_width = image_gray.shape[:2]
    st.caption(f"Loaded: {image_name} ({image_width} × {image_height} px)")

    click_point = st.session_state.get(CLICK_POINT_STATE_KEY)
    zoom_center = st.session_state.get(ZOOM_CENTER_STATE_KEY)
    preview_original = draw_click_preview(
        image_gray,
        click_point,
        zoom_center,
    )
    preview_gray = cv2.cvtColor(preview_original, cv2.COLOR_RGB2GRAY)
    display_image, scale_x_to_source, scale_y_to_source = (
        display_image_and_scale(preview_gray)
    )
    if click_point is not None:
        display_image = cv2.resize(
            preview_original,
            (display_image.shape[1], display_image.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    st.subheader("1. Choose a local area")
    st.caption("Click the full image once to open a magnified local view.")
    click_value = streamlit_image_coordinates(
        display_image,
        width=display_image.shape[1],
        key=(
            f"image_coordinates_{image_hash}_"
            f"{st.session_state[FULL_IMAGE_REVISION_KEY]}"
        ),
        png_compression_level=6,
        cursor="crosshair",
    )
    if click_value is not None:
        coarse_point = map_display_point_to_source(
            click_value["x"],
            click_value["y"],
            scale_x_to_source,
            scale_y_to_source,
            image_gray.shape,
        )
        st.session_state[ZOOM_CENTER_STATE_KEY] = coarse_point
        st.session_state.pop(CLICK_POINT_STATE_KEY, None)
        st.session_state.pop(RESULT_STATE_KEY, None)
        next_widget_revision(st.session_state, FULL_IMAGE_REVISION_KEY)
        next_widget_revision(st.session_state, ZOOM_IMAGE_REVISION_KEY)
        st.rerun()

    zoom_center = st.session_state.get(ZOOM_CENTER_STATE_KEY)
    click_point = st.session_state.get(CLICK_POINT_STATE_KEY)
    if zoom_center is None:
        st.info("Choose an area in the full image first.")
    else:
        st.subheader("2. Click the precise reference point")
        st.caption(
            "The local image is enlarged 3×. Click it to set the final point."
        )
        (
            zoom_display,
            zoom_x0,
            zoom_y0,
            zoom_scale_x,
            zoom_scale_y,
        ) = create_zoom_display(image_gray, zoom_center, click_point)
        zoom_value = streamlit_image_coordinates(
            zoom_display,
            width=zoom_display.shape[1],
            key=(
                f"zoom_coordinates_{image_hash}_"
                f"{st.session_state[ZOOM_IMAGE_REVISION_KEY]}"
            ),
            png_compression_level=6,
            cursor="crosshair",
        )
        if zoom_value is not None:
            precise_point = map_display_point_to_source(
                zoom_value["x"],
                zoom_value["y"],
                zoom_scale_x,
                zoom_scale_y,
                image_gray.shape,
                zoom_x0,
                zoom_y0,
            )
            st.session_state[CLICK_POINT_STATE_KEY] = precise_point
            st.session_state.pop(RESULT_STATE_KEY, None)
            next_widget_revision(st.session_state, ZOOM_IMAGE_REVISION_KEY)
            st.rerun()

    if click_point is not None:
        st.write(f"Selected reference point: `{click_point}`")

    if st.button(
        "Analyze selected point",
        type="primary",
        disabled=click_point is None,
    ):
        click_x, click_y = click_point
        output_dir = (
            INTERACTIVE_CONFIG.output_root_dir
            / f"{safe_stem(image_name)}_{image_hash}"
            / f"point_{click_x}_{click_y}"
        )
        try:
            with st.spinner("Analyzing ROI..."):
                st.session_state[RESULT_STATE_KEY] = run_interactive_case(
                    image_gray,
                    click_x,
                    click_y,
                    output_dir,
                    CONFIG,
                    INTERACTIVE_CONFIG,
                    image_name=image_name,
                )
        except (OSError, ValueError, cv2.error) as error:
            st.error(f"Analysis failed: {error}")

    result = st.session_state.get(RESULT_STATE_KEY)
    if result is not None:
        show_analysis_result(result)


if __name__ == "__main__":
    main()
