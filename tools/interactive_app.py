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


def draw_click_preview(image_gray, click_point: tuple[int, int] | None):
    """Draw the stored click and ROI on the browser display copy."""

    preview = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
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
    st.write(f"clicked/excluded track: `{interactive_result['clicked_track_id']}`")
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

    st.subheader("Rotation shadow (not applied to detection)")
    angle_column, score_column, separation_column = st.columns(3)
    angle_column.metric("Estimated correction", f"{rotation['best_angle_deg']}°")
    score_column.metric("Best score", rotation["best_score"])
    separation_column.metric("Peak separation", rotation["peak_separation"])
    rotation_chart, rotation_preview = st.columns(2)
    with rotation_chart:
        st.image(
            bgr_to_rgb(result.debug_images["rotation_score.png"]),
            width="stretch",
        )
    with rotation_preview:
        st.image(
            result.debug_images["rotation_preview.png"],
            caption="Best-angle preview only",
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
    try:
        image_gray = decode_grayscale_image(image_bytes)
    except ValueError as error:
        st.error(str(error))
        return

    click_state_key = f"click_{image_hash}"
    click_event_key = f"click_event_{image_hash}"
    result_state_key = f"result_{image_hash}"
    click_point = st.session_state.get(click_state_key)
    preview_original = draw_click_preview(image_gray, click_point)
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

    st.subheader("Click a reference point")
    click_value = streamlit_image_coordinates(
        display_image,
        width=display_image.shape[1],
        key=f"image_coordinates_{image_hash}",
        cursor="crosshair",
    )
    if click_value is not None:
        event_token = (
            click_value.get("x"),
            click_value.get("y"),
            click_value.get("unix_time"),
        )
        if event_token != st.session_state.get(click_event_key):
            x_global = min(
                image_gray.shape[1] - 1,
                max(0, round(click_value["x"] * scale_x_to_source)),
            )
            y_global = min(
                image_gray.shape[0] - 1,
                max(0, round(click_value["y"] * scale_y_to_source)),
            )
            st.session_state[click_state_key] = (x_global, y_global)
            st.session_state[click_event_key] = event_token
            st.session_state.pop(result_state_key, None)
            st.rerun()

    click_point = st.session_state.get(click_state_key)
    if click_point is None:
        st.info("Click the image to choose a reference point.")
    else:
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
                st.session_state[result_state_key] = run_interactive_case(
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

    result = st.session_state.get(result_state_key)
    if result is not None:
        show_analysis_result(result)


if __name__ == "__main__":
    main()
