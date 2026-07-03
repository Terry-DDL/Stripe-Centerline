"""Read-only Streamlit viewer for existing reference-sweep outputs."""

import csv
import json
from pathlib import Path
import re

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
REF_DIRECTORY_PATTERN = re.compile(r"^ref_(\d+)$")


def find_ref_directories(sample_dir: Path) -> list[tuple[int, Path]]:
    """Return numeric ref directories in ascending reference order."""

    ref_directories = []
    for path in sample_dir.iterdir():
        if not path.is_dir():
            continue
        match = REF_DIRECTORY_PATTERN.fullmatch(path.name)
        if match is not None:
            ref_directories.append((int(match.group(1)), path))
    return sorted(ref_directories, key=lambda item: item[0])


def find_sample_directories() -> list[Path]:
    """Return output directories that contain a summary or ref case."""

    if not OUTPUT_ROOT.is_dir():
        return []

    sample_directories = []
    for path in OUTPUT_ROOT.iterdir():
        if not path.is_dir():
            continue
        if (path / "sweep_summary.csv").is_file() or find_ref_directories(path):
            sample_directories.append(path)
    return sorted(sample_directories, key=lambda path: path.name)


def load_json_report(report_path: Path) -> tuple[dict | None, str | None]:
    """Read one JSON report without modifying it."""

    if not report_path.is_file():
        return None, f"Missing report: {report_path.name}"
    try:
        with report_path.open("r", encoding="utf-8") as report_file:
            report = json.load(report_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"Could not read {report_path.name}: {error}"
    if not isinstance(report, dict):
        return None, f"Invalid report structure: {report_path.name}"
    return report, None


def load_summary_row(
    summary_path: Path,
    ref_x_global: int,
) -> tuple[dict | None, str | None]:
    """Read the CSV row matching one reference x."""

    if not summary_path.is_file():
        return None, f"Missing summary: {summary_path.name}"
    try:
        with summary_path.open("r", encoding="utf-8", newline="") as summary_file:
            for row in csv.DictReader(summary_file):
                if row.get("ref_x_global") == str(ref_x_global):
                    return row, None
    except (OSError, UnicodeError, csv.Error) as error:
        return None, f"Could not read {summary_path.name}: {error}"
    return None, f"No summary row for ref_{ref_x_global}"


def selected_retention(report: dict, side_result: dict | None):
    """Find retention for a selected result using its candidate track id."""

    if side_result is None:
        return ""
    selected_track_id = side_result.get("track_id")
    for candidate in report.get("candidates", []):
        if candidate.get("track_id") == selected_track_id:
            return candidate.get("retention_ratio", "")
    return "unavailable"


def side_table_rows(report: dict) -> list[dict]:
    """Build compact left/right rows from one JSON report."""

    result = report.get("result", {})
    rows = []
    for side in ("left", "right"):
        side_result = result.get(side)
        if side_result is None:
            rows.append(
                {
                    "side": side,
                    "center_x_global": "",
                    "distance_px": "",
                    "valid_row_ratio": "",
                    "retention_ratio": "",
                }
            )
            continue
        rows.append(
            {
                "side": side,
                "center_x_global": side_result.get("center_x_global", ""),
                "distance_px": side_result.get("distance_px", ""),
                "valid_row_ratio": side_result.get("valid_row_ratio", ""),
                "retention_ratio": selected_retention(report, side_result),
            }
        )
    return rows


def show_image(image_path: Path, caption: str) -> None:
    """Display one existing image or a clear missing-file warning."""

    if image_path.is_file():
        st.image(image_path, caption=caption, width="stretch")
    else:
        st.warning(f"Missing image: {image_path.name}")


def main() -> None:
    """Render the read-only result viewer."""

    st.set_page_config(page_title="Stripe Sweep Results", layout="wide")
    st.title("Stripe Sweep Results")
    st.caption("Read-only viewer for existing files under outputs/.")

    sample_directories = find_sample_directories()
    if not sample_directories:
        st.info(f"No sweep outputs found under {OUTPUT_ROOT}")
        st.stop()

    sample_names = [path.name for path in sample_directories]
    selected_sample_name = st.sidebar.selectbox("Sample", sample_names)
    sample_dir = OUTPUT_ROOT / selected_sample_name
    ref_directories = find_ref_directories(sample_dir)
    if not ref_directories:
        st.warning(f"No ref_<x> directories found in {selected_sample_name}")
        st.stop()

    ref_values = [ref_x_global for ref_x_global, _path in ref_directories]
    selected_ref = st.sidebar.selectbox(
        "Reference",
        ref_values,
        format_func=lambda value: f"ref_{value}",
    )
    case_dir = sample_dir / f"ref_{selected_ref}"

    report, report_error = load_json_report(case_dir / "stripe_results.json")
    summary_row, summary_error = load_summary_row(
        sample_dir / "sweep_summary.csv",
        selected_ref,
    )
    if report_error is not None:
        st.error(report_error)
    if summary_error is not None:
        st.warning(summary_error)

    if report is not None:
        reference = report.get("reference", {})
        result = report.get("result", {})
        status_column, global_column, roi_column = st.columns(3)
        status_column.metric("Success", str(result.get("success", "unavailable")))
        global_column.metric(
            "ref_x_global", reference.get("x_ref_global", "unavailable")
        )
        roi_column.metric("ref_x_roi", reference.get("x_ref_roi", "unavailable"))

        if summary_row is None:
            warning_flags = "unavailable"
            notes = "unavailable"
        else:
            warning_flags = summary_row.get("warning_flags", "") or "none"
            notes = summary_row.get("notes", "") or "none"
        if warning_flags not in ("none", "unavailable"):
            st.warning(f"warning_flags: {warning_flags}")
        else:
            st.write(f"warning_flags: {warning_flags}")
        st.write(f"notes: {notes}")
        st.dataframe(side_table_rows(report), hide_index=True, width="stretch")

    st.subheader("Adjacent stripe result")
    show_image(
        case_dir / "adjacent_stripes_result.png",
        "adjacent_stripes_result.png",
    )
    candidates_column, votes_column = st.columns(2)
    with candidates_column:
        st.subheader("Black run candidates")
        show_image(
            case_dir / "black_run_candidates.png",
            "black_run_candidates.png",
        )
    with votes_column:
        st.subheader("Stripe center votes")
        show_image(
            case_dir / "stripe_center_votes.png",
            "stripe_center_votes.png",
        )


if __name__ == "__main__":
    main()
