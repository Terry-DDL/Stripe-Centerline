"""Validate the existing stripe detector at several reference x positions."""

import csv
from pathlib import Path

from config import CONFIG, ProcessingConfig
from image_processing import load_grayscale_image, validate_config
from pipeline import PipelineResult, run_reference_case


SUMMARY_FIELDS = (
    "image_name",
    "ref_x_global",
    "ref_x_roi",
    "success",
    "left_center_x_global",
    "right_center_x_global",
    "left_distance_px",
    "right_distance_px",
    "left_valid_row_ratio",
    "right_valid_row_ratio",
    "left_retention_ratio",
    "right_retention_ratio",
    "notes",
    "warning_flags",
)


def _selected_track_values(track, result: PipelineResult) -> dict:
    """Return CSV values for one selected track, or blanks when absent."""

    if track is None:
        return {
            "center_x_global": "",
            "distance_px": "",
            "valid_row_ratio": "",
            "retention_ratio": "",
        }

    analysis = result.stripe_analysis
    center_x_global = result.bounds_global.x0_global + track.center_x_roi
    return {
        "center_x_global": round(center_x_global, 3),
        "distance_px": round(
            abs(track.center_x_roi - analysis.x_ref_roi), 3
        ),
        "valid_row_ratio": round(track.valid_row_ratio, 6),
        "retention_ratio": round(track.retention_ratio, 6),
    }


def _summary_row(
    image_path: Path,
    image_width: int,
    result: PipelineResult,
    config: ProcessingConfig,
) -> dict:
    """Build one summary row with stable failure and ROI warning flags."""

    analysis = result.stripe_analysis
    left = _selected_track_values(analysis.left_track, result)
    right = _selected_track_values(analysis.right_track, result)
    half_width_roi = round(image_width * config.roi_half_width_ratio)

    warning_flags = []
    notes = []
    if analysis.left_track is None:
        warning_flags.append("no_valid_left_stripe")
    if analysis.right_track is None:
        warning_flags.append("no_valid_right_stripe")
    if result.x_ref_global - half_width_roi < 0:
        warning_flags.append("roi_clipped_left")
    if result.x_ref_global + half_width_roi > image_width:
        warning_flags.append("roi_clipped_right")
    if not analysis.success:
        notes.append("Detection incomplete; inspect per-ref debug output.")
    if any(flag.startswith("roi_clipped_") for flag in warning_flags):
        notes.append("ROI was clipped by the image boundary.")

    return {
        "image_name": image_path.name,
        "ref_x_global": result.x_ref_global,
        "ref_x_roi": analysis.x_ref_roi,
        "success": str(analysis.success).lower(),
        "left_center_x_global": left["center_x_global"],
        "right_center_x_global": right["center_x_global"],
        "left_distance_px": left["distance_px"],
        "right_distance_px": right["distance_px"],
        "left_valid_row_ratio": left["valid_row_ratio"],
        "right_valid_row_ratio": right["valid_row_ratio"],
        "left_retention_ratio": left["retention_ratio"],
        "right_retention_ratio": right["retention_ratio"],
        "notes": " ".join(notes),
        "warning_flags": "|".join(warning_flags),
    }


def _validate_sweep_refs(ref_x_globals: tuple[int, ...]) -> None:
    """Reject an empty or duplicate reference list before processing."""

    if not ref_x_globals:
        raise ValueError("sweep_ref_x_globals must not be empty")
    if any(not isinstance(ref_x_global, int) for ref_x_global in ref_x_globals):
        raise ValueError("every sweep reference must be an integer")
    if len(set(ref_x_globals)) != len(ref_x_globals):
        raise ValueError("sweep_ref_x_globals must not contain duplicates")


def _save_summary(summary_path: Path, rows: list[dict]) -> None:
    """Save one image's sweep results as a readable CSV file."""

    try:
        with summary_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as error:
        raise OSError(f"Could not save sweep summary: {summary_path}") from error


def main() -> int:
    """Run every configured image/reference case and write per-image CSVs."""

    try:
        validate_config(CONFIG)
        _validate_sweep_refs(CONFIG.sweep_ref_x_globals)

        for image_path in CONFIG.sweep_image_paths:
            image_gray = load_grayscale_image(image_path)
            image_width = image_gray.shape[1]
            image_output_dir = (
                CONFIG.output_root_dir / image_path.stem.replace(" ", "_")
            )
            rows = []

            for x_ref_global in CONFIG.sweep_ref_x_globals:
                case_output_dir = image_output_dir / f"ref_{x_ref_global}"
                result = run_reference_case(
                    image_gray,
                    x_ref_global,
                    case_output_dir,
                    CONFIG,
                )
                rows.append(
                    _summary_row(image_path, image_width, result, CONFIG)
                )
                print(
                    f"{image_path.name}: ref_x_global={x_ref_global}, "
                    f"success={result.stripe_analysis.success}"
                )

            _save_summary(image_output_dir / "sweep_summary.csv", rows)
            print(f"saved summary: {image_output_dir / 'sweep_summary.csv'}")

        return 0
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
