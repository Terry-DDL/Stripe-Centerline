"""Build a 40-case manual review package for Stripe 9 and Stripe 10."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import CONFIG, INTERACTIVE_CONFIG  # noqa: E402
from ground_truth_evaluation import (  # noqa: E402
    GROUND_TRUTH_SCHEMA_VERSION,
    validate_ground_truth_document,
)
from image_processing import (  # noqa: E402
    apply_vertical_close_roi,
    create_black_mask_roi,
    crop_roi_global,
    gaussian_blur_roi,
)
from interactive_analysis import select_interactive_tracks  # noqa: E402
from interactive_pipeline import (  # noqa: E402
    _neighbor_consistency_check,
    calculate_interactive_roi_bounds,
)
from pitch_reference import (  # noqa: E402
    build_pitch_reference_map,
    evaluate_pitch_guard,
)
from stripe_analysis import analyze_adjacent_stripes  # noqa: E402


ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 5
CASES_PER_CATEGORY = 5
CATEGORIES = (
    "agree_normal",
    "method_disagreement",
    "suspicious_or_defect",
    "negative_candidate",
)
SCREENSHOT_POINTS = {
    9: ((1357, 706), (1377, 754)),
    10: (
        (1759, 439),
        (1250, 217),
        (1916, 897),
        (714, 512),
        (1557, 324),
    ),
}
IMAGE_PATHS = {
    9: PROJECT_ROOT / "images" / "Stripe_09_e0_t200229235_v3p02565_do.bmp",
    10: PROJECT_ROOT / "images" / "Stripe_10_e0_t221236602_v8p56736_retry.bmp",
}


@dataclass(frozen=True)
class MethodCandidate:
    method: str
    selection: object
    analysis: object
    local_check: dict
    pitch_guard: dict
    black_mask: np.ndarray

    @property
    def successful(self) -> bool:
        return bool(self.selection.success)


@dataclass(frozen=True)
class ReviewCandidate:
    stripe_number: int
    image_path: Path
    click_x: int
    click_y: int
    category: str
    otsu: MethodCandidate
    adaptive: MethodCandidate
    screenshot_point: bool


def _threshold_images(image_gray_roi) -> dict[str, np.ndarray]:
    blurred = gaussian_blur_roi(image_gray_roi, CONFIG)
    _threshold, otsu = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )
    return {"otsu": otsu, "adaptive": adaptive}


def _evaluate_method(
    method: str,
    binary,
    bounds,
    click_x: int,
    click_y: int,
    pitch_map,
) -> MethodCandidate:
    closed = apply_vertical_close_roi(binary, CONFIG)
    black_mask = create_black_mask_roi(closed)
    click_x_roi = click_x - bounds.x0_global
    click_y_roi = click_y - bounds.y0_global
    analysis = analyze_adjacent_stripes(
        black_mask,
        click_x_roi,
        click_x,
        bounds.x0_global,
        CONFIG,
    )
    selection = select_interactive_tracks(
        black_mask,
        click_x_roi,
        click_y_roi,
        analysis,
        CONFIG,
    )
    local_check = _neighbor_consistency_check(
        selection,
        analysis,
        CONFIG,
        INTERACTIVE_CONFIG,
    )
    pitch_guard = evaluate_pitch_guard(
        selection,
        click_x,
        click_y,
        pitch_map,
        INTERACTIVE_CONFIG,
    )
    return MethodCandidate(
        method=method,
        selection=selection,
        analysis=analysis,
        local_check=local_check,
        pitch_guard=pitch_guard,
        black_mask=black_mask,
    )


def _global_center(track, bounds):
    if track is None:
        return None
    return round(bounds.x0_global + track.center_x_roi, 3)


def _selected_centers(candidate: MethodCandidate, bounds) -> tuple:
    selection = candidate.selection
    return (
        _global_center(selection.left_track, bounds),
        _global_center(selection.right_track, bounds),
    )


def _methods_disagree(
    otsu: MethodCandidate,
    adaptive: MethodCandidate,
    bounds,
) -> bool:
    if otsu.successful != adaptive.successful:
        return True
    if not otsu.successful:
        return False
    otsu_centers = _selected_centers(otsu, bounds)
    adaptive_centers = _selected_centers(adaptive, bounds)
    return any(
        abs(left - right) > 3.0
        for left, right in zip(otsu_centers, adaptive_centers)
    )


def _candidate_category(
    otsu: MethodCandidate,
    adaptive: MethodCandidate,
    bounds,
) -> str:
    if not otsu.successful and not adaptive.successful:
        return "negative_candidate"
    if _methods_disagree(otsu, adaptive, bounds):
        return "method_disagreement"
    checks = (otsu, adaptive)
    if any(
        candidate.local_check.get("passed") is False
        or candidate.pitch_guard.get("status") == "Suspicious"
        for candidate in checks
    ):
        return "suspicious_or_defect"
    return "agree_normal"


def _grid_points(image_shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = image_shape
    # Use a moderately dense pool so narrow defect bands are represented.
    # Only 20 points per image are selected from this pool.
    x_values = np.linspace(280, width - 281, 21).round().astype(int)
    y_values = np.linspace(120, height - 121, 17).round().astype(int)
    return [(int(x), int(y)) for y in y_values for x in x_values]


def _build_pool(stripe_number: int) -> list[ReviewCandidate]:
    image_path = IMAGE_PATHS[stripe_number]
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(f"could not load review image: {image_path}")
    pitch_map = build_pitch_reference_map(
        image_gray,
        CONFIG,
        INTERACTIVE_CONFIG,
    )
    points = list(SCREENSHOT_POINTS[stripe_number])
    points.extend(
        point
        for point in _grid_points(image_gray.shape)
        if point not in SCREENSHOT_POINTS[stripe_number]
    )
    pool = []
    for click_x, click_y in points:
        bounds = calculate_interactive_roi_bounds(
            image_gray.shape,
            click_x,
            click_y,
            INTERACTIVE_CONFIG,
        )
        roi = crop_roi_global(image_gray, bounds)
        thresholds = _threshold_images(roi)
        otsu = _evaluate_method(
            "otsu",
            thresholds["otsu"],
            bounds,
            click_x,
            click_y,
            pitch_map,
        )
        adaptive = _evaluate_method(
            "adaptive",
            thresholds["adaptive"],
            bounds,
            click_x,
            click_y,
            pitch_map,
        )
        pool.append(
            ReviewCandidate(
                stripe_number=stripe_number,
                image_path=image_path,
                click_x=click_x,
                click_y=click_y,
                category=_candidate_category(otsu, adaptive, bounds),
                otsu=otsu,
                adaptive=adaptive,
                screenshot_point=(
                    (click_x, click_y) in SCREENSHOT_POINTS[stripe_number]
                ),
            )
        )
    return pool


def _stratified_cases(pool: list[ReviewCandidate]) -> list[ReviewCandidate]:
    chosen = []
    used_points = set()
    for category in CATEGORIES:
        category_cases = [
            case for case in pool if case.category == category
        ]
        category_cases.sort(
            key=lambda case: (
                not case.screenshot_point,
                case.click_y,
                case.click_x,
            )
        )
        selected = category_cases[:CASES_PER_CATEGORY]
        if len(selected) != CASES_PER_CATEGORY:
            raise RuntimeError(
                f"not enough {category} cases: found {len(category_cases)}"
            )
        for case in selected:
            point = (case.click_x, case.click_y)
            if point in used_points:
                raise RuntimeError(f"duplicate selected review point: {point}")
            used_points.add(point)
            chosen.append(case)
    return chosen


def _candidate_to_dict(candidate: MethodCandidate, bounds) -> dict:
    selection = candidate.selection
    tracks = [
        track
        for track in (
            selection.clicked_track,
            selection.left_track,
            selection.right_track,
        )
        if track is not None
    ]
    minimum_support = (
        None
        if not tracks
        else round(min(track.valid_row_ratio for track in tracks), 6)
    )
    minimum_retention = (
        None
        if not tracks
        else round(min(track.retention_ratio for track in tracks), 6)
    )
    return {
        "method": candidate.method,
        "success": selection.success,
        "classification": selection.click_classification,
        "failure_reasons": list(selection.failure_reasons),
        "clicked_center_x": _global_center(selection.clicked_track, bounds),
        "left_center_x": _global_center(selection.left_track, bounds),
        "right_center_x": _global_center(selection.right_track, bounds),
        "minimum_valid_row_ratio": minimum_support,
        "minimum_retention_ratio": minimum_retention,
        "local_neighbor_check": candidate.local_check,
        "pitch_guard": candidate.pitch_guard,
    }


def _suggestion(case: ReviewCandidate, bounds) -> dict:
    candidates = (case.otsu, case.adaptive)

    def quality(candidate):
        pitch_rank = {
            "Normal": 2,
            "Unable to verify": 1,
            "Suspicious": 0,
            "Not applicable": -1,
        }.get(candidate.pitch_guard["status"], -1)
        local_rank = {
            True: 2,
            None: 1,
            False: 0,
        }[candidate.local_check["passed"]]
        tracks = [
            track
            for track in (
                candidate.selection.clicked_track,
                candidate.selection.left_track,
                candidate.selection.right_track,
            )
            if track is not None
        ]
        support = min(
            (track.valid_row_ratio for track in tracks),
            default=0.0,
        )
        legacy_tie_break = candidate.method == "otsu"
        return (
            candidate.successful,
            pitch_rank,
            local_rank,
            support,
            legacy_tie_break,
        )

    selected = max(candidates, key=quality)
    return {
        "method": selected.method,
        "valid_pair": selected.successful,
        "classification": selected.selection.click_classification,
        "clicked_center_x": _global_center(
            selected.selection.clicked_track,
            bounds,
        ),
        "left_center_x": _global_center(
            selected.selection.left_track,
            bounds,
        ),
        "right_center_x": _global_center(
            selected.selection.right_track,
            bounds,
        ),
        "warning": "algorithm suggestion only; human confirmation required",
    }


def _draw_panel(
    roi,
    candidate: MethodCandidate,
    bounds,
    click_x: int,
    click_y: int,
) -> np.ndarray:
    panel = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    height, width = panel.shape[:2]
    selection = candidate.selection
    for track, color in (
        (selection.clicked_track, (0, 140, 255)),
        (selection.left_track, (255, 0, 0)),
        (selection.right_track, (0, 255, 255)),
    ):
        if track is None:
            continue
        center_x = int(round(track.center_x_roi))
        cv2.line(panel, (center_x, 0), (center_x, height - 1), color, 2)
    cv2.drawMarker(
        panel,
        (click_x - bounds.x0_global, click_y - bounds.y0_global),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        17,
        2,
    )
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (width - 1, 64), (0, 0, 0), -1)
    panel = cv2.addWeighted(overlay, 0.78, panel, 0.22, 0)
    local = candidate.local_check["passed"]
    local_text = "pass" if local is True else "fail" if local is False else "n/a"
    cv2.putText(
        panel,
        (
            f"{candidate.method.upper()}  raw_success={candidate.successful}  "
            f"local={local_text}  pitch={candidate.pitch_guard['status']}"
        ),
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    centers = _candidate_to_dict(candidate, bounds)
    cv2.putText(
        panel,
        (
            f"L={centers['left_center_x']}  clicked="
            f"{centers['clicked_center_x']}  R={centers['right_center_x']}"
        ),
        (8, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return panel


def _save_review_image(
    output_path: Path,
    case_id: str,
    case: ReviewCandidate,
    image_gray,
    bounds,
) -> None:
    roi = crop_roi_global(image_gray, bounds)
    otsu_panel = _draw_panel(
        roi, case.otsu, bounds, case.click_x, case.click_y
    )
    adaptive_panel = _draw_panel(
        roi, case.adaptive, bounds, case.click_x, case.click_y
    )
    panels = np.hstack((otsu_panel, adaptive_panel))
    scale = 1.5
    panels = cv2.resize(
        panels,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )
    header_height = 72
    review = np.zeros(
        (panels.shape[0] + header_height, panels.shape[1], 3),
        dtype=np.uint8,
    )
    review[header_height:] = panels
    cv2.putText(
        review,
        (
            f"{case_id}  Stripe {case.stripe_number}  "
            f"click=({case.click_x},{case.click_y})  category={case.category}"
        ),
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        review,
        "Blue=left  Orange=clicked track  Yellow=right  Red cross=click",
        (12, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output_path), review):
        raise OSError(f"could not save review image: {output_path}")


def _write_csv(output_path: Path, cases: list[dict]) -> None:
    fields = (
        "id",
        "stripe",
        "category",
        "click_x",
        "click_y",
        "review_status",
        "valid_pair",
        "click_classification",
        "clicked_center_x",
        "left_center_x",
        "right_center_x",
        "notes",
    )
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            truth = case["ground_truth"]
            writer.writerow(
                {
                    "id": case["id"],
                    "stripe": case["stripe"],
                    "category": case["category"],
                    "click_x": case["click"]["x"],
                    "click_y": case["click"]["y"],
                    **truth,
                }
            )


def build_review_package(output_dir: Path) -> dict:
    """Create images and a pending review document; do not create truth."""

    output_dir.mkdir(parents=True, exist_ok=True)
    review_image_dir = output_dir / "cases"
    review_image_dir.mkdir(parents=True, exist_ok=True)
    document_cases = []
    for stripe_number in (9, 10):
        image_path = IMAGE_PATHS[stripe_number]
        image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        selected = _stratified_cases(_build_pool(stripe_number))
        for index, case in enumerate(selected, start=1):
            case_id = f"S{stripe_number:02d}-{index:02d}"
            bounds = calculate_interactive_roi_bounds(
                image_gray.shape,
                case.click_x,
                case.click_y,
                INTERACTIVE_CONFIG,
            )
            review_filename = f"{case_id}.png"
            _save_review_image(
                review_image_dir / review_filename,
                case_id,
                case,
                image_gray,
                bounds,
            )
            document_cases.append(
                {
                    "id": case_id,
                    "stripe": stripe_number,
                    "category": case.category,
                    "screenshot_point": case.screenshot_point,
                    "image_path": str(image_path.relative_to(PROJECT_ROOT)),
                    "click": {"x": case.click_x, "y": case.click_y},
                    "review_image": f"cases/{review_filename}",
                    "candidates": {
                        "otsu": _candidate_to_dict(case.otsu, bounds),
                        "adaptive": _candidate_to_dict(
                            case.adaptive,
                            bounds,
                        ),
                    },
                    "suggestion": _suggestion(case, bounds),
                    "ground_truth": {
                        "review_status": "pending",
                        "valid_pair": None,
                        "click_classification": None,
                        "clicked_center_x": None,
                        "left_center_x": None,
                        "right_center_x": None,
                        "notes": "",
                    },
                }
            )
    document = {
        "schema_version": GROUND_TRUTH_SCHEMA_VERSION,
        "review_status": "pending",
        "center_tolerance_px": 3.0,
        "case_count": len(document_cases),
        "instructions": (
            "Suggestions are not ground truth. Confirm valid_pair and centers "
            "from each review image before changing detection logic."
        ),
        "cases": document_cases,
    }
    validate_ground_truth_document(document)
    (output_dir / "review_cases.json").write_text(
        json.dumps(document, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_dir / "review_cases.csv", document_cases)
    (output_dir / "README.md").write_text(
        "\n".join(
            (
                "# Stripe 9/10 ground-truth review",
                "",
                "Review all 40 PNG files in `cases/`.",
                "Blue is the proposed left center, orange is the clicked track,",
                "yellow is the proposed right center, and the red cross is the click.",
                "",
                "For each row in `review_cases.csv`, confirm whether a valid nearest",
                "neighbor pair exists. If it does, fill the confirmed left/right",
                "global center X values. Suggestions must not be accepted blindly.",
                "",
                "The six user screenshot points have `screenshot_point=true` in JSON.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ground_truth_review",
    )
    args = parser.parse_args()
    document = build_review_package(args.output_dir)
    print(
        f"Created {document['case_count']} pending review cases in "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
