"""Build a small pending-review package for topology shadow evidence."""

from __future__ import annotations

import argparse
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
from grayscale_topology import evaluate_grayscale_topology  # noqa: E402
from image_processing import crop_roi_global  # noqa: E402
from interactive_analysis import select_interactive_tracks  # noqa: E402
from interactive_pipeline import (  # noqa: E402
    _identity_affine_matrix,
    _preprocess_roi,
    calculate_interactive_roi_bounds,
    run_interactive_case,
)
from pitch_reference import build_pitch_reference_map  # noqa: E402
from stripe_analysis import analyze_adjacent_stripes  # noqa: E402


CONFIRMED_TRUTH_PATH = (
    PROJECT_ROOT / "tests" / "data" / "stripe_09_10_ground_truth.json"
)
KNOWN_ERRORS = (
    ("sample2_split_01", (1613, 1745)),
    ("sample2_split_02", (661, 1157)),
)
EXCLUDED_NAMES = {
    "Sample 2.bmp",
    "Stripe_09_e0_t200229235_v3p02565_do.bmp",
    "Stripe_10_e0_t221236602_v8p56736_retry.bmp",
}


@dataclass(frozen=True)
class ProbeMethod:
    success: bool
    left_x: float | None
    right_x: float | None
    minimum_support: float
    topology_status: str
    strong_conflict: bool


@dataclass(frozen=True)
class PointProbe:
    click_x: int
    click_y: int
    otsu: ProbeMethod
    adaptive: ProbeMethod


def _method_probe(
    image_gray_roi,
    click_x_roi: int,
    click_y_roi: int,
    click_x_global: int,
    bounds,
    method: str,
) -> ProbeMethod:
    stages = _preprocess_roi(
        image_gray_roi,
        CONFIG,
        INTERACTIVE_CONFIG,
        method,
    )
    analysis = analyze_adjacent_stripes(
        stages.black_mask,
        click_x_roi,
        click_x_global,
        bounds.x0_global,
        CONFIG,
    )
    selection = select_interactive_tracks(
        stages.black_mask,
        click_x_roi,
        click_y_roi,
        analysis,
        CONFIG,
        image_gray_roi=stages.image_gray,
    )
    topology = evaluate_grayscale_topology(
        image_gray_roi,
        selection,
        _identity_affine_matrix(),
        INTERACTIVE_CONFIG,
    ).report
    selected_tracks = [
        track
        for track in (
            selection.left_track,
            selection.clicked_track,
            selection.right_track,
        )
        if track is not None
    ]
    return ProbeMethod(
        success=selection.success,
        left_x=(
            None
            if selection.left_track is None
            else selection.left_track.center_x_roi
        ),
        right_x=(
            None
            if selection.right_track is None
            else selection.right_track.center_x_roi
        ),
        minimum_support=min(
            (track.valid_row_ratio for track in selected_tracks),
            default=0.0,
        ),
        topology_status=topology["status"],
        strong_conflict=topology["strong_same_basin_conflict"],
    )


def _probe_point(
    image_gray,
    click_x: int,
    click_y: int,
) -> PointProbe:
    bounds = calculate_interactive_roi_bounds(
        image_gray.shape,
        click_x,
        click_y,
        INTERACTIVE_CONFIG,
    )
    roi = crop_roi_global(image_gray, bounds)
    click_x_roi = click_x - bounds.x0_global
    click_y_roi = click_y - bounds.y0_global
    return PointProbe(
        click_x=click_x,
        click_y=click_y,
        otsu=_method_probe(
            roi,
            click_x_roi,
            click_y_roi,
            click_x,
            bounds,
            "otsu",
        ),
        adaptive=_method_probe(
            roi,
            click_x_roi,
            click_y_roi,
            click_x,
            bounds,
            "adaptive",
        ),
    )


def _grid_points(image_shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = image_shape
    x_values = np.linspace(
        INTERACTIVE_CONFIG.roi_half_width_px,
        width - INTERACTIVE_CONFIG.roi_half_width_px - 1,
        7,
    ).round().astype(int)
    y_values = np.linspace(
        INTERACTIVE_CONFIG.roi_half_height_px,
        height - INTERACTIVE_CONFIG.roi_half_height_px - 1,
        5,
    ).round().astype(int)
    return [(int(x), int(y)) for y in y_values for x in x_values]


def _method_disagreement(probe: PointProbe) -> float:
    otsu = probe.otsu
    adaptive = probe.adaptive
    if otsu.success != adaptive.success:
        return 100.0
    if not otsu.success:
        return 40.0
    center_difference = max(
        abs(otsu.left_x - adaptive.left_x),
        abs(otsu.right_x - adaptive.right_x),
    )
    topology_bonus = 50.0 if (
        otsu.strong_conflict != adaptive.strong_conflict
    ) else 0.0
    return center_difference + topology_bonus


def _clear_score(probe: PointProbe) -> float | None:
    if not probe.otsu.success or not probe.adaptive.success:
        return None
    if max(
        abs(probe.otsu.left_x - probe.adaptive.left_x),
        abs(probe.otsu.right_x - probe.adaptive.right_x),
    ) > 3.0:
        return None
    if probe.otsu.strong_conflict or probe.adaptive.strong_conflict:
        return None
    consistent_count = sum(
        method.topology_status == "Consistent"
        for method in (probe.otsu, probe.adaptive)
    )
    return (
        min(probe.otsu.minimum_support, probe.adaptive.minimum_support)
        + consistent_count
    )


def _far_enough(
    probe: PointProbe,
    selected: list[PointProbe],
) -> bool:
    return all(
        abs(probe.click_x - other.click_x) >= 100
        or abs(probe.click_y - other.click_y) >= 100
        for other in selected
    )


def _select_probes(
    probes: list[PointProbe],
) -> tuple[list[tuple[str, PointProbe]], PointProbe | None]:
    clear_candidates = [
        (score, probe)
        for probe in probes
        if (score := _clear_score(probe)) is not None
    ]
    if clear_candidates:
        clear = max(clear_candidates, key=lambda item: item[0])[1]
    else:
        clear = min(probes, key=_method_disagreement)
    selected = [clear]
    challenge_candidates = [
        probe for probe in probes if _far_enough(probe, selected)
    ]
    challenge = max(
        challenge_candidates or probes,
        key=_method_disagreement,
    )
    selected.append(challenge)
    extra_candidates = [
        probe for probe in probes if _far_enough(probe, selected)
    ]
    extra = (
        None
        if not extra_candidates
        else max(extra_candidates, key=_method_disagreement)
    )
    return [("clear", clear), ("challenge", challenge)], extra


def _candidate_panel(
    roi_gray,
    click_x_roi: int,
    click_y_roi: int,
    candidate: dict,
    title: str,
) -> np.ndarray:
    panel = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    centers = candidate["selected_track_centers_roi"]
    for label, color in (
        ("left", (255, 0, 0)),
        ("clicked", (0, 140, 255)),
        ("right", (0, 255, 255)),
    ):
        center = centers[label]
        if center is not None:
            x_value = int(round(center))
            cv2.line(
                panel,
                (x_value, 0),
                (x_value, panel.shape[0] - 1),
                color,
                1,
            )
    cv2.drawMarker(
        panel,
        (click_x_roi, click_y_roi),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        15,
        2,
    )
    header = np.zeros((36, panel.shape[1], 3), dtype=np.uint8)
    topology = candidate["grayscale_topology"]
    cv2.putText(
        header,
        f"{title}: {topology['status']}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, panel))


def _review_card(result, case_id: str, selection_reason: str) -> np.ndarray:
    report = result.report
    roi = result.debug_images["roi_crop.png"]
    click_x_roi = report["click"]["x_roi"]
    click_y_roi = report["click"]["y_roi"]
    candidates = report["candidate_arbitration"]["candidates"]
    raw_candidate = {
        "selected_track_centers_roi": {
            "left": None,
            "clicked": None,
            "right": None,
        },
        "grayscale_topology": {"status": "Original ROI"},
    }
    panels = [
        _candidate_panel(
            roi,
            click_x_roi,
            click_y_roi,
            raw_candidate,
            "RAW",
        ),
        _candidate_panel(
            roi,
            click_x_roi,
            click_y_roi,
            candidates["original_otsu"],
            "OTSU",
        ),
        _candidate_panel(
            roi,
            click_x_roi,
            click_y_roi,
            candidates["original_adaptive"],
            "ADAPTIVE",
        ),
    ]
    card = np.hstack(panels)
    title = np.zeros((48, card.shape[1], 3), dtype=np.uint8)
    formal = report["interactive_result"]
    text = (
        f"{case_id} | {selection_reason} | click "
        f"({report['click']['x_global']}, {report['click']['y_global']}) | "
        f"formal {formal['geometry']}+{formal['threshold_method']}"
    )
    cv2.putText(
        title,
        text,
        (8, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((title, card))


def _case_manifest(
    case_id: str,
    image_path: Path,
    click: tuple[int, int],
    selection_reason: str,
    result,
) -> dict:
    report = result.report
    interactive = report["interactive_result"]
    return {
        "id": case_id,
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "click": {"x": click[0], "y": click[1]},
        "selection_reason": selection_reason,
        "formal_result": {
            "success": interactive["success"],
            "threshold_method": interactive["threshold_method"],
            "geometry": interactive["geometry"],
            "left_center_x": (
                None
                if interactive["left"] is None
                else interactive["left"]["center_x_global"]
            ),
            "right_center_x": (
                None
                if interactive["right"] is None
                else interactive["right"]["center_x_global"]
            ),
        },
        "shadow_arbitration": report["shadow_arbitration"],
        "original_candidates": {
            method: report["candidate_arbitration"]["candidates"][
                f"original_{method}"
            ]
            for method in ("otsu", "adaptive")
        },
        "review": {
            "status": "pending",
            "valid_pair": None,
            "left_center_x": None,
            "right_center_x": None,
            "notes": "",
        },
    }


def build_trial(output_dir: Path) -> dict:
    truth = json.loads(CONFIRMED_TRUTH_PATH.read_text(encoding="utf-8"))
    image_paths = [
        path
        for path in CONFIG.sweep_image_paths
        if path.name not in EXCLUDED_NAMES
    ]
    per_image = {}
    extra_rank = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        probes = [
            _probe_point(image, x_value, y_value)
            for x_value, y_value in _grid_points(image.shape)
        ]
        selected, extra = _select_probes(probes)
        per_image[image_path] = selected
        if extra is not None:
            extra_rank.append(
                (_method_disagreement(extra), image_path, extra)
            )
    for _score, image_path, extra in sorted(
        extra_rank,
        key=lambda item: item[0],
        reverse=True,
    )[:6]:
        per_image[image_path].append(("extra_disagreement", extra))

    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "cases"
    cards_dir = output_dir / "review_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    pending_cases = []
    case_index = 1
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        pitch_map = build_pitch_reference_map(
            image,
            CONFIG,
            INTERACTIVE_CONFIG,
        )
        for selection_reason, probe in per_image[image_path]:
            case_id = f"T{case_index:02d}"
            click = (probe.click_x, probe.click_y)
            result = run_interactive_case(
                image,
                click[0],
                click[1],
                cases_dir / case_id,
                CONFIG,
                INTERACTIVE_CONFIG,
                image_name=image_path.name,
                pitch_reference_map=pitch_map,
            )
            pending_cases.append(
                _case_manifest(
                    case_id,
                    image_path,
                    click,
                    selection_reason,
                    result,
                )
            )
            cv2.imwrite(
                str(cards_dir / f"{case_id}.png"),
                _review_card(result, case_id, selection_reason),
            )
            case_index += 1

    sample2_path = PROJECT_ROOT / "images" / "Sample 2.bmp"
    sample2 = cv2.imread(str(sample2_path), cv2.IMREAD_GRAYSCALE)
    sample2_pitch = build_pitch_reference_map(
        sample2,
        CONFIG,
        INTERACTIVE_CONFIG,
    )
    known_error_cases = []
    for case_id, click in KNOWN_ERRORS:
        result = run_interactive_case(
            sample2,
            click[0],
            click[1],
            cases_dir / case_id,
            CONFIG,
            INTERACTIVE_CONFIG,
            image_name=sample2_path.name,
            pitch_reference_map=sample2_pitch,
        )
        known_error_cases.append(
            _case_manifest(
                case_id,
                sample2_path,
                click,
                "known_same_basin_split",
                result,
            )
        )
        cv2.imwrite(
            str(cards_dir / f"{case_id}.png"),
            _review_card(
                result,
                case_id,
                "known_same_basin_split",
            ),
        )

    manifest = {
        "schema_version": 1,
        "mode": "grayscale_topology_shadow_trial",
        "confirmed_case_count": len(truth["cases"]),
        "confirmed_truth_path": str(
            CONFIRMED_TRUTH_PATH.relative_to(PROJECT_ROOT)
        ),
        "known_error_case_count": len(known_error_cases),
        "pending_review_case_count": len(pending_cases),
        "total_trial_case_count": (
            len(truth["cases"])
            + len(known_error_cases)
            + len(pending_cases)
        ),
        "known_error_cases": known_error_cases,
        "pending_review_cases": pending_cases,
    }
    (output_dir / "trial_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Grayscale topology shadow trial\n\n"
        "Review the numbered files in `review_cards/`. The 28 new cases "
        "remain pending and must not be treated as ground truth until a "
        "human confirms or corrects the pair.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "topology_shadow_trial",
    )
    args = parser.parse_args()
    manifest = build_trial(args.output)
    print(
        json.dumps(
            {
                "confirmed": manifest["confirmed_case_count"],
                "known_errors": manifest["known_error_case_count"],
                "pending_review": manifest["pending_review_case_count"],
                "total": manifest["total_trial_case_count"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
