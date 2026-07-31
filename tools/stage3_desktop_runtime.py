"""Desktop result adapter for the frozen cross-image v1.1 detector.

This module does not alter separator, basin, center, or raw-pitch decisions.
It only selects the frozen v1.1 runner and maps its result into the compact
data contract consumed by the Tk desktop result page.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from tools import basin_graph_joint_prototype as joint
from tools import cross_image_correction_prototype_v1_1 as cross_image_v1_1


RESULT_SOURCE = "cross_image_correction_v1_1"
UNAVAILABLE_REASON = "unable_to_determine_reliably"
FORMAL_REPORT_FILENAME = "interactive_results.json"
FORMAL_OVERLAY_FILENAME = "original_interactive_result.png"


@dataclass(frozen=True)
class Stage3DesktopResult:
    """Formal Stage 3.1 result data rendered by the desktop application."""

    click_x_global: int
    click_y_global: int
    bounds_global: object
    report: dict
    debug_images: dict[str, np.ndarray]
    output_dir: Path
    stage3_result: dict


def bounds_to_dict(bounds) -> dict:
    """Return the global half-open ROI bounds used by Stage 3.1."""

    return {
        "x0": int(bounds.x0_global),
        "y0": int(bounds.y0_global),
        "x1": int(bounds.x1_global),
        "y1": int(bounds.y1_global),
    }


def _run_cross_image_v1_1(
    image_gray,
    reference_global,
    roi_bounds_global,
    direction,
    _config,
):
    return cross_image_v1_1.run_joint_case_v1_1(
        image_gray,
        reference_global,
        roi_bounds_global,
        direction,
    )


def run_frozen_stage3(
    image_gray: np.ndarray,
    reference_global: dict,
    bounds,
    runner: Callable = _run_cross_image_v1_1,
) -> dict:
    """Run frozen cross-image v1.1 with the desktop image, point, and ROI."""

    return runner(
        image_gray,
        dict(reference_global),
        bounds_to_dict(bounds),
        "vertical",
        joint.DEFAULT_CONFIG,
    )


def _pitch_provenance(stage3_result: dict) -> dict:
    hypothesis = stage3_result.get("final_hypothesis")
    if hypothesis is not None:
        evidence = hypothesis["pitch_evidence"]
    else:
        evidence = stage3_result["debug"]["raw_pitch_result"]
    return {
        "algorithm_revision": evidence["algorithm_revision"],
        "configuration_checksum": evidence["configuration_checksum"],
        "confidence": evidence["confidence"],
        "success_eligible": bool(evidence["success_eligible"]),
        "diagnostic_pitch_px": evidence["diagnostic_pitch_px"],
        "usable_pitch_px": evidence["usable_pitch_px"],
        "harmonic_ambiguity": evidence["harmonic_ambiguity"],
        "joint_decision": evidence.get(
            "joint_decision",
            "no_joint_hypothesis",
        ),
        "geometry_spacing_ratios": list(
            evidence.get("geometry_spacing_ratios", [])
        ),
    }


def _basin_details(stage3_result: dict) -> dict[str, dict]:
    graph = stage3_result["debug"]["basin_graph"]
    return {
        basin["basin_id"]: basin
        for basin in graph["verified_basins"]
    }


def _formal_geometry(
    stage3_result: dict,
    reference_global: dict,
    bounds,
) -> dict | None:
    """Copy one successful atomic hypothesis into formal UI geometry."""

    hypothesis = stage3_result.get("final_hypothesis")
    if not stage3_result.get("success") or hypothesis is None:
        return None
    if hypothesis.get("atomic") is not True:
        return None

    geometry = hypothesis["geometry"]
    basin_by_id = _basin_details(stage3_result)
    sides = {}
    for side in ("left", "right"):
        basin_id = hypothesis["basin_ids"][side]
        basin_geometry = geometry["basins"][side]
        basin = basin_by_id.get(basin_id)
        if basin is None:
            return None
        center_roi = float(
            basin_geometry["center_x_at_reference_roi"]
        )
        center_global = float(
            basin_geometry.get(
                "center_x_at_reference_global",
                bounds.x0_global + center_roi,
            )
        )
        distance = (
            reference_global["x"] - center_global
            if side == "left"
            else center_global - reference_global["x"]
        )
        if distance < 0:
            return None
        sides[side] = {
            "center_x_global": center_global,
            "center_x_roi": center_roi,
            "distance_to_click_px": float(distance),
            "basin_id": basin_id,
            "left_separator_id": basin["left_separator_id"],
            "right_separator_id": basin["right_separator_id"],
            "basin_width_px": float(
                basin_geometry.get(
                    "width_at_reference_px",
                    basin["width_at_reference_px"],
                )
            ),
            "evidence_status": "verified",
        }

    if sides["left"]["center_x_global"] >= sides["right"][
        "center_x_global"
    ]:
        return None
    return {
        "reference_relation": hypothesis["reference_relation"],
        "left": sides["left"],
        "right": sides["right"],
        "basin_ids": dict(hypothesis["basin_ids"]),
        "basin_sequence": list(hypothesis["basin_sequence"]),
        "separator_ids": hypothesis.get("separator_ids"),
        "separator_sequence": list(hypothesis["separator_sequence"]),
        "clicked_separator_id": hypothesis.get(
            "clicked_separator_id"
        ),
        "shared_boundary": hypothesis.get("shared_boundary"),
        "stripe_spacing_px": (
            sides["right"]["center_x_global"]
            - sides["left"]["center_x_global"]
        ),
        "atomic": True,
    }


def _draw_formal_overlay(
    image_gray: np.ndarray,
    reference_global: dict,
    bounds,
    geometry: dict | None,
) -> np.ndarray:
    overlay = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(
        overlay,
        (bounds.x0_global, bounds.y0_global),
        (bounds.x1_global - 1, bounds.y1_global - 1),
        (0, 255, 0),
        2,
    )
    cv2.drawMarker(
        overlay,
        (reference_global["x"], reference_global["y"]),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        21,
        2,
    )
    if geometry is not None:
        for side, color in (
            ("left", (255, 0, 0)),
            ("right", (0, 255, 255)),
        ):
            center_x = int(
                round(geometry[side]["center_x_global"])
            )
            cv2.line(
                overlay,
                (center_x, bounds.y0_global),
                (center_x, bounds.y1_global - 1),
                color,
                2,
                cv2.LINE_AA,
            )
    return overlay


def build_desktop_result(
    image_gray: np.ndarray,
    image_name: str,
    reference_global: dict,
    bounds,
    stage3_result: dict,
    output_dir: Path,
) -> Stage3DesktopResult:
    """Create a formal result without changing the frozen Stage 3.1 output."""

    geometry = _formal_geometry(
        stage3_result,
        reference_global,
        bounds,
    )
    success = geometry is not None
    internal_reason = stage3_result.get("unavailable_reason")
    if stage3_result.get("success") and not success:
        internal_reason = "stage3_atomic_result_contract_invalid"
    pitch = _pitch_provenance(stage3_result)
    interactive_result = {
        "result_source": RESULT_SOURCE,
        "success": success,
        "status": "available" if success else "unavailable",
        "unavailable_reason": None if success else internal_reason,
        "failure_reasons": (
            []
            if success
            else [
                UNAVAILABLE_REASON,
                *(
                    [internal_reason]
                    if internal_reason
                    and internal_reason != UNAVAILABLE_REASON
                    else []
                ),
            ]
        ),
        "warning_flags": [],
        "left": None if geometry is None else geometry["left"],
        "right": None if geometry is None else geometry["right"],
        "stripe_spacing_px": (
            None if geometry is None else geometry["stripe_spacing_px"]
        ),
        "pitch_evidence": pitch,
        "pitch_guard": {
            "status": "Not applicable",
            "baseline_pitch_px": None,
            "observed_intervals_px": [],
            "interval_pitch_ratios": [],
        },
        "adjacency_verification": {
            "status": "verified" if success else "unavailable",
            "reason": None if success else internal_reason,
            "source": "frozen_separator_basin_hypothesis",
        },
        "geometry": geometry,
    }
    report = {
        "algorithm": "separator_path_basin_graph_raw_pitch_joint",
        "algorithm_revision": stage3_result["algorithm_revision"],
        "configuration_checksum": stage3_result[
            "configuration_checksum"
        ],
        "result_source": RESULT_SOURCE,
        "image_name": image_name,
        "image": {"shape": list(image_gray.shape)},
        "click": {
            "x_global": reference_global["x"],
            "y_global": reference_global["y"],
            "x_roi": reference_global["x"] - bounds.x0_global,
            "y_roi": reference_global["y"] - bounds.y0_global,
        },
        "roi": {
            "x0_global": bounds.x0_global,
            "x1_global": bounds.x1_global,
            "y0_global": bounds.y0_global,
            "y1_global": bounds.y1_global,
            "width_px": bounds.x1_global - bounds.x0_global,
            "height_px": bounds.y1_global - bounds.y0_global,
        },
        "interactive_result": interactive_result,
    }
    overlay = _draw_formal_overlay(
        image_gray,
        reference_global,
        bounds,
        geometry,
    )
    return Stage3DesktopResult(
        click_x_global=reference_global["x"],
        click_y_global=reference_global["y"],
        bounds_global=bounds,
        report=report,
        debug_images={"original_interactive_result.png": overlay},
        output_dir=output_dir,
        stage3_result=stage3_result,
    )


def persist_formal_result(result: Stage3DesktopResult) -> None:
    """Atomically save only the formal Stage 3.1 report and final overlay."""

    result.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = result.output_dir / FORMAL_REPORT_FILENAME
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(
        json.dumps(result.report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(report_path)

    overlay_path = result.output_dir / FORMAL_OVERLAY_FILENAME
    temporary_overlay = overlay_path.with_name(
        f"{overlay_path.stem}.tmp{overlay_path.suffix}"
    )
    if not cv2.imwrite(
        str(temporary_overlay),
        result.debug_images[FORMAL_OVERLAY_FILENAME],
    ):
        raise OSError(
            f"could not write Stage 3.1 formal overlay: "
            f"{temporary_overlay}"
        )
    temporary_overlay.replace(overlay_path)
