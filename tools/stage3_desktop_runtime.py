"""Desktop result adapter for the frozen cross-image v1.1 detector.

This module does not alter separator, basin, center, or raw-pitch decisions.
It only selects the frozen v1.1 runner and maps its result into the compact
data contract consumed by the Tk desktop result page.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from tools import basin_graph_joint_prototype as joint
from tools import cross_image_correction_prototype_v1_1 as cross_image_v1_1
from tools import separator_path_prototype as separator
from tools.lightweight_profile import (
    ProfileSession,
    activate as activate_profile,
    active_session,
    profiled,
    stage as profile_stage,
)
from config import CONFIG, INTERACTIVE_CONFIG
from image_processing import crop_roi_global
from interactive_pipeline import (
    _preprocess_roi,
    _rotation_guard_reasons,
    estimate_rotation_shadow,
)


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

    session = active_session()
    owns_session = session is None
    if session is None:
        session = ProfileSession()

    def run():
        with profile_stage("stage3_algorithm_total"):
            return runner(
                image_gray,
                dict(reference_global),
                bounds_to_dict(bounds),
                "vertical",
                joint.DEFAULT_CONFIG,
            )

    if owns_session:
        with activate_profile(session):
            result = run()
    else:
        result = run()
    result.setdefault("debug", {})["performance_profile"] = session.report()
    return result


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


def _partial_side_geometry(
    stage3_result: dict,
    reference_global: dict,
    bounds,
) -> tuple[dict | None, dict[str, dict]]:
    """Return independently verified output basins from one role hypothesis."""

    debug = stage3_result.get("debug", {})
    if stage3_result.get("unavailable_reason") not in {
        "basin_structure_not_verified",
        "path_order_conflict",
        "separator_adjacent_dark_basins_not_verified",
        "reference_separator_basin_safety_conflict",
        "continuous_dark_basins_not_verified",
        "intermediate_dark_basin_present",
        "output_basin_center_not_dark",
    }:
        return None, {
            side: {
                "status": "unavailable",
                "reason": stage3_result.get("unavailable_reason"),
            }
            for side in ("left", "right")
        }
    separator_result = debug.get("separator_result", {})
    arbitration = separator_result.get("arbitration_debug", {})
    hypotheses = arbitration.get("role_hypotheses", [])
    side_status = {
        side: {
            "status": "unavailable",
            "reason": stage3_result.get("unavailable_reason"),
        }
        for side in ("left", "right")
    }
    side_pairs = None
    reference_relation = None
    if len(hypotheses) == 1 and hypotheses[0].get("type") == (
        "clicked_basin_roles"
    ):
        selection = hypotheses[0].get("selection_candidate_ids", {})
        side_pairs = {
            "left": (
                selection.get("left_adjacent"),
                selection.get("left_clicked_boundary"),
            ),
            "right": (
                selection.get("right_clicked_boundary"),
                selection.get("right_adjacent"),
            ),
        }
        reference_relation = "inside_basin"
    else:
        reference_matches = [
            item
            for item in arbitration.get("competing_explanations", [])
            if item.get("type") == "reference_on_separator"
        ]
        ordered_ids = debug.get("basin_graph", {}).get(
            "ordered_separator_ids", []
        )
        if (
            len(reference_matches) == 1
            and reference_matches[0].get("candidate_id") in ordered_ids
        ):
            reference_id = reference_matches[0]["candidate_id"]
            index = ordered_ids.index(reference_id)
            side_pairs = {
                "left": (
                    ordered_ids[index - 1] if index > 0 else None,
                    reference_id,
                ),
                "right": (
                    reference_id,
                    ordered_ids[index + 1]
                    if index + 1 < len(ordered_ids)
                    else None,
                ),
            }
            reference_relation = "on_separator"
    if side_pairs is None:
        return None, side_status

    candidate_by_id = {
        candidate.get("candidate_id"): candidate
        for candidate in separator_result.get("candidates", [])
    }
    basin_by_pair = {
        (basin.get("left_separator_id"), basin.get("right_separator_id")): basin
        for basin in debug.get("basin_graph", {}).get("basin_candidates", [])
    }
    center_audits = {
        audit.get("role"): audit
        for audit in debug.get("output_basin_center_darkness_safety", {}).get(
            "basin_audits", []
        )
    }
    unsafe_intermediate = {
        item.get("side")
        for item in debug.get("intermediate_dark_basin_evidence", {}).get(
            "sides", []
        )
        if item.get("verified")
    }

    sides = {}
    for side, pair in side_pairs.items():
        if None in pair:
            continue
        paths = [candidate_by_id.get(candidate_id) for candidate_id in pair]
        if any(path is None or not path.get("accepted") for path in paths):
            side_status[side]["reason"] = "side_separator_path_unavailable"
            continue
        if any(
            path.get("support_fraction", 0.0)
            < separator.DEFAULT_CONFIG.minimum_role_path_support_fraction
            for path in paths
        ):
            side_status[side]["reason"] = (
                "role_path_insufficient_vertical_support"
            )
            continue
        if any(
            path.get("mean_step_px", math.inf)
            > separator.DEFAULT_CONFIG.maximum_role_path_mean_step_px
            for path in paths
        ):
            side_status[side]["reason"] = "role_path_geometry_unstable"
            continue
        basin = basin_by_pair.get(pair)
        if basin is None or not basin.get("verified"):
            side_status[side]["reason"] = "side_basin_not_verified"
            continue
        if side in unsafe_intermediate:
            side_status[side]["reason"] = "intermediate_dark_basin_present"
            continue
        audit = center_audits.get(side)
        if audit is not None and not audit.get("accepted"):
            side_status[side]["reason"] = "output_basin_center_not_dark"
            continue

        center_roi = float(basin["center_x_at_reference_roi"])
        center_global = bounds.x0_global + center_roi
        distance = (
            reference_global["x"] - center_global
            if side == "left"
            else center_global - reference_global["x"]
        )
        if distance < 0:
            side_status[side]["reason"] = "side_geometry_wrong_side"
            continue
        sides[side] = {
            "center_x_global": float(center_global),
            "center_x_roi": center_roi,
            "distance_to_click_px": float(distance),
            "basin_id": basin["basin_id"],
            "left_separator_id": basin["left_separator_id"],
            "right_separator_id": basin["right_separator_id"],
            "basin_width_px": float(basin["width_at_reference_px"]),
            "evidence_status": "verified",
        }
        side_status[side] = {"status": "available", "reason": None}

    if not sides:
        return None, side_status
    geometry = {
        "reference_relation": reference_relation,
        "left": sides.get("left"),
        "right": sides.get("right"),
        "stripe_spacing_px": None,
        "atomic": False,
        "partial": True,
    }
    if len(sides) == 2:
        geometry["stripe_spacing_px"] = (
            sides["right"]["center_x_global"]
            - sides["left"]["center_x_global"]
        )
    return geometry, side_status


def _draw_formal_overlay(
    image_gray: np.ndarray,
    reference_global: dict,
    bounds,
    geometry: dict | None,
    line_angle_deg: float = 0.0,
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
            if geometry.get(side) is None:
                continue
            center_x = float(geometry[side]["center_x_global"])
            slope_x_per_y = -math.tan(math.radians(line_angle_deg))
            top_x = center_x + slope_x_per_y * (
                bounds.y0_global - reference_global["y"]
            )
            bottom_y = bounds.y1_global - 1
            bottom_x = center_x + slope_x_per_y * (
                bottom_y - reference_global["y"]
            )
            cv2.line(
                overlay,
                (int(round(top_x)), bounds.y0_global),
                (int(round(bottom_x)), bottom_y),
                color,
                2,
                cv2.LINE_AA,
            )
    return overlay


def _estimate_formal_line_angle_deg(
    image_gray: np.ndarray,
    bounds,
    stage3_result: dict | None = None,
) -> float:
    """Choose a displayed angle consistent with verified basin trajectories."""

    image_gray_roi = crop_roi_global(image_gray, bounds)
    stages = _preprocess_roi(
        image_gray_roi,
        CONFIG,
        INTERACTIVE_CONFIG,
        "otsu",
    )
    angle_deg = estimate_rotation_shadow(
        stages.vertical_close,
        INTERACTIVE_CONFIG,
    )
    best_angle_deg = angle_deg.best_angle_deg
    if abs(best_angle_deg) < INTERACTIVE_CONFIG.rotation_min_abs_angle_deg:
        return 0.0
    if stage3_result is None:
        return float(best_angle_deg)

    structural = _reported_basin_orientation(stage3_result)
    if structural is None:
        return float(best_angle_deg)
    guard_reasons = _rotation_guard_reasons(
        angle_deg,
        list(stages.threshold_warning_flags),
        INTERACTIVE_CONFIG,
    )
    current_drawn_angle_deg = -float(best_angle_deg)
    basin_drawn_angle_deg = structural["angle_from_vertical_deg"]
    mismatch_drift_px = abs(
        math.tan(math.radians(current_drawn_angle_deg))
        - math.tan(math.radians(basin_drawn_angle_deg))
    ) * structural["vertical_span_px"]
    if (
        guard_reasons
        and mismatch_drift_px
        > separator.DEFAULT_CONFIG.trace_search_radius_px
    ):
        return -float(basin_drawn_angle_deg)
    return float(best_angle_deg)


def _reported_basin_orientation(
    stage3_result: dict,
) -> dict[str, float] | None:
    """Return a robust angle from the two basin trajectories being reported."""

    hypothesis = stage3_result.get("final_hypothesis") or {}
    basin_ids = hypothesis.get("basin_ids") or {}
    requested_ids = [basin_ids.get(role) for role in ("left", "right")]
    if any(basin_id is None for basin_id in requested_ids):
        return None
    basin_by_id = {
        basin.get("basin_id"): basin
        for basin in (
            stage3_result.get("debug", {})
            .get("basin_graph", {})
            .get("verified_basins", [])
        )
    }
    angles = []
    spans = []
    for basin_id in requested_ids:
        basin = basin_by_id.get(basin_id)
        if basin is None:
            return None
        y = np.asarray(basin.get("band_centers_y_roi", []), dtype=np.float64)
        x = np.asarray(basin.get("center_x_by_band_roi", []), dtype=np.float64)
        if y.size < 2 or x.size != y.size:
            return None
        slopes = [
            (x[j] - x[i]) / (y[j] - y[i])
            for i in range(y.size)
            for j in range(i + 1, y.size)
            if y[j] != y[i]
        ]
        if not slopes:
            return None
        angles.append(math.degrees(math.atan(float(np.median(slopes)))))
        spans.append(float(y[-1] - y[0]))
    return {
        "angle_from_vertical_deg": float(np.median(angles)),
        "vertical_span_px": min(spans),
    }


@profiled("desktop_result_construction", "result_construction")
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
    if geometry is None:
        geometry, side_status = _partial_side_geometry(
            stage3_result,
            reference_global,
            bounds,
        )
    else:
        side_status = {
            side: {"status": "available", "reason": None}
            for side in ("left", "right")
        }
    left = None if geometry is None else geometry.get("left")
    right = None if geometry is None else geometry.get("right")
    success = left is not None or right is not None
    bilateral_success = left is not None and right is not None
    internal_reason = stage3_result.get("unavailable_reason")
    if stage3_result.get("success") and not bilateral_success:
        internal_reason = "stage3_atomic_result_contract_invalid"
    pitch = _pitch_provenance(stage3_result)
    interactive_result = {
        "result_source": RESULT_SOURCE,
        "success": success,
        "status": "available" if success else "unavailable",
        "unavailable_reason": None if success else internal_reason,
        "bilateral_success": bilateral_success,
        "side_status": side_status,
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
        "left": left,
        "right": right,
        "stripe_spacing_px": (
            None if geometry is None else geometry.get("stripe_spacing_px")
        ),
        "pitch_evidence": pitch,
        "pitch_guard": {
            "status": "Not applicable",
            "baseline_pitch_px": None,
            "observed_intervals_px": [],
            "interval_pitch_ratios": [],
        },
        "adjacency_verification": {
            "status": (
                "verified"
                if bilateral_success
                else "partial" if success else "unavailable"
            ),
            "reason": None if bilateral_success else internal_reason,
            "source": "frozen_separator_basin_hypothesis",
            "sides": side_status,
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
    with profile_stage("overlay_preparation", "overlay_visualization"):
        line_angle_deg = (
            _estimate_formal_line_angle_deg(image_gray, bounds, stage3_result)
            if geometry is not None
            else 0.0
        )
    with profile_stage("overlay_drawing", "overlay_visualization"):
        overlay = _draw_formal_overlay(
            image_gray,
            reference_global,
            bounds,
            geometry,
            line_angle_deg,
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


def persist_formal_report(result: Stage3DesktopResult) -> None:
    """Atomically save the formal Stage 3.1 JSON report."""

    with profile_stage(
        "formal_report_json_serialization_and_write",
        "post_analysis_json_write",
    ):
        result.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = result.output_dir / FORMAL_REPORT_FILENAME
        temporary_report = report_path.with_suffix(".json.tmp")
        temporary_report.write_text(
            json.dumps(result.report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary_report.replace(report_path)


def persist_formal_overlay(
    result: Stage3DesktopResult,
    temporary_token: str | None = None,
) -> None:
    """Atomically save the already-built full-image formal overlay."""

    with profile_stage(
        "formal_overlay_image_encode_and_write",
        "post_analysis_overlay_write",
    ):
        overlay_path = result.output_dir / FORMAL_OVERLAY_FILENAME
        token = f".{temporary_token}" if temporary_token else ""
        temporary_overlay = overlay_path.with_name(
            f"{overlay_path.stem}{token}.tmp{overlay_path.suffix}"
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


def persist_formal_result(result: Stage3DesktopResult) -> None:
    """Atomically save the formal report and full-image overlay."""

    persist_formal_report(result)
    persist_formal_overlay(result)
