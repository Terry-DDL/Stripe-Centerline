"""Serialize and draw frozen Stage 3.1 shadow evidence."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


SHADOW_OVERLAY_FILENAME = "basin_shadow_overlay.png"
ACCEPTANCE_GEOMETRY_REVISION = "basin_shadow_acceptance_geometry_v1"


def _path_x_at_reference(path: dict, reference_y_roi: float) -> float:
    return float(
        np.interp(
            reference_y_roi,
            path["band_centers_y_roi"],
            path["x_by_band_roi"],
        )
    )


def build_acceptance_geometry(
    shadow_result: dict,
    reference_global: dict,
    roi_bounds_global: dict,
    formal_geometry: dict | None,
) -> dict:
    """Copy only evidence needed to review one shadow decision."""

    separator_result = shadow_result["debug"]["separator_result"]
    graph = shadow_result["debug"]["basin_graph"]
    hypothesis = shadow_result.get("final_hypothesis")
    candidates = separator_result["candidates"]
    roles_by_id: dict[str, str] = {}
    if hypothesis is not None:
        if hypothesis.get("separator_ids"):
            roles_by_id = {
                candidate_id: role
                for role, candidate_id in hypothesis[
                    "separator_ids"
                ].items()
            }
        else:
            roles_by_id = dict(
                zip(
                    hypothesis["separator_sequence"],
                    (
                        "left_adjacent",
                        "clicked_separator",
                        "right_adjacent",
                    ),
                )
            )
        selected_ids = set(hypothesis["separator_sequence"])
        visible_paths = [
            path
            for path in candidates
            if path["candidate_id"] in selected_ids
        ]
        candidate_scope = "final_hypothesis"
    else:
        selected_ids = set()
        visible_paths = list(candidates)
        candidate_scope = "all_candidates"

    path_snapshots = []
    for path in visible_paths:
        band_y = [float(value) for value in path["band_centers_y_roi"]]
        path_x = [float(value) for value in path["x_by_band_roi"]]
        reference_x = float(
            path.get(
                "x_at_reference_roi",
                _path_x_at_reference(
                    path,
                    separator_result["reference_y_roi"],
                ),
            )
        )
        path_snapshots.append(
            {
                "separator_id": path["candidate_id"],
                "role": roles_by_id.get(path["candidate_id"]),
                "used_by_final_hypothesis": (
                    path["candidate_id"] in selected_ids
                ),
                "accepted": bool(path["accepted"]),
                "rejection_reason": path.get("rejection_reason"),
                "band_centers_y_roi": band_y,
                "x_by_band_roi": path_x,
                "x_at_reference_roi": reference_x,
            }
        )
    path_snapshots.sort(
        key=lambda path: (
            path["x_at_reference_roi"],
            path["separator_id"],
        )
    )

    basin_snapshots = []
    if hypothesis is not None:
        basin_by_id = {
            basin["basin_id"]: basin
            for basin in graph["verified_basins"]
        }
        for role, basin_id in hypothesis["basin_ids"].items():
            basin = basin_by_id[basin_id]
            band_y = [
                float(value)
                for value in basin["band_centers_y_roi"]
            ]
            left_x = [
                float(value)
                for value in basin["left_x_by_band_roi"]
            ]
            right_x = [
                float(value)
                for value in basin["right_x_by_band_roi"]
            ]
            center_x = [
                float(value)
                for value in basin["center_x_by_band_roi"]
            ]
            basin_snapshots.append(
                {
                    "role": role,
                    "basin_id": basin_id,
                    "left_separator_id": basin[
                        "left_separator_id"
                    ],
                    "right_separator_id": basin[
                        "right_separator_id"
                    ],
                    "band_centers_y_roi": band_y,
                    "left_boundary_x_by_band_roi": left_x,
                    "right_boundary_x_by_band_roi": right_x,
                    "center_x_by_band_roi": center_x,
                    "center_x_at_reference_roi": float(
                        basin["center_x_at_reference_roi"]
                    ),
                    "center_x_at_reference_global": (
                        roi_bounds_global["x0"]
                        + float(basin["center_x_at_reference_roi"])
                    ),
                    "width_at_reference_px": float(
                        basin["width_at_reference_px"]
                    ),
                }
            )

    final = None
    if formal_geometry is not None:
        left_global = formal_geometry["left_basin_center_x_global"]
        right_global = formal_geometry[
            "right_basin_center_x_global"
        ]
        final = {
            "reference_relation": formal_geometry["reference_relation"],
            "left_center_x_global": left_global,
            "left_center_x_roi": left_global - roi_bounds_global["x0"],
            "right_center_x_global": right_global,
            "right_center_x_roi": (
                right_global - roi_bounds_global["x0"]
            ),
            "left_distance_px": formal_geometry[
                "left_distance_to_reference_px"
            ],
            "right_distance_px": formal_geometry[
                "right_distance_to_reference_px"
            ],
            "basin_ids": formal_geometry["basin_ids"],
            "separator_sequence": formal_geometry[
                "separator_sequence"
            ],
            "atomic": formal_geometry["atomic"],
        }
    return {
        "revision": ACCEPTANCE_GEOMETRY_REVISION,
        "mode": (
            "final_hypothesis"
            if hypothesis is not None
            else "unavailable_candidates"
        ),
        "candidate_scope": candidate_scope,
        "reference_global": dict(reference_global),
        "reference_roi": {
            "x": reference_global["x"] - roi_bounds_global["x0"],
            "y": reference_global["y"] - roi_bounds_global["y0"],
        },
        "roi_bounds_global": dict(roi_bounds_global),
        "separator_paths": path_snapshots,
        "basins": basin_snapshots,
        "final_geometry": final,
        "rejection_reason": shadow_result["unavailable_reason"],
    }


def _points(
    x_values: list[float],
    y_values: list[float],
    scale: int,
) -> np.ndarray:
    return np.rint(
        np.column_stack((x_values, y_values)) * scale
    ).astype(np.int32)


def draw_shadow_overlay(
    image_gray: np.ndarray,
    acceptance: dict,
) -> np.ndarray:
    """Draw review-only evidence without affecting any detector result."""

    bounds = acceptance["roi_bounds_global"]
    roi = image_gray[
        bounds["y0"] : bounds["y1"],
        bounds["x0"] : bounds["x1"],
    ]
    scale = 2
    view = cv2.cvtColor(
        cv2.resize(
            roi,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        ),
        cv2.COLOR_GRAY2BGR,
    )
    basin_colors = {
        "left": (255, 90, 20),
        "clicked": (130, 130, 130),
        "right": (20, 210, 255),
    }
    shaded = view.copy()
    for basin in acceptance["basins"]:
        left = _points(
            basin["left_boundary_x_by_band_roi"],
            basin["band_centers_y_roi"],
            scale,
        )
        right = _points(
            basin["right_boundary_x_by_band_roi"],
            basin["band_centers_y_roi"],
            scale,
        )
        cv2.fillPoly(
            shaded,
            [np.vstack((left, right[::-1]))],
            basin_colors.get(basin["role"], (140, 140, 140)),
            cv2.LINE_AA,
        )
    view = cv2.addWeighted(shaded, 0.22, view, 0.78, 0)

    path_colors = (
        (255, 255, 0),
        (255, 80, 255),
        (0, 255, 255),
        (80, 255, 80),
    )
    for index, path in enumerate(acceptance["separator_paths"]):
        color = (
            path_colors[index % len(path_colors)]
            if path["accepted"]
            else (140, 140, 140)
        )
        points = _points(
            path["x_by_band_roi"],
            path["band_centers_y_roi"],
            scale,
        )
        cv2.polylines(
            view,
            [points],
            False,
            color,
            3 if path["accepted"] else 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            view,
            path["separator_id"],
            (int(points[0][0]) + 3, max(18, int(points[0][1]) + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    reference = acceptance["reference_roi"]
    cv2.drawMarker(
        view,
        (
            int(round(reference["x"] * scale)),
            int(round(reference["y"] * scale)),
        ),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        28,
        3,
        cv2.LINE_AA,
    )
    final = acceptance["final_geometry"]
    if final is not None:
        for side, color in (
            ("left", (255, 90, 20)),
            ("right", (20, 210, 255)),
        ):
            x = int(round(final[f"{side}_center_x_roi"] * scale))
            cv2.line(
                view,
                (x, 0),
                (x, view.shape[0] - 1),
                color,
                3,
                cv2.LINE_AA,
            )

    header = np.full(
        (72, view.shape[1], 3),
        (28, 28, 28),
        dtype=np.uint8,
    )
    success = final is not None
    reference_global = acceptance["reference_global"]
    cv2.putText(
        header,
        (
            f"Stage 3.1 shadow: "
            f"{'SUCCESS' if success else 'UNAVAILABLE'}  "
            f"reference=({reference_global['x']}, "
            f"{reference_global['y']})"
        ),
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (80, 220, 80) if success else (60, 190, 255),
        2,
        cv2.LINE_AA,
    )
    detail = (
        (
            f"L={final['left_distance_px']:.2f}px  "
            f"R={final['right_distance_px']:.2f}px  "
            f"relation={final['reference_relation']}"
        )
        if success
        else (
            f"reason={acceptance['rejection_reason']}  "
            f"candidates={len(acceptance['separator_paths'])}"
        )
    )
    cv2.putText(
        header,
        detail,
        (12, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, view))


def atomic_write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if not cv2.imwrite(str(temporary), image):
        raise OSError(f"could not write shadow overlay: {temporary}")
    temporary.replace(path)
