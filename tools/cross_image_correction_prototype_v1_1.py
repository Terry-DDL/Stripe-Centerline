"""Offline cross-image correction v1.1.

This module extends the isolated v1 experiment without changing the frozen
separator detector, raw-pitch implementation, desktop application, or release
runtime.  Ground truth is used only by the evaluation functions near the end
of this file and is never an inference input.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable
from unittest.mock import patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from tools import basin_graph_joint_prototype as joint  # noqa: E402
from tools import cross_image_correction_prototype as v1  # noqa: E402
from tools import raw_local_pitch_prototype_v3 as raw_pitch  # noqa: E402
from tools import separator_path_prototype as separator  # noqa: E402


EXPERIMENT_START_COMMIT = (
    "0f6300a0ab40dd68af7a6425f27cbb7ca9e1a484"
)
RELEASE_COMMIT = "55b26e6deac5acbb50e3934c03d71e3c5b56b269"
ALGORITHM_REVISION = "cross_image_correction_offline_v1_1"
FREEZE_PATH = (
    PROJECT_ROOT
    / "tests"
    / "data"
    / "cross_image_correction_v1_1_freeze.json"
)
SMOKE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "tests"
    / "data"
    / "cross_image_correction_v1_1_preregistered_smoke.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "cross_image_correction_v1_1"
)
FROZEN_DETECT_SEPARATOR_PATHS = separator.detect_separator_paths
FROZEN_RUN_JOINT_CASE = joint.run_joint_case
V1_RUN_JOINT_CASE = v1.run_centerized_joint_case


def configuration_document() -> dict:
    """Return fixed semantic rules without introducing new tuned thresholds."""

    return {
        "algorithm_revision": ALGORITHM_REVISION,
        "experiment_start_commit": EXPERIMENT_START_COMMIT,
        "release_commit": RELEASE_COMMIT,
        "plateau_detection": {
            "source": "cross_image_correction_v1_frozen",
            "configuration_checksum": v1.correction_configuration_checksum(
                v1.DEFAULT_CORRECTION_CONFIG
            ),
        },
        "plateau_support": {
            "source_edge_supported_response": (
                separator.DEFAULT_CONFIG.supported_band_response
            ),
            "candidate_minimum_support": (
                separator.DEFAULT_CONFIG.minimum_supported_fraction
            ),
            "role_minimum_support": (
                separator.DEFAULT_CONFIG.minimum_role_path_support_fraction
            ),
            "minimum_pair_and_coverage_fraction": (
                v1.DEFAULT_CORRECTION_CONFIG
                .plateau_pair_minimum_band_fraction
            ),
            "maximum_center_mean_step": (
                separator.DEFAULT_CONFIG.maximum_role_path_mean_step_px
            ),
            "width_stability_ratio": (
                separator.DEFAULT_CONFIG
                .minimum_full_height_order_ratio
            ),
            "aggregation": "minimum_of_physical_evidence_fractions",
        },
        "width_aware_reference": {
            "core_majority_fraction": (
                v1.DEFAULT_CORRECTION_CONFIG
                .plateau_pair_minimum_band_fraction
            ),
            "fallback_guard_rule": "frozen_center_distance",
            "frozen_uncertainty_px": (
                separator.DEFAULT_CONFIG
                .reference_relationship_uncertainty_px
            ),
        },
        "center_refinement": {
            "source": "cross_image_correction_v1_frozen",
            "configuration_checksum": v1.correction_configuration_checksum(
                v1.DEFAULT_CORRECTION_CONFIG
            ),
        },
    }


def configuration_checksum() -> str:
    payload = json.dumps(
        configuration_document(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_fraction(
    values: np.ndarray,
    lower: float,
    upper: float,
) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean((values >= lower) & (values <= upper)))


def _source_edge_pair(group_paths: list[dict]) -> tuple[dict, dict]:
    ordered = sorted(
        group_paths,
        key=lambda item: float(np.median(item["x_by_band_roi"])),
    )
    return ordered[0], ordered[-1]


def _edge_support_flags(path: dict) -> np.ndarray:
    responses = np.asarray(path["response_by_band"], dtype=np.float64)
    return (
        responses
        >= separator.DEFAULT_CONFIG.supported_band_response
    )


def _robust_mad(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _plateau_band_geometry(
    band_results: list[dict],
    left_source: dict,
    right_source: dict,
) -> tuple[list[dict], dict]:
    """Add core/uncertainty geometry derived only from observed edges."""

    left_edges = np.asarray(
        [item["left_edge_x_roi"] for item in band_results],
        dtype=np.float64,
    )
    right_edges = np.asarray(
        [item["right_edge_x_roi"] for item in band_results],
        dtype=np.float64,
    )
    left_source_x = np.asarray(
        left_source["x_by_band_roi"],
        dtype=np.float64,
    )
    right_source_x = np.asarray(
        right_source["x_by_band_roi"],
        dtype=np.float64,
    )
    left_residual = np.abs(left_edges - left_source_x)
    right_residual = np.abs(right_edges - right_source_x)
    left_uncertainty = float(
        np.median(left_residual)
        + _robust_mad(left_residual)
        + _robust_mad(left_edges)
    )
    right_uncertainty = float(
        np.median(right_residual)
        + _robust_mad(right_residual)
        + _robust_mad(right_edges)
    )

    enriched = []
    for item in band_results:
        left_edge = float(item["left_edge_x_roi"])
        right_edge = float(item["right_edge_x_roi"])
        center = float(item["center_x_roi"])
        core_left = min(center, left_edge + left_uncertainty)
        core_right = max(center, right_edge - right_uncertainty)
        valid = bool(core_left <= center <= core_right)
        enriched.append(
            {
                **item,
                "core_left_x_roi": core_left,
                "core_right_x_roi": core_right,
                "left_edge_uncertainty_range_x_roi": [
                    left_edge,
                    core_left,
                ],
                "right_edge_uncertainty_range_x_roi": [
                    core_right,
                    right_edge,
                ],
                "valid": valid,
                "provenance": (
                    "raw_gray_plateau_interval_and_source_edges"
                ),
            }
        )

    widths = right_edges - left_edges + 1.0
    centers = np.asarray(
        [item["center_x_roi"] for item in enriched],
        dtype=np.float64,
    )
    valid_flags = np.asarray(
        [item["valid"] for item in enriched],
        dtype=bool,
    )
    return enriched, {
        "median_left_edge_x_roi": float(np.median(left_edges)),
        "median_right_edge_x_roi": float(np.median(right_edges)),
        "median_center_x_roi": float(np.median(centers)),
        "median_width_px": float(np.median(widths)),
        "valid_band_fraction": float(np.mean(valid_flags)),
        "left_edge_uncertainty_px": left_uncertainty,
        "right_edge_uncertainty_px": right_uncertainty,
    }


def _plateau_support_evidence(
    group_paths: list[dict],
    group_evidence: dict,
    band_results: list[dict],
) -> dict:
    """Aggregate original edge evidence before assigning center support."""

    left_source, right_source = _source_edge_pair(group_paths)
    left_flags = _edge_support_flags(left_source)
    right_flags = _edge_support_flags(right_source)
    paired_flags = left_flags & right_flags
    plateau_flags = np.asarray(
        [
            bool(item["same_plateau"])
            for item in group_evidence["bands"]
        ],
        dtype=bool,
    )
    center_path = np.asarray(
        [item["center_x_roi"] for item in band_results],
        dtype=np.float64,
    )
    widths = np.asarray(
        [item["plateau_width_px"] for item in band_results],
        dtype=np.float64,
    )
    steps = np.abs(np.diff(center_path))
    center_stable_fraction = (
        float(
            np.mean(
                steps
                <= separator.DEFAULT_CONFIG
                .maximum_role_path_mean_step_px
            )
        )
        if steps.size
        else 1.0
    )
    median_width = float(np.median(widths))
    width_ratio = (
        separator.DEFAULT_CONFIG.minimum_full_height_order_ratio
    )
    width_stable_fraction = _stable_fraction(
        widths,
        median_width * width_ratio,
        median_width / width_ratio,
    )
    components = {
        "left_edge_support": float(np.mean(left_flags)),
        "right_edge_support": float(np.mean(right_flags)),
        "paired_band_fraction": float(np.mean(paired_flags)),
        "plateau_coverage_fraction": float(np.mean(plateau_flags)),
        "center_stability_fraction": center_stable_fraction,
        "width_stability_fraction": width_stable_fraction,
    }
    final_support = float(min(components.values()))
    minimum_pair_fraction = (
        v1.DEFAULT_CORRECTION_CONFIG
        .plateau_pair_minimum_band_fraction
    )
    rejection_reasons = []
    if components["paired_band_fraction"] < minimum_pair_fraction:
        rejection_reasons.append("plateau_edge_pairing_insufficient")
    if components["plateau_coverage_fraction"] < minimum_pair_fraction:
        rejection_reasons.append("plateau_coverage_insufficient")
    if center_stable_fraction < minimum_pair_fraction:
        rejection_reasons.append("plateau_center_path_unstable")
    if width_stable_fraction < minimum_pair_fraction:
        rejection_reasons.append("plateau_width_unstable")
    if (
        components["left_edge_support"]
        < separator.DEFAULT_CONFIG.minimum_role_path_support_fraction
        or components["right_edge_support"]
        < separator.DEFAULT_CONFIG.minimum_role_path_support_fraction
    ):
        rejection_reasons.append("plateau_source_edge_support_insufficient")
    return {
        "source_edge_ids": {
            "left": left_source["raw_candidate_id"],
            "right": right_source["raw_candidate_id"],
        },
        "source_edge_seed_x_roi": {
            "left": left_source["seed_x_roi"],
            "right": right_source["seed_x_roi"],
        },
        "source_edge_support": {
            "left": left_source["support_fraction"],
            "right": right_source["support_fraction"],
        },
        "source_edge_response_by_band": {
            "left": left_source["response_by_band"],
            "right": right_source["response_by_band"],
        },
        **components,
        "final_plateau_support": final_support,
        "support_composition": (
            "minimum(left_edge_support,right_edge_support,"
            "paired_band_fraction,plateau_coverage_fraction,"
            "center_stability_fraction,width_stability_fraction)"
        ),
        "rejection_reasons": rejection_reasons,
    }


def _centerized_candidate_v1_1(
    frozen_candidate: dict,
    group_paths: list[dict],
    evidence: dict,
    local_pitch_px: float,
    group_evidence: dict,
) -> dict:
    path_matrix = np.asarray(
        [path["x_by_band_roi"] for path in group_paths],
        dtype=np.float64,
    )
    raw_band_results = [
        v1._plateau_center_for_band(  # noqa: SLF001
            np.asarray(profile, dtype=np.float64),
            path_matrix[:, band_index],
            local_pitch_px,
            v1.DEFAULT_CORRECTION_CONFIG,
        )
        for band_index, profile in enumerate(evidence["profiles"])
    ]
    left_source, right_source = _source_edge_pair(group_paths)
    band_results, geometry_summary = _plateau_band_geometry(
        raw_band_results,
        left_source,
        right_source,
    )
    support = _plateau_support_evidence(
        group_paths,
        group_evidence,
        band_results,
    )
    center_path = np.asarray(
        [item["center_x_roi"] for item in band_results],
        dtype=np.float64,
    )
    left_response = np.asarray(
        left_source["response_by_band"],
        dtype=np.float64,
    )
    right_response = np.asarray(
        right_source["response_by_band"],
        dtype=np.float64,
    )
    paired_response = np.minimum(left_response, right_response)
    physical_supported = (
        (_edge_support_flags(left_source))
        & (_edge_support_flags(right_source))
        & np.asarray(
            [item["same_plateau"] for item in group_evidence["bands"]],
            dtype=bool,
        )
    )
    top_count = min(
        separator.DEFAULT_CONFIG.aggregate_top_band_count,
        paired_response.size,
    )
    top_response = float(
        np.mean(np.sort(paired_response)[-top_count:])
    )
    mean_step = (
        float(np.mean(np.abs(np.diff(center_path))))
        if center_path.size > 1
        else 0.0
    )
    support_fraction = support["final_plateau_support"]
    path_score = float(
        top_response
        + 0.45 * support_fraction
        - 0.04 * mean_step
    )
    accepted = bool(
        not support["rejection_reasons"]
        and support_fraction
        >= separator.DEFAULT_CONFIG.minimum_supported_fraction
        and top_response
        >= separator.DEFAULT_CONFIG.minimum_top_response
        and path_score
        >= separator.DEFAULT_CONFIG.minimum_path_score
    )
    candidate = copy.deepcopy(frozen_candidate)
    candidate.update(
        {
            "seed_x_roi": int(round(float(np.median(center_path)))),
            "x_by_band_roi": center_path.tolist(),
            "response_by_band": paired_response.tolist(),
            "supported_band_indices": np.flatnonzero(
                physical_supported
            ).tolist(),
            "support_fraction": support_fraction,
            "top_response": top_response,
            "mean_step_px": mean_step,
            "path_score": path_score,
            "peak_width_px": int(
                round(geometry_summary["median_width_px"])
            ),
            "accepted": accepted,
            "rejection_reason": (
                None
                if accepted
                else (
                    support["rejection_reasons"][0]
                    if support["rejection_reasons"]
                    else separator._path_rejection_reason(  # noqa: SLF001
                        support_fraction,
                        top_response,
                        path_score,
                        separator.DEFAULT_CONFIG,
                    )
                )
            ),
            "plateau_centerization": {
                "attempted": True,
                "adopted": True,
                "reason": group_evidence["reason"],
                "local_pitch_px": local_pitch_px,
                "old_seed_x_roi": [
                    path["seed_x_roi"] for path in group_paths
                ],
                "old_path_x_by_band_roi": [
                    path["x_by_band_roi"] for path in group_paths
                ],
                "new_path_x_by_band_roi": center_path.tolist(),
                "group_evidence": group_evidence,
                "band_geometry": band_results,
                "geometry_summary": geometry_summary,
                "plateau_support": support,
                "evidence_revision": (
                    "source_edge_plateau_support_v1_1"
                ),
            },
        }
    )
    candidate.pop("x_at_reference_roi", None)
    return candidate


def centerize_frozen_candidates_v1_1(
    frozen_candidates: list[dict],
    raw_candidates: list[dict],
    evidence: dict,
    local_pitch_px: float,
) -> tuple[list[dict], list[dict]]:
    raw_by_seed = {
        item["seed_x_roi"]: item for item in raw_candidates
    }
    transformed = []
    mappings = []
    for frozen_candidate in frozen_candidates:
        group_seeds = [
            frozen_candidate["seed_x_roi"],
            *[
                item["seed_x_roi"]
                for item in frozen_candidate.get(
                    "suppressed_candidates",
                    [],
                )
            ],
        ]
        group_paths = [
            raw_by_seed[seed]
            for seed in group_seeds
            if seed in raw_by_seed
        ]
        group_evidence = v1._plateau_group_evidence(  # noqa: SLF001
            group_paths,
            evidence,
            local_pitch_px,
            v1.DEFAULT_CORRECTION_CONFIG,
        )
        if group_evidence["eligible"]:
            candidate = _centerized_candidate_v1_1(
                frozen_candidate,
                group_paths,
                evidence,
                local_pitch_px,
                group_evidence,
            )
        else:
            candidate = copy.deepcopy(frozen_candidate)
            candidate["plateau_centerization"] = {
                "attempted": len(group_paths) >= 2,
                "adopted": False,
                "reason": group_evidence["reason"],
                "local_pitch_px": local_pitch_px,
                "old_seed_x_roi": group_seeds,
                "group_evidence": group_evidence,
                "evidence_revision": "frozen_non_plateau_path",
            }
        transformed.append(candidate)
        mappings.append(
            {
                "candidate_id": frozen_candidate["candidate_id"],
                "old_seed_x_roi": group_seeds,
                "new_seed_x_roi": candidate["seed_x_roi"],
                "adopted": bool(
                    candidate["plateau_centerization"]["adopted"]
                ),
                "reason": candidate[
                    "plateau_centerization"
                ]["reason"],
                "plateau_support": candidate[
                    "plateau_centerization"
                ].get("plateau_support"),
            }
        )
    return transformed, mappings


def _plateau_reference_classification(
    candidate: dict,
    reference_x_roi: float,
) -> dict:
    centerization = candidate.get("plateau_centerization", {})
    if not centerization.get("adopted"):
        return {
            "candidate_id": candidate["candidate_id"],
            "classification": "not_plateau",
        }
    bands = centerization["band_geometry"]
    valid = [item for item in bands if item["valid"]]
    if not valid:
        return {
            "candidate_id": candidate["candidate_id"],
            "classification": "ambiguous",
            "reason": "plateau_geometry_no_valid_bands",
        }
    core_flags = np.asarray(
        [
            item["core_left_x_roi"]
            <= reference_x_roi
            <= item["core_right_x_roi"]
            for item in valid
        ],
        dtype=bool,
    )
    plateau_flags = np.asarray(
        [
            item["left_edge_x_roi"]
            <= reference_x_roi
            <= item["right_edge_x_roi"]
            for item in valid
        ],
        dtype=bool,
    )
    core_fraction = float(np.mean(core_flags))
    plateau_fraction = float(np.mean(plateau_flags))
    minimum = (
        v1.DEFAULT_CORRECTION_CONFIG
        .plateau_pair_minimum_band_fraction
    )
    support = centerization["plateau_support"]
    geometry_stable = bool(
        support["center_stability_fraction"] >= minimum
        and support["width_stability_fraction"] >= minimum
        and centerization["geometry_summary"][
            "valid_band_fraction"
        ]
        >= minimum
    )
    if geometry_stable and core_fraction >= minimum:
        classification = "definite_on_separator"
        reason = "reference_inside_stable_plateau_core"
    elif plateau_fraction >= minimum:
        classification = "ambiguous"
        reason = "reference_in_plateau_edge_uncertainty"
    else:
        classification = "outside_plateau"
        reason = "reference_outside_plateau_extent"
    return {
        "candidate_id": candidate["candidate_id"],
        "classification": classification,
        "reason": reason,
        "valid_band_count": len(valid),
        "valid_band_fraction": (
            len(valid) / len(bands) if bands else 0.0
        ),
        "core_band_fraction": core_fraction,
        "plateau_band_fraction": plateau_fraction,
        "geometry_stable": geometry_stable,
        "geometry_summary": centerization["geometry_summary"],
        "frozen_center_distance": abs(
            v1._path_x_at_y(  # noqa: SLF001
                candidate,
                float(np.median(candidate["band_centers_y_roi"])),
            )
            - reference_x_roi
        ),
    }


def select_clicked_basin_paths_v1_1(
    candidates: list[dict],
    reference_x_roi: float,
    reference_y_roi: float,
    evidence: dict,
) -> dict:
    """Apply width-aware relation only to confirmed plateau separators."""

    frozen = separator.select_clicked_basin_paths(
        candidates,
        reference_x_roi,
        reference_y_roi,
        evidence,
        separator.DEFAULT_CONFIG,
    )
    accepted = [item for item in candidates if item["accepted"]]
    classifications = [
        _plateau_reference_classification(item, reference_x_roi)
        for item in accepted
        if item.get("plateau_centerization", {}).get("adopted")
    ]
    definite = [
        item
        for item in classifications
        if item["classification"] == "definite_on_separator"
    ]
    ambiguous = [
        item
        for item in classifications
        if item["classification"] == "ambiguous"
    ]
    width_debug = {
        "algorithm_revision": "width_aware_plateau_reference_v1_1",
        "classifications": classifications,
        "definite_candidate_ids": [
            item["candidate_id"] for item in definite
        ],
        "ambiguous_candidate_ids": [
            item["candidate_id"] for item in ambiguous
        ],
    }
    if frozen["crossing"]:
        return {**frozen, "width_aware_reference": width_debug}
    if len(definite) > 1 or (definite and ambiguous):
        return {
            **frozen,
            "status": "unavailable",
            "unavailable_reason": "ambiguous_reference_relationship",
            "relationship_status": "not_unique",
            "selection": {},
            "competing_explanations": [
                *[
                    {
                        "type": (
                            "reference_on_separator"
                            if item in definite
                            else "reference_relationship_uncertain"
                        ),
                        "candidate_id": item["candidate_id"],
                        "width_aware": True,
                        "evidence": item,
                    }
                    for item in [*definite, *ambiguous]
                ],
                *[
                    item
                    for item in frozen["competing_explanations"]
                    if item.get("type") == "clicked_basin_roles"
                ],
            ],
            "rejection_reasons": sorted(
                set(
                    frozen["rejection_reasons"]
                    + ["ambiguous_reference_relationship"]
                )
            ),
            "width_aware_reference": width_debug,
        }
    if len(definite) == 1:
        relation = definite[0]
        role_explanations = [
            item
            for item in frozen["competing_explanations"]
            if item.get("type") == "clicked_basin_roles"
        ]
        role_rejections = sorted(
            {
                reason
                for hypothesis in frozen["role_hypotheses"]
                for reason in hypothesis["rejection_reasons"]
            }
        )
        return {
            **frozen,
            "status": "unavailable",
            "unavailable_reason": "reference_on_separator",
            "relationship_status": "reference_on_separator",
            "selection": {},
            "competing_explanations": [
                {
                    "type": "reference_on_separator",
                    "candidate_id": relation["candidate_id"],
                    "width_aware": True,
                    "evidence": relation,
                },
                *role_explanations,
            ],
            "rejection_reasons": sorted(
                set(["reference_on_separator", *role_rejections])
            ),
            "width_aware_reference": width_debug,
        }
    if ambiguous:
        return {
            **frozen,
            "status": "unavailable",
            "unavailable_reason": "ambiguous_reference_relationship",
            "relationship_status": "not_unique",
            "selection": {},
            "competing_explanations": [
                *[
                    {
                        "type": "reference_relationship_uncertain",
                        "candidate_id": item["candidate_id"],
                        "width_aware": True,
                        "evidence": item,
                    }
                    for item in ambiguous
                ],
                *[
                    item
                    for item in frozen["competing_explanations"]
                    if item.get("type") == "clicked_basin_roles"
                ],
            ],
            "rejection_reasons": sorted(
                set(
                    frozen["rejection_reasons"]
                    + ["ambiguous_reference_relationship"]
                )
            ),
            "width_aware_reference": width_debug,
        }
    return {**frozen, "width_aware_reference": width_debug}


def detect_centerized_separator_paths_v1_1(
    image_gray: np.ndarray,
    reference_global: dict,
    roi_bounds_global: dict,
    direction: str = "vertical",
) -> dict:
    frozen = FROZEN_DETECT_SEPARATOR_PATHS(
        image_gray,
        reference_global,
        roi_bounds_global,
        direction,
        separator.DEFAULT_CONFIG,
    )
    roi_gray = separator.extract_raw_roi(
        image_gray,
        roi_bounds_global,
    )
    directional = separator._directional_roi(  # noqa: SLF001
        roi_gray,
        direction,
    )
    evidence = separator.build_raw_gray_response(
        directional,
        separator.DEFAULT_CONFIG,
    )
    raw_candidates = v1._raw_candidates_before_dedup(evidence)  # noqa: SLF001
    pitch_result = raw_pitch.estimate_raw_local_pitch_v3(
        image_gray,
        reference_global,
        roi_bounds_global,
        direction,
        raw_pitch.DEFAULT_CONFIG,
    )
    local_pitch = v1._local_pitch_scale(  # noqa: SLF001
        pitch_result,
        frozen["candidates"],
        frozen["reference_y_roi"],
        directional.shape[1],
    )
    candidates, mappings = centerize_frozen_candidates_v1_1(
        frozen["candidates"],
        raw_candidates,
        evidence,
        local_pitch,
    )
    arbitration = select_clicked_basin_paths_v1_1(
        candidates,
        frozen["reference_x_roi"],
        frozen["reference_y_roi"],
        evidence,
    )
    return {
        **frozen,
        "algorithm_revision": ALGORITHM_REVISION,
        "configuration_checksum": configuration_checksum(),
        "status": arbitration["status"],
        "unavailable_reason": arbitration["unavailable_reason"],
        "crossing": arbitration["crossing"],
        "selection": arbitration["selection"],
        "arbitration_debug": {
            key: value
            for key, value in arbitration.items()
            if key not in {"status", "unavailable_reason", "selection"}
        },
        "candidates": candidates,
        "raw_gray_evidence": {
            **frozen["raw_gray_evidence"],
            "accepted_candidate_count": sum(
                item["accepted"] for item in candidates
            ),
        },
        "plateau_centerization": {
            "algorithm_revision": ALGORITHM_REVISION,
            "configuration_checksum": configuration_checksum(),
            "local_pitch_px": local_pitch,
            "raw_pitch_provenance": {
                key: pitch_result.get(key)
                for key in (
                    "algorithm_revision",
                    "configuration_checksum",
                    "diagnostic_pitch_px",
                    "confidence",
                    "success_eligible",
                    "harmonic_ambiguity",
                )
            },
            "mapping": mappings,
            "adopted_count": sum(
                item["adopted"] for item in mappings
            ),
        },
    }


def run_joint_case_v1_1(
    image_gray: np.ndarray,
    reference_global: dict,
    roi_bounds_global: dict,
    direction: str = "vertical",
) -> dict:
    """Run the frozen joint chain with v1.1 separator semantics."""

    frozen_result = FROZEN_RUN_JOINT_CASE(
        image_gray,
        reference_global,
        roi_bounds_global,
        direction,
        joint.DEFAULT_CONFIG,
    )
    separator_result = detect_centerized_separator_paths_v1_1(
        image_gray,
        reference_global,
        roi_bounds_global,
        direction,
    )
    raw_roi = separator.extract_raw_roi(
        image_gray,
        roi_bounds_global,
    )
    graph = joint.build_ordered_basin_graph(
        separator_result,
        raw_roi,
        direction,
    )
    relation = joint.resolve_reference_relation(
        separator_result,
        graph,
    )
    pitch_result = raw_pitch.estimate_raw_local_pitch_v3(
        image_gray,
        reference_global,
        roi_bounds_global,
        direction,
        raw_pitch.DEFAULT_CONFIG,
    )
    debug = {
        "separator_result": separator_result,
        "basin_graph": graph,
        "reference_relation": relation,
        "raw_pitch_result": pitch_result,
    }
    base = {
        "algorithm_revision": ALGORITHM_REVISION,
        "configuration_checksum": configuration_checksum(),
        "dependency_checks": joint.verify_frozen_dependencies(),
        "debug": debug,
        "experiment_only": True,
    }
    if relation["status"] != "unique":
        return {
            **base,
            "status": "unavailable",
            "success": False,
            "unavailable_reason": relation["unavailable_reason"],
            "final_hypothesis": None,
        }
    if len(relation["hypotheses"]) != 1:
        return {
            **base,
            "status": "unavailable",
            "success": False,
            "unavailable_reason": "joint_hypothesis_not_unique",
            "final_hypothesis": None,
        }
    relation_hypothesis = relation["hypotheses"][0]
    geometry = joint._basin_geometry(  # noqa: SLF001
        relation_hypothesis,
        graph,
        separator_result["reference_x_roi"],
        roi_bounds_global["x0"],
    )
    pitch_evidence = joint._pitch_evidence_for_geometry(  # noqa: SLF001
        pitch_result,
        geometry,
        joint.DEFAULT_CONFIG,
    )
    hypothesis = {
        **relation_hypothesis,
        "geometry": geometry,
        "pitch_evidence": pitch_evidence,
        "provenance": {
            "separator_algorithm_revision": ALGORITHM_REVISION,
            "separator_configuration_checksum": configuration_checksum(),
            "raw_pitch_algorithm_revision": pitch_result[
                "algorithm_revision"
            ],
            "raw_pitch_configuration_checksum": pitch_result[
                "configuration_checksum"
            ],
        },
        "atomic": True,
    }
    debug["joint_hypotheses"] = [hypothesis]
    if pitch_evidence["rejects_geometry"]:
        return {
            **base,
            "status": "unavailable",
            "success": False,
            "unavailable_reason": "high_pitch_geometry_conflict",
            "final_hypothesis": None,
        }
    if not frozen_result["success"]:
        debug["release_unavailable_audit"] = {
            "reason": "release_unavailable_cannot_become_success",
            "release_unavailable_reason": frozen_result[
                "unavailable_reason"
            ],
            "hard_veto_applied": False,
            "accepted_hypothesis": hypothesis,
        }
    return {
        **base,
        "status": "available",
        "success": True,
        "unavailable_reason": None,
        "final_hypothesis": hypothesis,
    }


def refine_success_geometry(
    result: dict,
    image_gray: np.ndarray,
    roi_bounds_global: dict,
    direction: str,
) -> dict:
    return v1.refine_accepted_basin_centers(
        result,
        image_gray,
        roi_bounds_global,
        direction,
        v1.DEFAULT_CORRECTION_CONFIG,
    )


def _evaluate_development_v1_1(
    annotation: dict,
    image_gray: np.ndarray,
) -> dict:
    def patched_separator(
        image,
        reference,
        bounds,
        direction="vertical",
        _config=None,
    ):
        return detect_centerized_separator_paths_v1_1(
            image,
            reference,
            bounds,
            direction,
        )

    def patched_joint(
        image,
        reference,
        bounds,
        direction="vertical",
        _config=None,
    ):
        return run_joint_case_v1_1(
            image,
            reference,
            bounds,
            direction,
        )

    with (
        patch.object(
            separator,
            "detect_separator_paths",
            side_effect=patched_separator,
        ),
        patch.object(
            joint,
            "run_joint_case",
            side_effect=patched_joint,
        ),
    ):
        return joint._evaluate_one(  # noqa: SLF001
            annotation,
            image_gray,
            joint.DEFAULT_CONFIG,
        )


def _load_image(
    images_dir: Path,
    image_name: str,
) -> np.ndarray:
    return separator.load_grayscale_image(images_dir / image_name)


def _compact_result(result: dict) -> dict:
    identity = v1._identity(result)  # noqa: SLF001
    separator_result = result["debug"]["separator_result"]
    plateau_centerization = separator_result.get(
        "plateau_centerization",
        {
            "algorithm_revision": "frozen_release_no_centerization",
            "configuration_checksum": None,
            "mapping": [],
            "adopted_count": 0,
        },
    )
    return {
        "identity": identity,
        "rejection_reason": result["unavailable_reason"],
        "separator_paths": separator_result["candidates"],
        "plateau_centerization": plateau_centerization,
        "reference_relationship": result["debug"][
            "reference_relation"
        ],
        "basin_graph": result["debug"]["basin_graph"],
        "final_hypothesis": result["final_hypothesis"],
        "raw_pitch_provenance": {
            key: result["debug"]["raw_pitch_result"].get(key)
            for key in (
                "algorithm_revision",
                "configuration_checksum",
                "diagnostic_pitch_px",
                "usable_pitch_px",
                "confidence",
                "success_eligible",
                "harmonic_ambiguity",
                "unavailable_reason",
            )
        },
        "safety_values": {
            "separator_parameters": vars(separator.DEFAULT_CONFIG),
            "joint_parameters": vars(joint.DEFAULT_CONFIG),
            "arbitration": separator_result["arbitration_debug"],
        },
    }


def _release_hashes_from_git() -> dict:
    freeze = json.loads(FREEZE_PATH.read_text())
    current_tree = v1._git_output(  # noqa: SLF001
        "rev-parse",
        f"{RELEASE_COMMIT}^{{tree}}",
    )
    return {
        "release_commit": RELEASE_COMMIT,
        "expected_tree": freeze["release_tree"],
        "actual_tree": current_tree,
        "tree_matches": current_tree == freeze["release_tree"],
        "file_sha256": freeze["release_file_sha256"],
    }


def verify_frozen_inputs() -> dict:
    freeze = json.loads(FREEZE_PATH.read_text())
    if freeze["experiment_start_commit"] != EXPERIMENT_START_COMMIT:
        raise ValueError("experiment start commit changed")
    if freeze["release_commit"] != RELEASE_COMMIT:
        raise ValueError("release commit changed")
    if freeze["separator_heldout_allowed"]:
        raise ValueError("separator held-out access is forbidden")
    if freeze["runtime_gt_allowed"]:
        raise ValueError("runtime GT access is forbidden")
    smoke_bytes = SMOKE_MANIFEST_PATH.read_bytes()
    smoke_checksum = hashlib.sha256(smoke_bytes).hexdigest()
    smoke = json.loads(smoke_bytes)
    if not smoke["registered_before_v1_1_execution"]:
        raise ValueError("smoke manifest is not preregistered")
    if smoke["selection_method"]["parameter_selection_allowed"]:
        raise ValueError("smoke points cannot select parameters")
    release = _release_hashes_from_git()
    if not release["tree_matches"]:
        raise ValueError("release tree changed")
    return {
        "freeze": freeze,
        "smoke_manifest_checksum": smoke_checksum,
        "release": release,
    }


def _result_for_case(
    runner,
    image_gray: np.ndarray,
    case: dict,
) -> tuple[dict, dict]:
    bounds = v1._case_roi(case, image_gray)  # noqa: SLF001
    result = runner(
        image_gray,
        case["reference_global"],
        bounds,
        case.get("direction", "vertical"),
    )
    refinement = v1.refine_accepted_basin_centers(
        result,
        image_gray,
        bounds,
        case.get("direction", "vertical"),
        v1.DEFAULT_CORRECTION_CONFIG,
    )
    return result, refinement


def _panel_with_title(
    image: np.ndarray,
    title: str,
    reference_roi: tuple[int, int],
) -> np.ndarray:
    panel = np.full(
        (image.shape[0] + 30, image.shape[1], 3),
        245,
        dtype=np.uint8,
    )
    panel[30:] = image
    cv2.putText(
        panel,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    cv2.drawMarker(
        panel,
        (reference_roi[0], reference_roi[1] + 30),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        17,
        2,
    )
    return panel


def save_four_version_overlay(
    image_gray: np.ndarray,
    case: dict,
    release_result: dict,
    v1_result: dict,
    v1_refinement: dict,
    v1_1_result: dict,
    v1_1_refinement: dict,
    output_path: Path,
) -> None:
    bounds = v1._case_roi(case, image_gray)  # noqa: SLF001
    roi = separator.extract_raw_roi(image_gray, bounds)
    reference = case["reference_global"]
    reference_roi = (
        reference["x"] - bounds["x0"],
        reference["y"] - bounds["y0"],
    )
    panels = [
        _panel_with_title(
            v1._draw_result_geometry(roi, release_result),  # noqa: SLF001
            "Release baseline",
            reference_roi,
        ),
        _panel_with_title(
            v1._draw_result_geometry(  # noqa: SLF001
                roi,
                v1_result,
                v1_refinement if v1_refinement["attempted"] else None,
            ),
            "Cross-image v1 combined",
            reference_roi,
        ),
        _panel_with_title(
            v1._draw_result_geometry(roi, v1_1_result),  # noqa: SLF001
            "v1.1 plateau semantics",
            reference_roi,
        ),
        _panel_with_title(
            v1._draw_result_geometry(  # noqa: SLF001
                roi,
                v1_1_result,
                (
                    v1_1_refinement
                    if v1_1_refinement["attempted"]
                    else None
                ),
            ),
            "v1.1 + basin refinement",
            reference_roi,
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), np.hstack(panels)):
        raise OSError(f"could not write {output_path}")


def _identity_equal_physical(first: dict, second: dict) -> bool:
    return (
        first["success"] == second["success"]
        and first["basin_ids"] == second["basin_ids"]
        and first["separator_sequence"]
        == second["separator_sequence"]
    )


def _valley_metrics_from_refinement(
    refinement: dict,
    version: str,
) -> dict | None:
    if not refinement["attempted"]:
        return None
    key = (
        "baseline_minimum_error"
        if version == "plateau"
        else "refined_minimum_error"
    )
    sides = {}
    for role, item in refinement["basins"].items():
        sides[role] = {
            "median_error_px": item[f"{key}_median_px"],
            "max_error_px": item[f"{key}_max_px"],
            "adopted": item["adopted"],
            "shift_px": item["median_shift_px"],
        }
    medians = [
        item["median_error_px"]
        for item in sides.values()
        if item["median_error_px"] is not None
    ]
    maxima = [
        item["max_error_px"]
        for item in sides.values()
        if item["max_error_px"] is not None
    ]
    if not medians:
        return None
    return {
        "median_error_px": float(np.median(medians)),
        "max_error_px": float(np.max(maxima)),
        "per_side": sides,
    }


def _group_lookup(samples: list[dict]) -> dict[str, dict]:
    return {item["sample_id"]: item for item in samples}


def _development_metrics(
    annotations: dict[str, dict],
    image_cache: dict[str, np.ndarray],
) -> dict:
    evaluations = [
        _evaluate_development_v1_1(
            annotation,
            image_cache[annotation["image_name"]],
        )
        for annotation in annotations.values()
    ]
    return joint._summarize(evaluations)  # noqa: SLF001


def _sample_record(
    case: dict,
    image_gray: np.ndarray,
    release_baseline: dict,
    v1_saved: dict,
) -> tuple[dict, Path | None]:
    bounds = v1._case_roi(case, image_gray)  # noqa: SLF001
    release_result, _ = _result_for_case(
        lambda image, reference, roi, direction: (
            FROZEN_RUN_JOINT_CASE(
                image,
                reference,
                roi,
                direction,
                joint.DEFAULT_CONFIG,
            )
        ),
        image_gray,
        case,
    )
    v1_result, v1_refinement = _result_for_case(
        lambda image, reference, roi, direction: (
            V1_RUN_JOINT_CASE(
                image,
                reference,
                roi,
                direction,
                v1.DEFAULT_CORRECTION_CONFIG,
            )
        ),
        image_gray,
        case,
    )
    result, refinement = _result_for_case(
        run_joint_case_v1_1,
        image_gray,
        case,
    )
    release_identity = v1._identity(release_result)  # noqa: SLF001
    v1_identity = v1._identity(v1_result)  # noqa: SLF001
    result_identity = v1._identity(result)  # noqa: SLF001
    return {
        "sample_id": case["sample_id"],
        "image_name": case["image_name"],
        "reference_global": case["reference_global"],
        "roi_bounds_global": bounds,
        "release_baseline": {
            "saved_identity": release_baseline["identity"],
            "replayed": _compact_result(release_result),
            "replay_matches_saved": (
                release_identity == release_baseline["identity"]
            ),
        },
        "cross_image_v1": {
            "saved_identity": v1_saved["plateau_only_identity"],
            "saved_combined_identity": v1_saved["combined_identity"],
            "replayed": _compact_result(v1_result),
            "center_refinement": v1_refinement,
            "replay_matches_saved": (
                v1_identity == v1_saved["plateau_only_identity"]
            ),
        },
        "v1_1_plateau_semantics": _compact_result(result),
        "v1_1_combined": {
            "identity": refinement["identity_after"],
            "center_refinement": refinement,
        },
        "changes": {
            "release_to_v1": {
                "identity_changed": (
                    release_identity != v1_identity
                ),
            },
            "v1_to_v1_1": {
                "identity_changed": (
                    v1_identity != result_identity
                ),
            },
            "refinement_changed_identity": (
                refinement["identity_before"]
                != refinement["identity_after"]
            ),
        },
    }, None


def _fixed_regression_gates(
    records: dict[str, list[dict]],
    development_metrics: dict,
    baseline_report: dict,
    v1_report: dict,
) -> dict:
    issue_records = records["sample2_issues"]
    random_records = records["sample2_random"]
    cross = _group_lookup(records["cross_image"])
    development = _group_lookup(records["development"])

    def release_identity(item):
        return item["release_baseline"]["replayed"]["identity"]

    def result_identity(item):
        return item["v1_1_plateau_semantics"]["identity"]

    sample2_issue_changes = [
        item["sample_id"]
        for item in issue_records
        if not _identity_equal_physical(
            release_identity(item),
            result_identity(item),
        )
    ]
    sample2_random_changes = [
        item["sample_id"]
        for item in random_records
        if not _identity_equal_physical(
            release_identity(item),
            result_identity(item),
        )
    ]
    sample2_adoptions = [
        item["sample_id"]
        for item in [*issue_records, *random_records]
        if item["v1_1_plateau_semantics"][
            "plateau_centerization"
        ]["adopted_count"]
    ]
    cross_release_upgrades = [
        item["sample_id"]
        for item in records["cross_image"]
        if (
            not release_identity(item)["success"]
            and result_identity(item)["success"]
        )
    ]
    refinement_identity_changes = [
        item["sample_id"]
        for group in records.values()
        for item in group
        if item["changes"]["refinement_changed_identity"]
    ]
    stripe5 = cross["X002"]
    stripe1 = cross["X003"]
    stripe5_v1 = _valley_metrics_from_refinement(
        stripe5["cross_image_v1"]["center_refinement"],
        "combined",
    )
    stripe5_v1_1 = _valley_metrics_from_refinement(
        stripe5["v1_1_combined"]["center_refinement"],
        "combined",
    )
    stripe1_v1 = _valley_metrics_from_refinement(
        stripe1["cross_image_v1"]["center_refinement"],
        "combined",
    )
    stripe1_v1_1 = _valley_metrics_from_refinement(
        stripe1["v1_1_combined"]["center_refinement"],
        "combined",
    )
    d016 = development["D016"]
    d017 = development["D017"]
    d016_result = d016["v1_1_plateau_semantics"]
    d016_selected = (
        d016_result["safety_values"]["arbitration"]
        .get("role_hypotheses", [])
    )
    d016_support_rejection = any(
        "role_path_insufficient_vertical_support"
        in item["rejection_reasons"]
        for item in d016_selected
    )
    d016_plateau_support = [
        mapping["plateau_support"]
        for mapping in d016_result["plateau_centerization"]["mapping"]
        if mapping["adopted"]
    ]
    d017_width = (
        d017_result := d017["v1_1_plateau_semantics"]
    )["safety_values"]["arbitration"].get(
        "width_aware_reference",
        {},
    )
    d017_definite = d017_width.get("definite_candidate_ids", [])
    stripe9_10_11 = [
        cross[sample_id]
        for sample_id in ("X004", "X005", "X006", "X007", "X008")
    ]
    checks = {
        "development_wrong_success_zero": (
            development_metrics["wrong_success"] == 0
        ),
        "development_false_split_zero": (
            development_metrics["false_split"] == 0
        ),
        "development_ambiguous_not_released": not (
            development_metrics["ambiguous_formally_available"]
        ),
        "development_unavailable_not_released": not (
            development_metrics["unavailable_formally_available"]
        ),
        "development_crossing_order_not_released": not (
            development_metrics[
                "crossing_or_order_conflict_formally_available"
            ]
        ),
        "refinement_identity_unchanged": not (
            refinement_identity_changes
        ),
        "sample2_issue_physical_identity_unchanged": not (
            sample2_issue_changes
        ),
        "sample2_random_physical_identity_unchanged": not (
            sample2_random_changes
        ),
        "sample2_no_plateau_adoption": not sample2_adoptions,
        "cross_image_release_unavailable_not_upgraded": not (
            cross_release_upgrades
        ),
        "stripe5_plateau_centerized": (
            stripe5["v1_1_plateau_semantics"][
                "plateau_centerization"
            ]["adopted_count"]
            > 0
        ),
        "stripe5_basin_identity_preserved": (
            release_identity(stripe5)["basin_ids"]
            == result_identity(stripe5)["basin_ids"]
        ),
        "stripe5_valley_error_not_worse": bool(
            stripe5_v1
            and stripe5_v1_1
            and stripe5_v1_1["median_error_px"]
            <= stripe5_v1["median_error_px"] + 1e-9
            and stripe5_v1_1["max_error_px"]
            <= stripe5_v1["max_error_px"] + 1e-9
        ),
        "stripe5_other_point_stays_unavailable": not (
            result_identity(cross["X001"])["success"]
        ),
        "stripe1_identity_preserved": _identity_equal_physical(
            release_identity(stripe1),
            result_identity(stripe1),
        ),
        "stripe1_error_not_worse": bool(
            stripe1_v1
            and stripe1_v1_1
            and stripe1_v1_1["median_error_px"]
            <= stripe1_v1["median_error_px"] + 1e-9
            and stripe1_v1_1["max_error_px"]
            <= stripe1_v1["max_error_px"] + 1e-9
        ),
        "d016_uses_plateau_support": bool(d016_plateau_support),
        "d016_not_rejected_by_center_sampled_support": not (
            d016_support_rejection
        ),
        "d017_width_aware_core_only": bool(
            d017_definite
            and d017_result["identity"]["success"]
        ),
        "stripe9_10_11_safe_failures_preserved": all(
            not result_identity(item)["success"]
            for item in stripe9_10_11
        ),
        "release_replays_match_saved": all(
            item["release_baseline"]["replay_matches_saved"]
            for group in records.values()
            for item in group
        ),
        "v1_replays_match_saved": all(
            item["cross_image_v1"]["replay_matches_saved"]
            for group in records.values()
            for item in group
        ),
    }
    return {
        "all_passed": all(checks.values()),
        "checks": checks,
        "sample2_issue_changes": sample2_issue_changes,
        "sample2_random_changes": sample2_random_changes,
        "sample2_adoptions": sample2_adoptions,
        "cross_image_release_upgrades": cross_release_upgrades,
        "refinement_identity_changes": refinement_identity_changes,
        "stripe5_metrics": {
            "v1_combined": stripe5_v1,
            "v1_1_combined": stripe5_v1_1,
        },
        "stripe1_metrics": {
            "v1_combined": stripe1_v1,
            "v1_1_combined": stripe1_v1_1,
        },
        "d016_plateau_support": d016_plateau_support,
        "d017_width_aware_reference": d017_width,
        "development_comparison": {
            "release": baseline_report["development_metrics"],
            "v1": v1_report["versions"]["combined"][
                "development_metrics"
            ],
            "v1_1": development_metrics,
        },
    }


def run_fixed_regression(
    images_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    frozen = verify_frozen_inputs()
    v1_output = PROJECT_ROOT / "outputs" / "cross_image_correction_v1"
    baseline_path = v1_output / "baseline_report.json"
    v1_report_path = v1_output / "three_version_report.json"
    if not baseline_path.is_file() or not v1_report_path.is_file():
        raise FileNotFoundError(
            "run the frozen v1 baseline and experiment reports first"
        )
    baseline_report = json.loads(baseline_path.read_text())
    v1_report = json.loads(v1_report_path.read_text())
    manifest = v1.load_evaluation_manifest()
    development = v1.load_development_annotations()
    grouped_cases = v1._group_cases(manifest, development)  # noqa: SLF001
    baseline_groups = {
        name: _group_lookup(items)
        for name, items in baseline_report["groups"].items()
    }
    v1_groups = {
        name: _group_lookup(items)
        for name, items in v1_report["versions"][
            "plateau_only"
        ]["groups"].items()
    }
    image_cache: dict[str, np.ndarray] = {}
    records: dict[str, list[dict]] = {}
    for group_name, cases in grouped_cases.items():
        group_records = []
        overlay_paths = []
        for case in cases:
            image_name = case["image_name"]
            if image_name not in image_cache:
                image_cache[image_name] = _load_image(
                    images_dir,
                    image_name,
                )
            image = image_cache[image_name]
            record, _ = _sample_record(
                case,
                image,
                baseline_groups[group_name][case["sample_id"]],
                v1_groups[group_name][case["sample_id"]],
            )
            group_records.append(record)

            release_result, _ = _result_for_case(
                lambda gray, reference, bounds, direction: (
                    FROZEN_RUN_JOINT_CASE(
                        gray,
                        reference,
                        bounds,
                        direction,
                        joint.DEFAULT_CONFIG,
                    )
                ),
                image,
                case,
            )
            v1_result, v1_refinement = _result_for_case(
                lambda gray, reference, bounds, direction: (
                    V1_RUN_JOINT_CASE(
                        gray,
                        reference,
                        bounds,
                        direction,
                        v1.DEFAULT_CORRECTION_CONFIG,
                    )
                ),
                image,
                case,
            )
            v1_1_result, v1_1_refinement = _result_for_case(
                run_joint_case_v1_1,
                image,
                case,
            )
            overlay_path = (
                output_dir
                / "fixed_regression_overlays"
                / group_name
                / f"{case['sample_id']}.png"
            )
            save_four_version_overlay(
                image,
                case,
                release_result,
                v1_result,
                v1_refinement,
                v1_1_result,
                v1_1_refinement,
                overlay_path,
            )
            overlay_paths.append(overlay_path)
        records[group_name] = group_records
        v1.save_contact_sheet(
            overlay_paths,
            output_dir
            / "fixed_regression_overlays"
            / f"{group_name}_contact_sheet.png",
            columns=2,
        )

    development_metrics = _development_metrics(
        development,
        image_cache,
    )
    gates = _fixed_regression_gates(
        records,
        development_metrics,
        baseline_report,
        v1_report,
    )
    report = {
        "report_revision": (
            "cross_image_correction_v1_1_fixed_regression_v1"
        ),
        "algorithm_revision": ALGORITHM_REVISION,
        "configuration": configuration_document(),
        "configuration_checksum": configuration_checksum(),
        "implementation_round": 1,
        "frozen_inputs": frozen,
        "heldout_read": False,
        "production_runtime_modified": False,
        "formal_ui_modified": False,
        "legacy_result_role": "safety_block_only_no_fallback",
        "versions": [
            "release_baseline",
            "cross_image_correction_v1_combined",
            "v1_1_plateau_semantics",
            "v1_1_plateau_semantics_plus_basin_refinement",
        ],
        "groups": records,
        "development_metrics": development_metrics,
        "safety_gates": gates,
        **v1.git_provenance(),
    }
    v1.atomic_write_json(
        output_dir / "fixed_regression_report.json",
        report,
    )
    return report


def _draw_smoke_overlay(
    image_gray: np.ndarray,
    case: dict,
    result: dict,
    refinement: dict,
) -> np.ndarray:
    bounds = v1._case_roi(case, image_gray)  # noqa: SLF001
    roi = separator.extract_raw_roi(image_gray, bounds)
    overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    separator_result = result["debug"]["separator_result"]
    for candidate in separator_result["candidates"]:
        centerization = candidate.get("plateau_centerization", {})
        for source_path in centerization.get(
            "old_path_x_by_band_roi",
            [],
        ):
            v1._draw_path(  # noqa: SLF001
                overlay,
                {
                    "x_by_band_roi": source_path,
                    "band_centers_y_roi": candidate[
                        "band_centers_y_roi"
                    ],
                },
                (0, 128, 255),
                1,
            )
        if centerization.get("adopted"):
            for edge_key, color in (
                ("left_edge_x_roi", (255, 180, 0)),
                ("right_edge_x_roi", (255, 180, 0)),
            ):
                v1._draw_path(  # noqa: SLF001
                    overlay,
                    {
                        "x_by_band_roi": [
                            item[edge_key]
                            for item in centerization[
                                "band_geometry"
                            ]
                        ],
                        "band_centers_y_roi": candidate[
                            "band_centers_y_roi"
                        ],
                    },
                    color,
                    1,
                )
        v1._draw_path(  # noqa: SLF001
            overlay,
            candidate,
            (0, 210, 0) if candidate["accepted"] else (0, 0, 200),
            1,
        )
    hypothesis = result["final_hypothesis"]
    if hypothesis is not None:
        basin_by_id = {
            item["basin_id"]: item
            for item in result["debug"]["basin_graph"][
                "verified_basins"
            ]
        }
        for role, color in (
            ("left", (255, 0, 0)),
            ("right", (0, 255, 255)),
        ):
            basin = basin_by_id[hypothesis["basin_ids"][role]]
            v1._draw_path(  # noqa: SLF001
                overlay,
                {
                    "x_by_band_roi": basin[
                        "center_x_by_band_roi"
                    ],
                    "band_centers_y_roi": basin[
                        "band_centers_y_roi"
                    ],
                },
                color,
                2,
            )
            if refinement["attempted"]:
                detail = refinement["basins"][role]
                v1._draw_path(  # noqa: SLF001
                    overlay,
                    {
                        "x_by_band_roi": detail[
                            "bandwise_center_path_x_roi"
                        ],
                        "band_centers_y_roi": basin[
                            "band_centers_y_roi"
                        ],
                    },
                    (255, 0, 255),
                    2,
                )
                v1._draw_path(  # noqa: SLF001
                    overlay,
                    {
                        "x_by_band_roi": [
                            item.get(
                                "gray_minimum_center_x_roi",
                                original,
                            )
                            for item, original in zip(
                                detail["bands"],
                                basin["center_x_by_band_roi"],
                            )
                        ],
                        "band_centers_y_roi": basin[
                            "band_centers_y_roi"
                        ],
                    },
                    (0, 80, 255),
                    1,
                )
    reference = case["reference_global"]
    cv2.drawMarker(
        overlay,
        (
            reference["x"] - bounds["x0"],
            reference["y"] - bounds["y0"],
        ),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        17,
        2,
    )
    label = (
        "success"
        if result["success"]
        else f"unavailable: {result['unavailable_reason']}"
    )
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 24), (245, 245, 245), -1)
    cv2.putText(
        overlay,
        label,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return overlay


def run_preregistered_smoke(
    images_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    frozen = verify_frozen_inputs()
    regression_path = output_dir / "fixed_regression_report.json"
    if not regression_path.is_file():
        raise FileNotFoundError("fixed regression report is required")
    regression = json.loads(regression_path.read_text())
    if not regression["safety_gates"]["all_passed"]:
        raise RuntimeError("fixed regression safety gates did not pass")
    smoke_manifest = json.loads(SMOKE_MANIFEST_PATH.read_text())
    rows = []
    details = []
    all_overlay_paths = []
    for image_entry in smoke_manifest["images"]:
        image = _load_image(images_dir, image_entry["image_name"])
        image_paths = []
        for point in image_entry["points"]:
            case = {
                **point,
                "image_name": image_entry["image_name"],
                "direction": "vertical",
            }
            bounds = v1._case_roi(case, image)  # noqa: SLF001
            result = run_joint_case_v1_1(
                image,
                case["reference_global"],
                bounds,
                "vertical",
            )
            refinement = refine_success_geometry(
                result,
                image,
                bounds,
                "vertical",
            )
            detail = {
                "sample_id": case["sample_id"],
                "image_name": case["image_name"],
                "reference_global": case["reference_global"],
                "roi_bounds_global": bounds,
                "result": _compact_result(result),
                "center_refinement": refinement,
                "human_gt_available": False,
                "correctness_claimed": False,
            }
            details.append(detail)
            geometry = (
                result["final_hypothesis"]["geometry"]
                if result["final_hypothesis"] is not None
                else {}
            )
            centers = geometry.get("basins", {})
            row = {
                "sample_id": case["sample_id"],
                "image_name": case["image_name"],
                "reference_x": case["reference_global"]["x"],
                "reference_y": case["reference_global"]["y"],
                "status": result["status"],
                "success": result["success"],
                "rejection_reason": (
                    result["unavailable_reason"] or ""
                ),
                "left_center_x_roi": (
                    centers.get("left", {}).get(
                        "center_x_at_reference_roi",
                        "",
                    )
                ),
                "right_center_x_roi": (
                    centers.get("right", {}).get(
                        "center_x_at_reference_roi",
                        "",
                    )
                ),
                "plateau_adopted_count": (
                    result["debug"]["separator_result"][
                        "plateau_centerization"
                    ]["adopted_count"]
                ),
                "raw_pitch_px": (
                    result["debug"]["raw_pitch_result"].get(
                        "diagnostic_pitch_px"
                    )
                    or ""
                ),
                "raw_pitch_confidence": (
                    result["debug"]["raw_pitch_result"]["confidence"]
                ),
            }
            rows.append(row)
            overlay = _draw_smoke_overlay(
                image,
                case,
                result,
                refinement,
            )
            overlay_path = (
                output_dir
                / "preregistered_smoke"
                / "overlays"
                / image_entry["image_name"].replace(".bmp", "")
                / f"{case['sample_id']}.png"
            )
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(overlay_path), overlay):
                raise OSError(f"could not write {overlay_path}")
            image_paths.append(overlay_path)
            all_overlay_paths.append(overlay_path)
            v1.atomic_write_json(
                output_dir
                / "preregistered_smoke"
                / "points"
                / f"{case['sample_id']}.json",
                detail,
            )
        v1.save_contact_sheet(
            image_paths,
            output_dir
            / "preregistered_smoke"
            / "contact_sheets"
            / f"{image_entry['image_name'].replace('.bmp', '')}.png",
            columns=3,
        )
    v1.save_contact_sheet(
        all_overlay_paths,
        output_dir
        / "preregistered_smoke"
        / "cross_image_contact_sheet.png",
        columns=3,
    )
    csv_path = (
        output_dir
        / "preregistered_smoke"
        / "preregistered_smoke.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "report_revision": (
            "cross_image_correction_v1_1_preregistered_smoke_v1"
        ),
        "algorithm_revision": ALGORITHM_REVISION,
        "configuration_checksum": configuration_checksum(),
        "manifest_path": str(SMOKE_MANIFEST_PATH),
        "manifest_checksum": frozen["smoke_manifest_checksum"],
        "point_count": len(details),
        "correctness_claimed": False,
        "heldout_read": False,
        "points": details,
        "csv_path": str(csv_path),
        "contact_sheet": str(
            output_dir
            / "preregistered_smoke"
            / "cross_image_contact_sheet.png"
        ),
        **v1.git_provenance(),
    }
    v1.atomic_write_json(
        output_dir
        / "preregistered_smoke"
        / "preregistered_smoke_report.json",
        report,
    )
    return report


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline cross-image correction v1.1",
    )
    parser.add_argument(
        "--phase",
        choices=("regression", "smoke"),
        required=True,
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    if arguments.phase == "regression":
        run_fixed_regression(
            arguments.images_dir,
            arguments.output_dir,
        )
    else:
        run_preregistered_smoke(
            arguments.images_dir,
            arguments.output_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
