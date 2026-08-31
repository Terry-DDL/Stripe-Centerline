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
from config import CONFIG, INTERACTIVE_CONFIG  # noqa: E402
from interactive_pipeline import _preprocess_roi  # noqa: E402
from tools.lightweight_profile import profiled, stage  # noqa: E402


EXPERIMENT_START_COMMIT = (
    "0f6300a0ab40dd68af7a6425f27cbb7ca9e1a484"
)
RELEASE_COMMIT = "55b26e6deac5acbb50e3934c03d71e3c5b56b269"
ALGORITHM_REVISION = (
    "cross_image_correction_offline_v1_1_local_contrast_recovery"
)
MINIMUM_RAW_GRAY_DYNAMIC = 8.0
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
ENABLE_REFERENCE_SEMANTIC_TIEBREAK = True
ENABLE_PITCH_GUIDED_MISSING_SEPARATOR_COMPLETION = True
ADAPTIVE_DIM_SEPARATOR_WHITE_FRACTION = 0.10


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
        "reference_grayscale_semantic_resolution": {
            "scope": "single_near_separator_and_single_basin_hypothesis",
            "plateau_source": "existing_raw_gray_ridge_interval",
            "core_source": "existing_edge_derived_plateau_core",
            "minimum_band_fraction": (
                v1.DEFAULT_CORRECTION_CONFIG
                .plateau_pair_minimum_band_fraction
            ),
            "distance_threshold_added": False,
        },
        "intermediate_dark_basin_veto": {
            "scope": "inside_basin_success_only",
            "gray_source": "raw_roi",
            "minimum_band_fraction": (
                separator.DEFAULT_CONFIG
                .minimum_dark_basin_band_fraction
            ),
            "minimum_row_fraction": (
                separator.DEFAULT_CONFIG
                .minimum_dark_basin_band_fraction
            ),
            "minimum_normalized_contrast": (
                separator.DEFAULT_CONFIG.minimum_dark_basin_contrast
            ),
            "maximum_mean_band_step_px": (
                separator.DEFAULT_CONFIG
                .maximum_role_path_mean_step_px
            ),
            "replacement_center_allowed": False,
        },
        "local_dark_valley_recovery": {
            "scope": "inside_basin_with_multiple_verified_clicked_valleys",
            "identity_source": "ordered_raw_gray_valley_paths",
            "center_source": "shared_dark_weighted_basin_center",
            "minimum_gray_dynamic": MINIMUM_RAW_GRAY_DYNAMIC,
            "minimum_band_fraction": (
                separator.DEFAULT_CONFIG
                .minimum_dark_basin_band_fraction
            ),
            "minimum_row_fraction": (
                separator.DEFAULT_CONFIG
                .minimum_dark_basin_band_fraction
            ),
            "maximum_mean_band_step_px": (
                separator.DEFAULT_CONFIG
                .maximum_role_path_mean_step_px
            ),
            "path_match_tolerance_px": (
                separator.DEFAULT_CONFIG.match_tolerance_px
            ),
            "coordinate_or_image_special_cases": False,
        },
        "local_contrast_coverage_recovery": {
            "scope": (
                "unique_clicked_basin_roles_rejected_only_by_"
                "dark_basin_sequence_not_verified"
            ),
            "identity_source": "four_consecutive_ordered_separator_paths",
            "local_profile_source": "median_rows_near_reference_y",
            "local_half_window_rows": int(
                round(separator.DEFAULT_CONFIG.match_tolerance_px)
            ),
            "minimum_local_normalized_contrast": (
                separator.DEFAULT_CONFIG.minimum_dark_basin_contrast
            ),
            "minimum_path_support": (
                separator.DEFAULT_CONFIG.minimum_role_path_support_fraction
            ),
            "maximum_path_mean_step_px": (
                separator.DEFAULT_CONFIG.maximum_role_path_mean_step_px
            ),
            "minimum_gap_width_consistency_ratio": (
                separator.DEFAULT_CONFIG.minimum_full_height_order_ratio
            ),
            "reference_core_guard_px": (
                separator.DEFAULT_CONFIG.reference_guard_px
            ),
            "pitch_requirement": "harmonically_safe_high_supports_geometry",
            "global_dark_basin_band_fraction_changed": False,
            "coordinate_or_image_special_cases": False,
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


def _reference_grayscale_semantic_classification(
    candidate: dict,
    candidates: list[dict],
    reference_x_roi: float,
    evidence: dict,
    local_pitch_px: float,
) -> dict:
    """Resolve a near-path reference from existing raw-gray band evidence."""

    accepted = sorted(
        [item for item in candidates if item["accepted"]],
        key=lambda item: float(np.median(item["x_by_band_roi"])),
    )
    candidate_index = accepted.index(candidate)
    candidate_median = float(np.median(candidate["x_by_band_roi"]))
    if reference_x_roi < candidate_median:
        neighbor_index = candidate_index - 1
    else:
        neighbor_index = candidate_index + 1
    if neighbor_index < 0 or neighbor_index >= len(accepted):
        return {
            "classification": "ambiguous",
            "reason": "adjacent_basin_boundary_missing",
            "candidate_id": candidate["candidate_id"],
            "bands": [],
        }
    neighbor = accepted[neighbor_index]

    band_results = []
    for band_index, profile in enumerate(evidence["profiles"]):
        ridge = v1._ridge_interval(  # noqa: SLF001
            np.asarray(profile, dtype=np.float64),
            candidate["x_by_band_roi"][band_index],
            local_pitch_px,
        )
        if ridge["status"] != "available":
            continue
        band_results.append(
            {
                "band_index": band_index,
                "left_edge_x_roi": ridge["left_x_roi"],
                "right_edge_x_roi": ridge["right_x_roi"],
                "center_x_roi": float(
                    candidate["x_by_band_roi"][band_index]
                ),
            }
        )
    if not band_results:
        return {
            "classification": "ambiguous",
            "reason": "local_plateau_geometry_unavailable",
            "candidate_id": candidate["candidate_id"],
            "bands": [],
        }

    # Reuse the existing edge-derived core construction. A single accepted
    # path supplies both observed source positions; no fixed core half-width
    # is introduced.
    source = {
        "x_by_band_roi": candidate["x_by_band_roi"],
    }
    enriched, geometry_summary = _plateau_band_geometry(
        band_results,
        source,
        source,
    )
    bands = []
    for item in enriched:
        band_index = item["band_index"]
        profile = np.asarray(
            evidence["profiles"][band_index],
            dtype=np.float64,
        )
        ref_index = int(
            np.clip(round(reference_x_roi), 0, profile.size - 1)
        )
        candidate_x = int(
            np.clip(
                round(candidate["x_by_band_roi"][band_index]),
                0,
                profile.size - 1,
            )
        )
        neighbor_x = int(
            np.clip(
                round(neighbor["x_by_band_roi"][band_index]),
                0,
                profile.size - 1,
            )
        )
        basin_left, basin_right = sorted(
            (candidate_x, neighbor_x)
        )
        basin_values = profile[basin_left + 1 : basin_right]
        if basin_values.size == 0:
            continue
        core_left = int(round(item["core_left_x_roi"]))
        core_right = int(round(item["core_right_x_roi"]))
        separator_gray = float(
            np.median(profile[core_left : core_right + 1])
        )
        basin_gray = float(np.min(basin_values))
        reference_gray = float(profile[ref_index])
        inside_plateau = bool(
            item["left_edge_x_roi"]
            <= reference_x_roi
            <= item["right_edge_x_roi"]
        )
        inside_core = bool(
            item["core_left_x_roi"]
            <= reference_x_roi
            <= item["core_right_x_roi"]
        )
        closer_to_separator = bool(
            abs(reference_gray - separator_gray)
            < abs(reference_gray - basin_gray)
        )
        closer_to_basin = bool(
            abs(reference_gray - basin_gray)
            < abs(reference_gray - separator_gray)
        )
        bands.append(
            {
                **item,
                "reference_gray": reference_gray,
                "separator_core_gray": separator_gray,
                "adjacent_basin_dark_gray": basin_gray,
                "reference_inside_plateau": inside_plateau,
                "reference_inside_core": inside_core,
                "reference_closer_to_separator": closer_to_separator,
                "reference_closer_to_basin": closer_to_basin,
            }
        )
    if not bands:
        return {
            "classification": "ambiguous",
            "reason": "adjacent_basin_gray_unavailable",
            "candidate_id": candidate["candidate_id"],
            "bands": [],
        }

    minimum = (
        v1.DEFAULT_CORRECTION_CONFIG
        .plateau_pair_minimum_band_fraction
    )
    on_separator_fraction = float(
        np.mean(
            [
                item["reference_inside_plateau"]
                and item["reference_inside_core"]
                and item["reference_closer_to_separator"]
                for item in bands
            ]
        )
    )
    inside_basin_fraction = float(
        np.mean(
            [
                not item["reference_inside_plateau"]
                and item["reference_closer_to_basin"]
                for item in bands
            ]
        )
    )
    if on_separator_fraction >= minimum:
        classification = "on_separator"
        reason = "stable_plateau_membership_and_separator_gray"
    elif inside_basin_fraction >= minimum:
        classification = "inside_basin"
        reason = "stable_plateau_exit_and_basin_gray"
    else:
        classification = "ambiguous"
        reason = "raw_gray_band_semantics_not_consistent"
    return {
        "algorithm_revision": "reference_grayscale_semantic_resolution",
        "candidate_id": candidate["candidate_id"],
        "classification": classification,
        "reason": reason,
        "minimum_band_fraction": minimum,
        "on_separator_band_fraction": on_separator_fraction,
        "inside_basin_band_fraction": inside_basin_fraction,
        "geometry_summary": geometry_summary,
        "bands": bands,
    }


def select_clicked_basin_paths_v1_1(
    candidates: list[dict],
    reference_x_roi: float,
    reference_y_roi: float,
    evidence: dict,
    local_pitch_px: float,
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
    relation_items = [
        item
        for item in frozen["competing_explanations"]
        if item.get("type")
        in {
            "reference_on_separator",
            "reference_relationship_uncertain",
        }
    ]
    role_hypotheses = frozen["role_hypotheses"]
    semantic = None
    if (
        not frozen["crossing"]
        and len(relation_items) == 1
        and len(role_hypotheses) == 1
    ):
        by_id = {item["candidate_id"]: item for item in accepted}
        relation_candidate = by_id[relation_items[0]["candidate_id"]]
        semantic = _reference_grayscale_semantic_classification(
            relation_candidate,
            candidates,
            reference_x_roi,
            evidence,
            local_pitch_px,
        )
        width_debug[
            "reference_grayscale_semantic_resolution"
        ] = semantic
        role = role_hypotheses[0]
        role_supports = [
            float(item["support_fraction"])
            for item in role.get("path_metrics", [])
            if item.get("support_fraction") is not None
        ]
        minimum_role_support = min(role_supports) if role_supports else 0.0
        semantic_relation = semantic["classification"]
        semantic_tiebreak = {
            "adopted": False,
            "source_classification": semantic_relation,
            "resolved_relation": semantic_relation,
            "minimum_role_support": minimum_role_support,
        }
        if (
            ENABLE_REFERENCE_SEMANTIC_TIEBREAK
            and semantic_relation == "ambiguous"
        ):
            on_fraction = float(
                semantic.get("on_separator_band_fraction", 0.0)
            )
            inside_fraction = float(
                semantic.get("inside_basin_band_fraction", 0.0)
            )
            if (
                on_fraction >= 0.50
                and inside_fraction < 0.50
                and minimum_role_support >= 0.80
                and role.get("verified") is True
            ):
                semantic_relation = "on_separator"
                semantic_tiebreak.update(
                    {
                        "adopted": True,
                        "resolved_relation": semantic_relation,
                        "reason": (
                            "ambiguous_semantic_with_local_separator_majority"
                        ),
                    }
                )
            elif (
                inside_fraction >= 0.50
                and on_fraction < 0.50
                and minimum_role_support >= 0.90
                and role.get("verified") is True
                and frozen.get("unavailable_reason")
                == "ambiguous_reference_relationship"
            ):
                semantic_relation = "inside_basin"
                semantic_tiebreak.update(
                    {
                        "adopted": True,
                        "resolved_relation": semantic_relation,
                        "reason": (
                            "ambiguous_semantic_with_local_basin_majority"
                        ),
                    }
                )
        width_debug["semantic_tiebreak"] = semantic_tiebreak
        if semantic_relation == "inside_basin":
            if not role["verified"]:
                return {
                    **frozen,
                    "status": "unavailable",
                    "unavailable_reason": role[
                        "rejection_reasons"
                    ][0],
                    "relationship_status": "basin_rejected",
                    "selection": {},
                    "competing_explanations": [
                        {
                            "type": "clicked_basin_roles",
                            "hypothesis_id": role["hypothesis_id"],
                            "selection_candidate_ids": role[
                                "selection_candidate_ids"
                            ],
                            "verified": False,
                            "rejection_reasons": role[
                                "rejection_reasons"
                            ],
                        }
                    ],
                    "rejection_reasons": role[
                        "rejection_reasons"
                    ],
                    "width_aware_reference": width_debug,
                }
            selection = {
                name: by_id[candidate_id]
                for name, candidate_id in role[
                    "selection_candidate_ids"
                ].items()
            }
            return {
                **frozen,
                "status": "available",
                "unavailable_reason": None,
                "relationship_status": "basin_unique",
                "selection": selection,
                "competing_explanations": [
                    {
                        "type": "clicked_basin_roles",
                        "hypothesis_id": role["hypothesis_id"],
                        "selection_candidate_ids": role[
                            "selection_candidate_ids"
                        ],
                        "verified": True,
                        "rejection_reasons": [],
                    }
                ],
                "rejection_reasons": [],
                "width_aware_reference": width_debug,
            }
        if semantic_relation == "on_separator":
            semantic_tiebreak = {
                **semantic_tiebreak,
                "adopted": ENABLE_REFERENCE_SEMANTIC_TIEBREAK,
                "resolved_relation": "on_separator",
                "reason": semantic_tiebreak.get(
                    "reason",
                    "strong_grayscale_separator_semantic",
                ),
            }
            width_debug["semantic_tiebreak"] = semantic_tiebreak
            return {
                **frozen,
                "status": "unavailable",
                "unavailable_reason": "reference_on_separator",
                "relationship_status": "reference_on_separator",
                "selection": {},
                "competing_explanations": [
                    {
                        "type": "reference_on_separator",
                        "candidate_id": relation_candidate[
                            "candidate_id"
                        ],
                        "grayscale_semantic": True,
                        "evidence": semantic,
                        "semantic_tiebreak": semantic_tiebreak,
                    },
                    {
                        "type": "clicked_basin_roles",
                        "hypothesis_id": role["hypothesis_id"],
                        "selection_candidate_ids": role[
                            "selection_candidate_ids"
                        ],
                        "verified": True,
                        "rejection_reasons": [],
                    },
                ],
                "rejection_reasons": ["reference_on_separator"],
                "width_aware_reference": width_debug,
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
    verified_roles = [
        item
        for item in frozen.get("role_hypotheses", [])
        if item.get("verified") is True
    ]
    if ambiguous and len(verified_roles) == 1:
        role = verified_roles[0]
        candidate_by_id = {
            item["candidate_id"]: item for item in accepted
        }
        width_debug["uncertain_relationship_diagnostic"] = {
            "candidate_ids": [item["candidate_id"] for item in ambiguous],
            "veto_applied": False,
            "reason": "verified_local_basin_identity_preferred",
        }
        return {
            **frozen,
            "status": "available",
            "unavailable_reason": None,
            "relationship_status": "basin_unique",
            "selection": {
                name: candidate_by_id[candidate_id]
                for name, candidate_id in role[
                    "selection_candidate_ids"
                ].items()
            },
            "rejection_reasons": [],
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


@profiled("centerized_separator_pass", "geometry_validation")
def detect_centerized_separator_paths_v1_1(
    image_gray: np.ndarray,
    reference_global: dict,
    roi_bounds_global: dict,
    direction: str = "vertical",
    *,
    frozen_separator_result: dict | None = None,
    frozen_pitch_result: dict | None = None,
) -> dict:
    frozen = (
        frozen_separator_result
        if frozen_separator_result is not None
        else FROZEN_DETECT_SEPARATOR_PATHS(
            image_gray,
            reference_global,
            roi_bounds_global,
            direction,
            separator.DEFAULT_CONFIG,
        )
    )
    with stage("centerized_roi_preparation", "image_roi_preparation"):
        roi_gray = separator.extract_raw_roi(
            image_gray,
            roi_bounds_global,
        )
        directional = separator._directional_roi(  # noqa: SLF001
            roi_gray,
            direction,
        )
    with stage("centerized_raw_preprocessing", "preprocessing"):
        evidence = separator.build_raw_gray_response(
            directional,
            separator.DEFAULT_CONFIG,
        )
    with stage(
        "raw_candidate_trace_generation",
        "candidate_detection_and_path_tracking",
    ):
        raw_candidates = v1._raw_candidates_before_dedup(  # noqa: SLF001
            evidence
        )
    with stage("centerized_pitch_estimation", "pitch_estimation"):
        pitch_result = _reuse_or_estimate_pitch_result(
            frozen_pitch_result,
            image_gray,
            reference_global,
            roi_bounds_global,
            direction,
        )
    with stage("candidate_centerization", "path_track_basin_search"):
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
    with stage("centerized_role_validation", "geometry_validation"):
        arbitration = select_clicked_basin_paths_v1_1(
            candidates,
            frozen["reference_x_roi"],
            frozen["reference_y_roi"],
            evidence,
            local_pitch,
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


def _click_local_four_path_revalidation(
    separator_result: dict,
    graph: dict,
) -> dict:
    """Revalidate one rejected four-path role near the clicked row only."""

    empty = {
        "applied": False,
        "success": False,
        "reason": "trigger_not_met",
    }
    arbitration = separator_result.get("arbitration_debug", {})
    roles = arbitration.get("role_hypotheses", [])
    if len(roles) != 1:
        return {**empty, "reason": "role_hypothesis_not_unique"}
    role = roles[0]
    rejection_reasons = set(role.get("rejection_reasons", []))
    soft_full_height_reasons = {
        "path_order_not_stable_full_height",
        "role_path_insufficient_vertical_support",
        "role_path_geometry_unstable",
        "dark_basin_sequence_not_verified",
        "dark_basin_width_sequence_irregular",
    }
    if (
        role.get("type") != "clicked_basin_roles"
        or not rejection_reasons
        or rejection_reasons - soft_full_height_reasons
        or arbitration.get("relationship_status")
        not in {"basin_rejected", "basin_structure_rejected"}
    ):
        return {**empty, "reason": "not_soft_full_height_failure"}
    if role.get("crossing") or any(
        item.get("crossing_band_indices")
        for item in role.get("full_height_pair_order", [])
    ):
        return {**empty, "reason": "path_crossing_full_height"}

    selection = role.get("selection_candidate_ids", {})
    try:
        selected_ids = [selection[name] for name in separator.ROLE_ORDER]
    except KeyError:
        return {**empty, "reason": "selected_separator_missing"}
    ordered_ids = graph.get("ordered_separator_ids", [])
    try:
        selected_indexes = [ordered_ids.index(item) for item in selected_ids]
    except ValueError:
        return {**empty, "reason": "selected_separator_missing"}
    if selected_indexes != list(
        range(selected_indexes[0], selected_indexes[0] + 4)
    ):
        return {**empty, "reason": "selected_separator_not_consecutive"}

    candidate_by_id = {
        item["candidate_id"]: item
        for item in separator_result.get("candidates", [])
    }
    try:
        paths = [candidate_by_id[item] for item in selected_ids]
    except KeyError:
        return {**empty, "reason": "selected_separator_missing"}
    band_centers = np.asarray(
        paths[0]["band_centers_y_roi"], dtype=np.float64
    )
    reference_y = float(separator_result["reference_y_roi"])
    local_indices = np.sort(
        np.argsort(np.abs(band_centers - reference_y))[:3]
    )
    if (
        local_indices.size != 3
        or not np.all(np.diff(local_indices) == 1)
    ):
        return {**empty, "reason": "local_bands_not_contiguous"}
    supported_sets = [
        set(int(value) for value in path.get("supported_band_indices", []))
        for path in paths
    ]
    if any(
        not all(int(index) in supported for index in local_indices)
        for supported in supported_sets
    ):
        return {
            **empty,
            "applied": True,
            "reason": "local_path_support_not_verified",
            "band_indices": local_indices.tolist(),
        }

    local_x = np.asarray(
        [
            np.asarray(path["x_by_band_roi"], dtype=np.float64)[
                local_indices
            ]
            for path in paths
        ]
    )
    local_steps = np.mean(np.abs(np.diff(local_x, axis=1)), axis=1)
    if np.any(
        local_steps
        > separator.DEFAULT_CONFIG.maximum_role_path_mean_step_px
    ):
        return {
            **empty,
            "applied": True,
            "reason": "local_path_geometry_unstable",
            "band_indices": local_indices.tolist(),
            "mean_step_px": local_steps.tolist(),
        }
    pair_gaps = np.diff(local_x, axis=0)
    if np.any(pair_gaps <= 0):
        return {
            **empty,
            "applied": True,
            "reason": "local_path_crossing",
            "band_indices": local_indices.tolist(),
        }
    pair_medians = np.median(pair_gaps, axis=1)
    if np.any(
        np.min(pair_gaps, axis=1) / pair_medians
        < separator.DEFAULT_CONFIG.minimum_full_height_order_ratio
    ):
        return {
            **empty,
            "applied": True,
            "reason": "local_path_order_unstable",
            "band_indices": local_indices.tolist(),
        }
    widths = np.asarray(
        [float(path["peak_width_px"]) for path in paths],
        dtype=np.float64,
    )
    local_edge_gaps = pair_gaps - (
        widths[:-1, None] + widths[1:, None]
    ) / 2.0
    if np.any(
        local_edge_gaps
        < separator.DEFAULT_CONFIG.minimum_dark_basin_edge_gap_px
    ):
        return {
            **empty,
            "applied": True,
            "reason": "local_separator_envelopes_overlap",
            "band_indices": local_indices.tolist(),
        }

    basin_by_pair = {
        (item["left_separator_id"], item["right_separator_id"]): item
        for item in graph.get("basin_candidates", [])
    }
    pairs = list(zip(selected_ids, selected_ids[1:]))
    if any(pair not in basin_by_pair for pair in pairs):
        return {**empty, "reason": "selected_basin_missing"}
    selected_basins = [basin_by_pair[pair] for pair in pairs]
    basin_audits = []
    for basin in selected_basins:
        contrast = np.asarray(
            basin["dark_basin_evidence"][
                "normalized_contrast_by_band"
            ],
            dtype=np.float64,
        )[local_indices]
        stable_fraction = float(
            np.mean(
                contrast
                >= separator.DEFAULT_CONFIG.minimum_dark_basin_contrast
            )
        )
        median_contrast = float(np.median(contrast))
        accepted = bool(
            stable_fraction
            >= separator.DEFAULT_CONFIG.minimum_dark_basin_band_fraction
            and median_contrast
            >= separator.DEFAULT_CONFIG.minimum_dark_basin_contrast
        )
        basin_audits.append(
            {
                "basin_id": basin["basin_id"],
                "contrast": contrast.tolist(),
                "stable_fraction": stable_fraction,
                "median_contrast": median_contrast,
                "accepted": accepted,
            }
        )
    # These coverage scores describe every basin in the four-path role,
    # including the clicked/auxiliary basin that is not reported.  They are
    # useful diagnostics, but cannot by themselves invalidate locally stable,
    # ordered separator identity.  The final output-center darkness invariant
    # remains responsible for rejecting an actually unreliable reported
    # left/right center.
    contrast_diagnostic_only = not all(
        item["accepted"] for item in basin_audits
    )

    patched_separator = copy.deepcopy(separator_result)
    patched_arbitration = patched_separator["arbitration_debug"]
    patched_role = copy.deepcopy(role)
    patched_role["verified"] = True
    patched_role["rejection_reasons"] = []
    patched_role["click_local_four_path_revalidation"] = True
    patched_arbitration["role_hypotheses"] = [patched_role]
    patched_arbitration["rejection_reasons"] = [
        reason
        for reason in patched_arbitration.get("rejection_reasons", [])
        if reason not in soft_full_height_reasons
    ]
    for explanation in patched_arbitration.get(
        "competing_explanations", []
    ):
        if explanation.get("hypothesis_id") == role.get("hypothesis_id"):
            explanation["verified"] = True
            explanation["rejection_reasons"] = []
    if patched_separator.get("unavailable_reason") != "reference_on_separator":
        patched_separator["status"] = "available"
        patched_separator["unavailable_reason"] = None
        patched_separator["selection"] = {
            name: candidate_by_id[selection[name]]
            for name in separator.ROLE_ORDER
        }

    patched_graph = copy.deepcopy(graph)
    recovered_basins = [
        {
            **basin,
            "verified": True,
            "rejection_reasons": [],
            "source": "click_local_four_path_revalidation",
            "click_local_basin_contrast": audit,
        }
        for basin, audit in zip(selected_basins, basin_audits)
    ]
    recovered_ids = {item["basin_id"] for item in recovered_basins}
    patched_graph["verified_basins"] = [
        item
        for item in patched_graph.get("verified_basins", [])
        if item["basin_id"] not in recovered_ids
    ] + recovered_basins
    patched_graph["verified_basin_ids"] = [
        item["basin_id"] for item in patched_graph["verified_basins"]
    ]
    return {
        **empty,
        "applied": True,
        "success": True,
        "reason": "click_local_four_path_verified",
        "source_rejections": sorted(rejection_reasons),
        "separator_sequence": selected_ids,
        "band_indices": local_indices.tolist(),
        "mean_step_px": local_steps.tolist(),
        "basin_audits": basin_audits,
        "contrast_diagnostic_only": contrast_diagnostic_only,
        "separator_result": patched_separator,
        "graph": patched_graph,
    }


def _recover_local_adaptive_dim_separator(
    image_gray: np.ndarray,
    roi_bounds_global: dict,
    direction: str,
    separator_result: dict,
    relation: dict,
    pitch_result: dict,
) -> dict:
    """Recover one dim separator only inside an already diagnosed merged gap."""

    empty = {
        "triggered": False,
        "success": False,
        "reason": "trigger_not_met",
        "separator_result": separator_result,
        "candidate": None,
    }
    if relation.get("status") != "unique" or len(relation.get("hypotheses", [])) != 1:
        return empty
    hypothesis = relation["hypotheses"][0]

    pitch = pitch_result.get("diagnostic_pitch_px")
    sequence = hypothesis.get("separator_sequence", [])
    if pitch is None or float(pitch) <= 0 or len(sequence) < 3:
        return {**empty, "triggered": True, "reason": "merged_gap_geometry_missing"}
    candidate_by_id = {
        item["candidate_id"]: item
        for item in separator_result.get("candidates", [])
        if item.get("accepted")
    }
    if any(candidate_id not in candidate_by_id for candidate_id in sequence):
        return {**empty, "triggered": True, "reason": "merged_gap_paths_missing"}
    paths = [candidate_by_id[candidate_id] for candidate_id in sequence]
    gap_by_band = [
        np.asarray(right["x_by_band_roi"], dtype=np.float64)
        - np.asarray(left["x_by_band_roi"], dtype=np.float64)
        for left, right in zip(paths, paths[1:])
    ]
    gaps = [float(np.median(values)) for values in gap_by_band]
    reference_x_roi = float(separator_result["reference_x_roi"])
    reference_y_roi = float(separator_result["reference_y_roi"])
    reference_gap_tolerance = float(
        separator.DEFAULT_CONFIG.match_tolerance_px
    )
    separator_roles = hypothesis.get("separator_ids", {})
    clicked_boundary_pair = (
        separator_roles.get("left_clicked_boundary"),
        separator_roles.get("right_clicked_boundary"),
    )
    clicked_separator_id = hypothesis.get(
        "clicked_separator_id",
        hypothesis.get("reference_separator_id"),
    )

    def is_output_local_gap(index: int) -> bool:
        pair = (
            paths[index]["candidate_id"],
            paths[index + 1]["candidate_id"],
        )
        if all(clicked_boundary_pair):
            return pair == clicked_boundary_pair
        return clicked_separator_id in pair

    eligible_gap_indices = [
        index
        for index, (left, right, values) in enumerate(
            zip(paths, paths[1:], gap_by_band)
        )
        if np.all(values > 0.0)
        and is_output_local_gap(index)
        and gaps[index] / float(pitch)
        > joint.DEFAULT_CONFIG.high_pitch_maximum_spacing_ratio
        and separator._path_x_at_y(  # noqa: SLF001
            left, reference_y_roi
        ) - reference_gap_tolerance
        <= reference_x_roi
        <= separator._path_x_at_y(  # noqa: SLF001
            right, reference_y_roi
        ) + reference_gap_tolerance
    ]
    if not eligible_gap_indices:
        return {**empty, "triggered": True, "reason": "no_local_merged_gap"}
    wide_index = max(eligible_gap_indices, key=gaps.__getitem__)

    left_path = paths[wide_index]
    right_path = paths[wide_index + 1]
    left_x = np.asarray(left_path["x_by_band_roi"], dtype=np.float64)
    right_x = np.asarray(right_path["x_by_band_roi"], dtype=np.float64)
    expected_x = (left_x + right_x) / 2.0
    raw_roi = separator.extract_raw_roi(image_gray, roi_bounds_global)
    adaptive_mask = _preprocess_roi(
        raw_roi,
        CONFIG,
        INTERACTIVE_CONFIG,
        "adaptive",
    ).threshold_binary
    directional_mask = separator._directional_roi(  # noqa: SLF001
        adaptive_mask,
        direction,
    )
    band_bounds = separator._band_bounds(  # noqa: SLF001
        directional_mask.shape[0],
        separator.DEFAULT_CONFIG.band_count,
    )
    if len(band_bounds) != len(expected_x):
        return {**empty, "triggered": True, "reason": "adaptive_band_count_mismatch"}

    adaptive_responses = np.zeros(
        (len(band_bounds), directional_mask.shape[1]),
        dtype=np.float64,
    )
    local_radius = int(round(separator.DEFAULT_CONFIG.match_tolerance_px))
    for band_index, (y0, y1) in enumerate(band_bounds):
        center = int(round(float(expected_x[band_index])))
        x0 = max(1, center - local_radius)
        x1 = min(directional_mask.shape[1] - 1, center + local_radius + 1)
        for x in range(x0, x1):
            white_fraction = float(
                np.mean(directional_mask[y0:y1, x - 1 : x + 2] > 0)
            )
            adaptive_responses[band_index, x] = (
                white_fraction / ADAPTIVE_DIM_SEPARATOR_WHITE_FRACTION
            )
    aggregate = np.mean(
        np.partition(
            adaptive_responses,
            adaptive_responses.shape[0]
            - separator.DEFAULT_CONFIG.aggregate_top_band_count,
            axis=0,
        )[-separator.DEFAULT_CONFIG.aggregate_top_band_count :],
        axis=0,
    )
    adaptive_evidence = {
        "band_bounds": band_bounds,
        "band_centers_y": np.asarray(
            [(y0 + y1 - 1) / 2.0 for y0, y1 in band_bounds],
            dtype=np.float64,
        ),
        "profiles": adaptive_responses,
        "responses": adaptive_responses,
        "aggregate_response": aggregate,
        "width": directional_mask.shape[1],
        "height": directional_mask.shape[0],
    }
    seed_x = int(round(float(np.median(expected_x))))
    candidate = separator._trace_one_seed(  # noqa: SLF001
        {"x": seed_x, "response": float(aggregate[seed_x])},
        adaptive_evidence,
        separator.DEFAULT_CONFIG,
    )
    candidate.update(
        {
            "candidate_id": "ADAPTIVE_LOCAL_DIM_01",
            "x_at_reference_roi": separator._path_x_at_y(  # noqa: SLF001
                candidate, float(separator_result["reference_y_roi"])
            ),
            "suppressed_candidates": [],
            "adaptive_local_recovery": {
                "origin": "existing_adaptive_white_mask",
                "trigger": "local_reference_merged_gap",
                "left_separator_id": left_path["candidate_id"],
                "right_separator_id": right_path["candidate_id"],
                "merged_gap_px": gaps[wide_index],
                "pitch_px": float(pitch),
                "minimum_white_fraction": ADAPTIVE_DIM_SEPARATOR_WHITE_FRACTION,
            },
        }
    )
    if not candidate["accepted"]:
        return {
            **empty,
            "triggered": True,
            "reason": candidate.get("rejection_reason") or "adaptive_path_rejected",
            "candidate": candidate,
        }

    candidate_x = np.asarray(candidate["x_by_band_roi"], dtype=np.float64)
    if not (
        np.all(candidate_x - left_x > 0.0)
        and np.all(right_x - candidate_x > 0.0)
    ):
        return {
            **empty,
            "triggered": True,
            "reason": "adaptive_separator_order_conflict",
            "candidate": candidate,
        }
    split_gaps = [
        float(np.median(candidate_x - left_x)),
        float(np.median(right_x - candidate_x)),
    ]
    pitch_tolerance = float(
        INTERACTIVE_CONFIG.neighbor_recovery_max_pitch_error_ratio
    )
    minimum_split_gap = (1.0 - pitch_tolerance) * float(pitch)
    maximum_split_gap = (1.0 + pitch_tolerance) * float(pitch)
    if not all(
        minimum_split_gap <= gap <= maximum_split_gap
        for gap in split_gaps
    ):
        return {
            **empty,
            "triggered": True,
            "reason": "adaptive_split_pitch_inconsistent",
            "candidate": candidate,
        }

    raw_directional = separator._directional_roi(  # noqa: SLF001
        raw_roi,
        direction,
    )
    raw_evidence = separator.build_raw_gray_response(
        raw_directional,
        separator.DEFAULT_CONFIG,
    )
    candidates = [*separator_result["candidates"], candidate]
    arbitration = select_clicked_basin_paths_v1_1(
        candidates,
        float(separator_result["reference_x_roi"]),
        float(separator_result["reference_y_roi"]),
        raw_evidence,
        float(pitch),
    )
    recovered = {
        **separator_result,
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
        "adaptive_local_dim_separator_recovery": {
            "triggered": True,
            "candidate_id": candidate["candidate_id"],
            "candidate_x_at_reference_roi": candidate["x_at_reference_roi"],
            "support_fraction": candidate["support_fraction"],
        },
    }
    return {
        **empty,
        "triggered": True,
        "success": True,
        "reason": "adaptive_separator_candidate_accepted",
        "separator_result": recovered,
        "candidate": candidate,
    }


def _basin_graph_input_key(separator_result: dict) -> tuple:
    """Return exactly the separator fields read by basin graph construction."""

    return (
        separator_result["reference_y_roi"],
        tuple(
            (
                path["candidate_id"],
                tuple(path["band_centers_y_roi"]),
                tuple(path["x_by_band_roi"]),
                bool(path.get("adaptive_local_recovery")),
            )
            for path in separator_result["candidates"]
            if path["accepted"]
        ),
    )


def _basin_graph_inputs_match(
    first: dict | None,
    second: dict | None,
) -> bool:
    """Return whether two separator results produce the same basin graph."""

    if first is None or second is None:
        return False
    try:
        return _basin_graph_input_key(first) == _basin_graph_input_key(
            second
        )
    except (KeyError, TypeError):
        return False


def _reuse_or_estimate_pitch_result(
    frozen_pitch_result: dict | None,
    image_gray: np.ndarray,
    reference_global: dict,
    roi_bounds_global: dict,
    direction: str,
) -> dict:
    """Return an isolated copy of the same-request frozen pitch result."""

    if frozen_pitch_result is not None:
        return copy.deepcopy(frozen_pitch_result)
    return raw_pitch.estimate_raw_local_pitch_v3(
        image_gray,
        reference_global,
        roi_bounds_global,
        direction,
        raw_pitch.DEFAULT_CONFIG,
    )


@profiled("cross_image_joint_pass", "geometry_validation")
def _run_joint_case_v1_1_once(
    image_gray: np.ndarray,
    reference_global: dict,
    roi_bounds_global: dict,
    direction: str = "vertical",
    *,
    enable_adaptive_recovery: bool = True,
) -> dict:
    """Run one joint-chain pass with optional adaptive recovery."""

    frozen_result = FROZEN_RUN_JOINT_CASE(
        image_gray,
        reference_global,
        roi_bounds_global,
        direction,
        joint.DEFAULT_CONFIG,
    )
    frozen_debug = frozen_result.get("debug", {})
    frozen_separator_result = frozen_debug.get(
        "separator_result"
    )
    frozen_pitch_result = frozen_debug.get("raw_pitch_result")
    separator_result = detect_centerized_separator_paths_v1_1(
        image_gray,
        reference_global,
        roi_bounds_global,
        direction,
        frozen_separator_result=frozen_separator_result,
        frozen_pitch_result=frozen_pitch_result,
    )
    with stage("cross_image_roi_preparation", "image_roi_preparation"):
        raw_roi = separator.extract_raw_roi(
            image_gray,
            roi_bounds_global,
        )
    with stage("cross_image_basin_graph", "path_track_basin_search"):
        frozen_separator = frozen_debug.get("separator_result")
        frozen_graph = frozen_debug.get("basin_graph")
        if _basin_graph_inputs_match(
            frozen_separator,
            separator_result,
        ) and frozen_graph is not None:
            graph = copy.deepcopy(frozen_graph)
        else:
            graph = joint.build_ordered_basin_graph(
                separator_result,
                raw_roi,
                direction,
            )
    with stage("cross_image_pitch_estimation", "pitch_estimation"):
        pitch_result = _reuse_or_estimate_pitch_result(
            frozen_pitch_result,
            image_gray,
            reference_global,
            roi_bounds_global,
            direction,
        )
    with stage("cross_image_relation_validation", "geometry_validation"):
        relation = joint.resolve_reference_relation(
            separator_result,
            graph,
            pitch_result,
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
        local_structure = _click_local_four_path_revalidation(
            separator_result,
            graph,
        )
        debug["click_local_four_path_revalidation"] = {
            key: value
            for key, value in local_structure.items()
            if key not in {"separator_result", "graph"}
        }
        if local_structure["success"]:
            separator_result = local_structure["separator_result"]
            graph = local_structure["graph"]
            relation = joint.resolve_reference_relation(
                separator_result,
                graph,
                pitch_result,
            )
            debug["separator_result"] = separator_result
            debug["basin_graph"] = graph
            debug["reference_relation"] = relation
            debug["staged_release"] = {
                "stage": "click_local_four_path_revalidation",
                "source_rejections": local_structure[
                    "source_rejections"
                ],
            }
    if enable_adaptive_recovery:
        adaptive_dim_recovery = _recover_local_adaptive_dim_separator(
            image_gray,
            roi_bounds_global,
            direction,
            separator_result,
            relation,
            pitch_result,
        )
    else:
        adaptive_dim_recovery = {
            "triggered": False,
            "success": False,
            "reason": "disabled_for_baseline_fallback",
            "separator_result": separator_result,
            "candidate": None,
        }
    debug["adaptive_local_dim_separator_recovery"] = {
        key: value
        for key, value in adaptive_dim_recovery.items()
        if key not in {"separator_result", "candidate"}
    }
    if adaptive_dim_recovery["success"]:
        separator_result = adaptive_dim_recovery["separator_result"]
        graph = joint.build_ordered_basin_graph(
            separator_result,
            raw_roi,
            direction,
        )
        relation = joint.resolve_reference_relation(
            separator_result,
            graph,
            pitch_result,
        )
        debug["separator_result"] = separator_result
        debug["basin_graph"] = graph
        debug["reference_relation"] = relation
    if relation["status"] != "unique":
        local_recovery = _recover_local_dark_valley_sequence(
            image_gray,
            roi_bounds_global,
            direction,
            separator_result,
            graph,
            None,
            pitch_result,
        )
        debug["local_dark_valley_recovery"] = local_recovery
        if local_recovery["success"]:
            recovered_hypothesis = local_recovery["hypothesis"]
            recovered_ids = {
                basin["basin_id"]
                for basin in local_recovery["selected_basins"]
            }
            graph["verified_basins"] = [
                basin
                for basin in graph.get("verified_basins", [])
                if basin["basin_id"] not in recovered_ids
            ] + local_recovery["selected_basins"]
            graph["verified_basin_ids"] = [
                basin["basin_id"] for basin in graph["verified_basins"]
            ]
            debug["joint_hypotheses"] = [recovered_hypothesis]
            if not frozen_result["success"]:
                debug["release_unavailable_audit"] = {
                    "reason": "release_unavailable_cannot_become_success",
                    "release_unavailable_reason": frozen_result[
                        "unavailable_reason"
                    ],
                    "hard_veto_applied": False,
                    "accepted_hypothesis": recovered_hypothesis,
                }
            return {
                **base,
                "status": "available",
                "success": True,
                "unavailable_reason": None,
                "final_hypothesis": recovered_hypothesis,
            }
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
        if ENABLE_PITCH_GUIDED_MISSING_SEPARATOR_COMPLETION:
            local_recovery = _recover_local_dark_valley_sequence(
                image_gray,
                roi_bounds_global,
                direction,
                separator_result,
                graph,
                hypothesis,
                pitch_result,
                allow_single_clicked_track=True,
                staged_trigger="high_pitch_geometry_conflict",
            )
            debug["local_dark_valley_recovery"] = local_recovery
            if local_recovery["success"]:
                recovered_hypothesis = local_recovery["hypothesis"]
                graph["verified_basins"].extend(
                    local_recovery["selected_basins"]
                )
                debug["joint_hypotheses"] = [recovered_hypothesis]
                debug["staged_release"] = {
                    "stage": "pitch_guided_missing_separator_completion",
                    "source_veto": "high_pitch_geometry_conflict",
                    "recovery_reason": local_recovery.get("reason"),
                }
                return {
                    **base,
                    "status": "available",
                    "success": True,
                    "unavailable_reason": None,
                    "final_hypothesis": recovered_hypothesis,
                }
        return {
            **base,
            "status": "unavailable",
            "success": False,
            "unavailable_reason": "high_pitch_geometry_conflict",
            "final_hypothesis": None,
        }
    intermediate_dark = _intermediate_dark_basin_evidence(
        image_gray,
        roi_bounds_global,
        direction,
        separator_result,
        graph,
        hypothesis,
    )
    debug["intermediate_dark_basin_evidence"] = intermediate_dark
    local_recovery = _recover_local_dark_valley_sequence(
        image_gray,
        roi_bounds_global,
        direction,
        separator_result,
        graph,
        hypothesis,
        pitch_result,
        allow_single_clicked_track=bool(
            ENABLE_PITCH_GUIDED_MISSING_SEPARATOR_COMPLETION
            and any(
                item["verified"] for item in intermediate_dark["sides"]
            )
        ),
        staged_trigger=(
            "intermediate_dark_basin_present"
            if ENABLE_PITCH_GUIDED_MISSING_SEPARATOR_COMPLETION
            else None
        ),
    )
    debug["local_dark_valley_recovery"] = local_recovery
    if local_recovery["triggered"]:
        if local_recovery["success"]:
            recovered_hypothesis = local_recovery["hypothesis"]
            graph["verified_basins"].extend(
                local_recovery["selected_basins"]
            )
            debug["joint_hypotheses"] = [recovered_hypothesis]
            if local_recovery.get("single_clicked_track_allowed"):
                debug["staged_release"] = {
                    "stage": "pitch_guided_missing_separator_completion",
                    "source_veto": "intermediate_dark_basin_present",
                    "recovery_reason": local_recovery.get("reason"),
                }
            if not frozen_result["success"]:
                debug["release_unavailable_audit"] = {
                    "reason": "release_unavailable_cannot_become_success",
                    "release_unavailable_reason": frozen_result[
                        "unavailable_reason"
                    ],
                    "hard_veto_applied": False,
                    "accepted_hypothesis": recovered_hypothesis,
                }
            return {
                **base,
                "status": "available",
                "success": True,
                "unavailable_reason": None,
                "final_hypothesis": recovered_hypothesis,
            }
        debug["intermediate_dark_basin_veto"] = {
            "reason": "intermediate_dark_basin_present",
            "triggered_sides": [
                item
                for item in intermediate_dark["sides"]
                if item["verified"]
            ],
            "recovery_failure": local_recovery["reason"],
            "rejected_hypothesis": hypothesis,
        }
        return {
            **base,
            "status": "unavailable",
            "success": False,
            "unavailable_reason": "intermediate_dark_basin_present",
            "final_hypothesis": None,
        }
    triggered = [
        item
        for item in intermediate_dark["sides"]
        if item["verified"]
    ]
    if triggered:
        debug["intermediate_dark_basin_veto"] = {
            "reason": "intermediate_dark_basin_present",
            "triggered_sides": triggered,
            "rejected_hypothesis": hypothesis,
        }
        return {
            **base,
            "status": "unavailable",
            "success": False,
            "unavailable_reason": "intermediate_dark_basin_present",
            "final_hypothesis": None,
        }
    output_center_safety = joint.validate_output_basin_center_darkness(
        hypothesis,
        graph,
        separator_result["reference_y_roi"],
        joint.DEFAULT_CONFIG,
    )
    debug["output_basin_center_darkness_safety"] = output_center_safety
    if not output_center_safety["success"]:
        return {
            **base,
            "status": "unavailable",
            "success": False,
            "unavailable_reason": output_center_safety["reason"],
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


@profiled("cross_image_v1_1_total")
def run_joint_case_v1_1(
    image_gray: np.ndarray,
    reference_global: dict,
    roi_bounds_global: dict,
    direction: str = "vertical",
) -> dict:
    """Prefer a fully validated recovery, otherwise retain a prior success."""

    recovered_result = _run_joint_case_v1_1_once(
        image_gray,
        reference_global,
        roi_bounds_global,
        direction,
        enable_adaptive_recovery=True,
    )
    recovery = recovered_result.get("debug", {}).get(
        "adaptive_local_dim_separator_recovery",
        {},
    )
    if recovered_result.get("success") or not recovery.get("success"):
        return recovered_result

    baseline_result = _run_joint_case_v1_1_once(
        image_gray,
        reference_global,
        roi_bounds_global,
        direction,
        enable_adaptive_recovery=False,
    )
    if not baseline_result.get("success"):
        return recovered_result

    baseline_result["debug"]["adaptive_local_dim_separator_recovery"] = {
        **recovery,
        "fallback_to_baseline_success": True,
        "recovered_unavailable_reason": recovered_result.get(
            "unavailable_reason"
        ),
    }
    return baseline_result


def _raw_dark_valleys_in_interval(
    profile: np.ndarray,
    left_x: float,
    right_x: float,
) -> list[dict]:
    """Return dark valleys bounded by consecutive raw-gray ridges."""

    left = int(np.clip(round(left_x), 0, profile.size - 1))
    right = int(np.clip(round(right_x), 0, profile.size - 1))
    if right - left < separator.DEFAULT_CONFIG.seed_minimum_distance_px:
        return []
    maxima = [left]
    maxima.extend(
        index
        for index in range(left + 1, right)
        if (
            profile[index] >= profile[index - 1]
            and profile[index] >= profile[index + 1]
        )
    )
    maxima.append(right)
    valleys = []
    for left_peak, right_peak in zip(maxima, maxima[1:]):
        if (
            right_peak - left_peak
            < separator.DEFAULT_CONFIG.seed_minimum_distance_px
        ):
            continue
        local = profile[left_peak : right_peak + 1]
        dark_x = left_peak + int(np.argmin(local))
        dark_gray = float(profile[dark_x])
        flank_gray = min(
            float(profile[left_peak]),
            float(profile[right_peak]),
        )
        gray_dynamic = flank_gray - dark_gray
        if gray_dynamic < MINIMUM_RAW_GRAY_DYNAMIC:
            continue
        valleys.append(
            {
                "left_x_roi": float(left_peak),
                "right_x_roi": float(right_peak),
                "minimum_x_roi": float(dark_x),
                "dark_gray": dark_gray,
                "flank_gray": flank_gray,
                "gray_dynamic": float(gray_dynamic),
            }
        )
    return valleys


def _match_valley_observations(
    observations_by_band: list[list[dict]],
) -> list[dict]:
    """Join band observations into ordered, vertically stable paths."""

    tracks: list[dict] = []
    tolerance = separator.DEFAULT_CONFIG.match_tolerance_px
    for band_index, observations in enumerate(observations_by_band):
        available_tracks = set(range(len(tracks)))
        available_observations = set(range(len(observations)))
        pairs = []
        for track_index, track in enumerate(tracks):
            expected = float(
                np.median(
                    [
                        item["minimum_x_roi"]
                        for item in track["observations"].values()
                    ]
                )
            )
            for observation_index, observation in enumerate(observations):
                distance = abs(
                    observation["minimum_x_roi"] - expected
                )
                if distance <= tolerance:
                    pairs.append(
                        (distance, track_index, observation_index)
                    )
        for _distance, track_index, observation_index in sorted(pairs):
            if (
                track_index not in available_tracks
                or observation_index not in available_observations
            ):
                continue
            tracks[track_index]["observations"][band_index] = (
                observations[observation_index]
            )
            available_tracks.remove(track_index)
            available_observations.remove(observation_index)
        for observation_index in sorted(available_observations):
            tracks.append(
                {
                    "observations": {
                        band_index: observations[observation_index]
                    }
                }
            )
    return tracks


def _interpolated_track_values(
    track: dict,
    key: str,
    band_count: int,
) -> np.ndarray:
    indices = np.asarray(sorted(track["observations"]), dtype=np.float64)
    values = np.asarray(
        [track["observations"][int(index)][key] for index in indices],
        dtype=np.float64,
    )
    return np.interp(np.arange(band_count), indices, values)


def _ordered_local_valley_tracks(
    directional: np.ndarray,
    band_centers_y: np.ndarray,
    outer_left_by_band: np.ndarray,
    outer_right_by_band: np.ndarray,
) -> tuple[list[dict], list[np.ndarray]]:
    """Build multi-band and row-supported dark-valley tracks."""

    band_bounds = separator._band_bounds(  # noqa: SLF001
        directional.shape[0],
        separator.DEFAULT_CONFIG.band_count,
    )
    band_profiles = []
    observations_by_band = []
    for band_index, (y0, y1) in enumerate(band_bounds):
        profile = separator._smooth_profile(  # noqa: SLF001
            np.median(
                directional[y0:y1].astype(np.float64),
                axis=0,
            ),
            separator.DEFAULT_CONFIG.profile_smoothing_sigma_px,
        )
        band_profiles.append(profile)
        observations_by_band.append(
            _raw_dark_valleys_in_interval(
                profile,
                outer_left_by_band[band_index],
                outer_right_by_band[band_index],
            )
        )

    minimum_fraction = (
        separator.DEFAULT_CONFIG.minimum_dark_basin_band_fraction
    )
    candidates = []
    for track in _match_valley_observations(observations_by_band):
        support = len(track["observations"]) / len(band_bounds)
        if support < minimum_fraction:
            continue
        center_path = _interpolated_track_values(
            track, "minimum_x_roi", len(band_bounds)
        )
        mean_step = float(
            np.mean(np.abs(np.diff(center_path)))
        )
        if (
            mean_step
            > separator.DEFAULT_CONFIG.maximum_role_path_mean_step_px
        ):
            continue
        candidates.append(
            {
                **track,
                "band_support": support,
                "mean_band_step_px": mean_step,
                "minimum_x_by_band_roi": center_path,
                "left_x_by_band_roi": _interpolated_track_values(
                    track, "left_x_roi", len(band_bounds)
                ),
                "right_x_by_band_roi": _interpolated_track_values(
                    track, "right_x_roi", len(band_bounds)
                ),
            }
        )

    row_matches = [0] * len(candidates)
    tolerance = separator.DEFAULT_CONFIG.match_tolerance_px
    for row_index in range(directional.shape[0]):
        profile = separator._smooth_profile(  # noqa: SLF001
            directional[row_index].astype(np.float64),
            separator.DEFAULT_CONFIG.profile_smoothing_sigma_px,
        )
        left_x = float(
            np.interp(
                row_index, band_centers_y, outer_left_by_band
            )
        )
        right_x = float(
            np.interp(
                row_index, band_centers_y, outer_right_by_band
            )
        )
        observations = _raw_dark_valleys_in_interval(
            profile, left_x, right_x
        )
        pairs = []
        for track_index, track in enumerate(candidates):
            expected = float(
                np.interp(
                    row_index,
                    band_centers_y,
                    track["minimum_x_by_band_roi"],
                )
            )
            for observation_index, observation in enumerate(observations):
                distance = abs(
                    observation["minimum_x_roi"] - expected
                )
                if distance <= tolerance:
                    pairs.append(
                        (distance, track_index, observation_index)
                    )
        used_tracks = set()
        used_observations = set()
        for _distance, track_index, observation_index in sorted(pairs):
            if (
                track_index in used_tracks
                or observation_index in used_observations
            ):
                continue
            row_matches[track_index] += 1
            used_tracks.add(track_index)
            used_observations.add(observation_index)

    verified = []
    for track, match_count in zip(candidates, row_matches):
        row_support = match_count / directional.shape[0]
        if row_support < minimum_fraction:
            continue
        verified.append({**track, "row_support": row_support})
    verified.sort(
        key=lambda item: float(
            np.median(item["minimum_x_by_band_roi"])
        )
    )
    return verified, band_profiles


def _calculate_local_valley_basin(
    track: dict,
    track_index: int,
    band_centers_y: np.ndarray,
    band_profiles: list[np.ndarray],
    reference_y_roi: float,
) -> tuple[dict | None, dict]:
    """Calculate one center after the physical valley identity is fixed."""

    left_path = track["left_x_by_band_roi"]
    right_path = track["right_x_by_band_roi"]
    midpoint_path = (left_path + right_path) / 2.0
    basin_id = f"LVB{track_index + 1:02d}"
    provisional = {
        "basin_id": basin_id,
        "left_separator_id": f"{basin_id}_LEFT_RAW_RIDGE",
        "right_separator_id": f"{basin_id}_RIGHT_RAW_RIDGE",
        "left_x_by_band_roi": left_path.tolist(),
        "right_x_by_band_roi": right_path.tolist(),
        "center_x_by_band_roi": midpoint_path.tolist(),
    }
    center_result = v1._refine_one_basin(  # noqa: SLF001
        provisional,
        [profile.tolist() for profile in band_profiles],
        v1.DEFAULT_CORRECTION_CONFIG,
    )
    if center_result["reason"] == "insufficient_or_inconsistent_band_evidence":
        return None, center_result
    center_path = np.asarray(
        center_result["bandwise_center_path_x_roi"],
        dtype=np.float64,
    )
    center_at_reference = float(
        np.interp(reference_y_roi, band_centers_y, center_path)
    )
    left_at_reference = float(
        np.interp(reference_y_roi, band_centers_y, left_path)
    )
    right_at_reference = float(
        np.interp(reference_y_roi, band_centers_y, right_path)
    )
    basin = {
        **provisional,
        "source": "ordered_raw_gray_valley_path",
        "band_centers_y_roi": band_centers_y.tolist(),
        "center_x_by_band_roi": center_path.tolist(),
        "width_by_band_px": (right_path - left_path).tolist(),
        "center_x_at_reference_roi": center_at_reference,
        "width_at_reference_px": right_at_reference - left_at_reference,
        "band_support": track["band_support"],
        "row_support": track["row_support"],
        "mean_step_px": track["mean_band_step_px"],
        "verified": True,
        "rejection_reasons": [],
        "center_definition": "shared_dark_weighted_basin_center",
    }
    return basin, center_result


def _local_reference_profile_basin_evidence(
    profile: np.ndarray,
    basin: dict,
    reference_y_roi: float,
) -> dict:
    """Verify one separator-defined dark gap near the clicked row."""

    band_centers = np.asarray(
        basin["band_centers_y_roi"], dtype=np.float64
    )
    left_x = int(
        round(
            np.interp(
                reference_y_roi,
                band_centers,
                basin["left_x_by_band_roi"],
            )
        )
    )
    right_x = int(
        round(
            np.interp(
                reference_y_roi,
                band_centers,
                basin["right_x_by_band_roi"],
            )
        )
    )
    x0, x1 = sorted((left_x, right_x))
    if x1 - x0 < 3:
        return {
            "supported": False,
            "reason": "local_gap_too_narrow",
            "left_x_roi": x0,
            "right_x_roi": x1,
        }

    interior = profile[x0 + 1 : x1]
    local = profile[
        max(0, x0 - 5) : min(profile.size, x1 + 6)
    ]
    if interior.size == 0 or local.size == 0:
        return {
            "supported": False,
            "reason": "local_profile_unavailable",
            "left_x_roi": x0,
            "right_x_roi": x1,
        }
    boundary_level = min(float(profile[x0]), float(profile[x1]))
    interior_level = float(np.percentile(interior, 40))
    scale = max(
        float(np.percentile(local, 90) - np.percentile(local, 10)),
        MINIMUM_RAW_GRAY_DYNAMIC,
    )
    normalized_contrast = (boundary_level - interior_level) / scale
    return {
        "supported": bool(
            normalized_contrast
            >= separator.DEFAULT_CONFIG.minimum_dark_basin_contrast
        ),
        "reason": None,
        "left_x_roi": x0,
        "right_x_roi": x1,
        "boundary_level": boundary_level,
        "interior_level": interior_level,
        "dynamic_range": scale,
        "normalized_contrast": float(normalized_contrast),
    }


def _recover_contrast_limited_separator_sequence(
    image_gray: np.ndarray,
    roi_bounds_global: dict,
    direction: str,
    separator_result: dict,
    graph: dict,
    pitch_result: dict,
) -> dict:
    """Recover one locally verified gap triplet rejected only by coverage."""

    empty = {
        "applied": False,
        "triggered": False,
        "success": False,
        "mode": "separator_sequence_local_contrast",
        "reason": "trigger_not_met",
    }
    if direction != "vertical":
        return {**empty, "reason": "non_vertical_direction"}
    if (
        separator_result.get("status") != "unavailable"
        or separator_result.get("unavailable_reason")
        != "basin_structure_not_verified"
    ):
        return {**empty, "reason": "not_basin_structure_failure"}

    arbitration = separator_result.get("arbitration_debug", {})
    hypotheses = arbitration.get("role_hypotheses", [])
    if len(hypotheses) != 1:
        return {**empty, "reason": "role_hypothesis_not_unique"}
    role_hypothesis = hypotheses[0]
    if (
        role_hypothesis.get("type") != "clicked_basin_roles"
        or role_hypothesis.get("crossing")
        or set(role_hypothesis.get("rejection_reasons", []))
        != {"dark_basin_sequence_not_verified"}
    ):
        return {**empty, "reason": "not_contrast_coverage_only"}

    selection = role_hypothesis.get("selection_candidate_ids", {})
    try:
        selected_sequence = [
            selection[role] for role in separator.ROLE_ORDER
        ]
        indexes = [
            graph["ordered_separator_ids"].index(candidate_id)
            for candidate_id in selected_sequence
        ]
    except (KeyError, ValueError):
        return {**empty, "reason": "selected_separator_missing"}
    if indexes != list(range(indexes[0], indexes[0] + 4)):
        return {**empty, "reason": "selected_separator_not_consecutive"}

    path_metrics = role_hypothesis.get("path_metrics", [])
    if len(path_metrics) != 4 or any(
        item["support_fraction"]
        < separator.DEFAULT_CONFIG.minimum_role_path_support_fraction
        or item["mean_step_px"]
        > separator.DEFAULT_CONFIG.maximum_role_path_mean_step_px
        for item in path_metrics
    ):
        return {**empty, "reason": "selected_path_geometry_unstable"}
    pair_order = role_hypothesis.get("full_height_pair_order", [])
    if len(pair_order) != 3 or any(
        not item["stable_full_height_order"]
        or item["crossing_band_indices"]
        for item in pair_order
    ):
        return {**empty, "reason": "selected_path_order_unstable"}

    edge_gaps = np.asarray(
        role_hypothesis.get("edge_gap_at_reference_px", []),
        dtype=np.float64,
    )
    if edge_gaps.size != 3 or np.any(edge_gaps <= 0):
        return {**empty, "reason": "selected_gap_width_unavailable"}
    if (
        float(np.min(edge_gaps)) / float(np.max(edge_gaps))
        < separator.DEFAULT_CONFIG.minimum_full_height_order_ratio
    ):
        return {**empty, "reason": "selected_gap_width_inconsistent"}

    basin_by_pair = {
        (item["left_separator_id"], item["right_separator_id"]): item
        for item in graph.get("basin_candidates", [])
    }
    pairs = list(zip(selected_sequence, selected_sequence[1:]))
    if any(pair not in basin_by_pair for pair in pairs):
        return {**empty, "reason": "selected_basin_missing"}
    selected_basins = [basin_by_pair[pair] for pair in pairs]
    if any(
        set(basin.get("rejection_reasons", []))
        - {"dark_basin_not_verified"}
        for basin in selected_basins
    ):
        return {**empty, "reason": "selected_basin_structural_conflict"}

    reference_x = float(separator_result["reference_x_roi"])
    reference_y = float(separator_result["reference_y_roi"])
    clicked_basin = selected_basins[1]
    clicked_centers = np.asarray(
        clicked_basin["band_centers_y_roi"], dtype=np.float64
    )
    clicked_left = float(
        np.interp(
            reference_y,
            clicked_centers,
            clicked_basin["left_x_by_band_roi"],
        )
    )
    clicked_right = float(
        np.interp(
            reference_y,
            clicked_centers,
            clicked_basin["right_x_by_band_roi"],
        )
    )
    reference_guard = separator.DEFAULT_CONFIG.reference_guard_px
    if not (
        clicked_left + reference_guard
        < reference_x
        < clicked_right - reference_guard
    ):
        return {**empty, "reason": "reference_not_inside_clicked_gap_core"}

    raw_roi = separator.extract_raw_roi(image_gray, roi_bounds_global)
    directional = separator._directional_roi(  # noqa: SLF001
        raw_roi, direction
    )
    half_window = max(
        1, int(round(separator.DEFAULT_CONFIG.match_tolerance_px))
    )
    y0 = max(0, int(round(reference_y)) - half_window)
    y1 = min(
        directional.shape[0],
        int(round(reference_y)) + half_window + 1,
    )
    if y1 <= y0:
        return {**empty, "reason": "local_reference_rows_unavailable"}
    local_profile = np.median(
        directional[y0:y1], axis=0
    ).astype(np.float64)
    local_evidence = [
        _local_reference_profile_basin_evidence(
            local_profile, basin, reference_y
        )
        for basin in selected_basins
    ]
    if any(not item["supported"] for item in local_evidence):
        return {
            **empty,
            "applied": True,
            "triggered": True,
            "reason": "local_reference_contrast_not_verified",
            "local_evidence": local_evidence,
        }

    recovered_basins = []
    for basin, evidence in zip(selected_basins, local_evidence):
        recovered_basins.append(
            {
                **basin,
                "source": "adjacent_separator_paths_local_contrast_recovery",
                "verified": True,
                "rejection_reasons": [],
                "local_reference_contrast_evidence": evidence,
            }
        )
    basin_ids = {
        role: basin["basin_id"]
        for role, basin in zip(
            ("left", "clicked", "right"), recovered_basins
        )
    }
    hypothesis = {
        "hypothesis_id": "JH01_LCR",
        "reference_relation": "inside_basin",
        "separator_ids": {
            role: selection[role] for role in separator.ROLE_ORDER
        },
        "basin_ids": basin_ids,
        "strictly_continuous": True,
        "separator_sequence": selected_sequence,
        "basin_sequence": [
            basin_ids[role] for role in ("left", "clicked", "right")
        ],
    }
    recovered_ids = set(basin_ids.values())
    recovery_graph = {
        **graph,
        "verified_basins": [
            basin
            for basin in graph.get("verified_basins", [])
            if basin["basin_id"] not in recovered_ids
        ]
        + recovered_basins,
    }
    geometry = joint._basin_geometry(  # noqa: SLF001
        hypothesis,
        recovery_graph,
        reference_x,
        roi_bounds_global["x0"],
    )
    pitch_evidence = joint._pitch_evidence_for_geometry(  # noqa: SLF001
        pitch_result,
        geometry,
        joint.DEFAULT_CONFIG,
    )
    if (
        pitch_evidence["joint_decision"]
        != "high_pitch_supports_geometry"
        or pitch_evidence["rejects_geometry"]
    ):
        return {
            **empty,
            "applied": True,
            "triggered": True,
            "reason": "high_pitch_does_not_support_recovery",
            "pitch_evidence": pitch_evidence,
            "local_evidence": local_evidence,
        }

    recovered_hypothesis = {
        **hypothesis,
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
            "local_contrast_recovery": ALGORITHM_REVISION,
        },
        "local_contrast_recovery": True,
        "atomic": True,
    }
    return {
        **empty,
        "applied": True,
        "triggered": True,
        "success": True,
        "reason": None,
        "source_failure": "dark_basin_sequence_not_verified",
        "local_row_range_roi": [y0, y1],
        "edge_gap_widths_px": edge_gaps.tolist(),
        "local_evidence": local_evidence,
        "pitch_evidence": pitch_evidence,
        "selected_basins": recovered_basins,
        "hypothesis": recovered_hypothesis,
    }


def _recover_local_dark_valley_sequence(
    image_gray: np.ndarray,
    roi_bounds_global: dict,
    direction: str,
    separator_result: dict,
    graph: dict,
    hypothesis: dict | None,
    pitch_result: dict | None = None,
    allow_single_clicked_track: bool = False,
    staged_trigger: str | None = None,
) -> dict:
    """Recover direct black-stripe neighbors from ordered raw valleys."""

    if hypothesis is None:
        if pitch_result is None:
            return {
                "applied": False,
                "triggered": False,
                "success": False,
                "reason": "pitch_result_missing",
            }
        return _recover_contrast_limited_separator_sequence(
            image_gray,
            roi_bounds_global,
            direction,
            separator_result,
            graph,
            pitch_result,
        )

    empty = {
        "applied": False,
        "triggered": False,
        "success": False,
        "reason": "trigger_not_met",
    }
    if hypothesis.get("reference_relation") != "inside_basin":
        return empty
    if direction != "vertical":
        return {**empty, "reason": "non_vertical_direction"}
    basin_by_id = {
        item["basin_id"]: item
        for item in graph.get("verified_basins", [])
    }
    basin_ids = hypothesis.get("basin_ids", {})
    if any(
        basin_ids.get(role) not in basin_by_id
        for role in ("left", "clicked", "right")
    ):
        return {**empty, "reason": "basin_paths_missing"}
    original_basins = {
        role: basin_by_id[basin_ids[role]]
        for role in ("left", "clicked", "right")
    }
    raw_roi = separator.extract_raw_roi(image_gray, roi_bounds_global)
    directional = separator._directional_roi(  # noqa: SLF001
        raw_roi, direction
    )
    band_centers_y = np.asarray(
        original_basins["clicked"]["band_centers_y_roi"],
        dtype=np.float64,
    )
    outer_left = np.asarray(
        original_basins["left"]["left_x_by_band_roi"],
        dtype=np.float64,
    )
    outer_right = np.asarray(
        original_basins["right"]["right_x_by_band_roi"],
        dtype=np.float64,
    )
    tracks, band_profiles = _ordered_local_valley_tracks(
        directional,
        band_centers_y,
        outer_left,
        outer_right,
    )
    reference_y = float(separator_result["reference_y_roi"])
    reference_x = float(separator_result["reference_x_roi"])
    track_geometry = []
    for index, track in enumerate(tracks):
        center = float(
            np.interp(
                reference_y,
                band_centers_y,
                track["minimum_x_by_band_roi"],
            )
        )
        left = float(
            np.interp(
                reference_y,
                band_centers_y,
                track["left_x_by_band_roi"],
            )
        )
        right = float(
            np.interp(
                reference_y,
                band_centers_y,
                track["right_x_by_band_roi"],
            )
        )
        track_geometry.append(
            {"index": index, "center": center, "left": left, "right": right}
        )

    if track_geometry:
        median_width = float(
            np.median(
                [item["right"] - item["left"] for item in track_geometry]
            )
        )
        kept = [
            (track, item)
            for track, item in zip(tracks, track_geometry)
            if (
                (item["right"] - item["left"]) / median_width
                >= separator.DEFAULT_CONFIG.minimum_full_height_order_ratio
                and (item["right"] - item["left"]) / median_width
                <= separator.DEFAULT_CONFIG.maximum_dark_basin_edge_gap_ratio
            )
        ]
        tracks = [item[0] for item in kept]
        track_geometry = [
            {**item[1], "index": index}
            for index, item in enumerate(kept)
        ]

    clicked_basin = original_basins["clicked"]
    clicked_left = float(
        np.interp(
            reference_y,
            band_centers_y,
            clicked_basin["left_x_by_band_roi"],
        )
    )
    clicked_right = float(
        np.interp(
            reference_y,
            band_centers_y,
            clicked_basin["right_x_by_band_roi"],
        )
    )
    clicked_track_count = sum(
        clicked_left < item["center"] < clicked_right
        for item in track_geometry
    )
    if clicked_track_count < 2 and not allow_single_clicked_track:
        return {
            **empty,
            "applied": True,
            "reason": "clicked_basin_has_fewer_than_two_stable_valleys",
            "verified_track_count": len(tracks),
            "clicked_track_count": clicked_track_count,
        }

    triggered = {
        **empty,
        "applied": True,
        "triggered": True,
        "reason": "recovery_not_yet_validated",
        "verified_track_count": len(tracks),
        "clicked_track_count": clicked_track_count,
        "staged_trigger": staged_trigger,
        "single_clicked_track_allowed": allow_single_clicked_track,
    }
    reference_matches = [
        item
        for item in track_geometry
        if item["left"] < reference_x < item["right"]
    ]
    if len(reference_matches) != 1:
        return {
            **triggered,
            "reason": "reference_valley_not_unique",
            "reference_match_count": len(reference_matches),
        }
    clicked_index = reference_matches[0]["index"]
    if clicked_index == 0 or clicked_index == len(tracks) - 1:
        return {**triggered, "reason": "direct_neighbor_missing"}
    selected_indices = {
        "left": clicked_index - 1,
        "clicked": clicked_index,
        "right": clicked_index + 1,
    }

    selected_basins = {}
    center_debug = {}
    for role, index in selected_indices.items():
        basin, calculation = _calculate_local_valley_basin(
            tracks[index],
            index,
            band_centers_y,
            band_profiles,
            reference_y,
        )
        center_debug[role] = calculation
        if basin is None:
            return {
                **triggered,
                "reason": f"{role}_center_not_reliable",
                "center_calculation": center_debug,
            }
        selected_basins[role] = basin

    center_paths = {
        role: np.asarray(basin["center_x_by_band_roi"], dtype=np.float64)
        for role, basin in selected_basins.items()
    }
    left_gaps = center_paths["clicked"] - center_paths["left"]
    right_gaps = center_paths["right"] - center_paths["clicked"]
    if np.any(left_gaps <= 0) or np.any(right_gaps <= 0):
        return {**triggered, "reason": "recovered_path_order_conflict"}
    for gaps in (left_gaps, right_gaps):
        median_gap = float(np.median(gaps))
        if (
            median_gap <= 0
            or float(np.min(gaps)) / median_gap
            < separator.DEFAULT_CONFIG.minimum_full_height_order_ratio
        ):
            return {**triggered, "reason": "recovered_path_order_unstable"}

    all_centers = np.asarray(
        [item["center"] for item in track_geometry], dtype=np.float64
    )
    spacings = np.diff(all_centers)
    local_scale = float(np.median(spacings))
    selected_spacing = np.asarray(
        [
            selected_basins["clicked"]["center_x_at_reference_roi"]
            - selected_basins["left"]["center_x_at_reference_roi"],
            selected_basins["right"]["center_x_at_reference_roi"]
            - selected_basins["clicked"]["center_x_at_reference_roi"],
        ],
        dtype=np.float64,
    )
    scale_ratios = selected_spacing / local_scale
    if any(
        ratio < separator.DEFAULT_CONFIG.minimum_full_height_order_ratio
        or ratio
        > separator.DEFAULT_CONFIG.maximum_dark_basin_edge_gap_ratio
        for ratio in scale_ratios
    ):
        return {
            **triggered,
            "reason": "recovered_local_scale_conflict",
            "local_scale_px": local_scale,
            "selected_scale_ratios": scale_ratios.tolist(),
        }

    left_center = selected_basins["left"]["center_x_at_reference_roi"]
    right_center = selected_basins["right"]["center_x_at_reference_roi"]
    if not left_center < reference_x < right_center:
        return {**triggered, "reason": "recovered_centers_do_not_straddle_reference"}

    recovered_pitch_evidence = _recovered_valley_pitch_evidence(
        pitch_result,
        selected_spacing,
    )
    if recovered_pitch_evidence["rejects_geometry"]:
        return {
            **triggered,
            "reason": "recovered_high_pitch_geometry_conflict",
            "pitch_evidence": recovered_pitch_evidence,
        }

    role_ids = {
        role: basin["basin_id"] for role, basin in selected_basins.items()
    }
    recovered_geometry = {
        "reference_relation": "inside_basin",
        "left_distance_px": reference_x - left_center,
        "right_distance_px": right_center - reference_x,
        "clicked_basin_width_px": selected_basins["clicked"][
            "width_at_reference_px"
        ],
        "local_valley_center_spacing_px": selected_spacing.tolist(),
        "local_valley_scale_px": local_scale,
        "basins": {
            role: {
                "basin_id": basin["basin_id"],
                "center_x_at_reference_roi": basin[
                    "center_x_at_reference_roi"
                ],
                "center_x_at_reference_global": (
                    roi_bounds_global["x0"]
                    + basin["center_x_at_reference_roi"]
                ),
                "width_at_reference_px": basin["width_at_reference_px"],
            }
            for role, basin in selected_basins.items()
        },
    }
    recovered_hypothesis = {
        **hypothesis,
        "hypothesis_id": f"{hypothesis['hypothesis_id']}_LVR",
        "basin_ids": role_ids,
        "basin_sequence": [
            role_ids[role] for role in ("left", "clicked", "right")
        ],
        "separator_ids": None,
        "separator_sequence": [],
        "strictly_continuous": True,
        "geometry": recovered_geometry,
        "pitch_evidence": recovered_pitch_evidence,
        "provenance": {
            **hypothesis["provenance"],
            "local_valley_recovery": ALGORITHM_REVISION,
        },
        "local_valley_recovery": True,
        "atomic": True,
    }
    return {
        **triggered,
        "success": True,
        "reason": None,
        "reference_track_index": clicked_index,
        "selected_track_indices": selected_indices,
        "local_scale_px": local_scale,
        "selected_scale_ratios": scale_ratios.tolist(),
        "pitch_evidence": recovered_pitch_evidence,
        "third_valley_conflict": False,
        "center_calculation": center_debug,
        "selected_basins": list(selected_basins.values()),
        "hypothesis": recovered_hypothesis,
    }


def _recovered_valley_pitch_evidence(
    pitch_result: dict | None,
    selected_spacing: np.ndarray,
) -> dict:
    """Reapply the frozen pitch safety bounds to recovered valley spacing."""

    if pitch_result is None:
        return {
            "confidence": "unavailable",
            "success_eligible": False,
            "diagnostic_pitch_px": None,
            "usable_pitch_px": None,
            "harmonic_ambiguity": {"detected": False},
            "geometry_spacing_ratios": [],
            "joint_decision": "unavailable_no_effect",
            "rejects_geometry": False,
        }
    harmonically_safe_high = bool(
        pitch_result["success_eligible"]
        and pitch_result["confidence"] == "high"
        and not pitch_result["harmonic_ambiguity"]["detected"]
        and pitch_result["usable_pitch_px"] is not None
    )
    evidence = {
        "algorithm_revision": pitch_result["algorithm_revision"],
        "configuration_checksum": pitch_result["configuration_checksum"],
        "confidence": pitch_result["confidence"],
        "success_eligible": pitch_result["success_eligible"],
        "harmonic_ambiguity": pitch_result["harmonic_ambiguity"],
        "diagnostic_pitch_px": pitch_result["diagnostic_pitch_px"],
        "usable_pitch_px": (
            pitch_result["usable_pitch_px"] if harmonically_safe_high else None
        ),
        "unavailable_reason": pitch_result["unavailable_reason"],
        "geometry_spacing_ratios": [],
        "joint_decision": "diagnostics_only",
        "rejects_geometry": False,
    }
    if not harmonically_safe_high:
        if pitch_result["confidence"] == "unavailable":
            evidence["joint_decision"] = "unavailable_no_effect"
        return evidence
    pitch_px = float(pitch_result["usable_pitch_px"])
    ratios = [float(spacing / pitch_px) for spacing in selected_spacing]
    evidence["geometry_spacing_ratios"] = ratios
    conflict = any(
        ratio < joint.DEFAULT_CONFIG.high_pitch_minimum_spacing_ratio
        or ratio > joint.DEFAULT_CONFIG.high_pitch_maximum_spacing_ratio
        for ratio in ratios
    )
    evidence["rejects_geometry"] = conflict
    evidence["joint_decision"] = (
        "high_pitch_geometry_conflict"
        if conflict
        else "high_pitch_supports_geometry"
    )
    return evidence


def _intermediate_dark_profile_candidate(
    profile: np.ndarray,
    clicked_center_x: float,
    selected_center_x: float,
) -> dict | None:
    """Find one dark valley strictly between two flanking bright ridges."""

    left_center, right_center = sorted(
        (float(clicked_center_x), float(selected_center_x))
    )
    left = int(np.clip(round(left_center), 0, profile.size - 1))
    right = int(np.clip(round(right_center), 0, profile.size - 1))
    if right - left < 5:
        return None
    local = profile[left : right + 1]
    low, high = (
        float(np.percentile(local, quantile))
        for quantile in (10, 90)
    )
    dynamic = high - low
    if dynamic < MINIMUM_RAW_GRAY_DYNAMIC:
        return None
    maxima = [
        index
        for index in range(left + 1, right)
        if (
            profile[index] >= profile[index - 1]
            and profile[index] >= profile[index + 1]
        )
    ]
    if len(maxima) < 2:
        return None
    left_peak = maxima[0]
    right_peak = maxima[-1]
    search_left = left_peak + 1
    search_right = right_peak - 1
    if search_right - search_left < 2:
        return None
    dark_x = search_left + int(
        np.argmin(profile[search_left : search_right + 1])
    )
    left_values = profile[left_peak:dark_x]
    right_values = profile[dark_x + 1 : right_peak + 1]
    if left_values.size == 0 or right_values.size == 0:
        return None
    dark_gray = float(profile[dark_x])
    left_bright = float(np.max(left_values))
    right_bright = float(np.max(right_values))
    flank_gray = min(left_bright, right_bright)
    normalized_contrast = (flank_gray - dark_gray) / dynamic
    return {
        "x_roi": float(dark_x),
        "supported": bool(
            normalized_contrast
            >= separator.DEFAULT_CONFIG.minimum_dark_basin_contrast
        ),
        "dark_gray": dark_gray,
        "left_bright_gray": left_bright,
        "right_bright_gray": right_bright,
        "normalized_contrast": float(normalized_contrast),
    }


def _raw_clicked_dark_center(
    profile: np.ndarray,
    reference_x_roi: float,
    left_selected_x: float,
    right_selected_x: float,
) -> float:
    """Locate the clicked raw-gray dark valley without using basin IDs."""

    left, right = sorted(
        (
            int(np.clip(round(left_selected_x), 0, profile.size - 1)),
            int(np.clip(round(right_selected_x), 0, profile.size - 1)),
        )
    )
    if right - left < 5:
        return float(reference_x_roi)
    local = profile[left : right + 1]
    low, high = (
        float(np.percentile(local, quantile))
        for quantile in (10, 90)
    )
    if high - low < MINIMUM_RAW_GRAY_DYNAMIC:
        return float(reference_x_roi)
    dark_limit = low + 0.5 * (high - low)
    minima = [
        index
        for index in range(left + 1, right)
        if (
            profile[index] <= dark_limit
            and profile[index] <= profile[index - 1]
            and profile[index] <= profile[index + 1]
        )
    ]
    if not minima:
        return float(left + int(np.argmin(local)))
    return float(
        min(minima, key=lambda index: abs(index - reference_x_roi))
    )


def _intermediate_dark_basin_evidence(
    image_gray: np.ndarray,
    roi_bounds_global: dict,
    direction: str,
    separator_result: dict,
    graph: dict,
    hypothesis: dict,
) -> dict:
    """Validate extra raw-gray dark valleys between clicked/selected stripes."""

    empty = {
        "applied": False,
        "reference_relation": hypothesis.get("reference_relation"),
        "sides": [],
    }
    if hypothesis.get("reference_relation") != "inside_basin":
        return empty
    if direction != "vertical":
        return {**empty, "unavailable_reason": "non_vertical_direction"}
    basin_by_id = {
        item["basin_id"]: item
        for item in graph.get("verified_basins", [])
    }
    basin_ids = hypothesis.get("basin_ids", {})
    if any(
        basin_ids.get(role) not in basin_by_id
        for role in ("clicked", "left", "right")
    ):
        return {**empty, "unavailable_reason": "basin_paths_missing"}

    basins = {
        role: basin_by_id[basin_ids[role]]
        for role in ("clicked", "left", "right")
    }
    raw_roi = separator.extract_raw_roi(image_gray, roi_bounds_global)
    directional = separator._directional_roi(  # noqa: SLF001
        raw_roi,
        direction,
    )
    band_bounds = separator._band_bounds(  # noqa: SLF001
        directional.shape[0],
        separator.DEFAULT_CONFIG.band_count,
    )
    reference_x = float(separator_result["reference_x_roi"])
    band_centers_y = np.asarray(
        basins["clicked"]["band_centers_y_roi"],
        dtype=np.float64,
    )
    paths = {
        role: np.asarray(
            basins[role]["center_x_by_band_roi"],
            dtype=np.float64,
        )
        for role in ("left", "right")
    }
    clicked_by_band = []
    band_profiles = []
    for band_index, (y0, y1) in enumerate(band_bounds):
        profile = separator._smooth_profile(  # noqa: SLF001
            np.median(
                directional[y0:y1].astype(np.float64),
                axis=0,
            ),
            separator.DEFAULT_CONFIG.profile_smoothing_sigma_px,
        )
        band_profiles.append(profile)
        clicked_by_band.append(
            _raw_clicked_dark_center(
                profile,
                reference_x,
                paths["left"][band_index],
                paths["right"][band_index],
            )
        )
    clicked_path = np.asarray(clicked_by_band, dtype=np.float64)

    side_results = []
    minimum_fraction = (
        separator.DEFAULT_CONFIG.minimum_dark_basin_band_fraction
    )
    for side in ("left", "right"):
        band_items = []
        for band_index, profile in enumerate(band_profiles):
            item = _intermediate_dark_profile_candidate(
                profile,
                clicked_path[band_index],
                paths[side][band_index],
            )
            if item is not None:
                band_items.append({"band_index": band_index, **item})

        row_items = []
        for row_index in range(directional.shape[0]):
            profile = separator._smooth_profile(  # noqa: SLF001
                directional[row_index].astype(np.float64),
                separator.DEFAULT_CONFIG.profile_smoothing_sigma_px,
            )
            left_at_row = float(
                np.interp(row_index, band_centers_y, paths["left"])
            )
            right_at_row = float(
                np.interp(row_index, band_centers_y, paths["right"])
            )
            clicked_at_row = _raw_clicked_dark_center(
                profile,
                reference_x,
                left_at_row,
                right_at_row,
            )
            selected_at_row = float(
                np.interp(row_index, band_centers_y, paths[side])
            )
            item = _intermediate_dark_profile_candidate(
                profile,
                clicked_at_row,
                selected_at_row,
            )
            if item is not None:
                row_items.append({"row_index": row_index, **item})

        supported_bands = [
            item for item in band_items if item["supported"]
        ]
        supported_rows = [
            item for item in row_items if item["supported"]
        ]
        band_support = (
            len(supported_bands) / len(band_items)
            if band_items
            else 0.0
        )
        row_support = (
            len(supported_rows) / len(row_items)
            if row_items
            else 0.0
        )
        x_values = np.asarray(
            [item["x_roi"] for item in band_items],
            dtype=np.float64,
        )
        dark_x_roi = (
            float(np.median(x_values)) if x_values.size else None
        )
        steps = np.abs(np.diff(x_values))
        mean_step = float(np.mean(steps)) if steps.size else 0.0
        normalized_contrast = (
            float(
                np.median(
                    [
                        item["normalized_contrast"]
                        for item in band_items
                    ]
                )
            )
            if band_items
            else None
        )
        verified = bool(
            band_support >= minimum_fraction
            and row_support >= minimum_fraction
            and normalized_contrast is not None
            and normalized_contrast
            >= separator.DEFAULT_CONFIG.minimum_dark_basin_contrast
            and mean_step
            <= separator.DEFAULT_CONFIG.maximum_role_path_mean_step_px
        )
        side_results.append(
            {
                "side": side,
                "verified": verified,
                "dark_basin_x_roi": dark_x_roi,
                "dark_basin_x_global": (
                    None
                    if dark_x_roi is None
                    else dark_x_roi + roi_bounds_global["x0"]
                ),
                "band_support": band_support,
                "row_support": row_support,
                "median_normalized_contrast": normalized_contrast,
                "mean_band_step_px": mean_step,
                "x_mad_px": (
                    None
                    if not x_values.size
                    else float(
                        np.median(
                            np.abs(x_values - dark_x_roi)
                        )
                    )
                ),
                "x_range_px": (
                    None if not x_values.size else float(np.ptp(x_values))
                ),
            }
        )
    return {
        "applied": True,
        "reference_relation": "inside_basin",
        "minimum_band_fraction": minimum_fraction,
        "minimum_row_fraction": minimum_fraction,
        "minimum_normalized_contrast": (
            separator.DEFAULT_CONFIG.minimum_dark_basin_contrast
        ),
        "maximum_mean_band_step_px": (
            separator.DEFAULT_CONFIG.maximum_role_path_mean_step_px
        ),
        "sides": side_results,
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
