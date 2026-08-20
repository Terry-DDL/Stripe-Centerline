"""Stage-3 development-only basin graph and raw-pitch joint prototype.

The graph is created exclusively from frozen separator-path candidates.
Adjacent separator paths may define one dark basin; no track, recovery, or
pitch result can create, split, delete, or relabel a basin.

Frozen raw pitch v3 is conditional evidence.  A harmonically safe high result
may support or reject an otherwise verified separator hypothesis.  Medium,
low, and unavailable pitch remain diagnostics and cannot change eligibility.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import raw_local_pitch_prototype_v3 as raw_pitch
from tools import separator_path_prototype as separator


ALGORITHM_REVISION = "basin_graph_raw_pitch_joint_stage3_v1_1"
FROZEN_SEPARATOR_COMMIT = (
    "136245853757fd7ade58537ee523fbf407042654"
)
FROZEN_SEPARATOR_CONFIGURATION_CHECKSUM = (
    "08cadf043e3061d24e4e90d332e3dceed625824c9625ebedbdad91dfb02204f4"
)
FROZEN_RAW_PITCH_COMMIT = (
    "7fdcf03e13a9c9e0887e266c574d94f45444728b"
)
FROZEN_RAW_PITCH_CONFIGURATION_CHECKSUM = (
    "d1df32d7130d89f0d3e2c6611ad2a1bac11d97ea0252e2de297b63c916e40b49"
)
FROZEN_STAGE3_INSIDE_GEOMETRY_CHECKSUM = (
    "7e970a26969ca8d852c8655b021e7ac12283a536f443be1b2eb4186fc32070fd"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "basin_graph_joint_stage3_v1_1"
ENABLE_LOCAL_REFERENCE_ADJACENCY_VALIDATION = True


@dataclass(frozen=True)
class BasinGraphJointConfig:
    """Stage-3-only safety parameters.

    These thresholds do not participate in separator candidate generation or
    raw-pitch estimation.
    """

    high_pitch_minimum_spacing_ratio: float = 0.67
    high_pitch_maximum_spacing_ratio: float = 1.50
    local_reference_minimum_path_support: float = 0.75
    local_reference_maximum_mean_step_px: float = 2.0
    local_reference_maximum_edge_gap_ratio: float = 6.0
    local_reference_maximum_inside_basin_fraction: float = 0.25
    local_reference_band_count: int = 3
    local_reference_minimum_contrast_fraction: float = 2.0 / 3.0
    local_reference_maximum_center_brightness_excess: float = 0.05


DEFAULT_CONFIG = BasinGraphJointConfig()


def canonical_configuration(config: BasinGraphJointConfig) -> dict:
    return {
        "algorithm_revision": ALGORITHM_REVISION,
        "frozen_dependencies": {
            "separator": {
                "commit": FROZEN_SEPARATOR_COMMIT,
                "configuration_checksum": (
                    FROZEN_SEPARATOR_CONFIGURATION_CHECKSUM
                ),
            },
            "raw_pitch_v3": {
                "commit": FROZEN_RAW_PITCH_COMMIT,
                "configuration_checksum": (
                    FROZEN_RAW_PITCH_CONFIGURATION_CHECKSUM
                ),
            },
        },
        "parameters": asdict(config),
    }


def configuration_checksum(config: BasinGraphJointConfig) -> str:
    payload = json.dumps(
        canonical_configuration(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_frozen_dependencies() -> dict:
    separator_checksum = separator.configuration_checksum(
        separator.DEFAULT_CONFIG
    )
    pitch_checksum = raw_pitch.configuration_checksum(
        raw_pitch.DEFAULT_CONFIG
    )
    if (
        separator_checksum
        != FROZEN_SEPARATOR_CONFIGURATION_CHECKSUM
    ):
        raise RuntimeError("frozen separator configuration changed")
    if pitch_checksum != FROZEN_RAW_PITCH_CONFIGURATION_CHECKSUM:
        raise RuntimeError("frozen raw pitch v3 configuration changed")
    return {
        "separator_configuration_checksum": separator_checksum,
        "raw_pitch_configuration_checksum": pitch_checksum,
    }


def _accepted_paths_in_reference_order(
    separator_result: dict,
) -> list[dict]:
    accepted = [
        path
        for path in separator_result["candidates"]
        if path["accepted"]
    ]
    reference_y = separator_result["reference_y_roi"]
    for path in accepted:
        path["x_at_reference_roi"] = separator._path_x_at_y(
            path,
            reference_y,
        )
    return sorted(
        accepted,
        key=lambda path: (
            path["x_at_reference_roi"],
            path["candidate_id"],
        ),
    )


def _basin_id(left_id: str, right_id: str) -> str:
    return f"B_{left_id}_{right_id}"


def _basin_center_darkness_evidence(
    left: dict,
    right: dict,
    evidence: dict,
) -> dict:
    """Measure whether the geometric basin center is an internal bright ridge."""

    left_x = np.rint(left["x_by_band_roi"]).astype(int)
    right_x = np.rint(right["x_by_band_roi"]).astype(int)
    excess_by_band = []
    for band_index, (left_value, right_value) in enumerate(
        zip(left_x, right_x)
    ):
        x0, x1 = sorted((int(left_value), int(right_value)))
        if x1 - x0 < 3:
            excess_by_band.append(float("inf"))
            continue
        profile = evidence["profiles"][band_index]
        center_x = int(round((x0 + x1) / 2.0))
        center_window = profile[
            max(x0 + 1, center_x - 1) : min(x1, center_x + 2)
        ]
        interior = profile[x0 + 1 : x1]
        local = profile[
            max(0, x0 - 5) : min(profile.size, x1 + 6)
        ]
        center_level = float(np.median(center_window))
        robust_dark_level = float(np.percentile(interior, 40))
        scale = max(
            float(
                np.percentile(local, 90)
                - np.percentile(local, 10)
            ),
            8.0,
        )
        excess_by_band.append(
            (center_level - robust_dark_level) / scale
        )
    return {
        "normalized_center_brightness_excess_by_band": excess_by_band,
        "provenance": "raw_gray_band_profile_center_vs_interior_p40",
    }


def validate_output_basin_center_darkness(
    hypothesis: dict,
    graph: dict,
    reference_y_roi: float,
    config: JointConfig = DEFAULT_CONFIG,
) -> dict:
    """Require each reported basin center to remain in robust dark interior.

    This is an output safety invariant, so it deliberately checks only the
    left/right basins whose centers are returned to the caller.  Clicked and
    auxiliary basins cannot invalidate an otherwise sound reported pair.
    """

    basin_by_id = {
        basin["basin_id"]: basin
        for basin in graph.get("basin_candidates", [])
    }
    audits = []
    for role in ("left", "right"):
        basin_id = hypothesis.get("basin_ids", {}).get(role)
        basin = basin_by_id.get(basin_id)
        if basin is None or "center_darkness_evidence" not in basin:
            return {
                "success": False,
                "reason": "output_basin_center_evidence_missing",
                "basin_audits": audits,
            }
        band_centers = np.asarray(
            basin["band_centers_y_roi"],
            dtype=np.float64,
        )
        count = min(config.local_reference_band_count, len(band_centers))
        local_indices = np.argsort(
            np.abs(band_centers - float(reference_y_roi))
        )[:count]
        excess = np.asarray(
            basin["center_darkness_evidence"][
                "normalized_center_brightness_excess_by_band"
            ],
            dtype=np.float64,
        )[local_indices]
        median_excess = float(np.median(excess))
        accepted = bool(
            len(local_indices) == config.local_reference_band_count
            and np.all(np.isfinite(excess))
            and median_excess
            <= config.local_reference_maximum_center_brightness_excess
        )
        audits.append(
            {
                "role": role,
                "basin_id": basin_id,
                "band_indices": sorted(int(value) for value in local_indices),
                "center_brightness_excess": excess.tolist(),
                "median_center_brightness_excess": median_excess,
                "maximum_center_brightness_excess": (
                    config.local_reference_maximum_center_brightness_excess
                ),
                "accepted": accepted,
            }
        )
    diagnostic_success = all(audit["accepted"] for audit in audits)
    partial_local_dark_evidence = bool(
        audits
        and all(
            any(
                value
                <= config.local_reference_maximum_center_brightness_excess
                for value in audit["center_brightness_excess"]
            )
            for audit in audits
        )
    )
    structurally_bound_reference = bool(
        hypothesis.get("reference_relation") == "on_separator"
        and hypothesis.get("strictly_continuous") is True
        and hypothesis.get("reference_separator_id")
        and len(hypothesis.get("basin_sequence", [])) == 2
        and partial_local_dark_evidence
    )
    success = bool(diagnostic_success or structurally_bound_reference)
    return {
        "success": success,
        "reason": (
            "output_basin_centers_dark"
            if diagnostic_success
            else (
                "output_basin_center_darkness_diagnostic_only"
                if structurally_bound_reference
                else "output_basin_center_not_dark"
            )
        ),
        "veto_applied": not success,
        "partial_local_dark_evidence": partial_local_dark_evidence,
        "basin_audits": audits,
    }


def _make_basin_node(
    left: dict,
    right: dict,
    evidence: dict,
) -> dict:
    order = separator._full_height_pair_order(
        left,
        right,
        separator.DEFAULT_CONFIG,
    )
    dark_evidence = separator._dark_basin_evidence(
        left,
        right,
        evidence,
        separator.DEFAULT_CONFIG,
    )
    left_x = np.asarray(left["x_by_band_roi"], dtype=np.float64)
    right_x = np.asarray(right["x_by_band_roi"], dtype=np.float64)
    width = right_x - left_x
    center = (left_x + right_x) / 2.0
    verified = bool(
        order["stable_full_height_order"]
        and dark_evidence["verified"]
    )
    rejection_reasons = []
    if order["crossing_band_indices"]:
        rejection_reasons.append("separator_paths_cross")
    if not order["stable_full_height_order"]:
        rejection_reasons.append("separator_order_not_stable")
    if not dark_evidence["verified"]:
        rejection_reasons.append("dark_basin_not_verified")
    return {
        "basin_id": _basin_id(
            left["candidate_id"],
            right["candidate_id"],
        ),
        "source": "adjacent_frozen_separator_paths",
        "left_separator_id": left["candidate_id"],
        "right_separator_id": right["candidate_id"],
        "band_centers_y_roi": left["band_centers_y_roi"],
        "left_x_by_band_roi": left_x.tolist(),
        "right_x_by_band_roi": right_x.tolist(),
        "center_x_by_band_roi": center.tolist(),
        "width_by_band_px": width.tolist(),
        "center_x_at_reference_roi": (
            left["x_at_reference_roi"]
            + right["x_at_reference_roi"]
        )
        / 2.0,
        "width_at_reference_px": (
            right["x_at_reference_roi"]
            - left["x_at_reference_roi"]
        ),
        "full_height_order": order,
        "dark_basin_evidence": dark_evidence,
        "center_darkness_evidence": _basin_center_darkness_evidence(
            left,
            right,
            evidence,
        ),
        "verified": verified,
        "rejection_reasons": rejection_reasons,
    }


def build_ordered_basin_graph(
    separator_result: dict,
    raw_roi: np.ndarray,
    direction: str,
) -> dict:
    """Create basin nodes only from adjacent frozen separator paths."""

    directional = separator._directional_roi(raw_roi, direction)
    evidence = separator.build_raw_gray_response(
        directional,
        separator.DEFAULT_CONFIG,
    )
    ordered_paths = _accepted_paths_in_reference_order(
        separator_result
    )
    basin_candidates = [
        _make_basin_node(left, right, evidence)
        for left, right in zip(ordered_paths, ordered_paths[1:])
    ]
    verified_basins = [
        basin for basin in basin_candidates if basin["verified"]
    ]
    order_conflicts = [
        {
            "left_separator_id": basin["left_separator_id"],
            "right_separator_id": basin["right_separator_id"],
            "rejection_reasons": basin["rejection_reasons"],
        }
        for basin in basin_candidates
        if (
            "separator_paths_cross" in basin["rejection_reasons"]
            or "separator_order_not_stable"
            in basin["rejection_reasons"]
        )
    ]
    return {
        "source": "frozen_separator_paths_only",
        "ordered_separator_ids": [
            path["candidate_id"] for path in ordered_paths
        ],
        "separator_x_at_reference_roi": {
            path["candidate_id"]: path["x_at_reference_roi"]
            for path in ordered_paths
        },
        "basin_candidates": basin_candidates,
        "verified_basins": verified_basins,
        "verified_basin_ids": [
            basin["basin_id"] for basin in verified_basins
        ],
        "path_order_conflicts": order_conflicts,
    }


def _basin_lookup(graph: dict) -> dict[tuple[str, str], dict]:
    return {
        (
            basin["left_separator_id"],
            basin["right_separator_id"],
        ): basin
        for basin in graph["verified_basins"]
    }


def _inside_basin_relation(
    separator_result: dict,
    graph: dict,
) -> dict:
    if separator_result["status"] != "available":
        return {
            "status": "unavailable",
            "relation": None,
            "hypotheses": [],
            "unavailable_reason": (
                "separator_roles_not_formally_verified"
            ),
        }
    selection = separator_result["selection"]
    separator_ids = {
        role: selection[role]["candidate_id"]
        for role in separator.ROLE_ORDER
    }
    ordered_ids = graph["ordered_separator_ids"]
    selected_sequence = [
        separator_ids[role] for role in separator.ROLE_ORDER
    ]
    indexes = [ordered_ids.index(item) for item in selected_sequence]
    if indexes != list(range(indexes[0], indexes[0] + 4)):
        return {
            "status": "unavailable",
            "relation": "inside_basin",
            "hypotheses": [],
            "unavailable_reason": "intermediate_separator_present",
        }

    lookup = _basin_lookup(graph)
    basin_pairs = [
        (selected_sequence[0], selected_sequence[1]),
        (selected_sequence[1], selected_sequence[2]),
        (selected_sequence[2], selected_sequence[3]),
    ]
    missing_pairs = [pair for pair in basin_pairs if pair not in lookup]
    if missing_pairs:
        return {
            "status": "unavailable",
            "relation": "inside_basin",
            "hypotheses": [],
            "unavailable_reason": "continuous_dark_basins_not_verified",
            "missing_separator_pairs": [
                list(pair) for pair in missing_pairs
            ],
        }
    basins = [lookup[pair] for pair in basin_pairs]
    hypothesis = {
        "hypothesis_id": "JH01",
        "reference_relation": "inside_basin",
        "separator_ids": separator_ids,
        "basin_ids": {
            "left": basins[0]["basin_id"],
            "clicked": basins[1]["basin_id"],
            "right": basins[2]["basin_id"],
        },
        "strictly_continuous": True,
        "separator_sequence": selected_sequence,
        "basin_sequence": [
            basin["basin_id"] for basin in basins
        ],
    }
    return {
        "status": "unique",
        "relation": "inside_basin",
        "hypotheses": [hypothesis],
        "unavailable_reason": None,
    }


def _separator_adjacency_relation(
    separator_result: dict,
    graph: dict,
    pitch_result: dict | None = None,
) -> dict:
    arbitration = separator_result["arbitration_debug"]
    competing_explanations = arbitration.get(
        "competing_explanations",
        [],
    )
    separator_relations = [
        item
        for item in competing_explanations
        if item.get("type") == "reference_on_separator"
    ]
    if len(separator_relations) != 1:
        return {
            "status": "unavailable",
            "relation": None,
            "hypotheses": separator_relations,
            "unavailable_reason": (
                "reference_relationship_not_unique"
            ),
        }
    reference_separator_id = separator_relations[0][
        "candidate_id"
    ]
    other_verified_explanations = [
        item
        for item in competing_explanations
        if (
            item.get("type") != "reference_on_separator"
            and item.get("verified")
            and reference_separator_id
            not in {
                item.get("selection_candidate_ids", {}).get(
                    "left_clicked_boundary"
                ),
                item.get("selection_candidate_ids", {}).get(
                    "right_clicked_boundary"
                ),
            }
        )
    ]
    semantic_tiebreak = separator_relations[0].get(
        "semantic_tiebreak",
        {},
    )
    prefer_reference_separator = bool(
        semantic_tiebreak.get("adopted")
        and semantic_tiebreak.get("resolved_relation")
        == "on_separator"
    )
    ordered_ids = graph["ordered_separator_ids"]
    index = ordered_ids.index(reference_separator_id)
    local_validation = None
    if other_verified_explanations and not prefer_reference_separator:
        if index > 0 and index + 1 < len(ordered_ids):
            local_validation = _local_reference_adjacency_validation(
                separator_result,
                graph,
                reference_separator_id,
                ordered_ids[index - 1],
                ordered_ids[index + 1],
                DEFAULT_CONFIG,
                pitch_result,
            )
        if not local_validation or not local_validation["success"]:
            return {
                "status": "unavailable",
                "relation": "on_separator",
                "hypotheses": other_verified_explanations,
                "unavailable_reason": (
                    "multiple_reasonable_reference_hypotheses"
                ),
                "local_reference_adjacency_validation": local_validation,
            }
    if index == 0 or index + 1 >= len(ordered_ids):
        return {
            "status": "unavailable",
            "relation": "on_separator",
            "hypotheses": [],
            "unavailable_reason": "separator_adjacency_boundary_missing",
        }
    left_id = ordered_ids[index - 1]
    right_id = ordered_ids[index + 1]
    if local_validation is None:
        local_validation = _local_reference_adjacency_validation(
            separator_result,
            graph,
            reference_separator_id,
            left_id,
            right_id,
            DEFAULT_CONFIG,
            pitch_result,
        )
    if local_validation["success"]:
        existing_ids = {
            basin["basin_id"]
            for basin in graph.get("verified_basins", [])
        }
        for basin in local_validation["basins"]:
            if basin["basin_id"] not in existing_ids:
                graph.setdefault("verified_basins", []).append(basin)
                graph.setdefault("verified_basin_ids", []).append(
                    basin["basin_id"]
                )
                existing_ids.add(basin["basin_id"])
    lookup = _basin_lookup(graph)
    left_pair = (left_id, reference_separator_id)
    right_pair = (reference_separator_id, right_id)
    if left_pair not in lookup or right_pair not in lookup:
        return {
            "status": "unavailable",
            "relation": "on_separator",
            "hypotheses": [],
            "unavailable_reason": (
                "separator_adjacent_dark_basins_not_verified"
            ),
            "local_reference_adjacency_validation": local_validation,
        }
    inherited_safety_conflicts = sorted(
        {
            reason
            for explanation in competing_explanations
            if (
                explanation.get("type") != "reference_on_separator"
                and explanation.get("verified") is True
            )
            for reason in explanation.get("rejection_reasons", [])
        }
    )
    if inherited_safety_conflicts and not local_validation["success"]:
        return {
            "status": "unavailable",
            "relation": "on_separator",
            "hypotheses": separator_relations,
            "unavailable_reason": (
                "reference_separator_basin_safety_conflict"
            ),
            "inherited_safety_conflicts": inherited_safety_conflicts,
            "local_reference_adjacency_validation": local_validation,
        }
    hypothesis = {
        "hypothesis_id": "JH01",
        "reference_relation": "on_separator",
        "clicked_separator_id": reference_separator_id,
        "reference_separator_id": reference_separator_id,
        "separator_sequence": [
            left_id,
            reference_separator_id,
            right_id,
        ],
        "basin_ids": {
            "left": lookup[left_pair]["basin_id"],
            "right": lookup[right_pair]["basin_id"],
        },
        "basin_sequence": [
            lookup[left_pair]["basin_id"],
            lookup[right_pair]["basin_id"],
        ],
        "shared_boundary": {
            "separator_id": reference_separator_id,
            "left_basin_id": lookup[left_pair]["basin_id"],
            "right_basin_id": lookup[right_pair]["basin_id"],
        },
        "strictly_continuous": True,
        "semantic_tiebreak": semantic_tiebreak,
        "local_reference_adjacency_validation": local_validation,
    }
    return {
        "status": "unique",
        "relation": "on_separator",
        "hypotheses": [hypothesis],
        "unavailable_reason": None,
    }


def _local_reference_adjacency_validation(
    separator_result: dict,
    graph: dict,
    reference_separator_id: str,
    left_id: str,
    right_id: str,
    config: BasinGraphJointConfig,
    pitch_result: dict | None,
) -> dict:
    """Validate only the two basins touching an established reference separator."""

    if not ENABLE_LOCAL_REFERENCE_ADJACENCY_VALIDATION:
        return {
            "success": False,
            "reason": "staged_local_reference_validation_disabled",
            "basins": [],
        }

    arbitration = separator_result["arbitration_debug"]
    roles = arbitration.get("role_hypotheses", [])
    if len(roles) != 1:
        return {"success": False, "reason": "role_hypothesis_not_unique", "basins": []}
    role = roles[0]
    if role.get("crossing"):
        return {"success": False, "reason": "role_paths_cross", "basins": []}
    hard_conflicts = {
        "path_crossing_full_height",
        "separator_envelopes_overlap",
    }
    if hard_conflicts.intersection(role.get("rejection_reasons", [])):
        return {"success": False, "reason": "hard_path_conflict", "basins": []}
    semantic = (
        (arbitration.get("width_aware_reference", {}) or {}).get(
            "reference_grayscale_semantic_resolution", {}
        )
        or {}
    )
    inside_basin_fraction = float(
        semantic.get("inside_basin_band_fraction", 0.0)
    )
    if (
        inside_basin_fraction
        > config.local_reference_maximum_inside_basin_fraction
    ):
        return {
            "success": False,
            "reason": "reference_semantic_not_separator_local",
            "basins": [],
            "inside_basin_band_fraction": inside_basin_fraction,
        }
    path_metrics = role.get("path_metrics", [])
    supports = [
        float(item["support_fraction"])
        for item in path_metrics
        if item.get("support_fraction") is not None
    ]
    steps = [
        float(item["mean_step_px"])
        for item in path_metrics
        if item.get("mean_step_px") is not None
    ]
    edge_ratio = role.get("edge_gap_ratio")
    global_shape_diagnostic = {
        "minimum_path_support": min(supports) if supports else None,
        "maximum_mean_step_px": max(steps) if steps else None,
        "edge_gap_ratio": edge_ratio,
    }
    if (
        not supports
        or not steps
        or edge_ratio is None
    ):
        return {
            "success": False,
            "reason": "global_shape_diagnostics_missing",
            "basins": [],
            **global_shape_diagnostic,
        }
    global_shape_within_legacy_limits = bool(
        min(supports) >= config.local_reference_minimum_path_support
        and max(steps) <= config.local_reference_maximum_mean_step_px
        and float(edge_ratio)
        <= config.local_reference_maximum_edge_gap_ratio
    )
    if (
        not global_shape_within_legacy_limits
        and semantic.get("classification") != "on_separator"
    ):
        return {
            "success": False,
            "reason": "global_shape_not_locally_overridden",
            "basins": [],
            **global_shape_diagnostic,
        }

    basin_by_pair = {
        (basin["left_separator_id"], basin["right_separator_id"]): basin
        for basin in graph.get("basin_candidates", [])
    }
    requested_pairs = [
        (left_id, reference_separator_id),
        (reference_separator_id, right_id),
    ]
    requested_basins = [basin_by_pair.get(pair) for pair in requested_pairs]
    if any(basin is None for basin in requested_basins):
        return {
            "success": False,
            "reason": "adjacent_basin_candidate_missing",
            "basins": [],
        }
    diagnostic_pitch = (
        pitch_result.get("diagnostic_pitch_px")
        if pitch_result is not None
        else None
    )
    if diagnostic_pitch is None or float(diagnostic_pitch) <= 0:
        return {
            "success": False,
            "reason": "local_reference_pitch_unavailable",
            "basins": [],
        }
    center_spacing = abs(
        float(requested_basins[1]["center_x_at_reference_roi"])
        - float(requested_basins[0]["center_x_at_reference_roi"])
    )
    pitch_ratio = center_spacing / float(diagnostic_pitch)
    if (
        pitch_ratio < config.high_pitch_minimum_spacing_ratio
        or pitch_ratio > config.high_pitch_maximum_spacing_ratio
    ):
        return {
            "success": False,
            "reason": "local_reference_pitch_geometry_conflict",
            "basins": [],
            "diagnostic_pitch_px": float(diagnostic_pitch),
            "basin_center_spacing_px": center_spacing,
            "pitch_ratio": pitch_ratio,
        }
    reference_y = float(separator_result["reference_y_roi"])
    accepted_basins = []
    basin_audits = []
    for pair in requested_pairs:
        basin = basin_by_pair[pair]
        band_centers = np.asarray(basin["band_centers_y_roi"], dtype=np.float64)
        count = min(config.local_reference_band_count, len(band_centers))
        local_indices = np.argsort(np.abs(band_centers - reference_y))[:count]
        gaps = np.asarray(basin["width_by_band_px"], dtype=np.float64)[local_indices]
        contrasts = np.asarray(
            basin["dark_basin_evidence"]["normalized_contrast_by_band"],
            dtype=np.float64,
        )[local_indices]
        contrast_fraction = float(
            np.mean(
                contrasts
                >= separator.DEFAULT_CONFIG.minimum_dark_basin_contrast
            )
        )
        center_excess_by_band = np.asarray(
            basin["center_darkness_evidence"][
                "normalized_center_brightness_excess_by_band"
            ],
            dtype=np.float64,
        )[local_indices]
        median_center_brightness_excess = float(
            np.median(center_excess_by_band)
        )
        center_is_dark = bool(
            np.all(np.isfinite(center_excess_by_band))
            and median_center_brightness_excess
            <= config.local_reference_maximum_center_brightness_excess
        )
        local_ok = bool(
            len(local_indices) == config.local_reference_band_count
            and np.all(
                gaps
                >= separator.DEFAULT_CONFIG.minimum_dark_basin_edge_gap_px
            )
            and contrast_fraction
            >= config.local_reference_minimum_contrast_fraction
            and float(np.median(contrasts))
            >= separator.DEFAULT_CONFIG.minimum_dark_basin_contrast
            and center_is_dark
        )
        basin_audits.append(
            {
                "basin_id": basin["basin_id"],
                "band_indices": sorted(int(value) for value in local_indices),
                "local_gap_px": gaps.tolist(),
                "local_contrast": contrasts.tolist(),
                "local_contrast_fraction": contrast_fraction,
                "local_center_brightness_excess": (
                    center_excess_by_band.tolist()
                ),
                "median_center_brightness_excess": (
                    median_center_brightness_excess
                ),
                "maximum_center_brightness_excess": (
                    config.local_reference_maximum_center_brightness_excess
                ),
                "center_is_dark": center_is_dark,
                "accepted": local_ok,
            }
        )
        if not local_ok:
            return {
                "success": False,
                "reason": (
                    "click_local_basin_center_not_dark"
                    if not center_is_dark
                    else "click_local_basin_evidence_not_verified"
                ),
                "basins": [],
                "basin_audits": basin_audits,
            }
        accepted_basins.append(
            {
                **basin,
                "verified": True,
                "rejection_reasons": [],
                "source": (
                    "adjacent_frozen_separator_paths_"
                    "click_local_reference_validation"
                ),
                "click_local_reference_validation": basin_audits[-1],
            }
        )
    return {
        "success": True,
        "reason": "click_local_reference_adjacency_verified",
        "basins": accepted_basins,
        "basin_audits": basin_audits,
        "minimum_path_support": min(supports),
        "maximum_mean_step_px": max(steps),
        "edge_gap_ratio": float(edge_ratio),
        "global_shape_diagnostic": global_shape_diagnostic,
        "inside_basin_band_fraction": inside_basin_fraction,
        "diagnostic_pitch_px": float(diagnostic_pitch),
        "basin_center_spacing_px": center_spacing,
        "pitch_ratio": pitch_ratio,
    }
def resolve_reference_relation(
    separator_result: dict,
    graph: dict,
    pitch_result: dict | None = None,
) -> dict:
    """Resolve exactly one basin or separator-adjacency interpretation."""

    if separator_result["status"] == "available":
        return _inside_basin_relation(separator_result, graph)
    if separator_result["unavailable_reason"] == "reference_on_separator":
        return _separator_adjacency_relation(
            separator_result,
            graph,
            pitch_result,
        )
    return {
        "status": "unavailable",
        "relation": None,
        "hypotheses": separator_result[
            "arbitration_debug"
        ].get("competing_explanations", []),
        "unavailable_reason": separator_result["unavailable_reason"],
    }


def _basin_geometry(
    hypothesis: dict,
    graph: dict,
    reference_x_roi: float,
    roi_x0_global: int,
) -> dict:
    basin_by_id = {
        basin["basin_id"]: basin
        for basin in graph["verified_basins"]
    }
    if hypothesis["reference_relation"] == "on_separator":
        basins = {
            role: basin_by_id[basin_id]
            for role, basin_id in hypothesis["basin_ids"].items()
        }
        reference_separator_id = hypothesis[
            "reference_separator_id"
        ]
        separator_x = graph["separator_x_at_reference_roi"][
            reference_separator_id
        ]
        left_center_x = basins["left"][
            "center_x_at_reference_roi"
        ]
        right_center_x = basins["right"][
            "center_x_at_reference_roi"
        ]
        return {
            "reference_relation": "on_separator",
            "reference_separator_x_roi": separator_x,
            "reference_separator_x_global": (
                roi_x0_global + separator_x
            ),
            "reference_separator_distance_px": abs(
                separator_x - reference_x_roi
            ),
            "reference_to_left_basin_center_distance_px": (
                reference_x_roi - left_center_x
            ),
            "reference_to_right_basin_center_distance_px": (
                right_center_x - reference_x_roi
            ),
            "shared_boundary_to_left_basin_center_distance_px": (
                separator_x - left_center_x
            ),
            "shared_boundary_to_right_basin_center_distance_px": (
                right_center_x - separator_x
            ),
            "basin_center_spacing_px": (
                right_center_x - left_center_x
            ),
            "basins": {
                role: {
                    "basin_id": basin["basin_id"],
                    "center_x_at_reference_roi": basin[
                        "center_x_at_reference_roi"
                    ],
                    "center_x_at_reference_global": (
                        roi_x0_global
                        + basin["center_x_at_reference_roi"]
                    ),
                    "width_at_reference_px": basin[
                        "width_at_reference_px"
                    ],
                }
                for role, basin in basins.items()
            },
        }

    role_ids = hypothesis["separator_ids"]
    separator_x = {
        role: graph["separator_x_at_reference_roi"][candidate_id]
        for role, candidate_id in role_ids.items()
    }
    basin_ids = hypothesis["basin_ids"]
    basins = {
        role: basin_by_id[basin_id]
        for role, basin_id in basin_ids.items()
    }
    sequence_x = [
        separator_x[role] for role in separator.ROLE_ORDER
    ]
    return {
        "reference_relation": "inside_basin",
        "separator_x_at_reference_roi": separator_x,
        "separator_spacing_px": np.diff(
            np.asarray(sequence_x, dtype=np.float64)
        ).tolist(),
        "left_distance_px": (
            reference_x_roi
            - separator_x["left_clicked_boundary"]
        ),
        "right_distance_px": (
            separator_x["right_clicked_boundary"]
            - reference_x_roi
        ),
        "clicked_basin_width_px": basins["clicked"][
            "width_at_reference_px"
        ],
        "basins": {
            role: {
                "basin_id": basin["basin_id"],
                "center_x_at_reference_roi": basin[
                    "center_x_at_reference_roi"
                ],
                "width_at_reference_px": basin[
                    "width_at_reference_px"
                ],
            }
            for role, basin in basins.items()
        },
    }


def _pitch_evidence_for_geometry(
    pitch_result: dict,
    geometry: dict,
    config: BasinGraphJointConfig,
) -> dict:
    harmonically_safe_high = bool(
        pitch_result["success_eligible"]
        and pitch_result["confidence"] == "high"
        and not pitch_result["harmonic_ambiguity"]["detected"]
        and pitch_result["usable_pitch_px"] is not None
    )
    evidence = {
        "algorithm_revision": pitch_result["algorithm_revision"],
        "configuration_checksum": pitch_result[
            "configuration_checksum"
        ],
        "confidence": pitch_result["confidence"],
        "success_eligible": pitch_result["success_eligible"],
        "harmonic_ambiguity": pitch_result["harmonic_ambiguity"],
        "diagnostic_pitch_px": pitch_result[
            "diagnostic_pitch_px"
        ],
        "usable_pitch_px": (
            pitch_result["usable_pitch_px"]
            if harmonically_safe_high
            else None
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
    if geometry["reference_relation"] == "inside_basin":
        geometry_spacings = geometry["separator_spacing_px"]
    else:
        geometry_spacings = [geometry["basin_center_spacing_px"]]
    ratios = [
        float(spacing / pitch_px) for spacing in geometry_spacings
    ]
    evidence["geometry_spacing_ratios"] = ratios
    conflict = any(
        ratio < config.high_pitch_minimum_spacing_ratio
        or ratio > config.high_pitch_maximum_spacing_ratio
        for ratio in ratios
    )
    evidence["rejects_geometry"] = conflict
    evidence["joint_decision"] = (
        "high_pitch_geometry_conflict"
        if conflict
        else "high_pitch_supports_geometry"
    )
    return evidence


def run_joint_case(
    image_gray: np.ndarray,
    reference_global: dict,
    roi_bounds_global: dict,
    direction: str = "vertical",
    config: BasinGraphJointConfig = DEFAULT_CONFIG,
) -> dict:
    """Build one immutable separator/basin/pitch hypothesis."""

    dependency_checks = verify_frozen_dependencies()
    separator_result = separator.detect_separator_paths(
        image_gray,
        reference_global,
        roi_bounds_global,
        direction,
        separator.DEFAULT_CONFIG,
    )
    raw_roi = separator.extract_raw_roi(
        image_gray,
        roi_bounds_global,
    )
    graph = build_ordered_basin_graph(
        separator_result,
        raw_roi,
        direction,
    )
    relation = resolve_reference_relation(
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
    if relation["status"] != "unique":
        return {
            "algorithm_revision": ALGORITHM_REVISION,
            "configuration_checksum": configuration_checksum(config),
            "dependency_checks": dependency_checks,
            "status": "unavailable",
            "success": False,
            "unavailable_reason": relation["unavailable_reason"],
            "final_hypothesis": None,
            "debug": debug,
        }
    if len(relation["hypotheses"]) != 1:
        return {
            "algorithm_revision": ALGORITHM_REVISION,
            "configuration_checksum": configuration_checksum(config),
            "dependency_checks": dependency_checks,
            "status": "unavailable",
            "success": False,
            "unavailable_reason": "joint_hypothesis_not_unique",
            "final_hypothesis": None,
            "debug": debug,
        }

    relation_hypothesis = relation["hypotheses"][0]
    geometry = _basin_geometry(
        relation_hypothesis,
        graph,
        separator_result["reference_x_roi"],
        roi_bounds_global["x0"],
    )
    pitch_evidence = _pitch_evidence_for_geometry(
        pitch_result,
        geometry,
        config,
    )
    atomic_hypothesis = {
        **relation_hypothesis,
        "geometry": geometry,
        "pitch_evidence": pitch_evidence,
        "provenance": {
            "separator_algorithm_revision": separator_result[
                "algorithm_revision"
            ],
            "separator_configuration_checksum": separator_result[
                "configuration_checksum"
            ],
            "raw_pitch_algorithm_revision": pitch_result[
                "algorithm_revision"
            ],
            "raw_pitch_configuration_checksum": pitch_result[
                "configuration_checksum"
            ],
        },
        "atomic": True,
    }
    debug["joint_hypotheses"] = [atomic_hypothesis]

    if pitch_evidence["rejects_geometry"]:
        return {
            "algorithm_revision": ALGORITHM_REVISION,
            "configuration_checksum": configuration_checksum(config),
            "dependency_checks": dependency_checks,
            "status": "unavailable",
            "success": False,
            "unavailable_reason": "high_pitch_geometry_conflict",
            "final_hypothesis": None,
            "debug": debug,
        }
    return {
        "algorithm_revision": ALGORITHM_REVISION,
        "configuration_checksum": configuration_checksum(config),
        "dependency_checks": dependency_checks,
        "status": "available",
        "success": True,
        "unavailable_reason": None,
        "final_hypothesis": atomic_hypothesis,
        "debug": debug,
    }


def _evaluate_reference_separator_hypothesis(
    separator_evaluation: dict,
    result: dict,
) -> dict:
    """Compare a formal shared-boundary result with development GT.

    The comparison happens only after inference.  It recognizes a reference
    separator at either annotated clicked boundary, or a stable separator
    between the two annotated clicked boundaries.
    """

    hypothesis = result["final_hypothesis"]
    if (
        not result["success"]
        or hypothesis is None
        or hypothesis["reference_relation"] != "on_separator"
    ):
        return {
            "status": "not_applicable",
            "correct": False,
            "matched_semantics": None,
            "predicted_separator_sequence": [],
            "expected_separator_sequences": [],
        }
    if (
        separator_evaluation["gt_all_ambiguous"]
        or separator_evaluation["gt_unavailable"]
    ):
        return {
            "status": "unsafe_gt_label",
            "correct": False,
            "matched_semantics": None,
            "predicted_separator_sequence": hypothesis[
                "separator_sequence"
            ],
            "expected_separator_sequences": [],
        }

    role_candidate_ids = {
        role: role_result["best_candidate_id"]
        for role, role_result in separator_evaluation["per_role"].items()
        if (
            role_result["numeric_gt"]
            and role_result["candidate_recalled"]
        )
    }
    required_roles = set(separator.ROLE_ORDER)
    if set(role_candidate_ids) != required_roles:
        return {
            "status": "numeric_gt_paths_not_all_recalled",
            "correct": False,
            "matched_semantics": None,
            "predicted_separator_sequence": hypothesis[
                "separator_sequence"
            ],
            "expected_separator_sequences": [],
        }

    predicted = hypothesis["separator_sequence"]
    left_boundary_sequence = [
        role_candidate_ids["left_adjacent"],
        role_candidate_ids["left_clicked_boundary"],
        role_candidate_ids["right_clicked_boundary"],
    ]
    right_boundary_sequence = [
        role_candidate_ids["left_clicked_boundary"],
        role_candidate_ids["right_clicked_boundary"],
        role_candidate_ids["right_adjacent"],
    ]
    expected_sequences = [
        left_boundary_sequence,
        right_boundary_sequence,
    ]
    if predicted == left_boundary_sequence:
        matched_semantics = "reference_is_left_clicked_boundary"
    elif predicted == right_boundary_sequence:
        matched_semantics = "reference_is_right_clicked_boundary"
    else:
        clicked_id = hypothesis["clicked_separator_id"]
        annotated_ids = set(role_candidate_ids.values())
        internal_separator = bool(
            predicted[0]
            == role_candidate_ids["left_clicked_boundary"]
            and predicted[2]
            == role_candidate_ids["right_clicked_boundary"]
            and clicked_id == predicted[1]
            and clicked_id not in annotated_ids
        )
        matched_semantics = (
            "reference_separator_inside_annotated_clicked_basin"
            if internal_separator
            else None
        )
    return {
        "status": "matched" if matched_semantics else "mismatch",
        "correct": matched_semantics is not None,
        "matched_semantics": matched_semantics,
        "predicted_separator_sequence": predicted,
        "expected_separator_sequences": expected_sequences,
    }


def _evaluate_one(
    annotation: dict,
    image_gray: np.ndarray,
    config: BasinGraphJointConfig,
) -> dict:
    separator_evaluation = separator.evaluate_one_annotation(
        annotation,
        image_gray,
        separator.DEFAULT_CONFIG,
    )
    result = run_joint_case(
        image_gray,
        annotation["reference_global"],
        annotation["roi_bounds_global"],
        annotation["direction"],
        config,
    )
    numeric_roles = [
        role
        for role in separator_evaluation["per_role"].values()
        if role["numeric_gt"]
    ]
    all_roles_correct = bool(
        numeric_roles
        and all(role["selected_recalled"] for role in numeric_roles)
    )
    unsafe_gt_label = bool(
        separator_evaluation["gt_all_ambiguous"]
        or separator_evaluation["gt_unavailable"]
    )
    candidate_recall_scorable = bool(
        numeric_roles
        and not unsafe_gt_label
        and not separator_evaluation["gt_reference_ambiguity"][
            "is_ambiguity_case"
        ]
    )
    reference_separator_assessment = (
        _evaluate_reference_separator_hypothesis(
            separator_evaluation,
            result,
        )
    )
    relation = result["debug"]["reference_relation"]["relation"]
    inside_basin_correct = bool(
        relation == "inside_basin"
        and all_roles_correct
        and not unsafe_gt_label
        and not separator_evaluation["gt_reference_ambiguity"][
            "is_ambiguity_case"
        ]
    )
    on_separator_correct = bool(
        relation == "on_separator"
        and reference_separator_assessment["correct"]
        and not unsafe_gt_label
    )
    formal_result_correct = (
        inside_basin_correct or on_separator_correct
    )
    if result["success"] and formal_result_correct:
        classification = "correct_success"
    elif result["success"]:
        classification = "wrong_success"
    else:
        classification = "safe_failure"
    stage3_reference_gate_case = bool(
        relation == "on_separator"
        and (
            result["success"]
            or result["unavailable_reason"]
            in {
                "multiple_reasonable_reference_hypotheses",
                "reference_separator_basin_safety_conflict",
            }
        )
    )
    if stage3_reference_gate_case and result["success"]:
        stage3_transition = (
            "safe_failure_to_correct_success"
            if classification == "correct_success"
            else "safe_failure_to_wrong_success"
        )
    elif stage3_reference_gate_case:
        stage3_transition = "safe_failure_reason_changed"
    else:
        stage3_transition = "unchanged"
    return {
        "sample_id": annotation["sample_id"],
        "image_name": annotation["image_name"],
        "reference_global": annotation["reference_global"],
        "classification": classification,
        "status": result["status"],
        "unavailable_reason": result["unavailable_reason"],
        "success": result["success"],
        "gt_all_ambiguous": separator_evaluation["gt_all_ambiguous"],
        "gt_unavailable": separator_evaluation["gt_unavailable"],
        "gt_reference_ambiguity": separator_evaluation[
            "gt_reference_ambiguity"
        ],
        "candidate_path_recall_count": (
            sum(role["candidate_recalled"] for role in numeric_roles)
            if candidate_recall_scorable
            else 0
        ),
        "numeric_path_count": (
            len(numeric_roles) if candidate_recall_scorable else 0
        ),
        "all_selected_roles_correct": all_roles_correct,
        "formal_result_correct": formal_result_correct,
        "reference_separator_gt_assessment": (
            reference_separator_assessment
        ),
        "stage3_to_stage3_1": {
            "transition": stage3_transition,
            "stage3_status": (
                "unavailable"
                if stage3_reference_gate_case
                else result["status"]
            ),
            "stage3_unavailable_reason": (
                "reference_on_separator_requires_side_choice"
                if stage3_reference_gate_case
                else result["unavailable_reason"]
            ),
            "stage3_1_status": result["status"],
            "stage3_1_unavailable_reason": result[
                "unavailable_reason"
            ],
        },
        "separator_v1_1_status": separator_evaluation["status"],
        "separator_v1_1_selection": separator_evaluation["selection"],
        "final_hypothesis": result["final_hypothesis"],
        "basin_graph": result["debug"]["basin_graph"],
        "reference_relation": result["debug"]["reference_relation"],
        "raw_pitch": {
            key: result["debug"]["raw_pitch_result"][key]
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
        "joint_debug": result["debug"],
    }


def _formal_hypothesis_has_order_conflict(sample: dict) -> bool:
    hypothesis = sample["final_hypothesis"]
    if not sample["success"] or hypothesis is None:
        return False
    sequence = hypothesis["separator_sequence"]
    selected_pairs = set(zip(sequence, sequence[1:]))
    conflict_pairs = {
        (
            conflict["left_separator_id"],
            conflict["right_separator_id"],
        )
        for conflict in sample["basin_graph"]["path_order_conflicts"]
    }
    return bool(selected_pairs & conflict_pairs)


def _summarize(samples: list[dict]) -> dict:
    failure_counts = {}
    pitch_decision_counts = {}
    for sample in samples:
        if sample["unavailable_reason"]:
            failure_counts[sample["unavailable_reason"]] = (
                failure_counts.get(sample["unavailable_reason"], 0) + 1
            )
        hypothesis = sample["final_hypothesis"]
        if hypothesis is not None:
            decision = hypothesis["pitch_evidence"]["joint_decision"]
        else:
            joint_hypotheses = sample["joint_debug"].get(
                "joint_hypotheses",
                [],
            )
            decision = (
                joint_hypotheses[0]["pitch_evidence"]["joint_decision"]
                if joint_hypotheses
                else "no_joint_hypothesis"
            )
        pitch_decision_counts[decision] = (
            pitch_decision_counts.get(decision, 0) + 1
        )
    candidate_payload = [
        {
            "sample_id": sample["sample_id"],
            "candidates": sample["joint_debug"]["separator_result"][
                "candidates"
            ],
        }
        for sample in samples
    ]
    candidate_checksum = hashlib.sha256(
        json.dumps(
            candidate_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    full_development = len(samples) == len(
        separator.load_development_document()["annotations"]
    )
    inside_geometry_payload = [
        {
            "sample_id": sample["sample_id"],
            "geometry": sample["final_hypothesis"]["geometry"],
        }
        for sample in samples
        if (
            sample["success"]
            and sample["final_hypothesis"]["reference_relation"]
            == "inside_basin"
        )
    ]
    inside_geometry_checksum = hashlib.sha256(
        json.dumps(
            inside_geometry_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "sample_count": len(samples),
        "correct_success": sum(
            sample["classification"] == "correct_success"
            for sample in samples
        ),
        "safe_failure": sum(
            sample["classification"] == "safe_failure"
            for sample in samples
        ),
        "wrong_success": sum(
            sample["classification"] == "wrong_success"
            for sample in samples
        ),
        "candidate_path_recall_count": sum(
            sample["candidate_path_recall_count"]
            for sample in samples
        ),
        "numeric_path_count": sum(
            sample["numeric_path_count"] for sample in samples
        ),
        "false_split": sum(
            sample["classification"] == "wrong_success"
            for sample in samples
        ),
        "separator_candidate_output_checksum": candidate_checksum,
        "separator_candidate_generation_unchanged": (
            candidate_checksum
            == separator.FROZEN_DEVELOPMENT_CANDIDATE_OUTPUT_CHECKSUM
            if full_development
            else None
        ),
        "stage3_inside_geometry_checksum": inside_geometry_checksum,
        "stage3_inside_geometry_unchanged": (
            inside_geometry_checksum
            == FROZEN_STAGE3_INSIDE_GEOMETRY_CHECKSUM
            if full_development
            else None
        ),
        "ambiguous_formally_available": [
            sample["sample_id"]
            for sample in samples
            if sample["gt_all_ambiguous"] and sample["success"]
        ],
        "unavailable_formally_available": [
            sample["sample_id"]
            for sample in samples
            if sample["gt_unavailable"] and sample["success"]
        ],
        "reference_ambiguity_formally_available": [
            sample["sample_id"]
            for sample in samples
            if (
                sample["gt_reference_ambiguity"]["is_ambiguity_case"]
                and sample["success"]
                and not sample["reference_separator_gt_assessment"][
                    "correct"
                ]
            )
        ],
        "reference_on_separator_correct_success": [
            sample["sample_id"]
            for sample in samples
            if (
                sample["success"]
                and sample["reference_relation"]["relation"]
                == "on_separator"
                and sample["formal_result_correct"]
            )
        ],
        "crossing_or_order_conflict_formally_available": [
            sample["sample_id"]
            for sample in samples
            if _formal_hypothesis_has_order_conflict(sample)
        ],
        "failure_reason_counts": dict(sorted(failure_counts.items())),
        "pitch_joint_decision_counts": dict(
            sorted(pitch_decision_counts.items())
        ),
        "correct_control_to_safe_failure": [
            sample["sample_id"]
            for sample in samples
            if (
                sample["separator_v1_1_status"] == "available"
                and sample["all_selected_roles_correct"]
                and sample["classification"] == "safe_failure"
            )
        ],
        "stage3_to_stage3_1_transition_counts": {
            transition: sum(
                sample["stage3_to_stage3_1"]["transition"]
                == transition
                for sample in samples
            )
            for transition in sorted(
                {
                    sample["stage3_to_stage3_1"]["transition"]
                    for sample in samples
                }
            )
        },
        "stage3_to_stage3_1_changed_samples": [
            {
                "sample_id": sample["sample_id"],
                **sample["stage3_to_stage3_1"],
                "classification": sample["classification"],
                "matched_semantics": sample[
                    "reference_separator_gt_assessment"
                ]["matched_semantics"],
            }
            for sample in samples
            if sample["stage3_to_stage3_1"]["transition"]
            != "unchanged"
        ],
    }


def _git_provenance() -> dict:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_dirty": bool(run("status", "--porcelain")),
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = (
        "sample_id",
        "classification",
        "status",
        "unavailable_reason",
        "separator_v1_1_status",
        "reference_relation",
        "pitch_confidence",
        "diagnostic_pitch_px",
        "usable_pitch_px",
        "pitch_joint_decision",
        "stage3_to_stage3_1_transition",
        "left_basin_center_x_roi",
        "right_basin_center_x_roi",
        "reference_to_left_basin_center_distance_px",
        "reference_to_right_basin_center_distance_px",
        "shared_boundary_to_left_basin_center_distance_px",
        "shared_boundary_to_right_basin_center_distance_px",
        "basin_center_spacing_px",
    )
    with temporary.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            hypothesis = sample["final_hypothesis"]
            if hypothesis is None:
                hypotheses = sample["joint_debug"].get(
                    "joint_hypotheses",
                    [],
                )
                hypothesis = hypotheses[0] if hypotheses else None
            geometry = (
                hypothesis["geometry"]
                if hypothesis is not None
                else None
            )
            on_separator_geometry = bool(
                geometry is not None
                and geometry["reference_relation"] == "on_separator"
            )
            writer.writerow(
                {
                    "sample_id": sample["sample_id"],
                    "classification": sample["classification"],
                    "status": sample["status"],
                    "unavailable_reason": sample["unavailable_reason"],
                    "separator_v1_1_status": sample[
                        "separator_v1_1_status"
                    ],
                    "reference_relation": sample[
                        "reference_relation"
                    ]["relation"],
                    "pitch_confidence": sample["raw_pitch"][
                        "confidence"
                    ],
                    "diagnostic_pitch_px": sample["raw_pitch"][
                        "diagnostic_pitch_px"
                    ],
                    "usable_pitch_px": sample["raw_pitch"][
                        "usable_pitch_px"
                    ],
                    "pitch_joint_decision": (
                        hypothesis["pitch_evidence"]["joint_decision"]
                        if hypothesis is not None
                        else "no_joint_hypothesis"
                    ),
                    "stage3_to_stage3_1_transition": sample[
                        "stage3_to_stage3_1"
                    ]["transition"],
                    "left_basin_center_x_roi": (
                        geometry["basins"]["left"][
                            "center_x_at_reference_roi"
                        ]
                        if on_separator_geometry
                        else None
                    ),
                    "right_basin_center_x_roi": (
                        geometry["basins"]["right"][
                            "center_x_at_reference_roi"
                        ]
                        if on_separator_geometry
                        else None
                    ),
                    "reference_to_left_basin_center_distance_px": (
                        geometry[
                            "reference_to_left_basin_center_distance_px"
                        ]
                        if on_separator_geometry
                        else None
                    ),
                    "reference_to_right_basin_center_distance_px": (
                        geometry[
                            "reference_to_right_basin_center_distance_px"
                        ]
                        if on_separator_geometry
                        else None
                    ),
                    "shared_boundary_to_left_basin_center_distance_px": (
                        geometry[
                            "shared_boundary_to_left_basin_center_distance_px"
                        ]
                        if on_separator_geometry
                        else None
                    ),
                    "shared_boundary_to_right_basin_center_distance_px": (
                        geometry[
                            "shared_boundary_to_right_basin_center_distance_px"
                        ]
                        if on_separator_geometry
                        else None
                    ),
                    "basin_center_spacing_px": (
                        geometry["basin_center_spacing_px"]
                        if on_separator_geometry
                        else None
                    ),
                }
            )
    temporary.replace(path)


def evaluate_development(
    sample_ids: list[str] | None = None,
    output_dir: Path = OUTPUT_DIR,
    config: BasinGraphJointConfig = DEFAULT_CONFIG,
) -> dict:
    verify_frozen_dependencies()
    document = separator.load_development_document()
    annotations = document["annotations"]
    requested = (
        list(annotations)
        if sample_ids is None
        else list(dict.fromkeys(sample_ids))
    )
    unknown = [sample_id for sample_id in requested if sample_id not in annotations]
    if unknown:
        raise ValueError(f"unknown development sample IDs: {unknown}")
    image_cache = {}
    samples = []
    for sample_id in requested:
        annotation = annotations[sample_id]
        image_name = annotation["image_name"]
        if image_name not in image_cache:
            image_cache[image_name] = separator.load_grayscale_image(
                separator.IMAGES_DIR / image_name
            )
        samples.append(
            _evaluate_one(
                annotation,
                image_cache[image_name],
                config,
            )
        )
    report = {
        "algorithm_revision": ALGORITHM_REVISION,
        "configuration": canonical_configuration(config),
        "configuration_checksum": configuration_checksum(config),
        "access_policy": "development_only",
        "source_document": str(
            separator.DEVELOPMENT_PATH.relative_to(PROJECT_ROOT)
        ),
        "evaluated_sample_ids": requested,
        **_git_provenance(),
        "metrics": _summarize(samples),
        "samples": samples,
    }
    report_name = (
        "development_report.json"
        if sample_ids is None
        else "quick_report.json"
    )
    _atomic_write_json(output_dir / report_name, report)
    _write_csv(output_dir / "sample_summary.csv", samples)
    return report


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-3 basin graph and raw-pitch joint prototype",
    )
    parser.add_argument("--sample", action="append", dest="sample_ids")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    report = evaluate_development(
        arguments.sample_ids,
        arguments.output_dir,
    )
    print(json.dumps(report["metrics"], indent=2))
    print(f"report: {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
