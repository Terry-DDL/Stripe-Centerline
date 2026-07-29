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


ALGORITHM_REVISION = "basin_graph_raw_pitch_joint_stage3_v1"
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "basin_graph_joint_stage3_v1"


@dataclass(frozen=True)
class BasinGraphJointConfig:
    """Stage-3-only safety parameters.

    These thresholds do not participate in separator candidate generation or
    raw-pitch estimation.
    """

    high_pitch_minimum_spacing_ratio: float = 0.67
    high_pitch_maximum_spacing_ratio: float = 1.50


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
) -> dict:
    arbitration = separator_result["arbitration_debug"]
    separator_relations = [
        item
        for item in arbitration.get("competing_explanations", [])
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
    ordered_ids = graph["ordered_separator_ids"]
    index = ordered_ids.index(reference_separator_id)
    if index == 0 or index + 1 >= len(ordered_ids):
        return {
            "status": "unavailable",
            "relation": "on_separator",
            "hypotheses": [],
            "unavailable_reason": "separator_adjacency_boundary_missing",
        }
    left_id = ordered_ids[index - 1]
    right_id = ordered_ids[index + 1]
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
        }
    hypothesis = {
        "hypothesis_id": "JH01",
        "reference_relation": "on_separator",
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
        "strictly_continuous": True,
    }
    return {
        "status": "unique",
        "relation": "on_separator",
        "hypotheses": [hypothesis],
        "unavailable_reason": None,
    }


def resolve_reference_relation(
    separator_result: dict,
    graph: dict,
) -> dict:
    """Resolve exactly one basin or separator-adjacency interpretation."""

    if separator_result["status"] == "available":
        return _inside_basin_relation(separator_result, graph)
    if separator_result["unavailable_reason"] == "reference_on_separator":
        return _separator_adjacency_relation(separator_result, graph)
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
        return {
            "reference_relation": "on_separator",
            "reference_separator_x_roi": separator_x,
            "reference_separator_distance_px": abs(
                separator_x - reference_x_roi
            ),
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
    if geometry["reference_relation"] != "inside_basin":
        evidence["joint_decision"] = "high_diagnostic_for_separator_relation"
        return evidence

    pitch_px = float(pitch_result["usable_pitch_px"])
    ratios = [
        float(spacing / pitch_px)
        for spacing in geometry["separator_spacing_px"]
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

    if relation_hypothesis["reference_relation"] == "on_separator":
        return {
            "algorithm_revision": ALGORITHM_REVISION,
            "configuration_checksum": configuration_checksum(config),
            "dependency_checks": dependency_checks,
            "status": "unavailable",
            "success": False,
            "unavailable_reason": (
                "reference_on_separator_requires_side_choice"
            ),
            "final_hypothesis": None,
            "debug": debug,
        }
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
    unsafe_label = bool(
        separator_evaluation["gt_all_ambiguous"]
        or separator_evaluation["gt_unavailable"]
        or separator_evaluation["gt_reference_ambiguity"][
            "is_ambiguity_case"
        ]
    )
    scorable = bool(numeric_roles and not unsafe_label)
    if result["success"] and all_roles_correct and not unsafe_label:
        classification = "correct_success"
    elif result["success"]:
        classification = "wrong_success"
    else:
        classification = "safe_failure"
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
            if scorable
            else 0
        ),
        "numeric_path_count": len(numeric_roles) if scorable else 0,
        "all_selected_roles_correct": all_roles_correct,
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
            if sample["gt_reference_ambiguity"]["is_ambiguity_case"]
            and sample["success"]
        ],
        "crossing_or_order_conflict_formally_available": [
            sample["sample_id"]
            for sample in samples
            if sample["success"]
            and (
                sample["basin_graph"]["path_order_conflicts"]
                or sample["separator_v1_1_status"] == "unavailable"
            )
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
