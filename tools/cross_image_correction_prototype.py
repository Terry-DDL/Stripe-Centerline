"""Offline cross-image separator/centerline correction experiment.

This tool is intentionally isolated from the desktop product.  The baseline
mode only replays the frozen Stage 3.1 detector and records raw-gray evidence.
Later experiment modes may transform offline hypotheses, but never alter the
formal desktop result.

Only separator development annotations are readable by this module.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable
from unittest.mock import patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from config import INTERACTIVE_CONFIG  # noqa: E402
from interactive_pipeline import (  # noqa: E402
    calculate_interactive_roi_bounds,
)
from tools import basin_graph_joint_prototype as joint  # noqa: E402
from tools import raw_local_pitch_prototype_v3 as raw_pitch  # noqa: E402
from tools import separator_path_prototype as separator  # noqa: E402


RELEASE_COMMIT = "55b26e6deac5acbb50e3934c03d71e3c5b56b269"
ALGORITHM_REVISION = "cross_image_correction_offline_v1"
EVALUATION_MANIFEST_PATH = (
    PROJECT_ROOT
    / "tests"
    / "data"
    / "cross_image_correction_eval_v1.json"
)
DEVELOPMENT_PATH = (
    PROJECT_ROOT
    / "tests"
    / "data"
    / "separator_path_gt_v1"
    / "development.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "cross_image_correction_v1"
)
FROZEN_DETECT_SEPARATOR_PATHS = separator.detect_separator_paths
FROZEN_RUN_JOINT_CASE = joint.run_joint_case


@dataclass(frozen=True)
class CrossImageCorrectionConfig:
    """Dimensionless limits for the isolated correction experiment."""

    broad_plateau_minimum_pitch_ratio: float = 0.20
    broad_plateau_maximum_pitch_ratio: float = 0.55
    plateau_pair_minimum_band_fraction: float = 2.0 / 3.0
    plateau_valley_minimum_relative_level: float = 0.55
    plateau_center_threshold_fraction: float = 0.50
    local_profile_radius_pitch_ratio: float = 0.48
    refinement_minimum_band_fraction: float = 0.75
    refinement_maximum_shift_mad_width_ratio: float = 0.08
    refinement_maximum_shift_width_ratio: float = 0.35
    refinement_maximum_weighted_to_minimum_width_ratio: float = 0.22


DEFAULT_CORRECTION_CONFIG = CrossImageCorrectionConfig()


def correction_configuration_checksum(
    config: CrossImageCorrectionConfig,
) -> str:
    payload = {
        "algorithm_revision": ALGORITHM_REVISION,
        "release_commit": RELEASE_COMMIT,
        "parameters": asdict(config),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_provenance() -> dict:
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_evaluation_manifest() -> dict:
    document = json.loads(EVALUATION_MANIFEST_PATH.read_text())
    if document["release_commit"] != RELEASE_COMMIT:
        raise ValueError("evaluation manifest release commit changed")
    if document["access_policy"]["separator_heldout_allowed"]:
        raise ValueError("held-out access must remain disabled")
    return document


def load_development_annotations() -> dict[str, dict]:
    """Load development only; this function has no held-out path."""

    document = json.loads(DEVELOPMENT_PATH.read_text())
    if document["split"] != "development":
        raise ValueError("development split marker is missing")
    annotations = document["annotations"]
    if len(annotations) != 26:
        raise ValueError(
            f"expected 26 development annotations, got {len(annotations)}"
        )
    if any(item["split"] != "development" for item in annotations.values()):
        raise ValueError("non-development annotation found")
    return annotations


def bounds_dict(bounds) -> dict:
    return {
        "x0": int(bounds.x0_global),
        "y0": int(bounds.y0_global),
        "x1": int(bounds.x1_global),
        "y1": int(bounds.y1_global),
    }


def _path_x_at_y(path: dict, y_roi: float) -> float:
    return float(
        np.interp(
            y_roi,
            path["band_centers_y_roi"],
            path["x_by_band_roi"],
        )
    )


def _raw_band_profiles(
    directional_roi: np.ndarray,
    evidence: dict,
) -> list[list[float]]:
    profiles = []
    for y0, y1 in evidence["band_bounds"]:
        profiles.append(
            np.percentile(
                directional_roi[y0:y1].astype(np.float64),
                separator.DEFAULT_CONFIG.band_profile_percentile,
                axis=0,
            ).tolist()
        )
    return profiles


def _ridge_interval(
    profile: np.ndarray,
    candidate_x: float,
    local_pitch_px: float,
) -> dict:
    """Describe the local bright ridge without using a fixed pixel scale."""

    width = profile.size
    center = int(np.clip(round(candidate_x), 0, width - 1))
    radius = max(1, int(round(0.48 * local_pitch_px)))
    local_x0 = max(0, center - radius)
    local_x1 = min(width, center + radius + 1)
    local = profile[local_x0:local_x1]
    dark = float(np.percentile(local, 10))
    bright = float(np.percentile(local, 90))
    dynamic = bright - dark
    if dynamic <= 1e-9:
        return {
            "status": "unavailable",
            "left_x_roi": float(center),
            "right_x_roi": float(center),
            "width_px": 1.0,
            "threshold": None,
            "dark_level": dark,
            "bright_level": bright,
            "internal_brightness": float(profile[center]),
            "normalized_candidate_position": 0.5,
        }

    threshold = dark + 0.5 * dynamic
    left = center
    right = center
    while left > local_x0 and profile[left - 1] >= threshold:
        left -= 1
    while right + 1 < local_x1 and profile[right + 1] >= threshold:
        right += 1
    ridge_width = max(1, right - left + 1)
    normalized = (candidate_x - left) / ridge_width
    return {
        "status": "available",
        "left_x_roi": float(left),
        "right_x_roi": float(right),
        "width_px": float(ridge_width),
        "threshold": threshold,
        "dark_level": dark,
        "bright_level": bright,
        "internal_brightness": float(
            np.median(profile[left : right + 1])
        ),
        "normalized_candidate_position": float(normalized),
    }


def _candidate_plateau_diagnostic(
    candidate: dict,
    raw_profiles: list[list[float]],
    local_pitch_px: float,
) -> dict:
    bands = []
    labels = []
    for band_index, (profile_values, candidate_x) in enumerate(
        zip(raw_profiles, candidate["x_by_band_roi"])
    ):
        ridge = _ridge_interval(
            np.asarray(profile_values, dtype=np.float64),
            float(candidate_x),
            local_pitch_px,
        )
        position = ridge["normalized_candidate_position"]
        if position < 0.34:
            label = "left_edge"
        elif position > 0.66:
            label = "right_edge"
        else:
            label = "ridge_center"
        labels.append(label)
        bands.append(
            {
                "band_index": band_index,
                "candidate_x_roi": float(candidate_x),
                "classification": label,
                **ridge,
            }
        )
    counts = {
        label: labels.count(label)
        for label in ("left_edge", "ridge_center", "right_edge")
    }
    representative = max(
        counts,
        key=lambda label: (counts[label], -list(counts).index(label)),
    )
    return {
        "candidate_id": candidate.get("candidate_id"),
        "seed_x_roi": candidate["seed_x_roi"],
        "representative_classification": representative,
        "classification_counts": counts,
        "bands": bands,
    }


def _basin_gray_diagnostic(
    basin: dict,
    raw_profiles: list[list[float]],
) -> dict:
    band_results = []
    for band_index, (left_x, right_x, midpoint, profile_values) in enumerate(
        zip(
            basin["left_x_by_band_roi"],
            basin["right_x_by_band_roi"],
            basin["center_x_by_band_roi"],
            raw_profiles,
        )
    ):
        profile = np.asarray(profile_values, dtype=np.float64)
        x0 = max(0, int(np.ceil(min(left_x, right_x))))
        x1 = min(profile.size - 1, int(np.floor(max(left_x, right_x))))
        if x1 <= x0:
            band_results.append(
                {
                    "band_index": band_index,
                    "status": "unavailable",
                    "reason": "empty_basin_interval",
                }
            )
            continue
        positions = np.arange(x0, x1 + 1, dtype=np.float64)
        values = profile[x0 : x1 + 1]
        minimum = float(np.min(values))
        minimum_positions = positions[np.isclose(values, minimum)]
        minimum_center = float(np.median(minimum_positions))
        bright_reference = float(np.percentile(values, 90))
        weights = np.maximum(0.0, bright_reference - values)
        weighted_center = (
            float(np.sum(positions * weights) / np.sum(weights))
            if float(np.sum(weights)) > 1e-9
            else float(midpoint)
        )
        band_results.append(
            {
                "band_index": band_index,
                "status": "available",
                "left_x_roi": float(left_x),
                "right_x_roi": float(right_x),
                "arithmetic_center_x_roi": float(midpoint),
                "gray_minimum": minimum,
                "gray_minimum_center_x_roi": minimum_center,
                "dark_weighted_center_x_roi": weighted_center,
                "arithmetic_to_minimum_error_px": float(
                    abs(midpoint - minimum_center)
                ),
                "weighted_to_minimum_error_px": float(
                    abs(weighted_center - minimum_center)
                ),
            }
        )
    available = [
        item for item in band_results if item["status"] == "available"
    ]
    return {
        "basin_id": basin["basin_id"],
        "left_separator_id": basin["left_separator_id"],
        "right_separator_id": basin["right_separator_id"],
        "verified": basin["verified"],
        "bands": band_results,
        "arithmetic_to_minimum_median_px": (
            float(
                np.median(
                    [
                        item["arithmetic_to_minimum_error_px"]
                        for item in available
                    ]
                )
            )
            if available
            else None
        ),
        "arithmetic_to_minimum_max_px": (
            float(
                np.max(
                    [
                        item["arithmetic_to_minimum_error_px"]
                        for item in available
                    ]
                )
            )
            if available
            else None
        ),
    }


def _raw_candidates_before_dedup(
    evidence: dict,
) -> list[dict]:
    traced = [
        separator._trace_one_seed(  # noqa: SLF001
            seed,
            evidence,
            separator.DEFAULT_CONFIG,
        )
        for seed in separator._seed_peaks(  # noqa: SLF001
            evidence["aggregate_response"],
            separator.DEFAULT_CONFIG,
        )
    ]
    traced.sort(
        key=lambda path: (
            not path["accepted"],
            -path["path_score"],
            path["seed_x_roi"],
        )
    )
    for index, path in enumerate(traced, start=1):
        path["raw_candidate_id"] = f"R{index:02d}"
    return traced


def _local_pitch_scale(
    pitch_result: dict,
    frozen_candidates: list[dict],
    reference_y_roi: float,
    roi_width_px: int,
) -> float:
    diagnostic = pitch_result.get("diagnostic_pitch_px")
    if diagnostic is not None and float(diagnostic) > 0:
        return float(diagnostic)
    accepted_x = sorted(
        _path_x_at_y(candidate, reference_y_roi)
        for candidate in frozen_candidates
        if candidate["accepted"]
    )
    spacings = np.diff(np.asarray(accepted_x, dtype=np.float64))
    if spacings.size:
        return float(np.median(spacings))
    if not frozen_candidates:
        return float(roi_width_px)
    return float(
        max(
            candidate["peak_width_px"]
            for candidate in frozen_candidates
        )
    )


def _plateau_pair_band_evidence(
    profile: np.ndarray,
    first_x: float,
    second_x: float,
    local_pitch_px: float,
    config: CrossImageCorrectionConfig,
) -> dict:
    x0, x1 = sorted((float(first_x), float(second_x)))
    span_ratio = (x1 - x0) / local_pitch_px
    center = (x0 + x1) / 2.0
    radius = max(
        1,
        int(
            round(
                config.local_profile_radius_pitch_ratio
                * local_pitch_px
            )
        ),
    )
    local_x0 = max(0, int(np.floor(center)) - radius)
    local_x1 = min(
        profile.size,
        int(np.ceil(center)) + radius + 1,
    )
    segment_x0 = max(0, int(np.floor(x0)))
    segment_x1 = min(profile.size, int(np.ceil(x1)) + 1)
    local = profile[local_x0:local_x1]
    segment = profile[segment_x0:segment_x1]
    dark = float(np.percentile(local, 10))
    endpoint_brightness = min(
        float(np.interp(x0, np.arange(profile.size), profile)),
        float(np.interp(x1, np.arange(profile.size), profile)),
    )
    dynamic = endpoint_brightness - dark
    valley = (
        float(np.percentile(segment, 10))
        if segment.size
        else dark
    )
    relative_valley = (
        (valley - dark) / dynamic if dynamic > 1e-9 else None
    )
    span_compatible = bool(
        config.broad_plateau_minimum_pitch_ratio
        <= span_ratio
        <= config.broad_plateau_maximum_pitch_ratio
    )
    no_dark_valley = bool(
        relative_valley is not None
        and relative_valley
        >= config.plateau_valley_minimum_relative_level
    )
    return {
        "span_px": x1 - x0,
        "span_pitch_ratio": span_ratio,
        "dark_level": dark,
        "endpoint_brightness": endpoint_brightness,
        "valley_level": valley,
        "relative_valley_level": relative_valley,
        "span_compatible": span_compatible,
        "no_dark_valley": no_dark_valley,
        "same_plateau": span_compatible and no_dark_valley,
    }


def _plateau_group_evidence(
    paths: list[dict],
    evidence: dict,
    local_pitch_px: float,
    config: CrossImageCorrectionConfig,
) -> dict:
    if len(paths) < 2:
        return {
            "eligible": False,
            "reason": "single_frozen_candidate",
            "band_agreement_fraction": 0.0,
            "median_span_pitch_ratio": 0.0,
            "bands": [],
        }
    path_matrix = np.asarray(
        [path["x_by_band_roi"] for path in paths],
        dtype=np.float64,
    )
    bands = []
    for band_index, profile in enumerate(evidence["profiles"]):
        bands.append(
            _plateau_pair_band_evidence(
                np.asarray(profile, dtype=np.float64),
                float(np.min(path_matrix[:, band_index])),
                float(np.max(path_matrix[:, band_index])),
                local_pitch_px,
                config,
            )
        )
    agreement = float(
        np.mean([band["same_plateau"] for band in bands])
    )
    median_span_ratio = float(
        np.median(
            [
                band["span_pitch_ratio"]
                for band in bands
            ]
        )
    )
    eligible = bool(
        agreement >= config.plateau_pair_minimum_band_fraction
        and config.broad_plateau_minimum_pitch_ratio
        <= median_span_ratio
        <= config.broad_plateau_maximum_pitch_ratio
    )
    if eligible:
        reason = "paired_edges_enclose_one_continuous_bright_plateau"
    elif (
        median_span_ratio
        < config.broad_plateau_minimum_pitch_ratio
    ):
        reason = "narrow_ridge_kept_on_frozen_path"
    elif (
        median_span_ratio
        > config.broad_plateau_maximum_pitch_ratio
    ):
        reason = "edge_span_not_compatible_with_local_pitch"
    else:
        reason = "dark_valley_or_insufficient_band_pairing"
    return {
        "eligible": eligible,
        "reason": reason,
        "band_agreement_fraction": agreement,
        "median_span_pitch_ratio": median_span_ratio,
        "bands": bands,
    }


def _plateau_center_for_band(
    profile: np.ndarray,
    group_x: np.ndarray,
    local_pitch_px: float,
    config: CrossImageCorrectionConfig,
) -> dict:
    x0 = float(np.min(group_x))
    x1 = float(np.max(group_x))
    center = (x0 + x1) / 2.0
    radius = max(
        1,
        int(
            round(
                config.local_profile_radius_pitch_ratio
                * local_pitch_px
            )
        ),
    )
    local_x0 = max(0, int(np.floor(center)) - radius)
    local_x1 = min(
        profile.size,
        int(np.ceil(center)) + radius + 1,
    )
    local = profile[local_x0:local_x1]
    dark = float(np.percentile(local, 10))
    segment_x0 = max(0, int(np.floor(x0)))
    segment_x1 = min(profile.size, int(np.ceil(x1)) + 1)
    segment = profile[segment_x0:segment_x1]
    bright = float(np.percentile(segment, 90))
    threshold = (
        dark
        + config.plateau_center_threshold_fraction
        * (bright - dark)
    )
    start = int(np.clip(round(center), 0, profile.size - 1))
    left = start
    right = start
    while left > local_x0 and profile[left - 1] >= threshold:
        left -= 1
    while right + 1 < local_x1 and profile[right + 1] >= threshold:
        right += 1
    positions = np.arange(left, right + 1, dtype=np.float64)
    values = profile[left : right + 1]
    weights = np.maximum(0.0, values - dark)
    robust_center = (
        float(np.sum(positions * weights) / np.sum(weights))
        if float(np.sum(weights)) > 1e-9
        else center
    )
    return {
        "center_x_roi": robust_center,
        "left_edge_x_roi": float(left),
        "right_edge_x_roi": float(right),
        "plateau_width_px": float(right - left + 1),
        "threshold": threshold,
        "dark_level": dark,
        "bright_level": bright,
    }


def _centerized_candidate(
    frozen_candidate: dict,
    group_paths: list[dict],
    evidence: dict,
    local_pitch_px: float,
    group_evidence: dict,
    config: CrossImageCorrectionConfig,
) -> dict:
    path_matrix = np.asarray(
        [path["x_by_band_roi"] for path in group_paths],
        dtype=np.float64,
    )
    band_results = [
        _plateau_center_for_band(
            np.asarray(profile, dtype=np.float64),
            path_matrix[:, band_index],
            local_pitch_px,
            config,
        )
        for band_index, profile in enumerate(evidence["profiles"])
    ]
    center_path = np.asarray(
        [band["center_x_roi"] for band in band_results],
        dtype=np.float64,
    )
    response_by_band = np.asarray(
        [
            np.interp(
                center_path[band_index],
                np.arange(evidence["responses"].shape[1]),
                evidence["responses"][band_index],
            )
            for band_index in range(center_path.size)
        ],
        dtype=np.float64,
    )
    supported = (
        response_by_band
        >= separator.DEFAULT_CONFIG.supported_band_response
    )
    support_fraction = float(np.mean(supported))
    top_count = min(
        separator.DEFAULT_CONFIG.aggregate_top_band_count,
        center_path.size,
    )
    top_response = float(
        np.mean(np.sort(response_by_band)[-top_count:])
    )
    mean_step = (
        float(np.mean(np.abs(np.diff(center_path))))
        if center_path.size > 1
        else 0.0
    )
    path_score = float(
        top_response
        + 0.45 * support_fraction
        - 0.04 * mean_step
    )
    accepted = bool(
        support_fraction
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
            "response_by_band": response_by_band.tolist(),
            "supported_band_indices": np.flatnonzero(
                supported
            ).tolist(),
            "support_fraction": support_fraction,
            "top_response": top_response,
            "mean_step_px": mean_step,
            "path_score": path_score,
            "peak_width_px": int(
                round(
                    float(
                        np.median(
                            [
                                band["plateau_width_px"]
                                for band in band_results
                            ]
                        )
                    )
                )
            ),
            "accepted": accepted,
            "rejection_reason": (
                None
                if accepted
                else separator._path_rejection_reason(  # noqa: SLF001
                    support_fraction,
                    top_response,
                    path_score,
                    separator.DEFAULT_CONFIG,
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
                "band_centers": band_results,
            },
        }
    )
    candidate.pop("x_at_reference_roi", None)
    return candidate


def centerize_frozen_candidates(
    frozen_candidates: list[dict],
    raw_candidates: list[dict],
    evidence: dict,
    local_pitch_px: float,
    config: CrossImageCorrectionConfig = DEFAULT_CORRECTION_CONFIG,
) -> tuple[list[dict], list[dict]]:
    """Replace only broad paired-edge groups with one ridge center path."""

    raw_by_seed = {
        path["seed_x_roi"]: path for path in raw_candidates
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
        group_evidence = _plateau_group_evidence(
            group_paths,
            evidence,
            local_pitch_px,
            config,
        )
        if group_evidence["eligible"]:
            candidate = _centerized_candidate(
                frozen_candidate,
                group_paths,
                evidence,
                local_pitch_px,
                group_evidence,
                config,
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
            }
        )
    return transformed, mappings


def detect_centerized_separator_paths(
    image_gray: np.ndarray,
    reference_global: dict,
    roi_bounds_global: dict,
    direction: str = "vertical",
    config: CrossImageCorrectionConfig = DEFAULT_CORRECTION_CONFIG,
) -> dict:
    """Run frozen candidate generation, then center only proven plateaus."""

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
    raw_candidates = _raw_candidates_before_dedup(evidence)
    pitch_result = raw_pitch.estimate_raw_local_pitch_v3(
        image_gray,
        reference_global,
        roi_bounds_global,
        direction,
        raw_pitch.DEFAULT_CONFIG,
    )
    local_pitch = _local_pitch_scale(
        pitch_result,
        frozen["candidates"],
        frozen["reference_y_roi"],
        directional.shape[1],
    )
    candidates, mappings = centerize_frozen_candidates(
        frozen["candidates"],
        raw_candidates,
        evidence,
        local_pitch,
        config,
    )
    v1_arbitration = separator._select_clicked_basin_paths_v1(  # noqa: SLF001
        candidates,
        frozen["reference_x_roi"],
        frozen["reference_y_roi"],
        separator.DEFAULT_CONFIG,
    )
    arbitration = separator.select_clicked_basin_paths(
        candidates,
        frozen["reference_x_roi"],
        frozen["reference_y_roi"],
        evidence,
        separator.DEFAULT_CONFIG,
    )
    return {
        **frozen,
        "status": arbitration["status"],
        "unavailable_reason": arbitration["unavailable_reason"],
        "crossing": arbitration["crossing"],
        "selection": arbitration["selection"],
        "arbitration_debug": {
            key: value
            for key, value in arbitration.items()
            if key not in {"status", "unavailable_reason", "selection"}
        },
        "v1_arbitration_baseline": {
            "status": v1_arbitration["status"],
            "unavailable_reason": v1_arbitration[
                "unavailable_reason"
            ],
            "selection_candidate_ids": {
                role: path["candidate_id"]
                for role, path in v1_arbitration["selection"].items()
            },
            "crossing": v1_arbitration["crossing"],
        },
        "candidates": candidates,
        "raw_gray_evidence": {
            **frozen["raw_gray_evidence"],
            "accepted_candidate_count": sum(
                path["accepted"] for path in candidates
            ),
        },
        "plateau_centerization": {
            "algorithm_revision": ALGORITHM_REVISION,
            "configuration_checksum": (
                correction_configuration_checksum(config)
            ),
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


def run_centerized_joint_case(
    image_gray: np.ndarray,
    reference_global: dict,
    roi_bounds_global: dict,
    direction: str = "vertical",
    correction_config: CrossImageCorrectionConfig = (
        DEFAULT_CORRECTION_CONFIG
    ),
) -> dict:
    """Run the frozen joint safety chain with centerized separators."""

    frozen_result = FROZEN_RUN_JOINT_CASE(
        image_gray,
        reference_global,
        roi_bounds_global,
        direction,
        joint.DEFAULT_CONFIG,
    )
    separator_result = detect_centerized_separator_paths(
        image_gray,
        reference_global,
        roi_bounds_global,
        direction,
        correction_config,
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
        "algorithm_revision": (
            "basin_graph_raw_pitch_joint_stage3_v1_1"
            "_plateau_centerization_offline_v1"
        ),
        "configuration_checksum": (
            correction_configuration_checksum(correction_config)
        ),
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
            "plateau_centerization_configuration_checksum": (
                correction_configuration_checksum(
                    correction_config
                )
            ),
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
            **base,
            "status": "unavailable",
            "success": False,
            "unavailable_reason": "high_pitch_geometry_conflict",
            "final_hypothesis": None,
        }
    if not frozen_result["success"]:
        debug["centerization_safe_upgrade_blocked"] = {
            "blocked": True,
            "reason": "frozen_unavailable_cannot_become_success",
            "frozen_unavailable_reason": frozen_result[
                "unavailable_reason"
            ],
            "proposed_hypothesis": atomic_hypothesis,
        }
        return {
            **base,
            "status": "unavailable",
            "success": False,
            "unavailable_reason": frozen_result[
                "unavailable_reason"
            ],
            "final_hypothesis": None,
        }
    return {
        **base,
        "status": "available",
        "success": True,
        "unavailable_reason": None,
        "final_hypothesis": atomic_hypothesis,
    }


def _refine_one_basin(
    basin: dict,
    raw_profiles: list[list[float]],
    config: CrossImageCorrectionConfig,
) -> dict:
    band_results = []
    for band_index, (left_x, right_x, midpoint, profile_values) in enumerate(
        zip(
            basin["left_x_by_band_roi"],
            basin["right_x_by_band_roi"],
            basin["center_x_by_band_roi"],
            raw_profiles,
        )
    ):
        profile = np.asarray(profile_values, dtype=np.float64)
        x0 = max(0, int(np.ceil(left_x)))
        x1 = min(profile.size - 1, int(np.floor(right_x)))
        basin_width = float(right_x - left_x)
        if x1 <= x0 or basin_width <= 0:
            band_results.append(
                {
                    "band_index": band_index,
                    "status": "unavailable",
                    "reason": "empty_basin_interval",
                }
            )
            continue
        positions = np.arange(x0, x1 + 1, dtype=np.float64)
        values = profile[x0 : x1 + 1]
        minimum = float(np.min(values))
        minimum_center = float(
            np.median(positions[np.isclose(values, minimum)])
        )
        bright = float(np.percentile(values, 90))
        dark = float(np.percentile(values, 10))
        weights = np.maximum(0.0, bright - values)
        weight_sum = float(np.sum(weights))
        weighted_center = (
            float(np.sum(positions * weights) / weight_sum)
            if weight_sum > 1e-9
            else float(midpoint)
        )
        weighted_to_minimum = abs(weighted_center - minimum_center)
        consistent = bool(
            dark < bright
            and left_x < weighted_center < right_x
            and weighted_to_minimum
            <= (
                config
                .refinement_maximum_weighted_to_minimum_width_ratio
                * basin_width
            )
        )
        band_results.append(
            {
                "band_index": band_index,
                "status": (
                    "available" if consistent else "unavailable"
                ),
                "reason": (
                    None
                    if consistent
                    else "band_valley_evidence_inconsistent"
                ),
                "left_x_roi": float(left_x),
                "right_x_roi": float(right_x),
                "basin_width_px": basin_width,
                "arithmetic_center_x_roi": float(midpoint),
                "gray_minimum": minimum,
                "gray_minimum_center_x_roi": minimum_center,
                "dark_weighted_center_x_roi": weighted_center,
                "weighted_to_minimum_error_px": weighted_to_minimum,
                "shift_from_arithmetic_px": float(
                    weighted_center - midpoint
                ),
                "profile_dark_level": dark,
                "profile_bright_level": bright,
            }
        )
    available = [
        item for item in band_results if item["status"] == "available"
    ]
    required_fraction = config.refinement_minimum_band_fraction
    support_fraction = len(available) / max(1, len(band_results))
    if available:
        shifts = np.asarray(
            [
                item["shift_from_arithmetic_px"]
                for item in available
            ],
            dtype=np.float64,
        )
        widths = np.asarray(
            [item["basin_width_px"] for item in available],
            dtype=np.float64,
        )
        median_shift = float(np.median(shifts))
        shift_mad = float(np.median(np.abs(shifts - median_shift)))
        median_width = float(np.median(widths))
        baseline_errors = [
            abs(
                item["arithmetic_center_x_roi"]
                - item["gray_minimum_center_x_roi"]
            )
            for item in available
        ]
        refined_errors = [
            abs(
                item["arithmetic_center_x_roi"]
                + median_shift
                - item["gray_minimum_center_x_roi"]
            )
            for item in available
        ]
    else:
        median_shift = 0.0
        shift_mad = float("inf")
        median_width = 0.0
        baseline_errors = []
        refined_errors = []
    consensus_stable = bool(
        available
        and support_fraction >= required_fraction
        and shift_mad
        <= (
            config.refinement_maximum_shift_mad_width_ratio
            * median_width
        )
        and abs(median_shift)
        <= (
            config.refinement_maximum_shift_width_ratio
            * median_width
        )
    )
    improves_valley_error = bool(
        baseline_errors
        and float(np.median(refined_errors))
        < float(np.median(baseline_errors))
    )
    adopted = consensus_stable and improves_valley_error
    reason = (
        "cross_band_dark_center_consensus"
        if adopted
        else (
            "insufficient_or_inconsistent_band_evidence"
            if not consensus_stable
            else "no_robust_valley_error_improvement"
        )
    )
    refined_path = [
        float(center + median_shift)
        if adopted
        else float(center)
        for center in basin["center_x_by_band_roi"]
    ]
    return {
        "basin_id": basin["basin_id"],
        "left_separator_id": basin["left_separator_id"],
        "right_separator_id": basin["right_separator_id"],
        "adopted": adopted,
        "reason": reason,
        "support_fraction": support_fraction,
        "median_shift_px": median_shift if adopted else 0.0,
        "proposed_median_shift_px": median_shift,
        "shift_mad_px": shift_mad if np.isfinite(shift_mad) else None,
        "median_basin_width_px": median_width,
        "baseline_minimum_error_median_px": (
            float(np.median(baseline_errors))
            if baseline_errors
            else None
        ),
        "baseline_minimum_error_max_px": (
            float(np.max(baseline_errors))
            if baseline_errors
            else None
        ),
        "refined_minimum_error_median_px": (
            float(np.median(refined_errors))
            if adopted and refined_errors
            else (
                float(np.median(baseline_errors))
                if baseline_errors
                else None
            )
        ),
        "refined_minimum_error_max_px": (
            float(np.max(refined_errors))
            if adopted and refined_errors
            else (
                float(np.max(baseline_errors))
                if baseline_errors
                else None
            )
        ),
        "bandwise_center_path_x_roi": refined_path,
        "bands": band_results,
    }


def refine_accepted_basin_centers(
    result: dict,
    image_gray: np.ndarray,
    roi_bounds_global: dict,
    direction: str = "vertical",
    config: CrossImageCorrectionConfig = DEFAULT_CORRECTION_CONFIG,
) -> dict:
    """Refine geometry only after the frozen safety decision is final."""

    identity_before = _identity(result)
    if not result["success"] or result["final_hypothesis"] is None:
        return {
            "attempted": False,
            "adopted": False,
            "reason": "formal_result_unavailable",
            "identity_before": identity_before,
            "identity_after": identity_before,
            "combined_geometry": None,
            "basins": {},
        }
    raw_roi = separator.extract_raw_roi(
        image_gray,
        roi_bounds_global,
    )
    directional = separator._directional_roi(  # noqa: SLF001
        raw_roi,
        direction,
    )
    evidence = separator.build_raw_gray_response(
        directional,
        separator.DEFAULT_CONFIG,
    )
    raw_profiles = _raw_band_profiles(directional, evidence)
    graph = result["debug"]["basin_graph"]
    basin_by_id = {
        basin["basin_id"]: basin
        for basin in graph["verified_basins"]
    }
    hypothesis = result["final_hypothesis"]
    refinements = {
        role: _refine_one_basin(
            basin_by_id[hypothesis["basin_ids"][role]],
            raw_profiles,
            config,
        )
        for role in ("left", "right")
    }
    combined_geometry = copy.deepcopy(hypothesis["geometry"])
    roi_x0 = roi_bounds_global["x0"]
    for role in ("left", "right"):
        original = combined_geometry["basins"][role]
        basin = basin_by_id[hypothesis["basin_ids"][role]]
        reference_y_roi = result["debug"]["separator_result"][
            "reference_y_roi"
        ]
        original_center = float(
            np.interp(
                reference_y_roi,
                basin["band_centers_y_roi"],
                basin["center_x_by_band_roi"],
            )
        )
        shift = refinements[role]["median_shift_px"]
        refined_center = original_center + shift
        left_boundary = float(
            np.interp(
                reference_y_roi,
                basin["band_centers_y_roi"],
                basin["left_x_by_band_roi"],
            )
        )
        right_boundary = float(
            np.interp(
                reference_y_roi,
                basin["band_centers_y_roi"],
                basin["right_x_by_band_roi"],
            )
        )
        if not left_boundary < refined_center < right_boundary:
            refinements[role]["adopted"] = False
            refinements[role]["reason"] = (
                "refined_center_would_leave_original_basin"
            )
            refinements[role]["median_shift_px"] = 0.0
            refined_center = original_center
        original["arithmetic_center_x_at_reference_roi"] = (
            original_center
        )
        original["center_x_at_reference_roi"] = refined_center
        original["center_x_at_reference_global"] = (
            roi_x0 + refined_center
        )
        original["refinement_adopted"] = refinements[role][
            "adopted"
        ]
    reference_x_roi = (
        result["debug"]["separator_result"]["reference_x_roi"]
    )
    left_center = combined_geometry["basins"]["left"][
        "center_x_at_reference_roi"
    ]
    right_center = combined_geometry["basins"]["right"][
        "center_x_at_reference_roi"
    ]
    if combined_geometry["reference_relation"] == "inside_basin":
        combined_geometry["refined_reference_to_left_distance_px"] = (
            reference_x_roi - left_center
        )
        combined_geometry["refined_reference_to_right_distance_px"] = (
            right_center - reference_x_roi
        )
    else:
        combined_geometry[
            "reference_to_left_basin_center_distance_px"
        ] = reference_x_roi - left_center
        combined_geometry[
            "reference_to_right_basin_center_distance_px"
        ] = right_center - reference_x_roi
        combined_geometry["basin_center_spacing_px"] = (
            right_center - left_center
        )
    identity_after = _identity(result)
    if identity_before != identity_after:
        raise AssertionError("center refinement changed formal identity")
    return {
        "attempted": True,
        "adopted": any(
            item["adopted"] for item in refinements.values()
        ),
        "reason": "per_basin_diagnostics_below",
        "identity_before": identity_before,
        "identity_after": identity_after,
        "combined_geometry": combined_geometry,
        "basins": refinements,
    }


def _identity(result: dict) -> dict:
    hypothesis = result.get("final_hypothesis")
    return {
        "success": bool(result["success"]),
        "status": result["status"],
        "unavailable_reason": result["unavailable_reason"],
        "basin_ids": (
            None if hypothesis is None else hypothesis["basin_ids"]
        ),
        "separator_ids": (
            None if hypothesis is None else hypothesis.get("separator_ids")
        ),
        "separator_sequence": (
            None
            if hypothesis is None
            else hypothesis["separator_sequence"]
        ),
    }


def _expected_identity_matches(actual: dict, expected: dict) -> bool:
    if actual["success"] != expected["success"]:
        return False
    if not expected["success"]:
        return (
            actual["unavailable_reason"]
            == expected["unavailable_reason"]
        )
    return (
        actual["basin_ids"] == expected["basin_ids"]
        and actual["separator_sequence"]
        == expected["separator_sequence"]
    )


def _case_roi(case: dict, image_gray: np.ndarray) -> dict:
    if "roi_bounds_global" in case:
        return dict(case["roi_bounds_global"])
    reference = case["reference_global"]
    bounds = calculate_interactive_roi_bounds(
        image_gray.shape,
        reference["x"],
        reference["y"],
        INTERACTIVE_CONFIG,
    )
    return bounds_dict(bounds)


def _diagnose_case(
    case: dict,
    image_gray: np.ndarray,
) -> dict:
    reference = case["reference_global"]
    roi_bounds = _case_roi(case, image_gray)
    result = joint.run_joint_case(
        image_gray,
        reference,
        roi_bounds,
        case.get("direction", "vertical"),
        joint.DEFAULT_CONFIG,
    )
    separator_result = result["debug"]["separator_result"]
    raw_roi = separator.extract_raw_roi(image_gray, roi_bounds)
    directional = separator._directional_roi(  # noqa: SLF001
        raw_roi,
        case.get("direction", "vertical"),
    )
    evidence = separator.build_raw_gray_response(
        directional,
        separator.DEFAULT_CONFIG,
    )
    raw_profiles = _raw_band_profiles(directional, evidence)
    raw_pitch = result["debug"]["raw_pitch_result"]
    diagnostic_pitch = raw_pitch.get("diagnostic_pitch_px")
    if diagnostic_pitch is None:
        accepted_x = sorted(
            _path_x_at_y(
                candidate,
                separator_result["reference_y_roi"],
            )
            for candidate in separator_result["candidates"]
            if candidate["accepted"]
        )
        diagnostic_pitch = (
            float(np.median(np.diff(accepted_x)))
            if len(accepted_x) >= 2
            else float(directional.shape[1])
        )
    local_pitch = max(float(diagnostic_pitch), 1.0)
    raw_candidates = _raw_candidates_before_dedup(evidence)
    plateau_diagnostics = [
        _candidate_plateau_diagnostic(
            candidate,
            raw_profiles,
            local_pitch,
        )
        for candidate in separator_result["candidates"]
    ]
    basin_diagnostics = [
        _basin_gray_diagnostic(basin, raw_profiles)
        for basin in result["debug"]["basin_graph"]["basin_candidates"]
    ]
    identity = _identity(result)
    return {
        "sample_id": case["sample_id"],
        "image_name": case["image_name"],
        "reference_global": reference,
        "roi_bounds_global": roi_bounds,
        "direction": case.get("direction", "vertical"),
        "identity": identity,
        "expected_identity_matches": (
            _expected_identity_matches(identity, case["expected"])
            if "expected" in case
            else None
        ),
        "raw_pitch": {
            key: raw_pitch.get(key)
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
        "raw_gray_profiles": {
            "profile_source": (
                "raw ROI percentile per frozen separator band"
            ),
            "percentile": (
                separator.DEFAULT_CONFIG.band_profile_percentile
            ),
            "band_bounds": [
                list(bounds) for bounds in evidence["band_bounds"]
            ],
            "band_centers_y_roi": evidence[
                "band_centers_y"
            ].tolist(),
            "profiles": raw_profiles,
        },
        "separator_candidates_before_dedup": raw_candidates,
        "separator_candidates_after_dedup": separator_result[
            "candidates"
        ],
        "candidate_plateau_diagnostics": plateau_diagnostics,
        "basin_gray_diagnostics": basin_diagnostics,
        "safety_checks": {
            "separator_parameters": asdict(
                separator.DEFAULT_CONFIG
            ),
            "joint_parameters": asdict(joint.DEFAULT_CONFIG),
            "separator_status": separator_result["status"],
            "separator_unavailable_reason": separator_result[
                "unavailable_reason"
            ],
            "arbitration_debug": separator_result[
                "arbitration_debug"
            ],
            "basin_graph": result["debug"]["basin_graph"],
            "reference_relation": result["debug"][
                "reference_relation"
            ],
        },
        "final_hypothesis": result["final_hypothesis"],
    }


def _draw_path(
    image: np.ndarray,
    path: dict,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    points = np.rint(
        np.column_stack(
            [
                path["x_by_band_roi"],
                path["band_centers_y_roi"],
            ]
        )
    ).astype(np.int32)
    cv2.polylines(
        image,
        [points.reshape(-1, 1, 2)],
        False,
        color,
        thickness,
        cv2.LINE_AA,
    )


def save_baseline_overlay(
    image_gray: np.ndarray,
    sample: dict,
    output_path: Path,
) -> None:
    bounds = sample["roi_bounds_global"]
    roi = separator.extract_raw_roi(image_gray, bounds)
    overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    for candidate in sample["separator_candidates_after_dedup"]:
        _draw_path(
            overlay,
            candidate,
            (0, 190, 0) if candidate["accepted"] else (0, 80, 220),
            1,
        )
    for basin in sample["safety_checks"]["basin_graph"][
        "verified_basins"
    ]:
        center_path = {
            "x_by_band_roi": basin["center_x_by_band_roi"],
            "band_centers_y_roi": basin["band_centers_y_roi"],
        }
        _draw_path(overlay, center_path, (150, 150, 150), 1)
    hypothesis = sample["final_hypothesis"]
    if hypothesis is not None:
        basin_by_id = {
            basin["basin_id"]: basin
            for basin in sample["safety_checks"]["basin_graph"][
                "verified_basins"
            ]
        }
        for role, color in (
            ("left", (255, 0, 0)),
            ("right", (0, 255, 255)),
        ):
            basin = basin_by_id[hypothesis["basin_ids"][role]]
            center_path = {
                "x_by_band_roi": basin["center_x_by_band_roi"],
                "band_centers_y_roi": basin["band_centers_y_roi"],
            }
            _draw_path(overlay, center_path, color, 2)
    reference = sample["reference_global"]
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise OSError(f"could not write {output_path}")


def save_contact_sheet(
    paths: Iterable[Path],
    output_path: Path,
    columns: int = 4,
) -> None:
    images = [cv2.imread(str(path)) for path in paths]
    images = [image for image in images if image is not None]
    if not images:
        return
    thumb_width, thumb_height = 500, 200
    rows = (len(images) + columns - 1) // columns
    sheet = np.full(
        (rows * thumb_height, columns * thumb_width, 3),
        245,
        dtype=np.uint8,
    )
    for index, image in enumerate(images):
        resized = cv2.resize(
            image,
            (thumb_width, thumb_height),
            interpolation=cv2.INTER_AREA,
        )
        row, column = divmod(index, columns)
        sheet[
            row * thumb_height : (row + 1) * thumb_height,
            column * thumb_width : (column + 1) * thumb_width,
        ] = resized
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise OSError(f"could not write {output_path}")


def _group_cases(
    manifest: dict,
    development: dict[str, dict],
) -> dict[str, list[dict]]:
    sample2_name = "Sample 2.bmp"
    issues = [
        {**case, "image_name": sample2_name}
        for case in manifest["sample2_issues"]
    ]
    random_cases = [
        {**case, "image_name": sample2_name}
        for case in manifest["sample2_random"]
    ]
    return {
        "cross_image": manifest["cross_image"],
        "sample2_issues": issues,
        "sample2_random": random_cases,
        "development": list(development.values()),
    }


def run_baseline(
    images_dir: Path,
    output_dir: Path,
) -> dict:
    manifest = load_evaluation_manifest()
    development = load_development_annotations()
    grouped_cases = _group_cases(manifest, development)
    image_cache: dict[str, np.ndarray] = {}
    samples_by_group = {}
    for group_name, cases in grouped_cases.items():
        group_samples = []
        overlay_paths = []
        for case in cases:
            image_name = case["image_name"]
            if image_name not in image_cache:
                image_path = images_dir / image_name
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                image_cache[image_name] = (
                    separator.load_grayscale_image(image_path)
                )
            sample = _diagnose_case(
                case,
                image_cache[image_name],
            )
            group_samples.append(sample)
            overlay_path = (
                output_dir
                / "baseline_overlays"
                / group_name
                / f"{case['sample_id']}.png"
            )
            save_baseline_overlay(
                image_cache[image_name],
                sample,
                overlay_path,
            )
            overlay_paths.append(overlay_path)
        samples_by_group[group_name] = group_samples
        save_contact_sheet(
            overlay_paths,
            output_dir
            / "baseline_overlays"
            / f"{group_name}_contact_sheet.png",
        )

    development_evaluations = []
    for case in grouped_cases["development"]:
        development_evaluations.append(
            joint._evaluate_one(  # noqa: SLF001
                case,
                image_cache[case["image_name"]],
                joint.DEFAULT_CONFIG,
            )
        )
    report = {
        "report_revision": (
            "cross_image_correction_phase1_baseline_v1"
        ),
        "algorithm_revision": ALGORITHM_REVISION,
        "release_commit": RELEASE_COMMIT,
        "heldout_read": False,
        "images_dir": str(images_dir.resolve()),
        "evaluation_manifest": str(
            EVALUATION_MANIFEST_PATH.relative_to(PROJECT_ROOT)
        ),
        "development_source": str(
            DEVELOPMENT_PATH.relative_to(PROJECT_ROOT)
        ),
        **git_provenance(),
        "groups": samples_by_group,
        "development_metrics": joint._summarize(  # noqa: SLF001
            development_evaluations
        ),
        "identity_mismatches": {
            group_name: [
                sample["sample_id"]
                for sample in samples
                if sample["expected_identity_matches"] is False
            ]
            for group_name, samples in samples_by_group.items()
            if group_name != "development"
        },
    }
    atomic_write_json(
        output_dir / "baseline_report.json",
        report,
    )
    return report


def _evaluate_development_centerized(
    annotation: dict,
    image_gray: np.ndarray,
    correction_config: CrossImageCorrectionConfig,
) -> dict:
    def patched_separator(
        image,
        reference,
        bounds,
        direction="vertical",
        _separator_config=None,
    ):
        return detect_centerized_separator_paths(
            image,
            reference,
            bounds,
            direction,
            correction_config,
        )

    def patched_joint(
        image,
        reference,
        bounds,
        direction="vertical",
        _joint_config=None,
    ):
        return run_centerized_joint_case(
            image,
            reference,
            bounds,
            direction,
            correction_config,
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


def _variant_case(
    case: dict,
    image_gray: np.ndarray,
    config: CrossImageCorrectionConfig,
) -> dict:
    roi_bounds = _case_roi(case, image_gray)
    result = run_centerized_joint_case(
        image_gray,
        case["reference_global"],
        roi_bounds,
        case.get("direction", "vertical"),
        config,
    )
    refinement = refine_accepted_basin_centers(
        result,
        image_gray,
        roi_bounds,
        case.get("direction", "vertical"),
        config,
    )
    separator_result = result["debug"]["separator_result"]
    return {
        "sample_id": case["sample_id"],
        "image_name": case["image_name"],
        "reference_global": case["reference_global"],
        "roi_bounds_global": roi_bounds,
        "plateau_only_identity": _identity(result),
        "combined_identity": refinement["identity_after"],
        "expected_identity_matches": (
            _expected_identity_matches(
                _identity(result),
                case["expected"],
            )
            if "expected" in case
            else None
        ),
        "plateau_centerization": separator_result[
            "plateau_centerization"
        ],
        "plateau_only_final_hypothesis": result[
            "final_hypothesis"
        ],
        "plateau_only_safety": {
            "separator_status": separator_result["status"],
            "separator_unavailable_reason": separator_result[
                "unavailable_reason"
            ],
            "arbitration_debug": separator_result[
                "arbitration_debug"
            ],
            "basin_graph": result["debug"]["basin_graph"],
            "reference_relation": result["debug"][
                "reference_relation"
            ],
        },
        "raw_pitch": {
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
        "combined_refinement": refinement,
    }


def _selected_valley_metrics(
    sample: dict,
    variant: str,
) -> dict | None:
    if variant == "baseline":
        hypothesis = sample["final_hypothesis"]
        if hypothesis is None:
            return None
        diagnostics = {
            item["basin_id"]: item
            for item in sample["basin_gray_diagnostics"]
        }
        selected = [
            diagnostics[hypothesis["basin_ids"][role]]
            for role in ("left", "right")
        ]
        medians = [
            item["arithmetic_to_minimum_median_px"]
            for item in selected
        ]
        maxima = [
            item["arithmetic_to_minimum_max_px"]
            for item in selected
        ]
        return {
            "median_error_px": float(np.median(medians)),
            "max_error_px": float(np.max(maxima)),
            "per_side": {
                role: {
                    "median_error_px": item[
                        "arithmetic_to_minimum_median_px"
                    ],
                    "max_error_px": item[
                        "arithmetic_to_minimum_max_px"
                    ],
                }
                for role, item in zip(("left", "right"), selected)
            },
        }

    refinement = sample["combined_refinement"]
    if not refinement["attempted"]:
        return None
    key = (
        "baseline_minimum_error"
        if variant == "plateau_only"
        else "refined_minimum_error"
    )
    per_side = {
        role: {
            "median_error_px": item[f"{key}_median_px"],
            "max_error_px": item[f"{key}_max_px"],
            "adopted": item["adopted"],
            "shift_px": item["median_shift_px"],
        }
        for role, item in refinement["basins"].items()
    }
    medians = [
        item["median_error_px"]
        for item in per_side.values()
        if item["median_error_px"] is not None
    ]
    maxima = [
        item["max_error_px"]
        for item in per_side.values()
        if item["max_error_px"] is not None
    ]
    if not medians:
        return None
    return {
        "median_error_px": float(np.median(medians)),
        "max_error_px": float(np.max(maxima)),
        "per_side": per_side,
    }


def _draw_result_geometry(
    roi_gray: np.ndarray,
    result: dict,
    refinement: dict | None = None,
) -> np.ndarray:
    overlay = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    separator_result = result["debug"]["separator_result"]
    for candidate in separator_result["candidates"]:
        _draw_path(
            overlay,
            candidate,
            (0, 190, 0) if candidate["accepted"] else (0, 80, 220),
            1,
        )
    hypothesis = result["final_hypothesis"]
    if hypothesis is None:
        return overlay
    basin_by_id = {
        basin["basin_id"]: basin
        for basin in result["debug"]["basin_graph"][
            "verified_basins"
        ]
    }
    for role, color in (
        ("left", (255, 0, 0)),
        ("right", (0, 255, 255)),
    ):
        basin = basin_by_id[hypothesis["basin_ids"][role]]
        original_path = {
            "x_by_band_roi": basin["center_x_by_band_roi"],
            "band_centers_y_roi": basin["band_centers_y_roi"],
        }
        _draw_path(overlay, original_path, color, 2)
        if refinement is not None:
            detail = refinement["basins"][role]
            refined_path = {
                "x_by_band_roi": detail[
                    "bandwise_center_path_x_roi"
                ],
                "band_centers_y_roi": basin[
                    "band_centers_y_roi"
                ],
            }
            _draw_path(
                overlay,
                refined_path,
                (255, 0, 255),
                2,
            )
            minimum_x = [
                band.get(
                    "gray_minimum_center_x_roi",
                    original,
                )
                for band, original in zip(
                    detail["bands"],
                    basin["center_x_by_band_roi"],
                )
            ]
            minimum_path = {
                "x_by_band_roi": minimum_x,
                "band_centers_y_roi": basin[
                    "band_centers_y_roi"
                ],
            }
            _draw_path(
                overlay,
                minimum_path,
                (0, 128, 255),
                1,
            )
    return overlay


def save_three_version_overlay(
    image_gray: np.ndarray,
    case: dict,
    baseline_result: dict,
    plateau_result: dict,
    refinement: dict,
    output_path: Path,
) -> None:
    bounds = _case_roi(case, image_gray)
    roi = separator.extract_raw_roi(image_gray, bounds)
    baseline_image = _draw_result_geometry(roi, baseline_result)
    plateau_image = _draw_result_geometry(roi, plateau_result)
    combined_image = _draw_result_geometry(
        roi,
        plateau_result,
        refinement,
    )
    labels = (
        "Baseline",
        "Plateau centerization",
        "Plateau + basin refinement",
    )
    panels = []
    for label, panel in zip(
        labels,
        (baseline_image, plateau_image, combined_image),
    ):
        titled = np.full(
            (panel.shape[0] + 28, panel.shape[1], 3),
            245,
            dtype=np.uint8,
        )
        titled[28:] = panel
        cv2.putText(
            titled,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        reference = case["reference_global"]
        cv2.drawMarker(
            titled,
            (
                reference["x"] - bounds["x0"],
                reference["y"] - bounds["y0"] + 28,
            ),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            17,
            2,
        )
        panels.append(titled)
    comparison = np.hstack(panels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), comparison):
        raise OSError(f"could not write {output_path}")


def _variant_identity_changes(
    baseline_samples: list[dict],
    variant_samples: list[dict],
) -> list[dict]:
    baseline_by_id = {
        item["sample_id"]: item for item in baseline_samples
    }
    changes = []
    for variant in variant_samples:
        baseline = baseline_by_id[variant["sample_id"]]
        if (
            baseline["identity"]
            != variant["plateau_only_identity"]
        ):
            changes.append(
                {
                    "sample_id": variant["sample_id"],
                    "baseline": baseline["identity"],
                    "plateau_only": variant[
                        "plateau_only_identity"
                    ],
                }
            )
    return changes


def _gate_report(
    baseline_report: dict,
    variants: dict[str, list[dict]],
    development_metrics: dict,
) -> dict:
    issue_changes = _variant_identity_changes(
        baseline_report["groups"]["sample2_issues"],
        variants["sample2_issues"],
    )
    random_changes = _variant_identity_changes(
        baseline_report["groups"]["sample2_random"],
        variants["sample2_random"],
    )
    sample2_plateau_adoptions = {
        group_name: [
            {
                "sample_id": sample["sample_id"],
                "adopted_count": sample[
                    "plateau_centerization"
                ]["adopted_count"],
            }
            for sample in variants[group_name]
            if sample["plateau_centerization"]["adopted_count"]
        ]
        for group_name in ("sample2_issues", "sample2_random")
    }
    cross_by_id = {
        sample["sample_id"]: sample
        for sample in variants["cross_image"]
    }
    baseline_cross = {
        sample["sample_id"]: sample
        for sample in baseline_report["groups"]["cross_image"]
    }
    stripe5 = cross_by_id["X002"]
    stripe1 = cross_by_id["X003"]
    stripe5_metrics = {
        version: _selected_valley_metrics(
            baseline_cross["X002"]
            if version == "baseline"
            else stripe5,
            version,
        )
        for version in ("baseline", "plateau_only", "combined")
    }
    stripe1_metrics = {
        version: _selected_valley_metrics(
            baseline_cross["X003"]
            if version == "baseline"
            else stripe1,
            version,
        )
        for version in ("baseline", "plateau_only", "combined")
    }
    baseline_unavailable_became_success = [
        change
        for change in _variant_identity_changes(
            baseline_report["groups"]["cross_image"],
            variants["cross_image"],
        )
        if (
            not change["baseline"]["success"]
            and change["plateau_only"]["success"]
        )
    ]
    checks = {
        "sample2_issue_identity_unchanged": not issue_changes,
        "sample2_random_identity_unchanged": not random_changes,
        "sample2_plateau_centerization_not_triggered": not any(
            sample2_plateau_adoptions.values()
        ),
        "development_wrong_success_zero": (
            development_metrics["wrong_success"] == 0
        ),
        "development_false_split_zero": (
            development_metrics["false_split"] == 0
        ),
        "development_ambiguous_unavailable_not_released": not (
            development_metrics["ambiguous_formally_available"]
            or development_metrics[
                "unavailable_formally_available"
            ]
        ),
        "development_crossing_not_released": not (
            development_metrics[
                "crossing_or_order_conflict_formally_available"
            ]
        ),
        "cross_image_safe_failures_not_released": not (
            baseline_unavailable_became_success
        ),
        "stripe5_plateau_centerized": (
            stripe5["plateau_centerization"]["adopted_count"] > 0
        ),
        "stripe5_basin_identity_preserved": (
            baseline_cross["X002"]["identity"]["basin_ids"]
            == stripe5["plateau_only_identity"]["basin_ids"]
        ),
        "stripe5_combined_median_error_at_most_2px": bool(
            stripe5_metrics["combined"]
            and stripe5_metrics["combined"]["median_error_px"]
            <= 2.0
        ),
        "stripe1_basin_identity_preserved": (
            baseline_cross["X003"]["identity"]["basin_ids"]
            == stripe1["plateau_only_identity"]["basin_ids"]
        ),
        "stripe1_combined_median_error_at_most_1_5px": bool(
            stripe1_metrics["combined"]
            and stripe1_metrics["combined"]["median_error_px"]
            <= 1.5
        ),
        "stripe1_right_error_improved": bool(
            stripe1_metrics["combined"]
            and stripe1_metrics["baseline"]
            and stripe1_metrics["combined"]["per_side"]["right"][
                "median_error_px"
            ]
            < stripe1_metrics["baseline"]["per_side"]["right"][
                "median_error_px"
            ]
        ),
    }
    return {
        "all_passed": all(checks.values()),
        "checks": checks,
        "sample2_issue_identity_changes": issue_changes,
        "sample2_random_identity_changes": random_changes,
        "sample2_plateau_adoptions": sample2_plateau_adoptions,
        "baseline_unavailable_became_success": (
            baseline_unavailable_became_success
        ),
        "stripe5_valley_metrics": stripe5_metrics,
        "stripe1_valley_metrics": stripe1_metrics,
    }


def run_experiment(
    images_dir: Path,
    output_dir: Path,
    config: CrossImageCorrectionConfig = DEFAULT_CORRECTION_CONFIG,
) -> dict:
    baseline_path = output_dir / "baseline_report.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(
            "run Phase 1 baseline before the experiment"
        )
    baseline_report = json.loads(baseline_path.read_text())
    manifest = load_evaluation_manifest()
    development = load_development_annotations()
    grouped_cases = _group_cases(manifest, development)
    image_cache: dict[str, np.ndarray] = {}
    variants: dict[str, list[dict]] = {}
    for group_name, cases in grouped_cases.items():
        group_variants = []
        comparison_paths = []
        baseline_by_id = {
            sample["sample_id"]: sample
            for sample in baseline_report["groups"][group_name]
        }
        for case in cases:
            image_name = case["image_name"]
            if image_name not in image_cache:
                image_cache[image_name] = (
                    separator.load_grayscale_image(
                        images_dir / image_name
                    )
                )
            image = image_cache[image_name]
            variant = _variant_case(case, image, config)
            group_variants.append(variant)
            baseline_result = FROZEN_RUN_JOINT_CASE(
                image,
                case["reference_global"],
                _case_roi(case, image),
                case.get("direction", "vertical"),
                joint.DEFAULT_CONFIG,
            )
            plateau_result = run_centerized_joint_case(
                image,
                case["reference_global"],
                _case_roi(case, image),
                case.get("direction", "vertical"),
                config,
            )
            comparison_path = (
                output_dir
                / "three_version_overlays"
                / group_name
                / f"{case['sample_id']}.png"
            )
            save_three_version_overlay(
                image,
                case,
                baseline_result,
                plateau_result,
                variant["combined_refinement"],
                comparison_path,
            )
            comparison_paths.append(comparison_path)
            variant["baseline_identity"] = baseline_by_id[
                case["sample_id"]
            ]["identity"]
            variant["identity_changed_from_baseline"] = (
                variant["baseline_identity"]
                != variant["plateau_only_identity"]
            )
        variants[group_name] = group_variants
        save_contact_sheet(
            comparison_paths,
            output_dir
            / "three_version_overlays"
            / f"{group_name}_contact_sheet.png",
            columns=2,
        )

    development_evaluations = [
        _evaluate_development_centerized(
            case,
            image_cache[case["image_name"]],
            config,
        )
        for case in grouped_cases["development"]
    ]
    development_metrics = joint._summarize(  # noqa: SLF001
        development_evaluations
    )
    gates = _gate_report(
        baseline_report,
        variants,
        development_metrics,
    )
    report = {
        "report_revision": (
            "cross_image_correction_three_version_evaluation_v1"
        ),
        "algorithm_revision": ALGORITHM_REVISION,
        "configuration": asdict(config),
        "configuration_checksum": (
            correction_configuration_checksum(config)
        ),
        "release_commit": RELEASE_COMMIT,
        "heldout_read": False,
        "production_runtime_modified": False,
        "formal_ui_modified": False,
        **git_provenance(),
        "baseline_report_sha256": sha256_file(baseline_path),
        "versions": {
            "baseline": {
                "source_report": str(baseline_path),
                "development_metrics": baseline_report[
                    "development_metrics"
                ],
            },
            "plateau_only": {
                "groups": variants,
                "development_metrics": development_metrics,
            },
            "combined": {
                "identity_source": "plateau_only_unchanged",
                "center_refinements": {
                    group_name: [
                        {
                            "sample_id": sample["sample_id"],
                            "refinement": sample[
                                "combined_refinement"
                            ],
                        }
                        for sample in samples
                    ]
                    for group_name, samples in variants.items()
                },
                "development_metrics": development_metrics,
            },
        },
        "safety_gates": gates,
    }
    atomic_write_json(
        output_dir / "three_version_report.json",
        report,
    )
    return report


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline cross-image correction audit",
    )
    parser.add_argument(
        "--phase",
        choices=("baseline", "experiment"),
        default="baseline",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    if arguments.phase == "baseline":
        report = run_baseline(
            arguments.images_dir,
            arguments.output_dir,
        )
        print(
            json.dumps(
                {
                    "report": str(
                        arguments.output_dir
                        / "baseline_report.json"
                    ),
                    "identity_mismatches": report[
                        "identity_mismatches"
                    ],
                    "development_metrics": report[
                        "development_metrics"
                    ],
                },
                indent=2,
            )
        )
    else:
        report = run_experiment(
            arguments.images_dir,
            arguments.output_dir,
        )
        print(
            json.dumps(
                {
                    "report": str(
                        arguments.output_dir
                        / "three_version_report.json"
                    ),
                    "configuration_checksum": report[
                        "configuration_checksum"
                    ],
                    "safety_gates": report["safety_gates"],
                    "development_metrics": report["versions"][
                        "plateau_only"
                    ]["development_metrics"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
