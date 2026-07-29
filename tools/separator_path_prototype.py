"""Stage-2 offline separator-path prototype.

This experiment reads only frozen development annotations and source grayscale
ROIs.  Separator paths are inferred before human control points are consulted.
The human paths are used only by the evaluation functions at the end.

The prototype deliberately does not import production detection code, tracks,
thresholding, morphology, pitch maps, or current detector results.

Run the complete development evaluation with:

    .venv/bin/python tools/separator_path_prototype.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_VERSION = "separator_path_gt_v1"
DEVELOPMENT_PATH = (
    PROJECT_ROOT
    / "tests"
    / "data"
    / DATASET_VERSION
    / "development.json"
)
IMAGES_DIR = PROJECT_ROOT / "images"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "separator_path_stage2_v1_1"
)
REPORT_PATH = OUTPUT_DIR / "development_report.json"
OVERLAY_DIR = OUTPUT_DIR / "overlays"
ALGORITHM_REVISION = "separator_path_stage2_v1_1"
FROZEN_V1_COMMIT = "2b31cb577dd1ae27b4bace6043194ab76d6fde6c"
FROZEN_V1_CONFIGURATION_CHECKSUM = (
    "8e28ca014c0abd378908adbaa3a54a1bd8b9281f450f5985a7e460af4733a184"
)
FROZEN_CANDIDATE_SOURCE_CHECKSUM = (
    "5b1b8f71a8b33b2caa432f62c83f42e896eebfa4ee96c4114fa5d329d58aa630"
)
FROZEN_DEVELOPMENT_CANDIDATE_OUTPUT_CHECKSUM = (
    "c48295fdacb145cef76a2062e9457a07b41d94fbc0c6934e54d0483630af9255"
)

ROLE_ORDER = (
    "left_adjacent",
    "left_clicked_boundary",
    "right_clicked_boundary",
    "right_adjacent",
)
ROLE_COLORS_BGR = {
    "left_adjacent": (255, 220, 0),
    "left_clicked_boundary": (255, 80, 30),
    "right_clicked_boundary": (0, 220, 255),
    "right_adjacent": (220, 70, 220),
}
NUMERIC_VISIBILITY = {"fully_visible", "partially_visible"}


@dataclass(frozen=True)
class SeparatorPathConfig:
    """Tunable parameters for raw-gray path evidence and arbitration."""

    band_count: int = 12
    band_profile_percentile: float = 78.0
    profile_smoothing_sigma_px: float = 1.1
    baseline_radius_px: int = 11
    intensity_weight: float = 0.45
    local_contrast_weight: float = 0.85
    aggregate_top_band_count: int = 4
    aggregate_smoothing_sigma_px: float = 1.0
    seed_response_floor: float = 0.85
    seed_minimum_distance_px: int = 5
    maximum_seed_count: int = 48
    trace_search_radius_px: int = 9
    transition_penalty_per_px: float = 0.12
    supported_band_response: float = 0.75
    minimum_supported_fraction: float = 0.24
    minimum_top_response: float = 0.95
    minimum_path_score: float = 1.05
    duplicate_path_distance_px: float = 4.5
    same_ridge_maximum_span_px: int = 32
    same_ridge_valley_ratio: float = 0.55
    same_ridge_minimum_band_fraction: float = 0.50
    reference_guard_px: float = 3.5
    match_tolerance_px: float = 6.0
    maximum_width_aware_match_tolerance_px: float = 20.0
    gt_reference_percentile_threshold: float = 0.95
    reference_relationship_uncertainty_px: float = 2.0
    minimum_role_path_support_fraction: float = 0.50
    maximum_role_path_mean_step_px: float = 1.50
    minimum_full_height_order_ratio: float = 0.70
    minimum_dark_basin_edge_gap_px: float = 1.0
    maximum_dark_basin_edge_gap_ratio: float = 2.70
    minimum_dark_basin_contrast: float = 0.15
    minimum_dark_basin_band_fraction: float = 0.75


DEFAULT_CONFIG = SeparatorPathConfig()


def canonical_configuration(config: SeparatorPathConfig) -> dict:
    return {
        "algorithm_revision": ALGORITHM_REVISION,
        "frozen_candidate_generation": {
            "parent_commit": FROZEN_V1_COMMIT,
            "v1_configuration_checksum": (
                FROZEN_V1_CONFIGURATION_CHECKSUM
            ),
            "source_checksum": FROZEN_CANDIDATE_SOURCE_CHECKSUM,
        },
        "parameters": asdict(config),
    }


def configuration_checksum(config: SeparatorPathConfig) -> str:
    payload = json.dumps(
        canonical_configuration(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_grayscale_image(path: Path) -> np.ndarray:
    """Decode source pixels directly as one grayscale array."""

    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image_gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image_gray is None or image_gray.size == 0:
        raise ValueError(f"OpenCV could not decode {path}")
    return image_gray


def extract_raw_roi(
    image_gray: np.ndarray,
    roi_bounds_global: dict,
) -> np.ndarray:
    if image_gray.ndim != 2:
        raise ValueError("separator prototype requires a grayscale image")
    height, width = image_gray.shape
    x0 = roi_bounds_global["x0"]
    y0 = roi_bounds_global["y0"]
    x1 = roi_bounds_global["x1"]
    y1 = roi_bounds_global["y1"]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("ROI is outside the source image")
    return image_gray[y0:y1, x0:x1]


def _directional_roi(
    roi_gray: np.ndarray,
    direction: str,
) -> np.ndarray:
    """Place the separator-offset axis on x without changing gray values."""

    if direction == "vertical":
        return roi_gray
    if direction == "horizontal":
        return roi_gray.T
    raise ValueError("direction must be vertical or horizontal")


def _directional_reference(
    reference_global: dict,
    bounds: dict,
    direction: str,
) -> tuple[float, float]:
    x_roi = reference_global["x"] - bounds["x0"]
    y_roi = reference_global["y"] - bounds["y0"]
    if direction == "vertical":
        return float(x_roi), float(y_roi)
    if direction == "horizontal":
        return float(y_roi), float(x_roi)
    raise ValueError("direction must be vertical or horizontal")


def _band_bounds(height: int, count: int) -> list[tuple[int, int]]:
    if count < 3:
        raise ValueError("at least three vertical bands are required")
    edges = np.linspace(0, height, count + 1).round().astype(int)
    return [
        (int(edges[index]), int(edges[index + 1]))
        for index in range(count)
        if edges[index + 1] > edges[index]
    ]


def _smooth_profile(profile: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(
        profile.astype(np.float64).reshape(1, -1),
        (0, 0),
        sigmaX=sigma,
        sigmaY=0,
        borderType=cv2.BORDER_REFLECT,
    ).reshape(-1)


def _box_profile(profile: np.ndarray, radius: int) -> np.ndarray:
    kernel_width = radius * 2 + 1
    return cv2.blur(
        profile.astype(np.float64).reshape(1, -1),
        (kernel_width, 1),
        borderType=cv2.BORDER_REFLECT,
    ).reshape(-1)


def _positive_robust_z(
    values: np.ndarray,
    scale_floor: float,
) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, scale_floor)
    return np.maximum(0.0, (values - median) / scale)


def build_raw_gray_response(
    directional_roi: np.ndarray,
    config: SeparatorPathConfig = DEFAULT_CONFIG,
) -> dict:
    """Build graded band evidence from source grayscale only."""

    height, width = directional_roi.shape
    bounds = _band_bounds(height, config.band_count)
    responses = []
    profiles = []
    band_centers = []
    for y0, y1 in bounds:
        band = directional_roi[y0:y1].astype(np.float64)
        profile = np.percentile(
            band,
            config.band_profile_percentile,
            axis=0,
        )
        smooth = _smooth_profile(
            profile,
            config.profile_smoothing_sigma_px,
        )
        baseline = _box_profile(smooth, config.baseline_radius_px)
        local_contrast = smooth - baseline
        intensity_evidence = _positive_robust_z(smooth, 8.0)
        contrast_evidence = _positive_robust_z(local_contrast, 3.0)
        response = (
            config.intensity_weight * intensity_evidence
            + config.local_contrast_weight * contrast_evidence
        )
        profiles.append(smooth)
        responses.append(response)
        band_centers.append((y0 + y1 - 1) / 2.0)

    response_matrix = np.stack(responses, axis=0)
    top_count = min(
        config.aggregate_top_band_count,
        response_matrix.shape[0],
    )
    top_values = np.partition(
        response_matrix,
        response_matrix.shape[0] - top_count,
        axis=0,
    )[-top_count:]
    aggregate = np.mean(top_values, axis=0)
    aggregate = _smooth_profile(
        aggregate,
        config.aggregate_smoothing_sigma_px,
    )
    return {
        "band_bounds": bounds,
        "band_centers_y": np.asarray(band_centers, dtype=np.float64),
        "profiles": np.stack(profiles, axis=0),
        "responses": response_matrix,
        "aggregate_response": aggregate,
        "width": width,
        "height": height,
    }


def _seed_peaks(
    aggregate_response: np.ndarray,
    config: SeparatorPathConfig,
) -> list[dict]:
    peaks = []
    for index in range(1, aggregate_response.size - 1):
        value = float(aggregate_response[index])
        if (
            value >= config.seed_response_floor
            and value >= aggregate_response[index - 1]
            and value > aggregate_response[index + 1]
        ):
            peaks.append({"x": index, "response": value})
    peaks.sort(key=lambda item: (-item["response"], item["x"]))

    kept = []
    for peak in peaks:
        if all(
            abs(peak["x"] - other["x"])
            >= config.seed_minimum_distance_px
            for other in kept
        ):
            kept.append(peak)
        if len(kept) >= config.maximum_seed_count:
            break
    return sorted(kept, key=lambda item: item["x"])


def _trace_one_seed(
    seed: dict,
    evidence: dict,
    config: SeparatorPathConfig,
) -> dict:
    responses = evidence["responses"]
    band_count, image_width = responses.shape
    x0 = max(0, seed["x"] - config.trace_search_radius_px)
    x1 = min(
        image_width,
        seed["x"] + config.trace_search_radius_px + 1,
    )
    states = np.arange(x0, x1, dtype=int)
    state_count = states.size
    scores = np.full((band_count, state_count), -np.inf)
    previous = np.full((band_count, state_count), -1, dtype=int)
    scores[0] = responses[0, states]

    for band_index in range(1, band_count):
        for current_index, current_x in enumerate(states):
            transition_scores = (
                scores[band_index - 1]
                - config.transition_penalty_per_px
                * np.abs(states - current_x)
            )
            best_previous = int(np.argmax(transition_scores))
            scores[band_index, current_index] = (
                float(transition_scores[best_previous])
                + float(responses[band_index, current_x])
            )
            previous[band_index, current_index] = best_previous

    final_index = int(np.argmax(scores[-1]))
    state_indices = [final_index]
    for band_index in range(band_count - 1, 0, -1):
        state_indices.append(
            int(previous[band_index, state_indices[-1]])
        )
    state_indices.reverse()
    path_x = np.asarray(
        [states[index] for index in state_indices],
        dtype=np.float64,
    )
    path_response = responses[
        np.arange(band_count),
        path_x.astype(int),
    ]
    supported = path_response >= config.supported_band_response
    support_fraction = float(np.mean(supported))
    top_count = min(config.aggregate_top_band_count, band_count)
    top_response = float(
        np.mean(np.sort(path_response)[-top_count:])
    )
    transition = (
        float(np.mean(np.abs(np.diff(path_x))))
        if path_x.size > 1
        else 0.0
    )
    path_score = float(
        top_response
        + 0.45 * support_fraction
        - 0.04 * transition
    )
    accepted = bool(
        support_fraction >= config.minimum_supported_fraction
        and top_response >= config.minimum_top_response
        and path_score >= config.minimum_path_score
    )

    aggregate = evidence["aggregate_response"]
    center = int(round(float(np.median(path_x))))
    peak_value = float(aggregate[np.clip(center, 0, len(aggregate) - 1)])
    half_height = peak_value * 0.5
    left = center
    right = center
    while left > 0 and aggregate[left - 1] >= half_height:
        left -= 1
    while right + 1 < len(aggregate) and aggregate[right + 1] >= half_height:
        right += 1

    return {
        "seed_x_roi": int(seed["x"]),
        "seed_response": float(seed["response"]),
        "band_centers_y_roi": evidence["band_centers_y"].tolist(),
        "x_by_band_roi": path_x.tolist(),
        "response_by_band": path_response.tolist(),
        "supported_band_indices": np.flatnonzero(supported).tolist(),
        "support_fraction": support_fraction,
        "top_response": top_response,
        "mean_step_px": transition,
        "path_score": path_score,
        "peak_width_px": int(right - left + 1),
        "accepted": accepted,
        "rejection_reason": (
            None
            if accepted
            else _path_rejection_reason(
                support_fraction,
                top_response,
                path_score,
                config,
            )
        ),
    }


def _path_rejection_reason(
    support_fraction: float,
    top_response: float,
    path_score: float,
    config: SeparatorPathConfig,
) -> str:
    if support_fraction < config.minimum_supported_fraction:
        return "insufficient_vertical_support"
    if top_response < config.minimum_top_response:
        return "weak_raw_gray_response"
    if path_score < config.minimum_path_score:
        return "low_path_score"
    return "rejected"


def _path_x_at_y(path: dict, y_roi: float) -> float:
    return float(
        np.interp(
            y_roi,
            path["band_centers_y_roi"],
            path["x_by_band_roi"],
        )
    )


def _paths_are_duplicates(
    left: dict,
    right: dict,
    evidence: dict,
    config: SeparatorPathConfig,
    maximum_distance: float,
) -> bool:
    left_x = np.asarray(left["x_by_band_roi"], dtype=np.float64)
    right_x = np.asarray(right["x_by_band_roi"], dtype=np.float64)
    if float(np.median(np.abs(left_x - right_x))) <= maximum_distance:
        return True
    return _paths_share_one_bright_ridge(left, right, evidence, config)


def _paths_share_one_bright_ridge(
    left: dict,
    right: dict,
    evidence: dict,
    config: SeparatorPathConfig,
) -> bool:
    """Return whether two response peaks sit on one continuous bright ridge."""

    profiles = evidence["profiles"]
    left_x = np.rint(left["x_by_band_roi"]).astype(int)
    right_x = np.rint(right["x_by_band_roi"]).astype(int)
    agreements = [
        _profile_positions_share_bright_ridge(
            profiles[band_index],
            int(first_x),
            int(second_x),
            config,
        )
        for band_index, (first_x, second_x) in enumerate(
            zip(left_x, right_x)
        )
    ]
    return (
        float(np.mean(agreements))
        >= config.same_ridge_minimum_band_fraction
    )


def _profile_positions_share_bright_ridge(
    profile: np.ndarray,
    first_x: int,
    second_x: int,
    config: SeparatorPathConfig,
) -> bool:
    x0, x1 = sorted((first_x, second_x))
    span = x1 - x0
    if span <= 0:
        return True
    if span > config.same_ridge_maximum_span_px:
        return False
    local_x0 = max(0, x0 - 8)
    local_x1 = min(profile.size, x1 + 9)
    local = profile[local_x0:local_x1]
    dark_level = float(np.percentile(local, 10))
    peak_level = min(float(profile[x0]), float(profile[x1]))
    denominator = peak_level - dark_level
    if denominator <= 1.0:
        return False
    valley_level = float(np.percentile(profile[x0 : x1 + 1], 10))
    valley_ratio = (valley_level - dark_level) / denominator
    return valley_ratio >= config.same_ridge_valley_ratio


def trace_separator_candidates(
    evidence: dict,
    config: SeparatorPathConfig = DEFAULT_CONFIG,
) -> list[dict]:
    """Trace and deduplicate raw-gray separator hypotheses."""

    traced = [
        _trace_one_seed(seed, evidence, config)
        for seed in _seed_peaks(
            evidence["aggregate_response"],
            config,
        )
    ]
    traced.sort(
        key=lambda path: (
            not path["accepted"],
            -path["path_score"],
            path["seed_x_roi"],
        )
    )
    unique = []
    for path in traced:
        duplicate = next(
            (
                existing
                for existing in unique
                if _paths_are_duplicates(
                    path,
                    existing,
                    evidence,
                    config,
                    config.duplicate_path_distance_px,
                )
            ),
            None,
        )
        if duplicate is None:
            path["suppressed_candidates"] = []
            unique.append(path)
        else:
            duplicate["suppressed_candidates"].append(
                {
                    "seed_x_roi": path["seed_x_roi"],
                    "path_score": path["path_score"],
                    "reason": "same_physical_bright_ridge",
                }
            )
    unique.sort(key=lambda path: path["seed_x_roi"])
    for candidate_id, path in enumerate(unique, start=1):
        path["candidate_id"] = f"C{candidate_id:02d}"
    return unique


def _selected_paths_cross(
    selection: dict[str, dict],
) -> bool:
    if any(role not in selection for role in ROLE_ORDER):
        return False
    matrices = [
        np.asarray(selection[role]["x_by_band_roi"], dtype=np.float64)
        for role in ROLE_ORDER
    ]
    return any(
        bool(np.any(left >= right))
        for left, right in zip(matrices, matrices[1:])
    )


def _select_clicked_basin_paths_v1(
    candidates: list[dict],
    reference_x_roi: float,
    reference_y_roi: float,
    config: SeparatorPathConfig = DEFAULT_CONFIG,
) -> dict:
    """Assign the four local roles using raw-gray paths around reference."""

    accepted = [path for path in candidates if path["accepted"]]
    for path in accepted:
        path["x_at_reference_roi"] = _path_x_at_y(
            path,
            reference_y_roi,
        )
    accepted.sort(
        key=lambda path: (
            path["x_at_reference_roi"],
            path["candidate_id"],
        )
    )
    if not accepted:
        return {
            "status": "unavailable",
            "unavailable_reason": "no_accepted_separator_paths",
            "selection": {},
            "crossing": False,
        }

    nearest = min(
        accepted,
        key=lambda path: abs(
            path["x_at_reference_roi"] - reference_x_roi
        ),
    )
    reference_guard = max(
        config.reference_guard_px,
        min(8.0, nearest["peak_width_px"] / 2.0),
    )
    if (
        abs(nearest["x_at_reference_roi"] - reference_x_roi)
        <= reference_guard
    ):
        return {
            "status": "unavailable",
            "unavailable_reason": "reference_on_separator",
            "reference_separator_candidate_id": nearest["candidate_id"],
            "reference_separator_distance_px": abs(
                nearest["x_at_reference_roi"] - reference_x_roi
            ),
            "reference_guard_px": reference_guard,
            "selection": {},
            "crossing": False,
        }

    left = [
        path
        for path in accepted
        if path["x_at_reference_roi"] < reference_x_roi
    ]
    right = [
        path
        for path in accepted
        if path["x_at_reference_roi"] > reference_x_roi
    ]
    if len(left) < 2 or len(right) < 2:
        return {
            "status": "unavailable",
            "unavailable_reason": "insufficient_boundary_paths",
            "left_path_count": len(left),
            "right_path_count": len(right),
            "selection": {},
            "crossing": False,
        }

    selection = {
        "left_adjacent": left[-2],
        "left_clicked_boundary": left[-1],
        "right_clicked_boundary": right[0],
        "right_adjacent": right[1],
    }
    crossing = _selected_paths_cross(selection)
    if crossing:
        return {
            "status": "unavailable",
            "unavailable_reason": "predicted_path_crossing",
            "selection": {},
            "debug_selection": selection,
            "crossing": True,
        }
    return {
        "status": "available",
        "unavailable_reason": None,
        "selection": selection,
        "crossing": False,
    }


def _reference_guard(
    path: dict,
    config: SeparatorPathConfig,
) -> float:
    return max(
        config.reference_guard_px,
        min(8.0, path["peak_width_px"] / 2.0),
    )


def _full_height_pair_order(
    left: dict,
    right: dict,
    config: SeparatorPathConfig,
) -> dict:
    left_x = np.asarray(left["x_by_band_roi"], dtype=np.float64)
    right_x = np.asarray(right["x_by_band_roi"], dtype=np.float64)
    gaps = right_x - left_x
    crossing_indices = np.flatnonzero(gaps <= 0).tolist()
    median_gap = float(np.median(gaps))
    minimum_gap = float(np.min(gaps))
    minimum_to_median_ratio = (
        minimum_gap / median_gap if median_gap > 0 else None
    )
    stable = bool(
        not crossing_indices
        and minimum_to_median_ratio is not None
        and minimum_to_median_ratio
        >= config.minimum_full_height_order_ratio
    )
    return {
        "left_candidate_id": left["candidate_id"],
        "right_candidate_id": right["candidate_id"],
        "gap_by_band_px": gaps.tolist(),
        "minimum_gap_px": minimum_gap,
        "median_gap_px": median_gap,
        "minimum_to_median_ratio": minimum_to_median_ratio,
        "crossing_band_indices": crossing_indices,
        "stable_full_height_order": stable,
    }


def _dark_basin_evidence(
    left: dict,
    right: dict,
    evidence: dict,
    config: SeparatorPathConfig,
) -> dict:
    left_x = np.rint(left["x_by_band_roi"]).astype(int)
    right_x = np.rint(right["x_by_band_roi"]).astype(int)
    contrast_by_band = []
    for band_index, (left_value, right_value) in enumerate(
        zip(left_x, right_x)
    ):
        x0, x1 = sorted((int(left_value), int(right_value)))
        if x1 - x0 < 3:
            contrast_by_band.append(-1.0)
            continue
        profile = evidence["profiles"][band_index]
        interior = profile[x0 + 1 : x1]
        local = profile[
            max(0, x0 - 5) : min(profile.size, x1 + 6)
        ]
        boundary_level = min(
            float(profile[x0]),
            float(profile[x1]),
        )
        interior_level = float(np.percentile(interior, 40))
        scale = max(
            float(
                np.percentile(local, 90)
                - np.percentile(local, 10)
            ),
            8.0,
        )
        contrast_by_band.append(
            (boundary_level - interior_level) / scale
        )
    contrast = np.asarray(contrast_by_band, dtype=np.float64)
    stable_fraction = float(
        np.mean(contrast >= config.minimum_dark_basin_contrast)
    )
    median_contrast = float(np.median(contrast))
    verified = bool(
        stable_fraction >= config.minimum_dark_basin_band_fraction
        and median_contrast >= config.minimum_dark_basin_contrast
    )
    return {
        "left_candidate_id": left["candidate_id"],
        "right_candidate_id": right["candidate_id"],
        "normalized_contrast_by_band": contrast.tolist(),
        "median_normalized_contrast": median_contrast,
        "stable_band_fraction": stable_fraction,
        "verified": verified,
    }


def _evaluate_role_hypothesis(
    selection: dict[str, dict],
    evidence: dict,
    config: SeparatorPathConfig,
) -> dict:
    paths = [selection[role] for role in ROLE_ORDER]
    path_metrics = [
        {
            "role": role,
            "candidate_id": path["candidate_id"],
            "support_fraction": path["support_fraction"],
            "mean_step_px": path["mean_step_px"],
        }
        for role, path in zip(ROLE_ORDER, paths)
    ]
    pair_order = [
        _full_height_pair_order(left, right, config)
        for left, right in zip(paths, paths[1:])
    ]
    basin_evidence = [
        _dark_basin_evidence(left, right, evidence, config)
        for left, right in zip(paths, paths[1:])
    ]
    center_x = np.asarray(
        [path["x_at_reference_roi"] for path in paths],
        dtype=np.float64,
    )
    widths = np.asarray(
        [path["peak_width_px"] for path in paths],
        dtype=np.float64,
    )
    edge_gaps = (
        np.diff(center_x) - (widths[:-1] + widths[1:]) / 2.0
    )
    minimum_edge_gap = float(np.min(edge_gaps))
    maximum_edge_gap = float(np.max(edge_gaps))
    edge_gap_ratio = (
        maximum_edge_gap / minimum_edge_gap
        if minimum_edge_gap > 0
        else None
    )

    rejection_reasons = []
    crossing = any(
        pair["crossing_band_indices"] for pair in pair_order
    )
    if crossing:
        rejection_reasons.append("path_crossing_full_height")
    if any(
        not pair["stable_full_height_order"]
        for pair in pair_order
    ):
        rejection_reasons.append("path_order_not_stable_full_height")
    if any(
        metric["support_fraction"]
        < config.minimum_role_path_support_fraction
        for metric in path_metrics
    ):
        rejection_reasons.append(
            "role_path_insufficient_vertical_support"
        )
    if any(
        metric["mean_step_px"]
        > config.maximum_role_path_mean_step_px
        for metric in path_metrics
    ):
        rejection_reasons.append("role_path_geometry_unstable")
    if (
        minimum_edge_gap
        < config.minimum_dark_basin_edge_gap_px
    ):
        rejection_reasons.append("separator_envelopes_overlap")
    if (
        edge_gap_ratio is None
        or edge_gap_ratio
        > config.maximum_dark_basin_edge_gap_ratio
    ):
        rejection_reasons.append(
            "dark_basin_width_sequence_irregular"
        )
    if any(not basin["verified"] for basin in basin_evidence):
        rejection_reasons.append("dark_basin_sequence_not_verified")

    return {
        "hypothesis_id": "H01",
        "type": "clicked_basin_roles",
        "selection_candidate_ids": {
            role: selection[role]["candidate_id"]
            for role in ROLE_ORDER
        },
        "path_metrics": path_metrics,
        "full_height_pair_order": pair_order,
        "dark_basin_evidence": basin_evidence,
        "edge_gap_at_reference_px": edge_gaps.tolist(),
        "edge_gap_ratio": edge_gap_ratio,
        "rejection_reasons": rejection_reasons,
        "verified": not rejection_reasons,
        "crossing": crossing,
    }


def select_clicked_basin_paths(
    candidates: list[dict],
    reference_x_roi: float,
    reference_y_roi: float,
    evidence: dict,
    config: SeparatorPathConfig = DEFAULT_CONFIG,
) -> dict:
    """Return roles only when reference relation and basin structure are unique."""

    accepted = [path for path in candidates if path["accepted"]]
    for path in accepted:
        path["x_at_reference_roi"] = _path_x_at_y(
            path,
            reference_y_roi,
        )
    accepted.sort(
        key=lambda path: (
            path["x_at_reference_roi"],
            path["candidate_id"],
        )
    )
    if not accepted:
        return {
            "status": "unavailable",
            "unavailable_reason": "no_accepted_separator_paths",
            "relationship_status": "unavailable",
            "selection": {},
            "crossing": False,
            "competing_explanations": [],
            "role_hypotheses": [],
            "rejection_reasons": ["no_accepted_separator_paths"],
        }

    relationship_candidates = []
    for path in accepted:
        distance = abs(
            path["x_at_reference_roi"] - reference_x_roi
        )
        guard = _reference_guard(path, config)
        if distance <= guard:
            relation = "reference_on_separator"
        elif (
            distance
            <= guard
            + config.reference_relationship_uncertainty_px
        ):
            relation = "reference_relationship_uncertain"
        else:
            continue
        relationship_candidates.append(
            {
                "type": relation,
                "candidate_id": path["candidate_id"],
                "distance_px": distance,
                "separator_guard_px": guard,
            }
        )

    definite_separator = [
        hypothesis
        for hypothesis in relationship_candidates
        if hypothesis["type"] == "reference_on_separator"
    ]
    uncertain_relationship = [
        hypothesis
        for hypothesis in relationship_candidates
        if hypothesis["type"]
        == "reference_relationship_uncertain"
    ]
    left = [
        path
        for path in accepted
        if path["x_at_reference_roi"] < reference_x_roi
    ]
    right = [
        path
        for path in accepted
        if path["x_at_reference_roi"] > reference_x_roi
    ]
    role_hypotheses = []
    candidate_selection = {}
    if len(left) >= 2 and len(right) >= 2:
        candidate_selection = {
            "left_adjacent": left[-2],
            "left_clicked_boundary": left[-1],
            "right_clicked_boundary": right[0],
            "right_adjacent": right[1],
        }
        role_hypotheses.append(
            _evaluate_role_hypothesis(
                candidate_selection,
                evidence,
                config,
            )
        )

    competing_explanations = [
        *relationship_candidates,
        *[
            {
                "type": hypothesis["type"],
                "hypothesis_id": hypothesis["hypothesis_id"],
                "selection_candidate_ids": hypothesis[
                    "selection_candidate_ids"
                ],
                "verified": hypothesis["verified"],
                "rejection_reasons": hypothesis[
                    "rejection_reasons"
                ],
            }
            for hypothesis in role_hypotheses
        ],
    ]
    crossing = any(
        hypothesis["crossing"] for hypothesis in role_hypotheses
    )
    role_rejection_reasons = sorted(
        {
            reason
            for hypothesis in role_hypotheses
            for reason in hypothesis["rejection_reasons"]
        }
    )

    if crossing:
        relationship_rejections = [
            hypothesis["type"]
            for hypothesis in relationship_candidates
        ]
        return {
            "status": "unavailable",
            "unavailable_reason": "path_order_conflict",
            "relationship_status": "conflicted",
            "selection": {},
            "crossing": True,
            "competing_explanations": competing_explanations,
            "role_hypotheses": role_hypotheses,
            "rejection_reasons": sorted(
                set(
                    role_rejection_reasons
                    + relationship_rejections
                )
            ),
        }
    if definite_separator:
        return {
            "status": "unavailable",
            "unavailable_reason": "reference_on_separator",
            "relationship_status": "reference_on_separator",
            "selection": {},
            "crossing": crossing,
            "competing_explanations": competing_explanations,
            "role_hypotheses": role_hypotheses,
            "rejection_reasons": sorted(
                set(
                    ["reference_on_separator"]
                    + role_rejection_reasons
                )
            ),
        }
    if uncertain_relationship:
        return {
            "status": "unavailable",
            "unavailable_reason": (
                "ambiguous_reference_relationship"
            ),
            "relationship_status": "not_unique",
            "selection": {},
            "crossing": crossing,
            "competing_explanations": competing_explanations,
            "role_hypotheses": role_hypotheses,
            "rejection_reasons": sorted(
                set(
                    ["ambiguous_reference_relationship"]
                    + role_rejection_reasons
                )
            ),
        }
    if not role_hypotheses:
        return {
            "status": "unavailable",
            "unavailable_reason": "insufficient_boundary_paths",
            "relationship_status": "basin_but_roles_incomplete",
            "selection": {},
            "crossing": False,
            "competing_explanations": competing_explanations,
            "role_hypotheses": [],
            "rejection_reasons": ["insufficient_boundary_paths"],
        }

    verified = [
        hypothesis
        for hypothesis in role_hypotheses
        if hypothesis["verified"]
    ]
    if len(verified) > 1:
        return {
            "status": "unavailable",
            "unavailable_reason": "ambiguous_role_hypotheses",
            "relationship_status": "basin_not_unique",
            "selection": {},
            "crossing": crossing,
            "competing_explanations": competing_explanations,
            "role_hypotheses": role_hypotheses,
            "rejection_reasons": ["ambiguous_role_hypotheses"],
        }
    if not verified:
        all_rejections = role_rejection_reasons
        unavailable_reason = (
            "path_order_conflict"
            if any(
                reason
                in {
                    "path_crossing_full_height",
                    "path_order_not_stable_full_height",
                }
                for reason in all_rejections
            )
            else "basin_structure_not_verified"
        )
        return {
            "status": "unavailable",
            "unavailable_reason": unavailable_reason,
            "relationship_status": "basin_structure_rejected",
            "selection": {},
            "crossing": crossing,
            "competing_explanations": competing_explanations,
            "role_hypotheses": role_hypotheses,
            "rejection_reasons": all_rejections,
        }

    return {
        "status": "available",
        "unavailable_reason": None,
        "relationship_status": "unique_verified_basin",
        "selection": candidate_selection,
        "crossing": False,
        "competing_explanations": competing_explanations,
        "role_hypotheses": role_hypotheses,
        "rejection_reasons": [],
    }


def detect_separator_paths(
    image_gray: np.ndarray,
    reference_global: dict,
    roi_bounds_global: dict,
    direction: str = "vertical",
    config: SeparatorPathConfig = DEFAULT_CONFIG,
) -> dict:
    """Infer local separator paths without consulting any human paths."""

    roi_gray = extract_raw_roi(image_gray, roi_bounds_global)
    directional = _directional_roi(roi_gray, direction)
    reference_x_roi, reference_y_roi = _directional_reference(
        reference_global,
        roi_bounds_global,
        direction,
    )
    evidence = build_raw_gray_response(directional, config)
    candidates = trace_separator_candidates(evidence, config)
    v1_arbitration = _select_clicked_basin_paths_v1(
        candidates,
        reference_x_roi,
        reference_y_roi,
        config,
    )
    arbitration = select_clicked_basin_paths(
        candidates,
        reference_x_roi,
        reference_y_roi,
        evidence,
        config,
    )
    return {
        "algorithm_revision": ALGORITHM_REVISION,
        "configuration_checksum": configuration_checksum(config),
        "direction": direction,
        "reference_x_roi": reference_x_roi,
        "reference_y_roi": reference_y_roi,
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
            "band_centers_y_roi": evidence[
                "band_centers_y"
            ].tolist(),
            "aggregate_response": evidence[
                "aggregate_response"
            ].tolist(),
            "candidate_seed_count": len(
                _seed_peaks(evidence["aggregate_response"], config)
            ),
            "accepted_candidate_count": sum(
                path["accepted"] for path in candidates
            ),
        },
    }


def _interpolated_gt_x(separator: dict, y_values: np.ndarray) -> np.ndarray:
    points = separator["control_points"]
    point_y = np.asarray([point["y_roi"] for point in points], dtype=float)
    point_x = np.asarray([point["x_roi"] for point in points], dtype=float)
    return np.interp(y_values, point_y, point_x)


def _path_error_against_gt(
    predicted: dict,
    separator: dict,
    evidence: dict | None = None,
    config: SeparatorPathConfig = DEFAULT_CONFIG,
) -> dict | None:
    points = separator.get("control_points", [])
    if (
        separator.get("visibility") not in NUMERIC_VISIBILITY
        or len(points) < 2
    ):
        return None
    y_values = np.asarray(
        predicted["band_centers_y_roi"],
        dtype=np.float64,
    )
    y0 = points[0]["y_roi"]
    y1 = points[-1]["y_roi"]
    overlap = (y_values >= y0) & (y_values <= y1)
    if not np.any(overlap):
        return None
    predicted_x = np.asarray(
        predicted["x_by_band_roi"],
        dtype=np.float64,
    )[overlap]
    gt_x = _interpolated_gt_x(separator, y_values[overlap])
    errors = np.abs(predicted_x - gt_x)
    same_bright_ridge = False
    if evidence is not None:
        overlap_indices = np.flatnonzero(overlap)
        agreements = [
            _profile_positions_share_bright_ridge(
                evidence["profiles"][int(band_index)],
                int(round(predicted_value)),
                int(round(gt_value)),
                config,
            )
            for band_index, predicted_value, gt_value in zip(
                overlap_indices,
                predicted_x,
                gt_x,
            )
        ]
        same_bright_ridge = bool(
            agreements
            and float(np.mean(agreements))
            >= config.same_ridge_minimum_band_fraction
        )
    return {
        "mean_error_px": float(np.mean(errors)),
        "median_error_px": float(np.median(errors)),
        "max_error_px": float(np.max(errors)),
        "evaluated_band_count": int(errors.size),
        "same_bright_ridge": same_bright_ridge,
    }


def _physical_path_match_tolerance(
    predicted: dict,
    config: SeparatorPathConfig,
) -> float:
    """Allow a human point anywhere on the same finite-width white ridge."""

    return max(
        config.match_tolerance_px,
        min(
            config.maximum_width_aware_match_tolerance_px,
            float(predicted["peak_width_px"]),
        ),
    )


def _gt_reference_ambiguity(
    annotation: dict,
    roi_gray: np.ndarray,
    config: SeparatorPathConfig,
) -> dict:
    reference = annotation["reference_global"]
    bounds = annotation["roi_bounds_global"]
    reference_x = reference["x"] - bounds["x0"]
    reference_y = reference["y"] - bounds["y0"]
    row = roi_gray[reference_y].astype(np.float64)
    local_x0 = max(0, reference_x - 30)
    local_x1 = min(row.size, reference_x + 31)
    local = row[local_x0:local_x1]
    percentile_rank = float(
        np.mean(local <= float(row[reference_x]))
    )
    separators = {
        separator["role"]: separator
        for separator in annotation["separators"]
    }
    boundary_distances = []
    for role in (
        "left_clicked_boundary",
        "right_clicked_boundary",
    ):
        separator = separators[role]
        if separator["visibility"] not in NUMERIC_VISIBILITY:
            continue
        gt_x = float(
            _interpolated_gt_x(
                separator,
                np.asarray([reference_y], dtype=np.float64),
            )[0]
        )
        boundary_distances.append(abs(gt_x - reference_x))
    near_boundary = bool(
        boundary_distances
        and min(boundary_distances) <= config.reference_guard_px
    )
    bright_reference = bool(
        percentile_rank >= config.gt_reference_percentile_threshold
    )
    return {
        "is_ambiguity_case": near_boundary or bright_reference,
        "near_annotated_boundary": near_boundary,
        "bright_reference": bright_reference,
        "reference_gray": int(row[reference_x]),
        "reference_local_percentile_rank": percentile_rank,
        "minimum_boundary_distance_px": (
            min(boundary_distances) if boundary_distances else None
        ),
    }


def evaluate_one_annotation(
    annotation: dict,
    image_gray: np.ndarray,
    config: SeparatorPathConfig = DEFAULT_CONFIG,
) -> dict:
    """Run inference first, then compare its paths with human development GT."""

    detection = detect_separator_paths(
        image_gray,
        annotation["reference_global"],
        annotation["roi_bounds_global"],
        annotation["direction"],
        config,
    )
    roi_gray = extract_raw_roi(
        image_gray,
        annotation["roi_bounds_global"],
    )
    gt_ambiguity = _gt_reference_ambiguity(
        annotation,
        roi_gray,
        config,
    )
    evaluation_evidence = build_raw_gray_response(
        _directional_roi(roi_gray, annotation["direction"]),
        config,
    )
    separators = {
        separator["role"]: separator
        for separator in annotation["separators"]
    }
    accepted = [
        path for path in detection["candidates"] if path["accepted"]
    ]
    all_visibilities = {
        separator["visibility"]
        for separator in annotation["separators"]
    }
    gt_unavailable = all_visibilities == {"unavailable"}
    gt_all_ambiguous = all_visibilities == {"ambiguous"}
    per_role = {}
    selected_false_splits = []
    all_numeric_gt = [
        separator
        for separator in annotation["separators"]
        if separator["visibility"] in NUMERIC_VISIBILITY
    ]
    for role in ROLE_ORDER:
        separator = separators[role]
        candidate_errors = [
            (
                candidate,
                _path_error_against_gt(
                    candidate,
                    separator,
                    evaluation_evidence,
                    config,
                ),
            )
            for candidate in accepted
        ]
        candidate_errors = [
            (candidate, error)
            for candidate, error in candidate_errors
            if error is not None
        ]
        candidate_errors.sort(
            key=lambda item: (
                item[1]["mean_error_px"],
                -item[0]["path_score"],
                item[0]["candidate_id"],
            )
        )
        best_candidate, best_error = (
            candidate_errors[0]
            if candidate_errors
            else (None, None)
        )
        selected = detection["selection"].get(role)
        selected_error = (
            _path_error_against_gt(
                selected,
                separator,
                evaluation_evidence,
                config,
            )
            if selected is not None
            else None
        )
        best_tolerance = (
            _physical_path_match_tolerance(best_candidate, config)
            if best_candidate is not None
            else None
        )
        selected_tolerance = (
            _physical_path_match_tolerance(selected, config)
            if selected is not None
            else None
        )
        per_role[role] = {
            "visibility": separator["visibility"],
            "confidence": separator["confidence"],
            "numeric_gt": separator["visibility"] in NUMERIC_VISIBILITY,
            "candidate_recalled": bool(
                best_error is not None
                and best_tolerance is not None
                and (
                    best_error["mean_error_px"] <= best_tolerance
                    or best_error["same_bright_ridge"]
                )
            ),
            "best_candidate_id": (
                best_candidate["candidate_id"]
                if best_candidate is not None
                else None
            ),
            "best_candidate_error": best_error,
            "best_candidate_match_tolerance_px": best_tolerance,
            "selected_candidate_id": (
                selected["candidate_id"] if selected is not None else None
            ),
            "selected_error": selected_error,
            "selected_match_tolerance_px": selected_tolerance,
            "selected_recalled": bool(
                selected_error is not None
                and selected_tolerance is not None
                and (
                    selected_error["mean_error_px"] <= selected_tolerance
                    or selected_error["same_bright_ridge"]
                )
            ),
        }

        if selected is not None and all_numeric_gt:
            errors_to_any_gt = [
                error
                for error in (
                    _path_error_against_gt(
                        selected,
                        target,
                        evaluation_evidence,
                        config,
                    )
                    for target in all_numeric_gt
                )
                if error is not None
            ]
            same_ridge_as_any_gt = any(
                error["same_bright_ridge"]
                for error in errors_to_any_gt
            )
            minimum_error = min(
                (error["mean_error_px"] for error in errors_to_any_gt),
                default=float("inf"),
            )
            false_split_tolerance = _physical_path_match_tolerance(
                selected,
                config,
            )
            if (
                minimum_error > false_split_tolerance
                and not same_ridge_as_any_gt
            ):
                selected_false_splits.append(
                    {
                        "role": role,
                        "candidate_id": selected["candidate_id"],
                        "minimum_gt_error_px": (
                            None
                            if not np.isfinite(minimum_error)
                            else minimum_error
                        ),
                        "match_tolerance_px": false_split_tolerance,
                    }
                )

    failure_reasons = []
    if gt_unavailable and detection["status"] == "available":
        selected_false_splits.extend(
            {
                "role": role,
                "candidate_id": path["candidate_id"],
                "minimum_gt_error_px": None,
                "match_tolerance_px": None,
                "reason": "ground_truth_unavailable",
            }
            for role, path in detection["selection"].items()
        )
    if detection["status"] != "available":
        failure_reasons.append(detection["unavailable_reason"])
    if selected_false_splits:
        failure_reasons.append("selected_false_split")
    if detection["crossing"]:
        failure_reasons.append("predicted_path_crossing")
    if gt_unavailable and detection["status"] == "available":
        failure_reasons.append("unavailable_gt_formally_available")
    if gt_all_ambiguous and detection["status"] == "available":
        failure_reasons.append("ambiguous_gt_formally_available")
    for role, role_result in per_role.items():
        if (
            role_result["numeric_gt"]
            and not role_result["selected_recalled"]
            and not gt_ambiguity["is_ambiguity_case"]
        ):
            failure_reasons.append(f"unmatched_{role}")

    v1_baseline = detection["v1_arbitration_baseline"]
    current_selection_ids = {
        role: path["candidate_id"]
        for role, path in detection["selection"].items()
    }
    if (
        v1_baseline["status"] == detection["status"]
        and v1_baseline["selection_candidate_ids"]
        == current_selection_ids
        and v1_baseline["unavailable_reason"]
        == detection["unavailable_reason"]
    ):
        transition = "unchanged"
    elif (
        v1_baseline["status"] == "available"
        and detection["status"] == "unavailable"
    ):
        transition = "v1_available_to_v1_1_safe_unavailable"
    elif (
        v1_baseline["status"] == "unavailable"
        and detection["status"] == "available"
    ):
        transition = "v1_unavailable_to_v1_1_available"
    else:
        transition = "decision_changed"

    return {
        "sample_id": annotation["sample_id"],
        "image_name": annotation["image_name"],
        "reference_global": annotation["reference_global"],
        "status": detection["status"],
        "unavailable_reason": detection["unavailable_reason"],
        "gt_reference_ambiguity": gt_ambiguity,
        "gt_all_ambiguous": gt_all_ambiguous,
        "gt_unavailable": gt_unavailable,
        "candidate_count": len(detection["candidates"]),
        "accepted_candidate_count": sum(
            candidate["accepted"]
            for candidate in detection["candidates"]
        ),
        "selection": {
            role: {
                key: value
                for key, value in path.items()
                if key
                in {
                    "candidate_id",
                    "seed_x_roi",
                    "band_centers_y_roi",
                    "x_by_band_roi",
                    "support_fraction",
                    "path_score",
                    "peak_width_px",
                    "x_at_reference_roi",
                }
            }
            for role, path in detection["selection"].items()
        },
        "per_role": per_role,
        "selected_false_splits": selected_false_splits,
        "crossing": detection["crossing"],
        "failure_reasons": sorted(set(failure_reasons)),
        "v1_to_v1_1": {
            "transition": transition,
            "v1_status": v1_baseline["status"],
            "v1_unavailable_reason": v1_baseline[
                "unavailable_reason"
            ],
            "v1_selection_candidate_ids": v1_baseline[
                "selection_candidate_ids"
            ],
            "v1_1_status": detection["status"],
            "v1_1_unavailable_reason": detection[
                "unavailable_reason"
            ],
            "v1_1_selection_candidate_ids": current_selection_ids,
        },
        "detection_debug": {
            "arbitration": detection["arbitration_debug"],
            "v1_arbitration_baseline": v1_baseline,
            "raw_gray_evidence": detection["raw_gray_evidence"],
            "candidates": detection["candidates"],
        },
    }


def _draw_polyline(
    canvas: np.ndarray,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if len(points) < 2:
        return
    cv2.polylines(
        canvas,
        [np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)],
        False,
        color,
        thickness,
        cv2.LINE_AA,
    )


def save_comparison_overlay(
    annotation: dict,
    image_gray: np.ndarray,
    evaluation: dict,
    output_path: Path,
) -> None:
    """Save raw ROI with human paths and inferred paths side by side."""

    roi = extract_raw_roi(
        image_gray,
        annotation["roi_bounds_global"],
    )
    raw = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    predicted = raw.copy()
    human = raw.copy()

    for candidate in evaluation["detection_debug"]["candidates"]:
        if not candidate["accepted"]:
            continue
        points = [
            (int(round(x)), int(round(y)))
            for x, y in zip(
                candidate["x_by_band_roi"],
                candidate["band_centers_y_roi"],
            )
        ]
        _draw_polyline(predicted, points, (60, 150, 60), 1)
    for role, path in evaluation["selection"].items():
        points = [
            (int(round(x)), int(round(y)))
            for x, y in zip(
                path["x_by_band_roi"],
                path["band_centers_y_roi"],
            )
        ]
        _draw_polyline(predicted, points, ROLE_COLORS_BGR[role], 2)

    for separator in annotation["separators"]:
        points = [
            (point["x_roi"], point["y_roi"])
            for point in separator["control_points"]
        ]
        _draw_polyline(
            human,
            points,
            ROLE_COLORS_BGR[separator["role"]],
            2,
        )
        for point in points:
            cv2.circle(
                human,
                point,
                2,
                ROLE_COLORS_BGR[separator["role"]],
                -1,
                cv2.LINE_AA,
            )

    bounds = annotation["roi_bounds_global"]
    reference = annotation["reference_global"]
    reference_point = (
        reference["x"] - bounds["x0"],
        reference["y"] - bounds["y0"],
    )
    for canvas in (predicted, human):
        cv2.drawMarker(
            canvas,
            reference_point,
            (0, 0, 255),
            cv2.MARKER_CROSS,
            14,
            2,
            cv2.LINE_AA,
        )

    title_height = 28
    gap = 8
    comparison = np.full(
        (
            roi.shape[0] + title_height,
            roi.shape[1] * 2 + gap,
            3,
        ),
        245,
        dtype=np.uint8,
    )
    comparison[title_height:, : roi.shape[1]] = predicted
    comparison[
        title_height:,
        roi.shape[1] + gap :,
    ] = human
    cv2.putText(
        comparison,
        "Raw-gray prediction",
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        comparison,
        "Human development GT",
        (roi.shape[1] + gap + 8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", comparison)
    if not success:
        raise ValueError(f"could not encode {output_path}")
    output_path.write_bytes(encoded.tobytes())


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


def load_development_document(
    path: Path = DEVELOPMENT_PATH,
) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("dataset_version") != DATASET_VERSION:
        raise ValueError("unexpected separator dataset version")
    if document.get("split") != "development":
        raise ValueError("prototype accepts only the development split")
    annotations = document.get("annotations")
    if not isinstance(annotations, dict) or not annotations:
        raise ValueError("development annotations are missing")
    if any(
        not sample_id.startswith("D") or annotation is None
        for sample_id, annotation in annotations.items()
    ):
        raise ValueError("development annotations are incomplete or mixed")
    return document


def _candidate_output_checksum(evaluations: list[dict]) -> str:
    payload = [
        {
            "sample_id": item["sample_id"],
            "candidates": item["detection_debug"]["candidates"],
        }
        for item in evaluations
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _summarize_evaluations(evaluations: list[dict]) -> dict:
    scorable = [
        item
        for item in evaluations
        if not item["gt_reference_ambiguity"]["is_ambiguity_case"]
        and not item["gt_all_ambiguous"]
        and not item["gt_unavailable"]
    ]
    numeric_roles = [
        role_result
        for item in scorable
        for role_result in item["per_role"].values()
        if role_result["numeric_gt"]
    ]
    candidate_recalled = sum(
        role["candidate_recalled"] for role in numeric_roles
    )
    selected_recalled = sum(
        role["selected_recalled"] for role in numeric_roles
    )
    selected_errors = [
        role["selected_error"]["mean_error_px"]
        for role in numeric_roles
        if role["selected_recalled"]
    ]
    failure_counts = {}
    for item in evaluations:
        for reason in item["failure_reasons"]:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
    ambiguity_cases = [
        item["sample_id"]
        for item in evaluations
        if item["gt_reference_ambiguity"]["is_ambiguity_case"]
    ]
    candidate_checksum = _candidate_output_checksum(evaluations)
    transition_counts = {}
    for item in evaluations:
        transition = item["v1_to_v1_1"]["transition"]
        transition_counts[transition] = (
            transition_counts.get(transition, 0) + 1
        )
    return {
        "sample_count": len(evaluations),
        "scorable_sample_count": len(scorable),
        "numeric_path_count": len(numeric_roles),
        "candidate_path_recall_count": candidate_recalled,
        "candidate_path_recall": (
            candidate_recalled / len(numeric_roles)
            if numeric_roles
            else None
        ),
        "selected_path_recall_count": selected_recalled,
        "path_recall": (
            selected_recalled / len(numeric_roles)
            if numeric_roles
            else None
        ),
        "matched_path_mean_error_median_px": (
            float(np.median(selected_errors))
            if selected_errors
            else None
        ),
        "matched_path_mean_error_max_px": (
            float(np.max(selected_errors))
            if selected_errors
            else None
        ),
        "false_split_count": sum(
            len(item["selected_false_splits"])
            for item in evaluations
        ),
        "crossing_count": sum(item["crossing"] for item in evaluations),
        "crossing_formally_available_count": sum(
            item["crossing"] and item["status"] == "available"
            for item in evaluations
        ),
        "available_sample_count": sum(
            item["status"] == "available" for item in evaluations
        ),
        "unavailable_sample_count": sum(
            item["status"] != "available" for item in evaluations
        ),
        "reference_ambiguity_cases": ambiguity_cases,
        "reference_ambiguity_safe_unavailable": sum(
            item["gt_reference_ambiguity"]["is_ambiguity_case"]
            and item["status"] == "unavailable"
            for item in evaluations
        ),
        "ambiguous_gt_formally_available": [
            item["sample_id"]
            for item in evaluations
            if item["gt_all_ambiguous"] and item["status"] == "available"
        ],
        "unavailable_gt_formally_available": [
            item["sample_id"]
            for item in evaluations
            if item["gt_unavailable"] and item["status"] == "available"
        ],
        "candidate_output_checksum": candidate_checksum,
        "candidate_generation_unchanged": (
            candidate_checksum
            == FROZEN_DEVELOPMENT_CANDIDATE_OUTPUT_CHECKSUM
        ),
        "v1_to_v1_1_transition_counts": dict(
            sorted(transition_counts.items())
        ),
        "v1_to_v1_1_changed_samples": [
            {
                "sample_id": item["sample_id"],
                **item["v1_to_v1_1"],
            }
            for item in evaluations
            if item["v1_to_v1_1"]["transition"] != "unchanged"
        ],
        "failure_reason_counts": dict(sorted(failure_counts.items())),
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_sample_summary(
    path: Path,
    evaluations: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = (
        "sample_id",
        "image_name",
        "status",
        "gt_reference_ambiguity",
        "numeric_path_count",
        "candidate_path_recalled",
        "selected_path_recalled",
        "false_split_count",
        "crossing",
        "relationship_status",
        "arbitration_rejection_reasons",
        "v1_to_v1_1_transition",
        "failure_reasons",
    )
    with temporary.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in evaluations:
            numeric_roles = [
                role
                for role in item["per_role"].values()
                if role["numeric_gt"]
            ]
            writer.writerow(
                {
                    "sample_id": item["sample_id"],
                    "image_name": item["image_name"],
                    "status": item["status"],
                    "gt_reference_ambiguity": item[
                        "gt_reference_ambiguity"
                    ]["is_ambiguity_case"],
                    "numeric_path_count": len(numeric_roles),
                    "candidate_path_recalled": sum(
                        role["candidate_recalled"]
                        for role in numeric_roles
                    ),
                    "selected_path_recalled": sum(
                        role["selected_recalled"]
                        for role in numeric_roles
                    ),
                    "false_split_count": len(
                        item["selected_false_splits"]
                    ),
                    "crossing": item["crossing"],
                    "relationship_status": item[
                        "detection_debug"
                    ]["arbitration"].get("relationship_status"),
                    "arbitration_rejection_reasons": "|".join(
                        item["detection_debug"]["arbitration"].get(
                            "rejection_reasons",
                            [],
                        )
                    ),
                    "v1_to_v1_1_transition": item[
                        "v1_to_v1_1"
                    ]["transition"],
                    "failure_reasons": "|".join(
                        item["failure_reasons"]
                    ),
                }
            )
    temporary.replace(path)


def evaluate_development(
    sample_ids: list[str] | None = None,
    output_dir: Path = OUTPUT_DIR,
    save_overlays: bool = True,
    config: SeparatorPathConfig = DEFAULT_CONFIG,
) -> dict:
    document = load_development_document()
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
    evaluations = []
    for sample_id in requested:
        annotation = annotations[sample_id]
        image_name = annotation["image_name"]
        if image_name not in image_cache:
            image_cache[image_name] = load_grayscale_image(
                IMAGES_DIR / image_name
            )
        evaluation = evaluate_one_annotation(
            annotation,
            image_cache[image_name],
            config,
        )
        evaluations.append(evaluation)
        if save_overlays:
            save_comparison_overlay(
                annotation,
                image_cache[image_name],
                evaluation,
                output_dir / "overlays" / f"{sample_id}.png",
            )

    report = {
        "algorithm_revision": ALGORITHM_REVISION,
        "configuration": canonical_configuration(config),
        "configuration_checksum": configuration_checksum(config),
        "access_policy": "development_only",
        "source_document": str(DEVELOPMENT_PATH.relative_to(PROJECT_ROOT)),
        "evaluated_sample_ids": requested,
        **_git_provenance(),
        "metrics": _summarize_evaluations(evaluations),
        "samples": evaluations,
    }
    report_name = (
        "development_report.json"
        if sample_ids is None
        else "quick_report.json"
    )
    _atomic_write_json(output_dir / report_name, report)
    _write_sample_summary(
        output_dir / "sample_summary.csv",
        evaluations,
    )
    return report


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline raw-gray separator path prototype",
    )
    parser.add_argument(
        "--sample",
        action="append",
        dest="sample_ids",
        help="development sample ID; repeat to select several",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    report = evaluate_development(
        sample_ids=arguments.sample_ids,
        output_dir=arguments.output_dir,
        save_overlays=not arguments.no_overlays,
    )
    print(json.dumps(report["metrics"], indent=2))
    print(f"report: {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
