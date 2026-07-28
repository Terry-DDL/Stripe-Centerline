"""Stage-1 raw local-pitch prototype.

The estimator in this file is deliberately independent of the production
detector.  It consumes only source grayscale pixels, one fixed ROI, a reference
point, and the stripe direction.  It does not import or call tracks,
thresholding, morphology, basins, the cached pitch map, or current results.

The command-line evaluation is intentionally locked to the development split:

    .venv/bin/python tools/raw_local_pitch_prototype.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_VERSION = "raw_pitch_gt_v1"
DEVELOPMENT_PATH = (
    PROJECT_ROOT / "tests" / "data" / DATASET_VERSION / "development.json"
)
P026_VALIDATION_PATH = (
    PROJECT_ROOT
    / "tests"
    / "data"
    / "raw_pitch_gt_v2"
    / "p026_validation.json"
)
IMAGES_DIR = PROJECT_ROOT / "images"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "raw_pitch_gt_v2"
    / "raw_local_pitch_stage1_v2"
)
REPORT_PATH = OUTPUT_DIR / "development_validation_report.json"
DEBUG_DIR = OUTPUT_DIR / "debug"
ALGORITHM_REVISION = "raw_local_pitch_stage1_v2"


@dataclass(frozen=True)
class RawLocalPitchConfig:
    """All tunable stage-1 parameters.

    These values operate on one-dimensional raw-gray profiles.  They are not
    image thresholds and do not change the source ROI pixels.
    """

    subwindow_count: int = 5
    profile_percentile: float = 70.0
    min_pitch_px: int = 8
    max_pitch_px: int = 90
    autocorrelation_floor: float = 0.18
    earliest_peak_relative_strength: float = 0.50
    minimum_spectral_support: float = 0.06
    band_pitch_tolerance_ratio: float = 0.14
    medium_relative_mad_limit: float = 0.14
    high_relative_mad_limit: float = 0.07
    medium_min_agreeing_bands: int = 3
    high_min_agreeing_bands: int = 4
    medium_autocorrelation_floor: float = 0.28
    high_autocorrelation_floor: float = 0.60
    harmonic_ambiguity_ratio: float = 0.78
    harmonic_resolution_min_support: float = 0.25
    harmonic_dominant_power_ratio: float = 1.50
    harmonic_integer_ratio_tolerance: float = 0.12
    harmonic_conflict_min_bands: int = 3
    minimum_profile_contrast: float = 6.0
    fft_padding_factor: int = 8


DEFAULT_CONFIG = RawLocalPitchConfig()


def canonical_configuration(config: RawLocalPitchConfig) -> dict:
    """Return the exact checksum payload for a frozen configuration."""

    return {
        "algorithm_revision": ALGORITHM_REVISION,
        "parameters": asdict(config),
    }


def configuration_checksum(config: RawLocalPitchConfig) -> str:
    """Return a stable SHA-256 for the algorithm revision and parameters."""

    payload = json.dumps(
        canonical_configuration(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_grayscale_image(path: Path) -> np.ndarray:
    """Decode source grayscale without applying any preprocessing."""

    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image_gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image_gray is None or image_gray.size == 0:
        raise ValueError(f"OpenCV could not decode {path}")
    return image_gray


def extract_raw_roi(
    image_gray: np.ndarray,
    roi_bounds_global: dict,
) -> np.ndarray:
    """Extract one half-open source ROI and validate its bounds."""

    if image_gray.ndim != 2:
        raise ValueError("raw local pitch requires one grayscale image")
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
    """Make the pitch axis horizontal while preserving raw gray values."""

    if direction == "vertical":
        return roi_gray
    if direction == "horizontal":
        return roi_gray.T
    raise ValueError("direction must be vertical or horizontal")


def _subwindow_bounds(
    height: int,
    count: int,
) -> list[tuple[int, int]]:
    """Return non-overlapping vertical subwindows covering the raw ROI."""

    if count < 3:
        raise ValueError("at least three subwindows are required")
    edges = np.linspace(0, height, count + 1).round().astype(int)
    return [
        (int(edges[index]), int(edges[index + 1]))
        for index in range(count)
        if edges[index + 1] > edges[index]
    ]


def raw_gray_profile(
    subwindow_gray: np.ndarray,
    percentile: float,
) -> np.ndarray:
    """Collapse raw grayscale rows into one pitch-axis profile."""

    return np.percentile(
        subwindow_gray.astype(np.float64),
        percentile,
        axis=0,
    )


def _quadratic_detrend(profile: np.ndarray) -> np.ndarray:
    """Remove only slow illumination tilt from a one-dimensional profile."""

    x = np.linspace(-1.0, 1.0, profile.size)
    coefficients = np.polyfit(x, profile, deg=2)
    baseline = np.polyval(coefficients, x)
    detrended = profile - baseline
    detrended -= float(np.mean(detrended))
    return detrended


def normalized_autocorrelation(
    signal: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    """Calculate overlap-normalized one-dimensional autocorrelation."""

    correlation = np.full(max_lag + 1, np.nan, dtype=np.float64)
    correlation[0] = 1.0
    for lag in range(1, max_lag + 1):
        left = signal[:-lag]
        right = signal[lag:]
        denominator = float(
            np.sqrt(np.dot(left, left) * np.dot(right, right))
        )
        if denominator > 1e-12:
            correlation[lag] = float(np.dot(left, right) / denominator)
    return correlation


def _spectral_support(
    signal: np.ndarray,
    min_pitch: int,
    max_pitch: int,
    padding_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return periods and normalized raw-profile periodogram support."""

    windowed = signal * np.hanning(signal.size)
    target_length = max(
        signal.size,
        int(2 ** np.ceil(np.log2(signal.size * padding_factor))),
    )
    spectrum = np.fft.rfft(windowed, n=target_length)
    power = np.square(np.abs(spectrum))
    frequencies = np.fft.rfftfreq(target_length)
    valid = (
        (frequencies >= 1.0 / max_pitch)
        & (frequencies <= 1.0 / min_pitch)
    )
    periods = 1.0 / frequencies[valid]
    target_power = power[valid]
    maximum = float(np.max(target_power)) if target_power.size else 0.0
    if maximum > 0:
        target_power = target_power / maximum
    return periods, target_power


def _support_at_period(
    periods: np.ndarray,
    support: np.ndarray,
    period: float,
) -> float:
    if not len(periods) or period <= 0:
        return 0.0
    index = int(np.argmin(np.abs(periods - period)))
    return float(support[index])


def _local_maxima(
    values: np.ndarray,
    start: int,
    stop: int,
) -> list[int]:
    peaks = []
    for index in range(max(1, start), min(stop, len(values) - 1)):
        value = values[index]
        if (
            np.isfinite(value)
            and value >= values[index - 1]
            and value > values[index + 1]
        ):
            peaks.append(index)
    return peaks


def _refine_peak(values: np.ndarray, index: int) -> float:
    """Refine one integer autocorrelation peak with a parabola."""

    if index <= 0 or index + 1 >= len(values):
        return float(index)
    left = float(values[index - 1])
    center = float(values[index])
    right = float(values[index + 1])
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return float(index)
    offset = 0.5 * (left - right) / denominator
    return float(index + np.clip(offset, -0.5, 0.5))


def estimate_subwindow_pitch(
    raw_profile: np.ndarray,
    config: RawLocalPitchConfig,
) -> dict:
    """Estimate one subwindow period from its raw-gray profile."""

    detrended = _quadratic_detrend(raw_profile)
    contrast = float(
        np.percentile(raw_profile, 90)
        - np.percentile(raw_profile, 10)
    )
    signal_scale = float(np.std(detrended))
    if (
        contrast < config.minimum_profile_contrast
        or signal_scale < 1e-6
    ):
        return {
            "available": False,
            "reason": "insufficient_raw_profile_contrast",
            "raw_profile_contrast": contrast,
        }

    max_pitch = min(
        config.max_pitch_px,
        max(config.min_pitch_px, raw_profile.size // 3),
    )
    autocorrelation = normalized_autocorrelation(detrended, max_pitch)
    periods, spectral = _spectral_support(
        detrended,
        config.min_pitch_px,
        max_pitch,
        config.fft_padding_factor,
    )
    dominant_spectral_index = int(np.argmax(spectral))
    dominant_spectral_period = float(periods[dominant_spectral_index])
    dominant_spectral_support = float(spectral[dominant_spectral_index])
    peaks = _local_maxima(
        autocorrelation,
        config.min_pitch_px,
        max_pitch + 1,
    )
    if not peaks:
        return {
            "available": False,
            "reason": "no_autocorrelation_peak",
            "raw_profile_contrast": contrast,
        }
    maximum_correlation = max(float(autocorrelation[peak]) for peak in peaks)
    required_correlation = max(
        config.autocorrelation_floor,
        maximum_correlation * config.earliest_peak_relative_strength,
    )
    eligible = [
        peak
        for peak in peaks
        if (
            autocorrelation[peak] >= required_correlation
            and _support_at_period(periods, spectral, peak)
            >= config.minimum_spectral_support
        )
    ]
    if not eligible:
        return {
            "available": False,
            "reason": "no_joint_periodic_peak",
            "raw_profile_contrast": contrast,
            "maximum_autocorrelation": maximum_correlation,
            "dominant_spectral_period_px": dominant_spectral_period,
            "dominant_spectral_support": dominant_spectral_support,
        }

    chosen = min(eligible)
    refined_pitch = _refine_peak(autocorrelation, chosen)
    fundamental_support = _support_at_period(
        periods,
        spectral,
        refined_pitch,
    )
    half_support = (
        _support_at_period(periods, spectral, refined_pitch / 2.0)
        if refined_pitch / 2.0 >= config.min_pitch_px
        else 0.0
    )
    double_support = (
        _support_at_period(periods, spectral, refined_pitch * 2.0)
        if refined_pitch * 2.0 <= max_pitch
        else 0.0
    )
    return {
        "available": True,
        "pitch_px": refined_pitch,
        "integer_peak_px": chosen,
        "autocorrelation": float(autocorrelation[chosen]),
        "maximum_autocorrelation": maximum_correlation,
        "raw_profile_contrast": contrast,
        "spectral_support": fundamental_support,
        "half_period_spectral_support": half_support,
        "double_period_spectral_support": double_support,
        "dominant_spectral_period_px": dominant_spectral_period,
        "dominant_spectral_support": dominant_spectral_support,
        "profile": raw_profile,
        "detrended_profile": detrended,
        "autocorrelation_curve": autocorrelation,
    }


def _relative_mad(values: list[float], center: float) -> float:
    if not values or center <= 0:
        return float("inf")
    deviations = [abs(value - center) for value in values]
    return float(np.median(deviations) / center)


def _integer_harmonic_conflict(
    selected_period: float,
    selected_spectral_support: float,
    dominant_period: float,
    dominant_spectral_support: float,
    config: RawLocalPitchConfig,
) -> dict | None:
    """Describe an unresolved 2x/3x conflict with the dominant spectrum."""

    if selected_period <= 0 or dominant_period <= 0:
        return None
    larger = max(selected_period, dominant_period)
    smaller = min(selected_period, dominant_period)
    ratio = larger / smaller
    multiple = int(round(ratio))
    if multiple not in {2, 3}:
        return None
    relative_distance = abs(ratio - multiple) / multiple
    if relative_distance > config.harmonic_integer_ratio_tolerance:
        return None
    if (
        dominant_spectral_support
        < selected_spectral_support
        * config.harmonic_dominant_power_ratio
    ):
        return None
    return {
        "selected_period_px": selected_period,
        "dominant_spectral_period_px": dominant_period,
        "ratio": ratio,
        "nearest_multiple": multiple,
        "relative_distance_to_multiple": relative_distance,
        "selected_spectral_support": selected_spectral_support,
        "dominant_spectral_support": dominant_spectral_support,
        "direction": (
            "selected_is_multiple_of_dominant"
            if selected_period > dominant_period
            else "dominant_is_multiple_of_selected"
        ),
    }


def estimate_raw_local_pitch(
    image_gray: np.ndarray,
    reference_global: dict,
    roi_bounds_global: dict,
    direction: str = "vertical",
    config: RawLocalPitchConfig = DEFAULT_CONFIG,
) -> dict:
    """Estimate local pitch from raw grayscale subwindow profiles only."""

    roi_gray = extract_raw_roi(image_gray, roi_bounds_global)
    directional = _directional_roi(roi_gray, direction)
    reference_axis = (
        reference_global["x"] - roi_bounds_global["x0"]
        if direction == "vertical"
        else reference_global["y"] - roi_bounds_global["y0"]
    )
    if not 0 <= reference_axis < directional.shape[1]:
        raise ValueError("reference point is outside the ROI pitch axis")

    band_results = []
    for index, (y0, y1) in enumerate(
        _subwindow_bounds(
            directional.shape[0],
            config.subwindow_count,
        )
    ):
        profile = raw_gray_profile(
            directional[y0:y1],
            config.profile_percentile,
        )
        result = estimate_subwindow_pitch(profile, config)
        result.update(
            {
                "band_index": index,
                "row_bounds": {"y0": y0, "y1": y1},
            }
        )
        band_results.append(result)

    available = [result for result in band_results if result["available"]]
    if len(available) < config.medium_min_agreeing_bands:
        return {
            "algorithm_revision": ALGORITHM_REVISION,
            "configuration_checksum": configuration_checksum(config),
            "pitch_px": None,
            "diagnostic_pitch_px": None,
            "usable_pitch_px": None,
            "success_eligible": False,
            "confidence": "unavailable",
            "subwindow_consistency": {
                "available_bands": len(available),
                "total_bands": len(band_results),
                "agreeing_bands": 0,
                "relative_mad": None,
            },
            "harmonic_ambiguity": {
                "detected": False,
                "reason": "insufficient_available_bands",
            },
            "unavailable_reason": "insufficient_periodic_subwindows",
            "reference_axis_roi": reference_axis,
            "band_results": band_results,
        }

    resolved_half_period_bands = []
    dominant_harmonic_conflicts = []
    for result in available:
        result["aggregation_pitch_px"] = result["pitch_px"]
        result["aggregation_spectral_support"] = result[
            "spectral_support"
        ]
        result["aggregation_half_period_spectral_support"] = result[
            "half_period_spectral_support"
        ]
        result["aggregation_double_period_spectral_support"] = result[
            "double_period_spectral_support"
        ]
        half_support = result["half_period_spectral_support"]
        if (
            result["pitch_px"] / 2.0 >= config.min_pitch_px
            and half_support >= config.harmonic_resolution_min_support
            and half_support
            >= (
                result["spectral_support"]
                * config.harmonic_ambiguity_ratio
            )
        ):
            result["aggregation_pitch_px"] = result["pitch_px"] / 2.0
            result["aggregation_spectral_support"] = half_support
            result["aggregation_half_period_spectral_support"] = 0.0
            result["aggregation_double_period_spectral_support"] = result[
                "spectral_support"
            ]
            result["harmonic_resolution"] = "selected_half_period"
            resolved_half_period_bands.append(result["band_index"])
        else:
            result["harmonic_resolution"] = None
        conflict = _integer_harmonic_conflict(
            result["aggregation_pitch_px"],
            result["aggregation_spectral_support"],
            result["dominant_spectral_period_px"],
            result["dominant_spectral_support"],
            config,
        )
        result["dominant_harmonic_conflict"] = conflict
        if conflict is not None:
            dominant_harmonic_conflicts.append(
                {
                    "band_index": result["band_index"],
                    **conflict,
                }
            )

    initial_center = float(
        np.median(
            [result["aggregation_pitch_px"] for result in available]
        )
    )
    agreeing = [
        result
        for result in available
        if (
            abs(result["aggregation_pitch_px"] - initial_center)
            / initial_center
            <= config.band_pitch_tolerance_ratio
        )
    ]
    if len(agreeing) < config.medium_min_agreeing_bands:
        return {
            "algorithm_revision": ALGORITHM_REVISION,
            "configuration_checksum": configuration_checksum(config),
            "pitch_px": None,
            "diagnostic_pitch_px": None,
            "usable_pitch_px": None,
            "success_eligible": False,
            "confidence": "unavailable",
            "subwindow_consistency": {
                "available_bands": len(available),
                "total_bands": len(band_results),
                "agreeing_bands": len(agreeing),
                "relative_mad": None,
                "band_pitches_px": [
                    result["aggregation_pitch_px"]
                    for result in available
                ],
                "raw_band_pitches_px": [
                    result["pitch_px"] for result in available
                ],
            },
            "harmonic_ambiguity": {
                "detected": False,
                "reason": "subwindow_cluster_not_established",
                "resolved_half_period_bands": (
                    resolved_half_period_bands
                ),
                "dominant_harmonic_conflicts": (
                    dominant_harmonic_conflicts
                ),
            },
            "unavailable_reason": "inconsistent_subwindow_periods",
            "reference_axis_roi": reference_axis,
            "band_results": band_results,
        }

    pitches = [result["aggregation_pitch_px"] for result in agreeing]
    pitch = float(np.median(pitches))
    relative_mad = _relative_mad(pitches, pitch)
    median_autocorrelation = float(
        np.median([result["autocorrelation"] for result in agreeing])
    )
    median_fundamental = float(
        np.median(
            [
                result["aggregation_spectral_support"]
                for result in agreeing
            ]
        )
    )
    median_half = float(
        np.median(
            [
                result["aggregation_half_period_spectral_support"]
                for result in agreeing
            ]
        )
    )
    median_double = float(
        np.median(
            [
                result["aggregation_double_period_spectral_support"]
                for result in agreeing
            ]
        )
    )
    denominator = max(median_fundamental, 1e-12)
    half_ratio = median_half / denominator
    double_ratio = median_double / denominator
    harmonic_detected = (
        half_ratio >= config.harmonic_ambiguity_ratio
        or double_ratio >= config.harmonic_ambiguity_ratio
    )
    agreeing_band_indexes = {
        result["band_index"] for result in agreeing
    }
    agreeing_dominant_conflicts = [
        conflict
        for conflict in dominant_harmonic_conflicts
        if conflict["band_index"] in agreeing_band_indexes
    ]
    hard_harmonic_ambiguity = (
        harmonic_detected
        or len(agreeing_dominant_conflicts)
        >= config.harmonic_conflict_min_bands
    )
    consistency = {
        "available_bands": len(available),
        "total_bands": len(band_results),
        "agreeing_bands": len(agreeing),
        "relative_mad": relative_mad,
        "band_pitches_px": [
            result["aggregation_pitch_px"] for result in available
        ],
        "raw_band_pitches_px": [
            result["pitch_px"] for result in available
        ],
        "median_autocorrelation": median_autocorrelation,
    }
    harmonic_details = {
        "detected": hard_harmonic_ambiguity,
        "half_period_support_ratio": half_ratio,
        "double_period_support_ratio": double_ratio,
        "fundamental_spectral_support": median_fundamental,
        "resolved_half_period_bands": resolved_half_period_bands,
        "dominant_harmonic_conflicts": agreeing_dominant_conflicts,
        "reason": (
            "dominant_spectral_integer_multiple_conflict"
            if agreeing_dominant_conflicts
            else (
                "half_or_double_period_support_conflict"
                if harmonic_detected
                else None
            )
        ),
    }
    if hard_harmonic_ambiguity:
        return {
            "algorithm_revision": ALGORITHM_REVISION,
            "configuration_checksum": configuration_checksum(config),
            "pitch_px": None,
            "diagnostic_pitch_px": pitch,
            "usable_pitch_px": None,
            "success_eligible": False,
            "confidence": "unavailable",
            "subwindow_consistency": consistency,
            "harmonic_ambiguity": harmonic_details,
            "unavailable_reason": "harmonic_ambiguous",
            "reference_axis_roi": reference_axis,
            "band_results": band_results,
        }

    if (
        len(agreeing) >= config.high_min_agreeing_bands
        and relative_mad <= config.high_relative_mad_limit
        and median_autocorrelation
        >= config.high_autocorrelation_floor
        and not harmonic_detected
        and not resolved_half_period_bands
    ):
        confidence = "high"
    elif (
        len(agreeing) >= config.medium_min_agreeing_bands
        and relative_mad <= config.medium_relative_mad_limit
        and median_autocorrelation
        >= config.medium_autocorrelation_floor
    ):
        confidence = "medium"
    else:
        confidence = "low"
    success_eligible = confidence == "high"

    return {
        "algorithm_revision": ALGORITHM_REVISION,
        "configuration_checksum": configuration_checksum(config),
        "pitch_px": pitch,
        "diagnostic_pitch_px": pitch,
        "usable_pitch_px": pitch if success_eligible else None,
        "success_eligible": success_eligible,
        "confidence": confidence,
        "subwindow_consistency": consistency,
        "harmonic_ambiguity": harmonic_details,
        "unavailable_reason": None,
        "reference_axis_roi": reference_axis,
        "band_results": band_results,
    }


def _serializable_result(result: dict) -> dict:
    """Drop profile arrays while retaining all numerical diagnostics."""

    cleaned = {
        key: value
        for key, value in result.items()
        if key != "band_results"
    }
    cleaned["band_results"] = []
    for band in result["band_results"]:
        cleaned["band_results"].append(
            {
                key: value
                for key, value in band.items()
                if key
                not in {
                    "profile",
                    "detrended_profile",
                    "autocorrelation_curve",
                }
            }
        )
    return cleaned


def _git_provenance() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty}


def _ground_truth_pitch(annotation: dict) -> dict | None:
    if annotation["label"] != "valid":
        return None
    centers = [center["x_global"] for center in annotation["centers"]]
    intervals = np.diff(centers).astype(np.float64)
    median = float(np.median(intervals))
    mad = float(np.median(np.abs(intervals - median)))
    return {
        "pitch_px": median,
        "mad_px": mad,
        "minimum_interval_px": float(np.min(intervals)),
        "maximum_interval_px": float(np.max(intervals)),
    }


def _development_validation_annotations() -> dict[str, dict]:
    """Load original development plus the transferred P026 validation case."""

    with DEVELOPMENT_PATH.open("r", encoding="utf-8") as input_file:
        document = json.load(input_file)
    if (
        document.get("dataset_version") != DATASET_VERSION
        or document.get("split") != "development"
        or document.get("access_policy")
        != "available_during_method_design"
    ):
        raise ValueError("refusing to evaluate a non-development document")
    with P026_VALIDATION_PATH.open("r", encoding="utf-8") as input_file:
        validation = json.load(input_file)
    if (
        validation.get("dataset_version") != "raw_pitch_gt_v2"
        or validation.get("split") != "development-validation"
        or set(validation.get("annotations", {})) != {"P026"}
    ):
        raise ValueError("P026 validation fixture is invalid")
    annotations = dict(document["annotations"])
    annotations.update(validation["annotations"])
    return annotations


def evaluate_development(
    config: RawLocalPitchConfig = DEFAULT_CONFIG,
) -> dict:
    """Evaluate original development plus P026, never the old held-out set."""

    annotations = _development_validation_annotations()
    image_cache: dict[str, np.ndarray] = {}
    rows = []
    valid_errors = []
    valid_estimated = 0
    valid_success_eligible = 0
    confident_harmonic_errors = 0
    prevented_harmonic_errors = []
    nonvalid_success_eligible = []
    medium_success_eligible = []
    for sample_id, annotation in annotations.items():
        image_name = annotation["image_name"]
        if image_name not in image_cache:
            image_cache[image_name] = load_grayscale_image(
                IMAGES_DIR / image_name
            )
        result = estimate_raw_local_pitch(
            image_cache[image_name],
            annotation["reference_global"],
            annotation["roi_bounds_global"],
            direction="vertical",
            config=config,
        )
        ground_truth = _ground_truth_pitch(annotation)
        relative_error = None
        harmonic_error = False
        if ground_truth is not None and result["pitch_px"] is not None:
            valid_estimated += 1
            relative_error = abs(
                result["pitch_px"] - ground_truth["pitch_px"]
            ) / ground_truth["pitch_px"]
            valid_errors.append(relative_error)
            ratio = result["pitch_px"] / ground_truth["pitch_px"]
            harmonic_error = ratio < 0.67 or ratio > 1.5
            if result["confidence"] == "high" and harmonic_error:
                confident_harmonic_errors += 1
        if ground_truth is not None and result["success_eligible"]:
            valid_success_eligible += 1
        if (
            ground_truth is not None
            and result["pitch_px"] is None
            and result["diagnostic_pitch_px"] is not None
        ):
            diagnostic_ratio = (
                result["diagnostic_pitch_px"]
                / ground_truth["pitch_px"]
            )
            if diagnostic_ratio < 0.67 or diagnostic_ratio > 1.5:
                prevented_harmonic_errors.append(
                    {
                        "sample_id": sample_id,
                        "diagnostic_ratio": diagnostic_ratio,
                        "unavailable_reason": result[
                            "unavailable_reason"
                        ],
                    }
                )
        if (
            annotation["label"] in {"ambiguous", "unavailable"}
            and result["success_eligible"]
        ):
            nonvalid_success_eligible.append(sample_id)
        if result["confidence"] == "medium" and result["success_eligible"]:
            medium_success_eligible.append(sample_id)
        rows.append(
            {
                "sample_id": sample_id,
                "evaluation_source": (
                    "p026_validation"
                    if sample_id == "P026"
                    else "original_development"
                ),
                "label": annotation["label"],
                "annotation_confidence": annotation["confidence"],
                "ground_truth": ground_truth,
                "estimate": _serializable_result(result),
                "relative_error": relative_error,
                "harmonic_error": harmonic_error,
            }
        )

    valid_total = sum(row["label"] == "valid" for row in rows)
    estimated_rate = valid_estimated / valid_total if valid_total else 0.0
    metrics = {
        "valid_total": valid_total,
        "valid_auxiliary_estimated": valid_estimated,
        "valid_auxiliary_estimated_rate": estimated_rate,
        "valid_success_eligible": valid_success_eligible,
        "median_relative_error": (
            float(np.median(valid_errors)) if valid_errors else None
        ),
        "maximum_relative_error": (
            float(np.max(valid_errors)) if valid_errors else None
        ),
        "confident_harmonic_errors": confident_harmonic_errors,
        "prevented_harmonic_errors": prevented_harmonic_errors,
        "medium_success_eligible": medium_success_eligible,
        "ambiguous_or_unavailable_success_eligible": (
            nonvalid_success_eligible
        ),
    }
    p026_row = next(row for row in rows if row["sample_id"] == "P026")
    p026_harmonic_safe = (
        p026_row["estimate"]["pitch_px"] is None
        and not p026_row["estimate"]["success_eligible"]
        and p026_row["estimate"]["unavailable_reason"]
        == "harmonic_ambiguous"
    )
    metrics["p026_harmonic_safe"] = p026_harmonic_safe
    metrics["acceptance_passed"] = (
        estimated_rate >= 0.80
        and metrics["median_relative_error"] is not None
        and metrics["median_relative_error"] <= 0.10
        and metrics["maximum_relative_error"] <= 0.20
        and confident_harmonic_errors == 0
        and not nonvalid_success_eligible
        and not medium_success_eligible
        and p026_harmonic_safe
    )
    return {
        "dataset_version": DATASET_VERSION,
        "evaluated_split": "development_plus_p026_validation",
        "heldout_accessed": False,
        "algorithm_revision": ALGORITHM_REVISION,
        "configuration": canonical_configuration(config),
        "configuration_checksum": configuration_checksum(config),
        **_git_provenance(),
        "metrics": metrics,
        "samples": rows,
    }


def _draw_polyline(
    canvas: np.ndarray,
    values: np.ndarray,
    x0: int,
    y0: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    scale = maximum - minimum
    if scale < 1e-12:
        return
    xs = np.linspace(x0, x0 + width - 1, len(values)).round().astype(int)
    ys = (
        y0
        + height
        - 1
        - (values - minimum) / scale * (height - 1)
    ).round().astype(int)
    points = np.column_stack((xs, ys)).astype(np.int32)
    cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)


def save_debug_image(
    annotation: dict,
    image_gray: np.ndarray,
    result: dict,
    output_path: Path,
) -> None:
    """Save raw ROI, five profiles, and autocorrelation diagnostics."""

    roi = extract_raw_roi(image_gray, annotation["roi_bounds_global"])
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    canvas = np.full((520, 900, 3), 248, dtype=np.uint8)
    display_width = 500
    display_height = max(1, round(roi.shape[0] * display_width / roi.shape[1]))
    roi_display = cv2.resize(
        roi_bgr,
        (display_width, display_height),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas[45 : 45 + display_height, 10:510] = roi_display
    title = (
        f"{annotation['sample_id']} label={annotation['label']} "
        f"pitch={result['pitch_px']} confidence={result['confidence']}"
    )
    cv2.putText(
        canvas,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    colors = [
        (220, 70, 40),
        (40, 150, 40),
        (30, 120, 220),
        (160, 70, 160),
        (80, 140, 180),
    ]
    for band in result["band_results"]:
        color = colors[band["band_index"] % len(colors)]
        if "profile" in band:
            _draw_polyline(
                canvas,
                band["profile"],
                10,
                250,
                500,
                105,
                color,
            )
        if "autocorrelation_curve" in band:
            _draw_polyline(
                canvas,
                band["autocorrelation_curve"][1:],
                540,
                45,
                340,
                310,
                color,
            )
        label = (
            f"B{band['band_index'] + 1}: "
            f"{band.get('pitch_px', band.get('reason'))}"
        )
        cv2.putText(
            canvas,
            label,
            (15, 382 + band["band_index"] * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "raw-gray subwindow profiles",
        (10, 238),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "normalized autocorrelation",
        (540, 382),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"could not save {output_path}")


def write_development_outputs(report: dict) -> None:
    """Persist the small development-only report and debug images."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    annotations = _development_validation_annotations()
    image_cache: dict[str, np.ndarray] = {}
    rows_by_id = {
        row["sample_id"]: row for row in report["samples"]
    }
    for sample_id, annotation in annotations.items():
        image_name = annotation["image_name"]
        if image_name not in image_cache:
            image_cache[image_name] = load_grayscale_image(
                IMAGES_DIR / image_name
            )
        result = estimate_raw_local_pitch(
            image_cache[image_name],
            annotation["reference_global"],
            annotation["roi_bounds_global"],
            direction="vertical",
            config=DEFAULT_CONFIG,
        )
        save_debug_image(
            annotation,
            image_cache[image_name],
            result,
            DEBUG_DIR / f"{sample_id}.png",
        )
        if (
            rows_by_id[sample_id]["estimate"]["pitch_px"]
            != result["pitch_px"]
        ):
            raise RuntimeError("debug rerun did not reproduce report estimate")


def main() -> None:
    report = evaluate_development(DEFAULT_CONFIG)
    write_development_outputs(report)
    print(
        json.dumps(
            {
                "evaluated_split": report["evaluated_split"],
                "heldout_accessed": report["heldout_accessed"],
                "configuration_checksum": report[
                    "configuration_checksum"
                ],
                "metrics": report["metrics"],
                "report_path": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
