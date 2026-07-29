"""Offline raw local-pitch prototype v3.

This module remains independent of the production detector.  It reads only
source grayscale pixels, a fixed ROI, a reference point, and stripe direction.
It reuses the frozen v2 raw-profile primitives, but not v2 candidate selection.

V3 keeps weak per-band evidence until cross-band aggregation.  Autocorrelation
peaks and stable spectral peaks are both represented in a unified hypothesis
pool.  Only one high-confidence, harmonically safe hypothesis can expose a
usable pitch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

import numpy as np

from tools import raw_local_pitch_prototype as v2


ALGORITHM_REVISION = "raw_local_pitch_stage1_v3"


@dataclass(frozen=True)
class RawLocalPitchV3Config:
    """Frozen tunable parameters for the offline v3 prototype."""

    subwindow_count: int = 5
    profile_percentile: float = 70.0
    min_pitch_px: int = 8
    max_pitch_px: int = 90
    minimum_profile_contrast: float = 6.0
    fft_padding_factor: int = 8
    spectral_peaks_per_band: int = 8
    hypothesis_merge_tolerance_ratio: float = 0.08
    minimum_hypothesis_bands: int = 2
    high_min_supporting_bands: int = 4
    high_min_cross_source_bands: int = 3
    high_relative_mad_limit: float = 0.07
    high_aggregate_score_floor: float = 0.72
    high_median_autocorrelation_floor: float = 0.60
    high_spectral_only_min_bands: int = 5
    high_spectral_only_support_floor: float = 0.75
    aggregate_autocorrelation_evidence_floor: float = 0.18
    aggregate_spectral_evidence_floor: float = 0.10
    medium_min_supporting_bands: int = 3
    medium_relative_mad_limit: float = 0.14
    medium_aggregate_score_floor: float = 0.52
    competing_hypothesis_score_ratio: float = 0.65
    competing_nonharmonic_period_ratio_limit: float = 1.75
    harmonic_competitor_score_ratio: float = 0.55
    harmonic_resolution_score_ratio: float = 1.15
    harmonic_resolution_spectral_ratio: float = 1.50
    harmonic_integer_ratio_tolerance: float = 0.12


DEFAULT_CONFIG = RawLocalPitchV3Config()


def canonical_configuration(config: RawLocalPitchV3Config) -> dict:
    """Return the exact payload used for the frozen checksum."""

    return {
        "algorithm_revision": ALGORITHM_REVISION,
        "parameters": asdict(config),
    }


def configuration_checksum(config: RawLocalPitchV3Config) -> str:
    """Return a deterministic checksum for revision plus parameters."""

    payload = json.dumps(
        canonical_configuration(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _spectral_peak_indexes(
    support: np.ndarray,
    limit: int,
) -> list[int]:
    """Return strongest deterministic local spectral peaks."""

    if not len(support):
        return []
    indexes = set(v2._local_maxima(support, 1, len(support)))
    indexes.add(int(np.argmax(support)))
    ranked = sorted(
        indexes,
        key=lambda index: (-float(support[index]), index),
    )
    return ranked[:limit]


def _safe_autocorrelation_value(
    autocorrelation: np.ndarray,
    period: float,
) -> float:
    index = int(np.clip(round(period), 1, len(autocorrelation) - 1))
    value = float(autocorrelation[index])
    return value if np.isfinite(value) else 0.0


def _band_candidate_evidence(
    raw_profile: np.ndarray,
    band_index: int,
    row_bounds: tuple[int, int],
    config: RawLocalPitchV3Config,
) -> dict:
    """Retain weak ACF and spectral candidates for one raw-gray band."""

    detrended = v2._quadratic_detrend(raw_profile)
    contrast = float(
        np.percentile(raw_profile, 90)
        - np.percentile(raw_profile, 10)
    )
    signal_scale = float(np.std(detrended))
    base = {
        "band_index": band_index,
        "row_bounds": {"y0": row_bounds[0], "y1": row_bounds[1]},
        "raw_profile_contrast": contrast,
        "detrended_standard_deviation": signal_scale,
        "candidates": [],
    }
    if contrast < config.minimum_profile_contrast or signal_scale < 1e-6:
        return {
            **base,
            "evidence_available": False,
            "reason": "insufficient_raw_profile_contrast",
        }

    max_pitch = min(
        config.max_pitch_px,
        max(config.min_pitch_px, raw_profile.size // 3),
    )
    autocorrelation = v2.normalized_autocorrelation(
        detrended,
        max_pitch,
    )
    periods, spectral = v2._spectral_support(
        detrended,
        config.min_pitch_px,
        max_pitch,
        config.fft_padding_factor,
    )
    acf_peaks = v2._local_maxima(
        autocorrelation,
        config.min_pitch_px,
        max_pitch + 1,
    )
    candidates = []
    for peak in acf_peaks:
        correlation = float(autocorrelation[peak])
        if not np.isfinite(correlation) or correlation <= 0:
            continue
        period = v2._refine_peak(autocorrelation, peak)
        candidates.append(
            {
                "source": "autocorrelation",
                "period_px": period,
                "integer_lag_px": peak,
                "autocorrelation": correlation,
                "spectral_support": v2._support_at_period(
                    periods,
                    spectral,
                    period,
                ),
            }
        )
    for index in _spectral_peak_indexes(
        spectral,
        config.spectral_peaks_per_band,
    ):
        period = float(periods[index])
        candidates.append(
            {
                "source": "spectrum",
                "period_px": period,
                "autocorrelation": _safe_autocorrelation_value(
                    autocorrelation,
                    period,
                ),
                "spectral_support": float(spectral[index]),
            }
        )
    candidates.sort(
        key=lambda candidate: (
            candidate["period_px"],
            candidate["source"],
        )
    )
    dominant_index = int(np.argmax(spectral))
    return {
        **base,
        "evidence_available": bool(candidates),
        "reason": None if candidates else "no_raw_period_candidates",
        "maximum_autocorrelation": (
            max(
                (
                    candidate["autocorrelation"]
                    for candidate in candidates
                    if candidate["source"] == "autocorrelation"
                ),
                default=0.0,
            )
        ),
        "dominant_spectral_period_px": float(periods[dominant_index]),
        "dominant_spectral_support": float(spectral[dominant_index]),
        "candidates": candidates,
    }


def _relative_distance(left: float, right: float) -> float:
    return abs(left - right) / max((left + right) / 2.0, 1e-12)


def _best_source_candidate(
    candidates: Iterable[dict],
    source: str,
    center: float,
    tolerance: float,
) -> dict | None:
    matches = [
        candidate
        for candidate in candidates
        if (
            candidate["source"] == source
            and _relative_distance(candidate["period_px"], center)
            <= tolerance
        )
    ]
    if not matches:
        return None
    strength_key = (
        "autocorrelation"
        if source == "autocorrelation"
        else "spectral_support"
    )
    return min(
        matches,
        key=lambda candidate: (
            -candidate[strength_key],
            _relative_distance(candidate["period_px"], center),
            candidate["period_px"],
        ),
    )


def _collect_hypothesis_evidence(
    center: float,
    band_results: list[dict],
    config: RawLocalPitchV3Config,
) -> tuple[float, list[dict]]:
    """Collect at most one ACF and spectral item per band around a center."""

    evidence_by_band = []
    band_periods = []
    for band in band_results:
        acf = _best_source_candidate(
            band["candidates"],
            "autocorrelation",
            center,
            config.hypothesis_merge_tolerance_ratio,
        )
        spectrum = _best_source_candidate(
            band["candidates"],
            "spectrum",
            center,
            config.hypothesis_merge_tolerance_ratio,
        )
        if acf is None and spectrum is None:
            continue
        source_periods = [
            candidate["period_px"]
            for candidate in (acf, spectrum)
            if candidate is not None
        ]
        band_period = float(np.median(source_periods))
        band_periods.append(band_period)
        evidence_by_band.append(
            {
                "band_index": band["band_index"],
                "period_px": band_period,
                "sources": [
                    source
                    for source, candidate in (
                        ("autocorrelation", acf),
                        ("spectrum", spectrum),
                    )
                    if candidate is not None
                ],
                "autocorrelation_candidate": acf,
                "spectral_candidate": spectrum,
            }
        )
    refined_center = (
        float(np.median(band_periods)) if band_periods else center
    )
    return refined_center, evidence_by_band


def _relative_mad(values: list[float], center: float) -> float:
    if not values or center <= 0:
        return float("inf")
    return float(
        np.median([abs(value - center) for value in values]) / center
    )


def _score_hypothesis(
    center: float,
    evidence_by_band: list[dict],
    total_bands: int,
    config: RawLocalPitchV3Config,
) -> dict:
    acf_evidence = [
        evidence["autocorrelation_candidate"]
        for evidence in evidence_by_band
        if evidence["autocorrelation_candidate"] is not None
    ]
    spectral_evidence = [
        evidence["spectral_candidate"]
        for evidence in evidence_by_band
        if evidence["spectral_candidate"] is not None
    ]
    qualified_acf = [
        candidate
        for candidate in acf_evidence
        if (
            candidate["autocorrelation"]
            >= config.aggregate_autocorrelation_evidence_floor
        )
    ]
    qualified_spectral = [
        candidate
        for candidate in spectral_evidence
        if (
            candidate["spectral_support"]
            >= config.aggregate_spectral_evidence_floor
        )
    ]
    qualified_cross_source = [
        evidence
        for evidence in evidence_by_band
        if (
            evidence["autocorrelation_candidate"] is not None
            and evidence["spectral_candidate"] is not None
            and evidence["autocorrelation_candidate"]["autocorrelation"]
            >= config.aggregate_autocorrelation_evidence_floor
            and evidence["spectral_candidate"]["spectral_support"]
            >= config.aggregate_spectral_evidence_floor
        )
    ]
    band_periods = [evidence["period_px"] for evidence in evidence_by_band]
    relative_mad = _relative_mad(band_periods, center)
    median_acf = (
        float(
            np.median(
                [
                    max(0.0, candidate["autocorrelation"])
                    for candidate in acf_evidence
                ]
            )
        )
        if acf_evidence
        else 0.0
    )
    median_spectral = (
        float(
            np.median(
                [
                    candidate["spectral_support"]
                    for candidate in spectral_evidence
                ]
            )
        )
        if spectral_evidence
        else 0.0
    )
    supporting_bands = len(evidence_by_band)
    coverage = supporting_bands / total_bands
    acf_weight = sum(
        max(0.0, candidate["autocorrelation"])
        for candidate in acf_evidence
    ) / total_bands
    spectral_weight = sum(
        candidate["spectral_support"]
        for candidate in spectral_evidence
    ) / total_bands
    consistency = max(0.0, 1.0 - relative_mad / 0.08)
    score = (
        0.30 * coverage
        + 0.25 * acf_weight
        + 0.30 * spectral_weight
        + 0.15 * consistency
    )
    return {
        "period_px": center,
        "supporting_band_count": supporting_bands,
        "autocorrelation_band_count": len(acf_evidence),
        "spectral_band_count": len(spectral_evidence),
        "qualified_autocorrelation_band_count": len(qualified_acf),
        "qualified_spectral_band_count": len(qualified_spectral),
        "cross_source_band_count": len(qualified_cross_source),
        "relative_mad": relative_mad,
        "median_autocorrelation": median_acf,
        "median_spectral_support": median_spectral,
        "aggregate_score": float(score),
        "band_support": evidence_by_band,
    }


def build_unified_hypotheses(
    band_results: list[dict],
    config: RawLocalPitchV3Config,
) -> list[dict]:
    """Build deterministic cross-band hypotheses from both evidence types."""

    anchors = sorted(
        {
            round(candidate["period_px"], 9)
            for band in band_results
            for candidate in band["candidates"]
        }
    )
    raw_hypotheses = []
    for anchor in anchors:
        center, _ = _collect_hypothesis_evidence(
            anchor,
            band_results,
            config,
        )
        center, evidence = _collect_hypothesis_evidence(
            center,
            band_results,
            config,
        )
        if len(evidence) < config.minimum_hypothesis_bands:
            continue
        raw_hypotheses.append(
            _score_hypothesis(
                center,
                evidence,
                len(band_results),
                config,
            )
        )

    ranked = sorted(
        raw_hypotheses,
        key=lambda hypothesis: (
            -hypothesis["aggregate_score"],
            -hypothesis["supporting_band_count"],
            -hypothesis["cross_source_band_count"],
            hypothesis["relative_mad"],
            hypothesis["period_px"],
        ),
    )
    unique = []
    for hypothesis in ranked:
        if any(
            _relative_distance(
                hypothesis["period_px"],
                existing["period_px"],
            )
            <= config.hypothesis_merge_tolerance_ratio / 2.0
            for existing in unique
        ):
            continue
        unique.append(hypothesis)
    unique.sort(
        key=lambda hypothesis: (
            -hypothesis["aggregate_score"],
            -hypothesis["supporting_band_count"],
            -hypothesis["cross_source_band_count"],
            hypothesis["relative_mad"],
            hypothesis["period_px"],
        )
    )
    for index, hypothesis in enumerate(unique, start=1):
        hypothesis["hypothesis_id"] = f"H{index:02d}"
    return unique


def _harmonic_relation(
    selected: dict,
    competitor: dict,
    config: RawLocalPitchV3Config,
) -> dict | None:
    larger = max(selected["period_px"], competitor["period_px"])
    smaller = min(selected["period_px"], competitor["period_px"])
    ratio = larger / smaller
    multiple = int(round(ratio))
    if multiple not in {2, 3}:
        return None
    relative_error = abs(ratio - multiple) / multiple
    if relative_error > config.harmonic_integer_ratio_tolerance:
        return None
    return {
        "selected_hypothesis_id": selected["hypothesis_id"],
        "competitor_hypothesis_id": competitor["hypothesis_id"],
        "selected_period_px": selected["period_px"],
        "competitor_period_px": competitor["period_px"],
        "ratio": ratio,
        "nearest_multiple": multiple,
        "direction": (
            "selected_is_larger"
            if selected["period_px"] > competitor["period_px"]
            else "selected_is_smaller"
        ),
        "relative_distance_to_multiple": relative_error,
    }


def _arbitrate_hypotheses(
    hypotheses: list[dict],
    config: RawLocalPitchV3Config,
) -> dict:
    if not hypotheses:
        return {
            "selected": None,
            "harmonic_competitors": [],
            "resolved_harmonic_competitors": [],
            "unresolved_harmonic_competitors": [],
            "close_nonharmonic_competitors": [],
            "rejection_reason": "no_cross_band_hypothesis",
        }

    selected = hypotheses[0]
    harmonic = []
    resolved_harmonic = []
    unresolved_harmonic = []
    close_nonharmonic = []
    for competitor in hypotheses[1:]:
        if (
            competitor["supporting_band_count"]
            < config.medium_min_supporting_bands
        ):
            continue
        score_ratio = competitor["aggregate_score"] / max(
            selected["aggregate_score"],
            1e-12,
        )
        relation = _harmonic_relation(selected, competitor, config)
        if relation is not None:
            relation["competitor_score_ratio"] = score_ratio
            if score_ratio < config.harmonic_competitor_score_ratio:
                relation["resolution"] = "competitor_too_weak"
                resolved_harmonic.append(relation)
            else:
                selected_spectral = selected[
                    "median_spectral_support"
                ]
                competitor_spectral = competitor[
                    "median_spectral_support"
                ]
                resolved = (
                    selected["aggregate_score"]
                    >= (
                        competitor["aggregate_score"]
                        * config.harmonic_resolution_score_ratio
                    )
                    and selected_spectral
                    >= (
                        competitor_spectral
                        * config.harmonic_resolution_spectral_ratio
                    )
                )
                relation["resolution"] = (
                    "selected_has_stronger_joint_evidence"
                    if resolved
                    else "unresolved"
                )
                (
                    resolved_harmonic
                    if resolved
                    else unresolved_harmonic
                ).append(relation)
            harmonic.append(relation)
        elif (
            score_ratio >= config.competing_hypothesis_score_ratio
            and max(
                selected["period_px"],
                competitor["period_px"],
            )
            / min(
                selected["period_px"],
                competitor["period_px"],
            )
            <= config.competing_nonharmonic_period_ratio_limit
        ):
            close_nonharmonic.append(
                {
                    "selected_hypothesis_id": selected["hypothesis_id"],
                    "competitor_hypothesis_id": competitor[
                        "hypothesis_id"
                    ],
                    "selected_period_px": selected["period_px"],
                    "competitor_period_px": competitor["period_px"],
                    "competitor_score_ratio": score_ratio,
                }
            )

    rejection_reason = None
    if unresolved_harmonic:
        rejection_reason = "harmonic_ambiguous"
    elif close_nonharmonic:
        rejection_reason = "ambiguous_competing_hypotheses"
    return {
        "selected": selected,
        "harmonic_competitors": harmonic,
        "resolved_harmonic_competitors": resolved_harmonic,
        "unresolved_harmonic_competitors": unresolved_harmonic,
        "close_nonharmonic_competitors": close_nonharmonic,
        "rejection_reason": rejection_reason,
    }


def estimate_raw_local_pitch_v3(
    image_gray: np.ndarray,
    reference_global: dict,
    roi_bounds_global: dict,
    direction: str = "vertical",
    config: RawLocalPitchV3Config = DEFAULT_CONFIG,
) -> dict:
    """Estimate raw local pitch with cross-band unified evidence."""

    roi_gray = v2.extract_raw_roi(image_gray, roi_bounds_global)
    directional = v2._directional_roi(roi_gray, direction)
    reference_axis = (
        reference_global["x"] - roi_bounds_global["x0"]
        if direction == "vertical"
        else reference_global["y"] - roi_bounds_global["y0"]
    )
    if not 0 <= reference_axis < directional.shape[1]:
        raise ValueError("reference point is outside the ROI pitch axis")

    band_results = []
    for index, bounds in enumerate(
        v2._subwindow_bounds(
            directional.shape[0],
            config.subwindow_count,
        )
    ):
        profile = v2.raw_gray_profile(
            directional[bounds[0] : bounds[1]],
            config.profile_percentile,
        )
        band_results.append(
            _band_candidate_evidence(
                profile,
                index,
                bounds,
                config,
            )
        )

    hypotheses = build_unified_hypotheses(band_results, config)
    arbitration = _arbitrate_hypotheses(hypotheses, config)
    selected = arbitration["selected"]
    base = {
        "algorithm_revision": ALGORITHM_REVISION,
        "configuration_checksum": configuration_checksum(config),
        "reference_axis_roi": reference_axis,
        "band_results": band_results,
        "hypotheses": hypotheses,
        "arbitration": {
            "selected_hypothesis_id": (
                selected["hypothesis_id"] if selected else None
            ),
            "ranked_hypothesis_ids": [
                hypothesis["hypothesis_id"]
                for hypothesis in hypotheses
            ],
            "harmonic_competitors": arbitration[
                "harmonic_competitors"
            ],
            "resolved_harmonic_competitors": arbitration[
                "resolved_harmonic_competitors"
            ],
            "unresolved_harmonic_competitors": arbitration[
                "unresolved_harmonic_competitors"
            ],
            "close_nonharmonic_competitors": arbitration[
                "close_nonharmonic_competitors"
            ],
            "rejection_reason": arbitration["rejection_reason"],
        },
    }
    if selected is None:
        return {
            **base,
            "pitch_px": None,
            "diagnostic_pitch_px": None,
            "usable_pitch_px": None,
            "success_eligible": False,
            "confidence": "unavailable",
            "harmonic_ambiguity": {
                "detected": False,
                "reason": None,
            },
            "unavailable_reason": "no_cross_band_hypothesis",
        }

    pitch = selected["period_px"]
    if arbitration["rejection_reason"] is not None:
        harmonic_ambiguous = (
            arbitration["rejection_reason"] == "harmonic_ambiguous"
        )
        return {
            **base,
            "pitch_px": None,
            "diagnostic_pitch_px": pitch,
            "usable_pitch_px": None,
            "success_eligible": False,
            "confidence": "unavailable",
            "harmonic_ambiguity": {
                "detected": harmonic_ambiguous,
                "reason": (
                    "unresolved_bidirectional_2x_or_3x_conflict"
                    if harmonic_ambiguous
                    else None
                ),
            },
            "unavailable_reason": arbitration["rejection_reason"],
        }

    cross_source_high = (
        selected["cross_source_band_count"]
        >= config.high_min_cross_source_bands
        and selected["median_autocorrelation"]
        >= config.high_median_autocorrelation_floor
    )
    spectral_only_high = (
        selected["qualified_spectral_band_count"]
        >= config.high_spectral_only_min_bands
        and selected["median_spectral_support"]
        >= config.high_spectral_only_support_floor
        and selected["qualified_autocorrelation_band_count"] == 0
    )
    is_high = (
        selected["supporting_band_count"]
        >= config.high_min_supporting_bands
        and selected["relative_mad"]
        <= config.high_relative_mad_limit
        and selected["aggregate_score"]
        >= config.high_aggregate_score_floor
        and (cross_source_high or spectral_only_high)
    )
    is_medium = (
        selected["supporting_band_count"]
        >= config.medium_min_supporting_bands
        and selected["relative_mad"]
        <= config.medium_relative_mad_limit
        and selected["aggregate_score"]
        >= config.medium_aggregate_score_floor
    )
    confidence = "high" if is_high else ("medium" if is_medium else "low")
    success_eligible = confidence == "high"
    rejection_reason = None if success_eligible else "not_high_confidence"
    base["arbitration"]["rejection_reason"] = rejection_reason
    return {
        **base,
        "pitch_px": pitch,
        "diagnostic_pitch_px": pitch,
        "usable_pitch_px": pitch if success_eligible else None,
        "success_eligible": success_eligible,
        "confidence": confidence,
        "harmonic_ambiguity": {
            "detected": False,
            "reason": None,
        },
        "unavailable_reason": None,
    }
