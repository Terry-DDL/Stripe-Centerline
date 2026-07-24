"""Whole-image reference pitch map used as an independent safety check."""

from dataclasses import dataclass

import cv2
import numpy as np

from config import InteractiveConfig, ProcessingConfig
from image_processing import (
    apply_vertical_close_roi,
    create_black_mask_roi,
    create_otsu_binary_roi,
    gaussian_blur_roi,
)


@dataclass(frozen=True)
class PitchTileEstimate:
    """One spatially local estimate in the whole-image pitch map."""

    x0: int
    x1: int
    y0: int
    y1: int
    pitch_px: float | None
    gap_count: int
    cluster_gap_count: int
    cluster_support_ratio: float
    valid: bool
    reason: str | None

    def contains(self, x: int, y: int) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def to_dict(self) -> dict:
        return {
            "bounds": {
                "x0": self.x0,
                "x1": self.x1,
                "y0": self.y0,
                "y1": self.y1,
            },
            "pitch_px": self.pitch_px,
            "gap_count": self.gap_count,
            "cluster_gap_count": self.cluster_gap_count,
            "cluster_support_ratio": self.cluster_support_ratio,
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PitchReferenceQuery:
    """Reference pitch available around one clicked image point."""

    baseline_pitch_px: float | None
    source_tile_count: int
    reason: str | None
    tile_pitches_px: tuple[float, ...]


@dataclass(frozen=True)
class PitchReferenceMap:
    """Sparse two-dimensional reference pitch estimates for one image."""

    image_shape: tuple[int, int]
    tiles: tuple[PitchTileEstimate, ...]
    tile_width_px: int
    tile_height_px: int
    stride_x_px: int
    stride_y_px: int

    @property
    def valid_tile_count(self) -> int:
        return sum(tile.valid for tile in self.tiles)

    def query(
        self,
        x: int,
        y: int,
        max_disagreement_ratio: float,
    ) -> PitchReferenceQuery:
        pitches = tuple(
            tile.pitch_px
            for tile in self.tiles
            if tile.valid and tile.pitch_px is not None and tile.contains(x, y)
        )
        if not pitches:
            return PitchReferenceQuery(
                baseline_pitch_px=None,
                source_tile_count=0,
                reason="no_reliable_pitch_tile_at_click",
                tile_pitches_px=(),
            )

        baseline = float(np.median(pitches))
        disagreement = (max(pitches) - min(pitches)) / max(baseline, 1e-12)
        if disagreement > max_disagreement_ratio:
            return PitchReferenceQuery(
                baseline_pitch_px=None,
                source_tile_count=len(pitches),
                reason="overlapping_pitch_tiles_disagree",
                tile_pitches_px=tuple(round(value, 3) for value in pitches),
            )
        return PitchReferenceQuery(
            baseline_pitch_px=round(baseline, 3),
            source_tile_count=len(pitches),
            reason=None,
            tile_pitches_px=tuple(round(value, 3) for value in pitches),
        )

    def to_dict(self) -> dict:
        return {
            "image_shape": {
                "height": self.image_shape[0],
                "width": self.image_shape[1],
            },
            "parameters": {
                "tile_width_px": self.tile_width_px,
                "tile_height_px": self.tile_height_px,
                "stride_x_px": self.stride_x_px,
                "stride_y_px": self.stride_y_px,
            },
            "tile_count": len(self.tiles),
            "valid_tile_count": self.valid_tile_count,
            "tiles": [tile.to_dict() for tile in self.tiles],
        }


def validate_pitch_reference_config(config: InteractiveConfig) -> None:
    """Reject invalid pitch-map and global-guard settings."""

    positive_integer_names = (
        "pitch_map_tile_width_px",
        "pitch_map_tile_height_px",
        "pitch_map_stride_x_px",
        "pitch_map_stride_y_px",
        "pitch_map_row_step_px",
        "pitch_map_min_gap_count",
    )
    for name in positive_integer_names:
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be greater than 0")
    ratio_names = (
        "pitch_map_gap_cluster_tolerance_ratio",
        "pitch_map_min_cluster_support_ratio",
        "pitch_map_max_neighbor_disagreement_ratio",
        "pitch_guard_min_interval_ratio",
        "pitch_guard_max_interval_ratio",
    )
    for name in ratio_names:
        if getattr(config, name) <= 0.0:
            raise ValueError(f"{name} must be greater than 0")
    if config.pitch_map_min_cluster_support_ratio > 1.0:
        raise ValueError(
            "pitch_map_min_cluster_support_ratio must not exceed 1"
        )
    if (
        config.pitch_guard_min_interval_ratio
        >= config.pitch_guard_max_interval_ratio
    ):
        raise ValueError(
            "pitch_guard_min_interval_ratio must be below maximum"
        )


def _axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Cover an axis with overlapping tiles, including the far edge."""

    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _row_run_centers(
    mask_row,
    processing_config: ProcessingConfig,
) -> list[float]:
    """Return accepted black-run centers from one binary mask row."""

    foreground = np.asarray(mask_row) != 0
    changes = np.diff(
        np.pad(foreground.astype(np.int8), (1, 1), constant_values=0)
    )
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    width = foreground.shape[0]
    centers = []
    for start, end in zip(starts, ends):
        run_width = int(end - start)
        if processing_config.reject_border_touching_runs and (
            start == 0 or end == width
        ):
            continue
        if not (
            processing_config.min_run_width_px
            <= run_width
            <= processing_config.max_run_width_px
        ):
            continue
        centers.append((float(start) + float(end) - 1.0) / 2.0)
    return centers


def _sample_adjacent_gaps(
    black_mask,
    processing_config: ProcessingConfig,
    row_step: int,
) -> list[float]:
    """Collect adjacent run-center gaps without requiring vertical tracks."""

    gaps = []
    minimum_gap = processing_config.center_cluster_tolerance_px
    for y in range(0, black_mask.shape[0], row_step):
        centers = _row_run_centers(black_mask[y], processing_config)
        gaps.extend(
            right - left
            for left, right in zip(centers, centers[1:])
            if right - left > minimum_gap
        )
    return gaps


def _dominant_gap_cluster(
    gaps: list[float],
    tolerance_ratio: float,
) -> list[float]:
    """Return the largest deterministic relative-tolerance gap cluster."""

    if not gaps:
        return []
    values = np.sort(np.asarray(gaps, dtype=np.float64))
    tolerances = np.maximum(1.0, values * tolerance_ratio)
    lower_indices = np.searchsorted(
        values,
        values - tolerances,
        side="left",
    )
    upper_indices = np.searchsorted(
        values,
        values + tolerances,
        side="right",
    )
    counts = upper_indices - lower_indices
    best_count = int(counts.max())
    best_indices = np.flatnonzero(counts == best_count)
    best_index = min(
        best_indices,
        key=lambda index: float(
            np.median(values[lower_indices[index] : upper_indices[index]])
        ),
    )
    return values[
        lower_indices[best_index] : upper_indices[best_index]
    ].tolist()


def _estimate_tile_pitch(
    image_gray_tile,
    bounds: tuple[int, int, int, int],
    processing_config: ProcessingConfig,
    interactive_config: InteractiveConfig,
) -> PitchTileEstimate:
    """Estimate one tile using the dominant robust adjacent-gap cluster."""

    x0, x1, y0, y1 = bounds
    blurred = gaussian_blur_roi(image_gray_tile, processing_config)
    binary = create_otsu_binary_roi(blurred)
    closed = apply_vertical_close_roi(binary, processing_config)
    black_mask = create_black_mask_roi(closed)
    gaps = _sample_adjacent_gaps(
        black_mask,
        processing_config,
        interactive_config.pitch_map_row_step_px,
    )
    cluster = _dominant_gap_cluster(
        gaps,
        interactive_config.pitch_map_gap_cluster_tolerance_ratio,
    )
    support_ratio = len(cluster) / len(gaps) if gaps else 0.0
    reason = None
    if len(cluster) < interactive_config.pitch_map_min_gap_count:
        reason = "insufficient_gap_count"
    elif (
        support_ratio
        < interactive_config.pitch_map_min_cluster_support_ratio
    ):
        reason = "dominant_gap_cluster_too_weak"
    pitch = None if reason is not None else round(float(np.median(cluster)), 3)
    return PitchTileEstimate(
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        pitch_px=pitch,
        gap_count=len(gaps),
        cluster_gap_count=len(cluster),
        cluster_support_ratio=round(support_ratio, 6),
        valid=reason is None,
        reason=reason,
    )


def build_pitch_reference_map(
    image_gray,
    processing_config: ProcessingConfig,
    interactive_config: InteractiveConfig,
) -> PitchReferenceMap:
    """Scan the whole image once and build a reusable 2-D pitch map."""

    validate_pitch_reference_config(interactive_config)
    if image_gray.ndim != 2:
        raise ValueError("image_gray must be a single-channel image")
    height, width = image_gray.shape
    tile_width = min(interactive_config.pitch_map_tile_width_px, width)
    tile_height = min(interactive_config.pitch_map_tile_height_px, height)
    x_starts = _axis_starts(
        width,
        tile_width,
        interactive_config.pitch_map_stride_x_px,
    )
    y_starts = _axis_starts(
        height,
        tile_height,
        interactive_config.pitch_map_stride_y_px,
    )
    tiles = []
    for y0 in y_starts:
        for x0 in x_starts:
            x1 = x0 + tile_width
            y1 = y0 + tile_height
            tiles.append(
                _estimate_tile_pitch(
                    image_gray[y0:y1, x0:x1],
                    (x0, x1, y0, y1),
                    processing_config,
                    interactive_config,
                )
            )
    return PitchReferenceMap(
        image_shape=(height, width),
        tiles=tuple(tiles),
        tile_width_px=tile_width,
        tile_height_px=tile_height,
        stride_x_px=interactive_config.pitch_map_stride_x_px,
        stride_y_px=interactive_config.pitch_map_stride_y_px,
    )


def create_pitch_reference_debug(
    image_gray,
    pitch_map: PitchReferenceMap,
) -> np.ndarray:
    """Draw the hidden whole-image pitch tiles for later diagnostics."""

    debug = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    for tile in pitch_map.tiles:
        color = (0, 180, 0) if tile.valid else (0, 0, 180)
        cv2.rectangle(
            debug,
            (tile.x0, tile.y0),
            (tile.x1 - 1, tile.y1 - 1),
            color,
            1,
        )
        label = (
            f"{tile.pitch_px:g}px"
            if tile.pitch_px is not None
            else "unverified"
        )
        cv2.putText(
            debug,
            label,
            (tile.x0 + 5, min(tile.y0 + 18, tile.y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return debug


def evaluate_pitch_guard(
    selection,
    click_x_global: int,
    click_y_global: int,
    pitch_map: PitchReferenceMap,
    interactive_config: InteractiveConfig,
) -> dict:
    """Compare selected immediate intervals with the independent map."""

    result = {
        "status": "Not applicable",
        "baseline_pitch_px": None,
        "observed_intervals_px": [],
        "interval_pitch_ratios": [],
        "source_tile_count": 0,
        "normal_interval_ratio_min": (
            interactive_config.pitch_guard_min_interval_ratio
        ),
        "normal_interval_ratio_max": (
            interactive_config.pitch_guard_max_interval_ratio
        ),
        "reason": "local_detection_not_successful",
    }
    if not selection.success:
        return result

    if (
        selection.left_track is None
        or selection.right_track is None
    ):
        result.update(
            {
                "status": "Unable to verify",
                "reason": "selected_interval_missing",
            }
        )
        return result
    if (
        selection.click_classification == "black_stripe"
        and selection.clicked_track is not None
    ):
        intervals = (
            selection.clicked_track.center_x_roi
            - selection.left_track.center_x_roi,
            selection.right_track.center_x_roi
            - selection.clicked_track.center_x_roi,
        )
    else:
        intervals = (
            selection.right_track.center_x_roi
            - selection.left_track.center_x_roi,
        )
    result["observed_intervals_px"] = [
        round(interval, 3) for interval in intervals
    ]
    query = pitch_map.query(
        click_x_global,
        click_y_global,
        interactive_config.pitch_map_max_neighbor_disagreement_ratio,
    )
    result["source_tile_count"] = query.source_tile_count
    if query.baseline_pitch_px is None:
        result.update(
            {
                "status": "Unable to verify",
                "reason": query.reason,
            }
        )
        return result

    ratios = tuple(
        interval / query.baseline_pitch_px for interval in intervals
    )
    normal = all(
        interactive_config.pitch_guard_min_interval_ratio
        <= ratio
        <= interactive_config.pitch_guard_max_interval_ratio
        for ratio in ratios
    )
    result.update(
        {
            "status": "Normal" if normal else "Suspicious",
            "baseline_pitch_px": query.baseline_pitch_px,
            "interval_pitch_ratios": [
                round(ratio, 3) for ratio in ratios
            ],
            "reason": None if normal else "interval_outside_normal_ratio",
        }
    )
    return result
