"""Export read-only candidate evidence for adjacency investigations."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import cv2
import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_S10_CASE_IDS = ("S10-11", "S10-12", "S10-14")
SAMPLE2_CASES = (
    {
        "id": "sample2_662_1046",
        "image_path": "images/Sample 2.bmp",
        "click": {"x": 662, "y": 1046},
        "ground_truth": {
            "valid_pair": True,
            "click_classification": "black_stripe",
            "left_center_x": 623.5,
            "clicked_center_x": 662.5,
            "right_center_x": 701.0,
        },
    },
    {
        "id": "sample2_662_1065",
        "image_path": "images/Sample 2.bmp",
        "click": {"x": 662, "y": 1065},
        "ground_truth": {
            "valid_pair": True,
            "click_classification": "black_stripe",
            "left_center_x": 623.5,
            "clicked_center_x": 662.5,
            "right_center_x": 701.0,
        },
    },
    {
        "id": "sample2_701_996",
        "image_path": "images/Sample 2.bmp",
        "click": {"x": 701, "y": 996},
        "ground_truth": {
            "valid_pair": True,
            "click_classification": "black_stripe",
            "left_center_x": 662.5,
            "clicked_center_x": 701.0,
            "right_center_x": 738.5,
        },
    },
)
CENTER_TOLERANCE_PX = 3.0


def git_provenance(project_root: Path) -> dict:
    """Return commit and dirty state for the code being diagnosed."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_output = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "git_commit": commit,
        "git_dirty": bool(dirty_output.strip()),
    }


def ensure_project_module_origins(project_root: Path) -> dict:
    """Reject cached pipeline modules from a different Git worktree."""

    src_dir = (project_root / "src").resolve()
    origins = {}
    for module_name in (
        "config",
        "interactive_pipeline",
        "pitch_reference",
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise RuntimeError(
                f"{module_name} has no file-backed module origin"
            )
        origin = Path(module_file).resolve()
        if origin.parent != src_dir:
            raise RuntimeError(
                "candidate diagnostics require a fresh process: "
                f"{module_name} is loaded from {origin}, expected {src_dir}"
            )
        origins[module_name] = str(origin)
    return origins


def brighten_with_diagnostics(
    image_gray: np.ndarray,
    brightness_offset: int,
) -> tuple[np.ndarray, float]:
    """Apply one non-negative offset and report the true clipping ratio."""

    if brightness_offset < 0:
        raise ValueError("brightness_offset must not be negative")
    expanded = image_gray.astype(np.int16) + brightness_offset
    clipped_pixel_ratio = float(
        np.count_nonzero(expanded > 255) / expanded.size
    )
    return (
        np.clip(expanded, 0, 255).astype(np.uint8),
        clipped_pixel_ratio,
    )


def morphology_statistics(stages) -> dict:
    """Summarize threshold and morphology pixels without changing them."""

    pixel_count = int(stages.threshold_binary.size)
    threshold_white = int(np.count_nonzero(stages.threshold_binary))
    closed_white = int(np.count_nonzero(stages.vertical_close))
    close_added = int(np.count_nonzero(stages.close_delta))
    close_removed = int(
        np.count_nonzero(
            (stages.threshold_binary != 0)
            & (stages.vertical_close == 0)
        )
    )
    changed = int(
        np.count_nonzero(
            stages.threshold_binary != stages.vertical_close
        )
    )
    black_mask = int(np.count_nonzero(stages.black_mask))
    return {
        "pixel_count": pixel_count,
        "threshold_white_pixel_count": threshold_white,
        "threshold_white_pixel_ratio": threshold_white / pixel_count,
        "post_close_white_pixel_count": closed_white,
        "post_close_white_pixel_ratio": closed_white / pixel_count,
        "close_added_white_pixel_count": close_added,
        "close_added_white_pixel_ratio": close_added / pixel_count,
        "close_removed_white_pixel_count": close_removed,
        "close_removed_white_pixel_ratio": close_removed / pixel_count,
        "threshold_to_close_changed_pixel_count": changed,
        "threshold_to_close_changed_pixel_ratio": changed / pixel_count,
        "post_close_black_mask_pixel_count": black_mask,
        "post_close_black_mask_pixel_ratio": black_mask / pixel_count,
    }


def _load_cases(
    project_root: Path,
    suite: str,
    case_ids: tuple[str, ...],
) -> tuple[dict, ...]:
    if suite == "sample2":
        return SAMPLE2_CASES
    truth_path = (
        project_root
        / "tests"
        / "data"
        / "stripe_09_10_ground_truth.json"
    )
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    cases_by_id = {case["id"]: case for case in truth["cases"]}
    missing = [case_id for case_id in case_ids if case_id not in cases_by_id]
    if missing:
        raise ValueError(f"unknown ground-truth case ids: {missing}")
    return tuple(cases_by_id[case_id] for case_id in case_ids)


def _mapped_x_global(candidate, track_x: float, y_roi: float, bounds) -> float:
    mapped = candidate.inverse_rotation_matrix @ np.array(
        [track_x, y_roi, 1.0],
        dtype=np.float64,
    )
    return float(bounds.x0_global + mapped[0])


def _track_summary(candidate, track, click_y_roi: int, bounds) -> dict:
    return {
        "track_id": track.track_id,
        "seed_x_roi": track.seed_x_roi,
        "center_x_roi": round(float(track.center_x_roi), 6),
        "center_x_global_at_click_y": round(
            _mapped_x_global(
                candidate,
                track.center_x_roi,
                click_y_roi,
                bounds,
            ),
            6,
        ),
        "median_width_px": round(float(track.median_width_px), 6),
        "assigned_row_count": track.assigned_row_count,
        "valid_row_count": track.valid_row_count,
        "valid_row_ratio": round(float(track.valid_row_ratio), 6),
        "retention_ratio": round(float(track.retention_ratio), 6),
        "crossing_assigned_count": track.crossing_assigned_count,
        "crossing_valid_count": track.crossing_valid_count,
        "side": track.side,
        "eligible": bool(track.eligible),
        "rejection_reasons": list(track.rejection_reasons),
        "legacy_analysis_selected": bool(track.selected),
        "legacy_analysis_selection_reason": track.selection_reason,
    }


def _post_close_run_evidence(
    candidate,
    expected_x_global: float,
    bounds,
    tolerance: float,
) -> dict:
    matching_runs = []
    for run in candidate.analysis.all_runs:
        center_global = _mapped_x_global(
            candidate,
            run.center_x_roi,
            run.y_roi,
            bounds,
        )
        if abs(center_global - expected_x_global) <= tolerance:
            matching_runs.append((run, center_global))

    rejection_counts = Counter(
        (
            "accepted"
            if run.accepted
            else run.rejection_reason or "unspecified"
        )
        for run, _center in matching_runs
    )
    return {
        "run_count": len(matching_runs),
        "distinct_row_count": len(
            {run.y_roi for run, _center in matching_runs}
        ),
        "accepted_run_count": rejection_counts["accepted"],
        "filter_result_counts": dict(sorted(rejection_counts.items())),
        "median_center_x_global": (
            None
            if not matching_runs
            else round(
                float(
                    np.median(
                        [center for _run, center in matching_runs]
                    )
                ),
                6,
            )
        ),
        "median_width_px": (
            None
            if not matching_runs
            else round(
                float(
                    np.median(
                        [run.width_px for run, _center in matching_runs]
                    )
                ),
                6,
            )
        ),
    }


def _preprocessing_column_evidence(
    candidate,
    expected_x_global: float,
    bounds,
    tolerance: float,
) -> dict:
    """Measure black support before and after close at an expected center."""

    if candidate.geometry != "original":
        return {
            "available": False,
            "reason": "column_evidence_limited_to_original_geometry",
        }
    center_x_roi = int(
        round(expected_x_global - bounds.x0_global)
    )
    width = candidate.stages.threshold_binary.shape[1]
    if not 0 <= center_x_roi < width:
        return {
            "available": False,
            "reason": "expected_center_outside_roi",
        }

    def coverage(x0: int, x1: int) -> dict:
        threshold_black = np.any(
            candidate.stages.threshold_binary[:, x0:x1] == 0,
            axis=1,
        )
        post_close_black = np.any(
            candidate.stages.black_mask[:, x0:x1] != 0,
            axis=1,
        )
        lost = threshold_black & ~post_close_black
        gained = ~threshold_black & post_close_black
        row_count = len(threshold_black)
        return {
            "x0_roi": x0,
            "x1_roi_exclusive": x1,
            "row_count": row_count,
            "threshold_black_row_count": int(
                np.count_nonzero(threshold_black)
            ),
            "threshold_black_row_ratio": float(
                np.count_nonzero(threshold_black) / row_count
            ),
            "post_close_black_row_count": int(
                np.count_nonzero(post_close_black)
            ),
            "post_close_black_row_ratio": float(
                np.count_nonzero(post_close_black) / row_count
            ),
            "black_rows_removed_by_close": int(np.count_nonzero(lost)),
            "black_rows_added_by_close": int(np.count_nonzero(gained)),
        }

    half_width = max(1, int(np.ceil(tolerance)))
    corridor_x0 = max(0, center_x_roi - half_width)
    corridor_x1 = min(width, center_x_roi + half_width + 1)
    return {
        "available": True,
        "expected_center_x_roi": center_x_roi,
        "center_column": coverage(center_x_roi, center_x_roi + 1),
        "tolerance_corridor": coverage(corridor_x0, corridor_x1),
    }


def _expected_track_evidence(
    candidate,
    expected_x_global: float,
    click_y_roi: int,
    bounds,
    tolerance: float,
) -> dict:
    tracks = [
        _track_summary(candidate, track, click_y_roi, bounds)
        for track in candidate.analysis.tracks
    ]
    matching = [
        track
        for track in tracks
        if abs(
            track["center_x_global_at_click_y"] - expected_x_global
        )
        <= tolerance
    ]
    closest = min(
        tracks,
        key=lambda track: abs(
            track["center_x_global_at_click_y"] - expected_x_global
        ),
        default=None,
    )
    return {
        "expected_center_x_global": expected_x_global,
        "matching_track_ids": [track["track_id"] for track in matching],
        "eligible_matching_track_ids": [
            track["track_id"] for track in matching if track["eligible"]
        ],
        "matching_tracks": matching,
        "closest_track": closest,
        "post_close_run_evidence": _post_close_run_evidence(
            candidate,
            expected_x_global,
            bounds,
            tolerance,
        ),
        "preprocessing_column_evidence": (
            _preprocessing_column_evidence(
                candidate,
                expected_x_global,
                bounds,
                tolerance,
            )
        ),
    }


def _role_eligibility_evidence(
    candidate,
    expected_evidence: dict,
    click_x_roi: int,
    click_y_roi: int,
    processing_config,
    interactive_config,
) -> dict:
    """Describe support thresholds separately for each interactive role."""

    clicked_pixel_is_black = bool(
        candidate.stages.black_mask[click_y_roi, click_x_roi]
    )
    outer_min_support = getattr(
        interactive_config,
        "outer_neighbor_recovery_min_support_ratio",
        None,
    )
    if outer_min_support is None:
        outer_min_support = (
            interactive_config.neighbor_recovery_min_support_ratio
        )
    roles = {}
    for label, evidence in expected_evidence.items():
        role_tracks = []
        for track in evidence["matching_tracks"]:
            stable = (
                track["valid_row_ratio"]
                >= processing_config.min_stripe_support_ratio
                and track["rejection_reasons"]
                in ([], ["center_on_reference"])
            )
            role_tracks.append(
                {
                    "track_id": track["track_id"],
                    "stable_selection_eligible": stable,
                    "meets_clicked_min_support": (
                        track["valid_row_ratio"]
                        >= processing_config.min_clicked_track_support_ratio
                    ),
                    "meets_outer_recovery_min_support": (
                        track["valid_row_ratio"] >= outer_min_support
                    ),
                    "covers_click_column": (
                        abs(track["center_x_roi"] - click_x_roi)
                        <= max(track["median_width_px"] / 2.0, 2.0)
                    ),
                }
            )
        roles[label] = role_tracks
    return {
        "clicked_pixel_is_black": clicked_pixel_is_black,
        "clicked_min_support_ratio": (
            processing_config.min_clicked_track_support_ratio
        ),
        "stable_min_support_ratio": (
            processing_config.min_stripe_support_ratio
        ),
        "outer_recovery_min_support_ratio": (
            outer_min_support
        ),
        "roles": roles,
    }


def _candidate_loss_stage(candidate_row: dict) -> dict:
    """Locate a truth pair loss without collapsing role-specific evidence."""

    if candidate_row["selection_matches_ground_truth"]:
        verification = candidate_row["adjacency_verification"]
        if (
            verification is not None
            and verification["status"] == "verified"
        ):
            return {"stage": "verified_candidate"}
        return {
            "stage": "adjacency_verification",
            "reason": (
                None if verification is None else verification["reason"]
            ),
        }

    missing_labels = [
        label
        for label, evidence in candidate_row[
            "expected_track_evidence"
        ].items()
        if not evidence["matching_track_ids"]
    ]
    if missing_labels:
        label_stages = {}
        for label in missing_labels:
            evidence = candidate_row["expected_track_evidence"][label]
            column = evidence["preprocessing_column_evidence"]
            if not column["available"]:
                stage = "preprocessing_or_track_generation"
            elif (
                column["center_column"]["threshold_black_row_count"]
                == 0
            ):
                stage = "threshold_representation"
            elif (
                column["center_column"]["post_close_black_row_count"]
                == 0
            ):
                stage = "morphology_close"
            elif (
                evidence["post_close_run_evidence"][
                    "accepted_run_count"
                ]
                == 0
            ):
                stage = "run_filtering"
            else:
                stage = "track_generation"
            label_stages[label] = stage
        unique_stages = set(label_stages.values())
        return {
            "stage": (
                next(iter(unique_stages))
                if len(unique_stages) == 1
                else "multiple_preselection_stages"
            ),
            "missing_labels": missing_labels,
            "label_stages": label_stages,
        }

    selected_matches = candidate_row[
        "selected_role_matches_ground_truth"
    ]
    role_evidence = candidate_row["role_eligibility"]
    contributing = []
    for label in ("left", "right"):
        if label not in role_evidence["roles"]:
            continue
        if any(
            track["stable_selection_eligible"]
            or track["meets_outer_recovery_min_support"]
            for track in role_evidence["roles"][label]
        ):
            continue
        contributing.append(f"{label}_outer_support_eligibility")

    if "clicked" in selected_matches and not selected_matches["clicked"]:
        return {
            "stage": "clicked_track_association",
            "clicked_pixel_is_black": role_evidence[
                "clicked_pixel_is_black"
            ],
            "contributing_stages": contributing,
        }
    if not selected_matches.get("left", True) or not selected_matches.get(
        "right",
        True,
    ):
        return {
            "stage": (
                "outer_track_support_eligibility"
                if contributing
                else "outer_neighbor_selection_or_recovery"
            ),
            "contributing_stages": contributing,
        }
    return {"stage": "candidate_combination"}


def _selected_track_summary(
    candidate,
    track,
    click_y_roi: int,
    bounds,
) -> dict | None:
    if track is None:
        return None
    return _track_summary(candidate, track, click_y_roi, bounds)


def _counterfactual_correct_selection(
    pipeline,
    candidate,
    case: dict,
    expected_evidence: dict,
    pitch_map,
    processing_config,
    interactive_config,
) -> dict:
    """Evaluate existing truth-matched tracks without changing selection."""

    if candidate.geometry != "original":
        return {
            "available": False,
            "reason": "counterfactual_limited_to_original_geometry",
        }
    if not hasattr(pipeline, "_build_adjacency_verification"):
        return {
            "available": False,
            "reason": "adjacency_verification_not_available_in_revision",
        }
    matched_tracks = {}
    tracks_by_id = {
        track.track_id: track for track in candidate.analysis.tracks
    }
    for label, evidence in expected_evidence.items():
        matching_ids = evidence["matching_track_ids"]
        if not matching_ids:
            return {
                "available": False,
                "reason": f"missing_truth_matched_{label}_track",
            }
        matched_tracks[label] = tracks_by_id[matching_ids[0]]

    truth = case["ground_truth"]
    selection = pipeline.InteractiveStripeSelection(
        click_classification=truth["click_classification"],
        clicked_track=matched_tracks.get("clicked"),
        left_track=matched_tracks.get("left"),
        right_track=matched_tracks.get("right"),
        success=True,
        failure_reasons=(),
        warning_flags=(),
    )
    guarded, neighbor_consistency = (
        pipeline._apply_neighbor_consistency_guard(
            selection,
            candidate.analysis,
            processing_config,
            interactive_config,
        )
    )
    pitch_guard = pipeline.evaluate_pitch_guard(
        guarded,
        case["click"]["x"],
        case["click"]["y"],
        pitch_map,
        interactive_config,
    )
    topology = pipeline.evaluate_grayscale_topology(
        candidate.stages.image_gray,
        guarded,
        candidate.inverse_rotation_matrix,
        interactive_config,
    )
    quality = pipeline.build_candidate_quality(
        guarded,
        candidate.analysis,
        neighbor_consistency,
        pitch_guard,
        (
            topology.report
            if interactive_config.enable_grayscale_topology_rejection
            else {"status": "Not considered"}
        ),
        candidate.analysis.x_ref_roi,
        processing_config,
    )
    verification = pipeline._build_adjacency_verification(
        guarded,
        neighbor_consistency,
        quality,
    )
    verification["arbitration_fields"] = {
        "adjacency_verification_status": verification["status"],
        "combined_pitch_status": quality["combined_pitch_status"],
        "grayscale_topology_status": quality[
            "grayscale_topology_status"
        ],
        "quality_success": quality["success"],
        "minimum_valid_row_ratio": quality[
            "minimum_valid_row_ratio"
        ],
        "minimum_retention_ratio": quality[
            "minimum_retention_ratio"
        ],
    }
    return {
        "available": True,
        "matched_track_ids": {
            label: track.track_id
            for label, track in matched_tracks.items()
        },
        "neighbor_consistency": neighbor_consistency,
        "pitch_guard": pitch_guard,
        "grayscale_topology": topology.report,
        "quality_success": quality["success"],
        "quality_hard_invalid_reasons": quality[
            "hard_invalid_reasons"
        ],
        "adjacency_verification": verification,
    }


def _preprocessing_diagnostics(
    pipeline,
    candidate,
    processing_config,
    interactive_config,
) -> dict:
    stages = candidate.stages
    blurred = pipeline.gaussian_blur_roi(
        stages.image_gray,
        processing_config,
    )
    otsu_threshold = None
    adaptive_block_size = None
    adaptive_c = None
    if candidate.threshold_method == "otsu":
        otsu_threshold = float(
            cv2.threshold(
                blurred,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[0]
        )
    else:
        adaptive_block_size = pipeline._effective_adaptive_block_size(
            stages.image_gray.shape,
            interactive_config,
        )[0]
        adaptive_c = float(interactive_config.adaptive_threshold_c)
    return {
        "otsu_threshold": otsu_threshold,
        "adaptive_effective_block_size": adaptive_block_size,
        "adaptive_c": adaptive_c,
        "morphology": morphology_statistics(stages),
    }


def _candidate_diagnostics(
    pipeline,
    candidate,
    case: dict,
    click_y_roi: int,
    bounds,
    processing_config,
    interactive_config,
    pitch_map,
    tolerance: float,
) -> dict:
    truth = case["ground_truth"]
    expected = {
        label: truth.get(f"{label}_center_x")
        for label in ("left", "clicked", "right")
        if truth.get(f"{label}_center_x") is not None
    }
    expected_evidence = {
        label: _expected_track_evidence(
            candidate,
            expected_x,
            click_y_roi,
            bounds,
            tolerance,
        )
        for label, expected_x in expected.items()
    }
    selection = candidate.selection
    selected = {
        "left": _selected_track_summary(
            candidate,
            selection.left_track,
            click_y_roi,
            bounds,
        ),
        "clicked": _selected_track_summary(
            candidate,
            selection.clicked_track,
            click_y_roi,
            bounds,
        ),
        "right": _selected_track_summary(
            candidate,
            selection.right_track,
            click_y_roi,
            bounds,
        ),
    }
    selected_role_matches = {
        label: (
            selected[label] is not None
            and abs(
                selected[label]["center_x_global_at_click_y"]
                - expected_x
            )
            <= tolerance
        )
        for label, expected_x in expected.items()
    }
    selection_matches_truth = all(selected_role_matches.values())
    track_set_exists = all(
        evidence["matching_track_ids"]
        for evidence in expected_evidence.values()
    )
    eligible_track_set_exists = all(
        evidence["eligible_matching_track_ids"]
        for evidence in expected_evidence.values()
    )
    adjacency = getattr(candidate, "adjacency_verification", None)
    near_click_tracks = [
        _track_summary(candidate, track, click_y_roi, bounds)
        for track in candidate.analysis.tracks
    ]
    interactive_roles_by_track_id = {}
    for label, track in selected.items():
        if track is None:
            continue
        interactive_roles_by_track_id.setdefault(
            track["track_id"],
            [],
        ).append(label)
    for track in near_click_tracks:
        track["interactive_selected_roles"] = (
            interactive_roles_by_track_id.get(track["track_id"], [])
        )
    role_eligibility = _role_eligibility_evidence(
        candidate,
        expected_evidence,
        candidate.analysis.x_ref_roi,
        click_y_roi,
        processing_config,
        interactive_config,
    )
    row = {
        "candidate_id": (
            f"{candidate.geometry}_{candidate.threshold_method}"
        ),
        "geometry": candidate.geometry,
        "threshold_method": candidate.threshold_method,
        "preprocessing": _preprocessing_diagnostics(
            pipeline,
            candidate,
            processing_config,
            interactive_config,
        ),
        "track_candidate_count": len(candidate.analysis.tracks),
        "track_candidate_count_definition": (
            "all analysis tracks inside the configured search radius"
        ),
        "run_filter_counts": candidate.analysis.run_filter_counts,
        "near_click_tracks": near_click_tracks,
        "expected_track_evidence": expected_evidence,
        "counterfactual_correct_selection": (
            _counterfactual_correct_selection(
                pipeline,
                candidate,
                case,
                expected_evidence,
                pitch_map,
                processing_config,
                interactive_config,
            )
        ),
        "correct_track_set_exists": track_set_exists,
        "eligible_correct_track_set_exists": eligible_track_set_exists,
        "role_eligibility": role_eligibility,
        "selected_tracks": selected,
        "selected_track_ids": {
            label: None if track is None else track["track_id"]
            for label, track in selected.items()
        },
        "selected_role_matches_ground_truth": selected_role_matches,
        "selection_matches_ground_truth": selection_matches_truth,
        "raw_success": bool(selection.success),
        "selection_failure_reasons": list(selection.failure_reasons),
        "selection_warning_flags": list(selection.warning_flags),
        "quality_success": bool(candidate.quality["success"]),
        "quality_hard_invalid_reasons": list(
            candidate.quality["hard_invalid_reasons"]
        ),
        "neighbor_consistency": candidate.neighbor_consistency,
        "adjacency_verification": adjacency,
        "grayscale_topology": candidate.grayscale_topology,
    }
    row["candidate_loss_stage"] = _candidate_loss_stage(row)
    return row


def diagnose_loss_stage(
    candidate_rows: list[dict],
    formal_success: bool = False,
) -> dict:
    """Name the earliest observed loss stage for a confirmed track set."""

    if formal_success:
        return {
            "stage": "not_applicable_formal_success",
            "candidate_ids": [
                row["candidate_id"]
                for row in candidate_rows
                if row["selection_matches_ground_truth"]
            ],
        }

    if candidate_rows and all(
        "candidate_loss_stage" in row for row in candidate_rows
    ):
        candidate_stages = {
            row["candidate_id"]: row["candidate_loss_stage"]
            for row in candidate_rows
        }
        unique_stages = {
            stage["stage"] for stage in candidate_stages.values()
        }
        if len(unique_stages) == 1:
            return {
                "stage": next(iter(unique_stages)),
                "candidate_ids": list(candidate_stages),
                "candidate_stages": candidate_stages,
            }
        return {
            "stage": "candidate_specific",
            "candidate_stages": candidate_stages,
        }

    matching_selections = [
        row for row in candidate_rows
        if row["selection_matches_ground_truth"]
    ]
    verified = [
        row
        for row in matching_selections
        if (
            row["adjacency_verification"] is not None
            and row["adjacency_verification"]["status"] == "verified"
        )
    ]
    if verified:
        return {
            "stage": "verified_candidate_exists",
            "candidate_ids": [
                row["candidate_id"] for row in verified
            ],
        }
    if matching_selections:
        return {
            "stage": "adjacency_verification",
            "candidate_ids": [
                row["candidate_id"] for row in matching_selections
            ],
        }

    eligible_sets = [
        row
        for row in candidate_rows
        if row["eligible_correct_track_set_exists"]
    ]
    if eligible_sets:
        return {
            "stage": "candidate_combination",
            "candidate_ids": [
                row["candidate_id"] for row in eligible_sets
            ],
        }

    generated_sets = [
        row
        for row in candidate_rows
        if row["correct_track_set_exists"]
    ]
    if generated_sets:
        return {
            "stage": "track_eligibility",
            "candidate_ids": [
                row["candidate_id"] for row in generated_sets
            ],
        }
    return {
        "stage": "preprocessing_or_track_generation",
        "candidate_ids": [],
    }


def _empty_debug_image(*_args, **_kwargs) -> np.ndarray:
    return np.zeros((1, 1), dtype=np.uint8)


def _capture_case(
    pipeline,
    case: dict,
    image_variant: np.ndarray,
    clipped_pixel_ratio: float,
    pitch_map,
    processing_config,
    interactive_config,
    tolerance: float,
) -> dict:
    captured = []
    original_builder = pipeline._build_detection_candidate

    def capture_candidate(*args, **kwargs):
        candidate = original_builder(*args, **kwargs)
        captured.append(candidate)
        return candidate

    patch_names = (
        "save_debug_image",
        "_save_report",
        "_create_original_overlay",
        "_create_interactive_roi_result",
        "_create_interactive_candidates_debug",
        "_create_interactive_votes_debug",
        "create_pitch_reference_debug",
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    pipeline,
                    "_build_detection_candidate",
                    side_effect=capture_candidate,
                )
            )
            for name in patch_names:
                if not hasattr(pipeline, name):
                    continue
                if name in ("save_debug_image", "_save_report"):
                    stack.enter_context(patch.object(pipeline, name))
                else:
                    stack.enter_context(
                        patch.object(
                            pipeline,
                            name,
                            side_effect=_empty_debug_image,
                        )
                    )
            result = pipeline.run_interactive_case(
                image_variant,
                case["click"]["x"],
                case["click"]["y"],
                Path(temporary_directory),
                processing_config,
                interactive_config,
                image_name=case["image_path"],
                pitch_reference_map=pitch_map,
            )

    candidate_rows = [
        _candidate_diagnostics(
            pipeline,
            candidate,
            case,
            result.click_y_roi,
            result.bounds_global,
            processing_config,
            interactive_config,
            pitch_map,
            tolerance,
        )
        for candidate in captured
    ]
    report = result.report
    return {
        "case_id": case["id"],
        "image_path": case["image_path"],
        "click": case["click"],
        "ground_truth": case["ground_truth"],
        "brightness_offset": 0,
        "clipped_pixel_ratio": clipped_pixel_ratio,
        "algorithm_revision": report.get("algorithm_revision"),
        "candidate_count": len(candidate_rows),
        "candidate_count_definition": (
            "evaluated geometry and threshold combinations"
        ),
        "loss_stage": diagnose_loss_stage(
            candidate_rows,
            formal_success=report["interactive_result"]["success"],
        ),
        "formal_result": report["interactive_result"],
        "candidate_arbitration": report["candidate_arbitration"],
        "shadow_arbitration": report.get("shadow_arbitration"),
        "adjacency_arbitration": report.get("adjacency_arbitration"),
        "candidates": candidate_rows,
    }


def run_diagnostics(
    project_root: Path,
    asset_root: Path,
    suite: str,
    case_ids: tuple[str, ...],
    brightness_offset: int,
    tolerance: float = CENTER_TOLERANCE_PX,
) -> dict:
    """Run one project revision in a fresh process and capture candidates."""

    src_dir = project_root / "src"
    ensure_project_module_origins(project_root)
    sys.path.insert(0, str(src_dir))
    import config as config_module  # noqa: PLC0415
    import interactive_pipeline as pipeline  # noqa: PLC0415
    import pitch_reference  # noqa: PLC0415
    module_origins = ensure_project_module_origins(project_root)
    CONFIG = config_module.CONFIG
    INTERACTIVE_CONFIG = config_module.INTERACTIVE_CONFIG

    cases = _load_cases(project_root, suite, case_ids)
    images = {}
    variants = {}
    clipping_ratios = {}
    pitch_maps = {}
    case_reports = []
    for case in cases:
        image_path = case["image_path"]
        if image_path not in images:
            image = cv2.imread(
                str(asset_root / image_path),
                cv2.IMREAD_GRAYSCALE,
            )
            if image is None:
                raise FileNotFoundError(asset_root / image_path)
            images[image_path] = image
            variant, clipping_ratio = brighten_with_diagnostics(
                image,
                brightness_offset,
            )
            variants[image_path] = variant
            clipping_ratios[image_path] = clipping_ratio
            pitch_maps[image_path] = (
                pitch_reference.build_pitch_reference_map(
                    variant,
                    CONFIG,
                    INTERACTIVE_CONFIG,
                )
            )
        case_report = _capture_case(
            pipeline,
            case,
            variants[image_path],
            clipping_ratios[image_path],
            pitch_maps[image_path],
            CONFIG,
            INTERACTIVE_CONFIG,
            tolerance,
        )
        case_report["brightness_offset"] = brightness_offset
        case_reports.append(case_report)

    provenance = git_provenance(project_root)
    return {
        "schema_version": 1,
        "diagnostics_schema_revision": "adjacency_diagnostics_v1",
        "behavior": "read_only_capture_no_pipeline_behavior_change",
        "project_root": str(project_root),
        "asset_root": str(asset_root),
        "module_origins": module_origins,
        **provenance,
        "suite": suite,
        "brightness_offset": brightness_offset,
        "center_tolerance_px": tolerance,
        "case_count": len(case_reports),
        "cases": case_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=SCRIPT_ROOT,
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="Optional root for untracked/ignored source images.",
    )
    parser.add_argument(
        "--suite",
        choices=("sample2", "s10"),
        required=True,
    )
    parser.add_argument(
        "--case-ids",
        default=",".join(DEFAULT_S10_CASE_IDS),
    )
    parser.add_argument("--brightness-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    asset_root = (
        project_root
        if args.asset_root is None
        else args.asset_root.resolve()
    )
    case_ids = tuple(
        case_id.strip()
        for case_id in args.case_ids.split(",")
        if case_id.strip()
    )
    report = run_diagnostics(
        project_root,
        asset_root,
        args.suite,
        case_ids,
        args.brightness_offset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "git_commit": report["git_commit"],
                "git_dirty": report["git_dirty"],
                "suite": report["suite"],
                "case_count": report["case_count"],
                "loss_stages": {
                    case["case_id"]: case["loss_stage"]
                    for case in report["cases"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
