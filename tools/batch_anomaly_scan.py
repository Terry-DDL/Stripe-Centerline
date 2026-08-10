"""Batch anomaly sweep using the frozen Windows v1.1 delivery detector.

This module is an audit/export tool only.  It calls
``stage3_desktop_runtime.run_frozen_stage3`` for every sampled point and never
changes detector configuration or decisions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import traceback
from typing import Iterable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from config import INTERACTIVE_CONFIG  # noqa: E402
from interactive_pipeline import calculate_interactive_roi_bounds  # noqa: E402
from tools import stage3_desktop_runtime as formal_runtime  # noqa: E402


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}
EXCLUDED_NAME_PARTS = {
    "adjacent_stripes_result",
    "binary",
    "black_mask",
    "brightened",
    "candidate",
    "centerline",
    "close_delta",
    "contact_sheet",
    "crop",
    "debug",
    "generated",
    "mask",
    "original_interactive_result",
    "otsu",
    "overlay",
    "result",
    "roi_gray",
    "screenshot",
    "synthetic",
    "threshold",
    "vertical_close",
    "vote",
}

SWEEP_FIELDS = [
    "scan_id",
    "image",
    "image_path",
    "image_width",
    "image_height",
    "x",
    "y",
    "grid_spacing_px",
    "algorithm_revision",
    "configuration_checksum",
    "current_result",
    "success",
    "primary_failure_reason",
    "earliest_failure_reason",
    "left_center",
    "right_center",
    "left_distance",
    "right_distance",
    "reference_relationship",
    "recovery_triggered",
    "recovery_success",
    "recovery_mode",
    "recovery_reason",
    "suspicious_success",
    "why_suspicious",
    "separator_status",
    "separator_failure_reason",
    "accepted_separator_count",
    "separator_candidate_count",
    "basin_candidate_count",
    "verified_basin_count",
    "role_hypothesis_count",
    "diagnostic_failure_reasons_json",
    "pitch_confidence",
    "pitch_px",
    "pitch_joint_decision",
    "pitch_harmonic_ambiguity",
    "pitch_rejects_geometry",
    "geometry_spacing_ratios_json",
    "separator_diagnostics_json",
    "basin_diagnostics_json",
    "arbitration_diagnostics_json",
    "recovery_diagnostics_json",
    "raw_diagnostics_path",
    "raw_diagnostics_line",
]

REVIEW_FIELDS = [
    "review_id",
    "image",
    "x",
    "y",
    "current_result",
    "left_center",
    "right_center",
    "category",
    "failure_reason",
    "suspicious",
    "normal_control",
    "why_selected",
    "why_suspicious",
    "overlay_path",
    "human_label",
    "human_notes",
]


@dataclass(frozen=True)
class ScanSettings:
    grid_spacing_px: int = 100
    target_points: int = 2800
    max_review_points: int = 40
    normal_control_points: int = 20
    review_min_distance_px: int = 150
    max_review_per_image: int = 4


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _compact_json(value) -> str:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_original_input(path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    lowered_name = path.name.lower()
    if any(
        part.lower() in {"outputs", "output", "docs", ".git"}
        for part in path.parts
    ):
        return False
    return not any(token in lowered_name for token in EXCLUDED_NAME_PARTS)


def discover_real_images(search_roots: Iterable[Path]) -> list[Path]:
    """Find plausible raw inputs and remove byte-identical duplicates."""

    candidates = []
    for root in search_roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            continue
        candidates.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and _looks_like_original_input(path)
        )

    unique_by_digest: dict[str, Path] = {}
    for path in sorted(set(candidates), key=lambda item: str(item).lower()):
        digest = _file_digest(path)
        current = unique_by_digest.get(digest)
        if current is None:
            unique_by_digest[digest] = path
            continue
        current_is_delivery = PROJECT_ROOT in current.parents
        candidate_is_delivery = PROJECT_ROOT in path.parents
        if candidate_is_delivery and not current_is_delivery:
            unique_by_digest[digest] = path
    return sorted(unique_by_digest.values(), key=lambda item: item.name.lower())


def grid_points(
    image_shape: tuple[int, ...],
    spacing_px: int,
    point_limit: int,
) -> list[tuple[int, int]]:
    """Return full-ROI-safe points on a regular lattice, evenly thinned."""

    height, width = image_shape[:2]
    margin_x = INTERACTIVE_CONFIG.roi_half_width_px + 20
    margin_y = INTERACTIVE_CONFIG.roi_half_height_px + 20
    if width <= 2 * margin_x or height <= 2 * margin_y:
        return []
    xs = list(range(margin_x, width - margin_x, spacing_px))
    ys = list(range(margin_y, height - margin_y, spacing_px))
    lattice = [(x, y) for y in ys for x in xs]
    if len(lattice) <= point_limit:
        return lattice
    indexes = np.linspace(0, len(lattice) - 1, point_limit, dtype=int)
    return [lattice[index] for index in indexes]


def _walk_rejection_reasons(value) -> list[str]:
    reasons = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"reason", "unavailable_reason"} and isinstance(nested, str):
                if nested and nested not in {"trigger_not_met", "not_applicable"}:
                    reasons.append(nested)
            elif key == "rejection_reasons" and isinstance(nested, list):
                reasons.extend(item for item in nested if isinstance(item, str))
            else:
                reasons.extend(_walk_rejection_reasons(nested))
    elif isinstance(value, list):
        for nested in value:
            reasons.extend(_walk_rejection_reasons(nested))
    return list(dict.fromkeys(reasons))


def _selected_geometry(result: dict, roi_x0: int) -> dict:
    hypothesis = result.get("final_hypothesis") or {}
    geometry = hypothesis.get("geometry") or {}
    basins = geometry.get("basins") or {}
    output = {}
    for side in ("left", "right"):
        basin = basins.get(side) or {}
        center_roi = basin.get("center_x_at_reference_roi")
        center_global = basin.get("center_x_at_reference_global")
        if center_global is None and center_roi is not None:
            center_global = roi_x0 + float(center_roi)
        output[side] = {
            "center": center_global,
            "distance": geometry.get(f"{side}_distance_px"),
        }
    return output


def suspicious_reasons(result: dict) -> list[str]:
    """Use existing formal diagnostics to flag successes for human review."""

    if not result.get("success"):
        return []
    debug = result.get("debug", {})
    recovery = debug.get("local_dark_valley_recovery") or {}
    hypothesis = result.get("final_hypothesis") or {}
    pitch = hypothesis.get("pitch_evidence") or debug.get("raw_pitch_result") or {}
    relation = debug.get("reference_relation") or {}
    reasons = []
    if recovery.get("triggered") or recovery.get("success"):
        reasons.append("recovery_triggered")
    if hypothesis.get("local_contrast_recovery"):
        reasons.append("local_contrast_recovery")
    if pitch.get("rejects_geometry"):
        reasons.append("pitch_rejects_geometry")
    if pitch.get("confidence") not in {None, "high"}:
        reasons.append(f"pitch_confidence_{pitch.get('confidence')}")
    if pitch.get("joint_decision") not in {None, "high_pitch_supports_geometry"}:
        reasons.append(f"pitch_{pitch.get('joint_decision')}")
    reference_relation = hypothesis.get("reference_relation") or relation.get("relation")
    if reference_relation not in {None, "inside_basin"}:
        reasons.append(f"reference_{reference_relation}")
    selected_basins = set((hypothesis.get("basin_ids") or {}).values())
    for basin in (debug.get("basin_graph") or {}).get("basin_candidates", []):
        if basin.get("basin_id") in selected_basins and basin.get("rejection_reasons"):
            reasons.append("selected_basin_has_intermediate_rejection")
    return list(dict.fromkeys(reasons))


def flatten_result(
    scan_id: str,
    image_path: Path,
    image_shape: tuple[int, ...],
    x: int,
    y: int,
    spacing_px: int,
    bounds,
    result: dict,
    diagnostics_path: Path,
    diagnostics_line: int,
) -> dict:
    """Flatten key formal diagnostics while retaining a raw-record pointer."""

    debug = result.get("debug", {})
    separator = debug.get("separator_result") or {}
    graph = debug.get("basin_graph") or {}
    relation = debug.get("reference_relation") or {}
    pitch = debug.get("raw_pitch_result") or {}
    hypothesis = result.get("final_hypothesis") or {}
    if hypothesis.get("pitch_evidence"):
        pitch = hypothesis["pitch_evidence"]
    recovery = debug.get("local_dark_valley_recovery") or {}
    arbitration = separator.get("arbitration_debug") or {}
    role_hypotheses = arbitration.get("role_hypotheses") or []
    all_reasons = _walk_rejection_reasons(debug)
    primary_reason = result.get("unavailable_reason") if not result.get("success") else None
    earliest_reason = all_reasons[0] if all_reasons else primary_reason
    geometry = _selected_geometry(result, bounds.x0_global)
    suspicious = suspicious_reasons(result)
    accepted_paths = separator.get("accepted_paths") or []
    candidate_paths = separator.get("candidate_paths") or separator.get("paths") or []
    compact_separator = {
        key: separator.get(key)
        for key in (
            "status",
            "unavailable_reason",
            "reference_x_roi",
            "reference_y_roi",
            "selected_candidate_ids",
            "selected_separator_ids",
        )
        if key in separator
    }
    compact_basin = {
        "ordered_separator_ids": graph.get("ordered_separator_ids"),
        "verified_basin_ids": [
            item.get("basin_id") for item in graph.get("verified_basins", [])
        ],
        "candidate_summaries": [
            {
                key: item.get(key)
                for key in (
                    "basin_id",
                    "left_separator_id",
                    "right_separator_id",
                    "verified",
                    "width_at_reference_px",
                    "dark_contrast_band_fraction",
                    "rejection_reasons",
                )
                if key in item
            }
            for item in graph.get("basin_candidates", [])
        ],
    }
    return {
        "scan_id": scan_id,
        "image": image_path.name,
        "image_path": str(image_path),
        "image_width": image_shape[1],
        "image_height": image_shape[0],
        "x": x,
        "y": y,
        "grid_spacing_px": spacing_px,
        "algorithm_revision": result.get("algorithm_revision", ""),
        "configuration_checksum": result.get("configuration_checksum", ""),
        "current_result": "success" if result.get("success") else "unavailable",
        "success": bool(result.get("success")),
        "primary_failure_reason": primary_reason or "",
        "earliest_failure_reason": earliest_reason or "",
        "left_center": geometry["left"]["center"],
        "right_center": geometry["right"]["center"],
        "left_distance": geometry["left"]["distance"],
        "right_distance": geometry["right"]["distance"],
        "reference_relationship": (
            hypothesis.get("reference_relation")
            or relation.get("relation")
            or relation.get("status")
            or ""
        ),
        "recovery_triggered": bool(recovery.get("triggered") or recovery.get("success")),
        "recovery_success": bool(recovery.get("success")),
        "recovery_mode": recovery.get("mode", ""),
        "recovery_reason": recovery.get("reason", ""),
        "suspicious_success": bool(suspicious),
        "why_suspicious": ";".join(suspicious),
        "separator_status": separator.get("status", ""),
        "separator_failure_reason": separator.get("unavailable_reason", ""),
        "accepted_separator_count": len(accepted_paths),
        "separator_candidate_count": len(candidate_paths),
        "basin_candidate_count": len(graph.get("basin_candidates", [])),
        "verified_basin_count": len(graph.get("verified_basins", [])),
        "role_hypothesis_count": len(role_hypotheses),
        "diagnostic_failure_reasons_json": _compact_json(all_reasons),
        "pitch_confidence": pitch.get("confidence", ""),
        "pitch_px": pitch.get("diagnostic_pitch_px"),
        "pitch_joint_decision": pitch.get("joint_decision", ""),
        "pitch_harmonic_ambiguity": bool(pitch.get("harmonic_ambiguity")),
        "pitch_rejects_geometry": bool(pitch.get("rejects_geometry")),
        "geometry_spacing_ratios_json": _compact_json(
            pitch.get("geometry_spacing_ratios", [])
        ),
        "separator_diagnostics_json": _compact_json(compact_separator),
        "basin_diagnostics_json": _compact_json(compact_basin),
        "arbitration_diagnostics_json": _compact_json(arbitration),
        "recovery_diagnostics_json": _compact_json(recovery),
        "raw_diagnostics_path": str(diagnostics_path),
        "raw_diagnostics_line": diagnostics_line,
    }


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("_")
    return stem or "image"


def scan_one_image(task: dict) -> list[dict]:
    """Worker: load one image once and call the formal detector per point."""

    image_path = Path(task["image_path"])
    diagnostics_path = Path(task["diagnostics_path"])
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Unable to read image: {image_path}")
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with gzip.open(diagnostics_path, "wt", encoding="utf-8") as stream:
        for index, (x, y) in enumerate(task["points"], start=1):
            bounds = calculate_interactive_roi_bounds(
                image.shape,
                int(x),
                int(y),
                INTERACTIVE_CONFIG,
            )
            try:
                result = formal_runtime.run_frozen_stage3(
                    image,
                    {"x": int(x), "y": int(y)},
                    bounds,
                )
            except Exception as error:  # preserve a batch-wide audit trail
                reason = f"formal_pipeline_exception:{type(error).__name__}"
                result = {
                    "status": "unavailable",
                    "success": False,
                    "unavailable_reason": reason,
                    "debug": {
                        "formal_pipeline_exception": {
                            "type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        },
                        "separator_result": {},
                        "basin_graph": {
                            "basin_candidates": [],
                            "verified_basins": [],
                        },
                        "reference_relation": {},
                        "raw_pitch_result": {},
                    },
                }
            scan_id = f"{task['image_id']}-{index:04d}"
            raw_record = {
                "scan_id": scan_id,
                "image": image_path.name,
                "image_path": str(image_path),
                "x": int(x),
                "y": int(y),
                "roi_bounds_global": formal_runtime.bounds_to_dict(bounds),
                "formal_pipeline": "tools.stage3_desktop_runtime.run_frozen_stage3",
                "stage3_result": result,
            }
            stream.write(_compact_json(raw_record) + "\n")
            records.append(
                flatten_result(
                    scan_id,
                    image_path,
                    image.shape,
                    int(x),
                    int(y),
                    task["spacing_px"],
                    bounds,
                    result,
                    diagnostics_path,
                    index,
                )
            )
    return records


def _far_enough(candidate: dict, selected: list[dict], distance_px: int) -> bool:
    minimum_squared = distance_px * distance_px
    for existing in selected:
        if existing["image_path"] != candidate["image_path"]:
            continue
        dx = int(existing["x"]) - int(candidate["x"])
        dy = int(existing["y"]) - int(candidate["y"])
        if dx * dx + dy * dy < minimum_squared:
            return False
    return True


def _diverse_pick(
    candidates: list[dict],
    limit: int,
    selected: list[dict],
    group_key,
    settings: ScanSettings,
) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        groups[str(group_key(candidate))].append(candidate)
    for items in groups.values():
        items.sort(key=lambda item: (item["image"], int(item["y"]), int(item["x"])))
    image_counts = Counter(item["image_path"] for item in selected)
    picked = []
    while len(picked) < limit:
        progressed = False
        for group in sorted(groups, key=lambda key: (len(groups[key]), key)):
            items = groups[group]
            for index, candidate in enumerate(items):
                if image_counts[candidate["image_path"]] >= settings.max_review_per_image:
                    continue
                if not _far_enough(candidate, selected + picked, settings.review_min_distance_px):
                    continue
                picked.append(candidate)
                image_counts[candidate["image_path"]] += 1
                del items[index]
                progressed = True
                break
            if len(picked) >= limit:
                break
        if not progressed:
            break
    return picked


def select_manual_review(records: list[dict], settings: ScanSettings) -> list[dict]:
    """Select failures, suspicious successes, and normal success controls."""

    normal = [
        item for item in records
        if item["success"] and not item["suspicious_success"]
    ]
    unavailable = [item for item in records if not item["success"]]
    suspicious = [item for item in records if item["suspicious_success"]]
    selected: list[dict] = []
    controls = _diverse_pick(
        normal,
        min(settings.normal_control_points, settings.max_review_points),
        selected,
        lambda item: f"{item['image']}:{int(item['x']) // 500}:{int(item['y']) // 500}",
        settings,
    )
    for item in controls:
        item["review_category"] = "normal_success_control"
        item["why_selected"] = "stratified normal-success control across image and position"
    selected.extend(controls)

    remaining = settings.max_review_points - len(selected)
    unavailable_target = min(10, math.ceil(remaining / 2))
    unavailable_picks = _diverse_pick(
        unavailable,
        unavailable_target,
        selected,
        lambda item: item["primary_failure_reason"] or "unknown_failure",
        settings,
    )
    for item in unavailable_picks:
        reason = item["primary_failure_reason"] or "unknown_failure"
        item["review_category"] = f"unavailable:{reason}"
        item["why_selected"] = "representative unavailable category with image/spatial diversity"
    selected.extend(unavailable_picks)

    remaining = settings.max_review_points - len(selected)
    suspicious_picks = _diverse_pick(
        suspicious,
        min(10, remaining),
        selected,
        lambda item: item["why_suspicious"] or "other_suspicious_diagnostic",
        settings,
    )
    for item in suspicious_picks:
        item["review_category"] = "suspicious_success"
        item["why_selected"] = "representative success with existing diagnostic warning"
    selected.extend(suspicious_picks)

    remaining = settings.max_review_points - len(selected)
    if remaining:
        leftovers = [
            item for item in unavailable + suspicious
            if item not in selected
        ]
        extra = _diverse_pick(
            leftovers,
            remaining,
            selected,
            lambda item: (
                item["primary_failure_reason"]
                or item["why_suspicious"]
                or item["image"]
            ),
            settings,
        )
        for item in extra:
            if item["success"]:
                item["review_category"] = "suspicious_success"
                item["why_selected"] = "additional diverse suspicious-success diagnostic"
            else:
                reason = item["primary_failure_reason"] or "unknown_failure"
                item["review_category"] = f"unavailable:{reason}"
                item["why_selected"] = "additional diverse unavailable category"
        selected.extend(extra)
    return selected[: settings.max_review_points]


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_overlays(selected: list[dict], output_dir: Path) -> None:
    """Generate only the final review overlays through the formal renderer."""

    overlay_dir = output_dir / "manual_review" / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    cached_images: dict[str, np.ndarray] = {}
    for index, item in enumerate(selected, start=1):
        image_path = item["image_path"]
        image = cached_images.get(image_path)
        if image is None:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Unable to read image: {image_path}")
            cached_images[image_path] = image
        reference = {"x": int(item["x"]), "y": int(item["y"])}
        bounds = calculate_interactive_roi_bounds(
            image.shape, reference["x"], reference["y"], INTERACTIVE_CONFIG
        )
        try:
            stage3 = formal_runtime.run_frozen_stage3(image, reference, bounds)
            rendered = formal_runtime.build_desktop_result(
                image,
                item["image"],
                reference,
                bounds,
                stage3,
                overlay_dir,
            )
            overlay = rendered.debug_images[
                formal_runtime.FORMAL_OVERLAY_FILENAME
            ]
        except Exception:
            overlay = formal_runtime._draw_formal_overlay(  # noqa: SLF001
                image,
                reference,
                bounds,
                None,
            )
        review_id = f"MR{index:03d}"
        overlay_path = overlay_dir / f"{review_id}_{_safe_stem(Path(item['image']))}_{item['x']}_{item['y']}.png"
        if not cv2.imwrite(
            str(overlay_path),
            overlay,
        ):
            raise RuntimeError(f"Unable to write overlay: {overlay_path}")
        item["review_id"] = review_id
        item["overlay_path"] = str(overlay_path)


def _review_rows(selected: list[dict]) -> list[dict]:
    rows = []
    for item in selected:
        rows.append(
            {
                "review_id": item["review_id"],
                "image": item["image"],
                "x": item["x"],
                "y": item["y"],
                "current_result": item["current_result"],
                "left_center": item["left_center"],
                "right_center": item["right_center"],
                "category": item["review_category"],
                "failure_reason": item["primary_failure_reason"],
                "suspicious": bool(item["suspicious_success"]),
                "normal_control": item["review_category"] == "normal_success_control",
                "why_selected": item["why_selected"],
                "why_suspicious": item["why_suspicious"],
                "overlay_path": item["overlay_path"],
                "human_label": "",
                "human_notes": "",
            }
        )
    return rows


def write_summary(
    output_path: Path,
    images: list[Path],
    records: list[dict],
    selected: list[dict],
    settings: ScanSettings,
) -> None:
    unavailable_counts = Counter(
        item["primary_failure_reason"] or "unknown_failure"
        for item in records
        if not item["success"]
    )
    success_count = sum(bool(item["success"]) for item in records)
    suspicious_count = sum(bool(item["suspicious_success"]) for item in records)
    controls = sum(
        item.get("review_category") == "normal_success_control"
        for item in selected
    )
    lines = [
        "# Batch anomaly sweep summary",
        "",
        f"- Formal pipeline: `tools.stage3_desktop_runtime.run_frozen_stage3`",
        f"- Images scanned: {len(images)}",
        f"- Sweep points: {len(records)}",
        f"- Grid lattice spacing: {settings.grid_spacing_px} px",
        f"- Success: {success_count}",
        f"- Unavailable: {len(records) - success_count}",
        f"- Suspicious success: {suspicious_count}",
        f"- Manual review points: {len(selected)}",
        f"- Normal success controls: {controls}",
        "",
        "## Scanned images",
        "",
    ]
    lines.extend(f"- `{path}`" for path in images)
    lines.extend(["", "## Unavailable categories", ""])
    lines.extend(
        f"- `{reason}`: {count}"
        for reason, count in unavailable_counts.most_common()
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scan(
    image_roots: list[Path],
    output_dir: Path,
    settings: ScanSettings,
    workers: int,
) -> dict:
    images = discover_real_images(image_roots)
    if not images:
        raise RuntimeError("No eligible real input images were found")
    if not 1000 <= settings.target_points <= 3000:
        raise ValueError("target_points must be between 1000 and 3000")
    if settings.max_review_points > 40:
        raise ValueError("max_review_points must not exceed 40")

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    base_budget, remainder = divmod(settings.target_points, len(images))
    tasks = []
    for image_index, image_path in enumerate(images):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        budget = base_budget + (1 if image_index < remainder else 0)
        points = grid_points(
            image.shape,
            settings.grid_spacing_px,
            budget,
        )
        image_id = f"I{image_index + 1:02d}_{_safe_stem(image_path)}"
        tasks.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "points": points,
                "spacing_px": settings.grid_spacing_px,
                "diagnostics_path": str(diagnostics_dir / f"{image_id}.jsonl.gz"),
            }
        )

    records = []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        futures = {pool.submit(scan_one_image, task): task for task in tasks}
        for future in as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda item: (item["image"].lower(), int(item["y"]), int(item["x"])))
    _write_csv(output_dir / "sweep_results.csv", SWEEP_FIELDS, records)

    selected = select_manual_review(records, settings)
    generate_overlays(selected, output_dir)
    review_rows = _review_rows(selected)
    _write_csv(
        output_dir / "manual_review" / "review.csv",
        REVIEW_FIELDS,
        review_rows,
    )
    write_summary(
        output_dir / "sweep_summary.md",
        images,
        records,
        selected,
        settings,
    )
    manifest = {
        "formal_pipeline": "tools.stage3_desktop_runtime.run_frozen_stage3",
        "formal_algorithm_revisions": sorted(
            {
                item["algorithm_revision"]
                for item in records
                if item["algorithm_revision"]
            }
        ),
        "formal_configuration_checksums": sorted(
            {
                item["configuration_checksum"]
                for item in records
                if item["configuration_checksum"]
            }
        ),
        "settings": settings.__dict__,
        "images": [str(path) for path in images],
        "sweep_points": len(records),
        "manual_review_points": len(selected),
        "overlay_count": len(list((output_dir / "manual_review" / "overlays").glob("*.png"))),
    }
    (output_dir / "scan_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "images": images,
        "records": records,
        "selected": selected,
        "output_dir": output_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-root",
        action="append",
        type=Path,
        dest="image_roots",
        help="Image directory to search; may be supplied more than once",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "batch_anomaly_scan",
    )
    parser.add_argument("--target-points", type=int, default=2800)
    parser.add_argument("--grid-spacing", type=int, default=100)
    parser.add_argument("--max-review", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_archive_images = PROJECT_ROOT.parent / "Stripe Centerline" / "images"
    image_roots = args.image_roots or [
        PROJECT_ROOT / "images",
        default_archive_images,
    ]
    settings = ScanSettings(
        grid_spacing_px=args.grid_spacing,
        target_points=args.target_points,
        max_review_points=args.max_review,
    )
    result = run_scan(
        image_roots,
        args.output_dir.resolve(),
        settings,
        args.workers,
    )
    success = sum(bool(item["success"]) for item in result["records"])
    print(
        _compact_json(
            {
                "images": len(result["images"]),
                "points": len(result["records"]),
                "success": success,
                "unavailable": len(result["records"]) - success,
                "suspicious_success": sum(
                    bool(item["suspicious_success"])
                    for item in result["records"]
                ),
                "manual_review": len(result["selected"]),
                "output_dir": str(result["output_dir"]),
            }
        )
    )


if __name__ == "__main__":
    main()
