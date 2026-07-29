"""Shadow-only bridge from the desktop pipeline to frozen Stage 3.1.

The formal desktop result is an input to provenance logging only.  This
module never changes that result and never exposes the shadow decision to the
desktop result view.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import threading
from typing import Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import basin_graph_joint_prototype as joint  # noqa: E402
from tools import basin_shadow_artifacts as artifacts  # noqa: E402


INTEGRATION_REVISION = "basin_shadow_integration_v1"
DEFAULT_LOG_ROOT = (
    PROJECT_ROOT / "outputs" / "basin_shadow_integration_v1" / "desktop"
)
PER_CLICK_FILENAME = "basin_shadow_result.json"
JSONL_FILENAME = "click_comparisons.jsonl"
CSV_FILENAME = "click_comparisons.csv"
SHADOW_OVERLAY_FILENAME = artifacts.SHADOW_OVERLAY_FILENAME
ACCEPTANCE_GEOMETRY_REVISION = artifacts.ACCEPTANCE_GEOMETRY_REVISION
_acceptance_geometry = artifacts.build_acceptance_geometry
_draw_shadow_overlay = artifacts.draw_shadow_overlay
_atomic_write_png = artifacts.atomic_write_png
_LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class ShadowLogConfig:
    """Logging settings only; no frozen algorithm parameter is exposed."""

    log_root: Path = DEFAULT_LOG_ROOT
    per_click_filename: str = PER_CLICK_FILENAME
    jsonl_filename: str = JSONL_FILENAME
    csv_filename: str = CSV_FILENAME


def _sha256_array(image: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(image)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bounds_dict(bounds) -> dict:
    return {
        "x0": int(bounds.x0_global),
        "y0": int(bounds.y0_global),
        "x1": int(bounds.x1_global),
        "y1": int(bounds.y1_global),
    }


def _production_summary(production_result) -> dict:
    report = production_result.report
    interactive = report["interactive_result"]

    def side_summary(side: str) -> dict | None:
        result = interactive.get(side)
        if result is None:
            return None
        return {
            "center_x_global": result.get("center_x_global"),
            "distance_to_click_px": result.get(
                "distance_to_click_px"
            ),
            "track_id": result.get("track_id"),
        }

    return {
        "success": bool(interactive["success"]),
        "failure_reasons": list(
            interactive.get("failure_reasons", [])
        ),
        "algorithm_revision": report.get("algorithm_revision"),
        "left": side_summary("left"),
        "right": side_summary("right"),
        "stripe_spacing_px": interactive.get("stripe_spacing_px"),
    }


def _pitch_provenance(shadow_result: dict) -> dict:
    hypothesis = shadow_result.get("final_hypothesis")
    if hypothesis is not None:
        evidence = hypothesis["pitch_evidence"]
        return {
            "algorithm_revision": evidence["algorithm_revision"],
            "configuration_checksum": evidence[
                "configuration_checksum"
            ],
            "confidence": evidence["confidence"],
            "success_eligible": evidence["success_eligible"],
            "diagnostic_pitch_px": evidence[
                "diagnostic_pitch_px"
            ],
            "usable_pitch_px": evidence["usable_pitch_px"],
            "harmonic_ambiguity": evidence["harmonic_ambiguity"],
            "joint_decision": evidence["joint_decision"],
            "geometry_spacing_ratios": evidence[
                "geometry_spacing_ratios"
            ],
        }
    raw_pitch = shadow_result["debug"]["raw_pitch_result"]
    return {
        "algorithm_revision": raw_pitch["algorithm_revision"],
        "configuration_checksum": raw_pitch[
            "configuration_checksum"
        ],
        "confidence": raw_pitch["confidence"],
        "success_eligible": raw_pitch["success_eligible"],
        "diagnostic_pitch_px": raw_pitch["diagnostic_pitch_px"],
        "usable_pitch_px": raw_pitch["usable_pitch_px"],
        "harmonic_ambiguity": raw_pitch["harmonic_ambiguity"],
        "joint_decision": "no_joint_hypothesis",
        "geometry_spacing_ratios": [],
    }


def _formal_geometry_summary(
    shadow_result: dict,
    reference_global: dict,
    roi_bounds_global: dict,
) -> dict | None:
    hypothesis = shadow_result.get("final_hypothesis")
    if hypothesis is None:
        return None
    geometry = hypothesis["geometry"]
    left_center_roi = geometry["basins"]["left"][
        "center_x_at_reference_roi"
    ]
    right_center_roi = geometry["basins"]["right"][
        "center_x_at_reference_roi"
    ]
    left_center_global = geometry["basins"]["left"].get(
        "center_x_at_reference_global",
        roi_bounds_global["x0"] + left_center_roi,
    )
    right_center_global = geometry["basins"]["right"].get(
        "center_x_at_reference_global",
        roi_bounds_global["x0"] + right_center_roi,
    )
    return {
        "reference_relation": hypothesis["reference_relation"],
        "left_basin_center_x_global": left_center_global,
        "right_basin_center_x_global": right_center_global,
        "left_distance_to_reference_px": (
            reference_global["x"] - left_center_global
        ),
        "right_distance_to_reference_px": (
            right_center_global - reference_global["x"]
        ),
        "basin_center_spacing_px": (
            right_center_global - left_center_global
        ),
        "basin_ids": hypothesis["basin_ids"],
        "basin_sequence": hypothesis["basin_sequence"],
        "separator_ids": hypothesis.get("separator_ids"),
        "separator_sequence": hypothesis["separator_sequence"],
        "clicked_separator_id": hypothesis.get(
            "clicked_separator_id"
        ),
        "shared_boundary": hypothesis.get("shared_boundary"),
        "atomic": hypothesis["atomic"],
    }


def _shadow_summary(
    shadow_result: dict,
    reference_global: dict,
    roi_bounds_global: dict,
) -> dict:
    relation = shadow_result["debug"]["reference_relation"]
    return {
        "success": bool(shadow_result["success"]),
        "status": shadow_result["status"],
        "unavailable_reason": shadow_result["unavailable_reason"],
        "algorithm_revision": shadow_result["algorithm_revision"],
        "configuration_checksum": shadow_result[
            "configuration_checksum"
        ],
        "geometry": _formal_geometry_summary(
            shadow_result,
            reference_global,
            roi_bounds_global,
        ),
        "pitch_provenance": _pitch_provenance(shadow_result),
        "rejection_evidence": {
            "reference_relation_status": relation["status"],
            "reference_relation": relation["relation"],
            "reference_relation_reason": relation[
                "unavailable_reason"
            ],
            "inherited_safety_conflicts": relation.get(
                "inherited_safety_conflicts",
                [],
            ),
            "path_order_conflicts": shadow_result["debug"][
                "basin_graph"
            ]["path_order_conflicts"],
        },
    }


def _input_contract(
    image_gray: np.ndarray,
    reference_global: dict,
    roi_bounds_global: dict,
    production_result,
) -> dict:
    production_bounds = _bounds_dict(production_result.bounds_global)
    production_reference = {
        "x": int(production_result.click_x_global),
        "y": int(production_result.click_y_global),
    }
    report_shape = tuple(
        production_result.report.get("image", {}).get(
            "shape",
            image_gray.shape,
        )
    )
    return {
        "same_image_shape": (
            tuple(image_gray.shape) == tuple(report_shape[:2])
        ),
        "same_reference": production_reference == reference_global,
        "same_roi_bounds": production_bounds == roi_bounds_global,
        "production_reference_global": production_reference,
        "production_roi_bounds_global": production_bounds,
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _csv_row(record: dict) -> dict:
    production = record["production"]
    shadow = record.get("shadow") or {}
    geometry = shadow.get("geometry") or {}
    pitch = shadow.get("pitch_provenance") or {}
    return {
        "timestamp": record["timestamp"],
        "image_name": record["image_name"],
        "reference_x": record["reference_global"]["x"],
        "reference_y": record["reference_global"]["y"],
        "production_success": production["success"],
        "shadow_success": shadow.get("success"),
        "shadow_status": shadow.get("status"),
        "shadow_unavailable_reason": shadow.get(
            "unavailable_reason"
        ),
        "reference_relation": geometry.get("reference_relation"),
        "left_center_x_global": geometry.get(
            "left_basin_center_x_global"
        ),
        "right_center_x_global": geometry.get(
            "right_basin_center_x_global"
        ),
        "left_distance_px": geometry.get(
            "left_distance_to_reference_px"
        ),
        "right_distance_px": geometry.get(
            "right_distance_to_reference_px"
        ),
        "basin_ids": json.dumps(
            geometry.get("basin_ids"),
            sort_keys=True,
        ),
        "separator_sequence": json.dumps(
            geometry.get("separator_sequence"),
        ),
        "pitch_confidence": pitch.get("confidence"),
        "usable_pitch_px": pitch.get("usable_pitch_px"),
        "pitch_joint_decision": pitch.get("joint_decision"),
        "integration_status": record["integration_status"],
    }


def _append_comparison_logs(
    record: dict,
    config: ShadowLogConfig,
) -> None:
    config.log_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = config.log_root / config.jsonl_filename
    csv_path = config.log_root / config.csv_filename
    row = _csv_row(record)
    with _LOG_LOCK:
        with jsonl_path.open("a", encoding="utf-8") as output_file:
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open(
            "a",
            encoding="utf-8",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=list(row),
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def run_shadow_and_log(
    image_gray: np.ndarray,
    image_name: str,
    production_result,
    output_dir: Path,
    log_config: ShadowLogConfig = ShadowLogConfig(),
    shadow_runner: Callable = joint.run_joint_case,
) -> dict:
    """Run Stage 3.1 without changing the formal desktop result."""

    reference_global = {
        "x": int(production_result.click_x_global),
        "y": int(production_result.click_y_global),
    }
    roi_bounds_global = _bounds_dict(production_result.bounds_global)
    report_hash_before = _canonical_sha256(production_result.report)
    image_hash_before = _sha256_array(image_gray)
    roi = image_gray[
        roi_bounds_global["y0"] : roi_bounds_global["y1"],
        roi_bounds_global["x0"] : roi_bounds_global["x1"],
    ]
    input_contract = _input_contract(
        image_gray,
        reference_global,
        roi_bounds_global,
        production_result,
    )
    record = {
        "integration_revision": INTEGRATION_REVISION,
        "timestamp": datetime.now().astimezone().isoformat(),
        "image_name": image_name,
        "image_sha256": image_hash_before,
        "roi_sha256": _sha256_array(roi),
        "reference_global": reference_global,
        "roi_bounds_global": roi_bounds_global,
        "input_contract": input_contract,
        "production": _production_summary(production_result),
        "shadow": None,
        "shadow_overlay": {
            "status": "not_generated",
            "filename": SHADOW_OVERLAY_FILENAME,
            "path": None,
            "error": None,
        },
        "integration_status": "pending",
        "integration_error": None,
    }
    if not all(
        input_contract[key]
        for key in (
            "same_image_shape",
            "same_reference",
            "same_roi_bounds",
        )
    ):
        record["integration_status"] = "input_contract_mismatch"
    else:
        try:
            shadow_result = shadow_runner(
                image_gray,
                reference_global,
                roi_bounds_global,
                "vertical",
                joint.DEFAULT_CONFIG,
            )
            record["shadow"] = _shadow_summary(
                shadow_result,
                reference_global,
                roi_bounds_global,
            )
            record["integration_status"] = "completed"
            try:
                acceptance = _acceptance_geometry(
                    shadow_result,
                    reference_global,
                    roi_bounds_global,
                    record["shadow"]["geometry"],
                )
                record["shadow"][
                    "acceptance_geometry"
                ] = acceptance
                overlay_path = output_dir / SHADOW_OVERLAY_FILENAME
                _atomic_write_png(
                    overlay_path,
                    _draw_shadow_overlay(image_gray, acceptance),
                )
                record["shadow_overlay"] = {
                    "status": "written",
                    "filename": SHADOW_OVERLAY_FILENAME,
                    "path": str(overlay_path),
                    "error": None,
                }
            except Exception as caught_error:
                record["shadow_overlay"]["status"] = "artifact_error"
                record["shadow_overlay"]["error"] = {
                    "type": type(caught_error).__name__,
                    "message": str(caught_error),
                }
        except Exception as caught_error:  # Shadow must never break formal UI.
            record["integration_status"] = "shadow_error"
            record["integration_error"] = {
                "type": type(caught_error).__name__,
                "message": str(caught_error),
            }

    record["formal_report_unchanged"] = (
        report_hash_before
        == _canonical_sha256(production_result.report)
    )
    record["image_unchanged"] = (
        image_hash_before == _sha256_array(image_gray)
    )
    per_click_path = output_dir / log_config.per_click_filename
    _atomic_write_json(per_click_path, record)
    _append_comparison_logs(record, log_config)
    return record
