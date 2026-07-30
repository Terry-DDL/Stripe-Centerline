"""Default-off desktop shadow runner for cross-image correction v1.1.

This bridge is intentionally output-only.  It runs the frozen offline v1.1
experiment with the same image, reference, and ROI as the formal Stage 3.1
request, but it cannot change the formal result or the desktop UI.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from tools import cross_image_correction_prototype_v1_1 as experiment


INTEGRATION_REVISION = "cross_image_correction_v1_1_shadow_v1"
FROZEN_ALGORITHM_COMMIT = (
    "c93dd086ebd5c5bb3ac614cd84ecd1a39cfb7274"
)
RESULT_FILENAME = "cross_image_correction_v1_1_shadow_result.json"
OVERLAY_FILENAME = "cross_image_correction_v1_1_shadow_overlay.png"


def _bounds_dict(bounds) -> dict:
    """Convert the formal desktop ROI object to the frozen dict contract."""

    return {
        "x0": int(bounds.x0_global),
        "y0": int(bounds.y0_global),
        "x1": int(bounds.x1_global),
        "y1": int(bounds.y1_global),
    }


def _sha256_array(image: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(image)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _formal_stage3_summary(stage3_result: dict | None) -> dict | None:
    """Keep comparison provenance without using Stage 3.1 as a fallback."""

    if stage3_result is None:
        return None
    hypothesis = stage3_result.get("final_hypothesis")
    return {
        "success": bool(stage3_result.get("success")),
        "status": stage3_result.get("status"),
        "unavailable_reason": stage3_result.get("unavailable_reason"),
        "algorithm_revision": stage3_result.get("algorithm_revision"),
        "configuration_checksum": stage3_result.get(
            "configuration_checksum"
        ),
        "reference_relation": (
            hypothesis.get("reference_relation")
            if hypothesis is not None
            else None
        ),
        "basin_ids": (
            hypothesis.get("basin_ids")
            if hypothesis is not None
            else None
        ),
        "separator_ids": (
            hypothesis.get("separator_ids")
            if hypothesis is not None
            else None
        ),
        "separator_sequence": (
            hypothesis.get("separator_sequence")
            if hypothesis is not None
            else None
        ),
    }


def _atomic_write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_ok, encoded = cv2.imencode(".png", image)
    if not encoded_ok:
        raise OSError(f"could not encode {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)


def run_cross_image_experiment_shadow(
    image_gray: np.ndarray,
    image_name: str,
    reference_global: dict,
    bounds,
    output_dir: Path,
    run_id: str,
    formal_stage3_result: dict | None,
    runner: Callable = experiment.run_joint_case_v1_1,
    refiner: Callable = experiment.refine_success_geometry,
    overlay_builder: Callable = experiment._draw_smoke_overlay,
) -> dict:
    """Run and persist one isolated v1.1 shadow result.

    The formal Stage 3.1 result is copied only into comparison provenance.  It
    is never supplied to the experimental detector and cannot restore an
    unavailable v1.1 result.
    """

    roi_bounds = _bounds_dict(bounds)
    reference = {
        "x": int(reference_global["x"]),
        "y": int(reference_global["y"]),
    }
    result = runner(
        image_gray,
        reference,
        roi_bounds,
        "vertical",
    )
    refinement = refiner(
        result,
        image_gray,
        roi_bounds,
        "vertical",
    )
    case = {
        "sample_id": run_id,
        "image_name": image_name,
        "reference_global": reference,
        "roi_bounds_global": roi_bounds,
        "direction": "vertical",
    }
    overlay = overlay_builder(
        image_gray,
        case,
        result,
        refinement,
    )

    shadow_dir = Path(output_dir)
    overlay_path = shadow_dir / OVERLAY_FILENAME
    result_path = shadow_dir / RESULT_FILENAME
    record = {
        "integration_revision": INTEGRATION_REVISION,
        "frozen_algorithm_commit": FROZEN_ALGORITHM_COMMIT,
        "algorithm_revision": experiment.ALGORITHM_REVISION,
        "configuration_checksum": experiment.configuration_checksum(),
        "run_id": run_id,
        "recorded_at": datetime.now().astimezone().isoformat(),
        "image_name": image_name,
        "image_shape": list(image_gray.shape),
        "image_gray_sha256": _sha256_array(image_gray),
        "reference_global": reference,
        "roi_bounds_global": roi_bounds,
        "formal_stage3_comparison": _formal_stage3_summary(
            formal_stage3_result
        ),
        "shadow": {
            "success": bool(result["success"]),
            "status": result["status"],
            "unavailable_reason": result["unavailable_reason"],
            "result": result,
            "center_refinement": refinement,
        },
        "contract": {
            "formal_result_used_as_inference_input": False,
            "legacy_fallback_allowed": False,
            "ui_mutation_allowed": False,
            "formal_timing_mutation_allowed": False,
        },
        "overlay_path": str(overlay_path),
    }
    _atomic_write_png(overlay_path, overlay)
    experiment.v1.atomic_write_json(result_path, record)
    return record
