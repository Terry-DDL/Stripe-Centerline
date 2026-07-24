"""Run the local interactive pipeline against confirmed review cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import CONFIG, INTERACTIVE_CONFIG  # noqa: E402
from ground_truth_evaluation import (  # noqa: E402
    evaluate_ground_truth_predictions,
)
from interactive_pipeline import run_interactive_case  # noqa: E402
from pitch_reference import build_pitch_reference_map  # noqa: E402


def run_regression(truth: dict, output_dir: Path) -> dict:
    """Evaluate all cases while reusing one pitch map per source image."""

    images = {}
    pitch_maps = {}
    predictions = {}
    for case in truth["cases"]:
        relative_image_path = case["image_path"]
        if relative_image_path not in images:
            image = cv2.imread(
                str(PROJECT_ROOT / relative_image_path),
                cv2.IMREAD_GRAYSCALE,
            )
            if image is None:
                raise FileNotFoundError(relative_image_path)
            images[relative_image_path] = image
            pitch_maps[relative_image_path] = build_pitch_reference_map(
                image,
                CONFIG,
                INTERACTIVE_CONFIG,
            )
        result = run_interactive_case(
            images[relative_image_path],
            case["click"]["x"],
            case["click"]["y"],
            output_dir / "cases" / case["id"],
            CONFIG,
            INTERACTIVE_CONFIG,
            image_name=relative_image_path,
            pitch_reference_map=pitch_maps[relative_image_path],
        )
        interactive = result.report["interactive_result"]
        rotation = result.report["rotation_shadow"]
        left = interactive["left"]
        right = interactive["right"]
        predictions[case["id"]] = {
            "success": interactive["success"],
            "click_classification": result.report["click"][
                "classification"
            ],
            "clicked_center_x": (
                None
                if interactive.get("clicked") is None
                else interactive["clicked"]["center_x_global"]
            ),
            "left_center_x": (
                None if left is None else left["center_x_global"]
            ),
            "right_center_x": (
                None if right is None else right["center_x_global"]
            ),
            "threshold_method": interactive.get(
                "threshold_method",
                "otsu",
            ),
            "geometry": interactive.get(
                "geometry",
                rotation["detection_space"],
            ),
            "pitch_status": interactive["pitch_guard"]["status"],
        }
    metrics = evaluate_ground_truth_predictions(truth, predictions)
    return {"metrics": metrics, "predictions": predictions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--truth",
        type=Path,
        default=(
            PROJECT_ROOT
            / "tests"
            / "data"
            / "stripe_09_10_ground_truth.json"
        ),
    )
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    output_dir = (
        PROJECT_ROOT / "outputs" / "ground_truth_regression" / args.label
    )
    result = run_regression(truth, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "regression_report.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
