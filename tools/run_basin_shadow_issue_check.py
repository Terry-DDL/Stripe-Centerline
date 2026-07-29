"""Run the desktop formal path and Stage 3.1 shadow on ten known issues."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CONFIG, INTERACTIVE_CONFIG  # noqa: E402
from interactive_pipeline import run_interactive_case  # noqa: E402
from pitch_reference import build_pitch_reference_map  # noqa: E402
from tools.basin_shadow_integration import (  # noqa: E402
    ShadowLogConfig,
    run_shadow_and_log,
)
from tools.desktop_app import decode_grayscale_image  # noqa: E402


ISSUE_COORDINATES = (
    (661, 921),
    (482, 704),
    (1044, 581),
    (1086, 525),
    (1044, 378),
    (507, 686),
    (150, 1829),
    (1143, 1239),
    (275, 1507),
    (546, 721),
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "basin_shadow_integration_v1"
    / "sample2_issues"
)


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


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_issue_check(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    image_path = PROJECT_ROOT / "images" / "Sample 2.bmp"
    image_gray = decode_grayscale_image(image_path.read_bytes())
    pitch_reference_map = build_pitch_reference_map(
        image_gray,
        CONFIG,
        INTERACTIVE_CONFIG,
    )
    records = []
    comparison_config = ShadowLogConfig(
        log_root=output_dir / "comparison_logs"
    )
    for click_x, click_y in ISSUE_COORDINATES:
        case_dir = output_dir / f"point_{click_x}_{click_y}"
        production_result = run_interactive_case(
            image_gray,
            click_x,
            click_y,
            case_dir / "formal",
            CONFIG,
            INTERACTIVE_CONFIG,
            image_name=image_path.name,
            pitch_reference_map=pitch_reference_map,
        )
        records.append(
            run_shadow_and_log(
                image_gray,
                image_path.name,
                production_result,
                case_dir,
                comparison_config,
            )
        )

    status_pairs = {}
    for record in records:
        shadow = record.get("shadow") or {}
        key = (
            f"formal_{record['production']['success']}"
            f"__shadow_{shadow.get('success')}"
        )
        status_pairs[key] = status_pairs.get(key, 0) + 1
    report = {
        "report_revision": "basin_shadow_sample2_issue_check_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "access_policy": "known_issues_only_no_heldout",
        "image_name": image_path.name,
        "issue_coordinates": [
            {"x": x, "y": y} for x, y in ISSUE_COORDINATES
        ],
        **_git_provenance(),
        "metrics": {
            "sample_count": len(records),
            "integration_completed": sum(
                record["integration_status"] == "completed"
                for record in records
            ),
            "input_contract_mismatch": sum(
                record["integration_status"]
                == "input_contract_mismatch"
                for record in records
            ),
            "shadow_error": sum(
                record["integration_status"] == "shadow_error"
                for record in records
            ),
            "formal_report_changed": sum(
                not record["formal_report_unchanged"]
                for record in records
            ),
            "input_image_changed": sum(
                not record["image_unchanged"] for record in records
            ),
            "formal_success": sum(
                record["production"]["success"] for record in records
            ),
            "shadow_success": sum(
                bool((record.get("shadow") or {}).get("success"))
                for record in records
            ),
            "status_pairs": dict(sorted(status_pairs.items())),
        },
        "samples": records,
    }
    _atomic_write_json(output_dir / "issue_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    arguments = parser.parse_args()
    report = run_issue_check(arguments.output_dir)
    print(json.dumps(report["metrics"], indent=2))
    print(f"report: {(arguments.output_dir / 'issue_report.json').resolve()}")


if __name__ == "__main__":
    main()
