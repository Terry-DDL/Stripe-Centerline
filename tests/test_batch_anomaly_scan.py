"""Tests for the audit-only batch anomaly scan tool."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools import batch_anomaly_scan as scan


class BatchAnomalyScanTests(unittest.TestCase):
    def test_image_discovery_excludes_outputs_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((20, 20), dtype=np.uint8)
            cv2.imwrite(str(root / "Sample.bmp"), image)
            cv2.imwrite(str(root / "Sample copy.bmp"), image)
            cv2.imwrite(str(root / "Sample_overlay.png"), image)
            cv2.imwrite(str(root / "Stripe_brightened_150.bmp"), image)
            found = scan.discover_real_images([root])
        self.assertEqual(len(found), 1)
        self.assertIn(found[0].name, {"Sample.bmp", "Sample copy.bmp"})

    def test_parent_directory_name_does_not_exclude_raw_image(self):
        with tempfile.TemporaryDirectory(prefix="Stripe Centerline ") as directory:
            root = Path(directory)
            image = np.zeros((20, 20), dtype=np.uint8)
            cv2.imwrite(str(root / "Stripe_10_retry.bmp"), image)
            found = scan.discover_real_images([root])
        self.assertEqual([path.name for path in found], ["Stripe_10_retry.bmp"])

    def test_grid_points_are_roi_safe_and_limited(self):
        points = scan.grid_points((2000, 2000), 100, 200)
        self.assertEqual(len(points), 200)
        self.assertTrue(all(270 <= x < 1730 for x, _ in points))
        self.assertTrue(all(120 <= y < 1880 for _, y in points))
        self.assertTrue(all(x % 100 == 70 for x, _ in points))
        self.assertTrue(all(y % 100 == 20 for _, y in points))

    def test_suspicious_success_uses_existing_diagnostics_only(self):
        result = {
            "success": True,
            "debug": {
                "local_dark_valley_recovery": {
                    "triggered": True,
                    "success": True,
                },
                "raw_pitch_result": {"confidence": "high"},
                "reference_relation": {"relation": "inside_basin"},
                "basin_graph": {"basin_candidates": []},
            },
            "final_hypothesis": {
                "reference_relation": "inside_basin",
                "basin_ids": {},
                "pitch_evidence": {
                    "confidence": "high",
                    "joint_decision": "high_pitch_supports_geometry",
                },
            },
        }
        self.assertEqual(scan.suspicious_reasons(result), ["recovery_triggered"])

    def test_manual_review_includes_twenty_normal_controls(self):
        records = []
        for index in range(60):
            records.append(
                {
                    "image": f"Image {index % 14}.bmp",
                    "image_path": f"/images/{index % 14}.bmp",
                    "x": 300 + (index % 5) * 200,
                    "y": 200 + (index // 5) * 200,
                    "success": True,
                    "suspicious_success": False,
                    "why_suspicious": "",
                    "primary_failure_reason": "",
                    "current_result": "success",
                }
            )
        for index in range(30):
            records.append(
                {
                    "image": f"Image {index % 14}.bmp",
                    "image_path": f"/images/{index % 14}.bmp",
                    "x": 350 + (index % 5) * 200,
                    "y": 250 + (index // 5) * 200,
                    "success": False,
                    "suspicious_success": False,
                    "why_suspicious": "",
                    "primary_failure_reason": f"failure_{index % 3}",
                    "current_result": "unavailable",
                }
            )
        selected = scan.select_manual_review(records, scan.ScanSettings())
        controls = [
            item for item in selected
            if item["review_category"] == "normal_success_control"
        ]
        self.assertLessEqual(len(selected), 40)
        self.assertEqual(len(controls), 20)

    def test_worker_calls_formal_runtime_for_every_point(self):
        fake_result = {
            "success": False,
            "unavailable_reason": "test_unavailable",
            "debug": {
                "separator_result": {
                    "status": "unavailable",
                    "unavailable_reason": "test_unavailable",
                },
                "basin_graph": {
                    "basin_candidates": [],
                    "verified_basins": [],
                },
                "reference_relation": {"status": "unavailable"},
                "raw_pitch_result": {},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "real.bmp"
            cv2.imwrite(str(image_path), np.zeros((400, 700), dtype=np.uint8))
            task = {
                "image_id": "I01_real",
                "image_path": str(image_path),
                "points": [(300, 150), (400, 250)],
                "spacing_px": 100,
                "diagnostics_path": str(root / "diagnostics.jsonl.gz"),
            }
            with patch.object(
                scan.formal_runtime,
                "run_frozen_stage3",
                return_value=fake_result,
            ) as formal:
                records = scan.scan_one_image(task)
        self.assertEqual(len(records), 2)
        self.assertEqual(formal.call_count, 2)

    def test_worker_records_formal_pipeline_exception_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "real.bmp"
            cv2.imwrite(str(image_path), np.zeros((400, 700), dtype=np.uint8))
            task = {
                "image_id": "I01_real",
                "image_path": str(image_path),
                "points": [(300, 150), (400, 250)],
                "spacing_px": 100,
                "diagnostics_path": str(root / "diagnostics.jsonl.gz"),
            }
            with patch.object(
                scan.formal_runtime,
                "run_frozen_stage3",
                side_effect=ValueError("formal failure"),
            ):
                records = scan.scan_one_image(task)
        self.assertEqual(len(records), 2)
        self.assertEqual(
            records[0]["primary_failure_reason"],
            "formal_pipeline_exception:ValueError",
        )


if __name__ == "__main__":
    unittest.main()
