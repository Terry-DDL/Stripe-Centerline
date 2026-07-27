"""Small real-image regressions for threshold fallback and brightness."""

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import CONFIG, INTERACTIVE_CONFIG  # noqa: E402
from interactive_pipeline import run_interactive_case  # noqa: E402


class RealThresholdRegressionTests(unittest.TestCase):
    def run_case(self, image_path: Path, click: tuple[int, int], output):
        if not image_path.is_file():
            self.skipTest(f"missing regression image: {image_path.name}")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)
        return run_interactive_case(
            image,
            click[0],
            click[1],
            output,
            CONFIG,
            INTERACTIVE_CONFIG,
            image_name=image_path.name,
        )

    def test_sample2_keeps_otsu_when_adaptive_pitch_is_worse(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self.run_case(
                PROJECT_ROOT / "images" / "Sample 2.bmp",
                (1000, 1000),
                Path(temporary_directory),
            )

        self.assertTrue(result.selection.success)
        self.assertEqual(
            result.report["interactive_result"]["threshold_method"],
            "otsu",
        )
        self.assertEqual(
            result.report["candidate_arbitration"][
                "threshold_selection_reasons"
            ]["original"],
            "better_combined_pitch_status",
        )

    def test_sample2_known_split_errors_use_safe_otsu_results(self):
        image_path = PROJECT_ROOT / "images" / "Sample 2.bmp"
        expected_formal = {
            (1613, 1745): (1573.0, 1612.0, 1648.0),
            (661, 1157): (623.5, 661.0, 701.0),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            for index, (click, expected_centers) in enumerate(
                expected_formal.items()
            ):
                with self.subTest(click=click):
                    result = self.run_case(
                        image_path,
                        click,
                        Path(temporary_directory) / str(index),
                    )
                    report = result.report
                    interactive = report["interactive_result"]
                    adaptive = report["candidate_arbitration"][
                        "candidates"
                    ]["original_adaptive"]
                    otsu = report["candidate_arbitration"][
                        "candidates"
                    ]["original_otsu"]
                    shadow = report["shadow_arbitration"]

                    self.assertEqual(
                        interactive["threshold_method"],
                        "otsu",
                    )
                    self.assertEqual(
                        interactive["left"]["center_x_global"],
                        expected_centers[0],
                    )
                    self.assertEqual(
                        interactive["clicked"]["center_x_global"],
                        expected_centers[1],
                    )
                    self.assertEqual(
                        interactive["right"]["center_x_global"],
                        expected_centers[2],
                    )
                    self.assertEqual(
                        adaptive["grayscale_topology"]["status"],
                        "Contradictory",
                    )
                    self.assertTrue(
                        adaptive["grayscale_topology"][
                            "strong_same_basin_conflict"
                        ]
                    )
                    self.assertFalse(
                        otsu["grayscale_topology"][
                            "strong_same_basin_conflict"
                        ]
                    )
                    self.assertTrue(shadow["enforced"])
                    self.assertTrue(shadow["rejection_applied"])
                    self.assertTrue(
                        shadow["would_change_formal_result"]
                    )
                    self.assertEqual(
                        shadow["hypothetical_winner"],
                        {
                            "geometry": "original",
                            "threshold_method": "otsu",
                        },
                    )
                    self.assertEqual(
                        shadow["final_winner"],
                        {
                            "geometry": "original",
                            "threshold_method": "otsu",
                        },
                    )

    def test_sample2_new_split_and_weak_neighbor_points(self):
        image_path = PROJECT_ROOT / "images" / "Sample 2.bmp"
        cases = {
            (1684, 1615): {
                "centers": (1648.0, 1685.0, 1722.5),
                "rejection_applied": True,
                "recovery_applied": False,
            },
            (396, 1506): {
                "centers": (356.0, 396.5, 437.75),
                "rejection_applied": False,
                "recovery_applied": True,
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            for index, (click, expected) in enumerate(cases.items()):
                with self.subTest(click=click):
                    result = self.run_case(
                        image_path,
                        click,
                        Path(temporary_directory) / str(index),
                    )
                    interactive = result.report["interactive_result"]
                    arbitration = result.report["shadow_arbitration"]
                    otsu = result.report["candidate_arbitration"][
                        "candidates"
                    ]["original_otsu"]

                    self.assertTrue(interactive["success"])
                    self.assertEqual(
                        interactive["threshold_method"],
                        "otsu",
                    )
                    for key, expected_center in zip(
                        ("left", "clicked", "right"),
                        expected["centers"],
                    ):
                        self.assertAlmostEqual(
                            interactive[key]["center_x_global"],
                            expected_center,
                            delta=1.0,
                        )
                    self.assertEqual(
                        arbitration["rejection_applied"],
                        expected["rejection_applied"],
                    )
                    self.assertEqual(
                        otsu["neighbor_recovery"]["applied"],
                        expected["recovery_applied"],
                    )

    def test_topology_rejection_fails_without_safe_alternative(self):
        image_path = (
            PROJECT_ROOT
            / "images"
            / "Stripe_08_e0_t160911003_v13p2083_do.bmp"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self.run_case(
                image_path,
                (1000, 550),
                Path(temporary_directory),
            )

        interactive = result.report["interactive_result"]
        arbitration = result.report["shadow_arbitration"]
        self.assertFalse(interactive["success"])
        self.assertIsNone(interactive["left"])
        self.assertIsNone(interactive["right"])
        self.assertEqual(
            interactive["failure_reasons"],
            ["strong_same_basin_conflict_no_safe_alternative"],
        )
        self.assertIsNone(arbitration["final_winner"])

    def test_topology_rejection_keeps_consistent_unverifiable_otsu(self):
        image_path = (
            PROJECT_ROOT
            / "images"
            / "Stripe_01_date20250601_t113239765.bmp"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self.run_case(
                image_path,
                (2201, 1491),
                Path(temporary_directory),
            )

        interactive = result.report["interactive_result"]
        self.assertTrue(interactive["success"])
        self.assertEqual(interactive["threshold_method"], "otsu")
        self.assertEqual(
            interactive["combined_pitch_status"],
            "Unable to verify",
        )
        for key, expected_center in (
            ("left", 2175.0),
            ("clicked", 2209.5),
            ("right", 2269.0),
        ):
            self.assertAlmostEqual(
                interactive[key]["center_x_global"],
                expected_center,
                delta=1.0,
            )

    def test_topology_rejection_can_be_disabled_for_comparison(self):
        image_path = PROJECT_ROOT / "images" / "Sample 2.bmp"
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)
        config = replace(
            INTERACTIVE_CONFIG,
            enable_grayscale_topology_rejection=False,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_interactive_case(
                image,
                1684,
                1615,
                Path(temporary_directory),
                CONFIG,
                config,
                image_name=image_path.name,
            )

        interactive = result.report["interactive_result"]
        arbitration = result.report["shadow_arbitration"]
        self.assertTrue(interactive["success"])
        self.assertEqual(interactive["threshold_method"], "adaptive")
        self.assertFalse(arbitration["enforced"])
        self.assertEqual(arbitration["mode"], "shadow")

    def test_stripe7_uses_otsu_when_adaptive_has_no_valid_pair(self):
        image_path = (
            PROJECT_ROOT
            / "images"
            / "Stripe_07_e0_t160816815_v12p0066_do.bmp"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self.run_case(
                image_path,
                (1719, 1109),
                Path(temporary_directory),
            )

        self.assertTrue(result.selection.success)
        self.assertEqual(
            result.report["interactive_result"]["threshold_method"],
            "otsu",
        )
        adaptive = result.report["candidate_arbitration"]["candidates"][
            "original_adaptive"
        ]
        self.assertFalse(adaptive["quality_success"])

    def test_real_plus_150_images_keep_centerlines(self):
        pairs = (
            (
                "Stripe_04_e0_t105403334_v7p8651_do.bmp",
                "Stripe_04_e0_t105403334_v7p8651_do_brightened_150.bmp",
            ),
            (
                "Stripe_08_e0_t160911003_v13p2083_do.bmp",
                "9f05c6ed-2276-5fb0-b608-28533c886120_brightened_150.bmp",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            for index, (original_name, brightened_name) in enumerate(pairs):
                with self.subTest(original=original_name):
                    original = self.run_case(
                        PROJECT_ROOT / "images" / original_name,
                        (1000, 1000),
                        Path(temporary_directory) / f"{index}_original",
                    )
                    brightened = self.run_case(
                        PROJECT_ROOT / "images" / brightened_name,
                        (1000, 1000),
                        Path(temporary_directory) / f"{index}_brightened",
                    )
                    self.assertTrue(original.selection.success)
                    self.assertTrue(brightened.selection.success)
                    for side in ("left", "right"):
                        original_x = original.report[
                            "interactive_result"
                        ][side]["center_x_global"]
                        brightened_x = brightened.report[
                            "interactive_result"
                        ][side]["center_x_global"]
                        self.assertAlmostEqual(
                            original_x,
                            brightened_x,
                            delta=1.0,
                        )

    def test_stripe9_faint_clicked_track_is_not_treated_as_white(self):
        image_path = (
            PROJECT_ROOT
            / "images"
            / "Stripe_09_e0_t200229235_v3p02565_do.bmp"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self.run_case(
                image_path,
                (564, 801),
                Path(temporary_directory),
            )

        interactive = result.report["interactive_result"]
        self.assertTrue(interactive["success"])
        self.assertEqual(result.report["click"]["classification"], "black_stripe")
        self.assertAlmostEqual(
            interactive["left"]["center_x_global"],
            545.0,
            delta=1.0,
        )
        self.assertAlmostEqual(
            interactive["right"]["center_x_global"],
            578.25,
            delta=1.0,
        )

    def test_stripe9_does_not_return_known_skipped_faint_neighbor(self):
        image_path = (
            PROJECT_ROOT
            / "images"
            / "Stripe_09_e0_t200229235_v3p02565_do.bmp"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self.run_case(
                image_path,
                (280, 801),
                Path(temporary_directory),
            )

        self.assertFalse(result.selection.success)
        self.assertIsNone(result.report["interactive_result"]["right"])

    def test_stripe10_original_failure_point_uses_nearest_adaptive_pair(self):
        image_path = (
            PROJECT_ROOT
            / "images"
            / "Stripe_10_e0_t221236602_v8p56736_retry.bmp"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self.run_case(
                image_path,
                (1250, 217),
                Path(temporary_directory),
            )

        interactive = result.report["interactive_result"]
        self.assertTrue(interactive["success"])
        self.assertEqual(interactive["threshold_method"], "adaptive")
        self.assertAlmostEqual(
            interactive["left"]["center_x_global"],
            1240.5,
            delta=1.0,
        )
        self.assertAlmostEqual(
            interactive["right"]["center_x_global"],
            1270.5,
            delta=1.0,
        )


if __name__ == "__main__":
    unittest.main()
