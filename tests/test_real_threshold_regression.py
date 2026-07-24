"""Small real-image regressions for threshold fallback and brightness."""

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
