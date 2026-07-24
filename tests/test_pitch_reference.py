"""Tests for the whole-image theoretical pitch reference and guard."""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import InteractiveConfig, ProcessingConfig  # noqa: E402
from interactive_pipeline import run_interactive_case  # noqa: E402
from pitch_reference import (  # noqa: E402
    PitchReferenceMap,
    PitchTileEstimate,
    build_pitch_reference_map,
    evaluate_pitch_guard,
)


def compact_config(**overrides):
    values = {
        "pitch_map_tile_width_px": 200,
        "pitch_map_tile_height_px": 100,
        "pitch_map_stride_x_px": 100,
        "pitch_map_stride_y_px": 50,
        "pitch_map_row_step_px": 2,
    }
    values.update(overrides)
    return InteractiveConfig(**values)


def stripe_image(
    height=300,
    width=400,
    pitch=40,
    stripe_width=10,
):
    image = np.full((height, width), 255, dtype=np.uint8)
    for center in range(20, width, pitch):
        image[:, center - stripe_width // 2 : center + stripe_width // 2] = 0
    return image


def one_tile_map(pitch=100.0):
    return PitchReferenceMap(
        image_shape=(300, 300),
        tiles=(
            PitchTileEstimate(
                x0=0,
                x1=300,
                y0=0,
                y1=300,
                pitch_px=pitch,
                gap_count=20,
                cluster_gap_count=20,
                cluster_support_ratio=1.0,
                valid=True,
                reason=None,
            ),
        ),
        tile_width_px=300,
        tile_height_px=300,
        stride_x_px=300,
        stride_y_px=300,
    )


def selection(
    left,
    right,
    classification="white_region",
    clicked=None,
    success=True,
):
    track = lambda center: SimpleNamespace(center_x_roi=float(center))
    return SimpleNamespace(
        success=success,
        click_classification=classification,
        left_track=None if left is None else track(left),
        right_track=None if right is None else track(right),
        clicked_track=None if clicked is None else track(clicked),
    )


class PitchReferenceTests(unittest.TestCase):
    def test_uniform_pitch_map_uses_dominant_gap_median(self):
        pitch_map = build_pitch_reference_map(
            stripe_image(),
            ProcessingConfig(),
            compact_config(),
        )

        query = pitch_map.query(200, 150, 0.25)

        self.assertEqual(query.baseline_pitch_px, 40.0)
        self.assertGreater(query.source_tile_count, 0)

    def test_brightness_shift_and_saturation_keep_pitch(self):
        image = stripe_image()
        brightened = np.clip(
            image.astype(np.int16) + 150,
            0,
            255,
        ).astype(np.uint8)

        pitch_map = build_pitch_reference_map(
            brightened,
            ProcessingConfig(),
            compact_config(),
        )

        self.assertEqual(pitch_map.query(200, 150, 0.25).baseline_pitch_px, 40.0)

    def test_small_tilt_keeps_horizontal_pitch_verifiable(self):
        image = stripe_image()
        tilted = cv2.warpAffine(
            image,
            cv2.getRotationMatrix2D((199.5, 149.5), 4.0, 1.0),
            (400, 300),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        pitch_map = build_pitch_reference_map(
            tilted,
            ProcessingConfig(),
            compact_config(),
        )
        query = pitch_map.query(200, 150, 0.25)

        self.assertIsNotNone(query.baseline_pitch_px)
        self.assertAlmostEqual(query.baseline_pitch_px, 40.0, delta=1.0)

    def test_spatial_map_keeps_different_left_and_right_pitch(self):
        image = np.full((300, 400), 255, dtype=np.uint8)
        for center in range(15, 200, 30):
            image[:, center - 4 : center + 5] = 0
        for center in range(225, 400, 50):
            image[:, center - 4 : center + 5] = 0

        pitch_map = build_pitch_reference_map(
            image,
            ProcessingConfig(),
            compact_config(),
        )

        self.assertAlmostEqual(
            pitch_map.query(50, 150, 0.25).baseline_pitch_px,
            30.0,
            delta=1.0,
        )
        self.assertAlmostEqual(
            pitch_map.query(350, 150, 0.25).baseline_pitch_px,
            50.0,
            delta=1.0,
        )

    def test_broken_stripe_rows_keep_dominant_pitch(self):
        image = stripe_image()
        for center in range(20, 400, 80):
            image[80:125, center - 5 : center + 5] = 255

        pitch_map = build_pitch_reference_map(
            image,
            ProcessingConfig(),
            compact_config(),
        )

        self.assertAlmostEqual(
            pitch_map.query(200, 150, 0.25).baseline_pitch_px,
            40.0,
            delta=1.0,
        )

    def test_flat_image_has_no_reliable_pitch(self):
        image = np.full((300, 400), 127, dtype=np.uint8)

        pitch_map = build_pitch_reference_map(
            image,
            ProcessingConfig(),
            compact_config(),
        )

        query = pitch_map.query(200, 150, 0.25)
        self.assertIsNone(query.baseline_pitch_px)
        self.assertEqual(query.reason, "no_reliable_pitch_tile_at_click")

    def test_conflicting_overlapping_tiles_are_unverifiable(self):
        tiles = (
            PitchTileEstimate(
                0, 300, 0, 300, 20.0, 10, 10, 1.0, True, None
            ),
            PitchTileEstimate(
                0, 300, 0, 300, 30.0, 10, 10, 1.0, True, None
            ),
        )
        pitch_map = PitchReferenceMap(
            (300, 300), tiles, 300, 300, 150, 150
        )

        query = pitch_map.query(150, 150, 0.25)

        self.assertIsNone(query.baseline_pitch_px)
        self.assertEqual(query.source_tile_count, 2)
        self.assertEqual(query.reason, "overlapping_pitch_tiles_disagree")

    def test_black_click_checks_two_one_pitch_intervals(self):
        result = evaluate_pitch_guard(
            selection(
                0,
                217,
                classification="black_stripe",
                clicked=67,
            ),
            100,
            100,
            one_tile_map(),
            InteractiveConfig(),
        )

        self.assertEqual(result["status"], "Normal")
        self.assertEqual(result["observed_intervals_px"], [67.0, 150.0])
        self.assertEqual(result["interval_pitch_ratios"], [0.67, 1.5])

    def test_interval_below_or_above_limits_is_suspicious(self):
        for right in (66, 151):
            with self.subTest(right=right):
                result = evaluate_pitch_guard(
                    selection(0, right),
                    100,
                    100,
                    one_tile_map(),
                    InteractiveConfig(),
                )
                self.assertEqual(result["status"], "Suspicious")

    def test_failed_local_detection_is_not_applicable(self):
        result = evaluate_pitch_guard(
            selection(None, None, success=False),
            100,
            100,
            one_tile_map(),
            InteractiveConfig(),
        )

        self.assertEqual(result["status"], "Not applicable")
        self.assertEqual(result["reason"], "local_detection_not_successful")

    def test_pipeline_saves_hidden_pitch_map_debug_files(self):
        image = stripe_image(height=300, width=400, pitch=80)
        config = compact_config(
            roi_half_width_px=150,
            roi_half_height_px=100,
        )
        pitch_map = build_pitch_reference_map(
            image,
            ProcessingConfig(),
            config,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            result = run_interactive_case(
                image,
                180,
                150,
                output_dir,
                ProcessingConfig(),
                config,
                pitch_reference_map=pitch_map,
            )

            self.assertIn("pitch_guard", result.report["interactive_result"])
            self.assertIn("pitch_reference_map.png", result.debug_images)
            self.assertTrue((output_dir / "pitch_reference_map.png").is_file())
            self.assertTrue((output_dir / "pitch_reference_map.json").is_file())


class RealStripePitchRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processing_config = ProcessingConfig()
        cls.interactive_config = InteractiveConfig()
        cls.images = {}
        cls.maps = {}
        for number, name in (
            (9, "Stripe_09_e0_t200229235_v3p02565_do.bmp"),
            (10, "Stripe_10_e0_t221236602_v8p56736_retry.bmp"),
            (11, "Stripe_11_e0_t225623761_v5p9823_do.bmp"),
        ):
            image = cv2.imread(
                str(PROJECT_ROOT / "images" / name),
                cv2.IMREAD_GRAYSCALE,
            )
            if image is None:
                raise unittest.SkipTest(f"real regression image missing: {name}")
            cls.images[number] = image
            cls.maps[number] = build_pitch_reference_map(
                image,
                cls.processing_config,
                cls.interactive_config,
            )

    def _run(self, number, x, y):
        with tempfile.TemporaryDirectory() as temporary_directory:
            return run_interactive_case(
                self.images[number],
                x,
                y,
                Path(temporary_directory),
                self.processing_config,
                self.interactive_config,
                image_name=f"Stripe {number}",
                pitch_reference_map=self.maps[number],
            )

    def test_stripe_9_close_minor_interval_is_suspicious(self):
        result = self._run(9, 1226, 1200)

        guard = result.report["interactive_result"]["pitch_guard"]
        self.assertTrue(result.selection.success)
        self.assertEqual(guard["status"], "Suspicious")
        self.assertAlmostEqual(guard["baseline_pitch_px"], 36.0, delta=1.0)

    def test_stripe_10_regular_region_is_normal(self):
        result = self._run(10, 400, 1000)

        guard = result.report["interactive_result"]["pitch_guard"]
        self.assertTrue(result.selection.success)
        self.assertEqual(guard["status"], "Normal")
        self.assertAlmostEqual(guard["baseline_pitch_px"], 14.5, delta=1.0)

    def test_stripe_11_bottom_is_normal_and_dark_middle_is_not_applicable(self):
        bottom = self._run(11, 1000, 1600)
        middle = self._run(11, 1000, 1000)

        bottom_guard = bottom.report["interactive_result"]["pitch_guard"]
        middle_guard = middle.report["interactive_result"]["pitch_guard"]
        self.assertEqual(bottom_guard["status"], "Normal")
        self.assertAlmostEqual(bottom_guard["baseline_pitch_px"], 25.0, delta=1.0)
        self.assertFalse(middle.selection.success)
        self.assertEqual(middle_guard["status"], "Not applicable")


if __name__ == "__main__":
    unittest.main()
