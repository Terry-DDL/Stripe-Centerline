import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tools import separator_path_gt_annotator as annotator


class SeparatorManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = annotator.load_json(annotator.MANIFEST_PATH)

    def test_manifest_is_frozen_with_exact_membership(self):
        self.assertEqual([], annotator.validate_manifest(self.manifest))
        samples = self.manifest["samples"]
        self.assertEqual(35, len(samples))
        self.assertEqual(
            [
                *[f"D{index:03d}" for index in range(1, 27)],
                *[f"H{index:03d}" for index in range(1, 10)],
            ],
            [sample["sample_id"] for sample in samples],
        )
        self.assertEqual(
            26,
            sum(sample["split"] == "development" for sample in samples),
        )
        self.assertEqual(
            9,
            sum(sample["split"] == "held-out" for sample in samples),
        )
        digest = hashlib.sha256(
            annotator.MANIFEST_PATH.read_bytes()
        ).hexdigest()
        frozen_digest = (
            annotator.DATA_DIR / "manifest.sha256"
        ).read_text(encoding="utf-8").split()[0]
        self.assertEqual(digest, frozen_digest)

    def test_known_issue_coordinates_are_all_in_development(self):
        expected = {
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
        }
        actual = {
            (
                sample["reference_global"]["x"],
                sample["reference_global"]["y"],
            )
            for sample in self.manifest["samples"]
            if (
                sample["split"] == "development"
                and "known_manual_issue" in sample["strata"]
            )
        }
        self.assertEqual(expected, actual)

    def test_heldout_images_are_isolated(self):
        development_images = {
            sample["image_name"]
            for sample in self.manifest["samples"]
            if sample["split"] == "development"
        }
        heldout_images = {
            sample["image_name"]
            for sample in self.manifest["samples"]
            if sample["split"] == "held-out"
        }
        self.assertTrue(development_images.isdisjoint(heldout_images))
        self.assertTrue(
            heldout_images.isdisjoint(
                annotator.FORBIDDEN_FINAL_PITCH_HELDOUT_IMAGES
            )
        )
        self.assertEqual(
            {
                "Sample 1.bmp",
                "Stripe_03_e0_t105348300_v-41p8_do.bmp",
                "Stripe_04_e0_t105403334_v7p8651_do.bmp",
            },
            heldout_images,
        )

    def test_split_files_are_blank_and_physically_separate(self):
        development = annotator.load_json(
            annotator.SPLIT_PATHS["development"]
        )
        heldout = annotator.load_json(
            annotator.SPLIT_PATHS["held-out"]
        )
        self.assertEqual(
            set(f"D{index:03d}" for index in range(1, 27)),
            set(development["annotations"]),
        )
        self.assertEqual(
            set(f"H{index:03d}" for index in range(1, 10)),
            set(heldout["annotations"]),
        )
        self.assertTrue(
            all(value is None for value in development["annotations"].values())
        )
        self.assertTrue(
            all(value is None for value in heldout["annotations"].values())
        )


class SeparatorPathStateTests(unittest.TestCase):
    def setUp(self):
        self.bounds = {"x0": 100, "y0": 200, "x1": 600, "y1": 440}

    def _three_point_record(
        self,
        role="left_clicked_boundary",
        visibility="fully_visible",
    ):
        draft = annotator.SeparatorPathDraft(
            role=role,
            visibility=visibility,
            confidence="high",
        )
        for x_global, y_global in (
            (150, 210),
            (154, 300),
            (158, 420),
        ):
            draft.add_point(
                x_global,
                y_global,
                self.bounds,
            )
        return draft.to_record()

    def test_points_must_increase_top_to_bottom_and_are_limited(self):
        draft = annotator.SeparatorPathDraft(
            role="left_clicked_boundary"
        )
        draft.add_point(150, 210, self.bounds)
        with self.assertRaises(ValueError):
            draft.add_point(151, 210, self.bounds)
        for index, y_global in enumerate(
            (240, 270, 300, 330, 360, 390),
            start=2,
        ):
            draft.add_point(150 + index, y_global, self.bounds)
        self.assertEqual(7, len(draft.control_points))
        with self.assertRaises(ValueError):
            draft.add_point(170, 420, self.bounds)
        draft.undo()
        self.assertEqual(6, len(draft.control_points))
        draft.clear_points()
        self.assertEqual([], draft.control_points)

    def test_visibility_and_required_role_contract(self):
        left = self._three_point_record()
        right = self._three_point_record(
            role="right_clicked_boundary",
            visibility="partially_visible",
        )
        sample = {"roi_bounds_global": self.bounds}
        self.assertEqual(
            [],
            annotator.annotation_validation_errors(
                sample,
                [left, right],
                "",
            ),
        )

        unavailable_with_points = dict(left)
        unavailable_with_points["visibility"] = "unavailable"
        errors = annotator.path_validation_errors(
            unavailable_with_points,
            self.bounds,
        )
        self.assertTrue(
            any("cannot contain points" in error for error in errors)
        )

        ambiguous = annotator.SeparatorPathDraft(
            role="left_clicked_boundary",
            visibility="ambiguous",
            confidence="low",
        ).to_record()
        self.assertEqual(
            [],
            annotator.path_validation_errors(ambiguous, self.bounds),
        )
        self.assertTrue(
            annotator.annotation_validation_errors(
                sample,
                [left],
                "",
            )
        )

    def test_visible_range_and_coordinate_mapping_match_points(self):
        record = self._three_point_record()
        self.assertEqual(
            {
                "y0_global": 210,
                "y1_global": 420,
                "y0_roi": 10,
                "y1_roi": 220,
            },
            record["visible_range"],
        )
        self.assertEqual(
            [50, 54, 58],
            [point["x_roi"] for point in record["control_points"]],
        )
        self.assertEqual(
            [10, 100, 220],
            [point["y_roi"] for point in record["control_points"]],
        )


class SeparatorPersistenceAndPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = annotator.load_json(annotator.MANIFEST_PATH)

    def test_store_routes_to_only_the_selected_split(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            paths = {}
            for split, source_path in annotator.SPLIT_PATHS.items():
                path = temporary / f"{split}.json"
                annotator.display_helpers.atomic_write_json(
                    path,
                    annotator.load_json(source_path),
                )
                paths[split] = path
            store = annotator.AnnotationStore(paths)
            sample = self.manifest["samples"][0]
            annotation = {"sample_id": sample["sample_id"]}
            store.save(sample, annotation)
            development = annotator.load_json(paths["development"])
            heldout = annotator.load_json(paths["held-out"])
            self.assertEqual(
                annotation,
                development["annotations"][sample["sample_id"]],
            )
            self.assertTrue(
                all(value is None for value in heldout["annotations"].values())
            )

    def test_preview_uses_exact_saved_control_points(self):
        sample = {
            "sample_id": "TEST",
            "reference_global": {"x": 350, "y": 320},
            "roi_bounds_global": {
                "x0": 100,
                "y0": 200,
                "x1": 600,
                "y1": 440,
            },
        }
        image = np.full((500, 700), 40, dtype=np.uint8)
        drafts = {
            role: annotator.SeparatorPathDraft(role)
            for role in annotator.ALL_ROLES
        }
        draft = drafts["left_clicked_boundary"]
        draft.visibility = "fully_visible"
        draft.confidence = "high"
        for x_global, y_global in (
            (150, 210),
            (154, 300),
            (158, 420),
        ):
            draft.add_point(
                x_global,
                y_global,
                sample["roi_bounds_global"],
            )
        annotation = {
            "separators": annotator.ordered_records(drafts),
        }
        overlay = annotator.render_roi_overlay(
            image,
            sample,
            annotation["separators"],
        )
        color = np.asarray(
            annotator.ROLE_COLORS_BGR["left_clicked_boundary"]
        )
        for point in draft.control_points:
            self.assertTrue(
                np.array_equal(
                    overlay[point["y_roi"], point["x_roi"]],
                    color,
                )
            )
        preview = annotator.render_preview(image, sample, annotation)
        self.assertEqual((380, 500, 3), preview.shape)
        with tempfile.TemporaryDirectory() as temporary_directory:
            preview_path = Path(temporary_directory) / "preview.png"
            annotator.save_preview(
                image,
                sample,
                annotation,
                preview_path,
            )
            reloaded = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)
            self.assertTrue(np.array_equal(preview, reloaded))

    def test_tool_has_no_detection_or_pitch_import(self):
        source = Path(annotator.__file__).read_text(encoding="utf-8")
        forbidden_imports = (
            "src.",
            "raw_local_pitch_prototype",
            "interactive_pipeline",
            "interactive_analysis",
            "pitch_reference",
            "prototype_column_basins",
        )
        for fragment in forbidden_imports:
            self.assertNotIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
