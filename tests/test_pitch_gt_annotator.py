import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tools import pitch_gt_annotator as annotator


class RawPitchManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = annotator.load_json(annotator.MANIFEST_PATH)

    def test_manifest_is_frozen_with_exact_split(self):
        samples = self.manifest["samples"]
        self.assertEqual(36, len(samples))
        self.assertEqual(
            [f"P{index:03d}" for index in range(1, 37)],
            [sample["sample_id"] for sample in samples],
        )
        self.assertEqual(
            25,
            sum(sample["split"] == "development" for sample in samples),
        )
        self.assertEqual(
            11,
            sum(sample["split"] == "held-out" for sample in samples),
        )
        self.assertEqual([], annotator.validate_manifest(self.manifest))

    def test_manifest_hashes_match_source_images(self):
        self.assertEqual(
            [],
            annotator.validate_image_hashes(self.manifest),
        )

    def test_blank_split_files_are_physically_separate(self):
        development = annotator.load_json(
            annotator.SPLIT_PATHS["development"]
        )
        heldout = annotator.load_json(annotator.SPLIT_PATHS["held-out"])
        self.assertEqual("development", development["split"])
        self.assertEqual("held-out", heldout["split"])
        self.assertEqual(
            "evaluation_only_after_parameter_freeze",
            heldout["access_policy"],
        )
        self.assertEqual(25, len(development["annotations"]))
        self.assertEqual(11, len(heldout["annotations"]))
        self.assertTrue(
            all(value is None for value in development["annotations"].values())
        )
        self.assertTrue(
            all(value is None for value in heldout["annotations"].values())
        )
        self.assertTrue(
            set(development["annotations"]).isdisjoint(
                heldout["annotations"]
            )
        )

    def test_tool_does_not_import_detection_pipeline(self):
        source = Path(annotator.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        forbidden_fragments = (
            "interactive_pipeline",
            "interactive_analysis",
            "pitch_reference",
            "stripe_analysis",
            "src.",
        )
        self.assertFalse(
            any(
                fragment in module
                for module in imported
                for fragment in forbidden_fragments
            ),
            imported,
        )


class AnnotationStateTests(unittest.TestCase):
    def setUp(self):
        self.bounds = {"x0": 100, "y0": 200, "x1": 600, "y1": 400}

    def test_display_to_roi_and_global_mapping(self):
        point = annotator.map_display_point_to_source(
            400,
            160,
            0.625,
            0.625,
            (2048, 2448),
            self.bounds["x0"],
            self.bounds["y0"],
        )
        self.assertEqual((350, 300), point)

    def test_magnifier_is_four_times_and_marks_source_center(self):
        image = np.arange(200 * 300, dtype=np.uint8).reshape(200, 300)
        display = annotator.create_magnifier_display(
            image,
            (150, 100),
        )
        self.assertEqual((240, 320, 3), display.shape)
        self.assertTrue(
            np.array_equal(display[120, 160], np.array([255, 0, 0]))
        )

    def test_centers_are_strictly_increasing_and_undo_clear_work(self):
        state = annotator.AnnotationEditorState(
            label="valid",
            confidence="high",
            notes="temporary",
        )
        state.add_center(120, 230, self.bounds)
        state.add_center(150, 240, self.bounds)
        self.assertEqual(
            {
                "order": 2,
                "x_global": 150,
                "x_roi": 50,
                "clicked_y_global": 240,
                "clicked_y_roi": 40,
            },
            state.centers[-1],
        )
        with self.assertRaises(ValueError):
            state.add_center(150, 250, self.bounds)
        with self.assertRaises(ValueError):
            state.add_center(149, 250, self.bounds)
        state.undo()
        self.assertEqual([120], [item["x_global"] for item in state.centers])
        state.clear()
        self.assertEqual([], state.centers)
        self.assertEqual("", state.label)
        self.assertEqual("", state.confidence)
        self.assertEqual("", state.notes)

    def test_invalid_annotation_states_cannot_save(self):
        centers = [
            {
                "order": index + 1,
                "x_global": 120 + index * 20,
                "x_roi": 20 + index * 20,
                "clicked_y_global": 230,
                "clicked_y_roi": 30,
            }
            for index in range(4)
        ]
        self.assertIn(
            "A valid annotation requires at least 5 centers.",
            annotator.annotation_validation_errors(
                "valid",
                "high",
                centers,
            ),
        )
        self.assertIn(
            "Unavailable annotations cannot contain centers.",
            annotator.annotation_validation_errors(
                "unavailable",
                "medium",
                centers,
            ),
        )
        self.assertEqual(
            [],
            annotator.annotation_validation_errors(
                "ambiguous",
                "low",
                centers,
            ),
        )
        self.assertEqual(
            [],
            annotator.annotation_validation_errors(
                "unavailable",
                "low",
                [],
            ),
        )


class AnnotationPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.manifest = annotator.load_json(annotator.MANIFEST_PATH)
        self.development_sample = next(
            sample
            for sample in self.manifest["samples"]
            if sample["split"] == "development"
        )
        self.heldout_sample = next(
            sample
            for sample in self.manifest["samples"]
            if sample["split"] == "held-out"
        )

    def _document(self, split, sample_id):
        policy = (
            "available_during_method_design"
            if split == "development"
            else "evaluation_only_after_parameter_freeze"
        )
        return {
            "dataset_version": annotator.DATASET_VERSION,
            "split": split,
            "access_policy": policy,
            "annotations": {sample_id: None},
        }

    def _annotation(self, sample, preview_path):
        bounds = sample["roi_bounds_global"]
        state = annotator.AnnotationEditorState(
            label="valid",
            confidence="high",
            notes="confirmed",
        )
        for offset in (30, 70, 110, 150, 190):
            state.add_center(
                bounds["x0"] + offset,
                bounds["y0"] + 80,
                bounds,
            )
        return annotator.build_annotation_record(
            sample,
            state,
            preview_path,
        )

    def test_split_routing_atomic_save_and_reload(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = {
                "development": root / "development.json",
                "held-out": root / "heldout.json",
            }
            annotator.atomic_write_json(
                paths["development"],
                self._document(
                    "development",
                    self.development_sample["sample_id"],
                ),
            )
            annotator.atomic_write_json(
                paths["held-out"],
                self._document(
                    "held-out",
                    self.heldout_sample["sample_id"],
                ),
            )
            heldout_before = paths["held-out"].read_bytes()
            store = annotator.AnnotationStore(paths)
            preview_path = annotator.PROJECT_ROOT / "outputs" / "P001.png"
            annotation = self._annotation(
                self.development_sample,
                preview_path,
            )
            store.save(self.development_sample, annotation)

            self.assertEqual(heldout_before, paths["held-out"].read_bytes())
            self.assertFalse(
                list(root.glob(".development.json.*.tmp")),
            )
            reloaded = annotator.AnnotationStore(paths)
            self.assertEqual(
                annotation,
                reloaded.annotation_for(self.development_sample),
            )
            self.assertEqual(
                self.development_sample["sample_id"],
                annotation["sample_id"],
            )
            self.assertEqual(
                annotator.DATASET_VERSION,
                annotation["dataset_version"],
            )
            self.assertRegex(
                annotation["annotated_at"],
                r"[+-]\d\d:\d\d$",
            )

    def test_preview_lines_match_json_order_and_coordinates(self):
        sample = copy.deepcopy(self.development_sample)
        sample["roi_bounds_global"] = {
            "x0": 100,
            "y0": 200,
            "x1": 600,
            "y1": 400,
        }
        sample["reference_global"] = {"x": 350, "y": 300}
        image = np.full((500, 700), 80, dtype=np.uint8)
        state = annotator.AnnotationEditorState(
            label="valid",
            confidence="medium",
        )
        for index, x_roi in enumerate((20, 80, 140, 200, 260)):
            state.add_center(100 + x_roi, 270 + index, sample["roi_bounds_global"])
        annotation = annotator.build_annotation_record(
            sample,
            state,
            annotator.PROJECT_ROOT / "outputs" / "preview.png",
        )
        overlay = annotator.render_roi_overlay(
            image,
            sample,
            annotation["centers"],
        )
        for center in annotation["centers"]:
            x_roi = center["x_roi"]
            self.assertTrue(
                np.all(overlay[40:, x_roi] == np.array([0, 220, 255])),
                x_roi,
            )
        preview = annotator.render_preview(image, sample, annotation)
        self.assertEqual((264, 500, 3), preview.shape)
        self.assertEqual(
            list(range(1, 6)),
            [center["order"] for center in annotation["centers"]],
        )


if __name__ == "__main__":
    unittest.main()
