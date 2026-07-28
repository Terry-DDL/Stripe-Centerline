import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from tools import final_pitch_gt_annotator as final_annotator
from tools import pitch_gt_annotator as base


class FinalPitchGtManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = base.load_json(final_annotator.MANIFEST_PATH)

    def test_manifest_is_frozen_and_uses_only_new_images(self):
        self.assertEqual(
            [],
            final_annotator.validate_final_manifest(self.manifest),
        )
        samples = self.manifest["samples"]
        self.assertEqual(15, len(samples))
        self.assertEqual(
            [f"F{index:03d}" for index in range(1, 16)],
            [sample["sample_id"] for sample in samples],
        )
        image_names = {sample["image_name"] for sample in samples}
        self.assertEqual(5, len(image_names))
        self.assertTrue(
            image_names.isdisjoint(
                final_annotator.FORBIDDEN_PRIOR_IMAGES
            )
        )
        digest = hashlib.sha256(
            final_annotator.MANIFEST_PATH.read_bytes()
        ).hexdigest()
        checksum_file = (
            final_annotator.DATA_DIR / "manifest.sha256"
        ).read_text(encoding="utf-8")
        self.assertEqual(digest, checksum_file.split()[0])

    def test_answers_are_blank_and_physically_isolated(self):
        document = base.load_json(final_annotator.ANNOTATIONS_PATH)
        self.assertEqual("held-out", document["split"])
        self.assertEqual(
            "single_evaluation_only_after_frozen_v2",
            document["access_policy"],
        )
        self.assertEqual(
            final_annotator.EXPECTED_PITCH_COMMIT,
            document["frozen_pitch_commit"],
        )
        self.assertEqual(15, len(document["annotations"]))
        self.assertTrue(
            all(
                annotation is None
                for annotation in document["annotations"].values()
            )
        )

    def test_shared_store_routes_only_to_final_answer_file(self):
        original_version = base.DATASET_VERSION
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                path = Path(temporary_directory) / "annotations.json"
                source = base.load_json(final_annotator.ANNOTATIONS_PATH)
                base.atomic_write_json(path, source)
                store = base.AnnotationStore({"held-out": path})
                sample = self.manifest["samples"][0]
                annotation = {
                    "sample_id": sample["sample_id"],
                    "label": "unavailable",
                    "confidence": "high",
                    "centers": [],
                }
                store.save(sample, annotation)
                reloaded = base.load_json(path)
                self.assertEqual(
                    annotation,
                    reloaded["annotations"]["F001"],
                )
                self.assertTrue(
                    all(
                        value is None
                        for key, value in reloaded["annotations"].items()
                        if key != "F001"
                    )
                )
        finally:
            base.DATASET_VERSION = original_version

    def test_entrypoint_has_no_pitch_or_detector_import(self):
        source = Path(final_annotator.__file__).read_text(
            encoding="utf-8"
        )
        forbidden = (
            "raw_local_pitch_prototype",
            "interactive_pipeline",
            "interactive_analysis",
            "pitch_reference",
            "threshold",
            "morphology",
            "track",
        )
        for fragment in forbidden:
            self.assertNotIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
