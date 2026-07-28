"""Manual annotator entrypoint for the frozen final raw-pitch held-out set."""

from pathlib import Path
import tkinter as tk

try:
    from tools import pitch_gt_annotator as base
except ModuleNotFoundError:
    import pitch_gt_annotator as base


DATASET_VERSION = "raw_pitch_final_heldout_v1"
DATA_DIR = (
    base.PROJECT_ROOT / "tests" / "data" / DATASET_VERSION
)
MANIFEST_PATH = DATA_DIR / "manifest.json"
ANNOTATIONS_PATH = DATA_DIR / "annotations.json"
PREVIEW_DIR = (
    base.PROJECT_ROOT / "outputs" / DATASET_VERSION / "previews"
)
EXPECTED_PITCH_COMMIT = (
    "41fe2f3b2870ec920ebd443307467e4bc02b1cee"
)
EXPECTED_CONFIG_CHECKSUM = (
    "2a5e234ea21807dabf58d45b5732604e5d95a50c4cd76033a66b96fbc2bb62a3"
)
FORBIDDEN_PRIOR_IMAGES = {
    "Sample 2.bmp",
    "Stripe_01_date20250601_t113239765.bmp",
    "Stripe_05_e0_t145409670_v0p10745_do.bmp",
    "Stripe_09_e0_t200229235_v3p02565_do.bmp",
    "Stripe_10_e0_t221236602_v8p56736_retry.bmp",
    "Stripe_11_e0_t225623761_v5p9823_do.bmp",
}


def validate_final_manifest(manifest: dict) -> list[str]:
    """Validate the frozen final manifest without consulting any detector."""

    errors = []
    samples = manifest.get("samples", [])
    if manifest.get("dataset_version") != DATASET_VERSION:
        errors.append("unexpected dataset version")
    if manifest.get("status") != "frozen_before_annotation":
        errors.append("manifest is not frozen before annotation")
    if manifest.get("frozen_pitch_commit") != EXPECTED_PITCH_COMMIT:
        errors.append("wrong frozen pitch commit")
    if (
        manifest.get("frozen_configuration_checksum")
        != EXPECTED_CONFIG_CHECKSUM
    ):
        errors.append("wrong frozen configuration checksum")
    if len(samples) != 15:
        errors.append("final held-out must contain exactly 15 samples")
        return errors
    expected_ids = [f"F{index:03d}" for index in range(1, 16)]
    if [sample.get("sample_id") for sample in samples] != expected_ids:
        errors.append("sample IDs must remain F001..F015 in order")
    image_names = {sample.get("image_name") for sample in samples}
    if len(image_names) != 5:
        errors.append("final held-out must use exactly five images")
    if image_names & FORBIDDEN_PRIOR_IMAGES:
        errors.append("a prior development/held-out image was reused")

    image_cache = {}
    hash_cache = {}
    for sample in samples:
        sample_id = sample["sample_id"]
        if sample.get("split") != "held-out":
            errors.append(f"{sample_id}: split changed")
        if sample.get("direction") not in {"vertical", "horizontal"}:
            errors.append(f"{sample_id}: invalid direction")
        image_name = sample["image_name"]
        image_path = base.IMAGES_DIR / image_name
        if not image_path.is_file():
            errors.append(f"{sample_id}: source image missing")
            continue
        if image_name not in image_cache:
            image_cache[image_name] = base.load_grayscale_image(image_path)
            hash_cache[image_name] = base.sha256_file(image_path)
        if hash_cache[image_name] != sample["image_sha256"]:
            errors.append(f"{sample_id}: source image hash mismatch")
        height, width = image_cache[image_name].shape
        bounds = sample["roi_bounds_global"]
        if not (
            0 <= bounds["x0"] < bounds["x1"] <= width
            and 0 <= bounds["y0"] < bounds["y1"] <= height
            and bounds["x1"] - bounds["x0"] == 500
            and bounds["y1"] - bounds["y0"] == 200
        ):
            errors.append(f"{sample_id}: invalid frozen ROI")
        reference = sample["reference_global"]
        if not (
            bounds["x0"] <= reference["x"] < bounds["x1"]
            and bounds["y0"] <= reference["y"] < bounds["y1"]
        ):
            errors.append(f"{sample_id}: reference outside ROI")
    return errors


def configure_base_module() -> None:
    """Point the shared human-only UI helpers at the final isolated files."""

    base.DATASET_VERSION = DATASET_VERSION
    base.DATA_DIR = DATA_DIR
    base.MANIFEST_PATH = MANIFEST_PATH
    base.SPLIT_PATHS = {"held-out": ANNOTATIONS_PATH}
    base.PREVIEW_DIR = PREVIEW_DIR


def main() -> None:
    configure_base_module()
    manifest = base.load_json(MANIFEST_PATH)
    errors = validate_final_manifest(manifest)
    if errors:
        raise ValueError(
            "Frozen final held-out manifest is invalid:\n- "
            + "\n- ".join(errors)
        )
    store = base.AnnotationStore()
    root = tk.Tk()
    patchlevel = str(root.tk.call("info", "patchlevel"))
    if not base.reliable_tk_runtime(patchlevel):
        root.destroy()
        raise RuntimeError(
            f"Tk {patchlevel} is unreliable; use .venv-desktop"
        )
    base.PitchGtAnnotatorApp(root, manifest, store)
    root.mainloop()


if __name__ == "__main__":
    main()
