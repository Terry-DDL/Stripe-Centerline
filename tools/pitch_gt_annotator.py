"""Manual raw-pitch ground-truth annotation tool.

This tool deliberately has no dependency on the detection pipeline.  It shows
only source grayscale pixels, the frozen sampling ROI, and human annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_VERSION = "raw_pitch_gt_v1"
DATA_DIR = PROJECT_ROOT / "tests" / "data" / DATASET_VERSION
MANIFEST_PATH = DATA_DIR / "manifest.json"
SPLIT_PATHS = {
    "development": DATA_DIR / "development.json",
    "held-out": DATA_DIR / "heldout.json",
}
IMAGES_DIR = PROJECT_ROOT / "images"
PREVIEW_DIR = PROJECT_ROOT / "outputs" / DATASET_VERSION / "previews"

VALID_LABELS = ("valid", "ambiguous", "unavailable")
VALID_CONFIDENCES = ("high", "medium", "low")
MIN_VALID_CENTERS = 5

MAGNIFIER_SOURCE_WIDTH = 80
MAGNIFIER_SOURCE_HEIGHT = 60
MAGNIFIER_SCALE = 4.0
MAGNIFIER_OFFSET = 18
MAGNIFIER_REFRESH_MS = 16
MINIMUM_RELIABLE_MACOS_TK = (8, 6, 13)


def version_numbers(version: str) -> tuple[int, ...]:
    """Return numeric version parts such as 8.6.13 -> (8, 6, 13)."""

    parts = []
    for part in version.split("."):
        digits = "".join(
            character for character in part if character.isdigit()
        )
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def reliable_tk_runtime(
    patchlevel: str,
    platform_name: str = sys.platform,
) -> bool:
    """Reject the Tk version known to lose mouse events on modern macOS."""

    if platform_name != "darwin":
        return True
    return version_numbers(patchlevel) >= MINIMUM_RELIABLE_MACOS_TK


def calculate_display_size(
    image_shape: tuple[int, ...],
    max_width: int,
    max_height: int | None = None,
    allow_enlarge: bool = False,
) -> tuple[int, int]:
    """Return an aspect-preserving display size."""

    height, width = image_shape[:2]
    scale = max_width / width
    if max_height is not None:
        scale = min(scale, max_height / height)
    if not allow_enlarge:
        scale = min(1.0, scale)
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize_for_display(
    image,
    max_width: int,
    max_height: int | None = None,
    nearest: bool = False,
    allow_enlarge: bool = False,
):
    """Resize one display copy and return exact source-coordinate scales."""

    display_width, display_height = calculate_display_size(
        image.shape,
        max_width,
        max_height,
        allow_enlarge,
    )
    height, width = image.shape[:2]
    if (display_width, display_height) == (width, height):
        display = image.copy()
    else:
        interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
        display = cv2.resize(
            image,
            (display_width, display_height),
            interpolation=interpolation,
        )
    return display, width / display_width, height / display_height


def map_display_point_to_source(
    x_display: float,
    y_display: float,
    scale_x_to_source: float,
    scale_y_to_source: float,
    image_shape: tuple[int, ...],
    x0_source: int = 0,
    y0_source: int = 0,
) -> tuple[int, int]:
    """Map one canvas point back to a clipped source coordinate."""

    height, width = image_shape[:2]
    x_source = x0_source + round(x_display * scale_x_to_source)
    y_source = y0_source + round(y_display * scale_y_to_source)
    return (
        min(width - 1, max(0, x_source)),
        min(height - 1, max(0, y_source)),
    )


def extract_centered_region(
    image,
    center_point: tuple[int, int],
    width: int,
    height: int,
):
    """Extract a fixed-size region, replicating pixels beyond image edges."""

    image_height, image_width = image.shape[:2]
    center_x, center_y = center_point
    x0 = center_x - width // 2
    y0 = center_y - height // 2
    x1 = x0 + width
    y1 = y0 + height
    clipped_x0 = max(0, x0)
    clipped_y0 = max(0, y0)
    clipped_x1 = min(image_width, x1)
    clipped_y1 = min(image_height, y1)
    region = image[clipped_y0:clipped_y1, clipped_x0:clipped_x1]
    return cv2.copyMakeBorder(
        region,
        clipped_y0 - y0,
        y1 - clipped_y1,
        clipped_x0 - x0,
        x1 - clipped_x1,
        cv2.BORDER_REPLICATE,
    )


def create_magnifier_display(
    image_gray,
    center_point: tuple[int, int],
    source_width: int = MAGNIFIER_SOURCE_WIDTH,
    source_height: int = MAGNIFIER_SOURCE_HEIGHT,
    scale: float = MAGNIFIER_SCALE,
):
    """Create the same nearest-neighbor 4x grayscale magnifier behavior."""

    region = extract_centered_region(
        image_gray,
        center_point,
        source_width,
        source_height,
    )
    display_width = max(1, round(source_width * scale))
    display_height = max(1, round(source_height * scale))
    display = cv2.cvtColor(region, cv2.COLOR_GRAY2RGB)
    display = cv2.resize(
        display,
        (display_width, display_height),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.drawMarker(
        display,
        (display_width // 2, display_height // 2),
        (255, 0, 0),
        cv2.MARKER_CROSS,
        17,
        1,
    )
    return display


def magnifier_canvas_position(
    pointer_x: int,
    pointer_y: int,
    magnifier_width: int,
    magnifier_height: int,
    canvas_width: int,
    canvas_height: int,
    offset: int = MAGNIFIER_OFFSET,
) -> tuple[int, int]:
    """Place the magnifier beside the pointer and inside the canvas."""

    x = pointer_x + offset
    y = pointer_y + offset
    if x + magnifier_width > canvas_width:
        x = pointer_x - offset - magnifier_width
    if y + magnifier_height > canvas_height:
        y = pointer_y - offset - magnifier_height
    return (
        max(0, min(x, max(0, canvas_width - magnifier_width))),
        max(0, min(y, max(0, canvas_height - magnifier_height))),
    )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local image."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_grayscale_image(path: Path) -> np.ndarray:
    """Decode one source image without using any preprocessing."""

    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image_gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image_gray is None or image_gray.size == 0:
        raise ValueError(f"OpenCV could not decode {path}")
    return image_gray


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON through a sibling temporary file, then replace atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(data, output_file, indent=2, ensure_ascii=False)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def validate_manifest(manifest: dict) -> list[str]:
    """Return manifest problems without consulting any detector output."""

    errors = []
    if manifest.get("dataset_version") != DATASET_VERSION:
        errors.append("unexpected dataset_version")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != 36:
        return errors + ["manifest must contain exactly 36 samples"]
    sample_ids = [sample.get("sample_id") for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        errors.append("sample IDs must be unique")
    expected_ids = [f"P{index:03d}" for index in range(1, 37)]
    if sample_ids != expected_ids:
        errors.append("sample IDs must remain P001..P036 in order")
    counts = {
        split: sum(sample.get("split") == split for sample in samples)
        for split in SPLIT_PATHS
    }
    if counts != {"development": 25, "held-out": 11}:
        errors.append("split counts must remain development=25 held-out=11")
    for sample in samples:
        image_name = sample.get("image_name")
        image_path = IMAGES_DIR / str(image_name)
        if not image_path.is_file():
            errors.append(f"{sample.get('sample_id')}: image missing")
            continue
        image_gray = load_grayscale_image(image_path)
        height, width = image_gray.shape
        bounds = sample.get("roi_bounds_global", {})
        x0 = bounds.get("x0")
        y0 = bounds.get("y0")
        x1 = bounds.get("x1")
        y1 = bounds.get("y1")
        if not (
            isinstance(x0, int)
            and isinstance(y0, int)
            and isinstance(x1, int)
            and isinstance(y1, int)
            and 0 <= x0 < x1 <= width
            and 0 <= y0 < y1 <= height
        ):
            errors.append(f"{sample.get('sample_id')}: invalid ROI")
        reference = sample.get("reference_global", {})
        if not (
            x0 <= reference.get("x", -1) < x1
            and y0 <= reference.get("y", -1) < y1
        ):
            errors.append(
                f"{sample.get('sample_id')}: reference outside ROI"
            )
    return errors


def validate_image_hashes(manifest: dict) -> list[str]:
    """Return source-image provenance mismatches."""

    errors = []
    expected_by_image = {}
    for sample in manifest["samples"]:
        expected_by_image.setdefault(
            sample["image_name"],
            sample["image_sha256"],
        )
    for image_name, expected in expected_by_image.items():
        actual = sha256_file(IMAGES_DIR / image_name)
        if actual != expected:
            errors.append(
                f"{image_name}: expected {expected}, found {actual}"
            )
    return errors


@dataclass
class AnnotationEditorState:
    """Editable human state for one frozen sample."""

    centers: list[dict] = field(default_factory=list)
    label: str = ""
    confidence: str = ""
    notes: str = ""

    def add_center(
        self,
        x_global: int,
        y_global: int,
        roi_bounds: dict,
    ) -> None:
        x_roi = x_global - roi_bounds["x0"]
        y_roi = y_global - roi_bounds["y0"]
        if self.centers and x_global <= self.centers[-1]["x_global"]:
            raise ValueError(
                "Centers must be clicked from left to right "
                "with strictly increasing x."
            )
        self.centers.append(
            {
                "order": len(self.centers) + 1,
                "x_global": int(x_global),
                "x_roi": int(x_roi),
                "clicked_y_global": int(y_global),
                "clicked_y_roi": int(y_roi),
            }
        )

    def undo(self) -> None:
        if self.centers:
            self.centers.pop()

    def clear(self) -> None:
        self.centers.clear()
        self.label = ""
        self.confidence = ""
        self.notes = ""

    @classmethod
    def from_annotation(
        cls,
        annotation: dict | None,
    ) -> "AnnotationEditorState":
        if annotation is None:
            return cls()
        return cls(
            centers=[
                dict(center) for center in annotation.get("centers", [])
            ],
            label=annotation.get("label", ""),
            confidence=annotation.get("confidence", ""),
            notes=annotation.get("notes", ""),
        )


def annotation_validation_errors(
    label: str,
    confidence: str,
    centers: list[dict],
) -> list[str]:
    """Return user-facing errors for one attempted annotation save."""

    errors = []
    if label not in VALID_LABELS:
        errors.append("Choose valid, ambiguous, or unavailable.")
    if confidence not in VALID_CONFIDENCES:
        errors.append("Choose high, medium, or low confidence.")
    x_values = [center["x_global"] for center in centers]
    if any(
        right <= left for left, right in zip(x_values, x_values[1:])
    ):
        errors.append("Centers must be strictly ordered left to right.")
    if label == "valid" and len(centers) < MIN_VALID_CENTERS:
        errors.append("A valid annotation requires at least 5 centers.")
    if label == "unavailable" and centers:
        errors.append("Unavailable annotations cannot contain centers.")
    return errors


class AnnotationStore:
    """Load and atomically update the two physically separate split files."""

    def __init__(
        self,
        split_paths: dict[str, Path] | None = None,
    ):
        self.split_paths = split_paths or SPLIT_PATHS
        self.documents = {
            split: load_json(path)
            for split, path in self.split_paths.items()
        }

    def annotation_for(self, sample: dict) -> dict | None:
        return self.documents[sample["split"]]["annotations"][
            sample["sample_id"]
        ]

    def save(self, sample: dict, annotation: dict) -> None:
        split = sample["split"]
        document = self.documents[split]
        document["annotations"][sample["sample_id"]] = annotation
        atomic_write_json(self.split_paths[split], document)

    def annotated_count(self) -> int:
        return sum(
            annotation is not None
            for document in self.documents.values()
            for annotation in document["annotations"].values()
        )


def build_annotation_record(
    sample: dict,
    state: AnnotationEditorState,
    preview_path: Path,
) -> dict:
    """Build the complete persisted annotation for one sample."""

    return {
        "dataset_version": DATASET_VERSION,
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "image_name": sample["image_name"],
        "image_sha256": sample["image_sha256"],
        "reference_global": dict(sample["reference_global"]),
        "roi_bounds_global": dict(sample["roi_bounds_global"]),
        "label": state.label,
        "confidence": state.confidence,
        "notes": state.notes.strip(),
        "centers": [dict(center) for center in state.centers],
        "annotated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "preview_path": str(preview_path.relative_to(PROJECT_ROOT)),
    }


def render_roi_overlay(
    image_gray: np.ndarray,
    sample: dict,
    centers: list[dict],
) -> np.ndarray:
    """Draw only the reference and current human center annotations."""

    bounds = sample["roi_bounds_global"]
    roi = image_gray[
        bounds["y0"] : bounds["y1"],
        bounds["x0"] : bounds["x1"],
    ]
    overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    reference = sample["reference_global"]
    reference_roi = (
        reference["x"] - bounds["x0"],
        reference["y"] - bounds["y0"],
    )
    cv2.drawMarker(
        overlay,
        reference_roi,
        (255, 0, 0),
        cv2.MARKER_CROSS,
        17,
        1,
    )
    for center in centers:
        x_roi = center["x_roi"]
        cv2.line(
            overlay,
            (x_roi, 0),
            (x_roi, overlay.shape[0] - 1),
            (0, 220, 255),
            2,
        )
        cv2.putText(
            overlay,
            str(center["order"]),
            (max(0, x_roi - 5), 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            str(center["order"]),
            (max(0, x_roi - 5), 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def render_preview(
    image_gray: np.ndarray,
    sample: dict,
    annotation: dict,
) -> np.ndarray:
    """Return a compact confirmation image for the saved annotation."""

    overlay = render_roi_overlay(
        image_gray,
        sample,
        annotation["centers"],
    )
    header = np.full(
        (64, overlay.shape[1], 3),
        245,
        dtype=np.uint8,
    )
    lines = [
        (
            f"{sample['sample_id']}  {sample['split']}  "
            f"{sample['image_name']}"
        ),
        (
            f"label={annotation['label']}  "
            f"confidence={annotation['confidence']}  "
            f"centers={len(annotation['centers'])}"
        ),
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            header,
            line,
            (8, 23 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(np.vstack((header, overlay)), cv2.COLOR_RGB2BGR)


def save_preview(
    image_gray: np.ndarray,
    sample: dict,
    annotation: dict,
    preview_path: Path,
) -> None:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview = render_preview(image_gray, sample, annotation)
    if not cv2.imwrite(str(preview_path), preview):
        raise OSError(f"Could not save preview {preview_path}")


class PitchGtAnnotatorApp:
    """Tk controller for the frozen raw-pitch annotation manifest."""

    def __init__(
        self,
        root: tk.Tk,
        manifest: dict,
        store: AnnotationStore,
    ):
        self.root = root
        self.manifest = manifest
        self.samples = manifest["samples"]
        self.store = store
        self.sample_index = 0
        self.state = AnnotationEditorState()
        self.image_cache: dict[str, np.ndarray] = {}
        self.image_gray: np.ndarray | None = None
        self.roi_display: np.ndarray | None = None
        self.roi_photo = None
        self.context_photo = None
        self.magnifier_photo = None
        self.roi_scale_x = 1.0
        self.roi_scale_y = 1.0
        self.magnifier_pointer = None
        self.magnifier_after_id = None
        self.magnifier_image_id = None
        self.magnifier_border_id = None
        self.dirty = False

        self.label_var = tk.StringVar()
        self.confidence_var = tk.StringVar()
        self.notes_var = tk.StringVar()
        self.header_var = tk.StringVar()
        self.detail_var = tk.StringVar()
        self.centers_var = tk.StringVar()
        self.status_var = tk.StringVar()

        self._build_ui()
        self._load_sample(0)

    @property
    def sample(self) -> dict:
        return self.samples[self.sample_index]

    def _build_ui(self) -> None:
        self.root.title("Raw Pitch GT Annotator")
        self.root.geometry("1450x820")
        self.root.minsize(1200, 720)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        header = ttk.Frame(self.root, padding=(16, 12))
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            textvariable=self.header_var,
            font=("TkDefaultFont", 17, "bold"),
        ).pack(anchor=tk.W)
        ttk.Label(
            header,
            textvariable=self.detail_var,
        ).pack(anchor=tk.W, pady=(3, 0))

        body = ttk.Frame(self.root, padding=(16, 0, 16, 8))
        body.pack(fill=tk.BOTH, expand=True)
        context_frame = ttk.LabelFrame(
            body,
            text="Full-image context (read only)",
            padding=8,
        )
        context_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        roi_frame = ttk.LabelFrame(
            body,
            text="Original grayscale ROI — click centers left to right",
            padding=8,
        )
        roi_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=(12, 0),
        )

        self.context_canvas = tk.Canvas(
            context_frame,
            width=540,
            height=500,
            background="#202020",
            highlightthickness=0,
        )
        self.context_canvas.pack(fill=tk.BOTH, expand=True)

        self.roi_canvas = tk.Canvas(
            roi_frame,
            width=800,
            height=400,
            background="#202020",
            cursor="crosshair",
            highlightthickness=0,
        )
        self.roi_canvas.pack(fill=tk.BOTH, expand=True)
        self.roi_canvas.bind("<Button-1>", self._add_center_from_click)
        self.roi_canvas.bind("<Motion>", self._queue_magnifier)
        self.roi_canvas.bind("<Leave>", self._hide_magnifier)

        ttk.Label(
            roi_frame,
            textvariable=self.centers_var,
            wraplength=780,
        ).pack(anchor=tk.W, pady=(8, 0))

        form = ttk.Frame(self.root, padding=(16, 4, 16, 6))
        form.pack(fill=tk.X)
        label_group = ttk.LabelFrame(form, text="Label", padding=6)
        label_group.pack(side=tk.LEFT)
        for label in VALID_LABELS:
            ttk.Radiobutton(
                label_group,
                text=label,
                value=label,
                variable=self.label_var,
                command=self._form_changed,
            ).pack(side=tk.LEFT, padx=5)

        ttk.Label(form, text="Confidence").pack(
            side=tk.LEFT,
            padx=(18, 5),
        )
        confidence = ttk.Combobox(
            form,
            width=9,
            state="readonly",
            values=VALID_CONFIDENCES,
            textvariable=self.confidence_var,
        )
        confidence.pack(side=tk.LEFT)
        confidence.bind("<<ComboboxSelected>>", self._form_changed)

        ttk.Label(form, text="Notes").pack(
            side=tk.LEFT,
            padx=(18, 5),
        )
        notes_entry = ttk.Entry(
            form,
            textvariable=self.notes_var,
        )
        notes_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        notes_entry.bind("<KeyRelease>", self._form_changed)

        controls = ttk.Frame(self.root, padding=(16, 4, 16, 12))
        controls.pack(fill=tk.X)
        ttk.Button(
            controls,
            text="Previous",
            command=self._previous,
        ).pack(side=tk.LEFT)
        ttk.Button(
            controls,
            text="Undo center",
            command=self._undo,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            controls,
            text="Clear / restart",
            command=self._clear,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            controls,
            text="Save",
            command=self._save,
        ).pack(side=tk.RIGHT)
        ttk.Button(
            controls,
            text="Save & Next",
            command=self._save_and_next,
        ).pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Label(
            controls,
            textvariable=self.status_var,
        ).pack(side=tk.LEFT, padx=(16, 0))

    def _image_for_sample(self, sample: dict) -> np.ndarray:
        image_name = sample["image_name"]
        if image_name not in self.image_cache:
            self.image_cache[image_name] = load_grayscale_image(
                IMAGES_DIR / image_name
            )
        return self.image_cache[image_name]

    def _load_sample(self, index: int) -> None:
        self._hide_magnifier()
        self.sample_index = min(max(index, 0), len(self.samples) - 1)
        sample = self.sample
        self.image_gray = self._image_for_sample(sample)
        annotation = self.store.annotation_for(sample)
        self.state = AnnotationEditorState.from_annotation(annotation)
        self.label_var.set(self.state.label)
        self.confidence_var.set(self.state.confidence)
        self.notes_var.set(self.state.notes)
        self.dirty = False
        self._refresh_header()
        self._refresh_images()
        self._refresh_center_text()

    def _refresh_header(self) -> None:
        sample = self.sample
        reference = sample["reference_global"]
        bounds = sample["roi_bounds_global"]
        policy = (
            "HELD-OUT — evaluation only after parameter freeze"
            if sample["split"] == "held-out"
            else "development"
        )
        self.header_var.set(
            f"{sample['sample_id']}  "
            f"({self.sample_index + 1}/{len(self.samples)})  {policy}"
        )
        self.detail_var.set(
            f"{sample['image_name']}  reference=({reference['x']},"
            f"{reference['y']})  ROI=[{bounds['x0']},{bounds['x1']})"
            f"×[{bounds['y0']},{bounds['y1']})  "
            f"strata={', '.join(sample['strata'])}"
        )
        self.status_var.set(
            f"Annotated {self.store.annotated_count()}/{len(self.samples)}"
        )

    def _context_overlay(self) -> np.ndarray:
        sample = self.sample
        bounds = sample["roi_bounds_global"]
        reference = sample["reference_global"]
        overlay = cv2.cvtColor(self.image_gray, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(
            overlay,
            (bounds["x0"], bounds["y0"]),
            (bounds["x1"] - 1, bounds["y1"] - 1),
            (0, 255, 0),
            3,
        )
        cv2.drawMarker(
            overlay,
            (reference["x"], reference["y"]),
            (255, 0, 0),
            cv2.MARKER_CROSS,
            25,
            2,
        )
        return overlay

    def _refresh_images(self) -> None:
        context, _sx, _sy = resize_for_display(
            self._context_overlay(),
            540,
            500,
        )
        self.context_photo = ImageTk.PhotoImage(
            Image.fromarray(context),
            master=self.root,
        )
        self.context_canvas.config(
            width=context.shape[1],
            height=context.shape[0],
        )
        self.context_canvas.delete("all")
        self.context_canvas.create_image(
            0,
            0,
            anchor=tk.NW,
            image=self.context_photo,
        )

        overlay = render_roi_overlay(
            self.image_gray,
            self.sample,
            self.state.centers,
        )
        display, self.roi_scale_x, self.roi_scale_y = resize_for_display(
            overlay,
            800,
            400,
            nearest=True,
            allow_enlarge=True,
        )
        self.roi_display = display
        self.roi_photo = ImageTk.PhotoImage(
            Image.fromarray(display),
            master=self.root,
        )
        self.roi_canvas.config(
            width=display.shape[1],
            height=display.shape[0],
        )
        self.roi_canvas.delete("all")
        self.roi_canvas.create_image(
            0,
            0,
            anchor=tk.NW,
            image=self.roi_photo,
            tags=("base_roi",),
        )

    def _refresh_center_text(self) -> None:
        if not self.state.centers:
            self.centers_var.set(
                "Centers: none. Click at least five consecutive centers "
                "for a valid annotation."
            )
            return
        centers = ", ".join(
            f"{center['order']}:{center['x_global']}"
            for center in self.state.centers
        )
        self.centers_var.set(
            f"Centers global x ({len(self.state.centers)}): {centers}"
        )

    def _form_changed(self, _event=None) -> None:
        self.dirty = True

    def _add_center_from_click(self, event) -> None:
        bounds = self.sample["roi_bounds_global"]
        x_global, y_global = map_display_point_to_source(
            event.x,
            event.y,
            self.roi_scale_x,
            self.roi_scale_y,
            self.image_gray.shape,
            bounds["x0"],
            bounds["y0"],
        )
        if not (
            bounds["x0"] <= x_global < bounds["x1"]
            and bounds["y0"] <= y_global < bounds["y1"]
        ):
            return
        try:
            self.state.add_center(
                x_global,
                y_global,
                bounds,
            )
        except ValueError as error:
            messagebox.showwarning(
                "Center order",
                str(error),
                parent=self.root,
            )
            return
        self.dirty = True
        self._refresh_images()
        self._refresh_center_text()

    def _queue_magnifier(self, event) -> None:
        self.magnifier_pointer = (event.x, event.y)
        if self.magnifier_after_id is None:
            self.magnifier_after_id = self.root.after(
                MAGNIFIER_REFRESH_MS,
                self._render_magnifier,
            )

    def _render_magnifier(self) -> None:
        self.magnifier_after_id = None
        if (
            self.image_gray is None
            or self.magnifier_pointer is None
            or self.roi_display is None
        ):
            return
        bounds = self.sample["roi_bounds_global"]
        pointer_x, pointer_y = self.magnifier_pointer
        source_point = map_display_point_to_source(
            pointer_x,
            pointer_y,
            self.roi_scale_x,
            self.roi_scale_y,
            self.image_gray.shape,
            bounds["x0"],
            bounds["y0"],
        )
        display = create_magnifier_display(
            self.image_gray,
            source_point,
        )
        self.magnifier_photo = ImageTk.PhotoImage(
            Image.fromarray(display),
            master=self.root,
        )
        magnifier_x, magnifier_y = magnifier_canvas_position(
            pointer_x,
            pointer_y,
            self.magnifier_photo.width(),
            self.magnifier_photo.height(),
            self.roi_display.shape[1],
            self.roi_display.shape[0],
        )
        self.roi_canvas.delete("magnifier")
        self.magnifier_image_id = self.roi_canvas.create_image(
            magnifier_x,
            magnifier_y,
            anchor=tk.NW,
            image=self.magnifier_photo,
            tags=("magnifier",),
        )
        self.magnifier_border_id = self.roi_canvas.create_rectangle(
            magnifier_x,
            magnifier_y,
            magnifier_x + self.magnifier_photo.width(),
            magnifier_y + self.magnifier_photo.height(),
            outline="#ff0000",
            width=1,
            tags=("magnifier",),
        )

    def _hide_magnifier(self, _event=None) -> None:
        if self.magnifier_after_id is not None:
            self.root.after_cancel(self.magnifier_after_id)
        self.magnifier_after_id = None
        self.magnifier_pointer = None
        if hasattr(self, "roi_canvas"):
            self.roi_canvas.delete("magnifier")
        self.magnifier_image_id = None
        self.magnifier_border_id = None
        self.magnifier_photo = None

    def _undo(self) -> None:
        self.state.undo()
        self.dirty = True
        self._refresh_images()
        self._refresh_center_text()

    def _clear(self) -> None:
        self.state.clear()
        self.label_var.set("")
        self.confidence_var.set("")
        self.notes_var.set("")
        self.dirty = True
        self._refresh_images()
        self._refresh_center_text()

    def _confirm_discard_if_dirty(self) -> bool:
        if not self.dirty:
            return True
        return messagebox.askyesno(
            "Discard unsaved changes?",
            "This sample has unsaved changes. Discard them?",
            parent=self.root,
        )

    def _previous(self) -> None:
        if self.sample_index == 0:
            return
        if self._confirm_discard_if_dirty():
            self._load_sample(self.sample_index - 1)

    def _save(self) -> bool:
        self.state.label = self.label_var.get()
        self.state.confidence = self.confidence_var.get()
        self.state.notes = self.notes_var.get()
        errors = annotation_validation_errors(
            self.state.label,
            self.state.confidence,
            self.state.centers,
        )
        if errors:
            messagebox.showerror(
                "Cannot save annotation",
                "\n".join(errors),
                parent=self.root,
            )
            return False

        preview_path = PREVIEW_DIR / f"{self.sample['sample_id']}.png"
        annotation = build_annotation_record(
            self.sample,
            self.state,
            preview_path,
        )
        try:
            save_preview(
                self.image_gray,
                self.sample,
                annotation,
                preview_path,
            )
            self.store.save(self.sample, annotation)
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Could not save",
                str(error),
                parent=self.root,
            )
            return False
        self.dirty = False
        self._refresh_header()
        self.status_var.set(
            f"Saved {self.sample['sample_id']} — "
            f"annotated {self.store.annotated_count()}/{len(self.samples)}"
        )
        return True

    def _save_and_next(self) -> None:
        if not self._save():
            return
        if self.sample_index + 1 < len(self.samples):
            self._load_sample(self.sample_index + 1)
        else:
            messagebox.showinfo(
                "Annotation set complete",
                "The final manifest sample has been saved.",
                parent=self.root,
            )

    def _on_close(self) -> None:
        if self._confirm_discard_if_dirty():
            self._hide_magnifier()
            self.root.destroy()


def main() -> None:
    manifest = load_json(MANIFEST_PATH)
    manifest_errors = validate_manifest(manifest)
    manifest_errors.extend(validate_image_hashes(manifest))
    if manifest_errors:
        raise ValueError(
            "Frozen GT manifest is invalid:\n- "
            + "\n- ".join(manifest_errors)
        )
    store = AnnotationStore()
    root = tk.Tk()
    patchlevel = str(root.tk.call("info", "patchlevel"))
    if not reliable_tk_runtime(patchlevel):
        messagebox.showerror(
            "Unsupported Tk runtime",
            (
                f"Tk {patchlevel} is unreliable for mouse annotation "
                "on this macOS version. Use .venv-desktop."
            ),
            parent=root,
        )
        root.destroy()
        return
    PitchGtAnnotatorApp(root, manifest, store)
    root.mainloop()


if __name__ == "__main__":
    main()
