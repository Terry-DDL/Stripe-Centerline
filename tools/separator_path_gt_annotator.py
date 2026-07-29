"""Human-only separator-path annotation tool.

The tool displays source grayscale pixels and the frozen sampling geometry.
It does not import or call any detection, pitch, threshold, ridge, or
morphology implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    from tools import pitch_gt_annotator as display_helpers
except ModuleNotFoundError:
    import pitch_gt_annotator as display_helpers


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_VERSION = "separator_path_gt_v1"
DATA_DIR = PROJECT_ROOT / "tests" / "data" / DATASET_VERSION
MANIFEST_PATH = DATA_DIR / "manifest.json"
SPLIT_PATHS = {
    "development": DATA_DIR / "development.json",
    "held-out": DATA_DIR / "heldout.json",
}
IMAGES_DIR = PROJECT_ROOT / "images"
PREVIEW_DIR = PROJECT_ROOT / "outputs" / DATASET_VERSION / "previews"

REQUIRED_ROLES = (
    "left_clicked_boundary",
    "right_clicked_boundary",
)
OPTIONAL_ROLES = (
    "left_adjacent",
    "right_adjacent",
)
ALL_ROLES = REQUIRED_ROLES + OPTIONAL_ROLES
ROLE_LABELS = {
    "left_clicked_boundary": "Clicked basin — left boundary",
    "right_clicked_boundary": "Clicked basin — right boundary",
    "left_adjacent": "Necessary left adjacent separator",
    "right_adjacent": "Necessary right adjacent separator",
}
ROLE_SHORT_LABELS = {
    "left_clicked_boundary": "LB",
    "right_clicked_boundary": "RB",
    "left_adjacent": "LA",
    "right_adjacent": "RA",
}
ROLE_COLORS_BGR = {
    "left_clicked_boundary": (255, 80, 30),
    "right_clicked_boundary": (0, 220, 255),
    "left_adjacent": (255, 220, 0),
    "right_adjacent": (220, 70, 220),
}
VISIBILITY_LABELS = (
    "fully_visible",
    "partially_visible",
    "unavailable",
    "ambiguous",
)
CONFIDENCE_LABELS = ("high", "medium", "low")
MIN_CONTROL_POINTS = 3
MAX_CONTROL_POINTS = 7

FORBIDDEN_FINAL_PITCH_HELDOUT_IMAGES = {
    "Stripe_02_e0_t105204227_v-41p8_do.bmp",
    "Stripe_06_e0_t160727286_v0p9056_do.bmp",
    "Stripe_07_e0_t160816815_v12p0066_do.bmp",
    "Stripe_08_e0_t160911003_v13p2083_do.bmp",
    "Stripe_12_e1_t220320710_v7p3386_do.bmp",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_grayscale_image(path: Path) -> np.ndarray:
    """Decode one source image without changing its pixels."""

    return display_helpers.load_grayscale_image(path)


def validate_manifest(manifest: dict) -> list[str]:
    """Validate fixed membership and geometry without consulting an algorithm."""

    errors = []
    samples = manifest.get("samples", [])
    if manifest.get("dataset_version") != DATASET_VERSION:
        errors.append("unexpected dataset_version")
    if manifest.get("status") != "frozen_before_annotation":
        errors.append("manifest is not frozen before annotation")
    if len(samples) != 35:
        return errors + ["manifest must contain exactly 35 samples"]

    expected_ids = [
        *[f"D{index:03d}" for index in range(1, 27)],
        *[f"H{index:03d}" for index in range(1, 10)],
    ]
    sample_ids = [sample.get("sample_id") for sample in samples]
    if sample_ids != expected_ids:
        errors.append("sample IDs or ordering changed")
    if len(set(sample_ids)) != len(sample_ids):
        errors.append("sample IDs must be unique")

    split_counts = {
        split: sum(sample.get("split") == split for sample in samples)
        for split in SPLIT_PATHS
    }
    if split_counts != {"development": 26, "held-out": 9}:
        errors.append("split counts must remain development=26 held-out=9")

    development_images = {
        sample["image_name"]
        for sample in samples
        if sample.get("split") == "development"
    }
    heldout_images = {
        sample["image_name"]
        for sample in samples
        if sample.get("split") == "held-out"
    }
    if development_images & heldout_images:
        errors.append("held-out images overlap development images")
    if heldout_images & FORBIDDEN_FINAL_PITCH_HELDOUT_IMAGES:
        errors.append("separator held-out reuses final pitch held-out images")

    image_cache = {}
    hash_cache = {}
    for sample in samples:
        sample_id = sample.get("sample_id", "unknown")
        image_name = sample.get("image_name")
        image_path = IMAGES_DIR / str(image_name)
        if not image_path.is_file():
            errors.append(f"{sample_id}: source image missing")
            continue
        if image_name not in image_cache:
            image_cache[image_name] = load_grayscale_image(image_path)
            hash_cache[image_name] = sha256_file(image_path)
        if hash_cache[image_name] != sample.get("image_sha256"):
            errors.append(f"{sample_id}: source image hash mismatch")

        image_height, image_width = image_cache[image_name].shape
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
            and 0 <= x0 < x1 <= image_width
            and 0 <= y0 < y1 <= image_height
            and x1 - x0 == 500
            and y1 - y0 == 240
        ):
            errors.append(f"{sample_id}: invalid frozen 500x240 ROI")
            continue
        reference = sample.get("reference_global", {})
        if not (
            x0 <= reference.get("x", -1) < x1
            and y0 <= reference.get("y", -1) < y1
        ):
            errors.append(f"{sample_id}: reference outside ROI")
        if sample.get("direction") != "vertical":
            errors.append(f"{sample_id}: unsupported direction")
    return errors


def visible_range_for_points(points: list[dict]) -> dict | None:
    if not points:
        return None
    first = points[0]
    last = points[-1]
    return {
        "y0_global": first["y_global"],
        "y1_global": last["y_global"],
        "y0_roi": first["y_roi"],
        "y1_roi": last["y_roi"],
    }


def path_validation_errors(path: dict, bounds: dict) -> list[str]:
    """Return validation errors for one human separator path."""

    role = path.get("role")
    visibility = path.get("visibility")
    confidence = path.get("confidence")
    points = path.get("control_points", [])
    errors = []
    prefix = role or "separator"
    if role not in ALL_ROLES:
        errors.append(f"{prefix}: invalid role")
    if visibility not in VISIBILITY_LABELS:
        errors.append(f"{prefix}: choose a visibility label")
    if confidence not in CONFIDENCE_LABELS:
        errors.append(f"{prefix}: choose confidence")
    if not isinstance(path.get("notes", ""), str):
        errors.append(f"{prefix}: notes must be text")
    if not isinstance(points, list):
        return errors + [f"{prefix}: control_points must be a list"]

    count = len(points)
    if visibility in {"fully_visible", "partially_visible"}:
        if not MIN_CONTROL_POINTS <= count <= MAX_CONTROL_POINTS:
            errors.append(f"{prefix}: visible path requires 3–7 points")
    elif visibility == "ambiguous":
        if count not in {0, *range(MIN_CONTROL_POINTS, MAX_CONTROL_POINTS + 1)}:
            errors.append(
                f"{prefix}: ambiguous path uses either 0 or 3–7 points"
            )
    elif visibility == "unavailable" and count:
        errors.append(f"{prefix}: unavailable path cannot contain points")

    previous_y = None
    for expected_order, point in enumerate(points, start=1):
        if point.get("order") != expected_order:
            errors.append(f"{prefix}: point order is not contiguous")
        x_global = point.get("x_global")
        y_global = point.get("y_global")
        x_roi = point.get("x_roi")
        y_roi = point.get("y_roi")
        if not all(
            isinstance(value, int)
            for value in (x_global, y_global, x_roi, y_roi)
        ):
            errors.append(f"{prefix}: point coordinates must be integers")
            continue
        if not (
            bounds["x0"] <= x_global < bounds["x1"]
            and bounds["y0"] <= y_global < bounds["y1"]
        ):
            errors.append(f"{prefix}: point outside ROI")
        if (
            x_roi != x_global - bounds["x0"]
            or y_roi != y_global - bounds["y0"]
        ):
            errors.append(f"{prefix}: ROI/global mapping mismatch")
        if previous_y is not None and y_global <= previous_y:
            errors.append(f"{prefix}: points must increase from top to bottom")
        previous_y = y_global

    if path.get("visible_range") != visible_range_for_points(points):
        errors.append(f"{prefix}: visible_range does not match points")
    return errors


def annotation_validation_errors(
    sample: dict,
    separators: list[dict],
    notes: str,
) -> list[str]:
    """Validate one complete sample annotation."""

    errors = []
    if not isinstance(notes, str):
        errors.append("Sample notes must be text.")
    roles = [separator.get("role") for separator in separators]
    if len(roles) != len(set(roles)):
        errors.append("Each separator role may appear only once.")
    for required_role in REQUIRED_ROLES:
        if required_role not in roles:
            errors.append(
                f"Required role is missing: {ROLE_LABELS[required_role]}"
            )
    bounds = sample["roi_bounds_global"]
    for separator in separators:
        errors.extend(path_validation_errors(separator, bounds))
    return errors


@dataclass
class SeparatorPathDraft:
    role: str
    visibility: str = ""
    confidence: str = ""
    notes: str = ""
    control_points: list[dict] = field(default_factory=list)

    def add_point(
        self,
        x_global: int,
        y_global: int,
        roi_bounds: dict,
    ) -> None:
        if len(self.control_points) >= MAX_CONTROL_POINTS:
            raise ValueError("A separator path may contain at most 7 points.")
        if (
            self.control_points
            and y_global <= self.control_points[-1]["y_global"]
        ):
            raise ValueError(
                "Control points must be clicked from top to bottom."
            )
        self.control_points.append(
            {
                "order": len(self.control_points) + 1,
                "x_global": x_global,
                "y_global": y_global,
                "x_roi": x_global - roi_bounds["x0"],
                "y_roi": y_global - roi_bounds["y0"],
            }
        )

    def undo(self) -> None:
        if self.control_points:
            self.control_points.pop()

    def clear_points(self) -> None:
        self.control_points.clear()

    def reset(self) -> None:
        self.visibility = ""
        self.confidence = ""
        self.notes = ""
        self.control_points.clear()

    def is_touched(self) -> bool:
        return bool(
            self.visibility
            or self.confidence
            or self.notes
            or self.control_points
        )

    def to_record(self) -> dict:
        points = [dict(point) for point in self.control_points]
        return {
            "role": self.role,
            "visibility": self.visibility,
            "confidence": self.confidence,
            "notes": self.notes,
            "control_points": points,
            "visible_range": visible_range_for_points(points),
        }

    @classmethod
    def from_record(cls, record: dict) -> "SeparatorPathDraft":
        return cls(
            role=record["role"],
            visibility=record.get("visibility", ""),
            confidence=record.get("confidence", ""),
            notes=record.get("notes", ""),
            control_points=[
                dict(point) for point in record.get("control_points", [])
            ],
        )


class AnnotationStore:
    """Load and atomically update physically separated split files."""

    def __init__(self, split_paths: dict[str, Path] | None = None):
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
        sample_id = sample["sample_id"]
        if sample_id not in document["annotations"]:
            raise ValueError(f"{sample_id} is not part of {split}")
        document["annotations"][sample_id] = annotation
        display_helpers.atomic_write_json(self.split_paths[split], document)

    def annotated_count(self) -> int:
        return sum(
            annotation is not None
            for document in self.documents.values()
            for annotation in document["annotations"].values()
        )


def records_from_annotation(annotation: dict | None) -> dict[str, SeparatorPathDraft]:
    drafts = {role: SeparatorPathDraft(role) for role in ALL_ROLES}
    if annotation is None:
        return drafts
    for record in annotation.get("separators", []):
        if record.get("role") in drafts:
            drafts[record["role"]] = SeparatorPathDraft.from_record(record)
    return drafts


def ordered_records(drafts: dict[str, SeparatorPathDraft]) -> list[dict]:
    return [
        drafts[role].to_record()
        for role in ALL_ROLES
        if drafts[role].is_touched()
    ]


def build_annotation_record(
    sample: dict,
    drafts: dict[str, SeparatorPathDraft],
    notes: str,
    preview_path: Path,
) -> dict:
    return {
        "dataset_version": DATASET_VERSION,
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "image_name": sample["image_name"],
        "image_sha256": sample["image_sha256"],
        "reference_global": dict(sample["reference_global"]),
        "roi_bounds_global": dict(sample["roi_bounds_global"]),
        "direction": sample["direction"],
        "separators": ordered_records(drafts),
        "notes": notes,
        "annotated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "preview_path": str(preview_path.relative_to(PROJECT_ROOT)),
    }


def extract_roi(image_gray: np.ndarray, sample: dict) -> np.ndarray:
    bounds = sample["roi_bounds_global"]
    return image_gray[
        bounds["y0"] : bounds["y1"],
        bounds["x0"] : bounds["x1"],
    ]


def render_roi_overlay(
    image_gray: np.ndarray,
    sample: dict,
    separators: list[dict],
    selected_role: str | None = None,
) -> np.ndarray:
    """Draw only human paths and the frozen reference on the raw ROI."""

    bounds = sample["roi_bounds_global"]
    reference = sample["reference_global"]
    overlay = cv2.cvtColor(extract_roi(image_gray, sample), cv2.COLOR_GRAY2BGR)
    cv2.drawMarker(
        overlay,
        (
            reference["x"] - bounds["x0"],
            reference["y"] - bounds["y0"],
        ),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        19,
        1,
    )
    for separator in separators:
        role = separator["role"]
        points = separator.get("control_points", [])
        if not points:
            continue
        color = ROLE_COLORS_BGR[role]
        coordinates = np.asarray(
            [
                (point["x_roi"], point["y_roi"])
                for point in points
            ],
            dtype=np.int32,
        )
        thickness = 3 if role == selected_role else 2
        if len(coordinates) >= 2:
            cv2.polylines(
                overlay,
                [coordinates],
                False,
                color,
                thickness,
                cv2.LINE_AA,
            )
        for point in points:
            coordinate = (point["x_roi"], point["y_roi"])
            cv2.circle(overlay, coordinate, 3, color, -1, cv2.LINE_AA)
            cv2.putText(
                overlay,
                str(point["order"]),
                (coordinate[0] + 4, max(10, coordinate[1] - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color,
                1,
                cv2.LINE_AA,
            )
        first = points[0]
        cv2.putText(
            overlay,
            ROLE_SHORT_LABELS[role],
            (first["x_roi"] + 5, min(235, first["y_roi"] + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def render_preview(
    image_gray: np.ndarray,
    sample: dict,
    annotation: dict,
) -> np.ndarray:
    """Render a deterministic human-annotation confirmation preview."""

    roi_overlay = render_roi_overlay(
        image_gray,
        sample,
        annotation["separators"],
    )
    canvas = np.full((380, 500, 3), 246, dtype=np.uint8)
    canvas[:240] = roi_overlay
    cv2.putText(
        canvas,
        (
            f"{sample['sample_id']}  reference="
            f"({sample['reference_global']['x']},"
            f"{sample['reference_global']['y']})"
        ),
        (8, 263),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    records = {
        record["role"]: record
        for record in annotation["separators"]
    }
    for index, role in enumerate(ALL_ROLES):
        record = records.get(role)
        if record is None:
            text = f"{ROLE_SHORT_LABELS[role]}: not annotated"
            color = (100, 100, 100)
        else:
            visible_range = record["visible_range"]
            range_text = (
                "none"
                if visible_range is None
                else (
                    f"{visible_range['y0_global']}.."
                    f"{visible_range['y1_global']}"
                )
            )
            text = (
                f"{ROLE_SHORT_LABELS[role]}: {record['visibility']}  "
                f"{record['confidence']}  "
                f"points={len(record['control_points'])}  "
                f"visible_y={range_text}"
            )
            color = ROLE_COLORS_BGR[role]
        cv2.putText(
            canvas,
            text,
            (8, 290 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


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


class SeparatorPathAnnotatorApp:
    """Tk controller for human separator-path annotation."""

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
        self.image_cache: dict[str, np.ndarray] = {}
        self.image_gray: np.ndarray | None = None
        self.paths = {
            role: SeparatorPathDraft(role) for role in ALL_ROLES
        }
        self.active_role = ALL_ROLES[0]
        self.sample_notes = ""
        self.dirty = False

        self.context_photo = None
        self.roi_photo = None
        self.roi_display = None
        self.roi_scale_x = 1.0
        self.roi_scale_y = 1.0
        self.magnifier_photo = None
        self.magnifier_pointer = None
        self.magnifier_after_id = None

        self.header_var = tk.StringVar()
        self.detail_var = tk.StringVar()
        self.role_var = tk.StringVar(value=self.active_role)
        self.visibility_var = tk.StringVar()
        self.confidence_var = tk.StringVar()
        self.path_notes_var = tk.StringVar()
        self.sample_notes_var = tk.StringVar()
        self.path_summary_var = tk.StringVar()
        self.status_var = tk.StringVar()

        self._build_ui()
        self._load_sample(0)

    @property
    def sample(self) -> dict:
        return self.samples[self.sample_index]

    def _build_ui(self) -> None:
        self.root.title("Separator Path GT Annotator")
        self.root.geometry("1540x920")
        self.root.minsize(1250, 780)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        header = ttk.Frame(self.root, padding=(16, 10))
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

        body = ttk.Frame(self.root, padding=(16, 0, 16, 6))
        body.pack(fill=tk.BOTH, expand=True)
        context_frame = ttk.LabelFrame(
            body,
            text="Full-image context (read only)",
            padding=8,
        )
        context_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        roi_frame = ttk.LabelFrame(
            body,
            text=(
                "Original grayscale ROI — select a role, then click "
                "3–7 points from top to bottom"
            ),
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
            width=500,
            height=500,
            background="#202020",
            highlightthickness=0,
        )
        self.context_canvas.pack(fill=tk.BOTH, expand=True)
        self.roi_canvas = tk.Canvas(
            roi_frame,
            width=900,
            height=432,
            background="#202020",
            cursor="crosshair",
            highlightthickness=0,
        )
        self.roi_canvas.pack(fill=tk.BOTH, expand=True)
        self.roi_canvas.bind("<Button-1>", self._add_point_from_click)
        self.roi_canvas.bind("<Motion>", self._queue_magnifier)
        self.roi_canvas.bind("<Leave>", self._hide_magnifier)
        ttk.Label(
            roi_frame,
            textvariable=self.path_summary_var,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(7, 0))

        path_form = ttk.LabelFrame(
            self.root,
            text="Selected separator",
            padding=(10, 6),
        )
        path_form.pack(fill=tk.X, padx=16, pady=(2, 4))
        ttk.Label(path_form, text="Role").grid(row=0, column=0, sticky=tk.W)
        role_box = ttk.Combobox(
            path_form,
            width=31,
            state="readonly",
            values=ALL_ROLES,
            textvariable=self.role_var,
        )
        role_box.grid(row=0, column=1, padx=(5, 16), sticky=tk.W)
        role_box.bind("<<ComboboxSelected>>", self._role_changed)

        ttk.Label(path_form, text="Visibility").grid(
            row=0,
            column=2,
            sticky=tk.W,
        )
        visibility_box = ttk.Combobox(
            path_form,
            width=18,
            state="readonly",
            values=VISIBILITY_LABELS,
            textvariable=self.visibility_var,
        )
        visibility_box.grid(row=0, column=3, padx=(5, 16), sticky=tk.W)
        visibility_box.bind("<<ComboboxSelected>>", self._form_changed)

        ttk.Label(path_form, text="Confidence").grid(
            row=0,
            column=4,
            sticky=tk.W,
        )
        confidence_box = ttk.Combobox(
            path_form,
            width=10,
            state="readonly",
            values=CONFIDENCE_LABELS,
            textvariable=self.confidence_var,
        )
        confidence_box.grid(row=0, column=5, padx=(5, 16), sticky=tk.W)
        confidence_box.bind("<<ComboboxSelected>>", self._form_changed)

        ttk.Label(path_form, text="Path notes").grid(
            row=1,
            column=0,
            sticky=tk.W,
            pady=(6, 0),
        )
        path_notes_entry = ttk.Entry(
            path_form,
            textvariable=self.path_notes_var,
        )
        path_notes_entry.grid(
            row=1,
            column=1,
            columnspan=5,
            sticky=tk.EW,
            padx=(5, 0),
            pady=(6, 0),
        )
        path_notes_entry.bind("<KeyRelease>", self._form_changed)
        path_form.columnconfigure(5, weight=1)

        notes_frame = ttk.Frame(self.root, padding=(16, 2))
        notes_frame.pack(fill=tk.X)
        ttk.Label(notes_frame, text="Sample notes").pack(side=tk.LEFT)
        sample_notes_entry = ttk.Entry(
            notes_frame,
            textvariable=self.sample_notes_var,
        )
        sample_notes_entry.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=(8, 0),
        )
        sample_notes_entry.bind("<KeyRelease>", self._sample_notes_changed)

        controls = ttk.Frame(self.root, padding=(16, 5, 16, 12))
        controls.pack(fill=tk.X)
        ttk.Button(
            controls,
            text="Previous",
            command=self._previous,
        ).pack(side=tk.LEFT)
        ttk.Button(
            controls,
            text="Undo point",
            command=self._undo_point,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            controls,
            text="Clear selected points",
            command=self._clear_selected_points,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            controls,
            text="Remove selected separator",
            command=self._remove_selected_separator,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(
            controls,
            textvariable=self.status_var,
        ).pack(side=tk.LEFT, padx=(14, 0))
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
        self.image_gray = self._image_for_sample(self.sample)
        annotation = self.store.annotation_for(self.sample)
        self.paths = records_from_annotation(annotation)
        self.sample_notes = (
            annotation.get("notes", "") if annotation is not None else ""
        )
        self.sample_notes_var.set(self.sample_notes)
        self.active_role = ALL_ROLES[0]
        self.role_var.set(self.active_role)
        self._load_active_form()
        self.dirty = False
        self._refresh_header()
        self._refresh_images()
        self._refresh_path_summary()

    def _refresh_header(self) -> None:
        sample = self.sample
        reference = sample["reference_global"]
        bounds = sample["roi_bounds_global"]
        policy = (
            "HELD-OUT — do not use before method/parameter freeze"
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

    def _display_records(self) -> list[dict]:
        return [
            self.paths[role].to_record()
            for role in ALL_ROLES
            if self.paths[role].is_touched()
        ]

    def _refresh_images(self) -> None:
        context, _scale_x, _scale_y = display_helpers.resize_for_display(
            self._context_overlay(),
            500,
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

        overlay_bgr = render_roi_overlay(
            self.image_gray,
            self.sample,
            self._display_records(),
            selected_role=self.active_role,
        )
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        display, self.roi_scale_x, self.roi_scale_y = (
            display_helpers.resize_for_display(
                overlay_rgb,
                900,
                440,
                nearest=True,
                allow_enlarge=True,
            )
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

    def _refresh_path_summary(self) -> None:
        lines = []
        for role in ALL_ROLES:
            draft = self.paths[role]
            requirement = (
                "required" if role in REQUIRED_ROLES else "optional"
            )
            visibility = draft.visibility or "not annotated"
            confidence = draft.confidence or "—"
            lines.append(
                f"{ROLE_SHORT_LABELS[role]} ({requirement}): "
                f"{visibility}, confidence={confidence}, "
                f"points={len(draft.control_points)}"
            )
        self.path_summary_var.set("\n".join(lines))

    def _sync_active_form(self) -> None:
        draft = self.paths[self.active_role]
        draft.visibility = self.visibility_var.get()
        draft.confidence = self.confidence_var.get()
        draft.notes = self.path_notes_var.get()

    def _load_active_form(self) -> None:
        draft = self.paths[self.active_role]
        self.visibility_var.set(draft.visibility)
        self.confidence_var.set(draft.confidence)
        self.path_notes_var.set(draft.notes)

    def _role_changed(self, _event=None) -> None:
        self._sync_active_form()
        self.active_role = self.role_var.get()
        self._load_active_form()
        self._refresh_images()
        self._refresh_path_summary()

    def _form_changed(self, _event=None) -> None:
        self._sync_active_form()
        self.dirty = True
        self._refresh_path_summary()

    def _sample_notes_changed(self, _event=None) -> None:
        self.sample_notes = self.sample_notes_var.get()
        self.dirty = True

    def _add_point_from_click(self, event) -> None:
        bounds = self.sample["roi_bounds_global"]
        x_global, y_global = display_helpers.map_display_point_to_source(
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
            self.paths[self.active_role].add_point(
                x_global,
                y_global,
                bounds,
            )
        except ValueError as error:
            messagebox.showwarning(
                "Control-point order",
                str(error),
                parent=self.root,
            )
            return
        self.dirty = True
        self._refresh_images()
        self._refresh_path_summary()

    def _queue_magnifier(self, event) -> None:
        self.magnifier_pointer = (event.x, event.y)
        if self.magnifier_after_id is None:
            self.magnifier_after_id = self.root.after(
                display_helpers.MAGNIFIER_REFRESH_MS,
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
        source_point = display_helpers.map_display_point_to_source(
            pointer_x,
            pointer_y,
            self.roi_scale_x,
            self.roi_scale_y,
            self.image_gray.shape,
            bounds["x0"],
            bounds["y0"],
        )
        magnifier = display_helpers.create_magnifier_display(
            self.image_gray,
            source_point,
        )
        self.magnifier_photo = ImageTk.PhotoImage(
            Image.fromarray(magnifier),
            master=self.root,
        )
        magnifier_x, magnifier_y = (
            display_helpers.magnifier_canvas_position(
                pointer_x,
                pointer_y,
                self.magnifier_photo.width(),
                self.magnifier_photo.height(),
                self.roi_display.shape[1],
                self.roi_display.shape[0],
            )
        )
        self.roi_canvas.delete("magnifier")
        self.roi_canvas.create_image(
            magnifier_x,
            magnifier_y,
            anchor=tk.NW,
            image=self.magnifier_photo,
            tags=("magnifier",),
        )
        self.roi_canvas.create_rectangle(
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
        self.magnifier_photo = None

    def _undo_point(self) -> None:
        self.paths[self.active_role].undo()
        self.dirty = True
        self._refresh_images()
        self._refresh_path_summary()

    def _clear_selected_points(self) -> None:
        self.paths[self.active_role].clear_points()
        self.dirty = True
        self._refresh_images()
        self._refresh_path_summary()

    def _remove_selected_separator(self) -> None:
        self.paths[self.active_role].reset()
        self._load_active_form()
        self.dirty = True
        self._refresh_images()
        self._refresh_path_summary()

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
        self._sync_active_form()
        self.sample_notes = self.sample_notes_var.get()
        separators = ordered_records(self.paths)
        errors = annotation_validation_errors(
            self.sample,
            separators,
            self.sample_notes,
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
            self.paths,
            self.sample_notes,
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
                "The final frozen sample has been saved.",
                parent=self.root,
            )

    def _on_close(self) -> None:
        if self._confirm_discard_if_dirty():
            self._hide_magnifier()
            self.root.destroy()


def main() -> None:
    manifest = load_json(MANIFEST_PATH)
    manifest_errors = validate_manifest(manifest)
    if manifest_errors:
        raise ValueError(
            "Frozen separator GT manifest is invalid:\n- "
            + "\n- ".join(manifest_errors)
        )
    store = AnnotationStore()
    root = tk.Tk()
    patchlevel = str(root.tk.call("info", "patchlevel"))
    if not display_helpers.reliable_tk_runtime(patchlevel):
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
    SeparatorPathAnnotatorApp(root, manifest, store)
    root.mainloop()


if __name__ == "__main__":
    main()
