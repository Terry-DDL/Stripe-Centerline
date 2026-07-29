# Raw Pitch GT v1 annotation

This tool records human judgments from the original grayscale pixels. It does
not run the current detector, tracks, pitch map, thresholding, or morphology.

## Start

From the repository root:

```bash
.venv-desktop/bin/python tools/pitch_gt_annotator.py
```

The left pane shows full-image context with the frozen ROI and reference point.
The right pane shows the original grayscale ROI. Move the mouse over the ROI
for the 4× magnifier.

## Annotation procedure

1. Work through the manifest in its fixed order. The tool labels each sample as
   `development` or `HELD-OUT`.
2. Click real black-gap centers in strictly left-to-right order. For `valid`,
   click at least five consecutive centers; do not skip a gap to choose a
   clearer one.
3. Choose one label:
   - `valid`: at least five consecutive, visually confirmed centers.
   - `ambiguous`: zero or more provisional centers may remain, but they are not
     numerical ground truth.
   - `unavailable`: no center may remain.
4. Choose `high`, `medium`, or `low` confidence. Notes are optional.
5. Use **Undo center** for the last click or **Clear / restart** to reset the
   current sample. Use **Previous** to return to an earlier sample.
6. Save the annotation. **Save & Next** follows the frozen manifest order.
7. Inspect the generated numbered preview in
   `outputs/raw_pitch_gt_v1/previews/`. If a center is wrong, return to that
   sample, clear it, and annotate it again.

The manifest cannot be edited in the UI: samples, reference coordinates, ROIs,
and split membership are fixed.

## Split policy

- Development annotations are stored in
  `tests/data/raw_pitch_gt_v1/development.json`.
- Held-out annotations are physically isolated in
  `tests/data/raw_pitch_gt_v1/heldout.json`.
- Pitch method design and parameter selection may read only the development
  file.
- Held-out data may be evaluated once only after the pitch method and all
  parameters are frozen. Held-out failures must not be used to tune parameters
  or add special cases.

