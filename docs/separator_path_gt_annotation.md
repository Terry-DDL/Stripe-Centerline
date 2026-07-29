# Separator path GT v1

This dataset records human-visible white separator paths near a clicked dark
basin. The tool displays only the source grayscale image, the frozen ROI, the
reference point, and paths clicked by the annotator.

Launch:

```bash
.venv-desktop/bin/python tools/separator_path_gt_annotator.py
```

## Frozen dataset

- `development.json`: 26 samples. The first ten are the known Sample 2 issue
  coordinates; the remainder cover Sample 2 and Stripe 01/05/09/10/11.
- `heldout.json`: 9 samples from Sample 1, Stripe 03, and Stripe 04.
- The held-out images do not overlap separator development or the final raw
  pitch held-out images.
- Membership, split, image hashes, reference coordinates, and 500×240 ROIs
  are fixed in `manifest.json`. The UI cannot add, remove, or move samples.

Held-out answers are physically isolated. They may be annotated now, but must
not be read during separator method design or parameter selection. Evaluate
them only once after the separator method and parameters are frozen.

## What to annotate

The two clicked-basin boundaries are required:

- `left_clicked_boundary`
- `right_clicked_boundary`

Add `left_adjacent` or `right_adjacent` only when that neighboring separator
is needed to describe a possible skipped or merged basin. Do not mark every
separator in the ROI.

For each selected role:

1. Choose visibility and confidence.
2. When a path is visible, click 3–7 control points from top to bottom.
   Horizontal movement is allowed, so a path may tilt or drift.
3. For `partially_visible`, place the first and last points at the visible
   endpoints. This defines the saved visible range.
4. For `fully_visible`, distribute points over the visible ROI height.
5. Use `ambiguous` rather than guessing. It accepts either no points or 3–7
   provisional points.
6. Use `unavailable` only with no control points.

The tool rejects non-increasing y order and more than seven points. `Undo
point`, `Clear selected points`, and `Remove selected separator` affect only
the currently selected role.

After saving, inspect the preview under:

```text
outputs/separator_path_gt_v1/previews/
```

The preview must match the intended path roles and point ordering. Return to
the sample and correct it if necessary.

## Evidence boundary

These are human labels, not detector output. Existing tracks, pitch estimates,
threshold masks, morphology, basin prototypes, and automatic bright-ridge
suggestions are not imported or displayed by this tool.
