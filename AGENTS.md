# AGENTS.md

This is a Python + OpenCV image-processing project.

## Core rules
- Work in small steps.
- Do not make large unexplained changes.
- Before implementation, explain the plan first.
- After implementation, explain changed files, functions, parameters, and known failure cases.
- Do not rewrite unrelated files.

## Project goal
Detect the centerline of the black region near a configurable vertical reference line in grayscale stripe images.

## Assumptions
- The default reference line is the image vertical center: x = width * 0.5.
- The reference line must remain configurable for future UI adjustment.
- The default ROI is centered around the image center.
- ROI width and height must remain configurable.
- White stripes may be broken, noisy, or uneven. Do not assume they are continuous.

## Coding rules
- Keep code simple and readable for an OpenCV beginner.
- Put tunable parameters in a config object or config.py.
- Save debug images for every major processing step.
- Do not add UI, automatic ROI detection, deep learning, or complex algorithms unless explicitly requested.

## Current product entrypoint
- The active user-facing product is the local Tk desktop application:
  `tools/desktop_app.py`.
- Launch it with:
  `.venv-desktop/bin/python tools/desktop_app.py`.
- `tools/interactive_app.py` is the older Streamlit/browser application.
  Do not modify, launch, or test it unless the user explicitly asks for the
  Streamlit or browser version.
- Before any UI change, state which entrypoint will be modified and verify it
  against this section. If the request says "local version", "desktop version",
  "magnifier", or "same window", use `tools/desktop_app.py`.

## Desktop UI contract
- Preserve the live mouse-following 4× magnifier on the original image.
- Preserve the optional enlarged precise-point selection view.
- Keep the original image and precise-point view side by side when possible;
  do not automatically scroll between them.
- After analysis, refresh the existing root window. Do not open a result
  `Toplevel`, browser page, or separate result window.
- The result view should fit in one window without normal mouse scrolling.
- Show only final user-facing information:
  - original image with the ROI rectangle, reference point, and final lines;
  - enlarged final ROI;
  - reference coordinates, left/right distances, centerline spacing, stripe
    widths, and row-support values;
  - whole-image pitch safety status, local theoretical pitch, and comparison
    ratios when available;
  - important success, failure, or warning status.
- Do not show Otsu images, masks, run candidates, vote charts, rotation charts,
  or other processing-stage images in the result UI.
- Continue saving processing-stage debug images in the background for
  troubleshooting.

## Current code map
- `tools/desktop_app.py`: active desktop UI, magnifier, point selection, and
  same-window result presentation.
- `src/interactive_pipeline.py`: image-processing and measurement pipeline.
- `src/interactive_analysis.py`: click-aware stripe selection.
- `src/pitch_reference.py`: cached whole-image 2-D theoretical pitch map and
  warning-only pitch safety check.
- `src/config.py`: tunable ROI, display, magnifier, and detection parameters.
- `tests/test_desktop_app.py`: desktop helper and UI-state regression tests.
- `tests/test_pitch_reference.py`: synthetic and Stripe 9/10/11 pitch-guard
  regression tests.

## UI completion checklist
- Confirm the target entrypoint before editing.
- Verify the magnifier still works.
- Verify display scaling maps clicks back to source-image coordinates.
- Verify analysis creates no additional result window.
- Verify final and enlarged ROI images fit the configured desktop window.
- Run:
  `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'`.
