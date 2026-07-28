# Raw pitch final held-out v1

This frozen set contains 15 coordinates from five source images that were not
used by the original development or held-out sets. Sampling coordinates,
images, hashes, ROIs, direction, ordering, and evaluation role were fixed
before annotation.

Start the human-only annotator:

```bash
.venv-desktop/bin/python tools/final_pitch_gt_annotator.py
```

Use the same annotation rules as raw pitch GT v1:

- Save `valid` only when at least five consecutive real black-gap centers can
  be confirmed from left to right.
- Use `ambiguous` when provisional centers may be visible but numerical GT is
  not reliable.
- Use `unavailable` with no centers when the ROI cannot be annotated.
- Always select confidence and inspect the numbered preview after saving.

Answers are isolated in
`tests/data/raw_pitch_final_heldout_v1/annotations.json`. They must not be read
by method design, parameter selection, or debugging. After annotation, the
frozen v2 algorithm may evaluate all 15 samples exactly once in a clean
detached worktree created from commit
`41fe2f3b2870ec920ebd443307467e4bc02b1cee`.

The final report must distinguish auxiliary numerical coverage from formal
`success_eligible`/usable coverage. A medium or low auxiliary estimate is not
a formal success.
