# Basin Shadow Integration v1

The active entrypoint remains:

```bash
.venv-desktop/bin/python tools/desktop_app.py
```

The visible desktop result and its formal success/failure decision still come
only from the existing production pipeline. After each formal analysis, the
same background worker runs frozen Stage 3.1 with the identical grayscale
image, reference coordinate, and the ROI bounds recorded by the formal
result.

Shadow comparison files:

- Per click: the existing interactive point output directory contains
  `basin_shadow_result.json`.
- Per-click acceptance overlay: the same directory contains
  `basin_shadow_overlay.png`.
- Cumulative JSONL:
  `outputs/basin_shadow_integration_v1/desktop/click_comparisons.jsonl`.
- Human-friendly cumulative CSV:
  `outputs/basin_shadow_integration_v1/desktop/click_comparisons.csv`.

The CSV includes formal and shadow success, shadow rejection reason,
reference relation, left/right basin centers and distances, basin/separator
IDs, and raw-pitch confidence and provenance. Re-running the same point
replaces its per-click JSON and appends another timestamped comparison row.

The per-click JSON additionally stores `shadow.acceptance_geometry`. For a
successful shadow result this is the exact atomic hypothesis: the used
separator paths and band coordinates, basin boundary paths, final centers,
and distances. For an unavailable result it stores all existing separator
candidates with their accepted/rejected state, the rejection reason, and no
final geometry. Overlay or artifact-write errors are recorded under
`shadow_overlay` and never change the formal desktop result.

The fixed ten-point Sample 2 integration check is:

```bash
.venv/bin/python tools/run_basin_shadow_issue_check.py
```

Its report is written to
`outputs/basin_shadow_integration_v1/sample2_issues/issue_report.json`.
This check verifies integration and provenance only; it does not authorize a
formal-result switch or establish prediction correctness.
