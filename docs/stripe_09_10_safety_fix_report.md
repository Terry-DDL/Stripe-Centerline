# Stripe 9/10 safety-fix report

Date: 2026-07-24

## Ground truth

- 40 manually reviewed points: 20 from Stripe 9 and 20 from Stripe 10.
- 30 points have a valid nearest-neighbor pair.
- 10 points are negative cases where the detector must not report a pair.
- Center tolerance: ±3 px.

## Before and after

| Metric | Before | Final |
|---|---:|---:|
| Detection success rate (all 40 cases) | 50.0% | 72.5% |
| Correct nearest-neighbor rate (30 valid cases) | 50.0% | 96.7% |
| False-success rate | 12.5% | 0.0% |
| Errors among successful results | 25.0% | 0.0% |
| Left center MAE | 1.700 px | 0.147 px |
| Right center MAE | 3.544 px | 0.168 px |
| Click classification accuracy | not recorded | 100% |
| Confirmed screenshot wrong successes | not recorded | 0 |

The final run passes every delivery gate. Among 29 successful results,
21 selected adaptive thresholding and 8 selected Otsu. No rotated result was
needed for these 40 near-vertical review points.

## Implemented behavior

- Original and rotated geometry spaces independently arbitrate adaptive/Otsu.
- Geometry arbitration runs only after threshold arbitration.
- Candidate comparison is lexicographic and records the deciding field.
- Whole-image pitch is combined with local spacing, track support,
  retention, center MAD, normalized width MAD, click association,
  crossing checks, and fragment counts.
- Local neighbor-spacing anomalies remain warnings and do not erase lines.
- Crossing contradictions and non-straddling pairs remain hard invalid.
- Adaptive threshold defaults are block size 31 and C=5.
- Illegal adaptive block sizes are rejected; small ROIs reduce the block
  size to a legal odd value or disable adaptive.
- All four candidate summaries and selection reasons are saved in the
  background JSON report. Processing images remain hidden from the UI.

## Known safe failure

S09-11 contains a faint closer right stripe around x=291 with only about
15% row support. The previous result incorrectly skipped it and returned
x=306. The final detector refuses that wrong result and reports failure.
This is counted as one false negative, not a false success.

## Verification

- 40-point final regression gate: passed.
- 76 unit and real-image tests: passed.
- Real Stripe 4 and Stripe 8 +150 saturated images retained centerlines
  within 1 px.
- Sample 2 and Stripe 7 confirmed Otsu fallback behavior.
- Stripe 10 point (1250, 217) returns approximately 1240.5 / 1270.5 px.
