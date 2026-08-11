# Output center-darkness safety fix — release-candidate summary

Date: 2026-08-10

Status: accepted local release-candidate safety fix after `c470970`.
This commit is not pushed and is not a formal release.

## Safety invariant

The final reported left and right basin centers must be consistent with the
robust dark interior in the three evidence bands nearest the reference row.
This closes the general failure mode where a weak or missing separator merges
multiple physical periods and the geometric basin center lands on the
internal bright ridge.  The invariant applies only to the reported basins;
Candidate 4 (`click_local_structural_fallback`) is not implemented.

## Human validation

- All 8 reviewed non-Stripe01/09 successes rejected by the output invariant
  were confirmed `wrong_detection` (6 Stripe10 and 2 Stripe12).
- No human-confirmed correct detection was rejected.
- Stripe01 and Stripe09 remain out of scope and are not release blockers.

## Targeted regression

- Restored `should_detect`: 18/18 retained.
- Existing `correct_detection`: 123/123 retained, with zero identity or
  geometry regression.
- New-success `correct_detection`: 71/71 retained.
- Existing `valid_unavailable`: 39/39 retained; zero false release.

## Full 2,800-point regression

| Metric | Original baseline | Final RC | Change |
|---|---:|---:|---:|
| Success | 1,895 | 1,970 | +75 |
| Unavailable | 905 | 830 | -75 |
| Suspicious success | 1,074 | 1,147 | +73 |

Coordinate-level transitions from the original baseline to the final RC:

- unavailable → success: 91
- success → unavailable: 16
- success → success: 1,879
- unavailable → unavailable: 814

Relative to the version immediately before the output invariant, the invariant
changed 16 successes to unavailable, introduced no new successes, and caused
zero identity or geometry changes among the 1,970 retained successes.

The full replay used the existing 2,800 coordinates and the formal desktop
runtime.  It did not implement Candidate 4 or introduce filename/coordinate
special cases.
