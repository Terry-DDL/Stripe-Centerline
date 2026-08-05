# Cross-image correction v1.1 manual identity audit

This record evaluates the audit-only release-unavailable change. It is not a
runtime input and does not replace or generate centerline ground truth.

- Baseline: `delivery/windows-v1-1` at `5eb6e189f2eecd8aabc15d9b119b1b738b6bc37b`
- Candidate patch: experimental commit `51ca2f5962cfad56fafbdc9f2253919cf6f7bce5`
- Manual evidence: candidate overlays were inspected to confirm the physical
  left/right black-seam identity for `S05_G02`, `S05_G05`, and `S05_G06`.
- `S05_G04` and `S10_G04` remain correct successes under the previously
  corrected human-evaluation interpretation; their older GT labels were wrong.
- The separate Stripe 5 field case at `(1589, 979)` was previously confirmed
  by manual inspection and is not part of the 14-point summary.

After manual identity confirmation, the 14-point result is:

- Correct success (9): `S05_G01`, `S05_G02`, `S05_G03`, `S05_G04`,
  `S05_G05`, `S05_G06`, `S10_G01`, `S10_G04`, `S11_G06`
- Wrong success (0)
- Unavailable (5): `S10_G02`, `S10_G03`, `S10_G06`, `S11_G04`, `S11_G05`
- Success precision: 100%
- Recognition rate: 64.3%

No GT value was inferred from the detector output, separator ID, or basin ID.
