# Raw Pitch v3 Internal Validation

Status: post-hoc implementation validation on consumed data. This is not an
independent held-out evaluation and is not evidence of generalization.

Frozen configuration checksum:
`d1df32d7130d89f0d3e2c6611ad2a1bac11d97ea0252e2de297b63c916e40b49`

## Acceptance summary

- 51 consumed/development samples, including 37 human-valid samples.
- All 19 correct v2 high-confidence samples remain correct high-confidence.
- V3 has 21 success-eligible valid estimates.
- V3 success-eligible median error: 1.15%; maximum error: 4.12%.
- Confident harmonic errors: 0.
- Human ambiguous/unavailable samples released as success: 0.
- F008 correct spectral hypothesis: 37.93 px, 5/5 spectral bands.
- F009 correct spectral hypothesis: 38.28 px, 5/5 spectral bands.

## Material v2 to v3 changes

Only confidence, success eligibility, unavailable reason, or the presence of a
diagnostic estimate is considered material. Small numerical refinements with
the same state are omitted.

| Sample | Label | v2 | v3 | Reason |
|---|---|---|---|---|
| P001 | ambiguous | unavailable/no bands | unavailable/no hypothesis | No stable cross-band candidate. |
| P004 | valid | unavailable/no bands | unavailable/36.06 diagnostic | Competing unified hypotheses remain too close. |
| P005 | valid | unavailable/inconsistent | unavailable/27.79 diagnostic | Competing unified hypotheses remain too close. |
| P010 | ambiguous | unavailable/no bands | unavailable/44.40 diagnostic | Competing unified hypotheses remain too close. |
| P012 | ambiguous | low/70.30 | medium/68.72 | Joint evidence remains below the unchanged high ACF standard. |
| P019 | valid | medium/18.41 | unavailable/18.13 | A newly retained competing hypothesis prevents formal use. |
| P021 | ambiguous | medium/18.61 | harmonic unavailable/18.37 | Unresolved bidirectional 2x/3x competition. |
| P022 | ambiguous | low/54.53 | harmonic unavailable/52.86 | Unresolved bidirectional 2x/3x competition. |
| P024 | ambiguous | medium/57.12 | unavailable/51.85 | Competing unified hypotheses remain too close. |
| P031 | unavailable | unavailable/no bands | harmonic unavailable/22.88 | Retained evidence exposes an unresolved harmonic conflict. |
| P033 | unavailable | unavailable/no bands | unavailable/no hypothesis | No stable cross-band candidate. |
| P034 | unavailable | unavailable/no bands | unavailable/no hypothesis | No stable cross-band candidate. |
| P002 | valid, consumed | medium/38.10 | harmonic unavailable/38.19 | Correct diagnostic exists, but harmonic competition is unresolved. |
| P006 | valid, consumed | medium/39.63 | unavailable/38.28 | Correct diagnostic exists, but nonharmonic competition is too close. |
| P011 | ambiguous, consumed | unavailable/inconsistent | unavailable/67.67 | Competing unified hypotheses remain too close. |
| P023 | ambiguous, consumed | medium/49.12 | unavailable/45.01 | Competing unified hypotheses remain too close. |
| P026 | valid validation | harmonic unavailable/42.92 | unavailable/14.36 | Correct spectral candidate wins, but another close hypothesis prevents high confidence. |
| P032 | unavailable, consumed | unavailable/no bands | harmonic unavailable/88.10 | Retained evidence exposes an unresolved harmonic conflict. |
| F006 | valid, consumed | medium/14.59 | **high/14.52** | Five-band ACF and spectral evidence jointly meet the frozen high standard. |
| F008 | valid, consumed | medium/15.92 half-period | unavailable/37.93 | Correct 5/5 spectral hypothesis enters and wins; close competing evidence keeps it unavailable. |
| F009 | valid, consumed | low/25.58 | unavailable/26.35 | Correct 38.28 hypothesis enters the pool, but the stronger 26.35 hypothesis is not uniquely distinguishable. |
| F010 | valid, consumed | unavailable/no bands | unavailable/36.82 | Correct diagnostic is recovered, but competing evidence prevents formal use. |
| F012 | ambiguous, consumed | unavailable/no bands | unavailable/no hypothesis | No stable cross-band candidate. |
| F015 | valid, consumed | unavailable/no bands | **high/14.38** | Five-band ACF and spectral evidence jointly meet the frozen high standard. |

F011 and F014 remain correct medium-confidence diagnostics. Their evidence does
not meet the unchanged high autocorrelation requirement, so v3 does not promote
them. All other v2 high-confidence correct samples remain high-confidence.
