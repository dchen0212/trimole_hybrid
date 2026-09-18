# P0 Findings: Family-Transfer Leakage and Corrective Reruns

Date: 2026-09-18

## Confirmed risk in the submitted workflow

The submitted family-transfer results cannot be retained.

- `clearance_hepatocyte_az`: target molecules overlap the other clearance endpoint.
- `cyp3a4_substrate_carbonmangels`: target molecules overlap the CYP2C9/CYP2D6 substrate endpoints.
- The historical workflow emitted diagnostic test leaderboards and used stage-specific target holdout identities when filtering source rows.
- Exact InChIKey filtering alone did not address near-identical chemistry.

The historical raw comparison sum of 1,518 is not a molecule count and must not be quoted as one because identities are counted repeatedly across source tasks and splits.

## Final pre-split, label-free policy

1. Before any model fitting, construct one label-free identity universe from all molecules in the target dataset (`train+valid+test`). No target labels are used.
2. Apply the same frozen source-row exclusion mask in selection and final refit: remove connectivity-level InChIKey matches and Morgan-fingerprint neighbours with Tanimoto similarity `>=0.90`.
3. Audit Bemis--Murcko scaffold overlap, but do not delete an entire scaffold solely because a target scaffold is present; this audit describes domain proximity rather than exact replay.
4. Rank candidates using target validation scores across five fixed seeds only, then write `selected_candidates.csv` before opening target test labels.
5. Refit only the frozen recipe on target/source development rows that pass the pre-specified mask, and score test once.

## Source-row filtering

| Target | Source task and split | Exact target-universe removals | Additional Tanimoto removals | Rows retained |
| --- | --- | ---: | ---: | ---: |
| `clearance_hepatocyte_az` | `clearance_microsome_az` train | 485/770 | 4 | 281 |
| `clearance_hepatocyte_az` | `clearance_microsome_az` valid | 66/111 | 0 | 45 |
| `cyp3a4_substrate_carbonmangels` | `cyp2c9_substrate_carbonmangels` train | 463/467 | 0 | 4 |
| `cyp3a4_substrate_carbonmangels` | `cyp2c9_substrate_carbonmangels` valid | 67/67 | 0 | 0 |
| `cyp3a4_substrate_carbonmangels` | `cyp2d6_substrate_carbonmangels` train | 460/465 | 0 | 5 |
| `cyp3a4_substrate_carbonmangels` | `cyp2d6_substrate_carbonmangels` valid | 67/67 | 0 | 0 |

## Independently validated corrected results

The verified runner and its tests were frozen in code snapshot `898ecc21e9dfe8c6ea6b9bd4708bb261067475fb` immediately after the formal reruns. Independent validation confirmed five selected-only prediction files, official test-label alignment, validation-only candidate selection, per-seed metrics, mean, sample standard deviation and ensemble score.

| Target | Submitted score | Corrected five-seed score | Ensemble score | Selected feature set |
| --- | ---: | ---: | ---: | --- |
| `clearance_hepatocyte_az` | 0.552040 +/- 0.009650 | 0.473590 +/- 0.008724 | 0.476533 | `fp` |
| `cyp3a4_substrate_carbonmangels` | 0.725249 +/- 0.004760 | 0.626605 +/- 0.034508 | 0.643535 | `fp_chemberta_kpgt_ept` |

The submitted values and associated top-reference claims are superseded. The corrected values are carried through the main tables, figures, supplement and response letter.

## Artifact coverage

The revision audit has sample-level predictions for all 22 benchmark tasks. The strict common-pool comparison additionally provides five seed-specific predictions for all eight controlled strategies on all 22 tasks. Its hierarchical bootstrap resamples both seeds and test samples. Historical comparisons that lack seed-specific public predictions remain descriptive and are explicitly labelled as such.
