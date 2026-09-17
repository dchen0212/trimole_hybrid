# P0 Findings: Family-Transfer Leakage

Date: 2026-09-17

## Confirmed risk in the submitted workflow

The submitted family-transfer results cannot be retained without rerunning.

- `clearance_hepatocyte_az`: 108 unique target-test molecular identities occur in the other clearance endpoint's train/validation data.
- `cyp3a4_substrate_carbonmangels`: 108 unique target-test molecular identities occur in the other CYP-substrate endpoints' train/validation data.
- The historical v1 scripts computed test scores for every candidate and emitted diagnostic test leaderboards.
- The historical final refit included validation rows in training while also passing the same validation rows as the early-stopping evaluation set.

The raw comparison sum of 1,518 is not a molecule count and must not be quoted as one; it double-counts identities across source-task and source-split comparisons.

## Leakage-safe policy now implemented

1. During candidate selection, pool train rows only and remove every source row whose connectivity-level InChIKey occurs in the target validation or target test split.
2. Rank candidates using the target validation metric across five fixed seeds only.
3. Write `selected_candidates.csv` before reading target test labels.
4. During final refit, pool train and validation rows, remove every source row whose identity occurs in target test, and fix boosting rounds from validation-stage best iterations.
5. Score only the frozen candidate on test.

## Exact filter preview

| Target | Stage | Rows removed | Rows before filtering |
| --- | ---: | ---: | ---: |
| `clearance_hepatocyte_az` | selection | 137 | 1,618 |
| `clearance_hepatocyte_az` | final | 109 | 1,851 |
| `cyp3a4_substrate_carbonmangels` | selection | 204 | 1,400 |
| `cyp3a4_substrate_carbonmangels` | final | 162 | 1,601 |

## Independently validated rerun results

Both formal reruns completed from Git commit `18b0f6a3d2f625958cfbfdddccb4b0ce331a8ede` in a clean checkout. Independent validation confirmed five selected-only test prediction files, official test-label alignment, validation-only candidate selection, per-seed metrics, mean, sample standard deviation and ensemble score.

| Target | Submitted score | Leakage-safe score | Selected feature set |
| --- | ---: | ---: | --- |
| `clearance_hepatocyte_az` | 0.552040 +/- 0.009650 | 0.273710 +/- 0.020235 | `fp_chemberta_kpgt_ept` |
| `cyp3a4_substrate_carbonmangels` | 0.725249 +/- 0.004760 | 0.655628 +/- 0.007590 | `fp` |

The submitted scores and their associated top-reference claims must be replaced. The new results are not top-reference results under the submitted comparison table.

## Artifact coverage

The revision audit now has sample-level final predictions for all 22 benchmark tasks:

- 10 tasks from the submitted prediction inventory;
- 5 tasks recovered and metric-verified from the fixed-endpoint archive;
- 5 tasks reconstructed from frozen v29 formulas and independently re-scored from the materialized CSV files;
- 2 family-transfer tasks replaced by the leakage-safe reruns above.

The v29 CSV-level score for `cyp2d6_veith` is 0.719167 +/- 0.005510. This differs slightly from the old in-memory value, 0.719172 +/- 0.005514, because CSV round-trip precision changes a small number of tied ranks in AUPRC. The materialized-file score is the reproducible audit value.

The original rerun provenance files reported `xgboost: not-installed` because the installed distribution is named `xgboost-cpu`. Non-overwriting correction records document that both the distribution and imported module were version 3.0.4; the original provenance files remain unchanged.
