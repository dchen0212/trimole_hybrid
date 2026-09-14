# Family-Transfer Leakage-Control Protocol

This protocol replaces the historical pooled-family `v1` scripts for the revised manuscript. Historical scripts remain in the repository solely to preserve provenance and are blocked by default.

## Molecular identity

SMILES are parsed with RDKit and represented by the first block of the standard InChIKey. This connectivity-level key is deliberately more conservative than literal or canonical-SMILES matching: stereochemical variants with the same molecular connectivity are treated as overlapping for exclusion and audit purposes.

## Selection stage

For each target endpoint, candidate feature sets and XGBoost configurations are trained using only pooled training rows. Before pooling, every row whose molecular identity occurs in the target validation or target test split is removed. Candidate ranking uses only the target validation metric across five fixed seeds. Test labels and test scores are not used to rank, filter, blend or otherwise choose candidates.

## Final-refit stage

After one candidate is frozen for a target, source and target training/validation rows are pooled. Every row whose identity occurs in the target test split is removed. The number of boosting rounds for each seed is fixed from that seed's validation-stage best iteration. The selected model is then refit without an evaluation set, so target validation rows are not simultaneously used as training rows and early-stopping observations.

## Test stage

The official target test metric is evaluated only for the frozen candidate. The revised script does not create test-based candidate leaderboards or selectors. It writes separate artifacts for candidate validation scores, selected candidates, selected-only test results, overlap filtering and input provenance.

## Commands

Run the standalone overlap audit before model fitting:

```bash
python tools/audit_family_transfer_overlap_v1.py \
  --data-root data/data_benchmark_official_v1 \
  --out-root results_strict/family_transfer_overlap_audit_v1
```

Run the two formal family-transfer experiments:

```bash
python tools/run_family_transfer_leakage_safe_v2.py \
  --family cyp_substrate \
  --data-root data/data_benchmark_official_v1

python tools/run_family_transfer_leakage_safe_v2.py \
  --family clearance \
  --data-root data/data_benchmark_official_v1
```

The commands must be executed in the recorded formal environment. Do not update manuscript scores until the generated provenance files, overlap audit and selected-only test outputs have been independently checked.
