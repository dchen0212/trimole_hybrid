# Family-Transfer Leakage-Control Protocol

This protocol replaces the historical pooled-family `v1` scripts for the revised manuscript. Historical scripts remain in the repository solely to preserve provenance and are blocked by default.

## Molecular identity

SMILES are parsed with RDKit and represented by the first block of the standard InChIKey. This connectivity-level key is deliberately more conservative than literal or canonical-SMILES matching: stereochemical variants with the same molecular connectivity are treated as overlapping for exclusion and audit purposes.

## Target-wide exclusion rule

For each target endpoint, the train, validation and test structures from the already defined official split are combined into one label-free target universe before any stage-specific fitting. Every non-target source split is filtered against the same connectivity-level identity set during candidate selection and final refitting. The exclusion rule therefore does not change according to target validation or test membership and never uses a target label.

After exact exclusion, each remaining source molecule is compared with the target universe using Morgan radius-2, 2,048-bit fingerprints. Source rows with maximum Tanimoto similarity greater than or equal to 0.90 are removed. Bemis-Murcko scaffold sharing is reported as an applicability diagnostic but is not itself an exclusion criterion.

## Selection stage

Candidate feature sets and XGBoost configurations are trained using pooled training rows after the frozen target-wide filter. Candidate ranking uses only the target validation metric across five fixed seeds. Test labels and test scores are not used to rank, filter, blend or otherwise choose candidates.

## Final-refit stage

After one candidate is frozen for a target, source and target training/validation rows are pooled. The same precomputed target-wide exact-identity and high-similarity source mask is applied. The number of boosting rounds for each seed is fixed from that seed's validation-stage best iteration. The selected model is then refit without an evaluation set, so target validation rows are not simultaneously used as training rows and early-stopping observations.

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
  --target cyp3a4_substrate_carbonmangels \
  --data-root data/data_benchmark_official_v1 \
  --cross-task-tanimoto-threshold 0.90 \
  --out-root results_strict/revision_20260918_cyp3a4_substrate_presplit_similarity_v4

python tools/run_family_transfer_leakage_safe_v2.py \
  --family clearance \
  --target clearance_hepatocyte_az \
  --data-root data/data_benchmark_official_v1 \
  --cross-task-tanimoto-threshold 0.90 \
  --out-root results_strict/revision_20260918_clearance_hepatocyte_presplit_similarity_v4
```

The non-target family tasks remain eligible source datasets after molecular
overlap filtering. The `--target` option limits candidate selection and final
test scoring to the endpoint used by the manuscript; omitting it preserves the
all-family-target behavior for separate exploratory work.

The commands must be executed in the recorded formal environment. Do not update manuscript scores until the generated provenance files, overlap audit and selected-only test outputs have been independently checked.
