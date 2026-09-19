# Trimole-Hybrid ADMET

Source code and audit artifacts for the manuscript:

**Task-wise multimodal model selection and ensemble learning for molecular ADMET prediction**

Trimole-Hybrid is a task-adaptive ADMET prediction framework that combines sequence, graph, 3D/EPT and chemistry-prior molecular evidence streams. The revision runners separate validation-only selection from test scoring. Because the candidate pool was assembled after historical benchmark development, that software gate is not evidence of a prospective untouched-test evaluation.

## Repository Scope

This repository is a manuscript source and audit package. It is intended to make the implementation, endpoint-selection logic, benchmark summaries, ablation summaries and case-study artifacts inspectable.

It is **not** a one-command full rerun bundle. Large files are intentionally excluded, including official datasets, trained model binaries, cached embeddings and serialized estimators.

## Main Contents

- `code/trimole_hybrid/`: core Trimole-Hybrid package, audit scripts, benchmark utilities and case-study scripts.
- `code/trimole_ept_swap_v1/`: endpoint construction, EPT/3D routing, chemistry sidecar and prediction-zoo scripts.
- `code/KPGT/`: KPGT graph-encoder source dependency used by the graph branch.
- `supplementary.tex`: LaTeX source for the Supplementary Information.
- `supplementary/supplementary.pdf`: compiled Supplementary Information PDF.
- `supplementary_figures/`: supplementary figure source exports, including the vector Supplementary Figure S1.
- `supplementary_tables/`: supplementary workbook and CSV tables used by the manuscript.
- `docs/MODEL_FAMILY_INDEX.md`: file-by-file map from manuscript model families to concrete source files.
- `docs/PULL_COMPLETENESS_AUDIT.md`: audit of what was pulled from the server and what was intentionally excluded.
- `code/trimole_ept_swap_v1/tools/run_expanded_candidate_pool_controls_v1.py`: validation-gated expanded common-pool sensitivity analysis used for Supplementary Table S24.
- `code/trimole_ept_swap_v1/tools/run_common_pool_fair_comparison_v1.py`: primary nine-family, eight-method, five-seed strict common-pool comparison used for Supplementary Tables S24g-S24j.
- `code/trimole_ept_swap_v1/tools/run_paired_bootstrap_v1.py`: hierarchical seed-and-sample bootstrap with BH-FDR correction.
- `code/trimole_ept_swap_v1/tools/audit_common_pool_compute_budget_v1.py`: meta-model fit-count audit for the common-pool reanalysis (Table S24k). The selector and AutoML each evaluate six options, but require 550 versus 3,300 cross-validation meta fits; historical base-family wall times are unavailable, so compute is not matched end to end.
- `tools/export_revision_prediction_bundle_v1.py`: constructs a companion label-free ZIP containing all strict-pool candidate and method predictions, with row counts and source SHA-256 checksums. This 2,860-file companion is separate from the code archive because the repository does not redistribute benchmark labels or large arrays.

## Model Families

The code package contains more than the KPGT graph branch:

- ChemBERTa-style SMILES/sequence wrappers.
- KPGT graph wrappers and KPGT source.
- UniMol/EPT/3D wrappers.
- Multimodal fusion modules with MLP, gated, residual dynamic and 3D-downweighted fusion variants.
- RDKit/Morgan/descriptor chemistry-prior sidecars.
- Classical endpoint heads including XGBoost, ExtraTrees, RandomForest, LogisticRegression and Ridge-style models.
- Prediction-level blends, seedbagging and validation-only endpoint-selection scripts.

See `docs/MODEL_FAMILY_INDEX.md` for exact paths.

## Family-Transfer Audit and Test-Blind Control

The revised family-transfer protocol is implemented in:

- `code/trimole_ept_swap_v1/tools/run_family_transfer_leakage_safe_v2.py`
- `code/trimole_ept_swap_v1/tools/audit_family_transfer_overlap_v1.py`
- `docs/FAMILY_TRANSFER_LEAKAGE_CONTROL.md`

The earlier target-wide protocol assembled a label-free identity universe from target train, validation and test structures before fitting. It applied one connectivity-InChIKey exclusion set and a Morgan radius-2, 2,048-bit Tanimoto filter at 0.90 to source rows. Candidate ranking was validation-only, but source-row selection consulted target test structures. These runs are transductive sensitivity evidence, not primary independent-test results. The historical pooled-family `v1` scripts and test-split-specific exact-only reruns remain for provenance.

The `--source-policy target_only` control excludes all cross-task source rows without using target holdout identity for training-row selection. Its independently rescored five-seed results are primary for the two leakage-sensitive endpoints (Table S20e). It is not evidence that the original transfer method is test-covariate independent; in particular, hepatocyte-clearance performance falls sharply without source-task rows.

`tools/propagate_target_only_v1.py` updates the frozen benchmark and provenance CSVs from the two independently validated run directories. It marks historical ablation and subgroup rows from these endpoints as protocol-incomparable and limits their paired summaries to the remaining 20 tasks; `tools/propagate_family_transfer_v4.py` supplies the shared table-update helpers.

## Data

The official ADMET benchmark data are available from Therapeutics Data Commons (TDC). This repository does not redistribute official TDC datasets or local data copies.

Formal manuscript results used the official TDC ADMET benchmark splits. Historical exploratory files may contain old path names or comments. The two leakage-sensitive primary endpoint scores must come from the independently validated target-only runs, not the target-wide sensitivity runs.

## Reproducibility Boundary

Included in this public-upload package:

- Source code for model branches, endpoint selection, sidecars, ensembles and audits.
- Lightweight benchmark, ablation and case-study summaries through `supplementary_tables/`.
- Supplementary Information source/PDF, supplementary figures and supplementary tables used in the manuscript.
- Validation-only numerical-stability audits for all 15 complete candidate families and the strict nine-family common-pool controls reported in Tables S24g-S24j.
- Target-wide exact-identity, Bemis-Murcko scaffold and Morgan-Tanimoto sensitivity audits in Tables S20b-S20d, plus test-blind target-only primary scores in S20e.
- A compute-budget audit (S24k) and independent recalculation of all 880 common-pool scores from a label-free prediction bundle (S24l). Candidate-pool parity does not imply matched compute.

Excluded:

- Official datasets.
- Large trained weights and serialized estimators.
- Cached embeddings and large arrays.
- Split-level audit prediction files that may contain benchmark labels.
- Local Python environments and logs.
- Non-formal exploratory data such as `data_new`.

For fuller reviewer inspection, upload the companion release asset `trimole_hybrid_server_code_pull_20260524.zip` separately rather than committing it to the Git repository. This public-upload copy intentionally excludes `results_audit/`; use `supplementary_tables/` for public result auditing and provide split-level prediction audits only through a private reviewer archive if appropriate.

The separate `trimole-hybrid-common-pool-label-free-predictions.zip` is suitable as a release/Zenodo companion after author review. It contains 2,860 CSVs with only `sample_idx,prediction`, no assay labels or SMILES. Its SHA-256 is `9accee59da3a5887f549d5c8cf3186654da10a338ab8deef224b8d1f4ad88f1c`. Reproducing scores still requires the official TDC labels and the documented AqSolDB alignment; generating the original candidate arrays additionally requires the excluded encoder weights and embeddings.

## Suggested Citation

Citation details will be updated after publication.

```bibtex
@article{huang2026trimolehybrid,
  title = {Task-wise multimodal model selection and ensemble learning for molecular ADMET prediction},
  author = {Luo, Zhensheng and Huang, Dachen and Shao, Yanruisheng and Yu, Qinze and Li, Yu},
  journal = {Bioinformatics},
  year = {2026},
  note = {Manuscript under review}
}
```

## License

The repository-level original code and revision audit package are released under Apache-2.0. Third-party components retain their upstream licenses; see `LICENSE` and `NOTICE`.
