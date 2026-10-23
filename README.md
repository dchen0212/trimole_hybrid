## October 2026 revision release

The updated revision source code and analysis archive is available at **[Zenodo DOI 10.5281/zenodo.23602309](https://doi.org/10.5281/zenodo.23602309)** and as `Trimole_Hybrid_revision_20261024.zip` on the [2026.10.24 release](https://github.com/dchen0212/trimole_hybrid/releases/tag/v2026.10.24).

The 2026.10.24 release ZIP includes the completed 180-s FLAML budget records and AMES replay script, together with the updated experiment scripts, corrected FLAML run ledgers, aggregate results, the 20-page Supplementary Information and a SHA-256 manifest. The `revision/` folder within that ZIP is the current revision material. Existing files in this repository retain historical source and results. Dataset, checkpoint and saved-prediction requirements are documented in the release; it is not a self-contained training bundle.

---

# Trimole-Hybrid ADMET

Source code and audit artifacts for the manuscript:

**A multimodal representation learning platform for accurate molecular ADMET prediction**

Trimole-Hybrid is a task-adaptive ADMET prediction framework that combines sequence, graph, 3D/EPT and chemistry-prior molecular evidence streams. Historical workflows used validation-based local searches and test-informed endpoint adoption. The current revision distinguishes historical results from the controlled revision experiments.

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

## Data

The official ADMET benchmark data are available from Therapeutics Data Commons (TDC). This repository does not redistribute official TDC datasets or local data copies.

Formal manuscript results used the official TDC ADMET benchmark splits. Historical exploratory files may contain old path names or comments, but the submitted benchmark, ablation and case-study reporting are documented in the strict audit tables and supplementary files.

## Reproducibility Boundary

Included in this public-upload package:

- Source code for model branches, endpoint selection, sidecars, ensembles and audits.
- Lightweight benchmark, ablation and case-study summaries through `supplementary_tables/`.
- Supplementary Information source/PDF, supplementary figures and supplementary tables used in the manuscript.

Excluded:

- Official datasets.
- Large trained weights and serialized estimators.
- Cached embeddings and large arrays.
- Split-level audit prediction files that may contain benchmark labels.
- Local Python environments and logs.
- Non-formal exploratory data such as `data_new`.

For fuller reviewer inspection, upload the companion release asset `trimole_hybrid_server_code_pull_20260524.zip` separately rather than committing it to the Git repository. This public-upload copy intentionally excludes `results_audit/`; use `supplementary_tables/` for public result auditing and provide split-level prediction audits only through a private reviewer archive if appropriate.

## Suggested Citation

Citation details will be updated after publication.

```bibtex
@article{huang2026trimolehybrid,
  title = {A multimodal representation learning platform for accurate molecular ADMET prediction},
  author = {Luo, Zhensheng and Huang, Dachen and Shao, Yanruisheng and Yu, Qinze and Li, Yu},
  journal = {Bioinformatics},
  year = {2026},
  note = {Manuscript under review}
}
```

## License

Original code and revision workflows are released under Apache-2.0; the complete terms are in `LICENSE`. Third-party components retain their own licenses and notices (see `NOTICE`).
