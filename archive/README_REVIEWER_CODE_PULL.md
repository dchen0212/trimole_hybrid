# Trimole-Hybrid Server Code Pull

This directory is a targeted pull from the server-side Trimole-Hybrid workspaces used for manuscript audit and reviewer access preparation.

Pull date: 2026-05-24

Remote sources:
- `<PROJECT_ROOT>/trimole_hybrid`
- `<PROJECT_ROOT>/trimole_ept_swap_v1`
- `<PROJECT_ROOT>/KPGT`

Local destination:
- local reviewer code-pull directory

## Scope

The remote workspaces are large and contain many exploratory runs:
- `trimole_hybrid`: approximately 16 GB
- `trimole_ept_swap_v1`: approximately 11 GB
- `KPGT`: approximately 2.9 GB

This pull intentionally excludes large datasets, trained model binaries, cached embeddings, logs, `data_new`, Python caches, and most exploratory run folders. It keeps the material needed to audit the manuscript claims:
- Core Trimole-Hybrid source modules.
- Benchmark, endpoint-selection, ablation, case-study and external-inference scripts.
- KPGT source and feature-extraction scripts used by the graph branch.
- Strict-run READMEs and small CSV/JSON/XLSX summary artifacts from the manuscript-relevant result directories.
- Split/source, endpoint-selection and case-study audit summaries where available as small text/table artifacts.

A supplemental lightweight pull was then performed across both `results_strict` trees to include small audit/source/result files from nearly all remote result directories. This substantially expands the local audit layer while still excluding datasets, weights, cached arrays and large artifacts. See `PULL_COMPLETENESS_AUDIT.md`.

## Included Top-Level Folders

- `trimole_hybrid/`
  - Core local package: `trimole/`
  - Benchmark and analysis scripts: `scripts/`
  - Strict rules and manifests: `notes/`, `manifests/`
  - Manuscript-relevant strict result summaries under selected `results_strict/` directories.
  - External case-study and perturbation scripts for AMES, P-gp, CYP3A4-S and related audits.

- `trimole_ept_swap_v1/`
  - Core local package: `trimole/`
  - Endpoint-construction tools: `tools/`
  - Formal endpoint, sidecar, EPT/3D, fixed-chemical and prediction-zoo scripts.
  - Manuscript-relevant strict result summaries under selected `results_strict/` directories.

- `KPGT/`
  - KPGT `scripts/` and `src/` source files.
  - `README.md`, `LICENSE`, and `environment.yml`.
  - Excludes pretrained models and large data.

## Important Exclusions

The following are intentionally not included:
- Official TDC datasets and local benchmark split files.
- `data_new` and any non-formal exploratory split material.
- Large model weights and serialized estimators (`.pt`, `.pth`, `.ckpt`, `.pkl`, `.joblib`, `.bin`, `.h5`, etc.).
- Large cached arrays or embeddings (`.npy`, `.npz`, `.parquet`, `.feather`).
- Full exploratory result trees that are not used as manuscript evidence.

Some legacy source files retained from the historical codebase may still mention `data_new` in comments, old command wrappers, or provenance-audit checks. These files are not evidence that `data_new` was used for the formal manuscript results. The manuscript-relevant strict-run READMEs and supplementary audit tables state the formal boundary: final benchmark, ablation and case-study reporting used official TDC benchmark splits, with test labels reserved for final reporting.

The manuscript supplementary tables remain the authoritative compact audit layer for the submitted article. This code pull is meant to support reviewer inspection of scripts and source provenance, not to serve as a one-command full rerun package.

## Model Family Map

This pull is not KPGT-only. KPGT is included as the graph-encoder dependency, but the package also contains ChemBERTa sequence wrappers, UniMol/EPT/3D wrappers, Trimole multimodal fusion models, chemistry-prior sidecar scripts, classical ML endpoint heads, prediction-level ensembles, seedbagging and validation-only endpoint-selection audits.

See `MODEL_FAMILY_INDEX.md` for the concrete file-by-file mapping from manuscript model families to source files and strict result artifacts.

See `PULL_COMPLETENESS_AUDIT.md` for the post-pull coverage check against the remote `results_strict` directories.

## Known Boundary

The manuscript table `Table_S11_split_and_source_provenance_audit.csv` was generated in the local manuscript package from server-side README/source-audit evidence and local audit assembly. A file with the exact generated name `v36_main_result_split_provenance_audit.csv` was not found on the server during this targeted pull. The generated audit table is available in the manuscript supplementary package:

- `supplementary_tables/Table_S11_split_and_source_provenance_audit.csv`

## File Manifest

The file `FILE_MANIFEST.txt` lists all files included in this pull.
