# Pull Completeness Audit

Audit date: 2026-05-24

This package is a targeted reviewer/audit pull, not a full mirror of the remote workspaces. The full server directories include official datasets, cached embeddings, trained model binaries, Python environments, logs and exploratory outputs that are intentionally excluded.

## Remote roots checked

- `/mnt/afs/250010150/zhensheng/trimole_hybrid`
- `/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1`
- `/mnt/afs/250010150/zhensheng/KPGT`

## What was pulled

The package includes:

- Core Trimole source code.
- KPGT graph-encoder source code.
- ChemBERTa, KPGT and UniMol/EPT embedding wrappers.
- Multimodal fusion model and training code.
- Chemistry-prior sidecar scripts.
- Classical endpoint-head and prediction-zoo scripts.
- Validation-only endpoint-selection, formal ablation and source-audit scripts.
- Lightweight result and audit artifacts from `results_strict`: `README`, `txt`, `csv`, `json`, `xlsx`, `py`, `sh`, `yaml` and `yml` files below 20 MB.

## What remains intentionally excluded

- Official TDC datasets and local data copies.
- `data_new` and other exploratory split data.
- Model weights and serialized estimators (`.pt`, `.pth`, `.ckpt`, `.pkl`, `.joblib`, `.bin`, `.h5`).
- Cached arrays and embeddings (`.npy`, `.npz`, `.parquet`, `.feather`).
- Python environments, `__pycache__`, logs, temporary image/structure caches and other non-audit artifacts.
- Files larger than 20 MB inside `results_strict`.

## Directory coverage after supplemental pull

After the supplemental lightweight pull:

- `trimole_hybrid/results_strict`: 155 local top-level directories out of 158 remote top-level directories.
- `trimole_ept_swap_v1/results_strict`: 165 local top-level directories out of 176 remote top-level directories.

The remaining missing top-level result directories were checked manually. They contained no small relevant audit/source files under the pull rules. They were cache/log/temp directories, empty directories, or directories containing only excluded non-audit artifacts.

Examples of intentionally unpulled directories:

- `__pycache__`
- `logs`
- `tmp_external_case_rdkit`
- empty exploratory result directories
- directories containing only non-target binary/cache artifacts

## Model family coverage check

The following manuscript model families are represented in the package:

- ChemBERTa sequence wrappers: present.
- KPGT graph wrappers and KPGT source: present.
- UniMol/EPT/3D wrappers and EPT routing summaries: present.
- Multimodal fusion model code: present.
- Chemistry-prior sidecars: present.
- Classical endpoint heads and sidecar result summaries: present.
- Prediction-level ensembles and seedbagging scripts: present.
- Formal benchmark, ablation and case-study audit artifacts: present.

See `MODEL_FAMILY_INDEX.md` for exact file paths.
