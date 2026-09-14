# Environment Notes

The original experiments used multiple environments because the project combines deep molecular encoders, graph feature extraction, RDKit chemistry features and classical machine-learning heads.

## Core benchmark environment

The recorded core benchmark environment is captured in `environment/trimole-benchmark-core.yml` and Supplementary Table S18. Its principal versions are:

- Python 3.10.20
- NumPy 2.2.6
- pandas 2.3.3
- scikit-learn 1.7.2
- XGBoost 3.2.0
- RDKit 2026.03.1
- PyTorch 2.5.1 with CUDA 12.1

The exact TDC package version was not captured in the original manifest. Before the revised formal run, record the installed TDC version and a checksum or release identifier for every official split file in the generated provenance artifact.

## KPGT dependency

The KPGT source dependency is included under `code/KPGT/`. Its original environment file is:

- `code/KPGT/environment.yml`

## Reproducibility note

This repository does not include official TDC datasets, trained model binaries or cached embeddings. A full rerun requires downloading the official TDC ADMET benchmark data and rebuilding intermediate features/artifacts. The environment file documents the recorded package versions; it does not replace an end-to-end installation test on the formal server.
