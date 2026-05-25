# Environment Notes

The original experiments used multiple environments because the project combines deep molecular encoders, graph feature extraction, RDKit chemistry features and classical machine-learning heads.

## Core dependencies

Typical dependencies include:

- Python 3.x
- PyTorch
- NumPy
- pandas
- scikit-learn
- XGBoost
- RDKit
- tqdm
- TDC

## KPGT dependency

The KPGT source dependency is included under `code/KPGT/`. Its original environment file is:

- `code/KPGT/environment.yml`

## Reproducibility note

This repository does not include official TDC datasets, trained model binaries or cached embeddings. A full rerun requires downloading the official TDC ADMET benchmark data and rebuilding intermediate features/artifacts.
