# Training inputs and model configurations

## Environments

The root environment file follows the recorded core experiment environment: Python 3.10.20, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, XGBoost 3.2.0, RDKit 2026.03.1 and PyTorch 2.5.1 with CUDA 12.1. The KPGT environment is separate; use `code/KPGT/environment.yml` for its graph feature extraction. The upstream checkpoint licenses apply to the corresponding pretrained encoders.

## Feature-head inputs

For the P-gp component example in the README, prepare this directory:

```text
data/pgp_broccatelli/
  train.csv
  valid.csv
  test.csv
  embeddings/
    chemberta.npy
    kpgt.npy
  embeddings_ept/
    train_ept.npy
    valid_ept.npy
    test_ept.npy
```

The CSV files contain a `label` column. ChemBERTa and KPGT arrays concatenate train, validation and test rows in that order. The EPT arrays are stored separately per split. Every feature row must correspond to the same molecular record as its CSV row.

The trainer fits EPT normalization and regression target transformations using training rows. The model mask order is ChemBERTa, KPGT, EPT. The final P-gp combination uses fixed component weights and a sigmoid of the weighted clipped logits; see the task configuration.

## Final task settings

Extract the final configuration package and open `configs/task_index.csv`. Each task JSON records the selected feature blocks, model family, fixed combination, available fitted-artifact IDs and parameter provenance. Source-level defaults are labeled separately from settings recovered from fitted-artifact metadata.

The feature-head API example trains one recorded component. The six fixed chemical endpoints, repeated AMES XGBoost, BBB ExtraTrees, target-only CYP3A4 SVMs and hepatocyte components have distinct implementations. Their recorded settings are included in the task JSONs and target-only training protocol in the supplementary-data download.

## Model files

The checkpoint manifest lists 155 located fitted artifacts. Some composite models have only part of their original fitted artifacts; eight tasks have no saved original estimator. The manifest and task configurations record availability. Later reconstructions are not substituted for original checkpoints.

## Analysis data

The supplementary-data index links each printed table and figure to its source records in the complete download. File hashes and the integrity checker are included there. The DOI-linked 2026.10.24 archive preserves the original research code and analysis record.
