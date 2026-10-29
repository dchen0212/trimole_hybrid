# Training and prediction

## Final task records

`configs/tasks/` contains the exact 22 JSON records published with the final configuration package. Training reads these records; it does not perform a candidate or hyperparameter search. Fields marked as source defaults retain that status. A missing fitted artifact is not replaced with an invented checkpoint.

`train.py --task TASK --inspect` prints the components, seeds, training parameters and source bindings. Composite tasks are fitted component by component. `--component` selects an explicitly recorded feature-head component and `--seed` selects its recorded run. The task JSON defines the final prediction combination.

## Prepared feature-head inputs

For the P-gp example, prepare:

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

Each CSV contains a `label` column. ChemBERTa and KPGT arrays concatenate train, validation and test rows in that order. EPT arrays are stored per split. All blocks must use the same molecular row order and the recorded feature dimensions. The original feature-head trainer and model architecture are retained. Training uses the recorded stopping rule and writes the checkpoint and metadata to a new output directory.

For feature-head prediction, provide an NPZ with `C`, `K` and `E_normalized`. The last block must use the fitted training normalization. Query-batch statistics must not replace those training statistics. Use `--metadata path/to/meta.json` with `--sha256 HASH`, or a registered `--artifact-id` to read bound metadata. A checkpoint hash alone does not provide missing preprocessing assets.

## Classical fitting and prediction

A classical fitting NPZ contains `X` and `y`, plus `smiles` for scaffold fold-bag models. `X` must be on the feature basis and column order named in the task configuration. Supply only the rows belonging to the recorded fitting partition. A query NPZ contains `X` on the same basis.

The fixed classical entry point supports BBB ExtraTrees, repeated AMES fingerprint XGBoost, the six recorded fixed chemical endpoints and the target-only CYP3A4 substrate SVMs. AMES uses `--run-index 1` through `5` for its recorded repeat groups. A refit uses recorded settings; it does not substitute for a missing original serialized estimator when comparing exact historical predictions.

For CYP3A4 substrate, feature NPZs contain `K`, `CK` and, for fitting, `y`. The three SVM heads use their recorded PCA/scaling, kernel parameters and equal rank averaging.

Other composite models use the recorded feature-head components and task-specific implementations linked by the paper-to-code guide. Encoder fine-tuning for the hepatocyte model uses the archived task-specific implementation and settings. It is not replaced by a generic feature-head fit.

## Prediction combinations

`--component-predictions` takes an NPZ with keys from the task JSON's `fixed_combination.component_order`. Each array is one-dimensional and uses the same rows and order. Apply the operation per recorded formal run before summarizing runs. Rank and z-score combinations use the complete requested prediction batch, following the recorded implementation, and their scores need not be calibrated probabilities.

For tasks with nested or seed-bag procedures, use the recorded component implementation and order from the configuration and archive. The command rejects an incomplete combination rather than treating unrelated components as a simple average. `prediction` can contain an already-completed task output for export.

## Checkpoints

Weights are not attached to the current release. `checkpoints/manifest.json` lists 155 located artifacts and their hashes; some tasks have only partial original component sets, and eight have no saved original estimator. The model guide records this status. Only load checkpoints from trusted sources with the expected hash.

## Reproduce supplementary analyses

Current supplementary tables and figures retain their existing paths and downloads. The result-to-code guide links their analysis implementations in the immutable research archive. These analysis and search programs are separate from the fixed model training/prediction entry points.
