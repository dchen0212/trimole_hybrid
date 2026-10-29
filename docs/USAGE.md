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

## Fingerprints and descriptors

`prepare_features.py --task TASK` reads the existing task record. It generates the original chemical basis in fixed column order: Morgan radius-2 count fingerprint (1,024), Avalon count fingerprint (1,024), ErG, then the recorded RDKit descriptor list. The expanded blocks retain MACCS's 167 columns and the recorded 2,048-column fingerprint families. The command writes an `X` matrix and a `.features.json` sidecar with the ordered column names, RDKit version and hashes. Use the pinned environment; descriptor definitions can vary with RDKit versions.

The six fixed chemical endpoints keep their task-specific profiles. For an embedding-plus-chemical profile, supply `--embeddings embeddings.npz` containing its recorded `C`, `K` or `E` blocks. These chemical models use the original embedding blocks; the feature-head `E_normalized` interface is a different input. If both inputs provide row identifiers, they must match. `--block PROFILE` exposes the recorded component bases for other composite models. Invalid SMILES cause an error; rows are not silently dropped, reordered or canonicalized. Feature scaling and feature selection remain inside the original fitting pipelines.

## Classical fitting and prediction

A classical fitting NPZ contains `X` and `y`, plus `smiles` for scaffold fold-bag models. `X` must be on the feature basis and column order named in the task configuration. Supply only the rows belonging to the recorded fitting partition. A query NPZ contains `X` on the same basis.

The fixed classical entry point supports BBB ExtraTrees, repeated AMES fingerprint XGBoost, the six recorded fixed chemical endpoints and the target-only CYP3A4 substrate SVMs. AMES uses `--run-index 1` through `5` for its recorded repeat groups. A refit uses recorded settings; it does not substitute for a missing original serialized estimator when comparing exact historical predictions.

For CYP3A4 substrate, feature NPZs contain `K`, `CK` and, for fitting, `y`. The three SVM heads use their recorded PCA/scaling, kernel parameters and equal rank averaging.

Other composite models use the recorded feature-head components and task-specific implementations linked by the paper-to-code guide. Encoder fine-tuning for the hepatocyte model uses the archived task-specific implementation and settings. It is not replaced by a generic feature-head fit.

## Prediction combinations

`--component-predictions` takes an NPZ with keys from the task JSON's `fixed_combination.component_order`. Each array is one-dimensional and uses the same rows and order. Apply the operation per recorded formal run before summarizing runs. Rank and z-score combinations use the complete requested prediction batch, following the recorded implementation, and their scores need not be calibrated probabilities.

The four additional fixed procedures use these NPZ inputs:

| Task | Input keys | Fixed operation |
|---|---|---|
| Microsome clearance | `seedbag_heads` with shape `(5, n_rows)` | Mean of the five raw predictions in one formal run |
| PPBR | `seedbag_heads`, `fp_only` | 0.7 times the five-head raw mean plus 0.3 times the fixed fingerprint stream |
| CYP2D6 substrate | `seedbag_heads`, `xl`, `descriptor_sidecar_v2` | Z-score each AP seed separately; combine with z-scored XL and descriptor streams at 0.8/0.1/0.1; then average the five combined vectors |
| Hepatocyte clearance | `c060`, `c070`, `c008`, `kpgt_finetuned`, `kpgt_staged`, `unimol` | Average the ranks of the three frozen heads, then apply the recorded successive 0.5, 0.25 and 0.25 ranking blends |

For the first three tasks, `--run-index 1` through `5` selects formal seed groups `[1–5]`, `[11–15]`, `[21–25]`, `[31–35]` and `[41–45]`. Optional `seed_ids` must match that group. Individual vectors named `seed_1`, etc. can replace `seedbag_heads`. Microsome and PPBR also accept an already-completed `seedbag_mean` vector. CYP2D6 requires the five raw AP vectors because normalization precedes averaging. Hepatocyte also accepts `frozen_feature_heads` as the three-row matrix or its completed rank mean. Each ranking stage is retained; the nested blends are not flattened.

```bash
python predict.py --task cyp2d6_substrate_carbonmangels --run-index 1 \
  --component-predictions outputs/cyp2d6_run1_components.npz \
  --output outputs/cyp2d6_run1_final.npy
```

All component vectors must refer to identical rows in identical order. The command checks dimensions, finite values and any supplied seed IDs. It cannot infer molecular identity from scores alone. A lone `prediction` array remains available for exporting an already-completed task output. [Source bindings](FIXED_PIPELINE_SOURCES.json) record the original functions and fixed settings.

Run the source-derived regression fixtures with:

```bash
python -m unittest discover -s tests -v
```

The fixtures cover chemical column order and all four final combinations. They do not refit benchmark models or replace reported scores.

## Checkpoints

Weights are not attached to the current release. `checkpoints/manifest.json` lists 155 located artifacts and their hashes; some tasks have only partial original component sets, and eight have no saved original estimator. The model guide records this status. Only load checkpoints from trusted sources with the expected hash.

## Reproduce supplementary analyses

Current supplementary tables and figures retain their existing paths and downloads. The result-to-code guide links their analysis implementations in the immutable research archive. These analysis and search programs are separate from the fixed model training/prediction entry points.
