# Trimole-Hybrid

Code and final task configurations for **Task-wise model selection for heterogeneous ADMET prediction**.

![Trimole-Hybrid](docs/architecture.png)

## Installation

```bash
git clone https://github.com/dchen0212/trimole_hybrid.git
cd trimole_hybrid
conda env create -f environment.yml
conda activate trimole-hybrid
```

Pretrained encoder feature extraction uses the upstream [KPGT](https://github.com/lihan97/KPGT) and [Uni-Mol](https://github.com/deepmodeling/Uni-Mol) environments. Feature interfaces are in [code/trimole/embeddings](code/trimole/embeddings/).

## Models and configurations

Each of the 22 tasks has one JSON file under [configs/tasks](configs/tasks/). Start with [configs/task_index.csv](configs/task_index.csv). The files specify model components, recorded training settings, formal seeds, fixed combination weights and available checkpoint identifiers.

| Location | Contents |
|---|---|
| [code/trimole](code/trimole/) | Molecular feature interfaces, feature-head architecture, losses and training implementation |
| [code/trimole_hybrid](code/trimole_hybrid/) | Fixed classical fitting, component prediction and recorded prediction combinations |
| [configs/tasks](configs/tasks/) | Final configuration records for all 22 tasks |
| [train.py](train.py) / [predict.py](predict.py) | Training and prediction entry points |
| [checkpoints/manifest.json](checkpoints/manifest.json) | Checkpoint identifiers, task/seed bindings, sizes and SHA256 hashes |

Checkpoint weights are not attached yet. The manifest lists located artifacts, including partial component sets; [model availability](models/) records which files are available and which original estimators were not saved.

## Usage

Inspect a final configuration:

```bash
python train.py --task pgp_broccatelli --inspect
```

Train one recorded feature-head component:

```bash
python train.py --task pgp_broccatelli --component layerwise --seed 1 \
  --data-dir data/pgp_broccatelli --output outputs/pgp_layerwise_seed1
```

Fit a recorded classical model on prepared fitting features:

```bash
python train.py --task bbb_martins --seed 1 \
  --features data/bbb_martins/fit_features.npz --output outputs/bbb_seed1.joblib
```

Predict with the saved classical model:

```bash
python predict.py --task bbb_martins --checkpoint outputs/bbb_seed1.joblib \
  --features data/bbb_martins/query_features.npz --output outputs/bbb_predictions.npy
```

Apply a recorded task combination to aligned component predictions:

```bash
python predict.py --task pgp_broccatelli \
  --component-predictions outputs/pgp_components.npz --output outputs/pgp_final.npy
```

For that example, the NPZ keys are `layerwise`, `kpgt_ept` and `kpgt`, with one prediction vector per component in identical row order. The fixed weights and transformation come from the task JSON. These commands do not search configurations. See [input formats and task-specific training](docs/USAGE.md).

## Data and supplementary results

- [Dataset definitions and splits](supplementary_tables/Table_S1_dataset_provenance.csv)
- [20-page supplement](supplementary/supplementary.pdf), [tables](supplementary_tables/) and [Figures S1–S3](supplementary_figures/)
- [Complete supporting data](https://github.com/dchen0212/trimole_hybrid/releases/download/v2026.10.28/Supplementary_Data.zip) and [table/figure source index](supplementary_data/)
- [Final model code/configuration package](https://github.com/dchen0212/trimole_hybrid/releases/download/v2026.10.29-models/Final_Model_Configuration_and_Code.zip) and [checkpoint manifest](checkpoints/manifest.json)
- [Paper-to-code guide](docs/RESULT_CODE_MAP.md)

## Research archive

The complete pre-cleanup repository, including configuration-search and analysis scripts, is preserved in the [archive branch](https://github.com/dchen0212/trimole_hybrid/tree/codex/archive-before-model-cleanup-20261029) and its [pinned source snapshot](https://github.com/dchen0212/trimole_hybrid/tree/6c263364f79b8c8f21fbf482812636fb801ec233). The [research archive DOI](https://doi.org/10.5281/zenodo.23602309) remains unchanged. [Archive navigation](archive/) links the earlier materials. Current manuscript and supplementary downloads retain their existing addresses.

## Citation

If you use Trimole-Hybrid, please cite the manuscript:

```bibtex
@unpublished{luo2026trimolehybrid,
  title = {Task-wise model selection for heterogeneous ADMET prediction},
  author = {Luo, Zhensheng and Huang, Dachen and Shao, Yanruisheng and Yu, Qinze and Li, Yu},
  year = {2026},
  note = {Manuscript},
  url = {https://github.com/dchen0212/trimole_hybrid}
}
```

The manuscript has not yet been published. The DOI above identifies the code archive.

## License

Original code is licensed under [Apache-2.0](LICENSE). Third-party code retains its own license; see [NOTICE](NOTICE).
