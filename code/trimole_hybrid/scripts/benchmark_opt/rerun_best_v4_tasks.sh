#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afs/250010150/zhensheng/trimole

TARGET_ENV="/mnt/afs/250010150/envs/trimole_bench310"
if [ "${CONDA_PREFIX:-}" != "${TARGET_ENV}" ]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${TARGET_ENV}"
  else
    echo "ERROR: conda not found, and current env is not ${TARGET_ENV}"
    exit 1
  fi
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DGLBACKEND=pytorch
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=.

DATA_DIR="./data/data_benchmark"
OUT_ROOT="results/model_log/final_best_v4_runs"
mkdir -p "${OUT_ROOT}"

echo "Python      : $(which python)"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<empty>}"

echo "============================================================"
echo "Running task: ames"
echo "Config      : fusion=mlp, loss=auto, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/ames" --tasks "ames" --fusion-type mlp --loss-type auto --hidden-dim 128 --dropout-head 0.2 --seed 42

echo "============================================================"
echo "Running task: bbb_martins"
echo "Config      : fusion=mlp, loss=auto, hidden_dim=128, dropout_head=0.3, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/bbb_martins" --tasks "bbb_martins" --fusion-type mlp --loss-type auto --hidden-dim 128 --dropout-head 0.3 --seed 42

echo "============================================================"
echo "Running task: bioavailability_ma"
echo "Config      : fusion=mlp, loss=auto, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/bioavailability_ma" --tasks "bioavailability_ma" --fusion-type mlp --loss-type auto --hidden-dim 128 --dropout-head 0.2 --seed 42

echo "============================================================"
echo "Running task: cyp2c9_substrate_carbonmangels"
echo "Config      : fusion=mlp, loss=auto, hidden_dim=128, dropout_head=0.3, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/cyp2c9_substrate_carbonmangels" --tasks "cyp2c9_substrate_carbonmangels" --fusion-type mlp --loss-type auto --hidden-dim 128 --dropout-head 0.3 --seed 42

echo "============================================================"
echo "Running task: cyp2c9_veith"
echo "Config      : fusion=gated, loss=focal, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/cyp2c9_veith" --tasks "cyp2c9_veith" --fusion-type gated --loss-type focal --hidden-dim 128 --dropout-head 0.2 --seed 42

echo "============================================================"
echo "Running task: cyp2d6_substrate_carbonmangels"
echo "Config      : fusion=gated, loss=focal, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/cyp2d6_substrate_carbonmangels" --tasks "cyp2d6_substrate_carbonmangels" --fusion-type gated --loss-type focal --hidden-dim 128 --dropout-head 0.2 --seed 42

echo "============================================================"
echo "Running task: cyp2d6_veith"
echo "Config      : fusion=mlp, loss=auto, hidden_dim=128, dropout_head=0.3, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/cyp2d6_veith" --tasks "cyp2d6_veith" --fusion-type mlp --loss-type auto --hidden-dim 128 --dropout-head 0.3 --seed 42

echo "============================================================"
echo "Running task: cyp3a4_substrate_carbonmangels"
echo "Config      : fusion=gated, loss=focal, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/cyp3a4_substrate_carbonmangels" --tasks "cyp3a4_substrate_carbonmangels" --fusion-type gated --loss-type focal --hidden-dim 128 --dropout-head 0.2 --seed 42

echo "============================================================"
echo "Running task: cyp3a4_veith"
echo "Config      : fusion=gated, loss=focal, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/cyp3a4_veith" --tasks "cyp3a4_veith" --fusion-type gated --loss-type focal --hidden-dim 128 --dropout-head 0.2 --seed 42

echo "============================================================"
echo "Running task: dili"
echo "Config      : fusion=gated, loss=auto, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/dili" --tasks "dili" --fusion-type gated --loss-type auto --hidden-dim 128 --dropout-head 0.2 --seed 42

echo "============================================================"
echo "Running task: herg"
echo "Config      : fusion=mlp, loss=auto, hidden_dim=128, dropout_head=0.3, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/herg" --tasks "herg" --fusion-type mlp --loss-type auto --hidden-dim 128 --dropout-head 0.3 --seed 42

echo "============================================================"
echo "Running task: hia_hou"
echo "Config      : fusion=gated, loss=focal, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/hia_hou" --tasks "hia_hou" --fusion-type gated --loss-type focal --hidden-dim 128 --dropout-head 0.2 --seed 42

echo "============================================================"
echo "Running task: pgp_broccatelli"
echo "Config      : fusion=gated, loss=auto, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/pgp_broccatelli" --tasks "pgp_broccatelli" --fusion-type gated --loss-type auto --hidden-dim 128 --dropout-head 0.2 --seed 42

echo "============================================================"
echo "Running task: caco2_wang"
echo "Config      : fusion=gated, loss=auto, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/caco2_wang" --tasks "caco2_wang" --fusion-type gated --loss-type auto --hidden-dim 128 --dropout-head 0.2 --seed 42

echo "============================================================"
echo "Running task: clearance_hepatocyte_az"
echo "Config      : fusion=mlp, loss=auto, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/clearance_hepatocyte_az" --tasks "clearance_hepatocyte_az" --fusion-type mlp --loss-type auto --hidden-dim 128 --dropout-head 0.2 --seed 42

echo "============================================================"
echo "Running task: clearance_microsome_az"
echo "Config      : fusion=gated, loss=auto, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/clearance_microsome_az" --tasks "clearance_microsome_az" --fusion-type gated --loss-type auto --hidden-dim 128 --dropout-head 0.2 --seed 42

echo "============================================================"
echo "Running task: half_life_obach"
echo "Config      : fusion=gated, loss=auto, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/half_life_obach" --tasks "half_life_obach" --fusion-type gated --loss-type auto --hidden-dim 128 --dropout-head 0.2 --seed 42

echo "============================================================"
echo "Running task: ld50_zhu"
echo "Config      : fusion=gated, loss=auto, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/ld50_zhu" --tasks "ld50_zhu" --fusion-type gated --loss-type auto --hidden-dim 128 --dropout-head 0.2 --seed 42

echo "============================================================"
echo "Running task: lipophilicity_astrazeneca"
echo "Config      : fusion=mlp, loss=auto, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/lipophilicity_astrazeneca" --tasks "lipophilicity_astrazeneca" --fusion-type mlp --loss-type auto --hidden-dim 128 --dropout-head 0.2 --seed 42

echo "============================================================"
echo "Running task: ppbr_az"
echo "Config      : fusion=mlp, loss=auto, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/ppbr_az" --tasks "ppbr_az" --fusion-type mlp --loss-type auto --hidden-dim 128 --dropout-head 0.2 --seed 42

echo "============================================================"
echo "Running task: solubility_aqsoldb"
echo "Config      : fusion=gated, loss=auto, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/solubility_aqsoldb" --tasks "solubility_aqsoldb" --fusion-type gated --loss-type auto --hidden-dim 128 --dropout-head 0.2 --seed 42

echo "============================================================"
echo "Running task: vdss_lombardo"
echo "Config      : fusion=mlp, loss=auto, hidden_dim=128, dropout_head=0.2, seed=42"
echo "============================================================"
python -m trimole.pipelines.batch_run_data_new --data-new "${DATA_DIR}" --out "${OUT_ROOT}/vdss_lombardo" --tasks "vdss_lombardo" --fusion-type mlp --loss-type auto --hidden-dim 128 --dropout-head 0.2 --seed 42

