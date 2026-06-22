#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/benchmark_opt/run_seed_ensemble_task.sh <task_name>"
  exit 1
fi

TASK_NAME="$1"

cd <PROJECT_ROOT>/trimole

TARGET_ENV="<ENV_ROOT>/trimole_bench310"

if [ "${CONDA_PREFIX:-}" != "${TARGET_ENV}" ]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${TARGET_ENV}"
  else
    echo "ERROR: conda not found, and current env is not ${TARGET_ENV}"
    echo "Current CONDA_PREFIX=${CONDA_PREFIX:-<empty>}"
    exit 1
  fi
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DGLBACKEND=pytorch
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=.

DATA_DIR="./data/data_benchmark"
OUT_ROOT="results/model_log/seed_ensemble_${TASK_NAME}"
mkdir -p "${OUT_ROOT}"

FUSION_TYPE="${FUSION_TYPE:-mlp}"
LOSS_TYPE="${LOSS_TYPE:-auto}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
DROPOUT_HEAD="${DROPOUT_HEAD:-0.3}"

SEEDS=(001 007 042 123 3407)

echo "Python      : $(which python)"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<empty>}"
echo "TASK_NAME   : ${TASK_NAME}"
echo

for seed in "${SEEDS[@]}"; do
  out_dir="${OUT_ROOT}/seed_${seed}"
  echo "============================================================"
  echo "Task   : ${TASK_NAME}"
  echo "Seed   : ${seed}"
  echo "Out dir: ${out_dir}"
  echo "============================================================"

  python -m trimole.pipelines.batch_run_data_new \
    --data-new "${DATA_DIR}" \
    --out "${out_dir}" \
    --tasks "${TASK_NAME}" \
    --fusion-type "${FUSION_TYPE}" \
    --loss-type "${LOSS_TYPE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --dropout-head "${DROPOUT_HEAD}" \
    --seed "${seed}"
done

echo
echo "Done. Seed runs are under: ${OUT_ROOT}"
