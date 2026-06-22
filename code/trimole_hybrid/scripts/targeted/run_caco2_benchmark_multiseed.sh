#!/usr/bin/env bash
set -euo pipefail

cd <PROJECT_ROOT>/trimole

OUT_ROOT="results/model_log/caco2_benchmark_multiseed"
SEEDS=(1 7 42 123 3407)

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DGLBACKEND=pytorch
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=.

mkdir -p "${OUT_ROOT}"

echo "PWD=$(pwd)"
echo "DATE=$(date)"
echo "TASK=caco2_wang"
echo "DATA=./data/data_benchmark"
echo "OUT_ROOT=${OUT_ROOT}"
echo "SEEDS=${SEEDS[*]}"

for SEED in "${SEEDS[@]}"; do
  RUN_OUT="${OUT_ROOT}/seed_${SEED}"
  mkdir -p "${RUN_OUT}"
  echo
  echo "=============================="
  echo "Running seed ${SEED}"
  echo "OUT=${RUN_OUT}"
  echo "=============================="

  python -m trimole.pipelines.batch_run_data_new \
    --data-new ./data/data_benchmark \
    --out "${RUN_OUT}" \
    --tasks caco2_wang \
    --seed "${SEED}" | tee "${RUN_OUT}/run_seed_${SEED}.log"
done
