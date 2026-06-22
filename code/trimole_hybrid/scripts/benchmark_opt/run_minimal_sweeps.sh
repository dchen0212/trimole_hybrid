#!/usr/bin/env bash
set -euo pipefail

cd <PROJECT_ROOT>/trimole

# ---- env bootstrap: 优先使用当前已激活环境；否则尝试 conda activate ----
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
OUT_ROOT="results/model_log/bench_opt_minimal"
SEED="${SEED:-42}"

mkdir -p "${OUT_ROOT}"

echo "Python      : $(which python)"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<empty>}"
echo "DATA_DIR    : ${DATA_DIR}"
echo "OUT_ROOT    : ${OUT_ROOT}"
echo

python scripts/benchmark_opt/discover_benchmark_tasks.py

mapfile -t CLS_TASKS < "${OUT_ROOT}/classification_tasks.txt"
mapfile -t REG_TASKS < "${OUT_ROOT}/regression_tasks.txt"

if [ "${#CLS_TASKS[@]}" -eq 0 ] && [ "${#REG_TASKS[@]}" -eq 0 ]; then
  echo "No tasks discovered under ${DATA_DIR}"
  exit 1
fi

run_cfg () {
  local tag="$1"
  shift
  local out_dir="${OUT_ROOT}/${tag}"
  mkdir -p "${out_dir}"

  echo "============================================================"
  echo "Running: ${tag}"
  echo "Output : ${out_dir}"
  echo "============================================================"

  python -m trimole.pipelines.batch_run_data_new \
    --data-new "${DATA_DIR}" \
    --out "${out_dir}" \
    "$@"
}

echo
echo "==== Sweep 1/7: classification | mlp | auto | dropout_head=0.3 ===="
if [ "${#CLS_TASKS[@]}" -gt 0 ]; then
  run_cfg "cls_mlp_auto_d03_seed${SEED}" \
    --tasks "${CLS_TASKS[@]}" \
    --fusion-type mlp \
    --loss-type auto \
    --hidden-dim 128 \
    --dropout-head 0.3 \
    --seed "${SEED}"
fi

echo
echo "==== Sweep 2/7: classification | mlp | auto | dropout_head=0.2 ===="
if [ "${#CLS_TASKS[@]}" -gt 0 ]; then
  run_cfg "cls_mlp_auto_d02_seed${SEED}" \
    --tasks "${CLS_TASKS[@]}" \
    --fusion-type mlp \
    --loss-type auto \
    --hidden-dim 128 \
    --dropout-head 0.2 \
    --seed "${SEED}"
fi

echo
echo "==== Sweep 3/7: classification | gated | auto | dropout_head=0.2 ===="
if [ "${#CLS_TASKS[@]}" -gt 0 ]; then
  run_cfg "cls_gated_auto_d02_seed${SEED}" \
    --tasks "${CLS_TASKS[@]}" \
    --fusion-type gated \
    --loss-type auto \
    --hidden-dim 128 \
    --dropout-head 0.2 \
    --seed "${SEED}"
fi

echo
echo "==== Sweep 4/7: classification | gated | focal | dropout_head=0.2 ===="
if [ "${#CLS_TASKS[@]}" -gt 0 ]; then
  run_cfg "cls_gated_focal_d02_seed${SEED}" \
    --tasks "${CLS_TASKS[@]}" \
    --fusion-type gated \
    --loss-type focal \
    --hidden-dim 128 \
    --dropout-head 0.2 \
    --seed "${SEED}"
fi

echo
echo "==== Sweep 5/7: regression | mlp | auto | dropout_head=0.3 ===="
if [ "${#REG_TASKS[@]}" -gt 0 ]; then
  run_cfg "reg_mlp_auto_d03_seed${SEED}" \
    --tasks "${REG_TASKS[@]}" \
    --fusion-type mlp \
    --loss-type auto \
    --hidden-dim 128 \
    --dropout-head 0.3 \
    --seed "${SEED}"
fi

echo
echo "==== Sweep 6/7: regression | mlp | auto | dropout_head=0.2 ===="
if [ "${#REG_TASKS[@]}" -gt 0 ]; then
  run_cfg "reg_mlp_auto_d02_seed${SEED}" \
    --tasks "${REG_TASKS[@]}" \
    --fusion-type mlp \
    --loss-type auto \
    --hidden-dim 128 \
    --dropout-head 0.2 \
    --seed "${SEED}"
fi

echo
echo "==== Sweep 7/7: regression | gated | auto | dropout_head=0.2 ===="
if [ "${#REG_TASKS[@]}" -gt 0 ]; then
  run_cfg "reg_gated_auto_d02_seed${SEED}" \
    --tasks "${REG_TASKS[@]}" \
    --fusion-type gated \
    --loss-type auto \
    --hidden-dim 128 \
    --dropout-head 0.2 \
    --seed "${SEED}"
fi

echo
echo "All sweeps finished."
echo "Now run:"
echo "python scripts/benchmark_opt/collect_best_from_sweeps.py"
