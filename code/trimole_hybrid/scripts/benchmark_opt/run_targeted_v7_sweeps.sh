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
OUT_ROOT="results/model_log/bench_opt_targeted_v7"
SEED="${SEED:-42}"

CLS_TASKS=(
  cyp3a4_substrate_carbonmangels
  cyp2c9_substrate_carbonmangels
  cyp2d6_substrate_carbonmangels
)

REG_TASKS=(
  clearance_hepatocyte_az
  vdss_lombardo
)

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

# -------- classification: try CE/Focal + gated/mlp + d0.15/0.20/0.25 --------
run_cfg "cls_gated_focal_h128_d015_seed${SEED}" \
  --tasks "${CLS_TASKS[@]}" \
  --fusion-type gated \
  --loss-type focal \
  --hidden-dim 128 \
  --dropout-head 0.15 \
  --seed "${SEED}"

run_cfg "cls_gated_focal_h128_d020_seed${SEED}" \
  --tasks "${CLS_TASKS[@]}" \
  --fusion-type gated \
  --loss-type focal \
  --hidden-dim 128 \
  --dropout-head 0.20 \
  --seed "${SEED}"

run_cfg "cls_gated_focal_h128_d025_seed${SEED}" \
  --tasks "${CLS_TASKS[@]}" \
  --fusion-type gated \
  --loss-type focal \
  --hidden-dim 128 \
  --dropout-head 0.25 \
  --seed "${SEED}"

run_cfg "cls_gated_auto_h128_d020_seed${SEED}" \
  --tasks "${CLS_TASKS[@]}" \
  --fusion-type gated \
  --loss-type auto \
  --hidden-dim 128 \
  --dropout-head 0.20 \
  --seed "${SEED}"

run_cfg "cls_mlp_auto_h128_d025_seed${SEED}" \
  --tasks "${CLS_TASKS[@]}" \
  --fusion-type mlp \
  --loss-type auto \
  --hidden-dim 128 \
  --dropout-head 0.25 \
  --seed "${SEED}"

run_cfg "cls_mlp_focal_h128_d020_seed${SEED}" \
  --tasks "${CLS_TASKS[@]}" \
  --fusion-type mlp \
  --loss-type focal \
  --hidden-dim 128 \
  --dropout-head 0.20 \
  --seed "${SEED}"

# -------- regression: keep d0.2, search hidden_dim and fusion --------
run_cfg "reg_gated_auto_h096_d020_seed${SEED}" \
  --tasks "${REG_TASKS[@]}" \
  --fusion-type gated \
  --loss-type auto \
  --hidden-dim 96 \
  --dropout-head 0.20 \
  --seed "${SEED}"

run_cfg "reg_gated_auto_h128_d020_seed${SEED}" \
  --tasks "${REG_TASKS[@]}" \
  --fusion-type gated \
  --loss-type auto \
  --hidden-dim 128 \
  --dropout-head 0.20 \
  --seed "${SEED}"

run_cfg "reg_gated_auto_h160_d020_seed${SEED}" \
  --tasks "${REG_TASKS[@]}" \
  --fusion-type gated \
  --loss-type auto \
  --hidden-dim 160 \
  --dropout-head 0.20 \
  --seed "${SEED}"

run_cfg "reg_gated_auto_h192_d020_seed${SEED}" \
  --tasks "${REG_TASKS[@]}" \
  --fusion-type gated \
  --loss-type auto \
  --hidden-dim 192 \
  --dropout-head 0.20 \
  --seed "${SEED}"

run_cfg "reg_mlp_auto_h096_d020_seed${SEED}" \
  --tasks "${REG_TASKS[@]}" \
  --fusion-type mlp \
  --loss-type auto \
  --hidden-dim 96 \
  --dropout-head 0.20 \
  --seed "${SEED}"

run_cfg "reg_mlp_auto_h160_d020_seed${SEED}" \
  --tasks "${REG_TASKS[@]}" \
  --fusion-type mlp \
  --loss-type auto \
  --hidden-dim 160 \
  --dropout-head 0.20 \
  --seed "${SEED}"

echo
echo "Done."
echo "Now run:"
echo "  python scripts/benchmark_opt/collect_targeted_v7_best.py"
echo "  python scripts/benchmark_opt/make_final_v7_selected.py"
