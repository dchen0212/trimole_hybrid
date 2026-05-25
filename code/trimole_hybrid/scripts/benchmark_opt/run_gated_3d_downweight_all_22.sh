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
OUT_ROOT="results/model_log/gated_3d_downweight_all_22"
SEED="${SEED:-42}"

CLS_TASKS=(
  ames
  bbb_martins
  bioavailability_ma
  cyp2c9_substrate_carbonmangels
  cyp2c9_veith
  cyp2d6_substrate_carbonmangels
  cyp2d6_veith
  cyp3a4_substrate_carbonmangels
  cyp3a4_veith
  dili
  herg
  hia_hou
  pgp_broccatelli
)

REG_TASKS=(
  caco2_wang
  clearance_hepatocyte_az
  clearance_microsome_az
  half_life_obach
  ld50_zhu
  lipophilicity_astrazeneca
  ppbr_az
  solubility_aqsoldb
  vdss_lombardo
)

run_one() {
  local tag="$1"
  shift
  local out_dir="${OUT_ROOT}/${tag}"
  mkdir -p "${out_dir}"

  echo "============================================================"
  echo "Running: ${tag}"
  echo "Out    : ${out_dir}"
  echo "============================================================"

  python -m trimole.pipelines.batch_run_data_new \
    --data-new "${DATA_DIR}" \
    --out "${out_dir}" \
    "$@"
}

# classification
run_one "cls_auto_seed${SEED}" \
  --tasks "${CLS_TASKS[@]}" \
  --fusion-type gated_3d_downweight \
  --loss-type auto \
  --hidden-dim 128 \
  --dropout-head 0.2 \
  --seed "${SEED}"

run_one "cls_focal_seed${SEED}" \
  --tasks "${CLS_TASKS[@]}" \
  --fusion-type gated_3d_downweight \
  --loss-type focal \
  --hidden-dim 128 \
  --dropout-head 0.2 \
  --seed "${SEED}"

# regression
run_one "reg_auto_seed${SEED}" \
  --tasks "${REG_TASKS[@]}" \
  --fusion-type gated_3d_downweight \
  --loss-type auto \
  --hidden-dim 128 \
  --dropout-head 0.2 \
  --seed "${SEED}"

echo
echo "Done."
echo "Now run:"
echo "  python scripts/benchmark_opt/collect_gated_3d_downweight_all_22.py"
echo "  python scripts/benchmark_opt/make_final_3d_downweight_all_22_patch.py"
