#!/usr/bin/env bash
set -euo pipefail

cd <PROJECT_ROOT>/trimole

TARGET_ENV="<ENV_ROOT>/trimole_bench310"
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
OUT_ROOT="results/model_log/boost_round2"

SEEDS=(001 007 042 123 3407)

run_one () {
  local task="$1"
  local cfg_tag="$2"
  local fusion="$3"
  local loss="$4"
  local hidden="$5"
  local dropout="$6"
  local seed="$7"

  local out_dir="${OUT_ROOT}/${task}/${cfg_tag}/seed_${seed}"
  mkdir -p "${out_dir}"

  echo "============================================================"
  echo "Task   : ${task}"
  echo "Cfg    : ${cfg_tag}"
  echo "Seed   : ${seed}"
  echo "Out    : ${out_dir}"
  echo "============================================================"

  python -m trimole.pipelines.batch_run_data_new \
    --data-new "${DATA_DIR}" \
    --out "${out_dir}" \
    --tasks "${task}" \
    --fusion-type "${fusion}" \
    --loss-type "${loss}" \
    --hidden-dim "${hidden}" \
    --dropout-head "${dropout}" \
    --seed "${seed}"
}

# -----------------------------
# bioavailability_ma
# -----------------------------
for seed in "${SEEDS[@]}"; do
  run_one bioavailability_ma bioava_mlp_auto_h128_d020 mlp   auto  128 0.20 "${seed}"
  run_one bioavailability_ma bioava_mlp_auto_h128_d030 mlp   auto  128 0.30 "${seed}"
  run_one bioavailability_ma bioava_gated_focal_h128_d015 gated focal 128 0.15 "${seed}"
done

# -----------------------------
# cyp3a4_substrate_carbonmangels
# -----------------------------
for seed in "${SEEDS[@]}"; do
  run_one cyp3a4_substrate_carbonmangels cyp3a4sub_gated_focal_h128_d020 gated focal 128 0.20 "${seed}"
  run_one cyp3a4_substrate_carbonmangels cyp3a4sub_gated_focal_h128_d015 gated focal 128 0.15 "${seed}"
  run_one cyp3a4_substrate_carbonmangels cyp3a4sub_mlp_auto_h128_d030    mlp   auto  128 0.30 "${seed}"
done

# -----------------------------
# half_life_obach
# -----------------------------
for seed in "${SEEDS[@]}"; do
  run_one half_life_obach halflife_gated_auto_h128_d020 gated auto 128 0.20 "${seed}"
  run_one half_life_obach halflife_gated_auto_h096_d020 gated auto  96 0.20 "${seed}"
  run_one half_life_obach halflife_mlp_auto_h128_d020   mlp   auto 128 0.20 "${seed}"
done

echo
echo "Done."
echo "Now run:"
echo "  python scripts/benchmark_opt/ensemble_boost_round2.py"
