#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afs/250010150/zhensheng/trimole_ept_swap_v1

PY=/mnt/afs/250010150/envs/trimole_bench310/bin/python
OUT=results_strict/paper_main_chemical_prior_xl_v4_remaining4_32core
N_JOBS=${N_JOBS:-32}
mkdir -p "$OUT"

# CPU-only tabular search. Keep this aligned with the full all-22 XL run,
# but restrict the task list to the four endpoints not completed previously.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$N_JOBS"
export MKL_NUM_THREADS="$N_JOBS"
export OPENBLAS_NUM_THREADS="$N_JOBS"
export NUMEXPR_NUM_THREADS="$N_JOBS"

"$PY" -u paper_main_chemical_prior_xl_v4.py \
  --out-root "$OUT" \
  --tasks pgp_broccatelli ppbr_az solubility_aqsoldb vdss_lombardo \
  --folds 3 \
  --seed 20260426 \
  --lambda-std 1.0 \
  --weight-step 0.1 \
  --n-jobs "$N_JOBS" \
  --fp-bits 2048 \
  --xgb-estimators 600 \
  --tree-estimators 900 \
  --cat-estimators 700 \
  --topk 512 1024 2048 4096 8192 \
  --backends xgb catboost extratrees rf linear \
  --chemical-blocks xl_morgan_family xl_topology_family xl_full_chemical_prior \
  --variants chem embed_chem chem_base_pred embed_chem_base_pred \
  2>&1 | tee "$OUT/run.log"
