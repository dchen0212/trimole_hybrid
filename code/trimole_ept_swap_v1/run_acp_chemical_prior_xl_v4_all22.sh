#!/usr/bin/env bash
set -euo pipefail

cd <PROJECT_ROOT>/trimole_ept_swap_v1

PY=<ENV_ROOT>/trimole_bench310/bin/python
OUT=results_strict/paper_main_chemical_prior_xl_v4_all22_32core
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

"$PY" -u paper_main_chemical_prior_xl_v4.py \
  --out-root "$OUT" \
  --folds 3 \
  --seed 20260426 \
  --lambda-std 1.0 \
  --weight-step 0.1 \
  --n-jobs 32 \
  --fp-bits 2048 \
  --xgb-estimators 600 \
  --tree-estimators 900 \
  --cat-estimators 700 \
  --topk 512 1024 2048 4096 8192 \
  --backends xgb catboost extratrees rf linear \
  --chemical-blocks xl_morgan_family xl_topology_family xl_full_chemical_prior \
  --variants chem embed_chem chem_base_pred embed_chem_base_pred \
  2>&1 | tee "$OUT/run.log"
