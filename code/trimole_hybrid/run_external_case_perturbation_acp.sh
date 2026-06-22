#!/usr/bin/env bash
set -euo pipefail

cd <PROJECT_ROOT>/trimole_hybrid

PY="${PY:-<ENV_ROOT>/trimole/bin/python}"
OUT="results_strict/external_case_perturbation_v1"

echo "[start] $(date)"
echo "[env] python=${PY}"
echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "[step1] generate perturbation candidates"
"${PY}" external_case_perturbation_v1.py --out "${OUT}" --generate-only

echo "[step2] run exact CYP3A4-S + P-gp perturbation predictions"
"${PY}" external_case_perturbation_v1.py --out "${OUT}" --run-cyp3a4 --run-pgp --python "${PY}"

echo "[done] $(date)"
echo "[outputs]"
ls -lh "${OUT}" || true
