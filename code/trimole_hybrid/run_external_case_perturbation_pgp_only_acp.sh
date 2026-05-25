#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afs/250010150/zhensheng/trimole_hybrid

PY="${PY:-/mnt/afs/250010150/envs/trimole/bin/python}"
OUT="results_strict/external_case_perturbation_v1"

echo "[start-pgp-only] $(date)"
echo "[env] python=${PY}"
echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

"${PY}" external_case_perturbation_v1.py --out "${OUT}" --generate-only
"${PY}" external_case_perturbation_v1.py --out "${OUT}" --run-pgp --python "${PY}"

echo "[done-pgp-only] $(date)"
find "${OUT}" -maxdepth 2 -type f | sort
