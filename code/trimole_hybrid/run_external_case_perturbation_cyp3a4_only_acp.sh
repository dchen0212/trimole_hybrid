#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afs/250010150/zhensheng/trimole_hybrid

PY="${PY:-/mnt/afs/250010150/envs/trimole/bin/python}"
OUT="results_strict/external_case_perturbation_v1"

echo "[start-cyp3a4-only] $(date)"
echo "[env] python=${PY}"

"${PY}" external_case_perturbation_v1.py --out "${OUT}" --generate-only
"${PY}" external_case_perturbation_v1.py --out "${OUT}" --run-cyp3a4

echo "[done-cyp3a4-only] $(date)"
find "${OUT}" -maxdepth 2 -type f | sort
