#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import pandas as pd

CFG = Path("results/model_log/final_best_v4_task_config.csv")
OUT = Path("scripts/benchmark_opt/rerun_best_v4_tasks.sh")

if not CFG.exists():
    raise FileNotFoundError(f"Missing file: {CFG}. Run make_final_best_v4.py first.")

df = pd.read_csv(CFG)

lines = []
lines.append("#!/usr/bin/env bash")
lines.append("set -euo pipefail")
lines.append("")
lines.append("cd /mnt/afs/250010150/zhensheng/trimole")
lines.append("")
lines.append('TARGET_ENV="/mnt/afs/250010150/envs/trimole_bench310"')
lines.append('if [ "${CONDA_PREFIX:-}" != "${TARGET_ENV}" ]; then')
lines.append('  if command -v conda >/dev/null 2>&1; then')
lines.append('    eval "$(conda shell.bash hook)"')
lines.append('    conda activate "${TARGET_ENV}"')
lines.append('  else')
lines.append('    echo "ERROR: conda not found, and current env is not ${TARGET_ENV}"')
lines.append('    exit 1')
lines.append('  fi')
lines.append('fi')
lines.append("")
lines.append('export HF_HUB_OFFLINE=1')
lines.append('export TRANSFORMERS_OFFLINE=1')
lines.append('export DGLBACKEND=pytorch')
lines.append('export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"')
lines.append('export PYTHONPATH=.')
lines.append("")
lines.append('DATA_DIR="./data/data_benchmark"')
lines.append('OUT_ROOT="results/model_log/final_best_v4_runs"')
lines.append('mkdir -p "${OUT_ROOT}"')
lines.append("")
lines.append('echo "Python      : $(which python)"')
lines.append('echo "CONDA_PREFIX: ${CONDA_PREFIX:-<empty>}"')
lines.append("")

for _, row in df.iterrows():
    task = row["task"]
    fusion = row["fusion_type"]
    loss = row["loss_type"]
    hidden = int(row["hidden_dim"])
    dhead = row["dropout_head"]
    seed = int(row["seed"]) if pd.notna(row["seed"]) else 42

    lines.append('echo "============================================================"')
    lines.append(f'echo "Running task: {task}"')
    lines.append(f'echo "Config      : fusion={fusion}, loss={loss}, hidden_dim={hidden}, dropout_head={dhead}, seed={seed}"')
    lines.append('echo "============================================================"')
    lines.append(
        'python -m trimole.pipelines.batch_run_data_new '
        f'--data-new "${{DATA_DIR}}" '
        f'--out "${{OUT_ROOT}}/{task}" '
        f'--tasks "{task}" '
        f'--fusion-type {fusion} '
        f'--loss-type {loss} '
        f'--hidden-dim {hidden} '
        f'--dropout-head {dhead} '
        f'--seed {seed}'
    )
    lines.append("")

OUT.write_text("\n".join(lines) + "\n")
print(f"Saved: {OUT}")
