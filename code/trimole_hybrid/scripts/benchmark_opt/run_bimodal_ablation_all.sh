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

SRC_DATA="./data/data_benchmark"
ABL_DATA="./data/data_benchmark_ablation"
OUT_ROOT="results/model_log/ablation_bimodal"

# 优先用 v6；没有就回退到 v5
CFG_FILE="results/model_log/final_best_of_all_22_v6_selected_with_config.csv"
if [ ! -f "${CFG_FILE}" ]; then
  CFG_FILE="results/model_log/final_best_of_all_22_v5_selected_with_config.csv"
fi

if [ ! -f "${CFG_FILE}" ]; then
  echo "ERROR: no config file found."
  exit 1
fi

echo "Using config file: ${CFG_FILE}"

python scripts/benchmark_opt/prepare_bimodal_ablation_data.py \
  --src-root "${SRC_DATA}" \
  --dst-root "${ABL_DATA}"

python - <<'PY'
from pathlib import Path
import pandas as pd
import math
import subprocess
import os

PROJECT = Path("/mnt/afs/250010150/zhensheng/trimole")
OUT_ROOT = PROJECT / "results/model_log/ablation_bimodal"
CFG_CANDIDATES = [
    PROJECT / "results/model_log/final_best_of_all_22_v6_selected_with_config.csv",
    PROJECT / "results/model_log/final_best_of_all_22_v5_selected_with_config.csv",
]
cfg_file = None
for p in CFG_CANDIDATES:
    if p.exists():
        cfg_file = p
        break
if cfg_file is None:
    raise FileNotFoundError("No v6/v5 config file found")

cfg = pd.read_csv(cfg_file)

# ensemble 行不能直接用于单模型 ablation，回退到 v5 里单模型配置
if (cfg["fusion_type"].astype(str) == "ensemble").any():
    v5 = PROJECT / "results/model_log/final_best_of_all_22_v5_selected_with_config.csv"
    if v5.exists():
        v5df = pd.read_csv(v5)
        ensemble_tasks = set(cfg.loc[cfg["fusion_type"].astype(str) == "ensemble", "task"])
        for task in ensemble_tasks:
            row = v5df.loc[v5df["task"] == task]
            if len(row):
                for col in ["fusion_type", "hidden_dim", "dropout_head"]:
                    if col in row.columns:
                        cfg.loc[cfg["task"] == task, col] = row.iloc[0][col]
                # 优先取自动配置列
                if "loss_type_cfg" in row.columns and pd.notna(row.iloc[0].get("loss_type_cfg", None)):
                    cfg.loc[cfg["task"] == task, "loss_type_cfg"] = row.iloc[0]["loss_type_cfg"]
                elif "loss_type" in row.columns and pd.notna(row.iloc[0].get("loss_type", None)):
                    val = str(row.iloc[0]["loss_type"])
                    if val in {"auto", "focal"}:
                        cfg.loc[cfg["task"] == task, "loss_type_cfg"] = val

settings = ["full", "drop_smiles", "drop_graph", "drop_3d"]

def pick_loss(row):
    # 先看规范化字段，再回退
    for col in ["loss_type_cfg", "loss_type"]:
        if col in row.index and pd.notna(row[col]):
            val = str(row[col])
            if val.lower() in {"auto", "focal"}:
                return val.lower()
            if val == "FocalLoss":
                return "focal"
            if val in {"CrossEntropyLoss", "SmoothL1Loss"}:
                return "auto"
    return "auto"

def pick_fusion(row):
    val = str(row.get("fusion_type", "mlp"))
    if val not in {"mlp", "gated"}:
        return "mlp"
    return val

def pick_hidden(row):
    try:
        return int(row.get("hidden_dim", 128))
    except Exception:
        return 128

def pick_dropout(row):
    try:
        return float(row.get("dropout_head", 0.2))
    except Exception:
        return 0.2

def pick_seed(row):
    try:
        s = int(row.get("seed", 42))
        return s
    except Exception:
        return 42

for setting in settings:
    data_dir = PROJECT / "data/data_benchmark_ablation" / setting
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing ablation data dir: {data_dir}")

    for _, row in cfg.iterrows():
        task = row["task"]
        fusion = pick_fusion(row)
        loss = pick_loss(row)
        hidden = pick_hidden(row)
        dropout = pick_dropout(row)
        seed = pick_seed(row)

        out_dir = OUT_ROOT / setting / task
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "-m", "trimole.pipelines.batch_run_data_new",
            "--data-new", str(data_dir),
            "--out", str(out_dir),
            "--tasks", str(task),
            "--fusion-type", str(fusion),
            "--loss-type", str(loss),
            "--hidden-dim", str(hidden),
            "--dropout-head", str(dropout),
            "--seed", str(seed),
        ]

        print("=" * 80)
        print(f"[ABLATION] setting={setting} task={task}")
        print("CMD:", " ".join(cmd))
        print("=" * 80)

        subprocess.run(cmd, cwd=str(PROJECT), check=True, env=os.environ.copy())
PY

echo
echo "All bimodal ablation runs finished."
echo "Now run:"
echo "  python scripts/benchmark_opt/collect_bimodal_ablation_results.py"
