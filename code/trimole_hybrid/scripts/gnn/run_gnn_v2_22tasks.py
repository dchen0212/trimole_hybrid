from __future__ import annotations

import itertools
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")
OUT_ROOT = PROJECT_ROOT / "results/model_log/gnn_v2_22tasks"
DATA_DIR = "./data/data_benchmark"

TASKS = [
    "ames",
    "bbb_martins",
    "bioavailability_ma",
    "caco2_wang",
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
    "cyp2c9_substrate_carbonmangels",
    "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels",
    "cyp2d6_veith",
    "cyp3a4_substrate_carbonmangels",
    "cyp3a4_veith",
    "dili",
    "half_life_obach",
    "herg",
    "hia_hou",
    "ld50_zhu",
    "lipophilicity_astrazeneca",
    "pgp_broccatelli",
    "ppbr_az",
    "solubility_aqsoldb",
    "vdss_lombardo",
]

SEEDS = [1, 42, 3407]

# 这里默认你当前可跑的单图模型入口，沿用你之前 kagnn 那条线
# 你如果有专门脚本，把 MODULE 改掉即可
MODULE = "trimole.pipelines.batch_run_data_new"

# 统一先只跑 graph-only
COMMON_ARGS = [
    "--data-new", DATA_DIR,
    "--modalities", "kpgt",
    "--hidden-dim", "128",
    "--dropout-head", "0.20",
    "--dropout-proj", "0.20",
    "--weight-decay", "1e-4",
    "--lr", "1e-4",
    "--max-epochs", "80",
    "--patience", "12",
    "--loss-type", "auto",
]

ENV = os.environ.copy()
ENV["PYTHONPATH"] = "."
ENV["HF_HUB_OFFLINE"] = "1"
ENV["TRANSFORMERS_OFFLINE"] = "1"
ENV["DGLBACKEND"] = "pytorch"
ENV["LD_LIBRARY_PATH"] = f'{ENV.get("CONDA_PREFIX", "")}/lib:{ENV.get("LD_LIBRARY_PATH", "")}'

def run_one(task: str, seed: int):
    out_dir = OUT_ROOT / task / f"seed_{seed}"
    result_file = out_dir / "results_all.csv"
    if result_file.exists():
        print(f"[skip] {task} seed={seed}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", MODULE,
        "--out", str(out_dir),
        "--tasks", task,
        "--seed", str(seed),
        *COMMON_ARGS,
    ]

    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=ENV, check=True)

def main():
    total = len(TASKS) * len(SEEDS)
    done = 0
    for task, seed in itertools.product(TASKS, SEEDS):
        done += 1
        print(f"\n=== [{done}/{total}] {task} seed={seed} ===")
        run_one(task, seed)

if __name__ == "__main__":
    main()
