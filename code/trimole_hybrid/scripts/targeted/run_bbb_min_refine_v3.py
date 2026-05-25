from __future__ import annotations

import itertools
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")
DATA_DIR = "./data/data_benchmark"
OUT_ROOT = PROJECT_ROOT / "results/model_log/bbb_min_refine_v3"
TASK = "bbb_martins"

GRID = [
    {"fusion_type": "gated_3d_downweight", "loss_type": "auto", "hidden_dim": 128, "dropout_head": 0.11, "weight_decay": 7.5e-5},
    {"fusion_type": "gated_3d_downweight", "loss_type": "auto", "hidden_dim": 128, "dropout_head": 0.12, "weight_decay": 7.5e-5},
    {"fusion_type": "gated_3d_downweight", "loss_type": "auto", "hidden_dim": 128, "dropout_head": 0.13, "weight_decay": 7.5e-5},
    {"fusion_type": "gated_3d_downweight", "loss_type": "auto", "hidden_dim": 128, "dropout_head": 0.14, "weight_decay": 7.5e-5},
    {"fusion_type": "gated_3d_downweight", "loss_type": "auto", "hidden_dim": 128, "dropout_head": 0.15, "weight_decay": 7.5e-5},
    {"fusion_type": "gated_3d_downweight", "loss_type": "auto", "hidden_dim": 128, "dropout_head": 0.10, "weight_decay": 1.5e-4},
    {"fusion_type": "gated_3d_downweight", "loss_type": "auto", "hidden_dim": 128, "dropout_head": 0.12, "weight_decay": 1.5e-4},
    {"fusion_type": "gated_3d_downweight", "loss_type": "auto", "hidden_dim": 128, "dropout_head": 0.15, "weight_decay": 1.5e-4},
    {"fusion_type": "gated_3d_downweight", "loss_type": "auto", "hidden_dim": 136, "dropout_head": 0.09, "weight_decay": 0.0},
    {"fusion_type": "gated_3d_downweight", "loss_type": "auto", "hidden_dim": 136, "dropout_head": 0.09, "weight_decay": 5e-5},
    {"fusion_type": "gated_3d_downweight", "loss_type": "auto", "hidden_dim": 136, "dropout_head": 0.11, "weight_decay": 5e-5},
    {"fusion_type": "gated_3d_downweight", "loss_type": "auto", "hidden_dim": 136, "dropout_head": 0.11, "weight_decay": 1e-4},
]

SEEDS = [1, 7, 42, 123, 3407]

ENV = os.environ.copy()
ENV["PYTHONPATH"] = "."
ENV["HF_HUB_OFFLINE"] = "1"
ENV["TRANSFORMERS_OFFLINE"] = "1"
ENV["DGLBACKEND"] = "pytorch"
ENV["LD_LIBRARY_PATH"] = f'{ENV.get("CONDA_PREFIX", "")}/lib:{ENV.get("LD_LIBRARY_PATH", "")}'

def fmt_wd(x: float) -> str:
    if x == 0:
        return "0"
    return f"{x:.8f}".rstrip("0").rstrip(".")

def run_one(cfg: dict, seed: int) -> None:
    tag = (
        f"fusion={cfg['fusion_type']}"
        f"__loss={cfg['loss_type']}"
        f"__hd={cfg['hidden_dim']}"
        f"__drop={cfg['dropout_head']:.2f}"
        f"__wd={fmt_wd(cfg['weight_decay'])}"
        f"__seed={seed}"
    )
    out_dir = OUT_ROOT / TASK / tag
    result_file = out_dir / "results_all.csv"
    if result_file.exists():
        print(f"[skip] {tag}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "trimole.pipelines.batch_run_data_new",
        "--data-new", DATA_DIR,
        "--out", str(out_dir),
        "--tasks", TASK,
        "--fusion-type", cfg["fusion_type"],
        "--loss-type", cfg["loss_type"],
        "--hidden-dim", str(cfg["hidden_dim"]),
        "--dropout-head", str(cfg["dropout_head"]),
        "--weight-decay", str(cfg["weight_decay"]),
        "--seed", str(seed),
    ]

    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=ENV, check=True)

def main() -> None:
    total = len(GRID) * len(SEEDS)
    done = 0
    for cfg, seed in itertools.product(GRID, SEEDS):
        done += 1
        print(f"\n=== [{done}/{total}] ===")
        run_one(cfg, seed)

if __name__ == "__main__":
    main()
