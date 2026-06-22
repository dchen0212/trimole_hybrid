from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("<PROJECT_ROOT>/trimole")
DATA_DIR = "./data/data_benchmark"
OUT_ROOT = PROJECT_ROOT / "results/model_log/bbb_best_v4_check"
TASK = "bbb_martins"

CFG = {
    "fusion_type": "gated_3d_downweight",
    "loss_type": "auto",
    "hidden_dim": 128,
    "dropout_head": 0.14,
    "weight_decay": 7.5e-5,
}
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

for i, seed in enumerate(SEEDS, 1):
    tag = (
        f"fusion={CFG['fusion_type']}"
        f"__loss={CFG['loss_type']}"
        f"__hd={CFG['hidden_dim']}"
        f"__drop={CFG['dropout_head']:.2f}"
        f"__wd={fmt_wd(CFG['weight_decay'])}"
        f"__seed={seed}"
    )
    out_dir = OUT_ROOT / TASK / tag
    result_file = out_dir / "results_all.csv"
    if result_file.exists():
        print(f"[skip {i}/{len(SEEDS)}] {tag}")
        continue

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-m", "trimole.pipelines.batch_run_data_new",
        "--data-new", DATA_DIR,
        "--out", str(out_dir),
        "--tasks", TASK,
        "--fusion-type", CFG["fusion_type"],
        "--loss-type", CFG["loss_type"],
        "--hidden-dim", str(CFG["hidden_dim"]),
        "--dropout-head", str(CFG["dropout_head"]),
        "--weight-decay", str(CFG["weight_decay"]),
        "--seed", str(seed),
    ]
    print(f"[run {i}/{len(SEEDS)}]", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=ENV, check=True)
