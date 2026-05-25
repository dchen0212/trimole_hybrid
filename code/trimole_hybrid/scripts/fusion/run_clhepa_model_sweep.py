from pathlib import Path
import itertools
import subprocess
import os

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")
OUT = ROOT / "results/model_log/clhepa_model_sweep"
OUT.mkdir(parents=True, exist_ok=True)

task = "clearance_hepatocyte_az"

fusion_types = ["mlp", "gated"]
hidden_dims = [128, 256, 384]
dropout_heads = [0.2, 0.3, 0.4, 0.5]
dropout_projs = [0.1, 0.2, 0.3]
lrs = [3e-4, 1e-4]
weight_decays = [0.0, 1e-5]
seeds = [7, 42, 123, 3407]

runs = list(itertools.product(
    fusion_types, hidden_dims, dropout_heads, dropout_projs, lrs, weight_decays, seeds
))

for i, (fusion, hd, dh, dp, lr, wd, seed) in enumerate(runs, 1):
    tag = f"f_{fusion}__hd_{hd}__dh_{dh}__dp_{dp}__lr_{lr}__wd_{wd}__seed_{seed}"
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "trimole.pipelines.batch_run_data_new",
        "--data-new", "./data/data_benchmark",
        "--out", str(out_dir),
        "--tasks", task,
        "--fusion-type", fusion,
        "--hidden-dim", str(hd),
        "--dropout-head", str(dh),
        "--dropout-proj", str(dp),
        "--lr", str(lr),
        "--weight-decay", str(wd),
        "--seed", str(seed),
    ]

    print(f"[{i}/{len(runs)}] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"[failed] {tag}", flush=True)
