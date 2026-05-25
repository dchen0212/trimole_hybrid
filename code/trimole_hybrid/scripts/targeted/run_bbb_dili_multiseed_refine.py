#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import subprocess
import pandas as pd
import math

PROJECT = Path("/mnt/afs/250010150/zhensheng/trimole")
OUT_ROOT = PROJECT / "results" / "model_log" / "bbb_dili_multiseed_refine"

# 只保留最值得试的几组
CONFIGS = [
    # BBB
    {"task": "bbb_martins", "fusion_type": "gated_3d_downweight", "loss_type": "auto",  "hidden_dim": 128, "dropout_head": 0.20},
    {"task": "bbb_martins", "fusion_type": "gated",               "loss_type": "focal", "hidden_dim": 128, "dropout_head": 0.20},
    {"task": "bbb_martins", "fusion_type": "gated",               "loss_type": "focal", "hidden_dim": 160, "dropout_head": 0.20},

    # DILI
    {"task": "dili",        "fusion_type": "gated",               "loss_type": "auto",  "hidden_dim": 128, "dropout_head": 0.20},
    {"task": "dili",        "fusion_type": "gated",               "loss_type": "focal", "hidden_dim": 160, "dropout_head": 0.25},
    {"task": "dili",        "fusion_type": "gated_3d_downweight", "loss_type": "focal", "hidden_dim": 128, "dropout_head": 0.20},
]

SEEDS = [1, 7, 42, 123, 3407]

def safe_name(x):
    return str(x).replace(".", "p")

def run_name(cfg, seed):
    return (
        f"{cfg['task']}"
        f"__fusion_{cfg['fusion_type']}"
        f"__loss_{cfg['loss_type']}"
        f"__h_{cfg['hidden_dim']}"
        f"__d_{safe_name(cfg['dropout_head'])}"
        f"__seed_{seed}"
    )

rows = []
OUT_ROOT.mkdir(parents=True, exist_ok=True)

for cfg in CONFIGS:
    for seed in SEEDS:
        name = run_name(cfg, seed)
        out_dir = OUT_ROOT / cfg["task"] / name
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "-m", "trimole.pipelines.batch_run_data_new",
            "--data-new", "./data/data_benchmark",
            "--tasks", cfg["task"],
            "--out", str(out_dir),
            "--fusion-type", str(cfg["fusion_type"]),
            "--loss-type", str(cfg["loss_type"]),
            "--hidden-dim", str(cfg["hidden_dim"]),
            "--dropout-head", str(cfg["dropout_head"]),
            "--seed", str(seed),
        ]

        print("RUN:", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=PROJECT, text=True, capture_output=True)
        (out_dir / "stdout.log").write_text(proc.stdout)
        (out_dir / "stderr.log").write_text(proc.stderr)

        cand = sorted(out_dir.glob("run_*/results_all.csv"))
        if proc.returncode != 0 or not cand:
            rows.append({
                **cfg, "seed": seed, "status": "FAILED",
                "best_valid_primary": math.nan, "primary_metric": math.nan,
                "results_csv": None,
            })
            pd.DataFrame(rows).to_csv(OUT_ROOT / "multiseed_live.csv", index=False)
            continue

        df = pd.read_csv(cand[-1])
        r = df[df["task"] == cfg["task"]].iloc[0]

        rows.append({
            **cfg,
            "seed": seed,
            "status": "OK",
            "primary_metric_name": r["primary_metric_name"],
            "best_valid_primary": r["best_valid_primary"],
            "primary_metric": r["primary_metric"],
            "test_auc": r.get("test_auc", math.nan),
            "test_auprc": r.get("test_auprc", math.nan),
            "test_acc": r.get("test_acc", math.nan),
            "results_csv": str(cand[-1]),
        })
        pd.DataFrame(rows).to_csv(OUT_ROOT / "multiseed_live.csv", index=False)

df = pd.DataFrame(rows)
df.to_csv(OUT_ROOT / "multiseed_summary.csv", index=False)

ok = df[df["status"] == "OK"].copy()
agg = (
    ok.groupby(["task", "fusion_type", "loss_type", "hidden_dim", "dropout_head"], as_index=False)
      .agg(
          n_runs=("seed", "count"),
          valid_mean=("best_valid_primary", "mean"),
          valid_std=("best_valid_primary", "std"),
          test_mean=("primary_metric", "mean"),
          test_std=("primary_metric", "std"),
          test_best=("primary_metric", "max"),
      )
      .sort_values(["task", "test_mean"], ascending=[True, False])
)
agg.to_csv(OUT_ROOT / "multiseed_agg.csv", index=False)

print("\n=== AGG ===")
print(agg.to_string(index=False))
print("\nSaved:")
print(" -", OUT_ROOT / "multiseed_summary.csv")
print(" -", OUT_ROOT / "multiseed_agg.csv")
