#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import itertools
import subprocess
import pandas as pd
import math

PROJECT = Path("<PROJECT_ROOT>/trimole")
OUT_ROOT = PROJECT / "results" / "model_log" / "bbb_final_refine"

TASK = "bbb_martins"
SEEDS = [1, 7, 42, 123, 3407]

GRID = {
    "hidden_dim": [96, 128, 160],
    "dropout_head": [0.10, 0.15, 0.20, 0.25],
    "weight_decay": [0.0, 1e-5, 1e-4],
}

def safe_name(x):
    return str(x).replace(".", "p").replace("-", "m")

rows = []
OUT_ROOT.mkdir(parents=True, exist_ok=True)

keys = list(GRID.keys())
vals = [GRID[k] for k in keys]

for combo in itertools.product(*vals):
    cfg = dict(zip(keys, combo))
    for seed in SEEDS:
        name = (
            f"{TASK}"
            f"__h_{cfg['hidden_dim']}"
            f"__d_{safe_name(cfg['dropout_head'])}"
            f"__wd_{safe_name(cfg['weight_decay'])}"
            f"__seed_{seed}"
        )
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 已完成就跳过
        cand = sorted(out_dir.glob("run_*/results_all.csv"))
        if cand:
            print("SKIP:", name)
            df = pd.read_csv(cand[-1])
            r = df[df["task"] == TASK].iloc[0]
            rows.append({
                "task": TASK,
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
            continue

        cmd = [
            "python", "-m", "trimole.pipelines.batch_run_data_new",
            "--data-new", "./data/data_benchmark",
            "--tasks", TASK,
            "--out", str(out_dir),
            "--fusion-type", "gated_3d_downweight",
            "--loss-type", "auto",
            "--hidden-dim", str(cfg["hidden_dim"]),
            "--dropout-head", str(cfg["dropout_head"]),
            "--weight-decay", str(cfg["weight_decay"]),
            "--seed", str(seed),
        ]

        print("RUN:", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=PROJECT, text=True)

        cand = sorted(out_dir.glob("run_*/results_all.csv"))
        if proc.returncode != 0 or not cand:
            rows.append({
                "task": TASK,
                **cfg,
                "seed": seed,
                "status": "FAILED",
                "best_valid_primary": math.nan,
                "primary_metric": math.nan,
                "results_csv": None,
            })
            pd.DataFrame(rows).to_csv(OUT_ROOT / "bbb_refine_live_resume.csv", index=False)
            continue

        df = pd.read_csv(cand[-1])
        r = df[df["task"] == TASK].iloc[0]

        rows.append({
            "task": TASK,
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
        pd.DataFrame(rows).to_csv(OUT_ROOT / "bbb_refine_live_resume.csv", index=False)

df = pd.DataFrame(rows)
df.to_csv(OUT_ROOT / "bbb_refine_summary_resume.csv", index=False)

ok = df[df["status"] == "OK"].copy()
agg = (
    ok.groupby(["hidden_dim", "dropout_head", "weight_decay"], as_index=False)
      .agg(
          n_runs=("seed", "count"),
          valid_mean=("best_valid_primary", "mean"),
          valid_std=("best_valid_primary", "std"),
          test_mean=("primary_metric", "mean"),
          test_std=("primary_metric", "std"),
          test_best=("primary_metric", "max"),
      )
      .sort_values("test_mean", ascending=False)
)
agg.to_csv(OUT_ROOT / "bbb_refine_agg_resume.csv", index=False)

print("\n=== AGG RESUME ===")
print(agg.to_string(index=False))
print("\nSaved:")
print(" -", OUT_ROOT / "bbb_refine_summary_resume.csv")
print(" -", OUT_ROOT / "bbb_refine_agg_resume.csv")
