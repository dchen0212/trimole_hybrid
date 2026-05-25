#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import itertools
import subprocess
import pandas as pd
import math

PROJECT = Path("/mnt/afs/250010150/zhensheng/trimole")
OUT_ROOT = PROJECT / "results" / "model_log" / "bbb_dili_refine_v2"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS = [1, 7, 42, 123, 3407]

TASK_CONFIGS = {
    "bbb_martins": {
        "fusion_type": ["gated_3d_downweight"],
        "loss_type": ["auto"],
        "hidden_dim": [120, 128, 136],
        "dropout_head": [0.10, 0.12, 0.15],
        "weight_decay": [0.0, 1e-5, 5e-5, 1e-4],
    },
    "dili": {
        "fusion_type": ["gated"],
        "loss_type": ["auto", "focal"],
        "hidden_dim": [128, 144, 160],
        "dropout_head": [0.18, 0.20, 0.22, 0.25],
        "weight_decay": [0.0, 1e-5, 5e-5],
    },
}

def safe_name(x):
    return str(x).replace(".", "p").replace("-", "m")

all_rows = []

for task, grid in TASK_CONFIGS.items():
    task_out = OUT_ROOT / task
    task_out.mkdir(parents=True, exist_ok=True)

    keys = list(grid.keys())
    vals = [grid[k] for k in keys]

    rows = []
    for combo in itertools.product(*vals):
        cfg = dict(zip(keys, combo))
        for seed in SEEDS:
            name = (
                f"{task}"
                f"__fusion_{cfg['fusion_type']}"
                f"__loss_{cfg['loss_type']}"
                f"__h_{cfg['hidden_dim']}"
                f"__d_{safe_name(cfg['dropout_head'])}"
                f"__wd_{safe_name(cfg['weight_decay'])}"
                f"__seed_{seed}"
            )
            out_dir = task_out / name
            out_dir.mkdir(parents=True, exist_ok=True)

            cand = sorted(out_dir.glob("run_*/results_all.csv"))
            if cand:
                print("SKIP:", name)
                df = pd.read_csv(cand[-1])
                r = df[df["task"] == task].iloc[0]
                row = {
                    "task": task,
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
                }
                rows.append(row)
                all_rows.append(row)
                continue

            cmd = [
                "python", "-m", "trimole.pipelines.batch_run_data_new",
                "--data-new", "./data/data_benchmark",
                "--tasks", task,
                "--out", str(out_dir),
                "--fusion-type", str(cfg["fusion_type"]),
                "--loss-type", str(cfg["loss_type"]),
                "--hidden-dim", str(cfg["hidden_dim"]),
                "--dropout-head", str(cfg["dropout_head"]),
                "--weight-decay", str(cfg["weight_decay"]),
                "--seed", str(seed),
                "--patience", "15",
                "--max-epochs", "80",
            ]

            print("RUN:", " ".join(cmd))
            proc = subprocess.run(cmd, cwd=PROJECT, text=True)

            cand = sorted(out_dir.glob("run_*/results_all.csv"))
            if proc.returncode != 0 or not cand:
                row = {
                    "task": task,
                    **cfg,
                    "seed": seed,
                    "status": "FAILED",
                    "primary_metric_name": "",
                    "best_valid_primary": math.nan,
                    "primary_metric": math.nan,
                    "test_auc": math.nan,
                    "test_auprc": math.nan,
                    "test_acc": math.nan,
                    "results_csv": "",
                }
                rows.append(row)
                all_rows.append(row)
                pd.DataFrame(rows).to_csv(task_out / f"{task}_live.csv", index=False)
                pd.DataFrame(all_rows).to_csv(OUT_ROOT / "live_all.csv", index=False)
                continue

            df = pd.read_csv(cand[-1])
            r = df[df["task"] == task].iloc[0]
            row = {
                "task": task,
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
            }
            rows.append(row)
            all_rows.append(row)

            pd.DataFrame(rows).to_csv(task_out / f"{task}_live.csv", index=False)
            pd.DataFrame(all_rows).to_csv(OUT_ROOT / "live_all.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(task_out / f"{task}_summary.csv", index=False)

    ok = df[df["status"] == "OK"].copy()
    agg = (
        ok.groupby(keys, as_index=False)
          .agg(
              n_runs=("seed", "count"),
              valid_mean=("best_valid_primary", "mean"),
              valid_std=("best_valid_primary", "std"),
              test_mean=("primary_metric", "mean"),
              test_std=("primary_metric", "std"),
              test_best=("primary_metric", "max"),
          )
    )

    # 关键：主排序看 valid_mean，不再看 test_mean
    agg = agg.sort_values(
        ["valid_mean", "test_mean", "test_best"],
        ascending=[False, False, False]
    )
    agg.to_csv(task_out / f"{task}_agg.csv", index=False)

    print(f"\n=== {task} AGG ===")
    print(agg.to_string(index=False))
    print(f"\nSaved: {task_out / f'{task}_agg.csv'}")

pd.DataFrame(all_rows).to_csv(OUT_ROOT / "summary_all.csv", index=False)
print("\nSaved:", OUT_ROOT / "summary_all.csv")
