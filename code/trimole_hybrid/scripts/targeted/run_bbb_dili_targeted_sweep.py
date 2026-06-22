#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import itertools
import subprocess
import pandas as pd
import json
import math

PROJECT = Path("<PROJECT_ROOT>/trimole")
OUT_ROOT = PROJECT / "results" / "model_log" / "bbb_dili_targeted_sweep"

TASKS = ["bbb_martins", "dili"]

GRID = {
    "fusion_type": ["gated", "gated_3d_downweight", "mlp"],
    "loss_type": ["auto", "focal"],
    "hidden_dim": [128, 160],
    "dropout_head": [0.15, 0.20, 0.25],
}

BASE_CMD = [
    "python", "-m", "trimole.pipelines.batch_run_data_new",
    "--data-new", "./data/data_benchmark",
    "--tasks",
]

def safe_name(x):
    return str(x).replace(".", "p")

def build_run_name(task, cfg):
    return (
        f"{task}"
        f"__fusion_{cfg['fusion_type']}"
        f"__loss_{cfg['loss_type']}"
        f"__h_{cfg['hidden_dim']}"
        f"__d_{safe_name(cfg['dropout_head'])}"
    )

def read_metric_from_results(results_csv: Path, task: str):
    df = pd.read_csv(results_csv)
    hit = df[df["task"] == task]
    if hit.empty:
        raise ValueError(f"{task} not found in {results_csv}")
    r = hit.iloc[0].to_dict()
    return r

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    configs = []
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]
    for vals in itertools.product(*values):
        cfg = dict(zip(keys, vals))
        configs.append(cfg)

    summary_rows = []

    for task in TASKS:
        task_root = OUT_ROOT / task
        task_root.mkdir(parents=True, exist_ok=True)

        for i, cfg in enumerate(configs, 1):
            run_name = build_run_name(task, cfg)
            out_dir = task_root / run_name
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = (
                BASE_CMD
                + [task, "--out", str(out_dir)]
                + ["--fusion-type", str(cfg["fusion_type"])]
                + ["--loss-type", str(cfg["loss_type"])]
                + ["--hidden-dim", str(cfg["hidden_dim"])]
                + ["--dropout-head", str(cfg["dropout_head"])]
            )

            print(f"\n[{task}] ({i}/{len(configs)}) running: {run_name}")
            print(" ".join(cmd))

            proc = subprocess.run(
                cmd,
                cwd=PROJECT,
                text=True,
                capture_output=True,
            )

            (out_dir / "stdout.log").write_text(proc.stdout)
            (out_dir / "stderr.log").write_text(proc.stderr)

            results_candidates = sorted(out_dir.glob("run_*/results_all.csv"))
            if proc.returncode != 0 or not results_candidates:
                summary_rows.append({
                    "task": task,
                    **cfg,
                    "status": "FAILED",
                    "returncode": proc.returncode,
                    "primary_metric_name": None,
                    "primary_metric": math.nan,
                    "best_valid_primary": math.nan,
                    "test_auc": math.nan,
                    "test_auprc": math.nan,
                    "test_acc": math.nan,
                    "results_csv": None,
                    "run_dir": str(out_dir),
                })
                continue

            results_csv = results_candidates[-1]
            row = read_metric_from_results(results_csv, task)

            summary_rows.append({
                "task": task,
                **cfg,
                "status": "OK",
                "returncode": proc.returncode,
                "primary_metric_name": row.get("primary_metric_name"),
                "primary_metric": row.get("primary_metric"),
                "best_valid_primary": row.get("best_valid_primary"),
                "test_auc": row.get("test_auc"),
                "test_auprc": row.get("test_auprc"),
                "test_acc": row.get("test_acc"),
                "results_csv": str(results_csv),
                "run_dir": str(results_csv.parent),
            })

            pd.DataFrame(summary_rows).to_csv(OUT_ROOT / "sweep_summary_live.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_ROOT / "sweep_summary.csv", index=False)

    print("\n=== DONE ===")
    print(summary.to_string(index=False))
    print("\nSaved:", OUT_ROOT / "sweep_summary.csv")

if __name__ == "__main__":
    main()
