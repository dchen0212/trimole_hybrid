from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


HIGHER_IS_BETTER = {"AUROC", "AUC", "AUPRC", "AUCPR", "ACC", "ACCURACY", "SPEARMAN"}
LOWER_IS_BETTER = {"MAE", "RMSE", "MSE"}


def _direction(metric_name: str) -> str:
    m = str(metric_name or "").upper()
    if m in LOWER_IS_BETTER:
        return "lower_better"
    return "higher_better"


def load_results(run_dir: Path) -> pd.DataFrame:
    run_dir = run_dir.resolve()
    p = run_dir / "results_all.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing results_all.csv under: {run_dir}")
    df = pd.read_csv(p)
    if "task" not in df.columns:
        raise ValueError(f"results_all.csv missing column 'task': {p}")
    # Ensure numeric primary metric
    df["primary_metric"] = pd.to_numeric(df.get("primary_metric"), errors="coerce")
    df["best_valid_primary"] = pd.to_numeric(df.get("best_valid_primary"), errors="coerce")
    df["primary_metric_name"] = df.get("primary_metric_name").astype(str)
    return df


def compare_runs(best: pd.DataFrame, other: pd.DataFrame, other_name: str) -> pd.DataFrame:
    b = best.copy()
    o = other.copy()

    b = b[["task", "task_type", "primary_metric_name", "primary_metric"]].rename(
        columns={"primary_metric": "best_primary"}
    )
    o = o[["task", "task_type", "primary_metric_name", "primary_metric"]].rename(
        columns={"primary_metric": f"{other_name}_primary"}
    )

    df = b.merge(o, on=["task"], how="inner", suffixes=("_best", "_other"))
    # Use best's metric_name as the reference.
    df["metric"] = df["primary_metric_name_best"].astype(str)
    df["direction"] = df["metric"].map(_direction)

    best_v = pd.to_numeric(df["best_primary"], errors="coerce")
    other_v = pd.to_numeric(df[f"{other_name}_primary"], errors="coerce")

    delta = []
    for m, d, bv, ov in zip(df["metric"].astype(str), df["direction"].astype(str), best_v, other_v):
        if not np.isfinite(bv) or not np.isfinite(ov):
            delta.append(np.nan)
            continue
        if d == "lower_better":
            # positive means other is better (lower)
            delta.append(float(bv - ov))
        else:
            # positive means other is better (higher)
            delta.append(float(ov - bv))
    df[f"{other_name}_improvement_vs_best"] = delta
    df["beat_best"] = df[f"{other_name}_improvement_vs_best"].apply(lambda x: bool(np.isfinite(x) and x > 0))

    # Nice sorting: biggest gains first.
    df = df.sort_values(f"{other_name}_improvement_vs_best", ascending=False, na_position="last")
    return df[
        [
            "task",
            "metric",
            "direction",
            "best_primary",
            f"{other_name}_primary",
            f"{other_name}_improvement_vs_best",
            "beat_best",
        ]
    ]


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Compare two run_* directories (results_all.csv) task-by-task")
    ap.add_argument("--best", type=str, required=True, help="Best run dir (contains results_all.csv)")
    ap.add_argument("--other", type=str, required=True, help="Other run dir to compare against best")
    ap.add_argument("--other-name", type=str, default="other", help="Label for other run columns")
    ap.add_argument("--out", type=str, default="", help="Output CSV path (default: <other>/compare_vs_<best>.csv)")
    args = ap.parse_args(argv)

    best_dir = Path(args.best).resolve()
    other_dir = Path(args.other).resolve()
    other_name = str(args.other_name).strip() or "other"

    best_df = load_results(best_dir)
    other_df = load_results(other_dir)
    comp = compare_runs(best_df, other_df, other_name=other_name)

    if args.out:
        out_csv = Path(args.out).resolve()
    else:
        out_csv = other_dir / f"compare_vs_{best_dir.name}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out_csv, index=False)

    beat = int(comp["beat_best"].sum())
    total = int(len(comp))
    print(f"Wrote: {out_csv}")
    print(f"{other_name} beats best on {beat}/{total} tasks (metric-direction aware).")


if __name__ == "__main__":
    main()

