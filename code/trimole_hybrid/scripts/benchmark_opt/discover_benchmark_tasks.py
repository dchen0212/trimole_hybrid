#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

DATA_ROOT = Path("data/data_benchmark")
OUT_DIR = Path("results/model_log/bench_opt_minimal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_TARGETS = [
    "Y", "y", "label", "labels", "target", "targets", "value"
]
NON_TARGET_HINTS = {
    "drug", "drugs", "smiles", "mol", "molecule", "id", "name",
    "split", "group", "scaffold"
}


def infer_target_col(df: pd.DataFrame) -> str:
    for c in CANDIDATE_TARGETS:
        if c in df.columns:
            return c

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    filtered = [c for c in numeric_cols if c.lower() not in NON_TARGET_HINTS]
    if len(filtered) == 1:
        return filtered[0]
    if len(filtered) >= 1:
        return filtered[-1]
    raise ValueError(f"Cannot infer target column from columns: {list(df.columns)}")


def infer_task_type(series: pd.Series) -> str:
    s = series.dropna()
    uniq = sorted(pd.unique(s))
    if len(uniq) <= 2 and set(map(float, uniq)).issubset({0.0, 1.0}):
        return "classification"
    return "regression"


rows = []
for task_dir in sorted(DATA_ROOT.iterdir()):
    if not task_dir.is_dir():
        continue
    train_csv = task_dir / "train.csv"
    valid_csv = task_dir / "valid.csv"
    test_csv = task_dir / "test.csv"
    emb_dir = task_dir / "embeddings"
    if not (train_csv.exists() and valid_csv.exists() and test_csv.exists()):
        continue

    df = pd.read_csv(train_csv)
    target_col = infer_target_col(df)
    task_type = infer_task_type(df[target_col])

    row = {
        "task": task_dir.name,
        "task_dir": str(task_dir),
        "target_col": target_col,
        "task_type": task_type,
        "n_train": len(pd.read_csv(train_csv)),
        "n_valid": len(pd.read_csv(valid_csv)),
        "n_test": len(pd.read_csv(test_csv)),
        "has_embeddings": emb_dir.exists(),
    }

    if task_type == "classification":
        s = df[target_col].dropna().astype(float)
        row["positive_rate_train"] = float(s.mean()) if len(s) else None
        row["n_unique_train"] = int(s.nunique())
    else:
        s = df[target_col].dropna().astype(float)
        row["target_mean_train"] = float(s.mean()) if len(s) else None
        row["target_std_train"] = float(s.std()) if len(s) else None
        row["n_unique_train"] = int(s.nunique())

    rows.append(row)

meta = pd.DataFrame(rows).sort_values(["task_type", "task"]).reset_index(drop=True)
meta.to_csv(OUT_DIR / "task_metadata.csv", index=False)

cls_tasks = meta.loc[meta["task_type"] == "classification", "task"].tolist()
reg_tasks = meta.loc[meta["task_type"] == "regression", "task"].tolist()

(OUT_DIR / "classification_tasks.txt").write_text("\n".join(cls_tasks) + ("\n" if cls_tasks else ""))
(OUT_DIR / "regression_tasks.txt").write_text("\n".join(reg_tasks) + ("\n" if reg_tasks else ""))

summary = {
    "data_root": str(DATA_ROOT),
    "n_tasks": int(len(meta)),
    "n_classification": int(len(cls_tasks)),
    "n_regression": int(len(reg_tasks)),
    "classification_tasks": cls_tasks,
    "regression_tasks": reg_tasks,
}
(OUT_DIR / "task_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

print("Saved:")
print(" -", OUT_DIR / "task_metadata.csv")
print(" -", OUT_DIR / "classification_tasks.txt")
print(" -", OUT_DIR / "regression_tasks.txt")
print(" -", OUT_DIR / "task_summary.json")
print()
print("Classification tasks:", len(cls_tasks))
print("Regression tasks    :", len(reg_tasks))
