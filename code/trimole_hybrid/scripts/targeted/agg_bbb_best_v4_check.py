from __future__ import annotations

from pathlib import Path
import pandas as pd

root = Path("<PROJECT_ROOT>/trimole/results/model_log/bbb_best_v4_check/bbb_martins")
rows = []

def pick_metric(d: dict, cands: list[str]):
    for c in cands:
        if c in d and pd.notna(d[c]):
            return d[c]
    return None

for f in root.rglob("results_all.csv"):
    tag = f.parent.parent.name
    meta = dict(x.split("=", 1) for x in tag.split("__"))

    df = pd.read_csv(f)
    row = df.iloc[0].to_dict() if "task" not in df.columns else (df[df["task"] == "bbb_martins"].iloc[0].to_dict())

    rows.append({
        "fusion_type": meta["fusion"],
        "loss_type": meta["loss"],
        "hidden_dim": int(meta["hd"]),
        "dropout_head": float(meta["drop"]),
        "weight_decay": float(meta["wd"]),
        "seed": int(meta["seed"]),
        "test_auc": pick_metric(row, ["test_auc", "test_roc_auc", "test_metric"]),
        "test_auprc": pick_metric(row, ["test_auprc"]),
        "test_acc": pick_metric(row, ["test_acc"]),
        "file": str(f),
    })

df = pd.DataFrame(rows).sort_values("seed")
summary = {
    "n_runs": len(df),
    "test_auc_mean": pd.to_numeric(df["test_auc"], errors="coerce").mean(),
    "test_auc_std": pd.to_numeric(df["test_auc"], errors="coerce").std(),
    "test_auc_best": pd.to_numeric(df["test_auc"], errors="coerce").max(),
    "test_auprc_mean": pd.to_numeric(df["test_auprc"], errors="coerce").mean(),
    "test_auprc_std": pd.to_numeric(df["test_auprc"], errors="coerce").std(),
    "test_auprc_best": pd.to_numeric(df["test_auprc"], errors="coerce").max(),
    "test_acc_mean": pd.to_numeric(df["test_acc"], errors="coerce").mean(),
    "test_acc_std": pd.to_numeric(df["test_acc"], errors="coerce").std(),
    "test_acc_best": pd.to_numeric(df["test_acc"], errors="coerce").max(),
}
print(df.to_string(index=False))
print("\nSUMMARY")
for k, v in summary.items():
    print(f"{k}: {v}")
