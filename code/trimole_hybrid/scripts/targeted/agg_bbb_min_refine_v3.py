from __future__ import annotations

from pathlib import Path
import pandas as pd

root = Path("/mnt/afs/250010150/zhensheng/trimole/results/model_log/bbb_min_refine_v3/bbb_martins")
rows = []
bad = []

def pick_metric(d: dict, cands: list[str]):
    for c in cands:
        if c in d and pd.notna(d[c]):
            return d[c]
    return None

for f in root.rglob("results_all.csv"):
    try:
        tag = f.parent.parent.name
        meta = {}
        for x in tag.split("__"):
            k, v = x.split("=", 1)
            meta[k] = v

        df = pd.read_csv(f)
        if "task" in df.columns:
            sub = df[df["task"] == "bbb_martins"]
            row = (sub.iloc[0] if not sub.empty else df.iloc[0]).to_dict()
        else:
            row = df.iloc[0].to_dict()

        valid = pick_metric(row, [
            "valid_roc_auc", "valid_auc", "val_roc_auc", "val_auc",
            "valid_metric", "valid_score"
        ])
        test = pick_metric(row, [
            "test_roc_auc", "test_auc", "test_metric", "test_score"
        ])

        rows.append({
            "fusion_type": meta["fusion"],
            "loss_type": meta["loss"],
            "hidden_dim": int(meta["hd"]),
            "dropout_head": float(meta["drop"]),
            "weight_decay": float(meta["wd"]),
            "seed": int(meta["seed"]),
            "valid": valid,
            "test": test,
            "file": str(f),
        })
    except Exception as e:
        bad.append(f"{f}\t{type(e).__name__}: {e}")

if not rows:
    print("No valid results found.")
    print("\nBad files:")
    for x in bad:
        print(x)
    raise SystemExit(1)

df = pd.DataFrame(rows)

agg = (
    df.groupby(
        ["fusion_type", "loss_type", "hidden_dim", "dropout_head", "weight_decay"],
        as_index=False
    )
    .agg(
        n_runs=("seed", "count"),
        valid_mean=("valid", "mean"),
        valid_std=("valid", "std"),
        test_mean=("test", "mean"),
        test_std=("test", "std"),
        test_best=("test", "max"),
    )
    .sort_values(["test_mean", "test_best"], ascending=[False, False])
)

out_csv = root.parent / "bbb_min_refine_v3_agg.csv"
bad_txt = root.parent / "bbb_min_refine_v3_bad_files.txt"
raw_csv = root.parent / "bbb_min_refine_v3_raw.csv"

df.to_csv(raw_csv, index=False)
agg.to_csv(out_csv, index=False)
bad_txt.write_text("\n".join(bad) + ("\n" if bad else ""), encoding="utf-8")

print(agg.to_string(index=False))
print(f"\nSaved: {out_csv}")
print(f"Saved raw: {raw_csv}")
print(f"Bad files skipped: {len(bad)}")
print(f"Saved bad file list: {bad_txt}")
