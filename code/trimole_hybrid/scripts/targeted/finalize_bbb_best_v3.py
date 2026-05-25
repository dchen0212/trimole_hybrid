from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")
SRC = ROOT / "results/model_log/bbb_min_refine_v3/bbb_martins"
OUT_DIR = ROOT / "results/model_log/bbb_best_v3_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = {
    "fusion": "gated_3d_downweight",
    "loss": "auto",
    "hd": "128",
    "drop": "0.14",
    "wd": "0.000075",
}

rows = []
picked = []

for f in SRC.rglob("results_all.csv"):
    tag = f.parent.parent.name
    meta = {}
    ok = True
    for x in tag.split("__"):
        if "=" not in x:
            ok = False
            break
        k, v = x.split("=", 1)
        meta[k] = v
    if not ok:
        continue

    if not (
        meta.get("fusion") == TARGET["fusion"]
        and meta.get("loss") == TARGET["loss"]
        and meta.get("hd") == TARGET["hd"]
        and meta.get("drop") == TARGET["drop"]
        and meta.get("wd") == TARGET["wd"]
    ):
        continue

    df = pd.read_csv(f)
    row = df.iloc[0].to_dict()
    row["seed"] = int(meta["seed"])
    row["source_file"] = str(f)
    rows.append(row)
    picked.append(str(f))

if not rows:
    raise SystemExit("No matching BBB best-v3 files found.")

raw = pd.DataFrame(rows).sort_values("seed")

metric_cols = [c for c in raw.columns if c.startswith("test_") or c.startswith("valid_")]
summary = {
    "fusion_type": TARGET["fusion"],
    "loss_type": TARGET["loss"],
    "hidden_dim": int(TARGET["hd"]),
    "dropout_head": float(TARGET["drop"]),
    "weight_decay": float(TARGET["wd"]),
    "n_runs": len(raw),
}

for c in metric_cols:
    s = pd.to_numeric(raw[c], errors="coerce")
    if s.notna().any():
        summary[f"{c}_mean"] = s.mean()
        summary[f"{c}_std"] = s.std()
        summary[f"{c}_best"] = s.max()

summary_df = pd.DataFrame([summary])

raw_csv = OUT_DIR / "bbb_best_v3_raw.csv"
summary_csv = OUT_DIR / "bbb_best_v3_summary.csv"
picked_txt = OUT_DIR / "bbb_best_v3_files.txt"

raw.to_csv(raw_csv, index=False)
summary_df.to_csv(summary_csv, index=False)
picked_txt.write_text("\n".join(picked) + "\n", encoding="utf-8")

print("=== BBB BEST V3 SUMMARY ===")
print(summary_df.to_string(index=False))
print(f"\nSaved raw: {raw_csv}")
print(f"Saved summary: {summary_csv}")
print(f"Saved file list: {picked_txt}")
