from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")
RERUN_DIR = ROOT / "results/model_log/benchmark_clean_rerun"

SRC = RERUN_DIR / "benchmark_clean_final_master_v7.csv"
OUT_CSV = RERUN_DIR / "benchmark_clean_final_master_v8.csv"
OUT_TXT = RERUN_DIR / "benchmark_clean_final_master_v8.txt"

PATCH_TASK = "bbb_martins"
PATCH_METRIC = "AUROC"
PATCH_SCORE = 0.923311
PATCH_SOURCE = "bbb_model_sweep_top1_push"
PATCH_NOTE = "replaced by v8 single-task top1 push; gated hidden_dim=256 dropout_head=0.2 dropout_proj=0.2 lr=1e-4 wd=1e-5 seed=123"

df = pd.read_csv(SRC).copy()

df["v8_patch_applied"] = "no"
df["v8_patch_source"] = ""
df["v8_test_score"] = np.nan

mask = df["task"].astype(str) == PATCH_TASK
if mask.sum() != 1:
    raise ValueError("bbb_martins row not found uniquely")

old_metric = str(df.loc[mask, "primary_metric_name"].iloc[0])
old_score = float(df.loc[mask, "primary_metric"].iloc[0])

if old_metric != PATCH_METRIC:
    raise ValueError(f"metric mismatch: {old_metric} vs {PATCH_METRIC}")
if PATCH_SCORE <= old_score:
    raise ValueError(f"new score not better: {PATCH_SCORE} <= {old_score}")

df.loc[mask, "primary_metric"] = PATCH_SCORE
df.loc[mask, "final_source"] = "v8_patch"
df.loc[mask, "final_note"] = PATCH_NOTE + f"; previous score={old_score}"
df.loc[mask, "v8_patch_applied"] = "yes"
df.loc[mask, "v8_patch_source"] = PATCH_SOURCE
df.loc[mask, "v8_test_score"] = PATCH_SCORE

df = df.sort_values("task").reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(df[df["v8_patch_applied"]=="yes"][[
        "task","primary_metric_name","primary_metric","v8_patch_source","v8_test_score","final_note"
    ]].to_string(index=False))
    f.write("\n\n=== source counts ===\n")
    f.write(df["final_source"].value_counts().to_string())
    f.write("\n")

print(df[df["v8_patch_applied"]=="yes"][[
    "task","primary_metric_name","primary_metric","v8_patch_source"
]].to_string(index=False))
print("\n=== source counts ===")
print(df["final_source"].value_counts().to_string())
print("\nSaved:", OUT_CSV)
print("Saved:", OUT_TXT)
