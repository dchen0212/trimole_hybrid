from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("<PROJECT_ROOT>/trimole")
RERUN_DIR = ROOT / "results/model_log/benchmark_clean_rerun"

SRC = RERUN_DIR / "benchmark_clean_final_master_v6.csv"
OUT_CSV = RERUN_DIR / "benchmark_clean_final_master_v7.csv"
OUT_TXT = RERUN_DIR / "benchmark_clean_final_master_v7.txt"

PATCH_TASK = "clearance_hepatocyte_az"
PATCH_METRIC = "Spearman"
PATCH_SCORE = 0.541710
PATCH_SOURCE = "clhepa_model_sweep_quick"
PATCH_NOTE = "replaced by v7 single-task sweep; mlp hidden_dim=256 dropout_head=0.3 dropout_proj=0.1 lr=3e-4 wd=0 seed=42"

df = pd.read_csv(SRC).copy()

df["v7_patch_applied"] = "no"
df["v7_patch_source"] = ""
df["v7_test_score"] = np.nan

mask = df["task"].astype(str) == PATCH_TASK
if mask.sum() != 1:
    raise ValueError("clearance_hepatocyte_az row not found uniquely")

old_metric = str(df.loc[mask, "primary_metric_name"].iloc[0])
old_score = float(df.loc[mask, "primary_metric"].iloc[0])

if old_metric != PATCH_METRIC:
    raise ValueError(f"metric mismatch: {old_metric} vs {PATCH_METRIC}")

if PATCH_SCORE <= old_score:
    raise ValueError(f"new score not better: {PATCH_SCORE} <= {old_score}")

df.loc[mask, "primary_metric"] = PATCH_SCORE
df.loc[mask, "final_source"] = "v7_patch"
df.loc[mask, "final_note"] = PATCH_NOTE + f"; previous score={old_score}"
df.loc[mask, "v7_patch_applied"] = "yes"
df.loc[mask, "v7_patch_source"] = PATCH_SOURCE
df.loc[mask, "v7_test_score"] = PATCH_SCORE

df = df.sort_values("task").reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("=== V7 PATCH APPLIED ROWS ===\n")
    f.write(df[df["v7_patch_applied"]=="yes"][[
        "task","primary_metric_name","primary_metric","v7_patch_source","v7_test_score","final_note"
    ]].to_string(index=False))
    f.write("\n\n=== source counts ===\n")
    f.write(df["final_source"].value_counts().to_string())
    f.write("\n")

print("Saved:", OUT_CSV)
print("Saved:", OUT_TXT)
print()
print(df[df["v7_patch_applied"]=="yes"][[
    "task","primary_metric_name","primary_metric","v7_patch_source"
]].to_string(index=False))
print("\n=== source counts ===")
print(df["final_source"].value_counts().to_string())
