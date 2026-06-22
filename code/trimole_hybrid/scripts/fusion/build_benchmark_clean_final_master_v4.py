from pathlib import Path
import pandas as pd

ROOT = Path("<PROJECT_ROOT>/trimole")
RERUN_DIR = ROOT / "results/model_log/benchmark_clean_rerun"

SRC = RERUN_DIR / "benchmark_clean_v3_patch.csv"
OUT_CSV = RERUN_DIR / "benchmark_clean_final_master_v4.csv"
OUT_TXT = RERUN_DIR / "benchmark_clean_final_master_v4.txt"

df = pd.read_csv(SRC).copy()

# final_source 已经在 v3 patch 里更新过，直接导出
out = df.sort_values("task").reset_index(drop=True)
out.to_csv(OUT_CSV, index=False)

summary = out["final_source"].value_counts()

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("=== benchmark_clean_final_master_v4 ===\n")
    f.write(out[["task","primary_metric_name","primary_metric","final_source","final_note"]].to_string(index=False))
    f.write("\n\n=== source counts ===\n")
    f.write(summary.to_string())
    f.write("\n")

print("Saved:", OUT_CSV)
print("Saved:", OUT_TXT)
print()
print(out[["task","primary_metric_name","primary_metric","final_source"]].to_string(index=False))
print("\n=== source counts ===")
print(summary.to_string())
