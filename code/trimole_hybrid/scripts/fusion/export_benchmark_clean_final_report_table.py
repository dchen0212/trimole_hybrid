from pathlib import Path
import pandas as pd

ROOT = Path("<PROJECT_ROOT>/trimole")
OUT_DIR = ROOT / "results/model_log/benchmark_clean_rerun"

SRC = OUT_DIR / "benchmark_clean_final_master_strict.csv"
OUT_CSV = OUT_DIR / "benchmark_clean_final_report_table.csv"
OUT_TXT = OUT_DIR / "benchmark_clean_final_report_table.txt"

df = pd.read_csv(SRC).copy()

keep_cols = [
    "task",
    "primary_metric_name",
    "primary_metric",
    "final_source",
    "final_note",
]
keep_cols = [c for c in keep_cols if c in df.columns]

out = df[keep_cols].copy()
out = out.sort_values("task").reset_index(drop=True)
out.to_csv(OUT_CSV, index=False)

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("=== benchmark_clean final report table ===\n")
    f.write(out.to_string(index=False))
    f.write("\n\n=== source counts ===\n")
    f.write(out["final_source"].value_counts().to_string())
    f.write("\n")

print("Saved:", OUT_CSV)
print("Saved:", OUT_TXT)
print()
print(out.to_string(index=False))
print()
print("=== source counts ===")
print(out["final_source"].value_counts().to_string())
