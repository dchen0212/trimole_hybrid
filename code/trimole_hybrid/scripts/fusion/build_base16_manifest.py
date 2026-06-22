from pathlib import Path
import pandas as pd

ROOT = Path("<PROJECT_ROOT>/trimole")
final_p = ROOT / "results/model_log/benchmark_clean_rerun/benchmark_clean_final_master_strict.csv"
out_p = ROOT / "results/model_log/benchmark_clean_rerun/base16_manifest.csv"

df = pd.read_csv(final_p)
base16 = df[df["final_source"] == "base_official_metrics"].copy()
base16 = base16[["task", "primary_metric_name", "primary_metric", "final_source", "final_note"]]
base16 = base16.sort_values("task").reset_index(drop=True)
base16.to_csv(out_p, index=False)

print("Saved:", out_p)
print(base16.to_string(index=False))
