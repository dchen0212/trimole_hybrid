from pathlib import Path
import pandas as pd

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")
RUN_ROOT = ROOT / "results/model_log/caco2_benchmark_multiseed"
FINAL_MASTER = ROOT / "results/model_log/benchmark_clean_rerun/benchmark_clean_final_master_strict.csv"
OUT_CSV = RUN_ROOT / "caco2_multiseed_summary.csv"
OUT_TXT = RUN_ROOT / "caco2_multiseed_summary.txt"

rows = []

def pick_cols(df):
    task_col = next((c for c in ["task", "dataset", "name"] if c in df.columns), None)
    metric_name_col = next((c for c in ["primary_metric_name", "official_metric", "metric_name", "metric"] if c in df.columns), None)
    metric_val_col = next((c for c in ["primary_metric", "official_score", "metric_value", "test_score", "score"] if c in df.columns), None)
    return task_col, metric_name_col, metric_val_col

for seed_dir in sorted(RUN_ROOT.glob("seed_*")):
    run_files = sorted(seed_dir.glob("run_*/results_all.csv"))
    if not run_files:
        continue
    p = run_files[-1]
    df = pd.read_csv(p)
    task_col, metric_name_col, metric_val_col = pick_cols(df)
    if not all([task_col, metric_name_col, metric_val_col]):
        rows.append({
            "seed_dir": seed_dir.name,
            "run_file": str(p),
            "task": "caco2_wang",
            "metric_name": None,
            "metric_value": None,
            "note": "could_not_identify_columns",
        })
        continue

    sub = df[df[task_col].astype(str) == "caco2_wang"].copy()
    if len(sub) == 0:
        rows.append({
            "seed_dir": seed_dir.name,
            "run_file": str(p),
            "task": "caco2_wang",
            "metric_name": None,
            "metric_value": None,
            "note": "task_not_found",
        })
        continue

    r = sub.iloc[0]
    rows.append({
        "seed_dir": seed_dir.name,
        "run_file": str(p),
        "task": "caco2_wang",
        "metric_name": r[metric_name_col],
        "metric_value": float(r[metric_val_col]),
        "note": "ok",
    })

out = pd.DataFrame(rows)

best_current = None
if FINAL_MASTER.exists():
    fm = pd.read_csv(FINAL_MASTER)
    sub = fm[fm["task"] == "caco2_wang"]
    if len(sub):
        best_current = float(sub.iloc[0]["primary_metric"])

if len(out) and out["metric_value"].notna().any():
    numeric = out["metric_value"].dropna()
    stats = {
        "n": int(numeric.shape[0]),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)) if numeric.shape[0] > 1 else 0.0,
        "best_min": float(numeric.min()),
        "worst_max": float(numeric.max()),
        "current_best_final_master": best_current,
    }
else:
    stats = {
        "n": 0,
        "mean": None,
        "std": None,
        "best_min": None,
        "worst_max": None,
        "current_best_final_master": best_current,
    }

out.to_csv(OUT_CSV, index=False)

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("=== caco2 benchmark multiseed summary ===\n")
    f.write(out.to_string(index=False))
    f.write("\n\n=== stats ===\n")
    for k, v in stats.items():
        f.write(f"{k}: {v}\n")

print("Saved:", OUT_CSV)
print("Saved:", OUT_TXT)
print()
print(out.to_string(index=False))
print("\n=== stats ===")
for k, v in stats.items():
    print(f"{k}: {v}")
