from pathlib import Path
import pandas as pd
import math

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole_hybrid")
RESULTS_ROOT = ROOT / "results_strict"
OUT_CSV = RESULTS_ROOT / "scoreboard_22_tdc.csv"

# Official TDC ADMET Group metrics
TDC_METRICS = {
    "ames": ("AUROC", "max"),
    "bbb_martins": ("AUROC", "max"),
    "bioavailability_ma": ("AUROC", "max"),
    "caco2_wang": ("MAE", "min"),
    "clearance_hepatocyte_az": ("Spearman", "max"),
    "clearance_microsome_az": ("Spearman", "max"),
    "cyp2c9_substrate_carbonmangels": ("AUPRC", "max"),
    "cyp2c9_veith": ("AUPRC", "max"),
    "cyp2d6_substrate_carbonmangels": ("AUPRC", "max"),
    "cyp2d6_veith": ("AUPRC", "max"),
    "cyp3a4_substrate_carbonmangels": ("AUROC", "max"),
    "cyp3a4_veith": ("AUPRC", "max"),
    "dili": ("AUROC", "max"),
    "half_life_obach": ("Spearman", "max"),
    "herg": ("AUROC", "max"),
    "hia_hou": ("AUROC", "max"),
    "ld50_zhu": ("MAE", "min"),
    "lipophilicity_astrazeneca": ("MAE", "min"),
    "pgp_broccatelli": ("AUROC", "max"),
    "ppbr_az": ("MAE", "min"),
    "solubility_aqsoldb": ("MAE", "min"),
    "vdss_lombardo": ("Spearman", "max"),
}

METRIC_TO_COL = {
    "AUROC": "test_auc",
    "AUPRC": "test_auprc",
    "MAE": "test_mae",
    "Spearman": "test_spearman",
}

def is_valid_number(x):
    try:
        return pd.notna(x) and not math.isnan(float(x))
    except Exception:
        return False

def better(direction, new_val, old_val):
    if old_val is None:
        return True
    if direction == "max":
        return new_val > old_val
    return new_val < old_val

def extract_row(results_csv: Path):
    try:
        df = pd.read_csv(results_csv)
    except Exception:
        return []

    rows = []
    for _, r in df.iterrows():
        task = str(r.get("task", "")).strip()
        if task not in TDC_METRICS:
            continue

        metric, direction = TDC_METRICS[task]
        score_col = METRIC_TO_COL[metric]
        score = r.get(score_col, None)

        if not is_valid_number(score):
            continue

        rows.append({
            "task": task,
            "tdc_metric": metric,
            "direction": direction,
            "official_score": float(score),
            "source_score_column": score_col,
            "task_type": r.get("task_type", None),
            "best_valid_primary": r.get("best_valid_primary", None),
            "pipeline_primary_metric_name": r.get("primary_metric_name", None),
            "pipeline_primary_metric": r.get("primary_metric", None),
            "loss_type": r.get("loss_type", None),
            "best_epoch": r.get("best_epoch", None),
            "device": r.get("device", None),
            "seed": r.get("seed", None),
            "source_run_file": str(results_csv),
        })
    return rows

# load existing scoreboard if present
if OUT_CSV.exists():
    current = pd.read_csv(OUT_CSV)
else:
    current = pd.DataFrame(columns=[
        "task","tdc_metric","direction","official_score","source_score_column",
        "task_type","best_valid_primary","pipeline_primary_metric_name",
        "pipeline_primary_metric","loss_type","best_epoch","device","seed",
        "source_run_file"
    ])

best_map = {}
for _, row in current.iterrows():
    best_map[str(row["task"])] = row.to_dict()

# scan all strict result files
all_result_files = sorted(RESULTS_ROOT.glob("**/results_all.csv"))
for p in all_result_files:
    for row in extract_row(p):
        task = row["task"]
        old = best_map.get(task)
        old_score = None if old is None or not is_valid_number(old.get("official_score")) else float(old["official_score"])
        if better(row["direction"], row["official_score"], old_score):
            best_map[task] = row

# ensure all 22 tasks exist in scoreboard, even if empty
final_rows = []
for task, (metric, direction) in sorted(TDC_METRICS.items()):
    if task in best_map:
        final_rows.append(best_map[task])
    else:
        final_rows.append({
            "task": task,
            "tdc_metric": metric,
            "direction": direction,
            "official_score": None,
            "source_score_column": METRIC_TO_COL[metric],
            "task_type": None,
            "best_valid_primary": None,
            "pipeline_primary_metric_name": None,
            "pipeline_primary_metric": None,
            "loss_type": None,
            "best_epoch": None,
            "device": None,
            "seed": None,
            "source_run_file": None,
        })

out = pd.DataFrame(final_rows)
out = out.sort_values("task").reset_index(drop=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_CSV, index=False)

print("saved:", OUT_CSV)
print(out.to_string(index=False))
