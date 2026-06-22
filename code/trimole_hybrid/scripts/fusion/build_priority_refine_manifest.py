from pathlib import Path
import pandas as pd

ROOT = Path("<PROJECT_ROOT>/trimole")
OUT_DIR = ROOT / "results/model_log/benchmark_clean_rerun"

SRC = OUT_DIR / "benchmark_clean_final_master_strict.csv"
OUT_CSV = OUT_DIR / "priority_refine_manifest.csv"
OUT_TXT = OUT_DIR / "priority_refine_manifest.txt"

priority_1 = [
    "ppbr_az",
    "pgp_broccatelli",
    "clearance_microsome_az",
    "solubility_aqsoldb",
]

priority_2 = [
    "cyp2c9_veith",
    "caco2_wang",
    "cyp3a4_veith",
]

df = pd.read_csv(SRC).copy()

def mark_priority(task):
    if task in priority_1:
        return "P1"
    if task in priority_2:
        return "P2"
    return ""

df["priority_group"] = df["task"].map(mark_priority)
manifest = df[df["priority_group"] != ""].copy()

manifest["recommended_action"] = manifest["priority_group"].map({
    "P1": "continue targeted refine now",
    "P2": "refine later if resources remain",
})

manifest = manifest[[
    "priority_group",
    "task",
    "primary_metric_name",
    "primary_metric",
    "final_source",
    "recommended_action",
    "final_note",
]].sort_values(["priority_group", "task"]).reset_index(drop=True)

manifest.to_csv(OUT_CSV, index=False)

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("=== priority refine manifest ===\n")
    f.write(manifest.to_string(index=False))
    f.write("\n")

print("Saved:", OUT_CSV)
print("Saved:", OUT_TXT)
print()
print(manifest.to_string(index=False))
