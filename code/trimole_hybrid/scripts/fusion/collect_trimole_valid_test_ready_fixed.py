#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import pandas as pd

PROJECT = Path("/mnt/afs/250010150/zhensheng/trimole")
ROOT = PROJECT / "results" / "model_log"
OUT_DIR = ROOT / "fusion_inputs_trimole_calib_ready_fixed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_TO_SRCS = {
    "bbb_martins": {
        "valid": ROOT / "validation_dump_rerun_top4" / "run_20260413_2212" / "bbb_martins" / "valid_predictions.csv",
        "test":  ROOT / "validation_dump_rerun_top4" / "run_20260413_2212" / "bbb_martins" / "test_predictions.csv",
    },
    "hia_hou": {
        "valid": ROOT / "validation_dump_rerun_top4" / "run_20260413_2212" / "hia_hou" / "valid_predictions.csv",
        "test":  ROOT / "validation_dump_rerun_top4" / "run_20260413_2212" / "hia_hou" / "test_predictions.csv",
    },
    "herg": {
        "valid": None,
        "test": None,
    },
    "dili": {
        "valid": ROOT / "validation_dump_rerun_top4" / "run_20260413_2212" / "dili" / "valid_predictions.csv",
        "test":  ROOT / "validation_dump_rerun_top4" / "run_20260413_2212" / "dili" / "test_predictions.csv",
    },
    "cyp3a4_veith": {
        "valid": ROOT / "validation_dump_rerun_keytasks" / "run_20260413_2116" / "cyp3a4_veith" / "valid_predictions.csv",
        "test":  ROOT / "validation_dump_rerun_keytasks" / "run_20260413_2116" / "cyp3a4_veith" / "test_predictions.csv",
    },
    "cyp2d6_veith": {
        "valid": ROOT / "validation_dump_rerun_keytasks" / "run_20260413_2116" / "cyp2d6_veith" / "valid_predictions.csv",
        "test":  ROOT / "validation_dump_rerun_keytasks" / "run_20260413_2116" / "cyp2d6_veith" / "test_predictions.csv",
    },
}

rows = []
missing = []

for task, srcs in TASK_TO_SRCS.items():
    v = srcs["valid"]
    t = srcs["test"]

    if v is None or t is None or (not Path(v).exists()) or (not Path(t).exists()):
        missing.append({
            "task": task,
            "valid_src": "" if v is None else str(v),
            "test_src": "" if t is None else str(t),
        })
        continue

    vdst = OUT_DIR / f"{task}_valid_predictions.csv"
    tdst = OUT_DIR / f"{task}_test_predictions.csv"
    shutil.copy2(v, vdst)
    shutil.copy2(t, tdst)

    rows.append({
        "task": task,
        "valid_src": str(v),
        "test_src": str(t),
        "valid_dst": str(vdst),
        "test_dst": str(tdst),
    })

pd.DataFrame(rows).to_csv(OUT_DIR / "collect_summary.csv", index=False)
pd.DataFrame(missing).to_csv(OUT_DIR / "collect_missing.csv", index=False)

print("=== collected ===")
print(pd.DataFrame(rows).to_string(index=False) if rows else "none")
print("\n=== missing ===")
print(pd.DataFrame(missing).to_string(index=False) if missing else "none")
print("\nSaved:")
print(" -", OUT_DIR / "collect_summary.csv")
print(" -", OUT_DIR / "collect_missing.csv")
