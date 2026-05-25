#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import shutil
import pandas as pd

PROJECT = Path("/mnt/afs/250010150/zhensheng/trimole")
ROOT = PROJECT / "results" / "model_log"

TASKS = [
    "bbb_martins",
    "hia_hou",
    "herg",
    "dili",
    "cyp3a4_veith",
    "cyp2d6_veith",
]

OUT_DIR = ROOT / "fusion_inputs_trimole_calib_ready"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 你前面实际跑过、最可能有 valid/test 的目录
KNOWN_RUN_ROOTS = [
    ROOT / "validation_dump_rerun_keytasks",
    ROOT / "validation_dump_rerun_regfix",
    ROOT / "final_best_v4_runs",
    ROOT / "final_patch_latefusion_22_fixed",
]

# 兼容两种命名：
# 1) task/valid_predictions.csv
# 2) task_valid_predictions.csv
def find_pred_file(task: str, split: str) -> Path | None:
    # 先扫 run_* 目录
    for base in KNOWN_RUN_ROOTS:
        if not base.exists():
            continue

        run_dirs = sorted(base.glob("run_*"))
        search_roots = run_dirs[::-1] if run_dirs else [base]

        for r in search_roots:
            candidates = [
                r / task / f"{split}_predictions.csv",
                r / f"{task}_{split}_predictions.csv",
                r / task / f"{split}.csv",
                r / f"{task}_{split}.csv",
            ]
            for c in candidates:
                if c.exists():
                    return c

    # 最后全局兜底搜一下，但只搜 model_log
    patterns = [
        f"**/{task}/{split}_predictions.csv",
        f"**/{task}_{split}_predictions.csv",
        f"**/{task}/{split}.csv",
        f"**/{task}_{split}.csv",
    ]
    hits = []
    for pat in patterns:
        hits.extend(ROOT.glob(pat))

    hits = sorted(set(hits), key=lambda p: str(p))
    if hits:
        # 取最后一个，通常是较新的路径
        return hits[-1]
    return None


rows = []
missing = []

for task in TASKS:
    valid_src = find_pred_file(task, "valid")
    test_src = find_pred_file(task, "test")

    if valid_src is None or test_src is None:
        missing.append({
            "task": task,
            "valid_src": str(valid_src) if valid_src else "",
            "test_src": str(test_src) if test_src else "",
        })
        continue

    valid_dst = OUT_DIR / f"{task}_valid_predictions.csv"
    test_dst = OUT_DIR / f"{task}_test_predictions.csv"

    shutil.copy2(valid_src, valid_dst)
    shutil.copy2(test_src, test_dst)

    rows.append({
        "task": task,
        "valid_src": str(valid_src),
        "test_src": str(test_src),
        "valid_dst": str(valid_dst),
        "test_dst": str(test_dst),
    })

summary_p = OUT_DIR / "collect_summary.csv"
pd.DataFrame(rows).to_csv(summary_p, index=False)

missing_p = OUT_DIR / "collect_missing.csv"
pd.DataFrame(missing).to_csv(missing_p, index=False)

print("=== collected ===")
if rows:
    print(pd.DataFrame(rows).to_string(index=False))
else:
    print("none")

print("\n=== missing ===")
if missing:
    print(pd.DataFrame(missing).to_string(index=False))
else:
    print("none")

print("\nSaved:")
print(" -", summary_p)
print(" -", missing_p)
