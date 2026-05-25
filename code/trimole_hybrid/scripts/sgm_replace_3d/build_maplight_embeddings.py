#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import sys

# 用官方 MapLight 仓库里的特征构造
MAPLIGHT_ROOT = Path("/mnt/afs/250010150/zhensheng/trimole/external/MapLight-TDC")
sys.path.insert(0, str(MAPLIGHT_ROOT))

from maplight import get_fingerprints  # noqa

TASKS_DEFAULT = [
    "bioavailability_ma",
    "bbb_martins",
    "cyp2c9_veith",
    "pgp_broccatelli",
    "herg",
]

def detect_smiles_col(df: pd.DataFrame) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in ["drug", "smiles"]:
        if cand in cols_lower:
            return cols_lower[cand]
    raise KeyError(f"Cannot find smiles column. columns={list(df.columns)}")

def build_one_csv(csv_path: Path, out_npy: Path):
    df = pd.read_csv(csv_path)
    smiles_col = detect_smiles_col(df)
    X = get_fingerprints(df[smiles_col].astype(str))
    X = np.asarray(X, dtype=np.float32)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, X)
    print(f"Saved: {out_npy} shape={X.shape}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=str)
    ap.add_argument("--tasks", nargs="*", default=TASKS_DEFAULT)
    ap.add_argument("--embed-dir-name", default="embeddings_maplight", type=str)
    args = ap.parse_args()

    data_root = Path(args.data_root)

    for task in args.tasks:
        task_dir = data_root / task
        out_dir = task_dir / args.embed_dir_name

        build_one_csv(task_dir / "train.csv", out_dir / "train_maplight.npy")
        build_one_csv(task_dir / "valid.csv", out_dir / "valid_maplight.npy")
        build_one_csv(task_dir / "test.csv",  out_dir / "test_maplight.npy")

if __name__ == "__main__":
    main()
