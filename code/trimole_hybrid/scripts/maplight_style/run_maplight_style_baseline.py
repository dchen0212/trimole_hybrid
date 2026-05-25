#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import rdReducedGraphs

RDLogger.DisableLog('rdApp.warning')

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)
from scipy.stats import spearmanr


CLASSIFICATION_TASKS = {
    "ames",
    "bbb_martins",
    "bioavailability_ma",
    "cyp2c9_substrate_carbonmangels",
    "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels",
    "cyp2d6_veith",
    "cyp3a4_substrate_carbonmangels",
    "cyp3a4_veith",
    "dili",
    "herg",
    "hia_hou",
    "pgp_broccatelli",
}

OFFICIAL_METRIC = {
    "caco2_wang": "MAE",
    "hia_hou": "AUROC",
    "pgp_broccatelli": "AUROC",
    "bioavailability_ma": "AUROC",
    "lipophilicity_astrazeneca": "MAE",
    "solubility_aqsoldb": "MAE",
    "bbb_martins": "AUROC",
    "ppbr_az": "MAE",
    "vdss_lombardo": "Spearman",
    "cyp2c9_veith": "AUPRC",
    "cyp2d6_veith": "AUPRC",
    "cyp3a4_veith": "AUPRC",
    "cyp2c9_substrate_carbonmangels": "AUPRC",
    "cyp2d6_substrate_carbonmangels": "AUPRC",
    "cyp3a4_substrate_carbonmangels": "AUROC",
    "half_life_obach": "Spearman",
    "clearance_hepatocyte_az": "Spearman",
    "clearance_microsome_az": "Spearman",
    "ld50_zhu": "MAE",
    "herg": "AUROC",
    "ames": "AUROC",
    "dili": "AUROC",
}

# 取一批稳定、常用的 RDKit 2D 描述符，避免太重
DESC_FUNCS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("RingCount", Descriptors.RingCount),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("NHOHCount", Descriptors.NHOHCount),
    ("NOCount", Descriptors.NOCount),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    ("BalabanJ", Descriptors.BalabanJ),
    ("BertzCT", Descriptors.BertzCT),
    ("Chi0", Descriptors.Chi0),
    ("Chi1", Descriptors.Chi1),
    ("HallKierAlpha", Descriptors.HallKierAlpha),
    ("LabuteASA", Descriptors.LabuteASA),
]


def find_smiles_and_label_cols(df: pd.DataFrame) -> Tuple[str, str]:
    smiles_candidates = ["Drug", "smiles", "SMILES", "Smiles"]
    label_candidates = ["Y", "y", "label", "Label", "target", "Target"]

    smiles_col = None
    label_col = None

    for c in smiles_candidates:
        if c in df.columns:
            smiles_col = c
            break
    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break

    if smiles_col is None:
        raise ValueError(f"Cannot find SMILES column in columns={list(df.columns)}")
    if label_col is None:
        raise ValueError(f"Cannot find label column in columns={list(df.columns)}")
    return smiles_col, label_col


def fp_to_np(fp) -> np.ndarray:
    arr = np.zeros((fp.GetNumBits(),), dtype=np.float32)
    onbits = list(fp.GetOnBits())
    arr[onbits] = 1.0
    return arr


def erg_to_np(fp) -> np.ndarray:
    # ErG fingerprint in RDKit is explicit vector-like object
    try:
        return np.array(list(fp), dtype=np.float32)
    except Exception:
        s = str(fp).strip().replace("[", "").replace("]", "").split(",")
        return np.array([float(x) for x in s if x.strip()], dtype=np.float32)


def calc_features(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # fallback zero vector
        morgan = np.zeros((2048,), dtype=np.float32)
        avalon = np.zeros((512,), dtype=np.float32)
        erg = np.zeros((315,), dtype=np.float32)
        desc = np.zeros((len(DESC_FUNCS),), dtype=np.float32)
        return np.concatenate([morgan, avalon, erg, desc], axis=0)

    morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    avalon_fp = pyAvalonTools.GetAvalonFP(mol, nBits=512)
    erg_fp = rdReducedGraphs.GetErGFingerprint(mol)

    morgan = fp_to_np(morgan_fp)
    avalon = fp_to_np(avalon_fp)
    erg = erg_to_np(erg_fp)

    desc = []
    for _, fn in DESC_FUNCS:
        try:
            v = float(fn(mol))
        except Exception:
            v = 0.0
        if not np.isfinite(v):
            v = 0.0
        desc.append(v)
    desc = np.array(desc, dtype=np.float32)

    return np.concatenate([morgan, avalon, erg, desc], axis=0)


def build_xy(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    smiles_col, label_col = find_smiles_and_label_cols(df)
    X = np.stack([calc_features(s) for s in df[smiles_col].astype(str).tolist()], axis=0)
    y = df[label_col].to_numpy()
    return X, y


def evaluate_classification(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "test_auc": float(roc_auc_score(y_true, y_prob)),
        "test_auprc": float(average_precision_score(y_true, y_prob)),
        "test_acc": float(accuracy_score(y_true, y_pred)),
    }
    return out


def evaluate_regression(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sp = spearmanr(y_true, y_pred).statistic
    if sp is None or not np.isfinite(sp):
        sp = np.nan
    out = {
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
        "test_rmse": rmse,
        "test_spearman": float(sp) if np.isfinite(sp) else np.nan,
    }
    return out


def fit_one_task(task_dir: Path, task_name: str, seed: int) -> dict:
    X_train, y_train = build_xy(task_dir / "train.csv")
    X_valid, y_valid = build_xy(task_dir / "valid.csv")
    X_test, y_test = build_xy(task_dir / "test.csv")

    is_cls = task_name in CLASSIFICATION_TASKS

    if is_cls:
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=400,
            learning_rate=0.03,
            depth=6,
            random_seed=seed,
            verbose=False,
            task_type="GPU",
            devices="0",
        )
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
        p_valid = model.predict_proba(X_valid)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

        official = OFFICIAL_METRIC[task_name]
        valid_auc = roc_auc_score(y_valid, p_valid)
        valid_auprc = average_precision_score(y_valid, p_valid)
        best_valid_primary = valid_auc if official == "AUROC" else valid_auprc

        out = {
            "task": task_name,
            "task_type": "classification",
            "primary_metric_name": official,
            "primary_metric": np.nan,
            "best_valid_primary": float(best_valid_primary),
            "loss_type": "CatBoost",
            "best_epoch": int(getattr(model, "get_best_iteration", lambda: -1)() or -1),
            "device": "cpu",
            "seed": seed,
        }
        out.update(evaluate_classification(y_test, p_test))
        out["primary_metric"] = out["test_auc"] if official == "AUROC" else out["test_auprc"]
        return out

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=400,
        learning_rate=0.03,
        depth=6,
        random_seed=seed,
        verbose=False,
        task_type="GPU",
        devices="0",
    )
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)

    official = OFFICIAL_METRIC[task_name]
    valid_mae = mean_absolute_error(y_valid, pred_valid)
    valid_sp = spearmanr(y_valid, pred_valid).statistic
    if valid_sp is None or not np.isfinite(valid_sp):
        valid_sp = np.nan
    best_valid_primary = valid_mae if official == "MAE" else valid_sp

    out = {
        "task": task_name,
        "task_type": "regression",
        "primary_metric_name": official,
        "primary_metric": np.nan,
        "best_valid_primary": float(best_valid_primary) if np.isfinite(best_valid_primary) else np.nan,
        "loss_type": "CatBoost",
        "best_epoch": int(getattr(model, "get_best_iteration", lambda: -1)() or -1),
        "device": "cpu",
        "seed": seed,
    }
    out.update(evaluate_regression(y_test, pred_test))
    out["primary_metric"] = out["test_mae"] if official == "MAE" else out["test_spearman"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-new", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_new)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    run_dir = out_root / "run_maplight_style"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = {}

    for task in args.tasks:
        try:
            row = fit_one_task(data_root / task, task, args.seed)
            rows.append(row)
            if row["task_type"] == "classification":
                print(f"[{task}] type=classification {row['primary_metric_name']}={row['primary_metric']:.6f}")
            else:
                print(f"[{task}] type=regression {row['primary_metric_name']}={row['primary_metric']:.6f}")
        except Exception as e:
            errors[task] = str(e)
            print(f"[{task}] FAILED: {e}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(run_dir / "results_all.csv", index=False)
        print(f"\nDone. Summary: {run_dir / 'results_all.csv'}")

    if errors:
        (run_dir / "errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False))
        print(f"Failures: {len(errors)} (see {run_dir / 'errors.json'})")


if __name__ == "__main__":
    main()
