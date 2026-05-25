from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
)

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

try:
    from rdkit.Avalon.pyAvalonTools import GetAvalonFP
    HAS_AVALON = True
except Exception:
    HAS_AVALON = False

from catboost import CatBoostClassifier, CatBoostRegressor


CLASSIFICATION_TASKS = {
    "ames",
    "bioavailability_ma",
    "bbb_martins",
    "cyp2c9_veith",
    "cyp2c9_substrate_carbonmangels",
    "cyp2d6_veith",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_veith",
    "cyp3a4_substrate_carbonmangels",
    "dili",
    "herg",
    "hia_hou",
    "pgp_broccatelli",
}

SPEARMAN_TASKS = {
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
    "half_life_obach",
    "vdss_lombardo",
}

MAE_TASKS = {
    "caco2_wang",
    "ld50_zhu",
    "lipophilicity_astrazeneca",
    "ppbr_az",
    "solubility_aqsoldb",
}


def infer_task_type(task_name: str) -> tuple[str, str]:
    t = task_name.lower().strip()
    if t in CLASSIFICATION_TASKS:
        return "classification", "AUROC"
    if t in SPEARMAN_TASKS:
        return "regression", "Spearman"
    if t in MAE_TASKS:
        return "regression", "MAE"
    return "classification", "AUROC"


def mol_from_smiles(smiles: str):
    if not isinstance(smiles, str):
        return None
    return Chem.MolFromSmiles(smiles)


def fp_to_np(fp, n_bits: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def safe_desc(mol, fn, default=0.0):
    try:
        v = fn(mol)
        if v is None or not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def build_features(smiles_list: list[str], ecfp_bits: int = 2048, avalon_bits: int = 1024) -> np.ndarray:
    rows = []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        if mol is None:
            ecfp = np.zeros((ecfp_bits,), dtype=np.float32)
            avalon = np.zeros((avalon_bits,), dtype=np.float32)
            erg = np.zeros((315,), dtype=np.float32)
            desc = np.zeros((12,), dtype=np.float32)
        else:
            ecfp = fp_to_np(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=ecfp_bits), ecfp_bits)

            if HAS_AVALON:
                avalon = fp_to_np(GetAvalonFP(mol, nBits=avalon_bits), avalon_bits)
            else:
                avalon = np.zeros((avalon_bits,), dtype=np.float32)

            try:
                erg = np.asarray(rdMolDescriptors.GetErGFingerprint(mol), dtype=np.float32)
            except Exception:
                erg = np.zeros((315,), dtype=np.float32)

            desc = np.asarray([
                safe_desc(mol, Descriptors.MolWt),
                safe_desc(mol, Descriptors.MolLogP),
                safe_desc(mol, Descriptors.TPSA),
                safe_desc(mol, Descriptors.NumHDonors),
                safe_desc(mol, Descriptors.NumHAcceptors),
                safe_desc(mol, Descriptors.NumRotatableBonds),
                safe_desc(mol, Descriptors.RingCount),
                safe_desc(mol, Descriptors.HeavyAtomCount),
                safe_desc(mol, Descriptors.FractionCSP3),
                safe_desc(mol, Descriptors.MolMR),
                safe_desc(mol, Descriptors.NHOHCount),
                safe_desc(mol, Descriptors.NOCount),
            ], dtype=np.float32)

        rows.append(np.concatenate([ecfp, avalon, erg, desc], axis=0))
    return np.vstack(rows).astype(np.float32)


def read_split(csv_path: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(csv_path)
    smiles_col = "Drug" if "Drug" in df.columns else ("smiles" if "smiles" in df.columns else df.columns[0])
    label_col = "Y" if "Y" in df.columns else ("label" if "label" in df.columns else df.columns[-1])
    smiles = df[smiles_col].astype(str).tolist()
    y = pd.to_numeric(df[label_col], errors="coerce").values
    return smiles, y


def fit_classifier(x_tr, y_tr, x_va, y_va):
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        iterations=2000,
        random_seed=42,
        verbose=False,
    )
    model.fit(
        x_tr, y_tr,
        eval_set=(x_va, y_va),
        use_best_model=True,
        verbose=False,
    )
    return model


def fit_regressor(primary_metric: str, x_tr, y_tr, x_va, y_va):
    eval_metric = "RMSE"
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric=eval_metric,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        iterations=2000,
        random_seed=42,
        verbose=False,
    )
    model.fit(
        x_tr, y_tr,
        eval_set=(x_va, y_va),
        use_best_model=True,
        verbose=False,
    )
    return model


def eval_classification(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) >= 2 else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)) if len(set(y_true.tolist())) >= 2 else float("nan"),
    }
    return out, y_pred


def eval_regression(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    try:
        sp = float(spearmanr(y_true, y_pred).statistic)
    except Exception:
        sp = float("nan")
    return {"mae": mae, "spearman": sp}


def run_task(task_dir: Path, out_root: Path):
    task_name = task_dir.name
    task_type, primary_metric = infer_task_type(task_name)

    tr_smiles, y_tr = read_split(task_dir / "train.csv")
    va_smiles, y_va = read_split(task_dir / "valid.csv")
    te_smiles, y_te = read_split(task_dir / "test.csv")

    x_tr = build_features(tr_smiles)
    x_va = build_features(va_smiles)
    x_te = build_features(te_smiles)

    out_dir = out_root / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if task_type == "classification":
        y_tr = y_tr.astype(int)
        y_va = y_va.astype(int)
        y_te = y_te.astype(int)

        model = fit_classifier(x_tr, y_tr, x_va, y_va)
        va_prob = model.predict_proba(x_va)[:, 1]
        te_prob = model.predict_proba(x_te)[:, 1]

        va_metrics, va_pred = eval_classification(y_va, va_prob)
        te_metrics, te_pred = eval_classification(y_te, te_prob)

        pd.DataFrame({
            "y_true": y_va,
            "y_prob": va_prob,
            "y_pred": va_pred,
        }).to_csv(out_dir / "valid_predictions.csv", index=False)

        pd.DataFrame({
            "y_true": y_te,
            "y_prob": te_prob,
            "y_pred": te_pred,
        }).to_csv(out_dir / "test_predictions.csv", index=False)

        result = {
            "task": task_name,
            "task_type": task_type,
            "model_family": "catboost",
            "feature_set": "ecfp4+avalon+erg+rdkit12",
            "primary_metric_name": primary_metric,
            "valid_acc": va_metrics["acc"],
            "valid_auc": va_metrics["auc"],
            "valid_auprc": va_metrics["auprc"],
            "test_acc": te_metrics["acc"],
            "test_auc": te_metrics["auc"],
            "test_auprc": te_metrics["auprc"],
            "primary_metric": te_metrics["auc"] if primary_metric == "AUROC" else te_metrics["auprc"],
        }

    else:
        y_tr = y_tr.astype(float)
        y_va = y_va.astype(float)
        y_te = y_te.astype(float)

        model = fit_regressor(primary_metric, x_tr, y_tr, x_va, y_va)
        va_pred = model.predict(x_va)
        te_pred = model.predict(x_te)

        va_metrics = eval_regression(y_va, va_pred)
        te_metrics = eval_regression(y_te, te_pred)

        pd.DataFrame({
            "y_true": y_va,
            "y_pred": va_pred,
        }).to_csv(out_dir / "valid_predictions.csv", index=False)

        pd.DataFrame({
            "y_true": y_te,
            "y_pred": te_pred,
        }).to_csv(out_dir / "test_predictions.csv", index=False)

        result = {
            "task": task_name,
            "task_type": task_type,
            "model_family": "catboost",
            "feature_set": "ecfp4+avalon+erg+rdkit12",
            "primary_metric_name": primary_metric,
            "valid_mae": va_metrics["mae"],
            "valid_spearman": va_metrics["spearman"],
            "test_mae": te_metrics["mae"],
            "test_spearman": te_metrics["spearman"],
            "primary_metric": te_metrics["spearman"] if primary_metric == "Spearman" else te_metrics["mae"],
        }

    pd.DataFrame([result]).to_csv(out_dir / "results_all.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {task_name} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-new", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--tasks", type=str, nargs="*", default=[])
    args = ap.parse_args()

    data_root = Path(args.data_new)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    task_dirs = [data_root / t for t in args.tasks] if args.tasks else [p for p in sorted(data_root.iterdir()) if p.is_dir()]
    for task_dir in task_dirs:
        if not ((task_dir / "train.csv").exists() and (task_dir / "valid.csv").exists() and (task_dir / "test.csv").exists()):
            continue
        run_task(task_dir, out_root)


if __name__ == "__main__":
    main()
