from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

# Sidecar is meant to be a light tabular calibrator, not a GPU job.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP
from rdkit.Chem import DataStructs, rdReducedGraphs
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    XGBClassifier = None  # type: ignore
    XGBRegressor = None  # type: ignore

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    HAS_SK = True
except Exception:
    HAS_SK = False
    RandomForestClassifier = None  # type: ignore
    RandomForestRegressor = None  # type: ignore


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
DATA_ROOT = REPO / "data" / "data_benchmark_official_v1"
MASTER = REPO / "results_strict" / "ept_family_routing_master_v1" / "ept_family_routing_master_v1_patched_v3_5seed.csv"
OUT_ROOT = REPO / "results_strict" / "descriptor_sidecar_official_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", type=str, default=str(MASTER))
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    p.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    p.add_argument("--tasks", nargs="*", default=[])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def count_to_array(fingerprint):
    n = int(fingerprint.GetLength()) if hasattr(fingerprint, "GetLength") else 0
    array = np.zeros((n,), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


def get_avalon_fingerprints(molecules, n_bits=1024):
    fps = molecules.apply(lambda x: GetAvalonCountFP(x, nBits=n_bits))
    fps = fps.apply(count_to_array)
    return np.stack(fps.values)


def get_morgan_fingerprints(molecules, n_bits=1024, radius=2):
    fps = molecules.apply(lambda x: GetHashedMorganFingerprint(x, nBits=n_bits, radius=radius))
    fps = fps.apply(count_to_array)
    return np.stack(fps.values)


def get_erg_fingerprints(molecules):
    fps = molecules.apply(rdReducedGraphs.GetErGFingerprint)
    return np.stack(fps.values)


def get_chosen_descriptors():
    return ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1',
        'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
        'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3',
        'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8',
        'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2',
        'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount',
        'HeavyAtomMolWt', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA',
        'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge',
        'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge',
        'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
        'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles',
        'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors',
        'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds',
        'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
        'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12',
        'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
        'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1',
        'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7',
        'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
        'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
        'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2',
        'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
        'VSA_EState8', 'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
        'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O',
        'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2',
        'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH',
        'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid',
        'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo',
        'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
        'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido',
        'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',
        'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone',
        'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
        'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime',
        'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid',
        'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
        'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone',
        'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene',
        'fr_unbrch_alkane', 'fr_urea', 'qed']


def get_rdkit_features(molecules):
    calc = MolecularDescriptorCalculator(get_chosen_descriptors())
    X = molecules.apply(lambda x: np.array(calc.CalcDescriptors(x)))
    return np.vstack(X.values)


def sanitize_features(x: np.ndarray, clip_value: float = 1e6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x.astype(np.float32, copy=False)


def get_fingerprints(smiles_series: pd.Series) -> np.ndarray:
    RDLogger.DisableLog('rdApp.*')
    mols = smiles_series.apply(Chem.MolFromSmiles)
    return sanitize_features(np.concatenate(
        [
            get_morgan_fingerprints(mols),
            get_avalon_fingerprints(mols),
            get_erg_fingerprints(mols),
            get_rdkit_features(mols),
        ],
        axis=1,
    ))


def find_smiles_col(df: pd.DataFrame) -> str:
    for col in ("smiles", "Drug", "SMILES", "drug"):
        if col in df.columns:
            return col
    raise KeyError(f"smiles column not found in {list(df.columns)}")


def find_label_col(df: pd.DataFrame) -> str:
    for col in ("label", "Y", "y", "target"):
        if col in df.columns:
            return col
    raise KeyError(f"label column not found in {list(df.columns)}")


def load_concat_2d_embeddings(task_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    root = task_dir / "embeddings"
    return np.load(root / "chemberta.npy"), np.load(root / "kpgt.npy")


def load_ept_embeddings(task_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = task_dir / "embeddings_ept"
    return (
        np.load(root / "train_ept.npy"),
        np.load(root / "valid_ept.npy"),
        np.load(root / "test_ept.npy"),
    )


def split_concat(arr: np.ndarray, n_tr: int, n_va: int, n_te: int):
    return arr[:n_tr], arr[n_tr:n_tr + n_va], arr[n_tr + n_va:n_tr + n_va + n_te]


def build_embedding_features(task_dir: Path, candidate: str, n_tr: int, n_va: int, n_te: int):
    chem_all, kpgt_all = load_concat_2d_embeddings(task_dir)
    chem_tr, chem_va, chem_te = split_concat(chem_all, n_tr, n_va, n_te)
    kpgt_tr, kpgt_va, kpgt_te = split_concat(kpgt_all, n_tr, n_va, n_te)
    ept_tr, ept_va, ept_te = load_ept_embeddings(task_dir)

    if candidate == "chemberta":
        return chem_tr, chem_va, chem_te
    if candidate == "kpgt":
        return kpgt_tr, kpgt_va, kpgt_te
    if candidate == "ept":
        return ept_tr, ept_va, ept_te
    if candidate == "chemberta_kpgt":
        return (
            np.concatenate([chem_tr, kpgt_tr], axis=1),
            np.concatenate([chem_va, kpgt_va], axis=1),
            np.concatenate([chem_te, kpgt_te], axis=1),
        )
    if candidate == "chemberta_ept":
        return (
            np.concatenate([chem_tr, ept_tr], axis=1),
            np.concatenate([chem_va, ept_va], axis=1),
            np.concatenate([chem_te, ept_te], axis=1),
        )
    if candidate == "kpgt_ept":
        return (
            np.concatenate([kpgt_tr, ept_tr], axis=1),
            np.concatenate([kpgt_va, ept_va], axis=1),
            np.concatenate([kpgt_te, ept_te], axis=1),
        )
    if candidate == "chemberta_kpgt_ept_gated":
        return (
            np.concatenate([chem_tr, kpgt_tr, ept_tr], axis=1),
            np.concatenate([chem_va, kpgt_va, ept_va], axis=1),
            np.concatenate([chem_te, kpgt_te, ept_te], axis=1),
        )
    raise ValueError(f"Unsupported candidate: {candidate}")


def build_features(task_dir: Path, candidate: str):
    train_df = pd.read_csv(task_dir / "train.csv")
    valid_df = pd.read_csv(task_dir / "valid.csv")
    test_df = pd.read_csv(task_dir / "test.csv")
    s_col = find_smiles_col(train_df)
    y_col = find_label_col(train_df)

    n_tr, n_va, n_te = len(train_df), len(valid_df), len(test_df)
    emb_tr, emb_va, emb_te = build_embedding_features(task_dir, candidate, n_tr, n_va, n_te)

    fp_tr = get_fingerprints(train_df[s_col])
    fp_va = get_fingerprints(valid_df[s_col])
    fp_te = get_fingerprints(test_df[s_col])

    X_tr = sanitize_features(np.concatenate([emb_tr.astype(np.float32), fp_tr], axis=1))
    X_va = sanitize_features(np.concatenate([emb_va.astype(np.float32), fp_va], axis=1))
    X_te = sanitize_features(np.concatenate([emb_te.astype(np.float32), fp_te], axis=1))
    y_tr = train_df[y_col].to_numpy()
    y_va = valid_df[y_col].to_numpy()
    y_te = test_df[y_col].to_numpy()
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def infer_task_type(y: np.ndarray) -> str:
    y = np.asarray(y)
    if np.all(np.isfinite(y)):
        r = np.rint(y)
        if np.allclose(y, r):
            unique = set(int(v) for v in np.unique(r))
            if unique.issubset({0, 1}):
                return "classification"
    return "regression"


def score_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    metric_name = str(metric_name).upper()
    if metric_name == "AUROC":
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_pred))
    if metric_name == "AUPRC":
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, y_pred))
    if metric_name == "MAE":
        from sklearn.metrics import mean_absolute_error
        return float(mean_absolute_error(y_true, y_pred))
    if metric_name == "RMSE":
        from sklearn.metrics import mean_squared_error
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric_name == "SPEARMAN":
        import scipy.stats as st
        return float(st.spearmanr(y_true, y_pred).correlation)
    raise ValueError(metric_name)


def predict_model(model, X: np.ndarray, task_type: str) -> np.ndarray:
    if task_type == "classification":
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        pred = model.predict(X)
        return np.asarray(pred, dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def fit_model(X_tr, y_tr, X_va, y_va, task_type: str, metric: str, seed: int = 42):
    if HAS_XGB:
        if task_type == "classification":
            model = XGBClassifier(
                n_estimators=800,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=2,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cpu",
                random_state=seed,
                n_jobs=8,
                early_stopping_rounds=60,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            return model, "xgboost"
        model = XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            objective="reg:squarederror",
            tree_method="hist",
            device="cpu",
            random_state=seed,
            n_jobs=8,
            early_stopping_rounds=60,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return model, "xgboost"

    if HAS_SK:
        if task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=8,
                random_state=seed,
            )
            model.fit(X_tr, y_tr)
            return model, "random_forest"
        model = RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=8,
            random_state=seed,
        )
        model.fit(X_tr, y_tr)
        return model, "random_forest"

    raise RuntimeError("Need xgboost or sklearn in runtime for descriptor sidecar")


def direction_better(new_value: float, old_value: float, direction: str) -> bool:
    return new_value > old_value if direction == "max" else new_value < old_value


def write_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> None:
    rows = []
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if task_type == "classification":
            rows.append({"sample_idx": i, "y_true": int(yt), "y_prob": float(yp)})
        else:
            rows.append({"sample_idx": i, "y_true": float(yt), "y_pred": float(yp)})
    pd.DataFrame(rows).to_csv(path, index=False)


def iter_plan(master_path: Path, tasks: list[str], limit: int):
    with master_path.open() as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if str(r.get("selected", "False")).lower() == "true"]
    if tasks:
        wanted = set(tasks)
        rows = [r for r in rows if r["task"] in wanted]
    if limit > 0:
        rows = rows[:limit]
    return rows


def run_one(entry: dict[str, str], data_root: Path, out_root: Path, force: bool):
    task = entry["task"]
    candidate = entry["candidate"]
    task_dir = data_root / task
    out_dir = out_root / f"{task}__{candidate}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "result.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text())

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = build_features(task_dir, candidate)
    task_type = infer_task_type(y_tr)
    model, backend = fit_model(X_tr, y_tr, X_va, y_va, task_type, entry["tdc_metric"])
    p_va = predict_model(model, X_va, task_type)
    p_te = predict_model(model, X_te, task_type)

    valid_score = score_metric(entry["tdc_metric"], y_va, p_va)
    test_score = score_metric(entry["tdc_metric"], y_te, p_te)

    write_predictions(out_dir / "valid_predictions.csv", y_va, p_va, task_type)
    write_predictions(out_dir / "test_predictions.csv", y_te, p_te, task_type)

    result = {
        "task": task,
        "candidate": candidate,
        "head": entry.get("head", ""),
        "tdc_metric": entry["tdc_metric"],
        "metric_direction": entry["metric_direction"],
        "backend": backend,
        "feature_type": "winner_embedding_plus_rdkit_fp",
        "valid_tdc_score": float(valid_score),
        "test_tdc_score": float(test_score),
        "incumbent_valid_tdc_score": float(entry["valid_tdc_score_mean"] or entry["valid_tdc_score"]),
        "incumbent_test_tdc_score": float(entry["test_tdc_score_mean"] or entry["test_tdc_score"]),
        "improved_valid": direction_better(float(valid_score), float(entry["valid_tdc_score_mean"] or entry["valid_tdc_score"]), entry["metric_direction"]),
        "improved_test": direction_better(float(test_score), float(entry["test_tdc_score_mean"] or entry["test_tdc_score"]), entry["metric_direction"]),
        "tdc_top1_ref": float(entry["tdc_top1_ref"]),
        "gap_vs_top1_ref": abs(float(test_score) - float(entry["tdc_top1_ref"])),
        "valid_pred_file": str(out_dir / "valid_predictions.csv"),
        "test_pred_file": str(out_dir / "test_predictions.csv"),
        "out_dir": str(out_dir),
    }
    summary_path.write_text(json.dumps(result, indent=2))
    return result


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = iter_plan(Path(args.master), args.tasks, args.limit)
    results = []
    total = len(rows)
    for idx, entry in enumerate(rows, start=1):
        print(f"[{idx}/{total}] {entry['task']}::{entry['candidate']}", flush=True)
        try:
            results.append(run_one(entry, data_root, out_root, args.force))
        except Exception as exc:
            results.append(
                {
                    "task": entry["task"],
                    "candidate": entry["candidate"],
                    "tdc_metric": entry["tdc_metric"],
                    "metric_direction": entry["metric_direction"],
                    "status": "error",
                    "error": str(exc),
                }
            )

    fieldnames = sorted({k for row in results for k in row})
    summary_csv = out_root / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    meta = {
        "master": str(args.master),
        "data_root": str(args.data_root),
        "count": len(rows),
        "tasks": [r["task"] for r in rows],
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2))
    print(summary_csv)


if __name__ == "__main__":
    main()
