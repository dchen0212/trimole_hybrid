from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path("/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import descriptor_sidecar_official_v1 as chem


DATA_ROOT = REPO / "data" / "data_benchmark_official_v1"
MASTER = REPO / "results_strict" / "ept_family_routing_master_v1" / "tdc_public_rank_audit_v23_valid_ose_endpoint.csv"
OUT = REPO / "results_strict" / "pure_chem_multibackend_endpoint_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--master", default=str(MASTER))
    p.add_argument("--out-root", default=str(OUT))
    p.add_argument("--tasks", nargs="*", default=["hia_hou", "bbb_martins", "cyp3a4_substrate_carbonmangels", "herg"])
    p.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    return p.parse_args()


def read_refs(path: Path):
    rows = list(csv.DictReader(path.open()))
    return {r["task"]: r for r in rows}


def label_col(df: pd.DataFrame) -> str:
    for col in ("label", "Y", "y", "target"):
        if col in df.columns:
            return col
    skip = {"smiles", "drug", "drug_id", "mol", "id", "sample_idx"}
    for col in df.columns:
        if col.lower() not in skip:
            return col
    raise KeyError("label col not found")


def smiles_col(df: pd.DataFrame) -> str:
    for col in ("smiles", "Drug", "SMILES", "drug"):
        if col in df.columns:
            return col
    raise KeyError("smiles col not found")


def task_type(y: np.ndarray) -> str:
    y = np.asarray(y)
    rounded = np.rint(y)
    if np.allclose(y, rounded) and set(int(x) for x in np.unique(rounded)).issubset({0, 1}):
        return "classification"
    return "regression"


def model_space(kind: str, seed: int):
    if kind == "classification":
        return {
            "extratrees": ExtraTreesClassifier(n_estimators=800, min_samples_leaf=1, max_features="sqrt", class_weight="balanced_subsample", random_state=seed, n_jobs=8),
            "rf": RandomForestClassifier(n_estimators=800, min_samples_leaf=1, max_features="sqrt", class_weight="balanced_subsample", random_state=seed, n_jobs=8),
            "gb": GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, random_state=seed),
            "hgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, l2_regularization=0.01, random_state=seed),
            "logreg": make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)),
        }
    return {
        "extratrees": ExtraTreesRegressor(n_estimators=800, min_samples_leaf=1, max_features="sqrt", random_state=seed, n_jobs=8),
        "rf": RandomForestRegressor(n_estimators=800, min_samples_leaf=1, max_features="sqrt", random_state=seed, n_jobs=8),
        "gb": GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, random_state=seed),
        "hgb": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, l2_regularization=0.01, random_state=seed),
        "ridge": make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0, random_state=seed)),
    }


def pred(model, X, kind: str):
    if kind == "classification" and hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.asarray(model.predict(X), dtype=float)


def better(metric: str, a: float, b: float) -> bool:
    return a < b if metric.upper() in {"MAE", "RMSE"} else a > b


def run_task(task: str, refs: dict[str, dict[str, str]], args: argparse.Namespace) -> dict[str, object]:
    task_dir = Path(args.data_root) / task
    tr = pd.read_csv(task_dir / "train.csv")
    va = pd.read_csv(task_dir / "valid.csv")
    te = pd.read_csv(task_dir / "test.csv")
    s = smiles_col(tr)
    y = label_col(tr)
    X_tr = chem.get_fingerprints(tr[s])
    X_va = chem.get_fingerprints(va[s])
    X_te = chem.get_fingerprints(te[s])
    y_tr = tr[y].to_numpy()
    y_va = va[y].to_numpy()
    y_te = te[y].to_numpy()
    kind = task_type(y_tr)
    metric = refs[task]["metric"]
    if not metric:
        metric = refs[task].get("tdc_metric", "")
    if not metric:
        metric = refs[task].get("metric", "")
    metric = {"SPEARMAN": "Spearman"}.get(metric.upper(), metric)
    top1 = float(refs[task].get("top1_displayed") or refs[task].get("top1_ref") or refs[task].get("tdc_top1_ref"))

    rows = []
    for seed in args.seeds:
        for name, model in model_space(kind, seed).items():
            model.fit(X_tr, y_tr)
            pv = pred(model, X_va, kind)
            pt = pred(model, X_te, kind)
            rows.append({
                "task": task,
                "seed": seed,
                "backend": name,
                "valid_score": chem.score_metric(metric, y_va, pv),
                "test_score_train_only": chem.score_metric(metric, y_te, pt),
            })
    df = pd.DataFrame(rows)
    ascending = metric.upper() in {"MAE", "RMSE"}
    selected_backend = df.groupby("backend")["valid_score"].mean().sort_values(ascending=ascending).index[0]

    refit_rows = []
    X_tv = np.concatenate([X_tr, X_va], axis=0)
    y_tv = np.concatenate([y_tr, y_va], axis=0)
    test_preds = []
    for seed in args.seeds:
        model = model_space(kind, seed)[selected_backend]
        model.fit(X_tv, y_tv)
        pte = pred(model, X_te, kind)
        test_preds.append(pte)
        refit_rows.append({
            "task": task,
            "seed": seed,
            "backend": selected_backend,
            "test_score": chem.score_metric(metric, y_te, pte),
        })
    ens_pred = np.mean(np.stack(test_preds, axis=0), axis=0)
    ens_score = chem.score_metric(metric, y_te, ens_pred)
    out_dir = Path(args.out_root) / task
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "valid_backend_search.csv", index=False)
    pd.DataFrame(refit_rows).to_csv(out_dir / "refit_seed_scores.csv", index=False)
    chem.write_predictions(out_dir / "test_ensemble_predictions.csv", y_te, ens_pred, kind)
    result = {
        "task": task,
        "metric": metric,
        "top1_ref": top1,
        "selected_backend": selected_backend,
        "selection_rule": "mean official-valid score across five seeds, then refit on train+valid",
        "valid_mean_selected": float(df[df.backend == selected_backend]["valid_score"].mean()),
        "test_mean_refit": float(np.mean([x["test_score"] for x in refit_rows])),
        "test_std_refit": float(np.std([x["test_score"] for x in refit_rows], ddof=0)),
        "test_ensemble": float(ens_score),
        "beats_top1_mean": bool(better(metric, float(np.mean([x["test_score"] for x in refit_rows])), top1)),
        "beats_top1_ensemble": bool(better(metric, float(ens_score), top1)),
    }
    Path(out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    refs = read_refs(Path(args.master))
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for task in args.tasks:
        print("[task]", task, flush=True)
        try:
            result = run_task(task, refs, args)
            results.append(result)
            print(result, flush=True)
        except Exception as exc:
            results.append({"task": task, "status": "error", "error": str(exc)})
            print("ERROR", task, exc, flush=True)
    fields = sorted({k for r in results for k in r})
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(out / "summary.csv")


if __name__ == "__main__":
    main()
