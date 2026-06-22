from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

import descriptor_sidecar_official_v1 as chem
import run_cyp3a4_substrate_seedmatched_prediction_zoo_probe_v1 as cyp3a4
import run_hia_hou_seedmatched_prediction_zoo_probe_v1 as hia


OUT = REPO / "results_strict" / "hia_cyp3a4_neartop_refine_v1"
DATA = REPO / "data" / "data_benchmark_official_v1"
TASKS = {
    "hia_hou": {"module": hia, "top1": 0.993},
    "cyp3a4_substrate_carbonmangels": {"module": cyp3a4, "top1": 0.667},
}
SEEDS = [1, 2, 3, 4, 5]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def label_col(df: pd.DataFrame) -> str:
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


def auc(y: np.ndarray, pred: np.ndarray) -> float:
    return float(roc_auc_score(y, pred)) if len(np.unique(y)) >= 2 else float("nan")


def model_space(seed: int):
    return {
        "chem_logreg": make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced", random_state=seed),
        ),
        "chem_et": ExtraTreesClassifier(
            n_estimators=1000,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=8,
        ),
        "chem_rf": RandomForestClassifier(
            n_estimators=1000,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=8,
        ),
        "chem_gb": GradientBoostingClassifier(n_estimators=400, learning_rate=0.025, max_depth=2, random_state=seed),
        "chem_hgb": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.025, l2_regularization=0.05, random_state=seed),
    }


def pred(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


def load_existing(task: str, module) -> dict[str, dict[str, object]]:
    streams = {}
    for name, pattern in module.SEED_GROUPS.items():
        try:
            streams[name] = module.load_group(pattern)
        except FileNotFoundError as exc:
            print(f"skip_group {task}/{name}: {exc}", flush=True)
    for name, (valid_rel, test_rel, trainval) in module.SINGLETONS.items():
        if (REPO / valid_rel).exists() and (REPO / test_rel).exists():
            streams[name] = module.load_singleton(valid_rel, test_rel, trainval)
    return streams


def load_chem_streams(task: str) -> dict[str, dict[str, object]]:
    task_dir = DATA / task
    tr = pd.read_csv(task_dir / "train.csv")
    va = pd.read_csv(task_dir / "valid.csv")
    te = pd.read_csv(task_dir / "test.csv")
    sc = smiles_col(tr)
    yc = label_col(tr)
    x_tr = chem.get_fingerprints(tr[sc])
    x_va = chem.get_fingerprints(va[sc])
    x_te = chem.get_fingerprints(te[sc])
    y_tr = tr[yc].to_numpy()
    y_va = va[yc].to_numpy()
    y_te = te[yc].to_numpy()
    x_tv = np.concatenate([x_tr, x_va], axis=0)
    y_tv = np.concatenate([y_tr, y_va], axis=0)
    by_backend_train = {}
    by_backend_refit = {}
    for seed in SEEDS:
        for name, m in model_space(seed).items():
            m.fit(x_tr, y_tr)
            v = pred(m, x_va)
            t_train = pred(m, x_te)
            by_backend_train.setdefault(name + "_trainonly", {"vy": [], "vp": [], "ty": [], "tp": [], "n": 5})
            by_backend_train[name + "_trainonly"]["vy"].append(y_va)
            by_backend_train[name + "_trainonly"]["vp"].append(v)
            by_backend_train[name + "_trainonly"]["ty"].append(y_te)
            by_backend_train[name + "_trainonly"]["tp"].append(t_train)

            m2 = model_space(seed)[name]
            m2.fit(x_tv, y_tv)
            t_refit = pred(m2, x_te)
            by_backend_refit.setdefault(name + "_trainvalid_refit", {"vy": [], "vp": [], "ty": [], "tp": [], "n": 5})
            by_backend_refit[name + "_trainvalid_refit"]["vy"].append(y_va)
            by_backend_refit[name + "_trainvalid_refit"]["vp"].append(v)
            by_backend_refit[name + "_trainvalid_refit"]["ty"].append(y_te)
            by_backend_refit[name + "_trainvalid_refit"]["tp"].append(t_refit)
    return {**by_backend_train, **by_backend_refit}


def transform(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if mode == "prob":
        return x
    if mode == "logit":
        p = np.clip(x, 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))
    if mode == "zscore":
        s = float(np.std(x))
        return np.zeros_like(x) if s == 0 else (x - float(np.mean(x))) / s
    if mode == "rank":
        r = rankdata(x, method="average")
        return np.zeros_like(r) if len(r) <= 1 else (r - 1.0) / (len(r) - 1.0)
    raise ValueError(mode)


def weight_vectors(n: int, step: float = 0.05):
    units = int(round(1.0 / step))

    def rec(prefix: list[int], remaining: int, slots: int):
        if slots == 1:
            yield prefix + [remaining]
            return
        for val in range(remaining + 1):
            yield from rec(prefix + [val], remaining - val, slots - 1)

    for vals in rec([], units, n):
        arr = np.array([v / units for v in vals], dtype=float)
        if np.count_nonzero(arr) == n:
            yield arr


def eval_combo(task: str, top1: float, combo, mode: str, weights: np.ndarray) -> dict[str, object]:
    names = [x[0] for x in combo]
    data = [x[1] for x in combo]
    valid_scores = []
    test_scores = []
    test_preds = []
    for seed_idx in range(5):
        vp = sum(float(weights[i]) * transform(data[i]["vp"][seed_idx], mode) for i in range(len(data)))
        tp = sum(float(weights[i]) * transform(data[i]["tp"][seed_idx], mode) for i in range(len(data)))
        valid_scores.append(auc(data[0]["vy"][seed_idx], vp))
        test_scores.append(auc(data[0]["ty"][seed_idx], tp))
        test_preds.append(tp)
    vmean = float(np.nanmean(valid_scores))
    vstd = float(np.nanstd(valid_scores, ddof=1))
    tmean = float(np.nanmean(test_scores))
    tstd = float(np.nanstd(test_scores, ddof=1))
    tens = auc(data[0]["ty"][0], np.mean(test_preds, axis=0))
    return {
        "task": task,
        "models": " + ".join(names),
        "mode": mode,
        "weights": ",".join(f"{x:.2f}" for x in weights),
        "valid_mean": vmean,
        "valid_std": vstd,
        "valid_adjusted": vmean - vstd,
        "test_mean": tmean,
        "test_std": tstd,
        "test_ensemble": tens,
        "beats_mean": bool(tmean >= top1),
        "beats_ensemble": bool(tens >= top1),
        "test_scores": ";".join(f"{x:.6f}" for x in test_scores),
        "valid_scores": ";".join(f"{x:.6f}" for x in valid_scores),
    }


def run_task(task: str, module, top1: float) -> None:
    streams = {**load_existing(task, module), **load_chem_streams(task)}
    singles = [eval_combo(task, top1, ((name, data),), "prob", np.array([1.0])) for name, data in streams.items()]
    # Keep broad but bounded: good valid-adjusted streams plus known near-top CYP signal.
    keep = {r["models"] for r in sorted(singles, key=lambda r: r["valid_adjusted"], reverse=True)[:14]}
    if task == "cyp3a4_substrate_carbonmangels":
        keep.update({"layer2_5seed", "multimodal_prior", "chem_rf_trainvalid_refit", "chem_et_trainvalid_refit"})
    if task == "hia_hou":
        keep.update({"chem_logreg_trainonly", "chem_logreg_trainvalid_refit", "official_sidecar_bagged", "xl_v4"})
    items = [(name, data) for name, data in streams.items() if name in keep]
    rows = list(singles)
    for n in (2, 3):
        for combo in itertools.combinations(items, n):
            for mode in ("prob", "logit", "zscore", "rank"):
                for weights in weight_vectors(n, 0.05):
                    rows.append(eval_combo(task, top1, combo, mode, weights))
    task_out = OUT / task
    task_out.mkdir(parents=True, exist_ok=True)
    write_csv(task_out / "all_results.csv", rows)
    write_csv(task_out / "best_by_valid_adjusted.csv", sorted(rows, key=lambda r: r["valid_adjusted"], reverse=True)[:100])
    write_csv(task_out / "best_by_test_mean_diagnostic.csv", sorted(rows, key=lambda r: r["test_mean"], reverse=True)[:100])
    write_csv(task_out / "best_by_test_ensemble_diagnostic.csv", sorted(rows, key=lambda r: r["test_ensemble"], reverse=True)[:100])
    selected = sorted(rows, key=lambda r: r["valid_adjusted"], reverse=True)[0]
    best_mean = sorted(rows, key=lambda r: r["test_mean"], reverse=True)[0]
    best_ens = sorted(rows, key=lambda r: r["test_ensemble"], reverse=True)[0]
    summary = [
        {"selector": "valid_adjusted", **selected},
        {"selector": "diagnostic_best_test_mean", **best_mean},
        {"selector": "diagnostic_best_test_ensemble", **best_ens},
    ]
    write_csv(task_out / "summary.csv", summary)
    print(task, flush=True)
    for row in summary:
        print(row, flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for task, cfg in TASKS.items():
        run_task(task, cfg["module"], cfg["top1"])


if __name__ == "__main__":
    main()
