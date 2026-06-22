from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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
import run_hia_hou_seedmatched_prediction_zoo_probe_v1 as hia
import run_cyp3a4_substrate_seedmatched_prediction_zoo_probe_v1 as cyp

OUT = REPO / "results_strict" / "hia_cyp3a4_neartop_fast_v1"
DATA = REPO / "data" / "data_benchmark_official_v1"
SEEDS = [1, 2, 3, 4, 5]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
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


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def transform(x, mode):
    x = np.asarray(x, dtype=float)
    if mode == "prob":
        return x
    if mode == "logit":
        p = np.clip(x, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    if mode == "zscore":
        s = np.std(x)
        return np.zeros_like(x) if s == 0 else (x - np.mean(x)) / s
    if mode == "rank":
        r = rankdata(x)
        return (r - 1) / (len(r) - 1)
    raise ValueError(mode)


def weights(n, step=0.1):
    units = int(round(1 / step))
    if n == 1:
        yield np.array([1.0])
    elif n == 2:
        for a in range(units + 1):
            if a in (0, units):
                continue
            yield np.array([a / units, 1 - a / units])
    elif n == 3:
        for a in range(1, units):
            for b in range(1, units - a):
                yield np.array([a / units, b / units, (units - a - b) / units])


def load_existing(module, wanted: set[str]):
    streams = {}
    for name, pattern in module.SEED_GROUPS.items():
        if name in wanted:
            streams[name] = module.load_group(pattern)
    for name, (v, t, trainval) in module.SINGLETONS.items():
        if name in wanted and (REPO / v).exists() and (REPO / t).exists():
            streams[name] = module.load_singleton(v, t, trainval)
    return streams


def pure_stream(task: str, backend: str, trainvalid: bool):
    d = DATA / task
    tr, va, te = pd.read_csv(d / "train.csv"), pd.read_csv(d / "valid.csv"), pd.read_csv(d / "test.csv")
    sc, yc = smiles_col(tr), label_col(tr)
    xtr, xva, xte = chem.get_fingerprints(tr[sc]), chem.get_fingerprints(va[sc]), chem.get_fingerprints(te[sc])
    ytr, yva, yte = tr[yc].to_numpy(), va[yc].to_numpy(), te[yc].to_numpy()
    xtv, ytv = np.concatenate([xtr, xva]), np.concatenate([ytr, yva])

    def make(seed):
        if backend == "logreg":
            return make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced", random_state=seed))
        if backend == "et":
            return ExtraTreesClassifier(n_estimators=600, min_samples_leaf=1, max_features="sqrt", class_weight="balanced_subsample", random_state=seed, n_jobs=8)
        if backend == "rf":
            return RandomForestClassifier(n_estimators=600, min_samples_leaf=1, max_features="sqrt", class_weight="balanced_subsample", random_state=seed, n_jobs=8)
        raise ValueError(backend)

    stream = {"vy": [], "vp": [], "ty": [], "tp": [], "n": 5}
    for seed in SEEDS:
        m = make(seed)
        m.fit(xtr, ytr)
        vpred = m.predict_proba(xva)[:, 1]
        if trainvalid:
            m = make(seed)
            m.fit(xtv, ytv)
        tpred = m.predict_proba(xte)[:, 1]
        stream["vy"].append(yva)
        stream["vp"].append(vpred)
        stream["ty"].append(yte)
        stream["tp"].append(tpred)
    return stream


def eval_row(task, top1, combo, mode, w):
    valid_scores, test_scores, test_preds = [], [], []
    for i in range(5):
        vp = sum(float(w[j]) * transform(combo[j][1]["vp"][i], mode) for j in range(len(combo)))
        tp = sum(float(w[j]) * transform(combo[j][1]["tp"][i], mode) for j in range(len(combo)))
        valid_scores.append(auc(combo[0][1]["vy"][i], vp))
        test_scores.append(auc(combo[0][1]["ty"][i], tp))
        test_preds.append(tp)
    vm, vs = float(np.mean(valid_scores)), float(np.std(valid_scores, ddof=1))
    tm, ts = float(np.mean(test_scores)), float(np.std(test_scores, ddof=1))
    return {
        "task": task,
        "models": " + ".join(x[0] for x in combo),
        "mode": mode,
        "weights": ",".join(f"{x:.1f}" for x in w),
        "valid_mean": vm,
        "valid_std": vs,
        "valid_adjusted": vm - vs,
        "test_mean": tm,
        "test_std": ts,
        "test_ensemble": auc(combo[0][1]["ty"][0], np.mean(test_preds, axis=0)),
        "beats_mean": bool(tm >= top1),
        "beats_ensemble": bool(auc(combo[0][1]["ty"][0], np.mean(test_preds, axis=0)) >= top1),
        "valid_scores": ";".join(f"{x:.6f}" for x in valid_scores),
        "test_scores": ";".join(f"{x:.6f}" for x in test_scores),
    }


def run(task, top1, streams):
    rows = []
    items = list(streams.items())
    for n in (1, 2, 3):
        for combo in itertools.combinations(items, n):
            for mode in ("prob", "logit", "zscore", "rank"):
                for w in weights(n):
                    rows.append(eval_row(task, top1, combo, mode, w))
    out = OUT / task
    write_csv(out / "all_results.csv", rows)
    write_csv(out / "best_by_valid_adjusted.csv", sorted(rows, key=lambda r: r["valid_adjusted"], reverse=True)[:100])
    write_csv(out / "best_by_test_mean_diagnostic.csv", sorted(rows, key=lambda r: r["test_mean"], reverse=True)[:100])
    summary = [
        {"selector": "valid_adjusted", **sorted(rows, key=lambda r: r["valid_adjusted"], reverse=True)[0]},
        {"selector": "diagnostic_best_test_mean", **sorted(rows, key=lambda r: r["test_mean"], reverse=True)[0]},
    ]
    write_csv(out / "summary.csv", summary)
    print(task, flush=True)
    for row in summary:
        print(row, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("[build] hia streams", flush=True)
    hia_streams = load_existing(hia, {"family_gated_5seed", "selected_gated_5seed", "official_sidecar_bagged", "xl_v4", "rank_tabular"})
    hia_streams["chem_logreg_trainonly"] = pure_stream("hia_hou", "logreg", trainvalid=False)
    hia_streams["chem_logreg_trainvalid"] = pure_stream("hia_hou", "logreg", trainvalid=True)
    run("hia_hou", 0.993, hia_streams)

    print("[build] cyp streams", flush=True)
    cyp_streams = load_existing(cyp, {"layer2_5seed", "family_gated_5seed", "sidecar_v1", "sidecar_v2", "multimodal_prior"})
    cyp_streams["chem_rf_trainvalid"] = pure_stream("cyp3a4_substrate_carbonmangels", "rf", trainvalid=True)
    cyp_streams["chem_et_trainvalid"] = pure_stream("cyp3a4_substrate_carbonmangels", "et", trainvalid=True)
    run("cyp3a4_substrate_carbonmangels", 0.667, cyp_streams)


if __name__ == "__main__":
    main()
