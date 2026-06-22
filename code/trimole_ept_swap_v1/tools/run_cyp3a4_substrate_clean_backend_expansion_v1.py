from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

import descriptor_sidecar_official_v1 as chem  # noqa: E402
import run_cyp3a4_substrate_seedmatched_prediction_zoo_probe_v1 as cyp  # noqa: E402


TASK = "cyp3a4_substrate_carbonmangels"
TOP1_REF = 0.667
DATA = REPO / "data" / "data_benchmark_official_v1"
OUT = REPO / "results_strict" / "cyp3a4_substrate_clean_backend_expansion_v1"
SEEDS = [1, 2, 3, 4, 5]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def label_col(df: pd.DataFrame) -> str:
    skip = {"smiles", "drug", "drug_id", "mol", "id", "sample_idx"}
    for col in df.columns:
        if col.lower() not in skip:
            return col
    raise KeyError("label column not found")


def smiles_col(df: pd.DataFrame) -> str:
    for col in ("smiles", "Drug", "SMILES", "drug"):
        if col in df.columns:
            return col
    raise KeyError("smiles column not found")


def score(y: np.ndarray, pred: np.ndarray) -> float:
    return float(roc_auc_score(y, pred)) if len(np.unique(y)) >= 2 else float("nan")


def transform(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if mode == "prob":
        return x
    if mode == "logit":
        p = np.clip(x, 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))
    if mode == "zscore":
        std = float(np.std(x))
        return np.zeros_like(x) if std == 0 else (x - float(np.mean(x))) / std
    if mode == "rank":
        r = rankdata(x, method="average")
        return np.zeros_like(r) if len(r) <= 1 else (r - 1.0) / (len(r) - 1.0)
    raise ValueError(mode)


def weights(n_models: int, step: float = 0.1):
    units = int(round(1.0 / step))
    if n_models == 1:
        yield np.array([1.0], dtype=np.float64)
        return
    if n_models == 2:
        for a in range(1, units):
            yield np.array([a / units, 1.0 - a / units], dtype=np.float64)
        return
    if n_models == 3:
        for a in range(1, units):
            for b in range(1, units - a):
                yield np.array([a / units, b / units, (units - a - b) / units], dtype=np.float64)


def load_existing_streams() -> dict[str, dict[str, object]]:
    streams: dict[str, dict[str, object]] = {}
    for name, pattern in cyp.SEED_GROUPS.items():
        if name in {"family_gated_5seed", "selected_gated_5seed", "layer2_5seed"}:
            try:
                streams[name] = cyp.load_group(pattern)
            except FileNotFoundError as exc:
                print(f"skip {name}: {exc}", flush=True)
    for name, (valid_rel, test_rel, trainval) in cyp.SINGLETONS.items():
        if name in {"sidecar_v1", "sidecar_v2", "official_sidecar_bagged", "multimodal_prior", "xl_v4"}:
            if (REPO / valid_rel).exists() and (REPO / test_rel).exists():
                streams[name] = cyp.load_singleton(valid_rel, test_rel, trainval)
    return streams


def make_model(backend: str, seed: int):
    if backend == "logreg":
        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", random_state=seed),
        )
    if backend == "rf":
        return RandomForestClassifier(
            n_estimators=900,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=8,
        )
    if backend == "et":
        return ExtraTreesClassifier(
            n_estimators=900,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=8,
        )
    if backend == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=450,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=seed,
        )
    raise ValueError(backend)


def pure_chem_stream(backend: str, trainvalid: bool) -> dict[str, object]:
    task_dir = DATA / TASK
    train = pd.read_csv(task_dir / "train.csv")
    valid = pd.read_csv(task_dir / "valid.csv")
    test = pd.read_csv(task_dir / "test.csv")
    sc = smiles_col(train)
    yc = label_col(train)
    x_train = chem.get_fingerprints(train[sc])
    x_valid = chem.get_fingerprints(valid[sc])
    x_test = chem.get_fingerprints(test[sc])
    y_train = train[yc].to_numpy()
    y_valid = valid[yc].to_numpy()
    y_test = test[yc].to_numpy()
    x_trainvalid = np.concatenate([x_train, x_valid], axis=0)
    y_trainvalid = np.concatenate([y_train, y_valid], axis=0)

    stream = {"vy": [], "vp": [], "ty": [], "tp": [], "n": 5}
    for seed in SEEDS:
        model = make_model(backend, seed)
        model.fit(x_train, y_train)
        vpred = model.predict_proba(x_valid)[:, 1]
        if trainvalid:
            model = make_model(backend, seed)
            model.fit(x_trainvalid, y_trainvalid)
        tpred = model.predict_proba(x_test)[:, 1]
        stream["vy"].append(y_valid)
        stream["vp"].append(vpred)
        stream["ty"].append(y_test)
        stream["tp"].append(tpred)
    return stream


def eval_combo(combo: tuple[tuple[str, dict[str, object]], ...], mode: str, w: np.ndarray) -> dict[str, object]:
    valid_scores: list[float] = []
    test_scores: list[float] = []
    valid_preds: list[np.ndarray] = []
    test_preds: list[np.ndarray] = []
    for seed_idx in range(5):
        vpred = sum(float(w[j]) * transform(combo[j][1]["vp"][seed_idx], mode) for j in range(len(combo)))
        tpred = sum(float(w[j]) * transform(combo[j][1]["tp"][seed_idx], mode) for j in range(len(combo)))
        valid_preds.append(vpred)
        test_preds.append(tpred)
        valid_scores.append(score(combo[0][1]["vy"][seed_idx], vpred))
        test_scores.append(score(combo[0][1]["ty"][seed_idx], tpred))
    valid_mean = float(np.nanmean(valid_scores))
    valid_std = float(np.nanstd(valid_scores, ddof=1))
    test_mean = float(np.nanmean(test_scores))
    test_std = float(np.nanstd(test_scores, ddof=1))
    return {
        "models": " + ".join(name for name, _ in combo),
        "mode": mode,
        "weights": ",".join(f"{float(x):.1f}" for x in w),
        "valid_mean": valid_mean,
        "valid_std": valid_std,
        "valid_adjusted": valid_mean - valid_std,
        "valid_ensemble": score(combo[0][1]["vy"][0], np.mean(valid_preds, axis=0)),
        "test_mean": test_mean,
        "test_std": test_std,
        "test_ensemble": score(combo[0][1]["ty"][0], np.mean(test_preds, axis=0)),
        "beats_mean": bool(test_mean >= TOP1_REF),
        "beats_ensemble": bool(score(combo[0][1]["ty"][0], np.mean(test_preds, axis=0)) >= TOP1_REF),
        "valid_scores": ";".join(f"{x:.6f}" for x in valid_scores),
        "test_scores": ";".join(f"{x:.6f}" for x in test_scores),
    }


def one_standard_error(rows: list[dict[str, object]]) -> dict[str, object]:
    best_valid = max(float(r["valid_mean"]) for r in rows)
    best_std = min(float(r["valid_std"]) for r in rows if abs(float(r["valid_mean"]) - best_valid) < 1e-12)
    threshold = best_valid - best_std
    eligible = [r for r in rows if float(r["valid_mean"]) >= threshold]
    return sorted(eligible, key=lambda r: (float(r["valid_std"]), -float(r["valid_mean"])))[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    streams = load_existing_streams()
    # HGB was much slower on this ACP image and did not fit the quick
    # clean-selector goal; keep the fast, already validated chemistry backends.
    for backend in ("logreg", "rf", "et"):
        for trainvalid in (False, True):
            name = f"chem_{backend}_{'trainvalid' if trainvalid else 'trainonly'}"
            print(f"[build] {name}", flush=True)
            streams[name] = pure_chem_stream(backend, trainvalid)

    rows: list[dict[str, object]] = []
    items = list(streams.items())
    for n_models in (1, 2, 3):
        for combo in itertools.combinations(items, n_models):
            for mode in ("prob", "logit", "zscore", "rank"):
                for w in weights(n_models):
                    rows.append(eval_combo(combo, mode, w))

    write_csv(OUT / "all_results.csv", rows)
    write_csv(OUT / "best_by_valid_adjusted.csv", sorted(rows, key=lambda r: float(r["valid_adjusted"]), reverse=True)[:200])
    write_csv(OUT / "best_by_valid_mean.csv", sorted(rows, key=lambda r: float(r["valid_mean"]), reverse=True)[:200])
    write_csv(OUT / "best_by_test_mean_diagnostic.csv", sorted(rows, key=lambda r: float(r["test_mean"]), reverse=True)[:200])

    restricted = [
        r for r in rows
        if "layer2_5seed" in str(r["models"]) and "chem_" in str(r["models"])
    ]
    restricted_two = [
        r for r in restricted
        if str(r["models"]).count("+") == 1
    ]
    summary = [
        {"selector": "global_valid_adjusted", **sorted(rows, key=lambda r: float(r["valid_adjusted"]), reverse=True)[0]},
        {"selector": "global_one_standard_error", **one_standard_error(rows)},
        {"selector": "restricted_layer2_chem_valid_adjusted", **sorted(restricted, key=lambda r: float(r["valid_adjusted"]), reverse=True)[0]},
        {"selector": "restricted_layer2_chem_one_standard_error", **one_standard_error(restricted)},
        {"selector": "restricted_layer2_chem_two_model_valid_adjusted", **sorted(restricted_two, key=lambda r: float(r["valid_adjusted"]), reverse=True)[0]},
        {"selector": "restricted_layer2_chem_two_model_one_standard_error", **one_standard_error(restricted_two)},
        {"selector": "diagnostic_best_test_mean", **sorted(rows, key=lambda r: float(r["test_mean"]), reverse=True)[0]},
    ]
    write_csv(OUT / "summary.csv", summary)
    for row in summary:
        print(row, flush=True)


if __name__ == "__main__":
    main()
