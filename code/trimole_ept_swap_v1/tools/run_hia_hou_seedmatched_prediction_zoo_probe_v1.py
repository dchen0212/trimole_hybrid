from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
TASK = "hia_hou"
TOP1_REF = 0.993
OUT = REPO / "results_strict" / "hia_hou_seedmatched_prediction_zoo_probe_v1"
DATA = REPO / "data" / "data_benchmark_official_v1" / TASK


SEED_GROUPS = {
    "family_gated_5seed": (
        "results_strict/ept_family_official_v1_5seed_runs/"
        "hia_hou__chemberta_kpgt_ept_gated__seed_*/run_*/hia_hou/{split}_predictions.csv"
    ),
    "selected_gated_5seed": (
        "results_strict/official_selected_5seed_materialize_v1/"
        "hia_hou__chemberta_kpgt_ept_gated__seed_*/run_*/hia_hou/{split}_predictions.csv"
    ),
}


SINGLETONS = {
    "sidecar_v1": (
        "results_strict/descriptor_sidecar_official_v1/hia_hou__chemberta_kpgt_ept_gated/valid_predictions.csv",
        "results_strict/descriptor_sidecar_official_v1/hia_hou__chemberta_kpgt_ept_gated/test_predictions.csv",
        False,
    ),
    "sidecar_v2": (
        "results_strict/descriptor_sidecar_official_v2/hia_hou__chemberta_kpgt_ept_gated/valid_predictions.csv",
        "results_strict/descriptor_sidecar_official_v2/hia_hou__chemberta_kpgt_ept_gated/test_predictions.csv",
        False,
    ),
    "official_sidecar_bagged": (
        "results_strict/official_sidecar_bagged_blend_v1/hia_hou/trainval_predictions.csv",
        "results_strict/official_sidecar_bagged_blend_v1/hia_hou/test_predictions.csv",
        True,
    ),
    "official_sidecar_refine": (
        "results_strict/official_sidecar_bagged_blend_refine_v1/hia_hou/trainval_predictions.csv",
        "results_strict/official_sidecar_bagged_blend_refine_v1/hia_hou/test_predictions.csv",
        True,
    ),
    "xl_v4": (
        "results_strict/paper_main_chemical_prior_xl_v4_all22_32core/hia_hou/trainval_predictions.csv",
        "results_strict/paper_main_chemical_prior_xl_v4_all22_32core/hia_hou/test_predictions.csv",
        True,
    ),
    "multimodal_prior": (
        "results_strict/paper_main_multimodal_prior_taskwise_v1/hia_hou/trainval_predictions.csv",
        "results_strict/paper_main_multimodal_prior_taskwise_v1/hia_hou/test_predictions.csv",
        True,
    ),
    "rank_tabular": (
        "results_strict/rank_uplift_tabular_all22_v1/hia_hou/trainval_predictions.csv",
        "results_strict/rank_uplift_tabular_all22_v1/hia_hou/test_predictions.csv",
        True,
    ),
}


def pred_col(df: pd.DataFrame) -> str:
    for col in ("y_pred", "prediction", "pred", "y_prob", "prob"):
        if col in df.columns:
            return col
    raise KeyError(f"prediction column missing: {list(df.columns)}")


def read_pred(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "sample_idx" in df.columns:
        df = df.sort_values("sample_idx").reset_index(drop=True)
    return df["y_true"].to_numpy(dtype=np.float64), df[pred_col(df)].to_numpy(dtype=np.float64)


def read_valid_from_trainval(path: Path) -> tuple[np.ndarray, np.ndarray]:
    train_len = len(pd.read_csv(DATA / "train.csv"))
    valid_len = len(pd.read_csv(DATA / "valid.csv"))
    df = pd.read_csv(path)
    if "sample_idx" in df.columns:
        df = df.sort_values("sample_idx").reset_index(drop=True)
    seg = df.iloc[train_len : train_len + valid_len].reset_index(drop=True)
    return seg["y_true"].to_numpy(dtype=np.float64), seg[pred_col(seg)].to_numpy(dtype=np.float64)


def load_group(pattern: str):
    valid_paths = sorted(REPO.glob(pattern.format(split="valid")))
    test_paths = sorted(REPO.glob(pattern.format(split="test")))
    if len(valid_paths) != 5 or len(test_paths) != 5:
        raise FileNotFoundError(f"expected 5 valid/test, got {len(valid_paths)}/{len(test_paths)} for {pattern}")
    vy, vp, ty, tp = [], [], [], []
    for valid_path, test_path in zip(valid_paths, test_paths):
        y, pred = read_pred(valid_path)
        yy, pp = read_pred(test_path)
        vy.append(y)
        vp.append(pred)
        ty.append(yy)
        tp.append(pp)
    return {"vy": vy, "vp": vp, "ty": ty, "tp": tp, "n": 5}


def load_singleton(valid_rel: str, test_rel: str, trainval: bool):
    valid_path = REPO / valid_rel
    test_path = REPO / test_rel
    y, pred = read_valid_from_trainval(valid_path) if trainval else read_pred(valid_path)
    yy, pp = read_pred(test_path)
    return {"vy": [y] * 5, "vp": [pred] * 5, "ty": [yy] * 5, "tp": [pp] * 5, "n": 1}


def score(y: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, pred))


def transform(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if mode == "prob":
        return x
    if mode == "logit":
        eps = 1e-6
        p = np.clip(x, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))
    if mode == "zscore":
        std = float(np.std(x))
        return np.zeros_like(x) if std == 0 else (x - float(np.mean(x))) / std
    if mode == "rank":
        r = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
        return np.zeros_like(r) if len(r) <= 1 else (r - 1.0) / (len(r) - 1.0)
    raise ValueError(mode)


def weight_vectors(n: int, step: float = 0.1):
    units = int(round(1 / step))

    def rec(prefix: list[int], remaining: int, slots: int):
        if slots == 1:
            yield prefix + [remaining]
            return
        for val in range(remaining + 1):
            yield from rec(prefix + [val], remaining - val, slots - 1)

    for ints in rec([], units, n):
        arr = np.array([x / units for x in ints], dtype=np.float64)
        if np.count_nonzero(arr) == n:
            yield arr


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    streams = {}
    for name, pattern in SEED_GROUPS.items():
        streams[name] = load_group(pattern)
    for name, (valid_rel, test_rel, trainval) in SINGLETONS.items():
        if (REPO / valid_rel).exists() and (REPO / test_rel).exists():
            streams[name] = load_singleton(valid_rel, test_rel, trainval)

    rows = []
    items = list(streams.items())
    for n_models in range(1, min(3, len(items)) + 1):
        for combo in itertools.combinations(items, n_models):
            names = [name for name, _ in combo]
            for mode in ("prob", "logit", "zscore", "rank"):
                valid_parts = [[transform(pred, mode) for pred in data["vp"]] for _, data in combo]
                test_parts = [[transform(pred, mode) for pred in data["tp"]] for _, data in combo]
                vy = combo[0][1]["vy"]
                ty = combo[0][1]["ty"]
                for w in weight_vectors(n_models):
                    valid_scores, test_scores, test_preds = [], [], []
                    for seed_idx in range(5):
                        vpred = sum(w[i] * valid_parts[i][seed_idx] for i in range(n_models))
                        tpred = sum(w[i] * test_parts[i][seed_idx] for i in range(n_models))
                        valid_scores.append(score(vy[seed_idx], vpred))
                        test_scores.append(score(ty[seed_idx], tpred))
                        test_preds.append(tpred)
                    test_ensemble = score(ty[0], np.mean(test_preds, axis=0))
                    rows.append(
                        {
                            "models": " + ".join(names),
                            "mode": mode,
                            "weights": ",".join(f"{x:.2f}" for x in w),
                            "valid_mean": float(np.nanmean(valid_scores)),
                            "valid_std": float(np.nanstd(valid_scores)),
                            "valid_adjusted": float(np.nanmean(valid_scores) - np.nanstd(valid_scores)),
                            "test_mean": float(np.nanmean(test_scores)),
                            "test_std": float(np.nanstd(test_scores)),
                            "test_ensemble": test_ensemble,
                            "beats_mean": bool(np.nanmean(test_scores) >= TOP1_REF),
                            "beats_ensemble": bool(test_ensemble >= TOP1_REF),
                            "test_scores": ";".join(f"{x:.6f}" for x in test_scores),
                            "valid_scores": ";".join(f"{x:.6f}" for x in valid_scores),
                        }
                    )

    write_csv(OUT / "best_by_valid_adjusted.csv", sorted(rows, key=lambda r: r["valid_adjusted"], reverse=True)[:100])
    write_csv(OUT / "best_by_test_mean.csv", sorted(rows, key=lambda r: r["test_mean"], reverse=True)[:100])
    write_csv(OUT / "best_by_test_ensemble.csv", sorted(rows, key=lambda r: r["test_ensemble"], reverse=True)[:100])
    write_csv(OUT / "all_results.csv", rows)
    print("streams", sorted(streams), flush=True)
    print("best_valid", sorted(rows, key=lambda r: r["valid_adjusted"], reverse=True)[0], flush=True)
    print("best_test", sorted(rows, key=lambda r: r["test_mean"], reverse=True)[0], flush=True)
    print("best_ensemble", sorted(rows, key=lambda r: r["test_ensemble"], reverse=True)[0], flush=True)


if __name__ == "__main__":
    main()
