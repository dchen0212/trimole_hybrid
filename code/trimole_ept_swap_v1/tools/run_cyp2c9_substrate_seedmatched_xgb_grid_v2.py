from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier


DEFAULT_REPO = Path("/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1")
TASK = "cyp2c9_substrate_carbonmangels"
TOP1_REF = 0.474
SEEDS = [1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=str(DEFAULT_REPO))
    p.add_argument("--data-root", default="")
    p.add_argument("--out-root", default="")
    p.add_argument("--n-jobs", type=int, default=12)
    p.add_argument("--exclude-maplight", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def add_import_paths(repo: Path) -> None:
    for path in (repo, repo / "results_strict", repo / "tools"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


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


def load_frames(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    task_dir = data_root / TASK
    train = pd.read_csv(task_dir / "train.csv")
    valid = pd.read_csv(task_dir / "valid.csv")
    test = pd.read_csv(task_dir / "test.csv")
    return train, valid, test, label_col(train)


def split_dense_concat(arr: np.ndarray, n_train: int, n_valid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return arr[:n_train], arr[n_train : n_train + n_valid], arr[n_train + n_valid :]


def load_dense(data_root: Path, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    task_dir = data_root / TASK
    n_train, n_valid, n_test = len(train), len(valid), len(test)
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in ("chemberta", "kpgt", "unimol"):
        path = task_dir / "embeddings" / f"{name}.npy"
        if path.exists():
            arr = np.load(path).astype(np.float32)
            if len(arr) == n_train + n_valid + n_test:
                out[name] = split_dense_concat(arr, n_train, n_valid)
    for name, root, suffix in (
        ("ept", "embeddings_ept", "ept"),
        ("maplight", "embeddings_maplight", "maplight"),
    ):
        paths = (
            task_dir / root / f"train_{suffix}.npy",
            task_dir / root / f"valid_{suffix}.npy",
            task_dir / root / f"test_{suffix}.npy",
        )
        if all(path.exists() for path in paths):
            out[name] = tuple(np.load(path).astype(np.float32) for path in paths)  # type: ignore[assignment]
    return out


def hstack(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([part[0] for part in parts], axis=1),
        np.concatenate([part[1] for part in parts], axis=1),
        np.concatenate([part[2] for part in parts], axis=1),
    )


def feature_sets(repo: Path, data_root: Path, exclude_maplight: bool) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    add_import_paths(repo)
    import descriptor_sidecar_official_v1 as sidecar  # noqa: PLC0415

    train, valid, test, _ = load_frames(data_root)
    sc = smiles_col(train)
    smiles = pd.concat([train[sc], valid[sc], test[sc]], axis=0).reset_index(drop=True)
    fp_all = sidecar.get_fingerprints(smiles)
    fp = split_dense_concat(fp_all, len(train), len(valid))
    dense = load_dense(data_root, train, valid, test)
    if exclude_maplight:
        dense.pop("maplight", None)
    sets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {"fp": fp}
    for name in ("kpgt", "chemberta", "ept", "maplight", "unimol"):
        if name in dense:
            sets[f"fp_{name}"] = hstack([fp, dense[name]])
    combo = [name for name in ("chemberta", "kpgt", "ept") if name in dense]
    if combo:
        sets["fp_chemberta_kpgt_ept"] = hstack([fp] + [dense[name] for name in combo])
    combo = [name for name in ("chemberta", "kpgt", "ept", "maplight") if name in dense]
    if combo:
        sets["fp_deep_all"] = hstack([fp] + [dense[name] for name in combo])
    return sets


def xgb_grid(y_train: np.ndarray) -> list[dict[str, float | int]]:
    pos = max(float(np.sum(y_train == 1)), 1.0)
    neg = max(float(np.sum(y_train == 0)), 1.0)
    ratio = neg / pos
    return [
        {"max_depth": 2, "learning_rate": 0.025, "min_child_weight": 1, "scale_pos_weight": ratio},
        {"max_depth": 2, "learning_rate": 0.040, "min_child_weight": 2, "scale_pos_weight": ratio},
        {"max_depth": 3, "learning_rate": 0.025, "min_child_weight": 1, "scale_pos_weight": ratio},
        {"max_depth": 3, "learning_rate": 0.040, "min_child_weight": 3, "scale_pos_weight": ratio * 1.25},
        {"max_depth": 4, "learning_rate": 0.025, "min_child_weight": 5, "scale_pos_weight": math.sqrt(ratio)},
        {"max_depth": 4, "learning_rate": 0.040, "min_child_weight": 1, "scale_pos_weight": 1.0},
    ]


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    cfg: dict[str, float | int],
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = XGBClassifier(
        n_estimators=1400,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.0,
        reg_lambda=1.0,
        n_jobs=n_jobs,
        early_stopping_rounds=80,
        **cfg,
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    return model.predict_proba(x_valid)[:, 1], model.predict_proba(x_test)[:, 1]


def write_pred(path: Path, y_true: np.ndarray, pred: np.ndarray) -> None:
    pd.DataFrame({"sample_idx": np.arange(len(y_true)), "y_true": y_true, "y_prob": pred}).to_csv(path, index=False)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo = Path(args.repo)
    data_root = Path(args.data_root) if args.data_root else repo / "data" / "data_benchmark_official_v1"
    out_root = Path(args.out_root) if args.out_root else repo / "results_strict" / "cyp2c9_substrate_seedmatched_xgb_grid_v2"
    out_root.mkdir(parents=True, exist_ok=True)

    train, valid, test, yc = load_frames(data_root)
    y_train = train[yc].to_numpy(dtype=np.float64)
    y_valid = valid[yc].to_numpy(dtype=np.float64)
    y_test = test[yc].to_numpy(dtype=np.float64)
    feats = feature_sets(repo, data_root, exclude_maplight=args.exclude_maplight)
    grid = xgb_grid(y_train)

    rows: list[dict[str, object]] = []
    for feat_name, (x_train, x_valid, x_test) in feats.items():
        for cfg_idx, cfg in enumerate(grid):
            tag = f"{feat_name}__cfg{cfg_idx:02d}"
            tag_dir = out_root / tag
            tag_dir.mkdir(parents=True, exist_ok=True)
            valid_scores: list[float] = []
            test_scores: list[float] = []
            valid_preds: list[np.ndarray] = []
            test_preds: list[np.ndarray] = []
            for seed in SEEDS:
                seed_dir = tag_dir / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                valid_path = seed_dir / "valid_predictions.csv"
                test_path = seed_dir / "test_predictions.csv"
                if valid_path.exists() and test_path.exists() and not args.force:
                    vp = pd.read_csv(valid_path)["y_prob"].to_numpy(dtype=np.float64)
                    tp = pd.read_csv(test_path)["y_prob"].to_numpy(dtype=np.float64)
                else:
                    vp, tp = fit_predict(x_train, y_train, x_valid, y_valid, x_test, seed, cfg, args.n_jobs)
                    write_pred(valid_path, y_valid, vp)
                    write_pred(test_path, y_test, tp)
                valid_preds.append(vp)
                test_preds.append(tp)
                valid_scores.append(float(average_precision_score(y_valid, vp)))
                test_scores.append(float(average_precision_score(y_test, tp)))
            valid_arr = np.asarray(valid_scores, dtype=float)
            test_arr = np.asarray(test_scores, dtype=float)
            valid_ensemble = float(average_precision_score(y_valid, np.mean(valid_preds, axis=0)))
            test_ensemble = float(average_precision_score(y_test, np.mean(test_preds, axis=0)))
            row = {
                "task": TASK,
                "feature_set": feat_name,
                "cfg_idx": cfg_idx,
                "tag": tag,
                **cfg,
                "valid_mean": float(np.mean(valid_arr)),
                "valid_std": float(np.std(valid_arr, ddof=1)),
                "valid_adjusted": float(np.mean(valid_arr) - np.std(valid_arr, ddof=1)),
                "valid_ensemble": valid_ensemble,
                "test_mean": float(np.mean(test_arr)),
                "test_std": float(np.std(test_arr, ddof=1)),
                "test_ensemble": test_ensemble,
                "top1_ref": TOP1_REF,
                "beats_mean": bool(float(np.mean(test_arr)) >= TOP1_REF),
                "beats_ensemble": bool(test_ensemble >= TOP1_REF),
                "valid_scores": ";".join(f"{x:.6f}" for x in valid_scores),
                "test_scores": ";".join(f"{x:.6f}" for x in test_scores),
            }
            rows.append(row)
            print(row, flush=True)

    write_csv(out_root / "all_results.csv", rows)
    best_valid_adj = sorted(rows, key=lambda r: float(r["valid_adjusted"]), reverse=True)
    best_valid_mean = sorted(rows, key=lambda r: float(r["valid_mean"]), reverse=True)
    best_test_mean = sorted(rows, key=lambda r: float(r["test_mean"]), reverse=True)
    write_csv(out_root / "best_by_valid_adjusted.csv", best_valid_adj[:50])
    write_csv(out_root / "best_by_valid_mean.csv", best_valid_mean[:50])
    write_csv(out_root / "best_by_test_mean_diagnostic.csv", best_test_mean[:50])
    summary = [
        {"selector": "valid_adjusted", **best_valid_adj[0]},
        {"selector": "valid_mean", **best_valid_mean[0]},
        {"selector": "diagnostic_best_test_mean", **best_test_mean[0]},
    ]
    write_csv(out_root / "summary.csv", summary)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
