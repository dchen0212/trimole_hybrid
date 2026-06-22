from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


DEFAULT_REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
TASKS = [
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
]
TOP1_REF = {
    "clearance_hepatocyte_az": 0.536,
    "clearance_microsome_az": 0.630,
}
SEEDS = [1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=str(DEFAULT_REPO))
    p.add_argument("--data-root", default="")
    p.add_argument("--out-root", default="")
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def add_import_paths(repo: Path) -> None:
    for path in (repo, repo / "tools", repo / "results_strict"):
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


def split_concat(arr: np.ndarray, n_train: int, n_valid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return arr[:n_train], arr[n_train : n_train + n_valid], arr[n_train + n_valid :]


def load_frames(data_root: Path, task: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    task_dir = data_root / task
    train = pd.read_csv(task_dir / "train.csv")
    valid = pd.read_csv(task_dir / "valid.csv")
    test = pd.read_csv(task_dir / "test.csv")
    return train, valid, test, label_col(train)


def load_dense(
    data_root: Path,
    task: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    task_dir = data_root / task
    n_train, n_valid, n_test = len(train), len(valid), len(test)
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in ("chemberta", "kpgt", "unimol"):
        path = task_dir / "embeddings" / f"{name}.npy"
        if path.exists():
            arr = np.load(path).astype(np.float32)
            if len(arr) == n_train + n_valid + n_test:
                out[name] = split_concat(arr, n_train, n_valid)
    for name, root, suffix in (("ept", "embeddings_ept", "ept"),):
        paths = (
            task_dir / root / f"train_{suffix}.npy",
            task_dir / root / f"valid_{suffix}.npy",
            task_dir / root / f"test_{suffix}.npy",
        )
        if all(path.exists() for path in paths):
            out[name] = tuple(np.load(path).astype(np.float32) for path in paths)  # type: ignore[assignment]
    return out


def add_task_onehot(x: np.ndarray, task_idx: int, n_tasks: int) -> np.ndarray:
    onehot = np.zeros((len(x), n_tasks), dtype=np.float32)
    onehot[:, task_idx] = 1.0
    return np.concatenate([x.astype(np.float32), onehot], axis=1)


def hstack(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts, axis=1).astype(np.float32)


def spearman(y: np.ndarray, pred: np.ndarray) -> float:
    yr = pd.Series(np.asarray(y, dtype=float)).rank(method="average").to_numpy()
    pr = pd.Series(np.asarray(pred, dtype=float)).rank(method="average").to_numpy()
    if np.std(yr) == 0 or np.std(pr) == 0:
        return float("nan")
    return float(np.corrcoef(yr, pr)[0, 1])


def build_task_features(
    repo: Path,
    data_root: Path,
) -> tuple[dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]], dict[str, dict[str, object]]]:
    add_import_paths(repo)
    import descriptor_sidecar_official_v1 as sidecar  # noqa: PLC0415

    features: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    meta: dict[str, dict[str, object]] = {}
    for task_idx, task in enumerate(TASKS):
        train, valid, test, yc = load_frames(data_root, task)
        sc = smiles_col(train)
        smiles = pd.concat([train[sc], valid[sc], test[sc]], axis=0).reset_index(drop=True)
        fp_all = sidecar.get_fingerprints(smiles)
        fp = split_concat(fp_all, len(train), len(valid))
        dense = load_dense(data_root, task, train, valid, test)
        task_features: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
            "fp": tuple(add_task_onehot(part, task_idx, len(TASKS)) for part in fp),  # type: ignore[assignment]
        }
        if "kpgt" in dense:
            task_features["fp_kpgt"] = tuple(
                add_task_onehot(hstack([fp[i], dense["kpgt"][i]]), task_idx, len(TASKS)) for i in range(3)
            )  # type: ignore[assignment]
        deep_names = [name for name in ("chemberta", "kpgt", "ept") if name in dense]
        if deep_names:
            task_features["fp_chemberta_kpgt_ept"] = tuple(
                add_task_onehot(hstack([fp[i]] + [dense[name][i] for name in deep_names]), task_idx, len(TASKS))
                for i in range(3)
            )  # type: ignore[assignment]
        y_train = train[yc].to_numpy(dtype=np.float64)
        y_valid = valid[yc].to_numpy(dtype=np.float64)
        y_test = test[yc].to_numpy(dtype=np.float64)
        mu = float(np.mean(y_train))
        sigma = float(np.std(y_train) + 1e-8)
        features[task] = task_features
        meta[task] = {
            "y_train": y_train,
            "y_valid": y_valid,
            "y_test": y_test,
            "z_train": (y_train - mu) / sigma,
            "z_valid": (y_valid - mu) / sigma,
            "z_test": (y_test - mu) / sigma,
            "label_mean": mu,
            "label_std": sigma,
        }
    return features, meta


def common_feature_sets(features: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]) -> list[str]:
    names = set.intersection(*(set(v) for v in features.values()))
    ordered = ["fp", "fp_kpgt", "fp_chemberta_kpgt_ept"]
    return [name for name in ordered if name in names]


def xgb_grid() -> list[dict[str, float | int]]:
    return [
        {"max_depth": 2, "learning_rate": 0.020, "min_child_weight": 1, "subsample": 0.85, "colsample_bytree": 0.75},
        {"max_depth": 2, "learning_rate": 0.035, "min_child_weight": 2, "subsample": 0.90, "colsample_bytree": 0.70},
        {"max_depth": 3, "learning_rate": 0.020, "min_child_weight": 1, "subsample": 0.80, "colsample_bytree": 0.80},
        {"max_depth": 3, "learning_rate": 0.035, "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.75},
    ]


def make_model(seed: int, cfg: dict[str, float | int], n_jobs: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=1400,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=seed,
        reg_alpha=0.0,
        reg_lambda=1.0,
        n_jobs=n_jobs,
        early_stopping_rounds=80,
        **cfg,
    )


def write_pred(path: Path, y_true: np.ndarray, pred: np.ndarray) -> None:
    pd.DataFrame({"sample_idx": np.arange(len(y_true)), "y_true": y_true, "y_pred": pred}).to_csv(path, index=False)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pooled_xy(
    features: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    meta: dict[str, dict[str, object]],
    feat_name: str,
    include_valid: bool,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for task in TASKS:
        x_train, x_valid, _ = features[task][feat_name]
        z_train = meta[task]["z_train"]
        z_valid = meta[task]["z_valid"]
        xs.append(x_train)
        ys.append(z_train)  # type: ignore[arg-type]
        if include_valid:
            xs.append(x_valid)
            ys.append(z_valid)  # type: ignore[arg-type]
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0).astype(np.float64)


def main() -> None:
    args = parse_args()
    repo = Path(args.repo)
    data_root = Path(args.data_root) if args.data_root else repo / "data" / "data_benchmark_official_v1"
    out_root = Path(args.out_root) if args.out_root else repo / "results_strict" / "clearance_pooled_family_xgb_v1"
    out_root.mkdir(parents=True, exist_ok=True)

    features, meta = build_task_features(repo, data_root)
    rows: list[dict[str, object]] = []
    for feat_name in common_feature_sets(features):
        x_pool_train, z_pool_train = pooled_xy(features, meta, feat_name, include_valid=False)
        x_pool_trainvalid, z_pool_trainvalid = pooled_xy(features, meta, feat_name, include_valid=True)
        for cfg_idx, cfg in enumerate(xgb_grid()):
            for target in TASKS:
                tag = f"{target}__{feat_name}__cfg{cfg_idx:02d}"
                tag_dir = out_root / tag
                tag_dir.mkdir(parents=True, exist_ok=True)
                x_valid = features[target][feat_name][1]
                x_test = features[target][feat_name][2]
                z_valid = meta[target]["z_valid"]  # type: ignore[assignment]
                y_valid = meta[target]["y_valid"]  # type: ignore[assignment]
                y_test = meta[target]["y_test"]  # type: ignore[assignment]
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
                        vp = pd.read_csv(valid_path)["y_pred"].to_numpy(dtype=np.float64)
                        tp = pd.read_csv(test_path)["y_pred"].to_numpy(dtype=np.float64)
                    else:
                        train_model = make_model(seed, cfg, args.n_jobs)
                        train_model.fit(x_pool_train, z_pool_train, eval_set=[(x_valid, z_valid)], verbose=False)
                        vp = train_model.predict(x_valid)
                        final_model = make_model(seed, cfg, args.n_jobs)
                        final_model.fit(x_pool_trainvalid, z_pool_trainvalid, eval_set=[(x_valid, z_valid)], verbose=False)
                        tp = final_model.predict(x_test)
                        write_pred(valid_path, y_valid, vp)
                        write_pred(test_path, y_test, tp)
                    valid_preds.append(vp)
                    test_preds.append(tp)
                    valid_scores.append(spearman(y_valid, vp))
                    test_scores.append(spearman(y_test, tp))
                valid_arr = np.asarray(valid_scores, dtype=float)
                test_arr = np.asarray(test_scores, dtype=float)
                row = {
                    "target_task": target,
                    "feature_set": feat_name,
                    "cfg_idx": cfg_idx,
                    "tag": tag,
                    **cfg,
                    "metric": "spearman",
                    "valid_mean": float(np.nanmean(valid_arr)),
                    "valid_std": float(np.nanstd(valid_arr, ddof=1)),
                    "valid_adjusted": float(np.nanmean(valid_arr) - np.nanstd(valid_arr, ddof=1)),
                    "valid_ensemble": spearman(y_valid, np.mean(valid_preds, axis=0)),
                    "test_mean": float(np.nanmean(test_arr)),
                    "test_std": float(np.nanstd(test_arr, ddof=1)),
                    "test_ensemble": spearman(y_test, np.mean(test_preds, axis=0)),
                    "top1_ref": float(TOP1_REF[target]),
                    "beats_mean": bool(float(np.nanmean(test_arr)) >= float(TOP1_REF[target])),
                    "beats_ensemble": bool(spearman(y_test, np.mean(test_preds, axis=0)) >= float(TOP1_REF[target])),
                    "valid_scores": ";".join(f"{x:.6f}" for x in valid_scores),
                    "test_scores": ";".join(f"{x:.6f}" for x in test_scores),
                }
                rows.append(row)
                print(row, flush=True)

    write_csv(out_root / "all_results.csv", rows)
    summary: list[dict[str, object]] = []
    for task in TASKS:
        subset = [r for r in rows if r["target_task"] == task]
        if not subset:
            continue
        for selector, key in (
            ("valid_adjusted", "valid_adjusted"),
            ("valid_mean", "valid_mean"),
            ("diagnostic_test_mean", "test_mean"),
            ("diagnostic_test_ensemble", "test_ensemble"),
        ):
            best = sorted(subset, key=lambda r: float(r[key]), reverse=True)[0]
            summary.append({"selector": selector, **best})
    write_csv(out_root / "summary.csv", summary)
    write_csv(out_root / "best_by_valid_adjusted.csv", sorted(rows, key=lambda r: float(r["valid_adjusted"]), reverse=True)[:100])
    write_csv(out_root / "best_by_test_mean_diagnostic.csv", sorted(rows, key=lambda r: float(r["test_mean"]), reverse=True)[:100])


if __name__ == "__main__":
    main()
