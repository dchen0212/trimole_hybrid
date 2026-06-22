from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import descriptor_sidecar_official_v1 as base
import official_sidecar_nested_refit_v1 as nested


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
DATA_ROOT = REPO / "data" / "data_benchmark_official_v1"
TASK = "ames"
TOP1_REF = 0.871


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument(
        "--out-root",
        default=str(REPO / "results_strict" / "ames_repeated_tabular_5seed_audit_v1"),
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--xgb-estimators", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def fit_xgb_classifier(X, y, seed: int, n_estimators: int):
    pos = float(np.sum(np.asarray(y) == 1))
    neg = float(len(y) - pos)
    scale_pos_weight = max(1.0, neg / max(pos, 1.0))
    model = base.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        random_state=seed,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X, y, verbose=False)
    return model


def write_predictions(path: Path, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "sample_idx": np.arange(len(y_true), dtype=int),
            "y_true": y_true,
            "y_prob": y_prob.astype(np.float32),
        }
    )
    df.to_csv(path, index=False)


def run_seed(
    seed: int,
    X_tv: np.ndarray,
    y_tv: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    smiles_tv: list[str],
    folds_n: int,
    n_estimators: int,
    out_root: Path,
    force: bool,
) -> dict[str, object]:
    out_dir = out_root / f"seed_{seed}"
    result_path = out_dir / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text())

    folds = nested.build_scaffold_folds(smiles_tv, folds_n, seed)
    oof = np.zeros(len(y_tv), dtype=np.float64)
    test_preds: list[np.ndarray] = []
    fold_rows: list[dict[str, object]] = []

    for fold_idx, valid_idx in enumerate(folds):
        train_mask = np.ones(len(y_tv), dtype=bool)
        train_mask[valid_idx] = False
        train_idx = np.where(train_mask)[0]

        model = fit_xgb_classifier(
            X_tv[train_idx],
            y_tv[train_idx],
            seed * 100 + fold_idx,
            n_estimators,
        )
        valid_pred = base.predict_model(model, X_tv[valid_idx], "classification").astype(np.float64)
        test_pred = base.predict_model(model, X_te, "classification").astype(np.float64)
        oof[valid_idx] = valid_pred
        test_preds.append(test_pred)
        fold_rows.append(
            {
                "seed": seed,
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
                "fold_oof_auc": float(roc_auc_score(y_tv[valid_idx], valid_pred)),
            }
        )

    test_mean_pred = np.mean(test_preds, axis=0)
    result = {
        "task": TASK,
        "seed": seed,
        "folds": folds_n,
        "backend": f"xgboost_{n_estimators}",
        "metric": "AUROC",
        "cv_oof_score": float(roc_auc_score(y_tv, oof)),
        "cv_fold_mean": float(np.mean([row["fold_oof_auc"] for row in fold_rows])),
        "cv_fold_std": float(np.std([row["fold_oof_auc"] for row in fold_rows], ddof=0)),
        "test_score": float(roc_auc_score(y_te, test_mean_pred)),
        "top1_ref": TOP1_REF,
        "beats_top1": bool(float(roc_auc_score(y_te, test_mean_pred)) >= TOP1_REF),
        "test_pred_file": str(out_dir / "test_predictions.csv"),
        "trainval_pred_file": str(out_dir / "trainval_predictions.csv"),
    }

    write_predictions(out_dir / "trainval_predictions.csv", y_tv, oof)
    write_predictions(out_dir / "test_predictions.csv", y_te, test_mean_pred)
    with (out_dir / "fold_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "fold", "n_train", "n_valid", "fold_oof_auc"])
        writer.writeheader()
        writer.writerows(fold_rows)
    result_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    task_dir = Path(args.data_root) / TASK
    train_df = pd.read_csv(task_dir / "train.csv")
    valid_df = pd.read_csv(task_dir / "valid.csv")
    test_df = pd.read_csv(task_dir / "test.csv")
    smiles_col = nested.get_smiles_col(train_df)
    label_col = base.find_label_col(train_df)

    X_tr = base.get_fingerprints(train_df[smiles_col])
    X_va = base.get_fingerprints(valid_df[smiles_col])
    X_te = base.get_fingerprints(test_df[smiles_col])
    y_tr = train_df[label_col].to_numpy()
    y_va = valid_df[label_col].to_numpy()
    y_te = test_df[label_col].to_numpy()
    X_tv = np.concatenate([X_tr, X_va], axis=0)
    y_tv = np.concatenate([y_tr, y_va], axis=0)
    smiles_tv = pd.concat([train_df[smiles_col], valid_df[smiles_col]], ignore_index=True).astype(str).tolist()

    results: list[dict[str, object]] = []
    test_seed_preds: list[np.ndarray] = []
    for seed in args.seeds:
        print(f"[seed {seed}]", flush=True)
        result = run_seed(
            seed,
            X_tv,
            y_tv,
            X_te,
            y_te,
            smiles_tv,
            args.folds,
            args.xgb_estimators,
            out_root,
            args.force,
        )
        results.append(result)
        pred = pd.read_csv(result["test_pred_file"]).sort_values("sample_idx")["y_prob"].to_numpy(dtype=np.float64)
        test_seed_preds.append(pred)

    scores = np.array([float(row["test_score"]) for row in results], dtype=np.float64)
    ensemble_pred = np.mean(test_seed_preds, axis=0)
    ensemble_score = float(roc_auc_score(y_te, ensemble_pred))
    summary = {
        "task": TASK,
        "metric": "AUROC",
        "top1_ref": TOP1_REF,
        "seeds": ",".join(str(seed) for seed in args.seeds),
        "seed_scores": ",".join(f"{score:.12f}" for score in scores),
        "test_mean": float(np.mean(scores)),
        "test_std": float(np.std(scores, ddof=0)),
        "test_ensemble_score": ensemble_score,
        "beats_top1_mean": bool(float(np.mean(scores)) >= TOP1_REF),
        "beats_top1_ensemble": bool(ensemble_score >= TOP1_REF),
        "backend": f"xgboost_{args.xgb_estimators}",
        "folds": args.folds,
    }

    with (out_root / "seed_results.csv").open("w", newline="") as f:
        fields = sorted({key for row in results for key in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    pd.DataFrame(
        {
            "sample_idx": np.arange(len(y_te), dtype=int),
            "y_true": y_te,
            "y_prob": ensemble_pred.astype(np.float32),
        }
    ).to_csv(out_root / "ensemble_test_predictions.csv", index=False)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(out_root / "summary.csv")


if __name__ == "__main__":
    main()
