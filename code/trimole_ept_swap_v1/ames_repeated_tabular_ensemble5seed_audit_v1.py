from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import ames_repeated_tabular_5seed_audit_v1 as seed_audit
import descriptor_sidecar_official_v1 as base
import official_sidecar_nested_refit_v1 as nested


REPO = Path("/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1")
DATA_ROOT = REPO / "data" / "data_benchmark_official_v1"
TASK = "ames"
TOP1_REF = 0.871


DEFAULT_GROUPS = [
    [11, 22, 33, 44, 55],
    [111, 122, 133, 144, 155],
    [211, 222, 233, 244, 255],
    [311, 322, 333, 344, 355],
    [411, 422, 433, 444, 455],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument(
        "--out-root",
        default=str(REPO / "results_strict" / "ames_repeated_tabular_ensemble5seed_audit_v1"),
    )
    parser.add_argument(
        "--seed-cache-root",
        default=str(REPO / "results_strict" / "ames_repeated_tabular_5seed_audit_v1"),
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--xgb-estimators", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_data(data_root: Path):
    task_dir = data_root / TASK
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
    return X_tv, y_tv, X_te, y_te, smiles_tv


def load_or_run_seed(
    seed: int,
    args: argparse.Namespace,
    X_tv: np.ndarray,
    y_tv: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    smiles_tv: list[str],
    out_root: Path,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    cache_root = Path(args.seed_cache_root)
    cache_result = cache_root / f"seed_{seed}" / "result.json"
    if cache_result.exists() and not args.force:
        result = json.loads(cache_result.read_text())
        train_pred_file = Path(result["trainval_pred_file"])
        test_pred_file = Path(result["test_pred_file"])
    else:
        result = seed_audit.run_seed(
            seed,
            X_tv,
            y_tv,
            X_te,
            y_te,
            smiles_tv,
            args.folds,
            args.xgb_estimators,
            out_root / "inner_seed_models",
            args.force,
        )
        train_pred_file = Path(result["trainval_pred_file"])
        test_pred_file = Path(result["test_pred_file"])

    train_pred = pd.read_csv(train_pred_file).sort_values("sample_idx")["y_prob"].to_numpy(dtype=np.float64)
    test_pred = pd.read_csv(test_pred_file).sort_values("sample_idx")["y_prob"].to_numpy(dtype=np.float64)
    return result, train_pred, test_pred


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    X_tv, y_tv, X_te, y_te, smiles_tv = load_data(Path(args.data_root))

    run_rows: list[dict[str, object]] = []
    all_run_test_preds: list[np.ndarray] = []
    for run_idx, seeds in enumerate(DEFAULT_GROUPS, 1):
        print(f"[formal run {run_idx}] seeds={','.join(map(str, seeds))}", flush=True)
        seed_results: list[dict[str, object]] = []
        train_preds: list[np.ndarray] = []
        test_preds: list[np.ndarray] = []
        for seed in seeds:
            result, train_pred, test_pred = load_or_run_seed(
                seed,
                args,
                X_tv,
                y_tv,
                X_te,
                y_te,
                smiles_tv,
                out_root,
            )
            seed_results.append(result)
            train_preds.append(train_pred)
            test_preds.append(test_pred)

        train_ensemble = np.mean(train_preds, axis=0)
        test_ensemble = np.mean(test_preds, axis=0)
        all_run_test_preds.append(test_ensemble)
        run_dir = out_root / f"formal_seed_{run_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "sample_idx": np.arange(len(y_tv), dtype=int),
                "y_true": y_tv,
                "y_prob": train_ensemble.astype(np.float32),
            }
        ).to_csv(run_dir / "trainval_predictions.csv", index=False)
        pd.DataFrame(
            {
                "sample_idx": np.arange(len(y_te), dtype=int),
                "y_true": y_te,
                "y_prob": test_ensemble.astype(np.float32),
            }
        ).to_csv(run_dir / "test_predictions.csv", index=False)
        run_rows.append(
            {
                "formal_seed": run_idx,
                "inner_seeds": ",".join(str(seed) for seed in seeds),
                "inner_seed_scores": ",".join(f"{float(row['test_score']):.12f}" for row in seed_results),
                "inner_seed_mean": float(np.mean([float(row["test_score"]) for row in seed_results])),
                "inner_seed_std": float(np.std([float(row["test_score"]) for row in seed_results], ddof=0)),
                "cv_oof_score": float(roc_auc_score(y_tv, train_ensemble)),
                "test_score": float(roc_auc_score(y_te, test_ensemble)),
                "beats_top1": bool(float(roc_auc_score(y_te, test_ensemble)) >= TOP1_REF),
            }
        )

    scores = np.array([float(row["test_score"]) for row in run_rows], dtype=np.float64)
    cross_run_ensemble = np.mean(all_run_test_preds, axis=0)
    summary = {
        "task": TASK,
        "metric": "AUROC",
        "top1_ref": TOP1_REF,
        "formal_run_scores": ",".join(f"{score:.12f}" for score in scores),
        "test_mean": float(np.mean(scores)),
        "test_std": float(np.std(scores, ddof=0)),
        "test_ensemble_score": float(roc_auc_score(y_te, cross_run_ensemble)),
        "beats_top1_mean": bool(float(np.mean(scores)) >= TOP1_REF),
        "beats_top1_ensemble": bool(float(roc_auc_score(y_te, cross_run_ensemble)) >= TOP1_REF),
        "backend": f"xgboost_{args.xgb_estimators}",
        "folds": args.folds,
        "internal_repeats_per_formal_seed": len(DEFAULT_GROUPS[0]),
        "selection_note": "Each formal run is a repeated scaffold fold-bagged tabular ensemble.",
    }

    with (out_root / "formal_seed_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(run_rows)
    pd.DataFrame(
        {
            "sample_idx": np.arange(len(y_te), dtype=int),
            "y_true": y_te,
            "y_prob": cross_run_ensemble.astype(np.float32),
        }
    ).to_csv(out_root / "cross_run_ensemble_test_predictions.csv", index=False)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(out_root / "summary.csv")


if __name__ == "__main__":
    main()
