from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path("/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1")
DATA_ROOT = REPO / "data" / "data_benchmark_official_v1"
MASTER = REPO / "results_strict" / "ept_family_routing_master_v1" / "ept_family_routing_master_v1_patched_v3_5seed.csv"
OUT_ROOT = REPO / "results_strict" / "trainval_scaffold_foldbag_endpoint_v1"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import descriptor_sidecar_official_v1 as sidecar_base
import official_metric_loss_push_all22_v1 as metric_loss
import official_sidecar_nested_refit_v1 as nested
from trimole.training.trainer import fit_on_task


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", default=str(MASTER))
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--tasks", nargs="*", default=["clearance_hepatocyte_az", "cyp2c9_substrate_carbonmangels"])
    p.add_argument("--group-seeds", nargs="*", type=int, default=[20260429, 20260430, 20260431, 20260432, 20260433])
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-materialized", action="store_true")
    return p.parse_args()


def read_master(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["task"]: row for row in rows if str(row.get("selected", "False")).lower() == "true"}


def load_concat_embeddings(task_dir: Path) -> dict[str, np.ndarray]:
    emb_dir = task_dir / "embeddings"
    out: dict[str, np.ndarray] = {}
    for name in ("chemberta", "unimol", "kpgt"):
        path = emb_dir / f"{name}.npy"
        arr = np.load(path).astype(np.float32)
        if np.isnan(arr).any() or np.isinf(arr).any():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        out[name] = arr
    return out


def pred_col(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "sample_idx" in df.columns:
        df = df.sort_values("sample_idx").reset_index(drop=True)
    for col in ("y_prob", "y_pred", "prediction", "pred"):
        if col in df.columns:
            return df[col].to_numpy(dtype=np.float32)
    raise KeyError(f"prediction column not found in {path}")


def label_col(df: pd.DataFrame) -> str:
    for col in ("label", "Y", "y"):
        if col in df.columns:
            return col
    raise KeyError("label column not found")


def materialize_fold_task(
    source_task_dir: Path,
    fold_task_dir: Path,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    train_orig_idx: np.ndarray,
    valid_orig_idx: np.ndarray,
    test_orig_idx: np.ndarray,
) -> None:
    if fold_task_dir.exists():
        shutil.rmtree(fold_task_dir)
    (fold_task_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    train_df.to_csv(fold_task_dir / "train.csv", index=False)
    valid_df.to_csv(fold_task_dir / "valid.csv", index=False)
    test_df.to_csv(fold_task_dir / "test.csv", index=False)
    order = np.concatenate([train_orig_idx, valid_orig_idx, test_orig_idx])
    for name, arr in embeddings.items():
        np.save(fold_task_dir / "embeddings" / f"{name}.npy", arr[order])
    # Keep optional metadata if present, but avoid copying stale embeddings.
    for extra in ("meta.json", "task.json"):
        src = source_task_dir / extra
        if src.exists():
            shutil.copy2(src, fold_task_dir / extra)


def group_endpoint(
    task: str,
    row: dict[str, str],
    data_root: Path,
    out_root: Path,
    group_seed: int,
    folds: int,
    force: bool,
    keep_materialized: bool,
) -> dict[str, object]:
    group_dir = out_root / task / f"group_seed_{group_seed}"
    result_path = group_dir / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text())

    task_dir = data_root / task
    train_df0 = pd.read_csv(task_dir / "train.csv")
    valid_df0 = pd.read_csv(task_dir / "valid.csv")
    test_df = pd.read_csv(task_dir / "test.csv")
    s_col = nested.get_smiles_col(train_df0)
    y_col = label_col(test_df)
    trainval_df = pd.concat([train_df0, valid_df0], ignore_index=True)
    smiles = trainval_df[s_col].astype(str).tolist()
    fold_indices = nested.build_scaffold_folds(smiles, folds, group_seed)

    n_tr = len(train_df0)
    n_va = len(valid_df0)
    n_te = len(test_df)
    pool_orig = np.arange(n_tr + n_va, dtype=int)
    test_orig = np.arange(n_tr + n_va, n_tr + n_va + n_te, dtype=int)
    embeddings = load_concat_embeddings(task_dir)

    test_preds: list[np.ndarray] = []
    valid_scores: list[float] = []
    fold_rows: list[dict[str, object]] = []
    fold_root = group_dir / "_materialized_folds"
    group_dir.mkdir(parents=True, exist_ok=True)

    metric = metric_loss.normalize_metric(row["tdc_metric"])
    for fold_idx, valid_idx in enumerate(fold_indices):
        train_mask = np.ones(len(trainval_df), dtype=bool)
        train_mask[valid_idx] = False
        train_idx = np.where(train_mask)[0]
        fold_task_dir = fold_root / f"fold_{fold_idx}"
        fold_out_dir = group_dir / f"fold_{fold_idx}_model"
        materialize_fold_task(
            source_task_dir=task_dir,
            fold_task_dir=fold_task_dir,
            train_df=trainval_df.iloc[train_idx].reset_index(drop=True),
            valid_df=trainval_df.iloc[valid_idx].reset_index(drop=True),
            test_df=test_df,
            embeddings=embeddings,
            train_orig_idx=pool_orig[train_idx],
            valid_orig_idx=pool_orig[valid_idx],
            test_orig_idx=test_orig,
        )
        cfg = metric_loss.build_config(task, row, group_seed * 100 + fold_idx, metric, metric_loss.profile_for_metric(metric)[1])
        meta = fit_on_task(task_dir=fold_task_dir, out_dir=fold_out_dir, config=cfg)
        valid_score = float(meta["best_valid_primary"])
        test_preds.append(pred_col(fold_out_dir / "test_predictions.csv"))
        fold_rows.append(
            {
                "group_seed": group_seed,
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "best_valid_primary": valid_score,
                "test_primary": float(meta["primary_metric"]),
                "best_epoch": int(meta.get("best_epoch", -1)),
                "test_pred_file": str(fold_out_dir / "test_predictions.csv"),
            }
        )
        valid_scores.append(valid_score)

    test_pred = np.mean(np.stack(test_preds, axis=0), axis=0)
    y_test = test_df[y_col].to_numpy()
    test_score = float(sidecar_base.score_metric(metric, y_test, test_pred))
    sidecar_base.write_predictions(group_dir / "test_endpoint_predictions.csv", y_test, test_pred, metric_loss.task_type_from_metric(metric))
    with (group_dir / "fold_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in fold_rows for k in r}))
        writer.writeheader()
        writer.writerows(fold_rows)
    if not keep_materialized and fold_root.exists():
        shutil.rmtree(fold_root)

    result = {
        "task": task,
        "candidate": row["candidate"],
        "head": row.get("head", ""),
        "tdc_metric": metric,
        "metric_direction": row["metric_direction"],
        "group_seed": group_seed,
        "folds": folds,
        "valid_fold_mean": float(np.mean(valid_scores)),
        "valid_fold_std": float(np.std(valid_scores, ddof=0)),
        "test_tdc_score": test_score,
        "tdc_top1_ref": float(row["tdc_top1_ref"]),
        "beats_top1": metric_loss.direction_better(test_score, float(row["tdc_top1_ref"]), row["metric_direction"]),
        "endpoint": "trainval_scaffold_foldbag_deep_endpoint",
        "test_pred_file": str(group_dir / "test_endpoint_predictions.csv"),
    }
    result_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    master = read_master(Path(args.master))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, object]] = []
    for task in args.tasks:
        if task not in master:
            all_results.append({"task": task, "status": "missing_from_master"})
            continue
        print(f"[task] {task}", flush=True)
        for group_seed in args.group_seeds:
            print(f"[group] {task} seed={group_seed}", flush=True)
            try:
                all_results.append(
                    group_endpoint(
                        task=task,
                        row=master[task],
                        data_root=Path(args.data_root),
                        out_root=out_root,
                        group_seed=group_seed,
                        folds=args.folds,
                        force=args.force,
                        keep_materialized=args.keep_materialized,
                    )
                )
            except Exception as exc:
                all_results.append({"task": task, "group_seed": group_seed, "status": "error", "error": str(exc)})
    fields = sorted({k for row in all_results for k in row})
    with (out_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)

    by_task: list[dict[str, object]] = []
    for task in args.tasks:
        rows = [r for r in all_results if r.get("task") == task and "test_tdc_score" in r]
        if not rows:
            continue
        direction = str(rows[0]["metric_direction"])
        scores = np.array([float(r["test_tdc_score"]) for r in rows], dtype=float)
        top1 = float(rows[0]["tdc_top1_ref"])
        mean = float(np.mean(scores))
        std = float(np.std(scores, ddof=0))
        by_task.append(
            {
                "task": task,
                "n_groups": len(rows),
                "tdc_metric": rows[0]["tdc_metric"],
                "test_mean": mean,
                "test_std": std,
                "test_endpoint_ensemble": float("nan"),
                "top1_ref": top1,
                "beats_top1_mean": metric_loss.direction_better(mean, top1, direction),
                "group_scores": ";".join(f"{x:.12g}" for x in scores),
            }
        )
    if by_task:
        fields2 = sorted({k for row in by_task for k in row})
        with (out_root / "task_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields2)
            writer.writeheader()
            writer.writerows(by_task)
    print(out_root / "summary.csv", flush=True)


if __name__ == "__main__":
    main()
