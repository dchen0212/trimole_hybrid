from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

import descriptor_sidecar_official_v1 as base


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
DATA_ROOT = REPO / "data" / "data_benchmark_official_v1"
MASTER = REPO / "results_strict" / "ept_family_routing_master_v1" / "ept_family_routing_master_v1_metric_cv_selected_v2.csv"
OUT_ROOT = REPO / "results_strict" / "descriptor_sidecar_official_v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", type=str, default=str(MASTER))
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    p.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    p.add_argument("--tasks", nargs="*", default=[])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_prediction_column(path_str: str | None, expected_len: int) -> np.ndarray | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "sample_idx" in df.columns:
        df = df.sort_values("sample_idx").reset_index(drop=True)
    for col in ("y_prob", "y_pred", "prediction", "pred"):
        if col in df.columns:
            arr = df[col].to_numpy(dtype=np.float32).reshape(-1, 1)
            if arr.shape[0] == expected_len:
                return arr
    return None


def build_features(task_dir: Path, candidate: str, entry: dict[str, str]):
    train_df = pd.read_csv(task_dir / "train.csv")
    valid_df = pd.read_csv(task_dir / "valid.csv")
    test_df = pd.read_csv(task_dir / "test.csv")
    s_col = base.find_smiles_col(train_df)
    y_col = base.find_label_col(train_df)

    n_tr, n_va, n_te = len(train_df), len(valid_df), len(test_df)
    emb_tr, emb_va, emb_te = base.build_embedding_features(task_dir, candidate, n_tr, n_va, n_te)

    fp_tr = base.get_fingerprints(train_df[s_col])
    fp_va = base.get_fingerprints(valid_df[s_col])
    fp_te = base.get_fingerprints(test_df[s_col])

    X_tr_parts = [emb_tr.astype(np.float32), fp_tr]
    X_va_parts = [emb_va.astype(np.float32), fp_va]
    X_te_parts = [emb_te.astype(np.float32), fp_te]
    feature_type = "winner_embedding_plus_rdkit_fp"

    tr_pred = load_prediction_column(entry.get("metric_loss_train_pred_file"), n_tr)
    va_pred = load_prediction_column(entry.get("metric_loss_valid_pred_file"), n_va)
    te_pred = load_prediction_column(entry.get("metric_loss_test_pred_file"), n_te)
    if tr_pred is not None and va_pred is not None and te_pred is not None:
        X_tr_parts.append(tr_pred)
        X_va_parts.append(va_pred)
        X_te_parts.append(te_pred)
        feature_type = "winner_embedding_plus_rdkit_fp_plus_base_pred"

    X_tr = base.sanitize_features(np.concatenate(X_tr_parts, axis=1))
    X_va = base.sanitize_features(np.concatenate(X_va_parts, axis=1))
    X_te = base.sanitize_features(np.concatenate(X_te_parts, axis=1))
    y_tr = train_df[y_col].to_numpy()
    y_va = valid_df[y_col].to_numpy()
    y_te = test_df[y_col].to_numpy()
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te), feature_type


def iter_plan(master_path: Path, tasks: list[str], limit: int):
    with master_path.open() as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if str(r.get("selected", "False")).lower() == "true"]
    if tasks:
        wanted = set(tasks)
        rows = [r for r in rows if r["task"] in wanted]
    if limit > 0:
        rows = rows[:limit]
    return rows


def run_one(entry: dict[str, str], data_root: Path, out_root: Path, force: bool):
    task = entry["task"]
    candidate = entry["candidate"]
    task_dir = data_root / task
    out_dir = out_root / f"{task}__{candidate}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "result.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text())

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), feature_type = build_features(task_dir, candidate, entry)
    task_type = base.infer_task_type(y_tr)
    model, backend = base.fit_model(X_tr, y_tr, X_va, y_va, task_type, entry["tdc_metric"])
    p_va = base.predict_model(model, X_va, task_type)
    p_te = base.predict_model(model, X_te, task_type)

    valid_score = base.score_metric(entry["tdc_metric"], y_va, p_va)
    test_score = base.score_metric(entry["tdc_metric"], y_te, p_te)

    base.write_predictions(out_dir / "valid_predictions.csv", y_va, p_va, task_type)
    base.write_predictions(out_dir / "test_predictions.csv", y_te, p_te, task_type)

    incumbent_valid = float(entry.get("valid_tdc_score_mean") or entry.get("valid_tdc_score"))
    incumbent_test = float(entry.get("test_tdc_score_mean") or entry.get("test_tdc_score"))
    result = {
        "task": task,
        "candidate": candidate,
        "head": entry.get("head", ""),
        "tdc_metric": entry["tdc_metric"],
        "metric_direction": entry["metric_direction"],
        "backend": backend,
        "feature_type": feature_type,
        "valid_tdc_score": float(valid_score),
        "test_tdc_score": float(test_score),
        "incumbent_valid_tdc_score": incumbent_valid,
        "incumbent_test_tdc_score": incumbent_test,
        "improved_valid": base.direction_better(float(valid_score), incumbent_valid, entry["metric_direction"]),
        "improved_test": base.direction_better(float(test_score), incumbent_test, entry["metric_direction"]),
        "tdc_top1_ref": float(entry["tdc_top1_ref"]),
        "gap_vs_top1_ref": abs(float(test_score) - float(entry["tdc_top1_ref"])),
        "valid_pred_file": str(out_dir / "valid_predictions.csv"),
        "test_pred_file": str(out_dir / "test_predictions.csv"),
        "out_dir": str(out_dir),
    }
    summary_path.write_text(json.dumps(result, indent=2))
    return result


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = iter_plan(Path(args.master), args.tasks, args.limit)
    results = []
    total = len(rows)
    for idx, entry in enumerate(rows, start=1):
        print(f"[{idx}/{total}] {entry['task']}::{entry['candidate']}", flush=True)
        try:
            results.append(run_one(entry, data_root, out_root, args.force))
        except Exception as exc:
            results.append(
                {
                    "task": entry["task"],
                    "candidate": entry["candidate"],
                    "status": "error",
                    "error": str(exc),
                }
            )

    fieldnames = sorted({k for row in results for k in row})
    summary = out_root / "summary.csv"
    with summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    (out_root / "meta.json").write_text(
        json.dumps(
            {
                "master": str(args.master),
                "tasks": [r["task"] for r in rows],
                "count": len(rows),
            },
            indent=2,
        )
    )
    print(summary)


if __name__ == "__main__":
    main()
