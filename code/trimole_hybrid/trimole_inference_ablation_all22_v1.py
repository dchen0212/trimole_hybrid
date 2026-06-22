from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader

THIS_FILE = Path(__file__).resolve()
REPO_FROM_FILE = THIS_FILE.parents[1]
if str(REPO_FROM_FILE) not in sys.path:
    sys.path.insert(0, str(REPO_FROM_FILE))

from trimole.models.model import MultiModalFusionMLP
from trimole.training.trainer import (
    MultiModalDataset,
    _load_concat_embeddings,
    _read_split,
    _slice,
    build_task_context_vector,
    expand_task_context,
    infer_task_type,
)


ROOT = Path("<PROJECT_ROOT>/trimole_hybrid")
MASTER = ROOT / "results_strict" / "tdc_aligned_22task_table_strict_v1" / "strict_master_writable.csv"
RUN_ROOT = ROOT / "results_strict" / "task_prior_v1_batch22_seed42" / "run_20260418_1521"
OUT_ROOT = ROOT / "results_strict" / "trimole_inference_ablation_all22_v1"

MODALITIES = [
    "chemberta",
    "unimol",
    "kpgt",
    "chemberta_unimol",
    "chemberta_kpgt",
    "unimol_kpgt",
    "all",
]

MODALITY_MAP = {
    "all": None,
    "chemberta": (True, False, False),
    "kpgt": (False, True, False),
    "unimol": (False, False, True),
    "chemberta_kpgt": (True, True, False),
    "unimol_kpgt": (False, True, True),
    "chemberta_unimol": (True, False, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=str, default=str(MASTER))
    parser.add_argument("--root", type=str, default=str(ROOT))
    parser.add_argument("--run-root", type=str, default=str(RUN_ROOT))
    parser.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--only-regression", action="store_true")
    return parser.parse_args()


def load_master_rows(master_path: Path) -> list[dict]:
    with master_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_task_bundle(task: str, root: Path, run_root: Path):
    task_dir = root / "data" / "data_new" / task
    meta_path = run_root / task / "meta.json"
    ckpt_path = run_root / task / "best_model.pth"
    if not meta_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"missing run artifacts for {task}")

    meta = json.loads(meta_path.read_text())
    train_csv = task_dir / "train.csv"
    valid_csv = task_dir / "valid.csv"
    test_csv = task_dir / "test.csv"

    n_tr, y_tr = _read_split(train_csv)
    n_va, y_va = _read_split(valid_csv)
    n_te, y_te = _read_split(test_csv)

    task_type = meta.get("task_type", "auto")
    if task_type == "auto":
        task_type = infer_task_type(np.asarray(y_tr))

    label_mean = meta.get("label_mean", None)
    label_std = meta.get("label_std", None)
    if task_type == "classification":
        y_tr = np.asarray(y_tr).astype(int)
        y_va = np.asarray(y_va).astype(int)
        y_te = np.asarray(y_te).astype(int)
    else:
        y_tr = np.asarray(y_tr).astype(np.float32)
        y_va = np.asarray(y_va).astype(np.float32)
        y_te = np.asarray(y_te).astype(np.float32)
        if label_mean is None or label_std is None:
            y_tr_finite = y_tr[np.isfinite(y_tr)]
            label_mean = float(np.mean(y_tr_finite)) if y_tr_finite.size else 0.0
            label_std = float(np.std(y_tr_finite)) + 1e-8 if y_tr_finite.size else 1.0
    emb_all_s, emb_all_3d, emb_all_g = _load_concat_embeddings(task_dir / "embeddings")
    off_tr = 0
    off_va = n_tr
    off_te = n_tr + n_va

    emb_tr_s = _slice(emb_all_s, off_tr, n_tr, "chemberta", task_dir / "embeddings")
    emb_tr_3d = _slice(emb_all_3d, off_tr, n_tr, "unimol", task_dir / "embeddings")
    emb_tr_g = _slice(emb_all_g, off_tr, n_tr, "kpgt", task_dir / "embeddings")
    emb_va_s = _slice(emb_all_s, off_va, n_va, "chemberta", task_dir / "embeddings")
    emb_va_3d = _slice(emb_all_3d, off_va, n_va, "unimol", task_dir / "embeddings")
    emb_va_g = _slice(emb_all_g, off_va, n_va, "kpgt", task_dir / "embeddings")
    emb_te_s = _slice(emb_all_s, off_te, n_te, "chemberta", task_dir / "embeddings")
    emb_te_3d = _slice(emb_all_3d, off_te, n_te, "unimol", task_dir / "embeddings")
    emb_te_g = _slice(emb_all_g, off_te, n_te, "kpgt", task_dir / "embeddings")

    train_ds = MultiModalDataset(
        emb_tr_s,
        emb_tr_3d,
        emb_tr_g,
        y_tr,
        task_type=task_type,
    )
    valid_ds = MultiModalDataset(
        emb_va_s,
        emb_va_3d,
        emb_va_g,
        y_va,
        task_type=task_type,
    )
    test_ds = MultiModalDataset(
        emb_te_s,
        emb_te_3d,
        emb_te_g,
        y_te,
        task_type=task_type,
    )

    cfg = meta["config"]
    model = MultiModalFusionMLP(
        dim_smiles=int(meta["dims"]["chemberta"]),
        dim_3d=int(meta["dims"]["unimol"]),
        dim_graph=int(meta["dims"]["kpgt"]),
        out_dim=2 if task_type == "classification" else 1,
        hidden_dim=int(cfg["hidden_dim"]),
        dropout_proj=float(cfg["dropout_proj"]),
        dropout_head=float(cfg["dropout_head"]),
        fusion_type=str(cfg["fusion_type"]),
        task_context_dim=int(cfg.get("task_context_dim", 0)),
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    task_context_vector = build_task_context_vector(
        task_type=task_type,
        primary_metric_name=str(meta["primary_metric_name"]),
        labels=np.asarray(y_tr),
    )
    return {
        "task": task,
        "task_type": task_type,
        "metric": meta["primary_metric_name"],
        "meta": meta,
        "model": model,
        "train_ds": train_ds,
        "valid_ds": valid_ds,
        "test_ds": test_ds,
        "task_context_vector": task_context_vector,
        "label_mean": float(label_mean) if label_mean is not None else None,
        "label_std": float(label_std) if label_std is not None else None,
    }


def eval_split(bundle: dict, dataset: MultiModalDataset, modality: str, device: torch.device):
    loader = DataLoader(dataset, batch_size=int(bundle["meta"]["config"]["batch_size"]), shuffle=False)
    model = bundle["model"].to(device)
    task_context_vector = bundle["task_context_vector"]
    modality_mask = MODALITY_MAP[modality]

    y_true = []
    y_score = []
    gate_rows = []
    with torch.no_grad():
        for batch in loader:
            emb1 = batch["emb1"].to(device)
            emb2 = batch["emb2"].to(device)
            emb3 = batch["emb3"].to(device)
            labels = batch["label"].to(device)
            task_context = expand_task_context(task_context_vector, emb1.shape[0], device)
            outputs = model(emb1, emb2, emb3, modality_mask=modality_mask, task_context=task_context)

            if bundle["task_type"] == "classification":
                probs = torch.softmax(outputs, dim=1)[:, 1]
                y_true.extend(labels.cpu().numpy().tolist())
                y_score.extend(probs.cpu().numpy().tolist())
            else:
                preds = outputs.view(-1).cpu().numpy().astype(float)
                if bundle["label_mean"] is not None and bundle["label_std"] is not None:
                    preds = preds * float(bundle["label_std"]) + float(bundle["label_mean"])
                y_true.extend(labels.cpu().numpy().astype(float).tolist())
                y_score.extend(preds.tolist())

            if modality == "all" and getattr(model, "gate", None) is not None and not getattr(model, "enable_dual_head", False):
                h_text = model.proj_smiles(emb1)
                h_graph = model.proj_graph(emb3)
                h_3d = model.proj_3d(emb2)
                gate_in = torch.cat([h_text, h_graph, h_3d], dim=1)
                gate_logits = model.gate(gate_in)
                gate = torch.softmax(gate_logits, dim=1)
                gate_rows.append(gate.cpu().numpy())

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if bundle["task_type"] == "classification":
        auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
        auprc = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
        ll = float(log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6)))
        metrics = {"AUROC": auc, "AUPRC": auprc, "LOGLOSS": ll}
    else:
        mae = float(mean_absolute_error(y_true, y_score))
        mse = float(mean_squared_error(y_true, y_score))
        rmse = float(math.sqrt(mse))
        r_true = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
        r_pred = pd.Series(y_score).rank(method="average").to_numpy(dtype=float)
        spearman = float(np.corrcoef(r_true, r_pred)[0, 1]) if np.std(r_true) > 0 and np.std(r_pred) > 0 else float("nan")
        metrics = {"MAE": mae, "RMSE": rmse, "Spearman": spearman}

    gate_stats = {}
    if gate_rows:
        gates = np.concatenate(gate_rows, axis=0)
        gate_stats = {
            "gate_mean_smiles": float(gates[:, 0].mean()),
            "gate_mean_graph": float(gates[:, 1].mean()),
            "gate_mean_3d": float(gates[:, 2].mean()),
            "gate_entropy": float((-(gates * np.log(np.clip(gates, 1e-12, 1.0))).sum(axis=1)).mean()),
        }
    return metrics, gate_stats


def metric_value(metrics: dict, metric_name: str) -> float:
    key = metric_name.upper()
    if key in {"AUROC", "AUPRC", "MAE", "RMSE"}:
        return float(metrics[key])
    if key == "SPEARMAN":
        return float(metrics["Spearman"])
    raise KeyError(metric_name)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    master_path = Path(args.master)
    run_root = Path(args.run_root)
    out_root = Path(args.out_root)

    out_root.mkdir(parents=True, exist_ok=True)
    rows = load_master_rows(master_path)
    if args.tasks:
        keep = set(args.tasks)
        rows = [row for row in rows if row["task"] in keep]
    if args.only_regression:
        rows = [row for row in rows if str(row["tdc_metric"]).upper() in {"MAE", "RMSE", "SPEARMAN"}]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = []
    for row in rows:
        task = row["task"]
        bundle = load_task_bundle(task, root=root, run_root=run_root)
        task_rows = []
        task_out = out_root / task
        task_out.mkdir(parents=True, exist_ok=True)
        tdc_metric = row["tdc_metric"]
        for modality in MODALITIES:
            valid_metrics, gate_stats = eval_split(bundle, bundle["valid_ds"], modality, device)
            test_metrics, _ = eval_split(bundle, bundle["test_ds"], modality, device)
            rec = {
                "task": task,
                "tdc_metric": tdc_metric,
                "modality": modality,
                "valid_tdc_score": metric_value(valid_metrics, tdc_metric),
                "test_tdc_score": metric_value(test_metrics, tdc_metric),
                "valid_auroc": valid_metrics.get("AUROC", float("nan")),
                "valid_auprc": valid_metrics.get("AUPRC", float("nan")),
                "valid_logloss": valid_metrics.get("LOGLOSS", float("nan")),
                "test_auroc": test_metrics.get("AUROC", float("nan")),
                "test_auprc": test_metrics.get("AUPRC", float("nan")),
                "test_mae": test_metrics.get("MAE", float("nan")),
                "test_rmse": test_metrics.get("RMSE", float("nan")),
                "test_spearman": test_metrics.get("Spearman", float("nan")),
                **gate_stats,
            }
            task_rows.append(rec)
            summary.append(rec)
        reverse = row["metric_direction"] == "max"
        task_rows = sorted(task_rows, key=lambda x: x["valid_tdc_score"], reverse=reverse)
        pd.DataFrame(task_rows).to_csv(task_out / "ranked_runs.csv", index=False)
    pd.DataFrame(summary).to_csv(out_root / "summary.csv", index=False)


if __name__ == "__main__":
    main()
