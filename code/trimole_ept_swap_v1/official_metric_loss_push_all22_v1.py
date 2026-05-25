from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO = Path("/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1")
MASTER = REPO / "results_strict" / "ept_family_routing_master_v1" / "ept_family_routing_master_v1_patched_v3_5seed.csv"
DATA_ROOT = REPO / "data" / "data_benchmark_official_v1"
OUT_ROOT = REPO / "results_strict" / "official_metric_loss_push_all22_v1"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from trimole.configs.task_configs import get_task_config
from trimole.training.trainer import TrainConfig, fit_on_task


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", type=str, default=str(MASTER))
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    p.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_master(master_path: Path) -> dict[str, dict[str, str]]:
    with master_path.open() as f:
        rows = list(csv.DictReader(f))
    return {row["task"]: row for row in rows if str(row.get("selected", "False")).lower() == "true"}


def raw_modalities(candidate: str) -> str:
    if candidate == "kpgt_ept":
        return "ept_kpgt"
    if candidate == "chemberta_kpgt_ept_gated":
        return "all"
    return candidate


def direction_better(new_value: float, old_value: float, direction: str) -> bool:
    return new_value > old_value if direction == "max" else new_value < old_value


def normalize_metric(metric: str) -> str:
    m = str(metric).upper()
    if m == "AUCPR":
        return "AUPRC"
    if m == "SPEARMAN":
        return "Spearman"
    return m


def task_type_from_metric(metric: str) -> str:
    m = normalize_metric(metric)
    return "classification" if m in {"AUROC", "AUPRC"} else "regression"


def profile_for_metric(metric: str) -> tuple[str, str]:
    metric = normalize_metric(metric)
    if metric == "AUPRC":
        return "ap_surrogate", "ap_surrogate"
    if metric == "AUROC":
        return "auc_margin", "auc_surrogate"
    return "metric_auto", "auto"


def build_config(task: str, base_row: dict[str, str], seed: int, metric: str, loss_type: str) -> TrainConfig:
    task_cfg = get_task_config(task)
    metric = normalize_metric(metric)
    task_type = task_type_from_metric(metric)
    use_weighted_sampler = bool(task_type == "classification" and metric == "AUPRC")
    return TrainConfig(
        seed=seed,
        task_name=task,
        modalities=raw_modalities(base_row["candidate"]),
        fusion_type=base_row["head"] or "mlp",
        hidden_dim=int(task_cfg.get("hidden_dim", 128)),
        batch_size=int(task_cfg.get("batch_size", 64)),
        lr=float(task_cfg.get("lr", 3e-4)),
        max_epochs=int(task_cfg.get("max_epochs", 80)),
        max_patience=int(task_cfg.get("max_patience", 15)),
        weight_decay=float(task_cfg.get("weight_decay", 0.0)),
        dropout_proj=float(task_cfg.get("dropout_proj", 0.2)),
        dropout_head=float(task_cfg.get("dropout_head", 0.3)),
        task_type=task_type,
        primary_metric_name=metric,
        focal_gamma=float(task_cfg.get("focal_gamma", 2.0)),
        label_smoothing=0.1,
        spearman_reg=0.1,
        loss_type=loss_type,
        use_weighted_sampler=use_weighted_sampler,
        sampler_pos_weight=2.0 if use_weighted_sampler else 1.0,
        task_context_dim=3,
        save_aux_predictions=False,
        aux_loss_weight=0.15,
        diversity_loss_weight=0.0,
    )


def extract_test_metric(meta: dict[str, object], metric: str) -> float:
    metric = normalize_metric(metric)
    if metric == "AUPRC":
        return float(meta["test_auprc"])
    if metric == "AUROC":
        return float(meta["test_auc"])
    if metric == "Spearman":
        return float(meta["test_spearman"])
    if metric == "MAE":
        return float(meta["test_mae"])
    if metric == "RMSE":
        return float(meta["test_rmse"])
    raise KeyError(f"unsupported metric: {metric}")


def run_one(task: str, base_row: dict[str, str], data_root: Path, out_root: Path, seed: int, force: bool) -> dict[str, object]:
    metric = normalize_metric(base_row["tdc_metric"])
    profile, loss_type = profile_for_metric(metric)
    out_dir = out_root / f"{task}__{profile}__seed_{seed}"
    result_json = out_dir / "result.json"
    if result_json.exists() and not force:
        return json.loads(result_json.read_text())

    meta = fit_on_task(
        task_dir=data_root / task,
        out_dir=out_dir,
        config=build_config(task, base_row, seed, metric, loss_type),
    )

    valid = float(meta["best_valid_primary"])
    test = extract_test_metric(meta, metric)
    incumbent_valid = float(base_row["valid_tdc_score_mean"] or base_row["valid_tdc_score"])
    incumbent_test = float(base_row["test_tdc_score_mean"] or base_row["test_tdc_score"])
    result = {
        "task": task,
        "candidate": base_row["candidate"],
        "head": base_row["head"],
        "loss_profile": profile,
        "loss_type": meta["loss_type"],
        "seed": seed,
        "tdc_metric": metric,
        "metric_direction": base_row["metric_direction"],
        "valid_tdc_score": valid,
        "test_tdc_score": test,
        "incumbent_valid_tdc_score": incumbent_valid,
        "incumbent_test_tdc_score": incumbent_test,
        "improved_valid": direction_better(valid, incumbent_valid, base_row["metric_direction"]),
        "improved_test": direction_better(test, incumbent_test, base_row["metric_direction"]),
        "tdc_top1_ref": float(base_row["tdc_top1_ref"]),
        "gap_vs_top1_ref": abs(test - float(base_row["tdc_top1_ref"])),
        "source_results_dir": str(out_dir),
        "train_pred_file": str(out_dir / "train_predictions.csv"),
        "valid_pred_file": str(out_dir / "valid_predictions.csv"),
        "test_pred_file": str(out_dir / "test_predictions.csv"),
    }
    result_json.write_text(json.dumps(result, indent=2))
    return result


def main():
    args = parse_args()
    master = load_master(Path(args.master))
    tasks = args.tasks or sorted(master)
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for task in tasks:
        try:
            results.append(run_one(task, master[task], data_root, out_root, args.seed, args.force))
        except Exception as exc:
            results.append(
                {
                    "task": task,
                    "loss_profile": "metric_all22",
                    "seed": args.seed,
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
                "tasks": tasks,
                "seed": args.seed,
                "count": len(tasks),
            },
            indent=2,
        )
    )
    print(summary)


if __name__ == "__main__":
    main()
