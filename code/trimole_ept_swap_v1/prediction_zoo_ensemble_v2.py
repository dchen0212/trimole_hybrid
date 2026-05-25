from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import descriptor_sidecar_official_v1 as base
import official_sidecar_nested_refit_v1 as nested
import paper_main_multimodal_prior_taskwise_v1 as v1


REPO = Path("/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1")
DATA_ROOT = REPO / "data" / "data_benchmark_official_v1"
MASTER = (
    REPO
    / "results_strict"
    / "ept_family_routing_master_v1"
    / "ept_family_routing_master_v1_metric_cv_sidecar_layerwise_selected_v4.csv"
)
OUT_ROOT = REPO / "results_strict" / "prediction_zoo_ensemble_v2"
FOCUS_TASKS = ["clearance_microsome_az", "hia_hou", "cyp2d6_substrate_carbonmangels", "ames"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--master", default=str(MASTER))
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--tasks", nargs="*", default=FOCUS_TASKS)
    p.add_argument("--max-models", type=int, default=3)
    p.add_argument("--top-candidates", type=int, default=12)
    p.add_argument("--weight-step", type=float, default=0.05)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_master(path: Path) -> dict[str, dict[str, str]]:
    return {row["task"]: row for row in csv.DictReader(path.open())}


def pred_col(path: Path) -> np.ndarray:
    return nested.load_pred_column(path)


def valid_tail_from_trainval(path: Path, task: str, data_root: Path) -> np.ndarray:
    pred = pred_col(path)
    n_valid = len(pd.read_csv(data_root / task / "valid.csv"))
    return pred[-n_valid:].astype(np.float32)


def labels(task: str, data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    valid = pd.read_csv(data_root / task / "valid.csv")
    test = pd.read_csv(data_root / task / "test.csv")
    y_col = base.find_label_col(valid)
    return valid[y_col].to_numpy(), test[y_col].to_numpy()


def rank(x: np.ndarray) -> np.ndarray:
    import scipy.stats as st

    return st.rankdata(np.asarray(x, dtype=float), method="average").astype(np.float32)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    std = float(np.std(x))
    if std < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - float(np.mean(x))) / std).astype(np.float32)


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x)).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32)))).astype(np.float32)


def transform(valid: np.ndarray, test: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "raw":
        return valid, test
    if mode == "rank":
        return rank(valid), rank(test)
    if mode == "zscore":
        return zscore(valid), zscore(test)
    if mode == "logit":
        return logit(valid), logit(test)
    raise ValueError(mode)


def inverse(pred: np.ndarray, mode: str) -> np.ndarray:
    return sigmoid(pred) if mode == "logit" else pred


def weight_vectors(n: int, step: float):
    units = int(round(1.0 / step))
    if n == 1:
        yield np.ones(1, dtype=np.float32)
    elif n == 2:
        for a in range(units + 1):
            yield np.array([a / units, 1 - a / units], dtype=np.float32)
    elif n == 3:
        for a in range(units + 1):
            for b in range(units + 1 - a):
                c = units - a - b
                yield np.array([a / units, b / units, c / units], dtype=np.float32)


def load_seeded_group(root: Path, task: str, label: str) -> dict[str, object] | None:
    valid_paths = sorted(root.glob(f"{task}__*__seed_*/run_*/{task}/valid_predictions.csv"))
    test_paths = sorted(root.glob(f"{task}__*__seed_*/run_*/{task}/test_predictions.csv"))
    if not valid_paths or len(valid_paths) != len(test_paths):
        return None
    try:
        valid = np.stack([pred_col(p) for p in valid_paths], axis=0).mean(axis=0).astype(np.float32)
        test = np.stack([pred_col(p) for p in test_paths], axis=0).mean(axis=0).astype(np.float32)
    except Exception:
        return None
    return {
        "name": label,
        "valid": valid,
        "test": test,
        "valid_paths": ";".join(str(p) for p in valid_paths),
        "test_paths": ";".join(str(p) for p in test_paths),
        "n_files": len(valid_paths),
    }


def load_trainval_pair(root: Path, task: str, label: str, data_root: Path) -> dict[str, object] | None:
    tv = root / task / "trainval_predictions.csv"
    te = root / task / "test_predictions.csv"
    if not (tv.exists() and te.exists()):
        return None
    try:
        return {
            "name": label,
            "valid": valid_tail_from_trainval(tv, task, data_root),
            "test": pred_col(te),
            "valid_paths": str(tv),
            "test_paths": str(te),
            "n_files": 1,
        }
    except Exception:
        return None


def discover(task: str, data_root: Path) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seeded_roots = [
        ("incumbent_5seed", REPO / "results_strict" / "ept_family_official_v1_5seed_runs"),
        ("selected_5seed", REPO / "results_strict" / "official_selected_5seed_materialize_v1"),
        ("layerwise_5seed", REPO / "results_strict" / "official_layerwise_selected_5seed_v1"),
    ]
    for label, root in seeded_roots:
        item = load_seeded_group(root, task, label)
        if item:
            candidates.append(item)

    trainval_roots = [
        ("paper_main_v1", REPO / "results_strict" / "paper_main_multimodal_prior_taskwise_v1"),
        ("chemical_prior_v2", REPO / "results_strict" / "paper_main_chemical_prior_v2_focus"),
        ("chem_select_multibackend_v3", REPO / "results_strict" / "paper_main_chem_select_multibackend_v3_focus"),
        ("tabular_fp_only", REPO / "results_strict" / "rank_uplift_tabular_fp_only_v1"),
        ("tabular_fp_repeated", REPO / "results_strict" / "rank_uplift_tabular_fp_repeated_v1_focus"),
        ("sidecar_bagged_blend", REPO / "results_strict" / "official_sidecar_bagged_blend_v1"),
        ("sidecar_bagged_refine", REPO / "results_strict" / "official_sidecar_bagged_blend_refine_v1"),
        ("sidecar_nested_refit", REPO / "results_strict" / "official_sidecar_nested_refit_v1"),
    ]
    for label, root in trainval_roots:
        item = load_trainval_pair(root, task, label, data_root)
        if item:
            candidates.append(item)

    seen = set()
    out = []
    for item in candidates:
        key = (item["name"], item["valid_paths"], item["test_paths"])
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def score(metric: str, y: np.ndarray, pred: np.ndarray) -> float:
    return float(base.score_metric(metric, y, pred))


def better(a: float, b: float, direction: str) -> bool:
    return a > b if direction == "max" else a < b


def run_task(task: str, master: dict[str, dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    row = master[task]
    metric = row["tdc_metric"]
    direction = row["metric_direction"]
    top1_ref = float(row["tdc_top1_ref"])
    yv, yt = labels(task, Path(args.data_root))
    candidates = []
    for item in discover(task, Path(args.data_root)):
        if len(item["valid"]) != len(yv) or len(item["test"]) != len(yt):  # type: ignore[arg-type]
            continue
        item["valid_score"] = score(metric, yv, item["valid"])  # type: ignore[arg-type]
        item["test_score"] = score(metric, yt, item["test"])  # type: ignore[arg-type]
        item["gap_vs_top1_ref"] = abs(float(item["test_score"]) - top1_ref)
        candidates.append(item)

    candidates.sort(key=lambda x: float(x["valid_score"]), reverse=(direction == "max"))
    candidates = candidates[: args.top_candidates]

    modes = ["raw"]
    if metric in ("AUROC", "AUPRC"):
        modes.append("logit")
    if metric == "Spearman":
        modes.extend(["rank", "zscore"])

    best_row: dict[str, object] | None = None
    result_rows: list[dict[str, object]] = []
    for n in range(1, min(args.max_models, len(candidates)) + 1):
        for combo in combinations(candidates, n):
            for mode in modes:
                transformed = [transform(c["valid"], c["test"], mode) for c in combo]  # type: ignore[arg-type]
                valid_stack = np.stack([x[0] for x in transformed], axis=0)
                test_stack = np.stack([x[1] for x in transformed], axis=0)
                for weights in weight_vectors(n, args.weight_step):
                    pv = inverse(np.tensordot(weights, valid_stack, axes=(0, 0)), mode)
                    pt = inverse(np.tensordot(weights, test_stack, axes=(0, 0)), mode)
                    valid_score = score(metric, yv, pv)
                    test_score = score(metric, yt, pt)
                    out = {
                        "task": task,
                        "n_models": n,
                        "mode": mode,
                        "models": " + ".join(str(c["name"]) for c in combo),
                        "weights": ",".join(f"{float(w):.3f}" for w in weights),
                        "valid_score": valid_score,
                        "test_score": test_score,
                        "tdc_metric": metric,
                        "metric_direction": direction,
                        "tdc_top1_ref": top1_ref,
                        "gap_vs_top1_ref": abs(test_score - top1_ref),
                        "is_top1_level": test_score >= top1_ref if direction == "max" else test_score <= top1_ref,
                    }
                    result_rows.append(out)
                    if best_row is None or better(valid_score, float(best_row["valid_score"]), direction):
                        best_row = out
    if best_row is None:
        best_row = {"task": task, "status": "no_candidates"}

    candidate_rows = []
    for item in candidates:
        candidate_rows.append({k: v for k, v in item.items() if k not in {"valid", "test"}} | {"task": task})
    return candidate_rows + result_rows, best_row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    master = load_master(Path(args.master))
    all_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    for task in args.tasks:
        print(f"[task] {task}", flush=True)
        rows, best = run_task(task, master, args)
        all_rows.extend(rows)
        best_rows.append(best)
    write_csv(out_root / "all_trials.csv", all_rows)
    write_csv(out_root / "summary.csv", best_rows)
    (out_root / "meta.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
