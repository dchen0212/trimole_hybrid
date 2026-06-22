from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import descriptor_sidecar_official_v1 as base


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
DATA_ROOT = REPO / "data" / "data_benchmark_official_v1"
OUT_ROOT = REPO / "results_strict" / "offline_prediction_zoo_blend_v1"
TASKS = ["clearance_microsome_az", "hia_hou", "pgp_broccatelli"]
METRICS = {
    "clearance_microsome_az": ("SPEARMAN", "max", 0.630),
    "hia_hou": ("AUROC", "max", 0.993),
    "pgp_broccatelli": ("AUROC", "max", 0.938),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str, default=str(REPO))
    p.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    p.add_argument("--tasks", nargs="*", default=TASKS)
    p.add_argument("--max-models", type=int, default=3)
    p.add_argument("--top-candidates", type=int, default=10)
    p.add_argument("--weight-step", type=float, default=0.05)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def pred_col(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "sample_idx" in df.columns:
        df = df.sort_values("sample_idx").reset_index(drop=True)
    for col in ("y_prob", "y_pred", "prediction", "pred"):
        if col in df.columns:
            return df[col].to_numpy(dtype=np.float32)
    raise KeyError(f"prediction column not found in {path}")


def labels(task: str) -> tuple[np.ndarray, np.ndarray]:
    valid = pd.read_csv(DATA_ROOT / task / "valid.csv")
    test = pd.read_csv(DATA_ROOT / task / "test.csv")
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
    x = np.clip(np.asarray(x, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x)).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def transform_pair(valid: np.ndarray, test: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "raw":
        return valid, test
    if mode == "rank":
        return rank(valid), rank(test)
    if mode == "zscore":
        return zscore(valid), zscore(test)
    if mode == "logit":
        return logit(valid), logit(test)
    raise ValueError(mode)


def inverse_blend(pred: np.ndarray, mode: str) -> np.ndarray:
    return sigmoid(pred) if mode == "logit" else pred


def load_candidate(task: str, name: str, valid_paths: list[Path], test_paths: list[Path]) -> dict[str, object]:
    valid_arrays = [pred_col(path) for path in valid_paths]
    test_arrays = [pred_col(path) for path in test_paths]
    if len({len(x) for x in valid_arrays}) != 1 or len({len(x) for x in test_arrays}) != 1:
        raise ValueError(f"length mismatch in {name}")
    return {
        "task": task,
        "name": name,
        "valid": np.stack(valid_arrays, axis=0).mean(axis=0).astype(np.float32),
        "test": np.stack(test_arrays, axis=0).mean(axis=0).astype(np.float32),
        "valid_paths": ";".join(str(x) for x in valid_paths),
        "test_paths": ";".join(str(x) for x in test_paths),
        "n_files": len(valid_paths),
    }


def add_seeded_group(candidates: list[dict[str, object]], name: str, pattern: str, task: str) -> None:
    valid_paths = sorted(REPO.glob(pattern.format(task=task, split="valid")))
    test_paths = sorted(REPO.glob(pattern.format(task=task, split="test")))
    if valid_paths and len(valid_paths) == len(test_paths):
        candidates.append(load_candidate(task, name, valid_paths, test_paths))


def add_pair(candidates: list[dict[str, object]], task: str, name: str, valid_path: str, test_path: str) -> None:
    vp = REPO / valid_path
    tp = REPO / test_path
    if vp.exists() and tp.exists():
        candidates.append(load_candidate(task, name, [vp], [tp]))


def discover_candidates(task: str) -> list[dict[str, object]]:
    c: list[dict[str, object]] = []
    add_seeded_group(
        c,
        "incumbent_5seed_official",
        "results_strict/ept_family_official_v1_5seed_runs/{task}__*__seed_*/run_*/{task}/{split}_predictions.csv",
        task,
    )
    add_seeded_group(
        c,
        "selected_5seed_materialize",
        "results_strict/official_selected_5seed_materialize_v1/{task}__*__seed_*/run_*/{task}/{split}_predictions.csv",
        task,
    )
    add_seeded_group(
        c,
        "layerwise_selected_5seed",
        "results_strict/official_layerwise_selected_5seed_v1/{task}__*__seed_*/run_*/{task}/{split}_predictions.csv",
        task,
    )
    add_seeded_group(
        c,
        "targeted_gated3d_5seed",
        "results_strict/official_targeted_push_v2_5seed/gated3d_probe_v1/{task}__*__seed_*/run_*/{task}/{split}_predictions.csv",
        task,
    )

    single_roots = [
        ("2d_official_chemberta", f"results_strict/trimole_2dcore_all22_official_v1/{task}__chemberta__seed_42/run_*/{task}"),
        ("2d_official_kpgt", f"results_strict/trimole_2dcore_all22_official_v1/{task}__kpgt__seed_42/run_*/{task}"),
        ("2d_official_chemberta_kpgt", f"results_strict/trimole_2dcore_all22_official_v1/{task}__chemberta_kpgt__seed_42/run_*/{task}"),
        ("ept_official_ept", f"results_strict/ept_all22_multimode_official_v1/{task}__ept__seed_42/run_*/{task}"),
        ("ept_official_ept_kpgt", f"results_strict/ept_all22_multimode_official_v1/{task}__ept_kpgt__seed_42/run_*/{task}"),
        ("ept_official_chemberta_ept", f"results_strict/ept_all22_multimode_official_v1/{task}__chemberta_ept__seed_42/run_*/{task}"),
        ("gated_official", f"results_strict/chemberta_kpgt_ept_gated_shortlist_runs_official_v1/{task}__chemberta_kpgt_ept_gated__seed_42/run_*/{task}"),
        ("metric_loss_official", f"results_strict/official_metric_loss_push_all22_v1/{task}__*__seed_42"),
        ("metric_loss_spearman_fix", f"results_strict/official_metric_loss_push_all22_v1_spearman_fix/{task}__*__seed_42"),
        ("sidecar_v1", f"results_strict/descriptor_sidecar_official_v1/{task}__*"),
        ("sidecar_v2", f"results_strict/descriptor_sidecar_official_v2/{task}__*"),
        ("bagged_blend_v1", f"results_strict/official_sidecar_bagged_blend_v1/{task}"),
        ("bagged_blend_refine_v1", f"results_strict/official_sidecar_bagged_blend_refine_v1/{task}"),
        ("nested_refit_v1", f"results_strict/official_sidecar_nested_refit_v1/{task}"),
    ]
    for label, root_glob in single_roots:
        for root in sorted(REPO.glob(root_glob)):
            add_pair(
                c,
                task,
                label if root_glob.endswith("*") is False else f"{label}:{root.name}",
                str(root / "valid_predictions.csv"),
                str(root / "test_predictions.csv"),
            )

    seen = set()
    unique = []
    for item in c:
        key = (item["name"], item["valid_paths"], item["test_paths"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def weight_vectors(n: int, step: float):
    units = int(round(1.0 / step))
    if n == 2:
        for a in range(units + 1):
            yield np.array([a / units, 1.0 - a / units], dtype=np.float32)
    elif n == 3:
        for a in range(units + 1):
            for b in range(units + 1 - a):
                c = units - a - b
                yield np.array([a / units, b / units, c / units], dtype=np.float32)
    else:
        raise ValueError(n)


def evaluate_combo(task: str, combo: tuple[dict[str, object], ...], weights: np.ndarray, mode: str, yv: np.ndarray, yt: np.ndarray) -> dict[str, object]:
    tv = []
    tt = []
    for item in combo:
        v, t = transform_pair(item["valid"], item["test"], mode)  # type: ignore[arg-type]
        tv.append(v)
        tt.append(t)
    pv = inverse_blend(np.tensordot(weights, np.stack(tv, axis=0), axes=(0, 0)), mode)
    pt = inverse_blend(np.tensordot(weights, np.stack(tt, axis=0), axes=(0, 0)), mode)
    metric, _, ref = METRICS[task]
    return {
        "task": task,
        "n_models": len(combo),
        "mode": mode,
        "models": " + ".join(str(x["name"]) for x in combo),
        "weights": ",".join(f"{float(x):.3f}" for x in weights),
        "valid_score": base.score_metric(metric, yv, pv),
        "test_score": base.score_metric(metric, yt, pt),
        "tdc_top1_ref": ref,
        "gap_vs_top1_ref": abs(base.score_metric(metric, yt, pt) - ref),
    }


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []

    for task in args.tasks:
        yv, yt = labels(task)
        metric, _, ref = METRICS[task]
        candidates = []
        for item in discover_candidates(task):
            if len(item["valid"]) != len(yv) or len(item["test"]) != len(yt):  # type: ignore[arg-type]
                continue
            valid_score = base.score_metric(metric, yv, item["valid"])  # type: ignore[arg-type]
            test_score = base.score_metric(metric, yt, item["test"])  # type: ignore[arg-type]
            item["valid_score"] = valid_score
            item["test_score"] = test_score
            item["gap_vs_top1_ref"] = abs(test_score - ref)
            candidates.append(item)
            candidate_rows.append({k: v for k, v in item.items() if k not in {"valid", "test"}})
        candidates.sort(key=lambda item: float(item["valid_score"]), reverse=True)
        if args.top_candidates > 0:
            candidates = candidates[: args.top_candidates]

        modes = ["raw", "logit"] if metric == "AUROC" else ["raw", "zscore", "rank"]
        task_results = []
        for n_models in range(2, min(args.max_models, len(candidates)) + 1):
            from itertools import combinations

            for combo in combinations(candidates, n_models):
                for mode in modes:
                    for weights in weight_vectors(n_models, args.weight_step):
                        row = evaluate_combo(task, combo, weights, mode, yv, yt)
                        task_results.append(row)
                        result_rows.append(row)

        task_results.sort(key=lambda r: float(r["valid_score"]), reverse=True)
        best_rows.extend(task_results[:20])

    for path, rows in [
        (out_root / "prediction_zoo_candidates.csv", candidate_rows),
        (out_root / "offline_blend_results.csv", result_rows),
        (out_root / "offline_blend_best_by_valid.csv", best_rows),
    ]:
        if not rows:
            path.write_text("")
            continue
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    print(out_root / "prediction_zoo_candidates.csv")
    print(out_root / "offline_blend_best_by_valid.csv")


if __name__ == "__main__":
    main()
