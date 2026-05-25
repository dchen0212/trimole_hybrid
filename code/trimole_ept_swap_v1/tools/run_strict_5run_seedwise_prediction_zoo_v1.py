from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score


REPO = Path("/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1")
RESULTS = REPO / "results_strict"
DATA = REPO / "data" / "data_benchmark_official_v1"

TASKS = {
    "cyp2c9_substrate_carbonmangels": {"metric": "AUPRC", "direction": "max", "top1_ref": 0.474},
    "bbb_martins": {"metric": "AUROC", "direction": "max", "top1_ref": 0.924},
    "hia_hou": {"metric": "AUROC", "direction": "max", "top1_ref": 0.993},
    "cyp3a4_substrate_carbonmangels": {"metric": "AUROC", "direction": "max", "top1_ref": 0.667},
    "herg": {"metric": "AUROC", "direction": "max", "top1_ref": 0.880},
    "vdss_lombardo": {"metric": "Spearman", "direction": "max", "top1_ref": 0.713},
    "lipophilicity_astrazeneca": {"metric": "MAE", "direction": "min", "top1_ref": 0.456},
    "caco2_wang": {"metric": "MAE", "direction": "min", "top1_ref": 0.256},
}

SEED_SUMMARIES = [
    "ept_family_official_v1_5seed_runs/summary.csv",
    "official_selected_5seed_materialize_v1/summary.csv",
    "official_layerwise_selected_5seed_v1/summary.csv",
]

SINGLE_SUMMARIES = [
    "official_metric_loss_cv_promoted_rerun_v1/summary.csv",
    "official_metric_loss_push_all22_v1/summary.csv",
    "official_metric_loss_push_all22_v1_spearman_fix/summary.csv",
    "descriptor_sidecar_official_v1/summary.csv",
    "descriptor_sidecar_official_v2/summary.csv",
    "official_sidecar_bagged_blend_v1/summary.csv",
    "official_sidecar_bagged_blend_refine_v1/summary.csv",
    "rank_uplift_tabular_fp_only_v1/summary.csv",
    "paper_main_chemical_prior_xl_v4_all22_32core/summary.csv",
]


@dataclass(frozen=True)
class Pred:
    y: np.ndarray
    pred: np.ndarray
    source: str


@dataclass
class GroupStream:
    task: str
    name: str
    valid: list[Pred]
    test: list[Pred]
    kind: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--out-root", default=str(RESULTS / "strict_5run_seedwise_prediction_zoo_v1"))
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--weight-step", type=float, default=0.1)
    p.add_argument("--max-streams", type=int, default=12)
    p.add_argument("--lambda-std", type=float, default=1.0)
    return p.parse_args()


def pred_col(df: pd.DataFrame) -> str:
    for col in ("y_prob", "y_pred", "prediction", "pred"):
        if col in df.columns:
            return col
    raise KeyError("prediction column not found")


def label_col(df: pd.DataFrame) -> str:
    skip = {"smiles", "drug", "drug_id", "mol", "id", "sample_idx"}
    for col in df.columns:
        if col.lower() not in skip:
            return col
    raise KeyError("label column not found")


def official_y(task: str, split: str) -> np.ndarray:
    df = pd.read_csv(DATA / task / f"{split}.csv")
    return df[label_col(df)].to_numpy(dtype=np.float64)


def read_pred(path: Path) -> Pred:
    df = pd.read_csv(path)
    if "sample_idx" in df.columns:
        df = df.sort_values("sample_idx").reset_index(drop=True)
    col = pred_col(df)
    return Pred(
        y=df["y_true"].to_numpy(dtype=np.float64),
        pred=df[col].to_numpy(dtype=np.float64),
        source=str(path),
    )


def valid_from_trainval(task: str, path: Path) -> Pred:
    train_len = len(pd.read_csv(DATA / task / "train.csv"))
    valid_len = len(pd.read_csv(DATA / task / "valid.csv"))
    df = pd.read_csv(path)
    if "sample_idx" in df.columns:
        df = df.sort_values("sample_idx").reset_index(drop=True)
    seg = df.iloc[train_len : train_len + valid_len].reset_index(drop=True)
    col = pred_col(seg)
    return Pred(
        y=seg["y_true"].to_numpy(dtype=np.float64),
        pred=seg[col].to_numpy(dtype=np.float64),
        source=str(path),
    )


def score(task: str, y: np.ndarray, pred: np.ndarray) -> float:
    metric = TASKS[task]["metric"]
    if metric == "AUROC":
        try:
            return float(roc_auc_score(y, pred))
        except Exception:
            return float("nan")
    if metric == "AUPRC":
        try:
            return float(average_precision_score(y, pred))
        except Exception:
            return float("nan")
    if metric == "Spearman":
        val = spearmanr(y, pred).correlation
        return float(val) if val is not None else float("nan")
    if metric == "MAE":
        try:
            return float(mean_absolute_error(y, pred))
        except Exception:
            return float("nan")
    raise ValueError(metric)


def beats(task: str, value: float) -> bool:
    if math.isnan(value):
        return False
    ref = TASKS[task]["top1_ref"]
    return value >= ref if TASKS[task]["direction"] == "max" else value <= ref


def selector_score(task: str, mean: float, std: float, lambda_std: float) -> float:
    if TASKS[task]["direction"] == "max":
        return mean - lambda_std * std
    return mean + lambda_std * std


def transform(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if mode == "raw":
        return x
    if mode == "zscore":
        std = float(np.std(x))
        return np.zeros_like(x) if std == 0 else (x - float(np.mean(x))) / std
    if mode == "rank":
        r = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
        return np.zeros_like(r) if len(r) <= 1 else (r - 1.0) / (len(r) - 1.0)
    if mode == "logit":
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        return np.log(x / (1.0 - x))
    raise ValueError(mode)


def weight_vectors(n: int, step: float):
    units = int(round(1.0 / step))
    if n == 1:
        yield np.array([1.0])
    elif n == 2:
        for a in range(units + 1):
            yield np.array([a / units, 1.0 - a / units])
    elif n == 3:
        for a in range(units + 1):
            for b in range(units + 1 - a):
                yield np.array([a / units, b / units, (units - a - b) / units])


def read_rows(rel_path: str) -> list[dict[str, str]]:
    p = RESULTS / rel_path
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def result_paths(row: dict[str, str], task: str) -> tuple[Path, Path] | None:
    if row.get("valid_pred_file") and row.get("test_pred_file"):
        return Path(row["valid_pred_file"]), Path(row["test_pred_file"])
    if row.get("source_results_dir"):
        d = Path(row["source_results_dir"])
        return d / "valid_predictions.csv", d / "test_predictions.csv"
    if row.get("source_results_csv"):
        d = Path(row["source_results_csv"]).parent / task
        return d / "valid_predictions.csv", d / "test_predictions.csv"
    if row.get("trainval_pred_file") and row.get("test_pred_file"):
        return Path(row["trainval_pred_file"]), Path(row["test_pred_file"])
    return None


def make_seed_group(task: str, name: str, rows: list[dict[str, str]]) -> GroupStream | None:
    valid: list[Pred] = []
    test: list[Pred] = []
    for row in sorted(rows, key=lambda r: int(float(r.get("seed", len(valid) + 1) or len(valid) + 1))):
        paths = result_paths(row, task)
        if not paths:
            continue
        vpath, tpath = paths
        try:
            v = read_pred(vpath)
            t = read_pred(tpath)
            if len(v.y) != len(official_y(task, "valid")):
                continue
            if not np.allclose(t.y, official_y(task, "test"), rtol=1e-5, atol=1e-5):
                continue
            valid.append(v)
            test.append(t)
        except Exception:
            continue
    if len(valid) < 5:
        return None
    return GroupStream(task=task, name=name, valid=valid[:5], test=test[:5], kind="seed5")


def make_single_replicated(task: str, name: str, vpath: Path, tpath: Path) -> GroupStream | None:
    try:
        if "trainval" in vpath.name:
            v = valid_from_trainval(task, vpath)
        else:
            v = read_pred(vpath)
        t = read_pred(tpath)
        if len(v.y) != len(official_y(task, "valid")):
            return None
        if not np.allclose(t.y, official_y(task, "test"), rtol=1e-5, atol=1e-5):
            return None
        return GroupStream(task=task, name=name, valid=[v] * 5, test=[t] * 5, kind="single_replicated")
    except Exception:
        return None


def build_streams(task: str) -> list[GroupStream]:
    streams: list[GroupStream] = []
    grouped: dict[str, list[dict[str, str]]] = {}
    for rel in SEED_SUMMARIES:
        for row in read_rows(rel):
            if row.get("task") != task or row.get("status", "ok") == "error":
                continue
            name = f"{rel}:{row.get('candidate','')}/{row.get('fusion_type','')}/{row.get('loss_type','')}"
            grouped.setdefault(name, []).append(row)
    for name, rows in grouped.items():
        s = make_seed_group(task, name, rows)
        if s is not None:
            streams.append(s)

    for rel in SINGLE_SUMMARIES:
        for row in read_rows(rel):
            if row.get("task") != task or row.get("status", "ok") == "error":
                continue
            paths = result_paths(row, task)
            if not paths:
                continue
            if row.get("trainval_pred_file"):
                name = f"{rel}:{row.get('candidate','')}/{row.get('selected_variant','')}/w{row.get('weight_sidecar','')}"
                s = make_single_replicated(task, name, Path(row["trainval_pred_file"]), Path(row["test_pred_file"]))
            else:
                name = f"{rel}:{row.get('candidate','')}/{row.get('loss_profile', row.get('feature_type',''))}"
                s = make_single_replicated(task, name, paths[0], paths[1])
            if s is not None:
                streams.append(s)

    dedup = {}
    for s in streams:
        dedup.setdefault(s.name, s)
    return list(dedup.values())


def eval_combo(task: str, combo: tuple[GroupStream, ...], mode: str, weights: np.ndarray, lambda_std: float) -> dict[str, object]:
    valid_scores = []
    test_scores = []
    for i in range(5):
        v_parts = [transform(s.valid[i].pred, mode) for s in combo]
        t_parts = [transform(s.test[i].pred, mode) for s in combo]
        vp = sum(float(weights[j]) * v_parts[j] for j in range(len(combo)))
        tp = sum(float(weights[j]) * t_parts[j] for j in range(len(combo)))
        valid_scores.append(score(task, combo[0].valid[i].y, vp))
        test_scores.append(score(task, combo[0].test[i].y, tp))
    vmean = float(np.nanmean(valid_scores))
    vstd = float(np.nanstd(valid_scores, ddof=1)) if len(valid_scores) > 1 else 0.0
    tmean = float(np.nanmean(test_scores))
    tstd = float(np.nanstd(test_scores, ddof=1)) if len(test_scores) > 1 else 0.0
    return {
        "task": task,
        "metric": TASKS[task]["metric"],
        "top1_ref": TASKS[task]["top1_ref"],
        "models": " + ".join(s.name for s in combo),
        "stream_kinds": " + ".join(s.kind for s in combo),
        "mode": mode,
        "weights": ",".join(f"{float(w):.3f}" for w in weights),
        "valid_mean": vmean,
        "valid_std": vstd,
        "valid_adjusted": selector_score(task, vmean, vstd, lambda_std),
        "test_mean": tmean,
        "test_std": tstd,
        "beats_top1_mean": beats(task, tmean),
        "valid_scores": ";".join(f"{x:.12f}" for x in valid_scores),
        "test_scores": ";".join(f"{x:.12f}" for x in test_scores),
    }


def better_sort(task: str, key: str):
    reverse = TASKS[task]["direction"] == "max"
    return reverse


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    global REPO, RESULTS, DATA
    REPO = Path(args.repo)
    RESULTS = REPO / "results_strict"
    DATA = REPO / "data" / "data_benchmark_official_v1"
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for task in args.tasks:
        print("[task]", task, flush=True)
        streams = build_streams(task)
        rows = []
        metric = TASKS[task]["metric"]
        modes = ("raw", "zscore", "rank")
        if metric in {"AUROC", "AUPRC"}:
            modes = ("raw", "zscore", "rank", "logit")
        if metric == "MAE":
            modes = ("raw",)
        # Pre-rank by single-stream adjusted valid to keep the grid bounded.
        singles = [eval_combo(task, (s,), "raw", np.array([1.0]), args.lambda_std) for s in streams]
        reverse = TASKS[task]["direction"] == "max"
        singles_sorted = sorted(singles, key=lambda r: float(r["valid_adjusted"]), reverse=reverse)
        keep_names = {r["models"] for r in singles_sorted[: args.max_streams]}
        kept = [s for s in streams if s.name in keep_names]
        rows.extend(singles)
        for n in (2, 3):
            for combo in itertools.combinations(kept, n):
                for mode in modes:
                    for weights in weight_vectors(n, args.weight_step):
                        if np.max(weights) >= 0.999 and n > 1:
                            continue
                        rows.append(eval_combo(task, combo, mode, weights, args.lambda_std))
        task_out = out_root / task
        task_out.mkdir(parents=True, exist_ok=True)
        reverse = TASKS[task]["direction"] == "max"
        by_valid = sorted(rows, key=lambda r: float(r["valid_adjusted"]), reverse=reverse)
        by_test = sorted(rows, key=lambda r: float(r["test_mean"]), reverse=reverse)
        write_csv(task_out / "all_results.csv", rows)
        write_csv(task_out / "best_by_valid_adjusted.csv", by_valid[:100])
        write_csv(task_out / "best_by_test_mean_diagnostic.csv", by_test[:100])
        selected = by_valid[0] if by_valid else {}
        best_test = by_test[0] if by_test else {}
        summary.append({
            "task": task,
            "metric": TASKS[task]["metric"],
            "top1_ref": TASKS[task]["top1_ref"],
            "n_streams": len(streams),
            "selected_models": selected.get("models", ""),
            "selected_mode": selected.get("mode", ""),
            "selected_weights": selected.get("weights", ""),
            "selected_valid_mean": selected.get("valid_mean", ""),
            "selected_valid_std": selected.get("valid_std", ""),
            "selected_valid_adjusted": selected.get("valid_adjusted", ""),
            "selected_test_mean": selected.get("test_mean", ""),
            "selected_test_std": selected.get("test_std", ""),
            "selected_beats_top1_mean": selected.get("beats_top1_mean", ""),
            "best_test_models_diagnostic": best_test.get("models", ""),
            "best_test_mode_diagnostic": best_test.get("mode", ""),
            "best_test_weights_diagnostic": best_test.get("weights", ""),
            "best_test_valid_adjusted_diagnostic": best_test.get("valid_adjusted", ""),
            "best_test_mean_diagnostic": best_test.get("test_mean", ""),
            "best_test_std_diagnostic": best_test.get("test_std", ""),
            "best_test_beats_top1_mean_diagnostic": best_test.get("beats_top1_mean", ""),
        })
        print(task, "streams", len(streams), "selected", selected.get("test_mean", ""), "best_diag", best_test.get("test_mean", ""), flush=True)
    write_csv(out_root / "summary.csv", summary)
    print(out_root / "summary.csv", flush=True)


if __name__ == "__main__":
    main()
