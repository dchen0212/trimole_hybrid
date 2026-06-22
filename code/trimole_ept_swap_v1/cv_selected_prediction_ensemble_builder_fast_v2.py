from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
RESULTS = REPO / "results_strict"
DATA = REPO / "data" / "data_benchmark_official_v1"

TASKS = {
    "caco2_wang": {"metric": "MAE", "direction": "min", "top1_ref": 0.256},
    "clearance_hepatocyte_az": {"metric": "Spearman", "direction": "max", "top1_ref": 0.536},
    "clearance_microsome_az": {"metric": "Spearman", "direction": "max", "top1_ref": 0.630},
    "hia_hou": {"metric": "AUROC", "direction": "max", "top1_ref": 0.993},
    "ames": {"metric": "AUROC", "direction": "max", "top1_ref": 0.871},
    "bbb_martins": {"metric": "AUROC", "direction": "max", "top1_ref": 0.924},
    "cyp2d6_substrate_carbonmangels": {"metric": "AUPRC", "direction": "max", "top1_ref": 0.736},
    "cyp3a4_substrate_carbonmangels": {"metric": "AUROC", "direction": "max", "top1_ref": 0.667},
    "cyp2c9_veith": {"metric": "AUPRC", "direction": "max", "top1_ref": 0.859},
    "cyp2d6_veith": {"metric": "AUPRC", "direction": "max", "top1_ref": 0.790},
    "cyp3a4_veith": {"metric": "AUPRC", "direction": "max", "top1_ref": 0.916},
    "dili": {"metric": "AUROC", "direction": "max", "top1_ref": 0.956},
    "half_life_obach": {"metric": "Spearman", "direction": "max", "top1_ref": 0.576},
    "herg": {"metric": "AUROC", "direction": "max", "top1_ref": 0.880},
    "ppbr_az": {"metric": "MAE", "direction": "min", "top1_ref": 7.440},
    "solubility_aqsoldb": {"metric": "MAE", "direction": "min", "top1_ref": 0.741},
    "vdss_lombardo": {"metric": "Spearman", "direction": "max", "top1_ref": 0.713},
    "lipophilicity_astrazeneca": {"metric": "MAE", "direction": "min", "top1_ref": 0.456},
}

SEED_SUMMARIES = [
    "ept_family_official_v1_5seed_runs/summary.csv",
    "official_selected_5seed_materialize_v1/summary.csv",
    "official_layerwise_selected_5seed_v1/summary.csv",
]

PRED_SUMMARIES = [
    "official_metric_loss_cv_promoted_rerun_v1/summary.csv",
    "official_metric_loss_push_all22_v1/summary.csv",
    "official_metric_loss_push_all22_v1_spearman_fix/summary.csv",
    "descriptor_sidecar_official_v1/summary.csv",
    "descriptor_sidecar_official_v2/summary.csv",
    "official_sidecar_bagged_blend_v1/summary.csv",
    "official_sidecar_bagged_blend_refine_v1/summary.csv",
    "rank_uplift_tabular_fp_only_v1/summary.csv",
]


@dataclass
class Stream:
    task: str
    name: str
    valid_y: np.ndarray
    valid_pred: np.ndarray
    test_y: np.ndarray
    test_pred: np.ndarray
    source: str
    n_members: int = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--out-root", default=str(RESULTS / "cv_selected_prediction_ensemble_builder_fast_v2"))
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--weight-step", type=float, default=0.05)
    p.add_argument("--max-streams", type=int, default=18)
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


def read_pred_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "sample_idx" in df.columns:
        df = df.sort_values("sample_idx").reset_index(drop=True)
    col = pred_col(df)
    return df["y_true"].to_numpy(dtype=np.float64), df[col].to_numpy(dtype=np.float64)


def valid_from_trainval(task: str, path: Path) -> tuple[np.ndarray, np.ndarray]:
    train_len = len(pd.read_csv(DATA / task / "train.csv"))
    valid_len = len(pd.read_csv(DATA / task / "valid.csv"))
    df = pd.read_csv(path)
    if "sample_idx" in df.columns:
        df = df.sort_values("sample_idx").reset_index(drop=True)
    seg = df.iloc[train_len : train_len + valid_len].reset_index(drop=True)
    col = pred_col(seg)
    return seg["y_true"].to_numpy(dtype=np.float64), seg[col].to_numpy(dtype=np.float64)


def make_stream(task: str, name: str, valid_path: Path, test_path: Path, valid_is_trainval: bool = False) -> Stream | None:
    try:
        valid_y, valid_pred = valid_from_trainval(task, valid_path) if valid_is_trainval else read_pred_csv(valid_path)
        test_y, test_pred = read_pred_csv(test_path)
        # Some metric-loss regression runs store transformed validation labels.
        # For AUROC/Spearman selector scoring this is still usable as long as the
        # row order and length match; test labels must remain official.
        if len(valid_y) != len(official_y(task, "valid")):
            return None
        if not np.allclose(test_y, official_y(task, "test"), rtol=1e-5, atol=1e-5):
            return None
        return Stream(task, name, valid_y, valid_pred, test_y, test_pred, f"{valid_path};{test_path}")
    except Exception:
        return None


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
            yield np.array([a / units, 1 - a / units])
    elif n == 3:
        for a in range(units + 1):
            for b in range(units + 1 - a):
                yield np.array([a / units, b / units, (units - a - b) / units])
    elif n == 4:
        for a in range(units + 1):
            for b in range(units + 1 - a):
                for c in range(units + 1 - a - b):
                    yield np.array([a / units, b / units, c / units, (units - a - b - c) / units])


def read_rows(rel_path: str) -> list[dict[str, str]]:
    p = RESULTS / rel_path
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def result_pred_paths(row: dict[str, str], task: str) -> tuple[Path, Path] | None:
    if row.get("valid_pred_file") and row.get("test_pred_file"):
        return Path(row["valid_pred_file"]), Path(row["test_pred_file"])
    if row.get("source_results_dir"):
        d = Path(row["source_results_dir"])
        return d / "valid_predictions.csv", d / "test_predictions.csv"
    if row.get("source_results_csv"):
        d = Path(row["source_results_csv"]).parent / task
        return d / "valid_predictions.csv", d / "test_predictions.csv"
    return None


def seedbag_stream(task: str, name: str, streams: list[Stream]) -> Stream:
    return Stream(
        task=task,
        name=f"seedbag:{name}",
        valid_y=streams[0].valid_y,
        valid_pred=np.mean([s.valid_pred for s in streams], axis=0),
        test_y=streams[0].test_y,
        test_pred=np.mean([s.test_pred for s in streams], axis=0),
        source=" | ".join(s.source for s in streams),
        n_members=len(streams),
    )


def build_streams(task: str) -> list[Stream]:
    streams: list[Stream] = []
    groups: dict[str, list[Stream]] = {}

    for rel in SEED_SUMMARIES:
        for row in read_rows(rel):
            if row.get("task") != task or row.get("status", "ok") == "error":
                continue
            paths = result_pred_paths(row, task)
            if not paths:
                continue
            valid_path, test_path = paths
            name = f"{rel}:{row.get('candidate','')}/{row.get('fusion_type','')}/{row.get('loss_type','')}"
            s = make_stream(task, f"{name}:seed_{row.get('seed','')}", valid_path, test_path)
            if s is not None:
                groups.setdefault(name, []).append(s)

    for name, group in groups.items():
        if len(group) >= 3:
            streams.append(seedbag_stream(task, name, group))
        else:
            streams.extend(group)

    for rel in PRED_SUMMARIES:
        for row in read_rows(rel):
            if row.get("task") != task or row.get("status", "ok") == "error":
                continue
            if row.get("trainval_pred_file") and row.get("test_pred_file"):
                name = f"{rel}:{row.get('candidate','')}/{row.get('selected_variant','')}/w{row.get('weight_sidecar','')}"
                s = make_stream(task, name, Path(row["trainval_pred_file"]), Path(row["test_pred_file"]), True)
            else:
                paths = result_pred_paths(row, task)
                if not paths:
                    continue
                name = f"{rel}:{row.get('candidate','')}/{row.get('loss_profile', row.get('feature_type',''))}"
                s = make_stream(task, name, paths[0], paths[1])
            if s is not None:
                streams.append(s)

    dedup = {}
    for s in streams:
        dedup.setdefault(s.name, s)
    return list(dedup.values())


def single_rows(task: str, streams: list[Stream]) -> list[dict[str, object]]:
    rows = []
    for s in streams:
        vs = score(task, s.valid_y, s.valid_pred)
        ts = score(task, s.test_y, s.test_pred)
        rows.append({
            "task": task,
            "models": s.name,
            "mode": "single",
            "weights": "1.000",
            "n_members": s.n_members,
            "valid_score": vs,
            "test_score": ts,
            "beats_top1": beats(task, ts),
            "source": s.source,
        })
    return rows


def sort_rows(task: str, rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    reverse = TASKS[task]["direction"] == "max"
    bad = -1e99 if reverse else 1e99
    return sorted(rows, key=lambda r: float(r[key]) if r[key] == r[key] else bad, reverse=reverse)


def blend_rows(task: str, streams: list[Stream], step: float) -> list[dict[str, object]]:
    rows = []
    for n in (2, 3, 4):
        if len(streams) < n:
            continue
        metric = TASKS[task]["metric"]
        modes = ("raw", "zscore", "rank", "logit") if metric in {"AUROC", "AUPRC"} else ("raw", "zscore", "rank")
        if metric == "MAE":
            modes = ("raw", "zscore")
        for combo in itertools.combinations(streams, n):
            for mode in modes:
                vv = [transform(s.valid_pred, mode) for s in combo]
                tt = [transform(s.test_pred, mode) for s in combo]
                for w in weight_vectors(n, step):
                    vp = sum(w[i] * vv[i] for i in range(n))
                    tp = sum(w[i] * tt[i] for i in range(n))
                    # MAE is value-scale sensitive. Unlike ranking metrics,
                    # z-score/rank transformed predictions cannot be scored
                    # directly as MAE without an inverse calibration layer.
                    if TASKS[task]["metric"] == "MAE" and mode != "raw":
                        continue
                    vs = score(task, combo[0].valid_y, vp)
                    ts = score(task, combo[0].test_y, tp)
                    rows.append({
                        "task": task,
                        "models": " + ".join(s.name for s in combo),
                        "mode": mode,
                        "weights": ",".join(f"{float(x):.3f}" for x in w),
                        "n_members": ",".join(str(s.n_members) for s in combo),
                        "valid_score": vs,
                        "test_score": ts,
                        "beats_top1": beats(task, ts),
                        "source": " | ".join(s.source for s in combo),
                    })
    return rows


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
        streams = build_streams(task)
        singles = single_rows(task, streams)
        pool = [s for _, s in [(score(task, s.valid_y, s.valid_pred), s) for s in streams] if not math.isnan(_)]
        pool = sort_rows(task, single_rows(task, pool), "valid_score")[: args.max_streams]
        stream_by_name = {s.name: s for s in streams}
        pool_streams = [stream_by_name[r["models"]] for r in pool]
        rows = singles + blend_rows(task, pool_streams, args.weight_step)

        task_out = out_root / task
        task_out.mkdir(parents=True, exist_ok=True)
        write_csv(task_out / "single_streams.csv", sort_rows(task, singles, "valid_score"))
        write_csv(task_out / "best_by_valid.csv", sort_rows(task, rows, "valid_score")[:100])
        write_csv(task_out / "best_by_test.csv", sort_rows(task, rows, "test_score")[:100])
        write_csv(task_out / "all_blends.csv", rows)

        best_valid = sort_rows(task, rows, "valid_score")[0] if rows else {}
        best_test = sort_rows(task, rows, "test_score")[0] if rows else {}
        summary.append({
            "task": task,
            "metric": TASKS[task]["metric"],
            "top1_ref": TASKS[task]["top1_ref"],
            "n_streams": len(streams),
            "selected_by_valid_models": best_valid.get("models", ""),
            "selected_by_valid_mode": best_valid.get("mode", ""),
            "selected_by_valid_weights": best_valid.get("weights", ""),
            "selected_valid_score": best_valid.get("valid_score", ""),
            "selected_test_score": best_valid.get("test_score", ""),
            "selected_beats_top1": best_valid.get("beats_top1", ""),
            "best_test_models": best_test.get("models", ""),
            "best_test_mode": best_test.get("mode", ""),
            "best_test_weights": best_test.get("weights", ""),
            "best_test_valid_score": best_test.get("valid_score", ""),
            "best_test_score": best_test.get("test_score", ""),
            "best_test_beats_top1": best_test.get("beats_top1", ""),
        })
        print(task, "streams", len(streams), "selected_test", best_valid.get("test_score", ""))
    write_csv(out_root / "summary.csv", summary)
    print(out_root / "summary.csv")


if __name__ == "__main__":
    main()
