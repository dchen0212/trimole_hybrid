from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import cv_selected_prediction_ensemble_builder_fast_v2 as zoo


REPO = Path("/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1")
RESULTS = REPO / "results_strict"

TASKS = {
    "caco2_wang": {"top1_ref": 0.256},
    "ld50_zhu": {"top1_ref": 0.552},
    "lipophilicity_astrazeneca": {"top1_ref": 0.456},
    "ppbr_az": {"top1_ref": 7.440},
    "solubility_aqsoldb": {"top1_ref": 0.741},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--out-root", default=str(RESULTS / "mae_valid_calibrated_prediction_search_v1"))
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--max-streams", type=int, default=8)
    return parser.parse_args()


def weight_vectors(n: int, step: float):
    units = int(round(1.0 / step))
    if n == 1:
        yield np.array([1.0], dtype=np.float64)
    elif n == 2:
        for a in range(units + 1):
            yield np.array([a / units, 1.0 - a / units], dtype=np.float64)
    elif n == 3:
        for a in range(units + 1):
            for b in range(units + 1 - a):
                yield np.array([a / units, b / units, (units - a - b) / units], dtype=np.float64)
    elif n == 4:
        for a in range(units + 1):
            for b in range(units + 1 - a):
                for c in range(units + 1 - a - b):
                    yield np.array([a / units, b / units, c / units, (units - a - b - c) / units], dtype=np.float64)


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(mean_absolute_error(y, pred))


def calibrate_l1(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    # For a fixed scale a, the L1-optimal intercept is median(y - a*x).
    # Search a compact scale range; include negative scales in case a regressor
    # learned inverse ordering on a noisy regression task.
    best = (float("inf"), 1.0, 0.0)
    scales = np.concatenate(
        [
            np.linspace(-1.5, -0.1, 29),
            np.array([0.0]),
            np.linspace(0.1, 2.0, 39),
        ]
    )
    for a in scales:
        b = float(np.median(y - a * pred))
        score = mae(y, a * pred + b)
        if score < best[0]:
            best = (score, float(a), b)
    valid_mae, a, b = best
    return valid_mae, a, b, a * pred + b


def rank_streams(task: str, streams: list[zoo.Stream]) -> list[zoo.Stream]:
    scored = []
    for stream in streams:
        score = mae(stream.valid_y, stream.valid_pred)
        if not math.isnan(score):
            scored.append((score, stream))
    scored.sort(key=lambda x: x[0])
    return [stream for _, stream in scored]


def search_task(task: str, out_root: Path, weight_step: float, max_streams: int) -> dict[str, object]:
    streams = rank_streams(task, zoo.build_streams(task))[:max_streams]
    rows: list[dict[str, object]] = []

    for n in (1, 2, 3, 4):
        if len(streams) < n:
            continue
        for combo in itertools.combinations(streams, n):
            for weights in weight_vectors(n, weight_step):
                valid_pred = sum(weights[i] * combo[i].valid_pred for i in range(n))
                test_pred = sum(weights[i] * combo[i].test_pred for i in range(n))
                raw_valid = mae(combo[0].valid_y, valid_pred)
                raw_test = mae(combo[0].test_y, test_pred)
                cal_valid, a, b, _ = calibrate_l1(combo[0].valid_y, valid_pred)
                cal_test = mae(combo[0].test_y, a * test_pred + b)
                rows.append(
                    {
                        "task": task,
                        "models": " + ".join(s.name for s in combo),
                        "weights": ",".join(f"{float(w):.3f}" for w in weights),
                        "raw_valid_mae": raw_valid,
                        "raw_test_mae": raw_test,
                        "cal_valid_mae": cal_valid,
                        "cal_test_mae": cal_test,
                        "cal_scale": a,
                        "cal_intercept": b,
                        "top1_ref": TASKS[task]["top1_ref"],
                        "cal_beats_top1": cal_test <= TASKS[task]["top1_ref"],
                        "raw_beats_top1": raw_test <= TASKS[task]["top1_ref"],
                        "source": " | ".join(s.source for s in combo),
                    }
                )

    rows.sort(key=lambda r: float(r["cal_valid_mae"]))
    task_dir = out_root / task
    task_dir.mkdir(parents=True, exist_ok=True)
    with (task_dir / "all_candidates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (task_dir / "best_by_valid.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows[:100])
    by_test = sorted(rows, key=lambda r: float(r["cal_test_mae"]))
    with (task_dir / "best_by_test.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(by_test[:100])
    selected = rows[0]
    best_test = by_test[0]
    return {
        "task": task,
        "n_streams": len(streams),
        "selected_models": selected["models"],
        "selected_weights": selected["weights"],
        "selected_raw_valid_mae": selected["raw_valid_mae"],
        "selected_raw_test_mae": selected["raw_test_mae"],
        "selected_cal_valid_mae": selected["cal_valid_mae"],
        "selected_cal_test_mae": selected["cal_test_mae"],
        "selected_cal_scale": selected["cal_scale"],
        "selected_cal_intercept": selected["cal_intercept"],
        "selected_beats_top1": selected["cal_beats_top1"],
        "best_test_models": best_test["models"],
        "best_test_weights": best_test["weights"],
        "best_test_cal_valid_mae": best_test["cal_valid_mae"],
        "best_test_cal_test_mae": best_test["cal_test_mae"],
        "best_test_beats_top1": best_test["cal_beats_top1"],
        "top1_ref": TASKS[task]["top1_ref"],
    }


def main() -> None:
    args = parse_args()
    repo = Path(args.repo)
    zoo.REPO = repo
    zoo.RESULTS = repo / "results_strict"
    zoo.DATA = repo / "data" / "data_benchmark_official_v1"
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for task in args.tasks:
        print(f"[{task}]", flush=True)
        summary.append(search_task(task, out_root, args.weight_step, args.max_streams))
    with (out_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(out_root / "summary.csv")


if __name__ == "__main__":
    main()
