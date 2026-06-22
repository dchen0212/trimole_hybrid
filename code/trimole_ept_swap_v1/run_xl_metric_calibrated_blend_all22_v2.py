from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

import cv_selected_prediction_ensemble_builder_fast_v2 as base


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
RESULTS = REPO / "results_strict"
OUT_ROOT = RESULTS / "xl_v4_metric_calibrated_blend_all22_v2"

ALL22_TASKS = [
    "caco2_wang",
    "hia_hou",
    "pgp_broccatelli",
    "bioavailability_ma",
    "lipophilicity_astrazeneca",
    "solubility_aqsoldb",
    "bbb_martins",
    "ppbr_az",
    "vdss_lombardo",
    "cyp2c9_veith",
    "cyp2d6_veith",
    "cyp3a4_veith",
    "cyp2c9_substrate_carbonmangels",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "half_life_obach",
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
    "herg",
    "ames",
    "dili",
    "ld50_zhu",
]

EXTRA_TASKS = {
    "pgp_broccatelli": {"metric": "AUROC", "direction": "max", "top1_ref": 0.938},
    "bioavailability_ma": {"metric": "AUROC", "direction": "max", "top1_ref": 0.942},
    "cyp2c9_substrate_carbonmangels": {"metric": "AUPRC", "direction": "max", "top1_ref": 0.474},
    "ld50_zhu": {"metric": "MAE", "direction": "min", "top1_ref": 0.552},
}

EXTRA_SUMMARIES = [
    "paper_main_chemical_prior_xl_v4_all22_32core/summary.csv",
    "paper_main_chemical_prior_xl_v4_remaining4_32core/summary.csv",
    "rank_uplift_tabular_fp_repeated_v1_focus/summary.csv",
    "cv_selected_prediction_ensemble_builder_fast_v4_rank_batch/summary.csv",
    "cv_selected_prediction_ensemble_builder_fast_v4_rank_batch_mae_raw/summary.csv",
]


def write_xl_summary(root: Path) -> None:
    rows = []
    for path in sorted(root.glob("*/result.json")):
        result = json.load(open(path))
        rows.append(
            {
                "task": result.get("task", path.parent.name),
                "candidate": f"xl_v4_{result.get('candidate', '')}",
                "head": result.get("head", ""),
                "tdc_metric": result.get("tdc_metric", ""),
                "metric_direction": result.get("metric_direction", ""),
                "selected_variant": result.get("selected_variant", ""),
                "selected_topk": result.get("selected_topk", ""),
                "selected_backend": result.get("selected_backend", ""),
                "weight_sidecar": result.get("weight_sidecar", ""),
                "cv_mean": result.get("cv_mean", ""),
                "cv_std": result.get("cv_std", ""),
                "test_tdc_score": result.get("test_tdc_score", ""),
                "incumbent_test_tdc_score": result.get("incumbent_test_tdc_score", ""),
                "improved_test": result.get("improved_test", ""),
                "tdc_top1_ref": result.get("tdc_top1_ref", ""),
                "is_top1_level": result.get("is_top1_level", ""),
                "trainval_pred_file": result.get("trainval_pred_file", ""),
                "test_pred_file": result.get("test_pred_file", ""),
                "endpoint": result.get("endpoint", ""),
            }
        )
    if not rows:
        return
    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def direction(task: str) -> str:
    return base.TASKS[task]["direction"]


def is_better(task: str, new_value: float, old_value: float) -> bool:
    if math.isnan(new_value):
        return False
    return new_value > old_value if direction(task) == "max" else new_value < old_value


def top_rows_by(task: str, rows: list[dict[str, object]], key: str, n: int) -> list[dict[str, object]]:
    return base.sort_rows(task, rows, key)[:n]


def weights(n: int, step: float):
    units = int(round(1.0 / step))
    if n == 1:
        yield np.array([1.0], dtype=float)
    elif n == 2:
        for a in range(units + 1):
            yield np.array([a / units, 1.0 - a / units], dtype=float)
    elif n == 3:
        for a in range(units + 1):
            for b in range(units + 1 - a):
                yield np.array([a / units, b / units, (units - a - b) / units], dtype=float)


def affine_calibrate(valid_pred: np.ndarray, valid_y: np.ndarray, test_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(valid_pred, dtype=float)
    y = np.asarray(valid_y, dtype=float)
    A = np.column_stack([x, np.ones_like(x)])
    try:
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    except Exception:
        a, b = 1.0, 0.0
    return a * x + b, a * np.asarray(test_pred, dtype=float) + b


def isotonic_calibrate(valid_pred: np.ndarray, valid_y: np.ndarray, test_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(valid_pred, dtype=float)
    y = np.asarray(valid_y, dtype=float)
    # Isotonic requires monotonic x ordering but handles duplicate x internally.
    increasing = np.corrcoef(x, y)[0, 1]
    increasing = True if np.isnan(increasing) else bool(increasing >= 0)
    try:
        iso = IsotonicRegression(out_of_bounds="clip", increasing=increasing)
        iso.fit(x, y)
        return iso.predict(x), iso.predict(np.asarray(test_pred, dtype=float))
    except Exception:
        return valid_pred, test_pred


def candidate_modes(task: str) -> list[str]:
    metric = base.TASKS[task]["metric"]
    if metric == "MAE":
        return ["raw", "affine", "isotonic"]
    if metric == "Spearman":
        return ["rank", "zscore", "raw", "rank_affine"]
    if metric in {"AUROC", "AUPRC"}:
        return ["logit", "rank", "zscore", "raw"]
    return ["raw"]


def transform_pair(task: str, valid_pred: np.ndarray, test_pred: np.ndarray, mode: str, valid_y: np.ndarray):
    if mode == "affine":
        return affine_calibrate(valid_pred, valid_y, test_pred)
    if mode == "isotonic":
        return isotonic_calibrate(valid_pred, valid_y, test_pred)
    if mode == "rank_affine":
        v = base.transform(valid_pred, "rank")
        t = base.transform(test_pred, "rank")
        return affine_calibrate(v, valid_y, t)
    return base.transform(valid_pred, mode), base.transform(test_pred, mode)


def disagreement(arrays: list[np.ndarray]) -> float:
    if len(arrays) <= 1:
        return 0.0
    stacked = np.vstack([base.transform(x, "rank") for x in arrays])
    return float(np.nanmean(np.nanstd(stacked, axis=0)))


def adjusted_score(task: str, valid_score: float, dis: float, lambda_dis: float = 0.01) -> float:
    if math.isnan(valid_score):
        return valid_score
    if direction(task) == "max":
        return valid_score - lambda_dis * dis
    return valid_score + lambda_dis * dis


def build_candidates(task: str, streams: list[base.Stream], step: float, max_streams: int) -> list[dict[str, object]]:
    singles = base.single_rows(task, streams)
    stream_by_name = {s.name: s for s in streams}
    pool_names = [r["models"] for r in top_rows_by(task, singles, "valid_score", max_streams)]
    pool = [stream_by_name[name] for name in pool_names if name in stream_by_name]

    rows: list[dict[str, object]] = []
    for row in singles:
        row = dict(row)
        row["selector_adjusted_score"] = adjusted_score(task, float(row["valid_score"]), 0.0)
        row["disagreement"] = 0.0
        rows.append(row)

    for n in (2, 3):
        if len(pool) < n:
            continue
        for combo in itertools.combinations(pool, n):
            model_names = " + ".join(s.name for s in combo)
            member_text = ",".join(str(s.n_members) for s in combo)
            source = " | ".join(s.source for s in combo)
            for mode in candidate_modes(task):
                transformed = [
                    transform_pair(task, s.valid_pred, s.test_pred, mode, s.valid_y)
                    for s in combo
                ]
                valid_parts = [x[0] for x in transformed]
                test_parts = [x[1] for x in transformed]
                dis = disagreement(valid_parts)
                for w in weights(n, step):
                    valid_pred = sum(w[i] * valid_parts[i] for i in range(n))
                    test_pred = sum(w[i] * test_parts[i] for i in range(n))
                    valid_score = base.score(task, combo[0].valid_y, valid_pred)
                    test_score = base.score(task, combo[0].test_y, test_pred)
                    rows.append(
                        {
                            "task": task,
                            "models": model_names,
                            "mode": mode,
                            "weights": ",".join(f"{float(x):.3f}" for x in w),
                            "n_members": member_text,
                            "valid_score": valid_score,
                            "test_score": test_score,
                            "selector_adjusted_score": adjusted_score(task, valid_score, dis),
                            "beats_top1": base.beats(task, test_score),
                            "disagreement": dis,
                            "source": source,
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for directory in [
        RESULTS / "paper_main_chemical_prior_xl_v4_all22_32core",
        RESULTS / "paper_main_chemical_prior_xl_v4_remaining4_32core",
    ]:
        write_xl_summary(directory)

    base.TASKS.update(EXTRA_TASKS)
    for summary in EXTRA_SUMMARIES:
        if summary not in base.PRED_SUMMARIES:
            base.PRED_SUMMARIES.append(summary)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for task in ALL22_TASKS:
        streams = base.build_streams(task)
        candidates = build_candidates(task, streams, step=0.05, max_streams=12)
        task_dir = OUT_ROOT / task
        write_csv(task_dir / "all_candidates.csv", candidates)
        by_valid = top_rows_by(task, candidates, "valid_score", 100)
        by_adjusted = top_rows_by(task, candidates, "selector_adjusted_score", 100)
        by_test = top_rows_by(task, candidates, "test_score", 100)
        write_csv(task_dir / "best_by_valid.csv", by_valid)
        write_csv(task_dir / "best_by_adjusted.csv", by_adjusted)
        write_csv(task_dir / "best_by_test.csv", by_test)

        selected = by_adjusted[0] if by_adjusted else {}
        best_test = by_test[0] if by_test else {}
        summary_rows.append(
            {
                "task": task,
                "metric": base.TASKS[task]["metric"],
                "top1_ref": base.TASKS[task]["top1_ref"],
                "n_streams": len(streams),
                "n_candidates": len(candidates),
                "selected_models": selected.get("models", ""),
                "selected_mode": selected.get("mode", ""),
                "selected_weights": selected.get("weights", ""),
                "selected_valid_score": selected.get("valid_score", ""),
                "selected_adjusted_score": selected.get("selector_adjusted_score", ""),
                "selected_test_score": selected.get("test_score", ""),
                "selected_beats_top1": selected.get("beats_top1", ""),
                "selected_disagreement": selected.get("disagreement", ""),
                "best_test_models": best_test.get("models", ""),
                "best_test_mode": best_test.get("mode", ""),
                "best_test_weights": best_test.get("weights", ""),
                "best_test_valid_score": best_test.get("valid_score", ""),
                "best_test_adjusted_score": best_test.get("selector_adjusted_score", ""),
                "best_test_score": best_test.get("test_score", ""),
                "best_test_beats_top1": best_test.get("beats_top1", ""),
            }
        )
        print(
            task,
            "streams",
            len(streams),
            "candidates",
            len(candidates),
            "selected_test",
            selected.get("test_score", ""),
            flush=True,
        )
    write_csv(OUT_ROOT / "summary.csv", summary_rows)
    print(OUT_ROOT / "summary.csv")


if __name__ == "__main__":
    main()
