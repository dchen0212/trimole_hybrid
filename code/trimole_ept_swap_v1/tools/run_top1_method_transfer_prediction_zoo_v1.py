from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import run_strict_5run_seedwise_prediction_zoo_v1 as zoo


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
STATUS = REPO / "results_strict" / "ept_family_routing_master_v1" / "tdc_current_22_status_v27_from_v10_v23.csv"
OUT = REPO / "results_strict" / "top1_method_transfer_prediction_zoo_v1"

EXTRA_SINGLE_SUMMARIES = [
    "paper_main_multimodal_prior_taskwise_v1/summary.csv",
    "paper_main_chem_select_multibackend_v3/summary.csv",
    "paper_main_chemical_prior_xl_v4_all22_32core/summary.csv",
    "paper_main_chemical_prior_xl_v4_remaining4_32core/summary.csv",
    "xl_v4_conservative_selector_all22_v3/summary.csv",
    "xl_v4_metric_calibrated_blend_all22_v2/summary.csv",
    "rank_uplift_tabular_fp_repeated_v1_focus/summary.csv",
    "top1_method_transfer_chem_select_v1/summary.csv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--status", default=str(STATUS))
    p.add_argument("--out-root", default=str(OUT))
    p.add_argument("--tasks", nargs="*", default=[])
    p.add_argument("--weight-step", type=float, default=0.1)
    p.add_argument("--max-streams", type=int, default=14)
    p.add_argument("--lambda-std", type=float, default=1.0)
    return p.parse_args()


def load_task_meta(status: Path, explicit_tasks: list[str]) -> dict[str, dict[str, object]]:
    with status.open() as f:
        rows = list(csv.DictReader(f))
    if explicit_tasks:
        wanted = set(explicit_tasks)
        rows = [r for r in rows if r["task"] in wanted]
    else:
        rows = [r for r in rows if str(r.get("top1", "")).lower() != "true"]
    out: dict[str, dict[str, object]] = {}
    for r in rows:
        out[r["task"]] = {
            "metric": r["metric"],
            "direction": r["direction"],
            "top1_ref": float(r["top1_ref"]),
        }
    return out


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


def main() -> None:
    args = parse_args()
    repo = Path(args.repo)
    zoo.REPO = repo
    zoo.RESULTS = repo / "results_strict"
    zoo.DATA = repo / "data" / "data_benchmark_official_v1"
    zoo.TASKS = load_task_meta(Path(args.status), args.tasks)
    for rel in EXTRA_SINGLE_SUMMARIES:
        if rel not in zoo.SINGLE_SUMMARIES:
            zoo.SINGLE_SUMMARIES.append(rel)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

    for task in zoo.TASKS:
        print("[task]", task, flush=True)
        streams = zoo.build_streams(task)
        rows: list[dict[str, object]] = []
        modes: tuple[str, ...] = ("raw", "zscore", "rank")
        metric = str(zoo.TASKS[task]["metric"])
        if metric in {"AUROC", "AUPRC"}:
            modes = ("raw", "zscore", "rank", "logit")
        if metric == "MAE":
            modes = ("raw",)

        singles = [zoo.eval_combo(task, (s,), "raw", np.array([1.0]), args.lambda_std) for s in streams]
        reverse = zoo.TASKS[task]["direction"] == "max"
        singles_sorted = sorted(singles, key=lambda r: float(r["valid_adjusted"]), reverse=reverse)
        keep_names = {r["models"] for r in singles_sorted[: args.max_streams]}
        kept = [s for s in streams if s.name in keep_names]
        rows.extend(singles)

        for n in (2, 3):
            if len(kept) < n:
                continue
            import itertools

            for combo in itertools.combinations(kept, n):
                for mode in modes:
                    for weights in weight_vectors(n, args.weight_step):
                        if n > 1 and np.max(weights) >= 0.999:
                            continue
                        rows.append(zoo.eval_combo(task, combo, mode, weights, args.lambda_std))

        task_out = out_root / task
        task_out.mkdir(parents=True, exist_ok=True)
        by_valid = sorted(rows, key=lambda r: float(r["valid_adjusted"]), reverse=reverse)
        by_test = sorted(rows, key=lambda r: float(r["test_mean"]), reverse=reverse)
        zoo.write_csv(task_out / "all_results.csv", rows)
        zoo.write_csv(task_out / "best_by_valid_adjusted.csv", by_valid[:100])
        zoo.write_csv(task_out / "best_by_test_mean_diagnostic.csv", by_test[:100])
        selected = by_valid[0] if by_valid else {}
        best_test = by_test[0] if by_test else {}
        summary.append(
            {
                "task": task,
                "metric": zoo.TASKS[task]["metric"],
                "top1_ref": zoo.TASKS[task]["top1_ref"],
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
            }
        )
        print(
            task,
            "streams",
            len(streams),
            "selected",
            selected.get("test_mean", ""),
            "best_diag",
            best_test.get("test_mean", ""),
            flush=True,
        )

    zoo.write_csv(out_root / "summary.csv", summary)
    print(out_root / "summary.csv", flush=True)


if __name__ == "__main__":
    main()
