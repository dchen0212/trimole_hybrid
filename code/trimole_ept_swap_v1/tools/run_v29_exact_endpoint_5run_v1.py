from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

import run_strict_5run_seedwise_prediction_zoo_v1 as zoo


REPO = Path("/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1")
STATUS = REPO / "results_strict" / "ept_family_routing_master_v1" / "tdc_current_22_status_v29_transfer_improved_partial.csv"
OUT = REPO / "results_strict" / "v29_exact_endpoint_5run_v1"

TASKS_12 = [
    "cyp2c9_substrate_carbonmangels",
    "vdss_lombardo",
    "cyp2d6_veith",
    "cyp3a4_veith",
    "lipophilicity_astrazeneca",
    "caco2_wang",
    "ld50_zhu",
    "dili",
    "herg",
    "half_life_obach",
    "cyp2c9_veith",
    "bioavailability_ma",
]

EXTRA_SINGLE_SUMMARIES = [
    "paper_main_chemical_prior_xl_v4_all22_32core/summary.csv",
    "paper_main_chemical_prior_xl_v4_remaining4_32core/summary.csv",
    "paper_main_multimodal_prior_taskwise_v1/summary.csv",
    "top1_method_transfer_chem_lite_v1/summary.csv",
    "rank_uplift_tabular_fp_repeated_v1_focus/summary.csv",
]

FIXED_FORMAL_SUMMARIES = [
    "fixed_chemical_endpoint_5run_v1/top1_method_transfer_chem_lite_v1_summary.csv",
    "fixed_chemical_endpoint_5run_v1/paper_main_chemical_prior_xl_v4_all22_32core_summary.csv",
]


# These recipes are the v29 endpoint formulas traced from the v29 status table
# plus the earlier v2 rank-uplift audit tables that generated the v29 scores.
RECIPES: dict[str, dict[str, object]] = {
    "cyp2c9_substrate_carbonmangels": {
        "relation": "v29_rank_uplift_formula_seedwise",
        "v29_score": 0.464920614944,
        "mode": "raw",
        "weights": [0.8, 0.2],
        "models": [
            "ept_family_official_v1_5seed_runs/summary.csv:kpgt/mlp/CrossEntropyLoss",
            "official_metric_loss_cv_promoted_rerun_v1/summary.csv:kpgt/ap_surrogate",
        ],
    },
    "vdss_lombardo": {
        "relation": "v29_rank_uplift_formula_seedwise",
        "v29_score": 0.680968937938,
        "mode": "rank",
        "weights": [0.9, 0.1],
        "models": [
            "ept_family_official_v1_5seed_runs/summary.csv:kpgt_ept/mlp/SpearmanLoss",
            "descriptor_sidecar_official_v1/summary.csv:kpgt_ept/winner_embedding_plus_rdkit_fp",
        ],
    },
    "cyp2d6_veith": {
        "relation": "v29_rank_uplift_formula_seedwise",
        "v29_score": 0.722437765284,
        "mode": "rank",
        "weights": [0.4, 0.3, 0.2, 0.1],
        "models": [
            "ept_family_official_v1_5seed_runs/summary.csv:kpgt_ept/mlp/CrossEntropyLoss",
            "descriptor_sidecar_official_v2/summary.csv:kpgt_ept/winner_embedding_plus_rdkit_fp_plus_base_pred",
            "official_metric_loss_cv_promoted_rerun_v1/summary.csv:kpgt_ept/ap_surrogate",
            "rank_uplift_tabular_fp_only_v1/summary.csv:kpgt_ept/tabular_fp_only/w",
        ],
    },
    "cyp3a4_veith": {
        "relation": "v29_rank_uplift_formula_seedwise",
        "v29_score": 0.885089197381,
        "mode": "logit",
        "weights": [0.2, 0.3, 0.3, 0.2],
        "models": [
            "paper_main_chemical_prior_xl_v4_all22_32core/summary.csv:xl_v4_kpgt_ept/embed_chem_xl_morgan_family/w1.0",
            "ept_family_official_v1_5seed_runs/summary.csv:kpgt_ept/mlp/CrossEntropyLoss",
            "official_metric_loss_cv_promoted_rerun_v1/summary.csv:kpgt_ept/ap_surrogate",
            "rank_uplift_tabular_fp_only_v1/summary.csv:kpgt_ept/tabular_fp_only/w",
        ],
    },
    "lipophilicity_astrazeneca": {
        "relation": "v29_rank_uplift_formula_seedwise",
        "v29_score": 0.470045761325,
        "mode": "raw",
        "weights": [0.2, 0.1, 0.3, 0.4],
        "models": [
            "paper_main_chemical_prior_xl_v4_all22_32core/summary.csv:xl_v4_kpgt/chem_xl_morgan_family/w1.0",
            "rank_uplift_tabular_fp_only_v1/summary.csv:kpgt/tabular_fp_only/w",
            "official_metric_loss_push_all22_v1/summary.csv:kpgt/metric_auto",
            "ept_family_official_v1_5seed_runs/summary.csv:kpgt/mlp/SmoothL1Loss",
        ],
    },
    "caco2_wang": {
        "relation": "v29_chem_lite_single_refit_plus_fixed_formal_5run",
        "v29_score": 0.282088043695,
        "mode": "raw",
        "weights": [1.0],
        "models": [
            "top1_method_transfer_chem_lite_v1/summary.csv:chemberta_kpgt/chem_core_pair_torsion/w1.0",
        ],
        "prefer_fixed_formal": "fixed_selected_chemical_scaffold_foldbag_5run",
    },
    "ld50_zhu": {
        "relation": "v29_prediction_zoo_single_formula_no_exact_seed_std",
        "v29_score": 0.597936091848,
        "mode": "raw",
        "weights": [0.2, 0.8],
        "models": [
            "descriptor_sidecar_official_v2/summary.csv:chemberta_kpgt/winner_embedding_plus_rdkit_fp",
            "paper_main_chemical_prior_xl_v4_all22_32core/summary.csv:xl_v4_chemberta_kpgt/chem_xl_full_chemical_prior/w1.0",
        ],
        "prefer_fixed_formal": "fixed_selected_chemical_scaffold_foldbag_5run",
    },
    "dili": {
        "relation": "v29_rank_uplift_formula_seedwise",
        "v29_score": 0.911739130435,
        "mode": "raw",
        "weights": [0.6, 0.2, 0.2],
        "models": [
            "paper_main_chemical_prior_xl_v4_all22_32core/summary.csv:xl_v4_kpgt_ept/chem_xl_morgan_family/w1.0",
            "ept_family_official_v1_5seed_runs/summary.csv:kpgt_ept/mlp/CrossEntropyLoss",
            "descriptor_sidecar_official_v2/summary.csv:kpgt_ept/winner_embedding_plus_rdkit_fp",
        ],
    },
    "herg": {
        "relation": "v29_prediction_zoo_single_formula_no_exact_seed_std",
        "v29_score": 0.855964653903,
        "mode": "rank",
        "weights": [0.6, 0.4],
        "models": [
            "official_metric_loss_push_all22_v1/summary.csv:ept/auc_margin",
            "rank_uplift_tabular_fp_only_v1/summary.csv:ept/tabular_fp_only/w",
        ],
        "prefer_fixed_formal": "fixed_selected_chemical_scaffold_foldbag_5run",
    },
    "half_life_obach": {
        "relation": "v29_rank_uplift_formula_seedwise",
        "v29_score": 0.491891947657,
        "mode": "zscore",
        "weights": [0.5, 0.1, 0.4],
        "models": [
            "official_metric_loss_push_all22_v1_spearman_fix/summary.csv:chemberta_kpgt/metric_auto",
            "ept_family_official_v1_5seed_runs/summary.csv:chemberta_kpgt/mlp/SpearmanLoss",
            "descriptor_sidecar_official_v1/summary.csv:chemberta_kpgt/winner_embedding_plus_rdkit_fp",
        ],
    },
    "cyp2c9_veith": {
        "relation": "v21_traceable_single_endpoint_plus_fixed_formal_5run",
        "v29_score": 0.780850950253,
        "mode": "raw",
        "weights": [1.0],
        "models": [
            "rank_uplift_tabular_fp_only_v1/summary.csv:kpgt_ept/tabular_fp_only/w",
        ],
        "prefer_fixed_formal": "fixed_selected_chemical_scaffold_foldbag_5run",
    },
    "bioavailability_ma": {
        "relation": "v21_traceable_single_endpoint_plus_fixed_formal_5run",
        "v29_score": 0.722314599268,
        "mode": "raw",
        "weights": [1.0],
        "models": [
            "rank_uplift_tabular_fp_only_v1/summary.csv:kpgt/tabular_fp_only/w",
        ],
        "prefer_fixed_formal": "fixed_selected_chemical_scaffold_foldbag_5run",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--status", default=str(STATUS))
    p.add_argument("--out-root", default=str(OUT))
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_task_meta(status: Path) -> dict[str, dict[str, object]]:
    rows = read_csv(status)
    out: dict[str, dict[str, object]] = {}
    for row in rows:
        if row.get("task") in TASKS_12:
            out[row["task"]] = {
                "metric": row["metric"],
                "direction": row["direction"],
                "top1_ref": float(row["top1_ref"]),
            }
    return out


def task_beats(task: str, value: float) -> bool:
    if math.isnan(value):
        return False
    meta = zoo.TASKS[task]
    if meta["direction"] == "max":
        return value >= float(meta["top1_ref"])
    return value <= float(meta["top1_ref"])


def metric_gap(task: str, value: float) -> float:
    ref = float(zoo.TASKS[task]["top1_ref"])
    if zoo.TASKS[task]["direction"] == "max":
        return ref - value
    return value - ref


def recompute_seedbag_average(task: str, streams: list[zoo.GroupStream], mode: str, weights: np.ndarray) -> tuple[float, float]:
    valid_parts = []
    test_parts = []
    for stream in streams:
        valid_avg = np.mean([p.pred for p in stream.valid], axis=0)
        test_avg = np.mean([p.pred for p in stream.test], axis=0)
        valid_parts.append(zoo.transform(valid_avg, mode))
        test_parts.append(zoo.transform(test_avg, mode))
    valid_pred = sum(float(weights[i]) * valid_parts[i] for i in range(len(streams)))
    test_pred = sum(float(weights[i]) * test_parts[i] for i in range(len(streams)))
    return (
        zoo.score(task, streams[0].valid[0].y, valid_pred),
        zoo.score(task, streams[0].test[0].y, test_pred),
    )


def fixed_formal_rows(task: str, results: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rel in FIXED_FORMAL_SUMMARIES:
        for row in read_csv(results / rel):
            if row.get("task") != task:
                continue
            rows.append(
                {
                    "fixed_formal_source": rel,
                    "fixed_formal_endpoint": row.get("endpoint", ""),
                    "fixed_formal_test_mean": row.get("test_mean", ""),
                    "fixed_formal_test_std": row.get("test_std", ""),
                    "fixed_formal_test_scores": row.get("test_scores", ""),
                    "fixed_formal_valid_mean": row.get("valid_mean", ""),
                    "fixed_formal_valid_std": row.get("valid_std", ""),
                    "fixed_formal_valid_scores": row.get("valid_scores", ""),
                    "fixed_formal_candidate": row.get("candidate", ""),
                    "fixed_formal_variant": row.get("selected_variant", ""),
                    "fixed_formal_backend": row.get("selected_backend", ""),
                    "fixed_formal_topk": row.get("selected_topk", ""),
                    "fixed_formal_weight_sidecar": row.get("weight_sidecar", ""),
                    "fixed_formal_relation": "same_method_formal_5run_not_same_refit_prediction",
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    repo = Path(args.repo)
    status = Path(args.status)
    out_root = Path(args.out_root)
    results = repo / "results_strict"

    zoo.REPO = repo
    zoo.RESULTS = results
    zoo.DATA = repo / "data" / "data_benchmark_official_v1"
    zoo.TASKS = load_task_meta(status)
    for rel in EXTRA_SINGLE_SUMMARIES:
        if rel not in zoo.SINGLE_SUMMARIES:
            zoo.SINGLE_SUMMARIES.append(rel)

    summary: list[dict[str, object]] = []
    per_seed_rows: list[dict[str, object]] = []

    for task in TASKS_12:
        recipe = RECIPES[task]
        print("[task]", task, flush=True)
        streams = zoo.build_streams(task)
        by_name = {stream.name: stream for stream in streams}
        missing = [name for name in recipe["models"] if name not in by_name]  # type: ignore[index]
        fixed_rows = fixed_formal_rows(task, results)

        row: dict[str, object] = {
            "task": task,
            "metric": zoo.TASKS[task]["metric"],
            "direction": zoo.TASKS[task]["direction"],
            "top1_ref": zoo.TASKS[task]["top1_ref"],
            "v29_score": recipe["v29_score"],
            "recipe_relation": recipe["relation"],
            "recipe_mode": recipe["mode"],
            "recipe_weights": ",".join(f"{float(w):.3f}" for w in recipe["weights"]),  # type: ignore[arg-type]
            "recipe_models": " + ".join(recipe["models"]),  # type: ignore[arg-type]
            "missing_models": " | ".join(missing),
            "n_available_streams": len(streams),
        }

        if missing:
            row.update({"status": "missing_recipe_stream"})
            if fixed_rows:
                row.update(fixed_rows[0])
            summary.append(row)
            print("  missing", missing, flush=True)
            continue

        selected_streams = [by_name[name] for name in recipe["models"]]  # type: ignore[index]
        weights = np.asarray(recipe["weights"], dtype=float)  # type: ignore[arg-type]
        mode = str(recipe["mode"])
        n_seed5 = sum(1 for stream in selected_streams if stream.kind == "seed5")

        formal = zoo.eval_combo(task, tuple(selected_streams), mode, weights, lambda_std=1.0)
        avg_valid, avg_test = recompute_seedbag_average(task, selected_streams, mode, weights)
        v29_score = float(recipe["v29_score"])

        relation = str(recipe["relation"])
        if n_seed5 == 0:
            status_name = "exact_singleton_formula_no_real_5seed_std"
        else:
            status_name = "formal_seedwise_formula_from_v29_recipe"

        row.update(
            {
                "status": status_name,
                "stream_kinds": formal["stream_kinds"],
                "n_seed5_streams_in_formula": n_seed5,
                "formal_valid_mean": formal["valid_mean"],
                "formal_valid_std": formal["valid_std"],
                "formal_valid_scores": formal["valid_scores"],
                "formal_test_mean": formal["test_mean"],
                "formal_test_std": formal["test_std"],
                "formal_test_scores": formal["test_scores"],
                "formal_beats_top1": formal["beats_top1_mean"],
                "formal_gap_to_top1": metric_gap(task, float(formal["test_mean"])),
                "v29_recomputed_valid_score": avg_valid,
                "v29_recomputed_test_score": avg_test,
                "v29_recompute_delta": avg_test - v29_score,
                "v29_beats_top1": task_beats(task, v29_score),
                "v29_gap_to_top1": metric_gap(task, v29_score),
            }
        )

        if fixed_rows:
            row.update(fixed_rows[0])
            try:
                fixed_mean = float(fixed_rows[0]["fixed_formal_test_mean"])
                row["fixed_formal_beats_top1"] = task_beats(task, fixed_mean)
                row["fixed_formal_gap_to_top1"] = metric_gap(task, fixed_mean)
            except Exception:
                pass

        test_scores = [float(x) for x in str(formal["test_scores"]).split(";") if x]
        valid_scores = [float(x) for x in str(formal["valid_scores"]).split(";") if x]
        for i, score in enumerate(test_scores, start=1):
            per_seed_rows.append(
                {
                    "task": task,
                    "seed_group": i,
                    "metric": zoo.TASKS[task]["metric"],
                    "test_score": score,
                    "valid_score": valid_scores[i - 1] if i - 1 < len(valid_scores) else "",
                    "status": status_name,
                    "recipe_relation": relation,
                    "stream_kinds": formal["stream_kinds"],
                }
            )

        summary.append(row)
        print(
            "  formal",
            f"{float(formal['test_mean']):.6f}",
            "+/-",
            f"{float(formal['test_std']):.6f}",
            "v29_recomputed",
            f"{avg_test:.6f}",
            "seed5_streams",
            n_seed5,
            flush=True,
        )

    write_csv(out_root / "summary.csv", summary)
    write_csv(out_root / "per_seed_scores.csv", per_seed_rows)
    print(out_root / "summary.csv", flush=True)


if __name__ == "__main__":
    main()
