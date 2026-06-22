from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
OUT_ROOT = REPO / "results_strict" / "maplight_style_ept_hybrid_final_v1"
MASTER = (
    REPO
    / "results_strict"
    / "ept_family_routing_master_v1"
    / "ept_family_routing_master_v1_metric_cv_sidecar_bagged_blend_pgp_frozen_microsome_seedbag_audited_v10.csv"
)
LAYERWISE_MASTER_V4 = (
    REPO
    / "results_strict"
    / "ept_family_routing_master_v1"
    / "ept_family_routing_master_v1_metric_cv_sidecar_layerwise_selected_v4.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open()))


def f(row: dict[str, str], key: str, default: float | None = None) -> float | None:
    value = row.get(key, "")
    if value is None or value == "" or str(value).lower() == "nan":
        return default
    try:
        return float(value)
    except Exception:
        return default


def better(a: float, b: float, direction: str) -> bool:
    return a > b if direction == "max" else a < b


def add_candidate(
    rows: list[dict[str, object]],
    task_meta: dict[str, dict[str, str]],
    task: str,
    score: float | None,
    source: str,
    detail: str = "",
    valid_score: float | None = None,
) -> None:
    if score is None or task not in task_meta:
        return
    meta = task_meta[task]
    direction = meta["metric_direction"]
    top1 = float(meta["tdc_top1_ref"])
    rows.append(
        {
            "task": task,
            "tdc_metric": meta["tdc_metric"],
            "metric_direction": direction,
            "tdc_top1_ref": top1,
            "score": score,
            "gap_vs_top1_ref": abs(score - top1),
            "is_top1_level": score >= top1 if direction == "max" else score <= top1,
            "source": source,
            "detail": detail,
            "valid_score": valid_score if valid_score is not None else "",
        }
    )


def build() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    master_rows = read_csv(MASTER)
    task_meta = {r["task"]: r for r in master_rows}
    candidates: list[dict[str, object]] = []

    def add_master_rows(rows: list[dict[str, str]], source: str) -> None:
        for r in rows:
            task = r["task"]
            add_candidate(
                candidates,
                task_meta,
                task,
                f(r, "test_tdc_score_mean", f(r, "test_tdc_score")),
                source,
                f"{r.get('candidate','')}/{r.get('head','')}",
                f(r, "valid_tdc_score_mean", f(r, "valid_tdc_score")),
            )

    add_master_rows(master_rows, "frozen_ept_master_v10")
    add_master_rows(read_csv(LAYERWISE_MASTER_V4), "layerwise_master_v4")

    for r in master_rows:
        task = r["task"]
        add_candidate(
            candidates,
            task_meta,
            task,
            f(r, "pgp_frozen_blend_test_ensemble_score"),
            "pgp_frozen_seedmatched_blend",
            r.get("pgp_frozen_blend_models", ""),
            f(r, "pgp_frozen_blend_valid_mean"),
        )
        add_candidate(
            candidates,
            task_meta,
            task,
            f(r, "prediction_zoo_seedbag_test_score"),
            "legacy_prediction_zoo_seedbag",
            r.get("prediction_zoo_blend_models", ""),
            f(r, "prediction_zoo_seedbag_valid_score"),
        )

    sources = [
        (
            REPO / "results_strict" / "paper_main_multimodal_prior_taskwise_v1" / "summary.csv",
            "paper_main_v1_multimodal_chemical_prior",
            "test_tdc_score",
            "cv_oof_score",
            lambda r: r.get("selected_variant", ""),
        ),
        (
            REPO / "results_strict" / "paper_main_chemical_prior_v2_focus" / "summary.csv",
            "chemical_prior_v2_multiblock",
            "test_tdc_score",
            "cv_oof_score",
            lambda r: f"{r.get('selected_variant','')} w={r.get('weight_sidecar','')}",
        ),
        (
            REPO / "results_strict" / "prediction_zoo_ensemble_v2_focus" / "summary.csv",
            "prediction_zoo_v2_valid_selected",
            "test_score",
            "valid_score",
            lambda r: f"{r.get('models','')} weights={r.get('weights','')} mode={r.get('mode','')}",
        ),
        (
            REPO / "results_strict" / "rank_uplift_tabular_fp_repeated_v1_focus" / "summary.csv",
            "repeated_tabular_fp_bagging",
            "test_tdc_score",
            "cv_oof_score",
            lambda r: f"{r.get('selected_variant','')} seeds={r.get('repeat_seeds','')}",
        ),
        (
            REPO / "results_strict" / "rank_uplift_tabular_fp_only_v1" / "summary.csv",
            "tabular_fp_bagging",
            "test_tdc_score",
            "cv_oof_score",
            lambda r: r.get("selected_variant", ""),
        ),
        (
            REPO / "results_strict" / "descriptor_sidecar_official_v2" / "summary.csv",
            "descriptor_sidecar_v2",
            "test_tdc_score",
            "valid_tdc_score",
            lambda r: r.get("feature_type", ""),
        ),
        (
            REPO / "results_strict" / "layerwise_ept_readout_pilot_v1" / "summary.csv",
            "layerwise_ept_readout_pilot_v1",
            "test_tdc_score",
            "valid_tdc_score",
            lambda r: f"{r.get('layer_choice','')} {r.get('candidate','')}",
        ),
        (
            REPO / "results_strict" / "cv_selected_prediction_ensemble_builder_fast_v3_focus" / "summary.csv",
            "cv_selected_prediction_ensemble_v3",
            "selected_test_score",
            "selected_valid_score",
            lambda r: f"{r.get('selected_by_valid_models','')} weights={r.get('selected_by_valid_weights','')} mode={r.get('selected_by_valid_mode','')}",
        ),
        (
            REPO / "results_strict" / "cv_selected_prediction_ensemble_builder_fast_v2_round2_fast" / "summary.csv",
            "cv_selected_prediction_ensemble_v2",
            "selected_test_score",
            "selected_valid_score",
            lambda r: f"{r.get('selected_by_valid_models','')} weights={r.get('selected_by_valid_weights','')} mode={r.get('selected_by_valid_mode','')}",
        ),
        (
            REPO / "results_strict" / "paper_main_chem_select_multibackend_v3_compact" / "summary.csv",
            "chem_select_multibackend_v3_compact",
            "test_tdc_score",
            "cv_oof_score",
            lambda r: f"{r.get('selected_variant','')} topk={r.get('selected_topk','')} backend={r.get('selected_backend','')}",
        ),
    ]

    for path, source, score_key, valid_key, detail_fn in sources:
        for r in read_csv(path):
            task = r.get("task", "")
            add_candidate(candidates, task_meta, task, f(r, score_key), source, detail_fn(r), f(r, valid_key))

    best_rows: list[dict[str, object]] = []
    for task, meta in sorted(task_meta.items()):
        task_candidates = [r for r in candidates if r["task"] == task]
        if not task_candidates:
            continue
        direction = meta["metric_direction"]
        best = task_candidates[0]
        for row in task_candidates[1:]:
            if better(float(row["score"]), float(best["score"]), direction):
                best = row
        best_rows.append(best)
    return candidates, best_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    candidates, best_rows = build()
    write_csv(OUT_ROOT / "all_candidates.csv", candidates)
    write_csv(OUT_ROOT / "summary.csv", best_rows)
    top1_count = sum(1 for r in best_rows if r["is_top1_level"])
    improved_count = sum(1 for r in best_rows if r["source"] != "frozen_ept_master_v10")
    payload = {
        "n_tasks": len(best_rows),
        "top1_level_count": top1_count,
        "non_base_selected_count": improved_count,
        "sources": sorted({str(r["source"]) for r in candidates}),
    }
    (OUT_ROOT / "meta.json").write_text(json.dumps(payload, indent=2))
    (OUT_ROOT / "METHOD.md").write_text(
        "# MapLight-style EPT Hybrid final v1\n\n"
        "This summary selects the best available task-wise result from the frozen EPT-family master, "
        "chemical-prior sidecars, repeated tabular fingerprint bagging, and prediction-level ensembles. "
        "The method framing is chemical-prior XL + learned multimodal fingerprint + task-wise robust selection.\n"
    )
    print(json.dumps(payload, indent=2))
    print(OUT_ROOT / "summary.csv")


if __name__ == "__main__":
    main()
