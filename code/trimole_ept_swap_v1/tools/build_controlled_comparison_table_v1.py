from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_paired_bootstrap_v1 import load_prediction, load_test_labels, metric_value


CONTROL_MODELS = {
    "global_single",
    "per_task_single",
    "validation_top2_average",
    "validation_top3_average",
    "uniform_average",
    "oof_stacking",
}
CLASSIFICATION_METRICS = {"AUROC", "AUPRC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trimole-manifest", type=Path, required=True)
    parser.add_argument("--controlled-manifest", type=Path, required=True)
    parser.add_argument("--flaml-manifest", type=Path, required=True)
    parser.add_argument("--trimole-selection-evidence", type=Path, required=True)
    parser.add_argument("--flaml-provenance", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    return parser.parse_args()


def normalize_manifests(
    trimole: pd.DataFrame,
    controlled: pd.DataFrame,
    flaml: pd.DataFrame,
) -> pd.DataFrame:
    required = {"task", "metric", "seed", "prediction_file"}
    for name, frame in (("trimole", trimole), ("controlled", controlled), ("flaml", flaml)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} manifest missing columns: {sorted(missing)}")
    if "model" not in trimole:
        raise ValueError("Trimole manifest lacks model column")
    if "control" not in controlled:
        raise ValueError("controlled manifest lacks control column")

    trimole = trimole[list(required) + ["model"]].copy()
    trimole["source_family"] = "submitted_taskwise_system"
    controlled = controlled[list(required) + ["control"]].rename(
        columns={"control": "model"}
    )
    controlled["source_family"] = "controlled_three_view_pool"
    flaml = flaml.copy()
    flaml["model"] = "flaml_automl"
    flaml["source_family"] = "matched_budget_automl"
    return pd.concat(
        [
            trimole,
            controlled[list(required) + ["model", "source_family"]],
            flaml[list(required) + ["model", "source_family"]],
        ],
        ignore_index=True,
    )


def validate_grid(manifest: pd.DataFrame) -> None:
    tasks = sorted(manifest.task.unique())
    if len(tasks) != 22:
        raise ValueError(f"expected 22 tasks, found {len(tasks)}")
    observed_controls = set(
        manifest.loc[manifest.source_family == "controlled_three_view_pool", "model"]
    )
    if observed_controls != CONTROL_MODELS:
        raise ValueError(
            f"controlled method mismatch: missing={sorted(CONTROL_MODELS-observed_controls)}, "
            f"extra={sorted(observed_controls-CONTROL_MODELS)}"
        )
    for (task, model), rows in manifest.groupby(["task", "model"]):
        expected = 1 if task == "bbb_martins" and model == "trimole_hybrid" else 5
        if len(rows) != expected or rows.seed.nunique() != expected:
            raise ValueError(f"unexpected run count for {task}/{model}: {len(rows)}")


def main() -> None:
    args = parse_args()
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    trimole = pd.read_csv(args.trimole_manifest)
    controlled = pd.read_csv(args.controlled_manifest)
    flaml = pd.read_csv(args.flaml_manifest)
    manifest = normalize_manifests(trimole, controlled, flaml)
    validate_grid(manifest)

    selection = pd.read_csv(args.trimole_selection_evidence)
    trimole_counts = dict(
        zip(selection.task, selection.formal_variant_count_in_checked_audit)
    )
    flaml_provenance = json.loads(args.flaml_provenance.read_text())
    flaml_time = flaml_provenance["time_budget_seconds_per_task_seed"]
    flaml_iter = flaml_provenance["max_iterations_per_task_seed"]

    rows: list[dict[str, object]] = []
    label_cache: dict[str, np.ndarray] = {}
    for item in manifest.to_dict("records"):
        task, metric, model = str(item["task"]), str(item["metric"]), str(item["model"])
        if task not in label_cache:
            label_cache[task] = load_test_labels(args.data_root, task)
        labels = label_cache[task]
        prediction = load_prediction(item["prediction_file"], len(labels))
        if model == "trimole_hybrid":
            candidate_count = int(trimole_counts[task])
            budget = "historical validation-only search; six formal variants audited"
            prediction_evidence_note = (
                "five-run aggregate scores archived; one five-model ensemble prediction array archived"
                if task == "bbb_martins"
                else "five seed-level prediction arrays archived"
            )
        elif model == "flaml_automl":
            candidate_count = 4 if metric in CLASSIFICATION_METRICS else 3
            budget = f"{flaml_time:g} seconds and <= {flaml_iter} iterations per task/seed"
            prediction_evidence_note = "five seed-level prediction arrays generated in revision"
        else:
            candidate_count = 3
            budget = "fixed three-view pool; fixed linear learner; no adaptive time search"
            prediction_evidence_note = "five seed-level prediction arrays generated in revision"
        rows.append(
            {
                **item,
                "n_samples": len(labels),
                "recomputed_test_score": metric_value(metric, labels, prediction),
                "candidate_count": candidate_count,
                "candidate_count_definition": (
                    "formal audited variants" if model == "trimole_hybrid" else "available estimators/views"
                ),
                "selection_budget": budget,
                "prediction_evidence_note": prediction_evidence_note,
            }
        )

    table = pd.DataFrame(rows).sort_values(["task", "model", "seed"])
    if not np.isfinite(table.recomputed_test_score.to_numpy(dtype=float)).all():
        raise ValueError("non-finite recomputed score")
    summary = table.groupby(
        [
            "task",
            "metric",
            "model",
            "source_family",
            "candidate_count",
            "candidate_count_definition",
            "selection_budget",
            "prediction_evidence_note",
        ],
        as_index=False,
    ).agg(
        n_runs=("seed", "nunique"),
        n_samples=("n_samples", "first"),
        score_mean=("recomputed_test_score", "mean"),
        score_sd=("recomputed_test_score", "std"),
    )

    args.out_root.mkdir(parents=True)
    table.to_csv(args.out_root / "Table_S21_controlled_baselines_by_seed.csv", index=False)
    summary.to_csv(args.out_root / "Table_S21_controlled_baselines_summary.csv", index=False)
    table[["task", "metric", "model", "seed", "prediction_file"]].to_csv(
        args.out_root / "combined_prediction_manifest.csv", index=False
    )
    (args.out_root / "provenance.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "policy": "all test scores recomputed from sample-level predictions against frozen official labels",
                "trimole_manifest": str(args.trimole_manifest),
                "controlled_manifest": str(args.controlled_manifest),
                "flaml_manifest": str(args.flaml_manifest),
                "tasks": 22,
                "controlled_models": sorted(CONTROL_MODELS),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
