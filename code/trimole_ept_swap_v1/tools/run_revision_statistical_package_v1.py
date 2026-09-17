from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trimole-manifest", type=Path, required=True)
    parser.add_argument("--controlled-manifest", type=Path, required=True)
    parser.add_argument("--flaml-manifest", type=Path, required=True)
    parser.add_argument("--trimole-selection-evidence", type=Path, required=True)
    parser.add_argument("--flaml-provenance", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--benchmark-table", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser.parse_args()


def run(script: Path, *arguments: object) -> None:
    command = [sys.executable, str(script), *map(str, arguments)]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    tools = Path(__file__).resolve().parent
    s21 = args.out_root / "s21"
    bootstrap = args.out_root / "paired_bootstrap"
    subgroup = args.out_root / "subgroup_uncertainty"
    s22 = args.out_root / "s22"
    figure_stem = args.out_root / "figures" / "Figure2_revision"

    args.out_root.mkdir(parents=True)
    run(
        tools / "build_controlled_comparison_table_v1.py",
        "--trimole-manifest",
        args.trimole_manifest,
        "--controlled-manifest",
        args.controlled_manifest,
        "--flaml-manifest",
        args.flaml_manifest,
        "--trimole-selection-evidence",
        args.trimole_selection_evidence,
        "--flaml-provenance",
        args.flaml_provenance,
        "--data-root",
        args.data_root,
        "--out-root",
        s21,
    )
    run(
        tools / "run_paired_bootstrap_v1.py",
        "--prediction-manifest",
        s21 / "combined_prediction_manifest.csv",
        "--data-root",
        args.data_root,
        "--out-root",
        bootstrap,
        "--reference-model",
        "trimole_hybrid",
        "--bootstrap-replicates",
        args.bootstrap_replicates,
        "--seed",
        args.seed,
        "--n-jobs",
        args.n_jobs,
    )
    run(
        tools / "build_subgroup_uncertainty_v1.py",
        "--prediction-manifest",
        s21 / "combined_prediction_manifest.csv",
        "--data-root",
        args.data_root,
        "--out-root",
        subgroup,
        "--reference-model",
        "trimole_hybrid",
        "--comparator-model",
        "per_task_single",
        "--bootstrap-replicates",
        args.bootstrap_replicates,
        "--seed",
        args.seed,
        "--n-jobs",
        args.n_jobs,
    )
    run(
        tools / "combine_s22_outputs_v1.py",
        "--paired-bootstrap",
        bootstrap / "paired_bootstrap_results.csv",
        "--subgroup-uncertainty",
        subgroup / "Table_S22_subgroup_uncertainty.csv",
        "--out-root",
        s22,
    )
    run(
        tools / "generate_figure2_revision_v1.py",
        "--benchmark-table",
        args.benchmark_table,
        "--controlled-summary",
        s21 / "Table_S21_controlled_baselines_summary.csv",
        "--bootstrap-results",
        bootstrap / "paired_bootstrap_results.csv",
        "--subgroup-table",
        subgroup / "Table_S22_subgroup_uncertainty.csv",
        "--output-stem",
        figure_stem,
        "--seed",
        args.seed,
    )


if __name__ == "__main__":
    main()
