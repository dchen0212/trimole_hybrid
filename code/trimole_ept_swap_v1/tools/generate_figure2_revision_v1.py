#!/usr/bin/env python3
"""Generate the corrected four-panel benchmark and uncertainty figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


COLORS = {
    "Absorption": "#3978A8",
    "Distribution": "#D79B36",
    "Metabolism": "#2B8C82",
    "Excretion": "#67A77A",
    "Toxicity": "#C96855",
}
INK = "#18232D"
MUTED = "#667681"
MODEL_LABELS = {
    "trimole_hybrid": "Trimole-Hybrid",
    "flaml_automl": "FLAML AutoML",
    "global_single": "Global single",
    "per_task_single": "Per-task single",
    "validation_top2_average": "Validation top-2",
    "validation_top3_average": "Validation top-3",
    "uniform_average": "Uniform average",
    "oof_stacking": "OOF stacking",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-table", type=Path, required=True)
    parser.add_argument("--controlled-summary", type=Path, required=True)
    parser.add_argument("--bootstrap-results", type=Path, required=True)
    parser.add_argument("--subgroup-table", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=13, weight="bold")
    ax.text(-0.04, 1.04, title, transform=ax.transAxes, fontsize=10.5, weight="bold", color=INK)


def task_labels(benchmark: pd.DataFrame) -> dict[str, str]:
    task_order = [
        "bioavailability_ma", "caco2_wang", "hia_hou", "lipophilicity_astrazeneca",
        "pgp_broccatelli", "solubility_aqsoldb", "bbb_martins", "ppbr_az",
        "vdss_lombardo", "cyp2c9_substrate_carbonmangels", "cyp2c9_veith",
        "cyp2d6_substrate_carbonmangels", "cyp2d6_veith",
        "cyp3a4_substrate_carbonmangels", "cyp3a4_veith",
        "clearance_hepatocyte_az", "clearance_microsome_az", "half_life_obach",
        "ames", "dili", "herg", "ld50_zhu",
    ]
    if len(benchmark) != len(task_order):
        raise ValueError("benchmark table must contain 22 rows")
    return dict(zip(task_order, benchmark["Dataset"].astype(str)))


def draw_public_margin(ax: plt.Axes, benchmark: pd.DataFrame) -> None:
    data = benchmark.copy()
    data["Margin"] = pd.to_numeric(data["Margin"])
    data = data.sort_values("Margin")
    y = np.arange(len(data))
    colors = [COLORS[str(value)] for value in data.Category]
    ax.barh(y, data.Margin, color=colors, height=0.72, alpha=0.92)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(y, data.Dataset, fontsize=7.2)
    ax.set_xlabel("Direction-normalized margin vs frozen public top-1", fontsize=8.5)
    ax.grid(axis="x", color="#D8DEE2", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    panel_label(ax, "a", "Corrected public-reference margins (descriptive)")


def normalized_task_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    data = summary.copy()
    direction = data.metric.map(
        {"AUROC": 1.0, "AUPRC": 1.0, "Spearman": 1.0, "MAE": -1.0}
    )
    if direction.isna().any():
        raise ValueError("unsupported metric in controlled summary")
    data["utility"] = data.score_mean * direction
    data["rank"] = data.groupby("task").utility.rank(method="average", ascending=False)
    counts = data.groupby("task").model.transform("count")
    data["normalized_rank_utility"] = 1.0 - (data["rank"] - 1.0) / (counts - 1.0)
    return data


def draw_controlled(ax: plt.Axes, summary: pd.DataFrame, rng: np.random.Generator) -> dict[str, float]:
    ranked = normalized_task_ranks(summary)
    order = (
        ranked.groupby("model").normalized_rank_utility.mean().sort_values().index.tolist()
    )
    rows = []
    for model in order:
        values = ranked.loc[ranked.model.eq(model), "normalized_rank_utility"].to_numpy()
        boot = np.mean(rng.choice(values, size=(10_000, len(values)), replace=True), axis=1)
        rows.append((model, values.mean(), *np.quantile(boot, [0.025, 0.975])))
    result = pd.DataFrame(rows, columns=["model", "mean", "lower", "upper"])
    y = np.arange(len(result))
    colors = ["#C96855" if model == "trimole_hybrid" else "#3978A8" for model in result.model]
    ax.barh(y, result["mean"], color=colors, alpha=0.9, height=0.68)
    ax.errorbar(
        result["mean"], y,
        xerr=np.vstack([result["mean"] - result.lower, result.upper - result["mean"]]),
        fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.2,
    )
    ax.set_yticks(y, [MODEL_LABELS.get(value, value) for value in result.model], fontsize=7.4)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Mean within-task normalized rank utility (95% task-bootstrap CI)", fontsize=8.2)
    ax.grid(axis="x", color="#D8DEE2", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    panel_label(ax, "b", "Matched controls on identical official splits")
    return dict(zip(result.model, result["mean"]))


def draw_paired_ci(ax: plt.Axes, bootstrap: pd.DataFrame, labels: dict[str, str]) -> dict[str, int]:
    data = bootstrap[
        bootstrap.comparator_model.eq("per_task_single")
        & bootstrap.reference_model.eq("trimole_hybrid")
    ].copy()
    if len(data) != 22:
        raise ValueError(f"expected 22 paired task rows, found {len(data)}")
    scale = data.comparator_score.abs().clip(lower=1e-12)
    data["relative_improvement"] = data.improvement / scale
    data["relative_ci95_lower"] = data.ci95_lower / scale
    data["relative_ci95_upper"] = data.ci95_upper / scale
    data["label"] = data.task.map(labels)
    data = data.sort_values("relative_improvement")
    y = np.arange(len(data))
    significant = data.significant_fdr_0_05.astype(bool)
    colors = np.where(significant, "#C96855", "#3978A8")
    ax.errorbar(
        data.relative_improvement,
        y,
        xerr=np.vstack(
            [
                data.relative_improvement - data.relative_ci95_lower,
                data.relative_ci95_upper - data.relative_improvement,
            ]
        ),
        fmt="none",
        ecolor=MUTED,
        elinewidth=1.0,
        capsize=2.0,
    )
    ax.scatter(
        data.relative_improvement,
        y,
        c=colors,
        s=18,
        zorder=3,
        edgecolors="white",
        linewidths=0.35,
    )
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(y, data.label, fontsize=7.1)
    ax.set_xlabel(
        "Relative improvement over per-task single model (paired 95% CI)",
        fontsize=8.2,
    )
    ax.grid(axis="x", color="#D8DEE2", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    panel_label(ax, "c", "Sample-paired uncertainty (10,000 resamples)")
    return {"significant_fdr_0_05": int(significant.sum()), "tasks": len(data)}


def draw_subgroups(ax: plt.Axes, subgroup: pd.DataFrame, labels: dict[str, str]) -> dict[str, int]:
    valid = subgroup[subgroup.metric_valid.astype(bool)].copy()
    valid["relative_improvement"] = valid.improvement / valid.comparator_score.abs().clip(
        lower=1e-12
    )
    valid["column"] = valid.descriptor.map(
        {"molecular_weight": "MW", "clogp": "cLogP", "tpsa": "TPSA"}
    ) + valid.group.map({"low_or_equal_median": " low", "above_median": " high"})
    columns = ["MW low", "MW high", "cLogP low", "cLogP high", "TPSA low", "TPSA high"]
    tasks = list(labels)
    matrix = valid.pivot(
        index="task", columns="column", values="relative_improvement"
    ).reindex(index=tasks, columns=columns)
    finite = matrix.to_numpy(dtype=float)
    limit = float(np.nanquantile(np.abs(finite), 0.95))
    limit = max(limit, 1e-6)
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(columns)), columns, rotation=35, ha="right", fontsize=7.2)
    ax.set_yticks(np.arange(len(tasks)), [labels[t] for t in tasks], fontsize=6.8)
    invalid = subgroup.groupby(["task", "descriptor", "group"]).metric_valid.first()
    invalid_count = int((~invalid.astype(bool)).sum())
    for task_index, task in enumerate(tasks):
        for column_index, column in enumerate(columns):
            descriptor_label, side = column.split()
            descriptor = {"MW": "molecular_weight", "cLogP": "clogp", "TPSA": "tpsa"}[descriptor_label]
            group = "low_or_equal_median" if side == "low" else "above_median"
            if (task, descriptor, group) in invalid.index and not bool(invalid.loc[(task, descriptor, group)]):
                ax.text(column_index, task_index, "×", ha="center", va="center", fontsize=7, color=INK)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Relative improvement vs per-task single", fontsize=7.5)
    colorbar.ax.tick_params(labelsize=6.5)
    panel_label(ax, "d", "Matched molecular-property subgroup effects")
    return {"valid_groups": int(valid.shape[0]), "invalid_groups": invalid_count}


def main() -> None:
    args = parse_args()
    benchmark = pd.read_csv(args.benchmark_table)
    controlled = pd.read_csv(args.controlled_summary)
    bootstrap = pd.read_csv(args.bootstrap_results)
    subgroup = pd.read_csv(args.subgroup_table)
    labels = task_labels(benchmark)
    rng = np.random.default_rng(args.seed)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.4), constrained_layout=True)
    draw_public_margin(axes[0, 0], benchmark)
    rank_summary = draw_controlled(axes[0, 1], controlled, rng)
    paired_summary = draw_paired_ci(axes[1, 0], bootstrap, labels)
    subgroup_summary = draw_subgroups(axes[1, 1], subgroup, labels)
    fig.suptitle(
        "Leakage-safe benchmark results, matched controls and sample-level uncertainty",
        fontsize=13,
        weight="bold",
        color=INK,
    )

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    summary = {
        "controlled_mean_normalized_rank_utility": rank_summary,
        "paired_bootstrap": paired_summary,
        "subgroups": subgroup_summary,
    }
    args.output_stem.with_name(args.output_stem.name + "_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
