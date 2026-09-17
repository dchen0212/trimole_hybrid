from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


VARIANT_LABELS = {
    "no_chemistry_sidecar": "No chemistry\nsidecar",
    "no_ept_or_no_3d": "No EPT / 3D",
    "no_prediction_level_ensemble": "No prediction\nensemble",
    "no_seedbag_single_seed": "Single seed",
    "no_task_adaptive_selection": "No task-adaptive\nselection",
}
TASK_LABELS = {
    "ames": "AMES",
    "bbb_martins": "BBB",
    "bioavailability_ma": "Bioavailability",
    "caco2_wang": "Caco-2",
    "clearance_hepatocyte_az": "CL-Hepa",
    "clearance_microsome_az": "CL-Micro",
    "cyp2c9_substrate_carbonmangels": "CYP2C9-S",
    "cyp2c9_veith": "CYP2C9-I",
    "cyp2d6_substrate_carbonmangels": "CYP2D6-S",
    "cyp2d6_veith": "CYP2D6-I",
    "cyp3a4_substrate_carbonmangels": "CYP3A4-S",
    "cyp3a4_veith": "CYP3A4-I",
    "dili": "DILI",
    "half_life_obach": "Half-life",
    "herg": "hERG",
    "hia_hou": "HIA",
    "ld50_zhu": "LD50",
    "lipophilicity_astrazeneca": "Lipophilicity",
    "pgp_broccatelli": "P-gp",
    "ppbr_az": "PPBR",
    "solubility_aqsoldb": "AqSolDB",
    "vdss_lombardo": "VDss",
}
METRIC_COLORS = {
    "AUROC": "#5B8DB8",
    "AUPRC": "#D9A441",
    "Spearman": "#5F9E7D",
    "MAE": "#9B83C7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s23", type=Path, required=True)
    parser.add_argument("--mlp-control", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--supp-output-stem", type=Path)
    return parser.parse_args()


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.09, 1.04, label, transform=axis.transAxes, fontsize=12, fontweight="bold")


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.s23)
    rank = data[data.record_type == "ablation_rank_loss"].copy()
    stability = data[data.record_type == "validation_selection_stability"].copy()
    expected_variants = list(VARIANT_LABELS)
    rank = rank[rank.variant.isin(expected_variants)]
    if rank.task.nunique() != 22 or len(rank) != 22 * len(expected_variants):
        raise ValueError("S23 does not contain a complete 22-task ablation grid")
    if stability.task.nunique() != 22:
        raise ValueError("S23 selection stability does not cover 22 tasks")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    figure = plt.figure(figsize=(13.2, 11.2), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=[0.75, 1.8, 1.25], width_ratios=[1, 1.45])
    ax_a = figure.add_subplot(grid[0, 0])
    ax_d = figure.add_subplot(grid[0, 1])
    ax_b = figure.add_subplot(grid[1, 0])
    ax_c = figure.add_subplot(grid[1, 1])
    ax_e = figure.add_subplot(grid[2, :])

    summary = rank.groupby("variant").normalized_rank_loss.agg(["mean", "median"]).loc[expected_variants]
    x = np.arange(len(summary))
    width = 0.36
    ax_a.bar(x - width / 2, summary["mean"], width, label="Mean", color="#A9C7DC")
    ax_a.bar(x + width / 2, summary["median"], width, label="Median", color="#557C9A")
    ax_a.axhline(0, color="#555555", linewidth=0.8)
    ax_a.set_xticks(x, [VARIANT_LABELS[value] for value in summary.index], rotation=30, ha="right")
    ax_a.set_ylabel("Normalized rank loss")
    ax_a.set_title("Overall scale-free ablation effect", loc="left", fontweight="bold")
    ax_a.legend(ncol=2, loc="upper left")
    panel_label(ax_a, "a")

    rank_pivot = rank.pivot(index="task", columns="variant", values="normalized_rank_loss")
    task_order = rank.groupby("task").normalized_rank_loss.mean().sort_values(ascending=False).index
    matrix = rank_pivot.loc[task_order, expected_variants].to_numpy()
    bound = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.1)
    image = ax_b.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
    ax_b.set_yticks(np.arange(len(task_order)), [TASK_LABELS.get(value, value) for value in task_order])
    ax_b.set_xticks(np.arange(len(expected_variants)), [VARIANT_LABELS[value] for value in expected_variants], rotation=30, ha="right")
    ax_b.set_title("Per-task normalized rank loss", loc="left", fontweight="bold")
    ax_b.spines[:].set_visible(False)
    colorbar = figure.colorbar(image, ax=ax_b, fraction=0.035, pad=0.02)
    colorbar.set_label("Rank loss")
    panel_label(ax_b, "b")

    metric_summary = rank.groupby(["metric", "variant"]).normalized_rank_loss.mean().unstack()
    metric_order = [value for value in ("AUROC", "AUPRC", "Spearman", "MAE") if value in metric_summary.index]
    metric_matrix = metric_summary.loc[metric_order, expected_variants].to_numpy()
    bound_metric = max(abs(np.nanmin(metric_matrix)), abs(np.nanmax(metric_matrix)), 0.1)
    image_d = ax_d.imshow(metric_matrix, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-bound_metric, vcenter=0, vmax=bound_metric))
    ax_d.set_yticks(np.arange(len(metric_order)), metric_order)
    ax_d.set_xticks(np.arange(len(expected_variants)), [VARIANT_LABELS[value] for value in expected_variants], rotation=25, ha="right")
    for row in range(metric_matrix.shape[0]):
        for column in range(metric_matrix.shape[1]):
            ax_d.text(column, row, f"{metric_matrix[row, column]:.2f}", ha="center", va="center", fontsize=7)
    ax_d.set_title("Mean rank loss by metric", loc="left", fontweight="bold")
    ax_d.spines[:].set_visible(False)
    colorbar_d = figure.colorbar(image_d, ax=ax_d, fraction=0.04, pad=0.02)
    colorbar_d.set_label("Mean rank loss")
    panel_label(ax_d, "d")

    frequency = stability.pivot(index="task", columns="candidate", values="selection_frequency").fillna(0)
    switch = stability.groupby("task").switch_rate.first()
    stability_order = switch.sort_values(ascending=True).index
    frequency = frequency.reindex(stability_order)
    y = np.arange(len(stability_order))
    left = np.zeros(len(stability_order))
    candidate_colors = {"chemberta": "#6E9EC2", "kpgt": "#78B59A", "ept": "#E2B35B"}
    for candidate in ("chemberta", "kpgt", "ept"):
        values = frequency.get(candidate, pd.Series(0, index=frequency.index)).to_numpy()
        ax_c.barh(y, values, left=left, height=0.68, color=candidate_colors[candidate], label=candidate)
        left += values
    ax_c.scatter(switch.loc[stability_order], y, s=16, facecolors="white", edgecolors="#202020", linewidths=0.8, label="switch rate", zorder=3)
    ax_c.set_yticks(y, [TASK_LABELS.get(value, value) for value in stability_order])
    ax_c.set_xlim(0, 1)
    ax_c.set_xlabel("Validation-bootstrap selection frequency")
    ax_c.set_title("Selection frequency and switching", loc="left", fontweight="bold")
    ax_c.legend(ncol=4, loc="lower right", fontsize=7)
    panel_label(ax_c, "c")

    mlp = pd.read_csv(args.mlp_control)
    mlp = mlp[mlp.variant == "mlp_stacking_baseline_v1"].set_index("task")
    full = data[(data.record_type == "ablation_rank_loss") & (data.variant == "full_v36_final")].drop_duplicates("task").set_index("task")
    shared = full.index.intersection(mlp.index)
    advantage = pd.Series(index=shared, dtype=float)
    for task in shared:
        raw_advantage = (
            full.loc[task, "score_mean"] - mlp.loc[task, "score_mean"]
            if full.loc[task, "direction"] == "max"
            else mlp.loc[task, "score_mean"] - full.loc[task, "score_mean"]
        )
        advantage.loc[task] = raw_advantage / max(abs(full.loc[task, "score_mean"]), 1e-12)
    advantage = advantage.sort_values()
    colors = [METRIC_COLORS[full.loc[task, "metric"]] for task in advantage.index]
    ax_e.bar(np.arange(len(advantage)), advantage.to_numpy(), color=colors, width=0.78)
    ax_e.axhline(0, color="#333333", linewidth=0.8)
    ax_e.set_xticks(np.arange(len(advantage)), [TASK_LABELS.get(value, value) for value in advantage.index], rotation=55, ha="right")
    ax_e.set_ylabel("Relative performance loss\nof naive MLP")
    ax_e.set_title("Naive MLP late-fusion control", loc="left", fontweight="bold")
    handles = [plt.Line2D([0], [0], color=color, linewidth=6, label=metric) for metric, color in METRIC_COLORS.items()]
    ax_e.legend(handles=handles, ncol=4, loc="upper left")
    panel_label(ax_e, "e")

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output_stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(args.output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    args.output_stem.with_name(args.output_stem.name + "_summary.json").write_text(
        json.dumps(
            {
                "tasks": int(rank.task.nunique()),
                "variants": expected_variants,
                "median_rank_loss": summary["median"].to_dict(),
                "median_switch_rate": float(switch.median()),
                "maximum_switch_rate": float(switch.max()),
                "mlp_worse_tasks": int((advantage > 0).sum()),
            },
            indent=2,
        )
        + "\n"
    )

    if args.supp_output_stem is not None:
        supp_figure, (supp_a, supp_b) = plt.subplots(
            1, 2, figsize=(12.8, 8.2), gridspec_kw={"width_ratios": [0.7, 1.5]}, constrained_layout=True
        )
        full_rows = data[
            (data.record_type == "ablation_rank_loss") & (data.variant == "full_v36_final")
        ].drop_duplicates("task").set_index("task")
        component_columns = [
            "uses_chemistry_sidecar",
            "uses_ept_or_3d",
            "uses_prediction_level_ensemble",
            "uses_seedbag",
        ]
        component_labels = ["Chemistry sidecar", "EPT / 3D", "Prediction ensemble", "Seed bagging"]
        component_matrix = full_rows.loc[task_order, component_columns].astype(float).to_numpy()
        supp_a.imshow(component_matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        supp_a.set_yticks(np.arange(len(task_order)), [TASK_LABELS.get(value, value) for value in task_order])
        supp_a.set_xticks(np.arange(len(component_labels)), component_labels, rotation=35, ha="right")
        for row in range(component_matrix.shape[0]):
            for column in range(component_matrix.shape[1]):
                supp_a.text(column, row, "●" if component_matrix[row, column] else "○", ha="center", va="center", fontsize=8, color="white" if component_matrix[row, column] else "#777777")
        supp_a.set_title("Components used by the fixed task setup", loc="left", fontweight="bold")
        supp_a.spines[:].set_visible(False)
        panel_label(supp_a, "a")

        supp_image = supp_b.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
        supp_b.set_yticks(np.arange(len(task_order)), [TASK_LABELS.get(value, value) for value in task_order])
        supp_b.set_xticks(np.arange(len(expected_variants)), [VARIANT_LABELS[value] for value in expected_variants], rotation=35, ha="right")
        supp_b.set_title("Component-removal normalized rank loss", loc="left", fontweight="bold")
        supp_b.spines[:].set_visible(False)
        supp_colorbar = supp_figure.colorbar(supp_image, ax=supp_b, fraction=0.04, pad=0.02)
        supp_colorbar.set_label("Normalized rank loss")
        panel_label(supp_b, "b")
        args.supp_output_stem.parent.mkdir(parents=True, exist_ok=True)
        supp_figure.savefig(args.supp_output_stem.with_suffix(".pdf"), bbox_inches="tight")
        supp_figure.savefig(args.supp_output_stem.with_suffix(".svg"), bbox_inches="tight")
        supp_figure.savefig(args.supp_output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(supp_figure)


if __name__ == "__main__":
    main()
