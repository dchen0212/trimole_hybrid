#!/usr/bin/env python3
"""Generate the revised Figure 1 with explicit selection/test boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


NAVY = "#153B5B"
BLUE = "#3978A8"
TEAL = "#2B8C82"
GREEN = "#67A77A"
OCHRE = "#D79B36"
CORAL = "#C96855"
INK = "#18232D"
MUTED = "#586875"
PALE_BLUE = "#E8F1F7"
PALE_TEAL = "#E7F4F1"
PALE_OCHRE = "#FBF0DB"
PALE_CORAL = "#F8E9E5"
PALE_GRAY = "#F2F4F5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-stem", type=Path, required=True)
    return parser.parse_args()


def box(ax, xy, width, height, text, face, edge, fontsize=8.5, weight="normal", text_color=INK, radius=0.02):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        transform=ax.transAxes,
        linewidth=1.0,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color=text_color,
        linespacing=1.18,
    )
    return patch


def arrow(ax, start, end, color=INK, style="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            linestyle=style,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def panel_label(ax, label, title):
    ax.text(0.0, 1.025, label, transform=ax.transAxes, fontsize=13, weight="bold", va="bottom")
    ax.text(0.036, 1.025, title, transform=ax.transAxes, fontsize=12, weight="bold", va="bottom", color=NAVY)


def draw_protocol(ax):
    ax.set_axis_off()
    panel_label(ax, "a", "Leakage-controlled development and evaluation")
    steps = [
        ("1  Frozen splits", "train / valid / test\nscaffold partitions", PALE_BLUE, BLUE),
        ("2  Feature extraction", "SMILES · graph · EPT/3D\nfingerprint · descriptors", PALE_TEAL, TEAL),
        ("3  Base fitting", "train or inner folds only\nno official-test labels", PALE_OCHRE, OCHRE),
        ("4  Validation decisions", "hyperparameters · candidate\nweights · calibration · rank", PALE_CORAL, CORAL),
        ("5  Frozen refit", "write selection manifest\nrefit fixed recipe on train+valid", PALE_BLUE, BLUE),
        ("6  Test reporting", "write predictions first\nthen read labels once", PALE_TEAL, TEAL),
    ]
    left = 0.015
    gap = 0.024
    width = (0.97 - gap * 5) / 6
    for index, (title, body, face, edge) in enumerate(steps):
        x = left + index * (width + gap)
        box(ax, (x, 0.42), width, 0.38, f"{title}\n\n{body}", face, edge, fontsize=8.1, weight="normal")
        if index < 5:
            arrow(ax, (x + width, 0.61), (x + width + gap, 0.61), color=NAVY)

    ax.add_patch(
        Rectangle((0.012, 0.20), 0.976, 0.10, transform=ax.transAxes, facecolor="#FFF8F3", edgecolor=CORAL, linewidth=1.0)
    )
    ax.text(
        0.5,
        0.25,
        "Hard gate: official test labels are inaccessible to feature fitting, candidate ranking, blend-weight estimation, calibration and refitting",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        color="#8C3C2F",
        weight="bold",
    )
    ax.text(
        0.5,
        0.08,
        "Family transfer: source-train identities matching target validation/test are removed during selection; source train+valid identities matching target test are removed during final refit",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.6,
        color=MUTED,
    )


def draw_task_map(ax):
    ax.set_axis_off()
    panel_label(ax, "b", "Twenty-two TDC ADMET tasks")
    categories = [
        ("Absorption  6", ["Caco-2", "HIA", "Lipophilicity", "P-gp inhibition", "Solubility", "Bioavailability"], BLUE, PALE_BLUE),
        ("Distribution  3", ["BBB", "PPBR", "VDss"], OCHRE, PALE_OCHRE),
        ("Metabolism  6", ["CYP2C9-S", "CYP2D6-S", "CYP3A4-S", "CYP2C9-I", "CYP2D6-I", "CYP3A4-I"], TEAL, PALE_TEAL),
        ("Excretion  3", ["CL-hepatocyte", "CL-microsome", "Half-life"], GREEN, "#EBF5EC"),
        ("Toxicity  4", ["AMES", "DILI", "hERG", "LD50"], CORAL, PALE_CORAL),
    ]
    x_positions = [0.01, 0.225, 0.395, 0.67, 0.84]
    widths = [0.20, 0.155, 0.26, 0.155, 0.15]
    for (title, tasks, edge, face), x, width in zip(categories, x_positions, widths):
        box(ax, (x, 0.11), width, 0.78, "", face, edge, radius=0.02)
        ax.text(x + width / 2, 0.82, title, transform=ax.transAxes, ha="center", va="center", fontsize=8.8, weight="bold", color=edge)
        for row, task in enumerate(tasks):
            ax.text(x + 0.015, 0.70 - row * 0.102, task, transform=ax.transAxes, fontsize=7.8, color=INK)
    ax.text(
        0.5,
        0.005,
        "Official metrics: AUROC and AUPRC for classification; MAE and Spearman correlation for continuous endpoints",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.6,
        color=MUTED,
    )


def draw_candidate_pool(ax):
    ax.set_axis_off()
    panel_label(ax, "c", "Established representations and controlled combination rules")
    ax.text(
        0.5,
        0.935,
        "The contribution is the validation-governed task-wise protocol, not a new molecular encoder",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
        weight="bold",
        color="#8C3C2F",
    )

    group_specs = [
        ("Molecular views", ["SMILES\nChemBERTa", "2D graph\nKPGT", "EPT / 3D\nenvironment", "Morgan\nfingerprint", "RDKit / XL\ndescriptors"], PALE_TEAL, TEAL),
        ("Prediction heads", ["Ridge /\nlogistic", "XGBoost", "ExtraTrees"], PALE_OCHRE, OCHRE),
        ("Combination rules", ["single", "validation\ntop-2 / top-3", "uniform\naverage", "OOF\nstacking", "frozen task\nensemble", "matched\nFLAML"], PALE_BLUE, BLUE),
    ]
    y_positions = [0.66, 0.39, 0.10]
    for (title, items, face, edge), y in zip(group_specs, y_positions):
        ax.text(0.015, y + 0.085, title, transform=ax.transAxes, fontsize=8.7, weight="bold", color=edge, va="center")
        start_x = 0.27
        gap = 0.018
        width = min(0.12, (0.72 - gap * (len(items) - 1)) / len(items))
        for index, item in enumerate(items):
            box(ax, (start_x + index * (width + gap), y), width, 0.17, item, face, edge, fontsize=7.4, radius=0.018)
    ax.text(
        0.5,
        0.00,
        "Each task receives one machine-readable frozen recipe; all matched controls use the same official splits, representation inputs and five seeds",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=MUTED,
    )


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
        }
    )
    fig = plt.figure(figsize=(12.8, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.44, 0.56], width_ratios=[0.56, 0.44], hspace=0.12, wspace=0.08)
    draw_protocol(fig.add_subplot(grid[0, :]))
    draw_task_map(fig.add_subplot(grid[1, 0]))
    draw_candidate_pool(fig.add_subplot(grid[1, 1]))

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
