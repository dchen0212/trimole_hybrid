#!/usr/bin/env python3
"""Generate the corrected Figure 4 from the versioned case-study audit table.

The two stages deliberately have disjoint dependencies.  ``molecules`` needs
RDKit and writes molecule PNGs; ``figure`` needs pandas/matplotlib and combines
those PNGs with the audited prediction values.  This permits reproducible
generation in the archived server environments without relabeling an external
compound as an experimentally positive P-gp-inhibition example.
"""

from __future__ import annotations

import argparse
from pathlib import Path


CASE_ORDER = ["Figure4a_AMES_test297", "Figure4b_Deutivacaftor_Pgp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("molecules", "figure", "all"), default="all")
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path)
    return parser.parse_args()


def load_rows(table: Path):
    import pandas as pd

    frame = pd.read_csv(table, keep_default_na=False)
    frame = frame.set_index("case_id").loc[CASE_ORDER].reset_index()
    return frame


def draw_molecules(table: Path, asset_dir: Path) -> None:
    from rdkit import Chem
    from rdkit.Chem import Draw

    asset_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(table)
    for _, row in rows.iterrows():
        for state, column in (("original", "canonical_smiles"), ("perturbed", "perturbed_smiles")):
            molecule = Chem.MolFromSmiles(row[column])
            if molecule is None:
                raise ValueError(f"Invalid {state} SMILES for {row['case_id']}")
            Draw.MolToFile(
                molecule,
                str(asset_dir / f"{row['case_id']}__{state}.png"),
                size=(700, 420),
                legend=state.capitalize(),
            )


def render_figure(table: Path, asset_dir: Path, output_stem: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    rows = load_rows(table)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.linewidth": 0.8,
        }
    )
    fig = plt.figure(figsize=(11.8, 4.5), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, wspace=0.08)
    colors = ["#9A86C8", "#8EB9D4", "#D9B47E"]

    for panel_index, (_, row) in enumerate(rows.iterrows()):
        inner = outer[panel_index].subgridspec(2, 2, height_ratios=[0.14, 0.86], width_ratios=[1.35, 1.0])
        title_ax = fig.add_subplot(inner[0, :])
        title_ax.axis("off")
        if row["case_id"].startswith("Figure4a"):
            title = "a  AMES official-test molecule | y = 1"
        else:
            title = "b  Deutivacaftor | P-gp inhibition | external, unlabeled"
        title_ax.text(0.0, 0.55, title, fontsize=11, va="center")

        molecule_ax = fig.add_subplot(inner[1, 0])
        molecule_ax.axis("off")
        original = Image.open(asset_dir / f"{row['case_id']}__original.png")
        perturbed = Image.open(asset_dir / f"{row['case_id']}__perturbed.png")
        combined = np.concatenate([np.asarray(original), np.asarray(perturbed)], axis=1)
        molecule_ax.imshow(combined)

        bar_ax = fig.add_subplot(inner[1, 1])
        values = [
            float(row["minimol_score_mean"]),
            float(row["trimole_score_mean"]),
            float(row["perturbed_score_mean"]),
        ]
        errors = [
            float(row["minimol_score_std"] or 0),
            float(row["trimole_score_std"] or 0),
            float(row["perturbed_score_std"] or 0),
        ]
        x = np.arange(3)
        bars = bar_ax.bar(x, values, yerr=errors, width=0.62, color=colors, capsize=2.5, edgecolor="none")
        bar_ax.axhline(0.5, color="#7A7A7A", linewidth=0.8, linestyle=(0, (3, 3)))
        bar_ax.text(2.45, 0.515, "0.5", color="#666666", fontsize=7)
        bar_ax.set_xticks(x, ["MiniMol", "Trimole", "Perturbed"])
        bar_ax.set_ylim(0, 1.08)
        bar_ax.set_ylabel("Model output")
        bar_ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            bar_ax.text(bar.get_x() + bar.get_width() / 2, min(value + 0.045, 1.035), f"{value:.3f}", ha="center")
        if panel_index == 1:
            bar_ax.text(
                0.02,
                0.98,
                "No experimental inhibition label;\nnot a correctness comparison",
                transform=bar_ax.transAxes,
                va="top",
                fontsize=7.2,
                color="#8B2F23",
            )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.stage in {"molecules", "all"}:
        draw_molecules(args.table, args.asset_dir)
    if args.stage in {"figure", "all"}:
        if args.output_stem is None:
            raise SystemExit("--output-stem is required for figure generation")
        render_figure(args.table, args.asset_dir, args.output_stem)


if __name__ == "__main__":
    main()
