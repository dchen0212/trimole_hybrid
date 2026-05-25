#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D


ROOT = Path("/mnt/afs/250010150/zhensheng/trimole_hybrid")
INFILE = ROOT / "results_strict" / "pgp_external_matched_replacement_sweep_v1" / "matched_replacement_case_ranking.csv"
OUTDIR = ROOT / "results_strict" / "pgp_external_matched_replacement_sweep_v1" / "figure"


def replace_atom(mol, atom_idx: int, atomic_num: int):
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(int(atom_idx))
    atom.SetAtomicNum(int(atomic_num))
    atom.SetFormalCharge(0)
    atom.SetNoImplicit(False)
    atom.SetNumExplicitHs(0)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def mol_png(mol, highlight_atoms: list[int], path: Path, size=(1100, 780)) -> None:
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2.1
    opts.minFontSize = 15
    opts.maxFontSize = 28
    opts.padding = 0.08
    colors = {int(i): (0.91, 0.37, 0.16) for i in highlight_atoms}
    radii = {int(i): 0.42 for i in highlight_atoms}
    drawer.DrawMolecule(mol, highlightAtoms=[int(i) for i in highlight_atoms], highlightAtomColors=colors, highlightAtomRadii=radii)
    drawer.FinishDrawing()
    path.write_bytes(drawer.GetDrawingText())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INFILE)
    row = df[
        (df["drug"] == "Deutivacaftor")
        & (df["edit"] == "O -> CH2")
        & (df["score_delta_vs_original"] < -0.1)
    ].sort_values("score_delta_vs_original").iloc[0]

    atom_idx = int(float(row["atom_indices"]))
    original_smiles = str(row["original_input_smiles"])
    original = Chem.MolFromSmiles(original_smiles)
    perturbed = replace_atom(original, atom_idx, 6)

    original_png = OUTDIR / "deutivacaftor_original_highlight.png"
    perturbed_png = OUTDIR / "deutivacaftor_o_to_ch2_highlight.png"
    mol_png(original, [atom_idx], original_png)
    mol_png(perturbed, [atom_idx], perturbed_png)

    original_score = float(row["original_score"])
    perturbed_score = float(row["trimole_exact_reconstructed_pred_mean"])
    perturbed_std = float(row["trimole_exact_reconstructed_pred_std"])
    delta = float(row["score_delta_vs_original"])

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 1.0

    fig = plt.figure(figsize=(7.2, 2.75), dpi=350)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.38, 1.38, 0.95], wspace=0.18)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    for ax, img_path, title, subtitle in [
        (ax0, original_png, "Original", f"P-gp score = {original_score:.3f}"),
        (ax1, perturbed_png, "O→CH$_2$ replacement", f"P-gp score = {perturbed_score:.3f}"),
    ]:
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=10, pad=3, weight="bold")
        ax.text(0.5, -0.03, subtitle, ha="center", va="top", transform=ax.transAxes, fontsize=9)

    ax0.text(
        0.02,
        0.98,
        "highlighted atom: replaced O",
        transform=ax0.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#8a3a16",
    )
    ax1.annotate(
        "",
        xy=(-0.06, 0.50),
        xytext=(-0.20, 0.50),
        xycoords=ax1.transAxes,
        textcoords=ax1.transAxes,
        arrowprops=dict(arrowstyle="->", lw=1.4, color="#555555"),
    )

    colors = ["#d6b98c", "#7fb3d5"]
    ax2.bar([0, 1], [original_score, perturbed_score], color=colors, width=0.62, edgecolor="#333333", linewidth=0.45)
    ax2.errorbar([1], [perturbed_score], yerr=[perturbed_std], color="#333333", capsize=2.4, lw=0.8)
    ax2.axhline(0.5, color="#777777", ls="--", lw=0.8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Exact v36 P-gp score", fontsize=9)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Original", "O→CH$_2$"], rotation=20, ha="right", fontsize=8.5)
    ax2.tick_params(axis="y", labelsize=8.5, length=3)
    ax2.set_title("Prediction sensitivity", fontsize=10, pad=3, weight="bold")
    for x, y in zip([0, 1], [original_score, perturbed_score]):
        ax2.text(x, y + 0.035, f"{y:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax2.text(
        0.5,
        0.12,
        f"Δ = {delta:.3f}",
        transform=ax2.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#1f5f8b",
        weight="bold",
    )

    fig.suptitle("Deutivacaftor / P-gp: exact v36 matched-replacement sensitivity", x=0.02, ha="left", fontsize=11.5, weight="bold")
    fig.text(
        0.02,
        0.015,
        "Prediction-supported local sensitivity; not a causal mechanistic assignment. Endpoint replay: exact v36 reconstruction, 5 seeds.",
        ha="left",
        va="bottom",
        fontsize=7.8,
        color="#4a4a4a",
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.82, bottom=0.24)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUTDIR / f"deutivacaftor_pgp_matched_replacement_sensitivity.{ext}", dpi=350)

    summary = pd.DataFrame(
        [
            {
                "drug": "Deutivacaftor",
                "task": "P-gp",
                "edit": "O -> CH2",
                "atom_idx": atom_idx,
                "original_score": original_score,
                "perturbed_score": perturbed_score,
                "perturbed_score_std": perturbed_std,
                "delta": delta,
                "original_smiles": Chem.MolToSmiles(original, canonical=True, isomericSmiles=True),
                "perturbed_smiles": Chem.MolToSmiles(perturbed, canonical=True, isomericSmiles=True),
            }
        ]
    )
    summary.to_csv(OUTDIR / "deutivacaftor_pgp_matched_replacement_sensitivity_data.csv", index=False)
    print(OUTDIR)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
