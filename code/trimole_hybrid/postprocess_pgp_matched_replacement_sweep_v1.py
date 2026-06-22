#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors


OUT = Path("<PROJECT_ROOT>/trimole_hybrid/results_strict/pgp_external_matched_replacement_sweep_v1")


def descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {}
    return {
        "MolWt": Descriptors.MolWt(mol),
        "cLogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
    }


def main() -> None:
    pred = pd.read_csv(OUT / "external_predictions_pgp_exact_reconstructed.csv")
    desc = pd.DataFrame([descriptors(s) for s in pred["smiles"]])
    df = pd.concat([pred.reset_index(drop=True), desc.reset_index(drop=True)], axis=1)
    orig = (
        df[df["perturbation_id"].str.endswith("__original")]
        [["drug", "trimole_exact_reconstructed_pred_mean"]]
        .rename(columns={"trimole_exact_reconstructed_pred_mean": "original_score"})
    )
    df = df.merge(orig, on="drug", how="left")
    df["score_delta_vs_original"] = df["trimole_exact_reconstructed_pred_mean"] - df["original_score"]
    df["abs_score_delta_vs_original"] = df["score_delta_vs_original"].abs()
    orig_desc = df[df["perturbation_id"].str.endswith("__original")].set_index("drug")
    for col in ["MolWt", "cLogP", "TPSA", "HBD", "HBA", "RotBonds", "HeavyAtoms"]:
        df[f"delta_{col}_vs_original"] = [row[col] - orig_desc.loc[row["drug"], col] for _, row in df.iterrows()]

    df.to_csv(OUT / "matched_replacement_predictions.csv", index=False)
    ranking = df[~df["perturbation_id"].str.endswith("__original")].sort_values(
        "abs_score_delta_vs_original", ascending=False
    )
    ranking.to_csv(OUT / "matched_replacement_case_ranking.csv", index=False)
    cols = [
        "drug",
        "external_label",
        "original_score",
        "edit",
        "target_region",
        "trimole_exact_reconstructed_pred_mean",
        "trimole_exact_reconstructed_pred_std",
        "score_delta_vs_original",
        "abs_score_delta_vs_original",
        "delta_TPSA_vs_original",
        "delta_cLogP_vs_original",
        "delta_HeavyAtoms_vs_original",
        "smiles",
    ]
    print(ranking[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
