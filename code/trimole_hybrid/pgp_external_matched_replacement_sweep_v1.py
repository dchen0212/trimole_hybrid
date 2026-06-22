#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path("<PROJECT_ROOT>/trimole_hybrid")
OUT = ROOT / "results_strict" / "pgp_external_matched_replacement_sweep_v1"
TASK = "pgp_broccatelli"

BASE_CANDIDATES = [
    {
        "drug": "Mavorixafor",
        "task_sheet": "TDC.Pgp",
        "task": TASK,
        "external_label": 1.0,
        "smiles": "NCCCCN(Cc1nc2ccccc2[nH]1)[C@H]1CCCc2cccnc21",
        "source_note": "external PGP sensitivity candidate",
    },
    {
        "drug": "Tezacaftor",
        "task_sheet": "TDC.Pgp",
        "task": TASK,
        "external_label": 1.0,
        "smiles": "CC(C)(CO)c1cc2cc(NC(=O)C3(c4ccc5c(c4)OC(F)(F)O5)CC3)c(F)cc2n1C[C@@H](O)CO",
        "source_note": "FDA/PDF PGP candidate",
    },
    {
        "drug": "Deutivacaftor",
        "task_sheet": "TDC.Pgp",
        "task": TASK,
        "external_label": 1.0,
        "smiles": "[2H]C([2H])([2H])C(C1=C(C=C(C(=C1)C(C)(C)C)NC(=O)C2=CNC3=CC=CC=C3C2=O)O)(C([2H])([2H])[2H])C([2H])([2H])[2H]",
        "source_note": "FDA/PDF PGP candidate",
    },
    {
        "drug": "Vanzacaftor",
        "task_sheet": "TDC.Pgp",
        "task": TASK,
        "external_label": 1.0,
        "smiles": "CC1(C)C[C@H]2CN1C1=NC(=CC=C1C(=O)NS(=O)(=O)C1=CC=CC(NCCC2)=N1)N1C=CC(OCCC2C3(CC3)C22CC2)=N1",
        "source_note": "FDA/PDF PGP candidate",
    },
]


def mol(smiles: str):
    from rdkit import Chem

    return Chem.MolFromSmiles(str(smiles))


def canon(m):
    from rdkit import Chem

    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)


def sanitize(rw):
    from rdkit import Chem

    try:
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def replace_atom(m, idx: int, atomic_num: int):
    from rdkit import Chem

    rw = Chem.RWMol(m)
    atom = rw.GetAtomWithIdx(int(idx))
    atom.SetAtomicNum(int(atomic_num))
    atom.SetFormalCharge(0)
    atom.SetNoImplicit(False)
    atom.SetNumExplicitHs(0)
    return sanitize(rw)


def remove_terminal_atom(m, idx: int):
    from rdkit import Chem

    rw = Chem.RWMol(m)
    atom = rw.GetAtomWithIdx(int(idx))
    if atom.GetDegree() != 1:
        return None
    rw.RemoveAtom(int(idx))
    return sanitize(rw)


def descriptors(smiles: str) -> dict:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return {}
    return {
        "MolWt": Descriptors.MolWt(m),
        "cLogP": Crippen.MolLogP(m),
        "TPSA": rdMolDescriptors.CalcTPSA(m),
        "HBD": rdMolDescriptors.CalcNumHBD(m),
        "HBA": rdMolDescriptors.CalcNumHBA(m),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(m),
        "HeavyAtoms": m.GetNumHeavyAtoms(),
    }


def enumerate_variants(drug: str, smiles: str, max_n: int = 18) -> list[dict]:
    m = mol(smiles)
    if m is None:
        return []
    rows = [
        {
            "drug": drug,
            "perturbation_id": f"{drug}__original",
            "source_note": "original",
            "smiles": canon(m),
            "edit": "none",
            "target_region": "original",
            "atom_indices": "",
        }
    ]
    seen = {rows[0]["smiles"]}

    def add(name: str, edit: str, region: str, atom_ids: list[int], pm) -> None:
        if pm is None or len(rows) >= max_n:
            return
        s = canon(pm)
        if s in seen or s == rows[0]["smiles"]:
            return
        seen.add(s)
        rows.append(
            {
                "drug": drug,
                "perturbation_id": f"{drug}__{name}__{len(rows):02d}",
                "source_note": "matched replacement perturbation",
                "smiles": s,
                "edit": edit,
                "target_region": region,
                "atom_indices": ";".join(map(str, atom_ids)),
            }
        )

    for atom in m.GetAtoms():
        if len(rows) >= max_n:
            break
        idx = atom.GetIdx()
        sym = atom.GetSymbol()
        deg = atom.GetDegree()
        aromatic = atom.GetIsAromatic()
        if sym == "N" and not aromatic:
            if deg <= 2:
                add("aliphatic_N_to_O", "aliphatic N -> O", "basic amine / polar linker region", [idx], replace_atom(m, idx, 8))
            add("aliphatic_N_to_C", "aliphatic N -> C", "basic amine / linker region", [idx], replace_atom(m, idx, 6))
        if sym == "O" and not aromatic and deg <= 2:
            add("O_to_CH2", "O -> CH2", "oxygen linker / polarity region", [idx], replace_atom(m, idx, 6))
        if sym in {"F", "Cl", "Br"} and deg == 1:
            add(f"{sym}_removed", f"{sym} removal", "halogenated hydrophobic region", [idx], remove_terminal_atom(m, idx))

    for ring in m.GetRingInfo().AtomRings():
        if len(rows) >= max_n:
            break
        atoms = [m.GetAtomWithIdx(int(i)) for i in ring]
        if any(x.GetIsAromatic() for x in atoms):
            continue
        for x in atoms:
            if x.GetSymbol() == "C" and x.GetDegree() == 2:
                add(
                    "sat_ring_CH2_to_O",
                    "saturated ring CH2 -> O",
                    "saturated ring polarity/shape region",
                    [x.GetIdx()],
                    replace_atom(m, x.GetIdx(), 8),
                )
                break

    return rows


def build_candidates() -> pd.DataFrame:
    rows = []
    for candidate in BASE_CANDIDATES:
        for variant in enumerate_variants(candidate["drug"], candidate["smiles"]):
            variant.update({k: candidate[k] for k in ["drug", "task_sheet", "task", "external_label"]})
            variant["original_input_smiles"] = candidate["smiles"]
            rows.append(variant)
    df = pd.DataFrame(rows)
    df["canonical_smiles"] = [canon(mol(s)) if mol(s) is not None else "" for s in df["smiles"]]
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates()
    candidates.to_csv(OUT / "pgp_external_candidates.csv", index=False)

    for dirname in ["external_embeddings", "external_ept", "external_ept_task", "external_ept_processed"]:
        path = OUT / dirname
        if path.exists():
            shutil.rmtree(path)

    sys.path.insert(0, str(ROOT / "results_strict"))
    import reconstruct_pgp_exact_external_v36 as pgp

    pgp.OUT = OUT
    pgp.MAIN_OUT = OUT
    pgp.DEFAULT_CANDIDATES = []
    sys.argv = ["reconstruct_pgp_exact_external_v36.py"]
    pgp.main()

    pred = pd.read_csv(OUT / "external_predictions_pgp_exact_reconstructed.csv")
    meta = candidates[["drug", "perturbation_id", "edit", "target_region", "atom_indices", "smiles"]].copy()
    merged = pred.merge(meta, on=["drug", "perturbation_id", "smiles"], how="left")
    orig = (
        merged[merged["perturbation_id"].str.endswith("__original")]
        [["drug", "trimole_exact_reconstructed_pred_mean"]]
        .rename(columns={"trimole_exact_reconstructed_pred_mean": "original_score"})
    )
    merged = merged.merge(orig, on="drug", how="left")
    merged["score_delta_vs_original"] = merged["trimole_exact_reconstructed_pred_mean"] - merged["original_score"]
    merged["abs_score_delta_vs_original"] = merged["score_delta_vs_original"].abs()

    desc = pd.DataFrame([descriptors(s) for s in merged["smiles"]])
    merged = pd.concat([merged.reset_index(drop=True), desc.reset_index(drop=True)], axis=1)
    orig_desc = merged[merged["perturbation_id"].str.endswith("__original")].set_index("drug")
    for col in ["MolWt", "cLogP", "TPSA", "HBD", "HBA", "RotBonds", "HeavyAtoms"]:
        merged[f"delta_{col}_vs_original"] = [row[col] - orig_desc.loc[row["drug"], col] for _, row in merged.iterrows()]

    merged.to_csv(OUT / "matched_replacement_predictions.csv", index=False)
    ranking = merged[~merged["perturbation_id"].str.endswith("__original")].sort_values(
        "abs_score_delta_vs_original", ascending=False
    )
    ranking.to_csv(OUT / "matched_replacement_case_ranking.csv", index=False)

    print("\nTop PGP matched-replacement cases")
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
    ]
    print(ranking[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
