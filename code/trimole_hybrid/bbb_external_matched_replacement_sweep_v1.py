#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path("/mnt/afs/250010150/zhensheng/trimole_hybrid")
EPT_REPO = Path("/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1")
OUT = ROOT / "results_strict" / "bbb_external_matched_replacement_sweep_v1"
TASK = "bbb_martins"
SEEDS = [1, 2, 3, 4, 5]


def add_paths() -> None:
    for p in [EPT_REPO / "tools", EPT_REPO, EPT_REPO / "results_strict", ROOT / "tools", ROOT]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def canonical_smiles(mol) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def safe_sanitize(rw):
    from rdkit import Chem

    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def replace_atom(mol, atom_idx: int, atomic_num: int):
    from rdkit import Chem

    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(int(atom_idx))
    atom.SetAtomicNum(int(atomic_num))
    atom.SetFormalCharge(0)
    atom.SetNoImplicit(False)
    atom.SetNumExplicitHs(0)
    return safe_sanitize(rw)


def remove_terminal_atom(mol, atom_idx: int):
    from rdkit import Chem

    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(int(atom_idx))
    if atom.GetDegree() != 1:
        return None
    rw.RemoveAtom(int(atom_idx))
    return safe_sanitize(rw)


def neutralize_quaternary_n(mol, atom_idx: int):
    from rdkit import Chem

    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(int(atom_idx))
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
        return None
    atom.SetFormalCharge(0)
    atom.SetNoImplicit(False)
    return safe_sanitize(rw)


def descriptors(smiles: str) -> dict:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {k: np.nan for k in ["MolWt", "cLogP", "TPSA", "HBD", "HBA", "RotBonds", "HeavyAtoms"]}
    return {
        "MolWt": Descriptors.MolWt(mol),
        "cLogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
    }


def load_candidates() -> pd.DataFrame:
    p = (
        ROOT
        / "results_strict"
        / "alyftrek_minimol_trimole_case_v1"
        / "exact_reconstructed_bbb_v36"
        / "external_predictions_bbb_exact_reconstructed.csv"
    )
    df = pd.read_csv(p)
    df = df[df["prediction_status"].astype(str).str.contains("exact", na=False)].copy()
    df = df.drop_duplicates("canonical_smiles").reset_index(drop=True)
    return df


def enumerate_replacements_for_mol(drug: str, smiles: str, max_per_drug: int = 28) -> list[dict]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return []
    rows = [
        {
            "drug": drug,
            "replacement_id": f"{drug}__original",
            "replacement": "original",
            "edit": "none",
            "target_region": "original",
            "source_atom_indices": "",
            "smiles": canonical_smiles(mol),
            "status": "valid",
        }
    ]
    seen = {rows[0]["smiles"]}

    def add_variant(name: str, edit: str, region: str, atom_ids: list[int], pmol) -> None:
        if pmol is None or len(rows) >= max_per_drug:
            return
        psmi = canonical_smiles(pmol)
        if psmi in seen or psmi == rows[0]["smiles"]:
            return
        seen.add(psmi)
        rows.append(
            {
                "drug": drug,
                "replacement_id": f"{drug}__{name}__{len(rows):02d}",
                "replacement": name,
                "edit": edit,
                "target_region": region,
                "source_atom_indices": ";".join(str(x) for x in atom_ids),
                "smiles": psmi,
                "status": "valid",
            }
        )

    # BBB-oriented conservative local replacements: polarity, basicity, halogens and saturated rings.
    for atom in mol.GetAtoms():
        if len(rows) >= max_per_drug:
            break
        idx = atom.GetIdx()
        sym = atom.GetSymbol()
        aromatic = atom.GetIsAromatic()
        deg = atom.GetDegree()

        if sym == "N" and not aromatic:
            if atom.GetFormalCharge() > 0:
                add_variant("quaternary_N_neutralized", "quaternary N+ -> neutral N", "charged/basic nitrogen region", [idx], neutralize_quaternary_n(mol, idx))
            if deg <= 2:
                add_variant("aliphatic_N_to_O", "aliphatic N -> O", "basic amine / polarity region", [idx], replace_atom(mol, idx, 8))
            add_variant("aliphatic_N_to_C", "aliphatic N -> C", "basic amine / polarity region", [idx], replace_atom(mol, idx, 6))

        if sym == "O" and not aromatic and deg <= 2:
            add_variant("ether_or_alcohol_O_to_CH2", "O -> CH2", "oxygen linker / HBA-HBD region", [idx], replace_atom(mol, idx, 6))

        if sym == "S" and not aromatic and deg <= 2:
            add_variant("thioether_S_to_O", "S -> O", "sulfur linker / polarity region", [idx], replace_atom(mol, idx, 8))
            add_variant("thioether_S_to_CH2", "S -> CH2", "sulfur linker / hydrophobicity region", [idx], replace_atom(mol, idx, 6))

        if sym in {"Cl", "Br", "F"} and deg == 1:
            add_variant(f"halogen_{sym}_removed", f"{sym} removal", "halogenated hydrophobic region", [idx], remove_terminal_atom(mol, idx))

    for ring in mol.GetRingInfo().AtomRings():
        if len(rows) >= max_per_drug:
            break
        atoms = [mol.GetAtomWithIdx(int(i)) for i in ring]
        if any(a.GetIsAromatic() for a in atoms):
            continue
        for a in atoms:
            idx = a.GetIdx()
            if a.GetSymbol() == "C" and a.GetDegree() == 2:
                add_variant("saturated_ring_CH2_to_O", "saturated ring CH2 -> O", "saturated ring polarity/shape region", [idx], replace_atom(mol, idx, 8))
                break

    patt = Chem.MolFromSmarts("[CX3](=O)[NX3,O]")
    for match in mol.GetSubstructMatches(patt):
        if len(rows) >= max_per_drug:
            break
        c_idx = int(match[0])
        add_variant("amide_or_ester_carbonyl_C_to_CH2", "amide/ester carbonyl C -> CH2", "carbonyl linker / HBA region", [c_idx], replace_atom(mol, c_idx, 6))

    return rows


def build_replacement_table(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in candidates.iterrows():
        variants = enumerate_replacements_for_mol(str(r["drug"]), str(r["canonical_smiles"]))
        for v in variants:
            v.update(
                {
                    "task": TASK,
                    "task_sheet": r.get("task_sheet", "TDC.BBB"),
                    "external_label": r["external_label"],
                    "original_input_smiles": r["smiles"],
                    "original_exact_bbb_score": r["trimole_exact_reconstructed_pred_mean"],
                    "original_exact_bbb_score_std": r["trimole_exact_reconstructed_pred_std"],
                }
            )
            rows.append(v)
    df = pd.DataFrame(rows)
    desc = pd.DataFrame([descriptors(s) for s in df["smiles"]])
    df = pd.concat([df.reset_index(drop=True), desc.reset_index(drop=True)], axis=1)
    orig_desc = df[df["replacement"].eq("original")].set_index("drug")
    for col in ["MolWt", "cLogP", "TPSA", "HBD", "HBA", "RotBonds", "HeavyAtoms"]:
        df[f"delta_{col}_vs_original"] = [row[col] - orig_desc.loc[row["drug"], col] for _, row in df.iterrows()]
    return df


def predict_bbb(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    add_paths()
    import descriptor_sidecar_official_v1 as chem
    import run_pure_chem_multibackend_endpoint_v1 as pure

    data_root = EPT_REPO / "data" / "data_benchmark_official_v1"
    task_dir = data_root / TASK
    tr = pd.read_csv(task_dir / "train.csv")
    va = pd.read_csv(task_dir / "valid.csv")
    te = pd.read_csv(task_dir / "test.csv")
    s = pure.smiles_col(tr)
    y = pure.label_col(tr)

    X_tr = chem.get_fingerprints(tr[s])
    X_va = chem.get_fingerprints(va[s])
    X_te = chem.get_fingerprints(te[s])
    X_ext = chem.get_fingerprints(df["smiles"].astype(str))
    y_tr = tr[y].to_numpy()
    y_va = va[y].to_numpy()
    y_te = te[y].to_numpy()
    X_tv = np.concatenate([X_tr, X_va], axis=0)
    y_tv = np.concatenate([y_tr, y_va], axis=0)

    seed_rows = []
    test_preds = []
    ext_preds = []
    for seed in SEEDS:
        model = pure.model_space("classification", seed)["extratrees"]
        model.fit(X_tv, y_tv)
        test_pred = pure.pred(model, X_te, "classification")
        ext_pred = pure.pred(model, X_ext, "classification")
        test_preds.append(test_pred)
        ext_preds.append(ext_pred)
        seed_rows.append({"seed": seed, "test_auroc": float(roc_auc_score(y_te, test_pred))})

    test_mean = np.mean(np.vstack(test_preds), axis=0)
    ext_stack = np.vstack(ext_preds)
    df = df.copy()
    df["bbb_score_mean"] = np.mean(ext_stack, axis=0)
    df["bbb_score_std"] = np.std(ext_stack, axis=0, ddof=1)

    materialized = pd.read_csv(
        ROOT / "results_strict" / "case_study_v36_exact_full" / "full_predictions" / TASK / "test_predictions_full_seed_mean.csv"
    )
    mat_col = "y_prob" if "y_prob" in materialized.columns else "y_pred"
    mat = materialized[mat_col].to_numpy(dtype=float)
    replay = {
        "task": TASK,
        "endpoint_config": "pure_chem_multibackend_endpoint_v1; ExtraTrees; trainvalid refit; seeds 1-5 ensemble",
        "reconstructed_test_auroc": float(roc_auc_score(y_te, test_mean)),
        "materialized_test_auroc": float(roc_auc_score(materialized["y_true"].to_numpy(dtype=float), mat)),
        "max_abs_diff_vs_materialized_seed_mean": float(np.max(np.abs(test_mean - mat))),
        "mean_abs_diff_vs_materialized_seed_mean": float(np.mean(np.abs(test_mean - mat))),
        "status": "exact_reconstruction_pass" if float(np.max(np.abs(test_mean - mat))) < 1e-12 else "reconstruction_mismatch",
        "seed_metrics": seed_rows,
    }
    return df, replay


def summarize(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    orig = preds[preds["replacement"].eq("original")][["drug", "bbb_score_mean"]].rename(columns={"bbb_score_mean": "original_score"})
    out = preds.merge(orig, on="drug", how="left")
    out["score_delta_vs_original"] = out["bbb_score_mean"] - out["original_score"]
    out["abs_score_delta_vs_original"] = out["score_delta_vs_original"].abs()

    candidates = out[~out["replacement"].eq("original")].copy()
    candidates["desired_direction"] = np.where(candidates["external_label"].astype(float) >= 0.5, "score_decrease_from_positive", "score_increase_from_negative")
    candidates["directional_change"] = np.where(
        candidates["external_label"].astype(float) >= 0.5,
        -candidates["score_delta_vs_original"],
        candidates["score_delta_vs_original"],
    )
    candidates["descriptor_shift_penalty"] = (
        candidates["delta_HeavyAtoms_vs_original"].abs() * 0.04
        + candidates["delta_TPSA_vs_original"].abs() * 0.003
        + candidates["delta_MolWt_vs_original"].abs() * 0.001
    )
    candidates["case_score"] = candidates["directional_change"] - candidates["descriptor_shift_penalty"]
    ranking = candidates.sort_values(["case_score", "directional_change"], ascending=False).reset_index(drop=True)

    top_by_drug = (
        ranking.sort_values(["drug", "case_score"], ascending=[True, False])
        .groupby("drug", as_index=False)
        .head(3)
        .sort_values(["case_score", "directional_change"], ascending=False)
        .reset_index(drop=True)
    )
    return out, top_by_drug


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates()
    candidates.to_csv(OUT / "bbb_external_candidates_used.csv", index=False)

    replacement_df = build_replacement_table(candidates)
    replacement_df.to_csv(OUT / "matched_replacement_candidates.csv", index=False)

    preds, replay = predict_bbb(replacement_df)
    full_pred_table, top_by_drug = summarize(preds)
    full_pred_table.to_csv(OUT / "matched_replacement_predictions.csv", index=False)
    top_by_drug.to_csv(OUT / "top_replacement_changes_by_drug.csv", index=False)

    case_ranking = top_by_drug[
        [
            "drug",
            "external_label",
            "original_score",
            "replacement",
            "edit",
            "target_region",
            "source_atom_indices",
            "bbb_score_mean",
            "bbb_score_std",
            "score_delta_vs_original",
            "directional_change",
            "case_score",
            "smiles",
            "delta_MolWt_vs_original",
            "delta_cLogP_vs_original",
            "delta_TPSA_vs_original",
            "delta_HBD_vs_original",
            "delta_HBA_vs_original",
            "delta_RotBonds_vs_original",
            "delta_HeavyAtoms_vs_original",
        ]
    ].copy()
    case_ranking.to_csv(OUT / "matched_replacement_case_ranking.csv", index=False)
    (OUT / "replay_summary.json").write_text(json.dumps(replay, indent=2))

    readme = (
        "# BBB external matched-replacement sensitivity sweep\n\n"
        "This sweep uses the exact reconstructed v36 selected BBB endpoint: "
        "`pure_chem_multibackend_endpoint_v1; ExtraTrees; trainvalid refit; seeds 1-5 ensemble`. "
        "No model is retrained beyond replaying the frozen train+valid refit protocol, and no endpoint selection is repeated.\n\n"
        "Local replacements are RDKit-valid matched-replacement style perturbations over nitrogen, oxygen, sulfur, halogen, "
        "saturated-ring and carbonyl regions. The resulting score changes are prediction-supported sensitivity signals, "
        "not causal chemical mechanisms.\n\n"
        f"```json\n{json.dumps(replay, indent=2)}\n```\n"
    )
    (OUT / "README.md").write_text(readme)

    print(json.dumps(replay, indent=2))
    print("\nTop BBB matched-replacement sensitivity cases")
    print(case_ranking.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
