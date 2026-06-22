#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("<PROJECT_ROOT>/trimole_hybrid")
EPT_REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
OUT = ROOT / "results_strict" / "cyp3a4s_external_matched_replacement_sweep_v1"
TASK = "cyp3a4_substrate_carbonmangels"
SEEDS = [1, 2, 3, 4, 5]


def add_paths() -> None:
    for p in [EPT_REPO / "tools", EPT_REPO, EPT_REPO / "results_strict", ROOT / "tools", ROOT]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def load_candidates() -> pd.DataFrame:
    paths = [
        ROOT / "results_strict" / "alyftrek_pdf_cyp3a4_candidates.csv",
        ROOT / "results_strict" / "alyftrek_minimol_trimole_case_v1" / "pdf_cyp3a4_external_minimol_vs_trimole_exact_summary.csv",
        ROOT / "results_strict" / "alyftrek_minimol_trimole_case_v1" / "exact_reconstructed_cyp3a4s_v36" / "external_predictions_cyp3a4s_exact_reconstructed.csv",
    ]
    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError("No CYP3A4-S external candidate file found")
    if "drug name" in df.columns:
        df = df.rename(columns={"drug name": "drug", "task1": "task_sheet", "label1": "external_label"})
        df["task"] = TASK
    if "task" in df.columns:
        df = df[df["task"].eq(TASK)].copy()
    df = df[["drug", "smiles", "task_sheet", "task", "external_label"]].drop_duplicates("drug").reset_index(drop=True)
    return df


def canonical_smiles(mol) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def mol_from_smiles(smiles: str):
    from rdkit import Chem

    return Chem.MolFromSmiles(str(smiles))


def safe_sanitize(rw):
    from rdkit import Chem

    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


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


def replace_atom(mol, atom_idx: int, atomic_num: int):
    rw = __import__("rdkit").Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(int(atom_idx))
    atom.SetAtomicNum(int(atomic_num))
    atom.SetFormalCharge(0)
    atom.SetNoImplicit(False)
    atom.SetNumExplicitHs(0)
    return safe_sanitize(rw)


def remove_terminal_atom(mol, atom_idx: int):
    rw = __import__("rdkit").Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(int(atom_idx))
    if atom.GetDegree() != 1:
        return None
    rw.RemoveAtom(int(atom_idx))
    return safe_sanitize(rw)


def enumerate_replacements_for_mol(drug: str, smiles: str, max_per_drug: int = 18) -> list[dict]:
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
        if pmol is None:
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

    # Local bioisosteric-like atom replacements.
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        sym = atom.GetSymbol()
        aromatic = atom.GetIsAromatic()
        deg = atom.GetDegree()

        if sym == "N" and not aromatic:
            if deg <= 2:
                add_variant("aliphatic_N_to_O", "aliphatic N -> O", "aliphatic/basic amine region", [idx], replace_atom(mol, idx, 8))
            add_variant("aliphatic_N_to_C", "aliphatic N -> C", "aliphatic/basic amine region", [idx], replace_atom(mol, idx, 6))

        if sym == "O" and not aromatic and deg <= 2:
            add_variant("ether_or_alcohol_O_to_CH2", "O -> CH2", "oxygen linker/polar region", [idx], replace_atom(mol, idx, 6))

        if sym == "S" and not aromatic and deg <= 2:
            add_variant("thioether_S_to_O", "S -> O", "sulfur linker region", [idx], replace_atom(mol, idx, 8))
            add_variant("thioether_S_to_CH2", "S -> CH2", "sulfur linker region", [idx], replace_atom(mol, idx, 6))

        if sym in {"Cl", "Br", "F"} and deg == 1:
            add_variant(f"halogen_{sym}_to_H", f"{sym} removal", "aryl/alkyl halogen region", [idx], remove_terminal_atom(mol, idx))

        if len(rows) >= max_per_drug:
            break

    # Conservative ring heteroatom replacement for saturated heterocycles.
    for ring in mol.GetRingInfo().AtomRings():
        if len(rows) >= max_per_drug:
            break
        atoms = [mol.GetAtomWithIdx(int(i)) for i in ring]
        if any(a.GetIsAromatic() for a in atoms):
            continue
        for a in atoms:
            idx = a.GetIdx()
            if a.GetSymbol() == "C" and a.GetDegree() == 2:
                add_variant("ring_CH2_to_O", "saturated ring CH2 -> O", "saturated ring shape/polarity region", [idx], replace_atom(mol, idx, 8))
                break

    # Carbonyl polarity replacement is more aggressive but still local.
    patt = Chem.MolFromSmarts("[CX3](=O)[NX3,O]")
    for match in mol.GetSubstructMatches(patt):
        if len(rows) >= max_per_drug:
            break
        c_idx = int(match[0])
        add_variant("carbonyl_C_to_CH2", "amide/ester carbonyl C -> CH2", "carbonyl linker region", [c_idx], replace_atom(mol, c_idx, 6))

    return rows


def build_replacement_table(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in candidates.iterrows():
        variants = enumerate_replacements_for_mol(str(r["drug"]), str(r["smiles"]))
        for v in variants:
            v.update({"task": TASK, "task_sheet": "TDC.CYP3A4-S", "external_label": r["external_label"], "original_input_smiles": r["smiles"]})
            rows.append(v)
    df = pd.DataFrame(rows)
    desc = pd.DataFrame([descriptors(s) for s in df["smiles"]])
    df = pd.concat([df.reset_index(drop=True), desc.reset_index(drop=True)], axis=1)
    orig_desc = df[df["replacement"].eq("original")].set_index("drug")
    for col in ["MolWt", "cLogP", "TPSA", "HBD", "HBA", "RotBonds", "HeavyAtoms"]:
        df[f"delta_{col}_vs_original"] = [
            row[col] - orig_desc.loc[row["drug"], col] for _, row in df.iterrows()
        ]
    return df


def predict(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    add_paths()
    import descriptor_sidecar_official_v1 as sidecar
    import run_cyp_substrate_pooled_family_xgb_v1 as cypxgb
    from sklearn.metrics import roc_auc_score

    data_root = EPT_REPO / "data" / "data_benchmark_official_v1"
    features, meta = cypxgb.build_task_features(EPT_REPO, data_root)
    feat = "fp"
    x_pool_train, y_pool_train = cypxgb.pooled_xy(features, meta, feat, include_valid=False)
    x_pool_trainvalid, y_pool_trainvalid = cypxgb.pooled_xy(features, meta, feat, include_valid=True)
    cfg = cypxgb.xgb_grid(y_pool_train)[2]
    x_valid = features[TASK][feat][1]
    y_valid = meta[TASK]["y_valid"]
    x_test = features[TASK][feat][2]
    y_test = meta[TASK]["y_test"]
    task_idx = cypxgb.TASKS.index(TASK)
    n_tasks = len(cypxgb.TASKS)
    fp_ext = sidecar.get_fingerprints(df["smiles"].astype(str))
    x_ext = cypxgb.add_task_onehot(fp_ext.astype(np.float32), task_idx, n_tasks)

    ext_preds, test_preds, seed_rows = [], [], []
    for seed in SEEDS:
        model = cypxgb.make_model(seed, cfg, n_jobs=8)
        model.fit(x_pool_trainvalid, y_pool_trainvalid, eval_set=[(x_valid, y_valid)], verbose=False)
        tp = model.predict_proba(x_test)[:, 1]
        ep = model.predict_proba(x_ext)[:, 1]
        test_preds.append(tp)
        ext_preds.append(ep)
        seed_rows.append({"seed": seed, "test_auroc": float(roc_auc_score(y_test, tp))})
        seed_df = df[["drug", "replacement_id", "replacement", "smiles"]].copy()
        seed_df["seed"] = seed
        seed_df["pred"] = ep
        seed_df.to_csv(OUT / f"replacement_predictions_seed_{seed}.csv", index=False)

    test_mean = np.mean(np.vstack(test_preds), axis=0)
    mat = pd.read_csv(ROOT / "results_strict" / "case_study_v36_exact_full" / "full_predictions" / TASK / "test_predictions_full_seed_mean.csv")
    mat_col = "y_prob" if "y_prob" in mat.columns else "y_pred"
    mat_pred = mat[mat_col].to_numpy(float)
    replay = {
        "task": TASK,
        "endpoint_config": "cyp_substrate_pooled_family_xgb_v1; fp; cfg02; trainvalid pooled; seeds 1-5",
        "reconstructed_test_auroc": float(roc_auc_score(y_test, test_mean)),
        "materialized_test_auroc": float(roc_auc_score(mat["y_true"].to_numpy(float), mat_pred)),
        "official_test_replay_max_abs_diff_vs_materialized": float(np.max(np.abs(test_mean - mat_pred))),
        "official_test_replay_mean_abs_diff_vs_materialized": float(np.mean(np.abs(test_mean - mat_pred))),
    }
    replay["attribution_source"] = "exact_selected_endpoint_reconstruction" if replay["official_test_replay_max_abs_diff_vs_materialized"] < 1e-6 else "selected_endpoint_family_reconstruction_proxy"

    out = df.copy()
    out["pred_mean"] = np.mean(np.vstack(ext_preds), axis=0)
    out["pred_std"] = np.std(np.vstack(ext_preds), axis=0, ddof=1)
    original = out[out["replacement"].eq("original")].set_index("drug")["pred_mean"]
    out["original_pred_mean"] = out["drug"].map(original)
    out["delta_pred_vs_original"] = out["pred_mean"] - out["original_pred_mean"]
    out["abs_delta_pred_vs_original"] = np.abs(out["delta_pred_vs_original"])
    out["n_runs"] = len(SEEDS)
    out["attribution_source"] = replay["attribution_source"]
    out["endpoint_config"] = replay["endpoint_config"]
    pd.DataFrame(seed_rows).to_csv(OUT / "official_replay_seed_metrics.csv", index=False)
    return out, replay


def score_cases(pred: pd.DataFrame) -> pd.DataFrame:
    variants = pred[~pred["replacement"].eq("original")].copy()
    variants["descriptor_shift_score"] = (
        variants["delta_MolWt_vs_original"].abs() / 80.0
        + variants["delta_cLogP_vs_original"].abs() / 2.0
        + variants["delta_TPSA_vs_original"].abs() / 50.0
        + variants["delta_HBA_vs_original"].abs() / 4.0
        + variants["delta_HBD_vs_original"].abs() / 3.0
    )
    variants["is_descriptor_reasonable"] = (
        (variants["delta_MolWt_vs_original"].abs() <= 80)
        & (variants["delta_cLogP_vs_original"].abs() <= 2.0)
        & (variants["delta_TPSA_vs_original"].abs() <= 50)
        & (variants["delta_HBA_vs_original"].abs() <= 4)
        & (variants["delta_HBD_vs_original"].abs() <= 3)
    )
    variants["is_drop"] = variants["delta_pred_vs_original"] <= -0.15
    variants["is_strong_drop"] = variants["delta_pred_vs_original"] <= -0.25
    rows = []
    for drug, g in variants.groupby("drug"):
        original_pred = float(g["original_pred_mean"].iloc[0])
        reasonable = g[g["is_descriptor_reasonable"]].copy()
        drops = reasonable[reasonable["is_drop"]].copy()
        strong = reasonable[reasonable["is_strong_drop"]].copy()
        best_drop = reasonable.sort_values("delta_pred_vs_original").head(1)
        rows.append(
            {
                "drug": drug,
                "external_label": g["external_label"].iloc[0],
                "original_pred_mean": original_pred,
                "n_replacements": int(len(g)),
                "n_reasonable_replacements": int(len(reasonable)),
                "n_drop_ge_0p15": int(len(drops)),
                "n_drop_ge_0p25": int(len(strong)),
                "best_delta_pred_vs_original": float(best_drop["delta_pred_vs_original"].iloc[0]) if not best_drop.empty else np.nan,
                "best_replacement": best_drop["replacement"].iloc[0] if not best_drop.empty else "",
                "best_edit": best_drop["edit"].iloc[0] if not best_drop.empty else "",
                "best_replacement_smiles": best_drop["smiles"].iloc[0] if not best_drop.empty else "",
                "case_score": float((-best_drop["delta_pred_vs_original"].iloc[0] if not best_drop.empty else 0) + 0.1 * len(drops) + 0.05 * len(strong) + max(original_pred - 0.5, 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_drop_ge_0p15", "case_score"], ascending=[False, False])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates()
    candidates.to_csv(OUT / "cyp3a4s_external_candidates_used.csv", index=False)
    repl = build_replacement_table(candidates)
    repl.to_csv(OUT / "matched_replacement_candidates.csv", index=False)
    pred, replay = predict(repl)
    pred.to_csv(OUT / "matched_replacement_predictions.csv", index=False)
    ranking = score_cases(pred)
    ranking.to_csv(OUT / "matched_replacement_case_ranking.csv", index=False)
    top_variants = pred[~pred["replacement"].eq("original")].sort_values("delta_pred_vs_original").groupby("drug", group_keys=False).head(5)
    top_variants.to_csv(OUT / "top_replacement_drops_by_drug.csv", index=False)
    (OUT / "replay_summary.json").write_text(json.dumps(replay, indent=2))
    (OUT / "README.md").write_text(
        "# CYP3A4-S external matched replacement sweep v1\n\n"
        "This sweep searches external CYP3A4-S molecules for conservative matched-replacement sensitivity cases. "
        "It does not use MiniMol and does not use deletion-only perturbation as evidence. "
        "A potentially useful case should have high original CYP3A4-S score and one or more descriptor-reasonable replacements that reduce the score.\n\n"
        f"```json\n{json.dumps(replay, indent=2)}\n```\n"
    )
    print("\nCase ranking")
    print(ranking.to_string(index=False))
    print("\nTop replacement drops")
    cols = ["drug", "replacement", "edit", "original_pred_mean", "pred_mean", "delta_pred_vs_original", "MolWt", "cLogP", "TPSA", "smiles"]
    print(top_variants[cols].to_string(index=False))
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
