#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("<PROJECT_ROOT>/trimole_hybrid")
EPT_REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
OUT = ROOT / "results_strict" / "ritlecitinib_matched_replacement_cyp3a4s_v1"
TASK = "cyp3a4_substrate_carbonmangels"
SEEDS = [1, 2, 3, 4, 5]
ORIGINAL = "C[C@H]1CC[C@H](CN1C(=O)C=C)NC2=NC=NC3=C2C=CN3"


def add_paths() -> None:
    for p in [EPT_REPO / "tools", EPT_REPO, EPT_REPO / "results_strict", ROOT / "tools", ROOT]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def canonicalize(mol) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def sanitize_or_none(rw):
    from rdkit import Chem

    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def find_exocyclic_secondary_amine(mol) -> int | None:
    from rdkit import Chem

    patt = Chem.MolFromSmarts("[NX3;H1]([#6;!a])([#6;a])")
    matches = mol.GetSubstructMatches(patt)
    return int(matches[0][0]) if matches else None


def find_acylated_ring_n(mol) -> int | None:
    from rdkit import Chem

    # Tertiary amide N in the piperidine ring, attached to carbonyl C and two aliphatic ring atoms.
    patt = Chem.MolFromSmarts("[NX3;H0]([CX3](=O))([#6;!a])([#6;!a])")
    matches = mol.GetSubstructMatches(patt)
    return int(matches[0][0]) if matches else None


def piperidine_ring_atoms(mol) -> tuple[int, ...]:
    ring_atoms = []
    for ring in mol.GetRingInfo().AtomRings():
        atoms = tuple(int(x) for x in ring)
        symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in atoms]
        if len(atoms) == 6 and symbols.count("N") == 1 and all(mol.GetAtomWithIdx(i).GetIsAromatic() is False for i in atoms):
            ring_atoms.append(atoms)
    return ring_atoms[0] if ring_atoms else tuple()


def replace_atom(mol, atom_idx: int, atomic_num: int, name: str):
    from rdkit import Chem

    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(atom_idx)
    atom.SetAtomicNum(atomic_num)
    atom.SetFormalCharge(0)
    atom.SetNoImplicit(False)
    atom.SetNumExplicitHs(0)
    out = sanitize_or_none(rw)
    if out is None:
        return None
    return {"replacement": name, "smiles": canonicalize(out), "status": "valid"}


def generate_replacements() -> pd.DataFrame:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(ORIGINAL)
    if mol is None:
        raise RuntimeError("Invalid original SMILES")

    rows = [
        {
            "replacement": "original",
            "smiles": canonicalize(mol),
            "status": "valid",
            "edit": "none",
            "target_region": "piperidine/basic-amine candidate sensitivity region",
        }
    ]

    exo_n = find_exocyclic_secondary_amine(mol)
    if exo_n is not None:
        for atomic_num, name, edit in [
            (8, "exocyclic_N_to_O_ether", "secondary anilino N -> O"),
            (6, "exocyclic_N_to_CH2_linker", "secondary anilino N -> CH2"),
        ]:
            r = replace_atom(mol, exo_n, atomic_num, name)
            if r:
                r["edit"] = edit
                r["target_region"] = "exocyclic secondary amine linker"
                rows.append(r)

    ring_n = find_acylated_ring_n(mol)
    if ring_n is not None:
        r = replace_atom(mol, ring_n, 6, "acylated_ring_N_to_CH")
        if r:
            r["edit"] = "amide ring N -> CH"
            r["target_region"] = "acylated piperidine ring nitrogen"
            rows.append(r)

    ring = piperidine_ring_atoms(mol)
    if ring:
        # Pick a ring CH2 carbon with degree two and away from the substituted stereocenters when possible.
        candidates = [
            i
            for i in ring
            if mol.GetAtomWithIdx(i).GetSymbol() == "C"
            and mol.GetAtomWithIdx(i).GetDegree() == 2
            and not mol.GetAtomWithIdx(i).HasProp("_CIPCode")
        ]
        if candidates:
            r = replace_atom(mol, candidates[0], 8, "piperidine_CH2_to_morpholine_O")
            if r:
                r["edit"] = "one piperidine CH2 -> O"
                r["target_region"] = "piperidine ring heteroatom/shape replacement"
                rows.append(r)

    df = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
    return df


def add_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    rows = []
    for _, r in df.iterrows():
        mol = Chem.MolFromSmiles(r["smiles"])
        d = r.to_dict()
        if mol is None:
            d.update({"MolWt": np.nan, "cLogP": np.nan, "TPSA": np.nan, "HBD": np.nan, "HBA": np.nan, "RotBonds": np.nan, "HeavyAtoms": np.nan})
        else:
            d.update(
                {
                    "MolWt": Descriptors.MolWt(mol),
                    "cLogP": Crippen.MolLogP(mol),
                    "TPSA": rdMolDescriptors.CalcTPSA(mol),
                    "HBD": rdMolDescriptors.CalcNumHBD(mol),
                    "HBA": rdMolDescriptors.CalcNumHBA(mol),
                    "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                    "HeavyAtoms": mol.GetNumHeavyAtoms(),
                }
            )
        rows.append(d)
    out = pd.DataFrame(rows)
    orig = out[out["replacement"] == "original"].iloc[0]
    for col in ["MolWt", "cLogP", "TPSA", "HBD", "HBA", "RotBonds", "HeavyAtoms"]:
        out[f"delta_{col}_vs_original"] = out[col] - orig[col]
    return out


def predict_cyp3a4(df: pd.DataFrame) -> pd.DataFrame:
    add_paths()
    import descriptor_sidecar_official_v1 as sidecar
    import run_cyp_substrate_pooled_family_xgb_v1 as cypxgb
    from sklearn.metrics import roc_auc_score

    data_root = EPT_REPO / "data" / "data_benchmark_official_v1"
    features, meta = cypxgb.build_task_features(EPT_REPO, data_root)
    feat_name = "fp"
    x_pool_train, y_pool_train = cypxgb.pooled_xy(features, meta, feat_name, include_valid=False)
    x_pool_trainvalid, y_pool_trainvalid = cypxgb.pooled_xy(features, meta, feat_name, include_valid=True)
    cfg = cypxgb.xgb_grid(y_pool_train)[2]
    x_valid = features[TASK][feat_name][1]
    y_valid = meta[TASK]["y_valid"]
    x_test = features[TASK][feat_name][2]
    y_test = meta[TASK]["y_test"]
    task_idx = cypxgb.TASKS.index(TASK)
    n_tasks = len(cypxgb.TASKS)
    fp_ext = sidecar.get_fingerprints(df["smiles"].astype(str))
    x_ext = cypxgb.add_task_onehot(fp_ext.astype(np.float32), task_idx, n_tasks)

    ext_preds = []
    test_preds = []
    seed_rows = []
    for seed in SEEDS:
        model = cypxgb.make_model(seed, cfg, n_jobs=8)
        model.fit(x_pool_trainvalid, y_pool_trainvalid, eval_set=[(x_valid, y_valid)], verbose=False)
        tp = model.predict_proba(x_test)[:, 1]
        ep = model.predict_proba(x_ext)[:, 1]
        test_preds.append(tp)
        ext_preds.append(ep)
        seed_rows.append({"seed": seed, "test_auroc": float(roc_auc_score(y_test, tp))})
        seed_df = df[["replacement", "smiles"]].copy()
        seed_df["seed"] = seed
        seed_df["pred"] = ep
        seed_df.to_csv(OUT / f"replacement_predictions_seed_{seed}.csv", index=False)

    test_mean = np.mean(np.vstack(test_preds), axis=0)
    mat_path = ROOT / "results_strict" / "case_study_v36_exact_full" / "full_predictions" / TASK / "test_predictions_full_seed_mean.csv"
    mat = pd.read_csv(mat_path)
    mat_col = "y_prob" if "y_prob" in mat.columns else "y_pred"
    mat_pred = mat[mat_col].to_numpy(dtype=float)
    replay = {
        "task": TASK,
        "endpoint_config": "cyp_substrate_pooled_family_xgb_v1; fp; cfg02; trainvalid pooled; seeds 1-5",
        "reconstructed_test_auroc": float(roc_auc_score(y_test, test_mean)),
        "materialized_test_auroc": float(roc_auc_score(mat["y_true"].to_numpy(dtype=float), mat_pred)),
        "official_test_replay_max_abs_diff_vs_materialized": float(np.max(np.abs(test_mean - mat_pred))),
        "official_test_replay_mean_abs_diff_vs_materialized": float(np.mean(np.abs(test_mean - mat_pred))),
        "attribution_source": "selected_endpoint_family_reconstruction_proxy"
        if float(np.max(np.abs(test_mean - mat_pred))) > 1e-6
        else "exact_selected_endpoint_reconstruction",
    }

    pred_mean = np.mean(np.vstack(ext_preds), axis=0)
    pred_std = np.std(np.vstack(ext_preds), axis=0, ddof=1)
    out = df.copy()
    out["pred_mean"] = pred_mean
    out["pred_std"] = pred_std
    original_pred = float(out.loc[out["replacement"] == "original", "pred_mean"].iloc[0])
    out["delta_pred_vs_original"] = out["pred_mean"] - original_pred
    out["abs_delta_pred_vs_original"] = np.abs(out["delta_pred_vs_original"])
    out["n_runs"] = len(SEEDS)
    out["attribution_source"] = replay["attribution_source"]
    out["endpoint_config"] = replay["endpoint_config"]
    out["official_test_replay_max_abs_diff_vs_materialized"] = replay["official_test_replay_max_abs_diff_vs_materialized"]
    pd.DataFrame(seed_rows).to_csv(OUT / "official_replay_seed_metrics.csv", index=False)
    (OUT / "replay_summary.json").write_text(json.dumps(replay, indent=2))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    repl = generate_replacements()
    repl = add_descriptors(repl)
    repl.to_csv(OUT / "ritlecitinib_matched_replacement_candidates.csv", index=False)
    pred = predict_cyp3a4(repl)
    pred.to_csv(OUT / "ritlecitinib_matched_replacement_predictions.csv", index=False)
    md = (
        "# Ritlecitinib matched replacement CYP3A4-S sensitivity\n\n"
        "This analysis makes conservative local matched replacements around the candidate piperidine/basic-amine sensitivity region. "
        "It does not delete the whole molecule and should be interpreted as prediction-supported sensitivity, not causal chemical proof.\n\n"
        "The CYP3A4-S predictor is the same selected endpoint family reconstruction used for external Ritlecitinib prediction. "
        "If replay against the materialized full prediction is not numerically exact, the source is labeled as selected_endpoint_family_reconstruction_proxy.\n"
    )
    (OUT / "README.md").write_text(md)
    cols = [
        "replacement",
        "edit",
        "target_region",
        "pred_mean",
        "pred_std",
        "delta_pred_vs_original",
        "MolWt",
        "cLogP",
        "TPSA",
        "HBD",
        "HBA",
        "RotBonds",
        "smiles",
    ]
    print(pred[cols].to_string(index=False))
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
