#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("<PROJECT_ROOT>/trimole_hybrid")
EPT_REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
MAIN_OUT = ROOT / "results_strict" / "external_case_perturbation_v1"
ALY_OUT = ROOT / "results_strict" / "alyftrek_minimol_trimole_case_v1"
PGP_RECON_OUT = ALY_OUT / "exact_reconstructed_pgp_v36"

SEEDS = [1, 2, 3, 4, 5]

CASES = [
    {
        "case_id": "k",
        "drug": "Ritlecitinib",
        "task_sheet": "TDC.CYP3A4-S",
        "task": "cyp3a4_substrate_carbonmangels",
        "external_label": 1.0,
        "smiles": "C[C@H]1CC[C@H](CN1C(=O)C=C)NC2=NC=NC3=C2C=CN3",
        "baseline_note": "MiniMol-style 0.49522677 +/- 0.058744248; Trimole exact v36 0.5859176 +/- 0.041352496",
    },
    {
        "case_id": "l",
        "drug": "Mavorixafor",
        "task_sheet": "TDC.Pgp",
        "task": "pgp_broccatelli",
        "external_label": 1.0,
        "smiles": "C1C[C@@H](C2=C(C1)C=CC=N2)N(CCCCN)CC3=NC4=CC=CC=C4N3",
        "baseline_note": "MiniMol-style 0.37249684 +/- 0.046613913; Trimole exact v36 0.5346837671 +/- 0.1314741602",
    },
]

SMARTS = {
    "acrylamide_or_enamide": "C=CC(=O)N",
    "amide": "C(=O)N",
    "primary_amine": "[NX3;H2]",
    "secondary_amine": "[NX3;H1]",
    "tertiary_amine": "[NX3;H0;!$(N=*)]",
    "heteroaromatic_n": "[n]",
    "pyridine_like_n": "[nX2]",
    "aromatic_ring_atom": "a",
    "basic_amine_chain": "NCCCCN",
}


def add_paths() -> None:
    for p in [EPT_REPO / "tools", EPT_REPO, EPT_REPO / "results_strict", ROOT / "tools", ROOT]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def canonicalize(smiles: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def atom_set_to_smiles(mol, atom_ids: tuple[int, ...]) -> str:
    from rdkit import Chem

    if not atom_ids:
        return ""
    try:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=list(atom_ids), canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def remove_atoms_largest_fragment(mol, atom_ids: tuple[int, ...]) -> tuple[bool, str, str]:
    from rdkit import Chem

    keep = set(range(mol.GetNumAtoms())) - set(atom_ids)
    if len(keep) < 5:
        return False, "", "too_few_atoms_after_deletion"
    rw = Chem.RWMol(mol)
    for idx in sorted(atom_ids, reverse=True):
        rw.RemoveAtom(int(idx))
    try:
        pmol = rw.GetMol()
        Chem.SanitizeMol(pmol)
    except Exception as exc:
        return False, "", f"sanitize_failed:{type(exc).__name__}"
    frags = Chem.GetMolFrags(pmol, asMols=True, sanitizeFrags=True)
    if not frags:
        return False, "", "no_fragment"
    frag = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    if frag.GetNumHeavyAtoms() < 5:
        return False, "", "largest_fragment_too_small"
    return True, Chem.MolToSmiles(frag, canonical=True, isomericSmiles=True), "largest_valid_fragment_after_substructure_deletion"


def generate_case_candidates(case: dict, max_candidates: int = 40) -> list[dict]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(case["smiles"])
    if mol is None:
        raise ValueError(f"invalid base smiles for {case['drug']}")
    base = dict(case)
    base["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    rows = []
    rows.append(
        {
            **base,
            "perturbation_id": f"{case['case_id']}_{case['drug']}_original",
            "substructure_type": "original",
            "substructure_name": "original",
            "substructure_smarts_or_smiles": base["canonical_smiles"],
            "atom_indices": "",
            "n_atoms": mol.GetNumAtoms(),
            "perturbation_method": "none",
            "perturbed_smiles": base["canonical_smiles"],
            "perturbation_valid": True,
            "perturbation_note": "original molecule",
        }
    )

    seen_atom_sets: set[tuple[int, ...]] = set()
    candidates: list[tuple[str, str, tuple[int, ...], str]] = []

    for name, smarts in SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt, uniquify=True):
            atom_ids = tuple(sorted(set(int(x) for x in match)))
            if 1 <= len(atom_ids) <= max(8, mol.GetNumAtoms() // 2) and atom_ids not in seen_atom_sets:
                seen_atom_sets.add(atom_ids)
                candidates.append(("functional_group", name, atom_ids, smarts))

    ring_info = mol.GetRingInfo()
    for i, ring in enumerate(ring_info.AtomRings(), start=1):
        atom_ids = tuple(sorted(set(int(x) for x in ring)))
        if atom_ids not in seen_atom_sets:
            seen_atom_sets.add(atom_ids)
            candidates.append(("ring", f"ring_{i}", atom_ids, atom_set_to_smiles(mol, atom_ids)))

    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=bit_info)
    for bit, envs in sorted(bit_info.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True):
        for center, radius in envs[:1]:
            if radius == 0:
                atom_ids = (int(center),)
            else:
                bond_ids = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center)
                atoms = {int(center)}
                for bidx in bond_ids:
                    bond = mol.GetBondWithIdx(int(bidx))
                    atoms.add(bond.GetBeginAtomIdx())
                    atoms.add(bond.GetEndAtomIdx())
                atom_ids = tuple(sorted(atoms))
            if 2 <= len(atom_ids) <= max(8, mol.GetNumAtoms() // 2) and atom_ids not in seen_atom_sets:
                seen_atom_sets.add(atom_ids)
                candidates.append(("morgan_env_r2", f"morgan_bit_{bit}", atom_ids, atom_set_to_smiles(mol, atom_ids)))
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    for idx, (stype, name, atom_ids, pattern) in enumerate(candidates, start=1):
        valid, psmi, note = remove_atoms_largest_fragment(mol, atom_ids)
        rows.append(
            {
                **base,
                "perturbation_id": f"{case['case_id']}_{case['drug']}_perturb_{idx:02d}",
                "substructure_type": stype,
                "substructure_name": name,
                "substructure_smarts_or_smiles": pattern,
                "atom_indices": ";".join(str(x) for x in atom_ids),
                "n_atoms": len(atom_ids),
                "perturbation_method": "delete_substructure_keep_largest_valid_fragment",
                "perturbed_smiles": psmi,
                "perturbation_valid": bool(valid),
                "perturbation_note": note,
            }
        )
    return rows


def generate_candidates(out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for case in CASES:
        rows.extend(generate_case_candidates(case))
    df = pd.DataFrame(rows)
    df["smiles_for_prediction"] = np.where(df["perturbation_valid"], df["perturbed_smiles"], "")
    df.to_csv(out / "perturbation_candidates.csv", index=False)
    valid = df[df["perturbation_valid"]].copy()
    valid.to_csv(out / "perturbation_candidates_valid_only.csv", index=False)
    return df


def predict_cyp3a4(candidates: pd.DataFrame, out: Path) -> pd.DataFrame:
    add_paths()
    import run_cyp_substrate_pooled_family_xgb_v1 as cypxgb
    import descriptor_sidecar_official_v1 as sidecar
    from sklearn.metrics import roc_auc_score

    task = "cyp3a4_substrate_carbonmangels"
    sub = candidates[(candidates["task"] == task) & (candidates["perturbation_valid"])].copy().reset_index(drop=True)
    if sub.empty:
        return pd.DataFrame()
    pred_out = out / "cyp3a4s_exact_reconstructed"
    pred_out.mkdir(parents=True, exist_ok=True)

    data_root = EPT_REPO / "data" / "data_benchmark_official_v1"
    features, meta = cypxgb.build_task_features(EPT_REPO, data_root)
    feat_name = "fp"
    x_pool_train, y_pool_train = cypxgb.pooled_xy(features, meta, feat_name, include_valid=False)
    x_pool_trainvalid, y_pool_trainvalid = cypxgb.pooled_xy(features, meta, feat_name, include_valid=True)
    cfg = cypxgb.xgb_grid(y_pool_train)[2]
    x_valid = features[task][feat_name][1]
    y_valid = meta[task]["y_valid"]
    x_test = features[task][feat_name][2]
    y_test = meta[task]["y_test"]
    task_idx = cypxgb.TASKS.index(task)
    n_tasks = len(cypxgb.TASKS)
    fp_ext = sidecar.get_fingerprints(sub["smiles_for_prediction"].astype(str))
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
        seed_rows.append({"task": task, "seed": seed, "test_auroc": float(roc_auc_score(y_test, tp))})
        seed_df = sub[["case_id", "drug", "task", "external_label", "perturbation_id", "substructure_name", "smiles_for_prediction"]].copy()
        seed_df["seed"] = seed
        seed_df["trimole_pred"] = ep
        seed_df.to_csv(pred_out / f"external_perturbation_predictions_seed_{seed}.csv", index=False)

    test_mean = np.mean(np.vstack(test_preds), axis=0)
    materialized = pd.read_csv(
        ROOT
        / "results_strict"
        / "case_study_v36_exact_full"
        / "full_predictions"
        / task
        / "test_predictions_full_seed_mean.csv"
    )
    mat_col = "y_prob" if "y_prob" in materialized.columns else "y_pred"
    mat = materialized[mat_col].to_numpy(dtype=float)
    max_abs_diff = float(np.max(np.abs(test_mean - mat)))

    ext_mean = np.mean(np.vstack(ext_preds), axis=0)
    ext_std = np.std(np.vstack(ext_preds), axis=0, ddof=1)
    res = sub.copy()
    res["model"] = "Trimole-Hybrid"
    res["prediction_source"] = "exact_selected_endpoint_reconstruction"
    res["endpoint_config"] = "cyp_substrate_pooled_family_xgb_v1; fp; cfg02; trainvalid pooled; seeds 1-5"
    res["official_test_replay_max_abs_diff_vs_materialized"] = max_abs_diff
    res["pred_mean"] = ext_mean
    res["pred_std"] = ext_std
    res["n_runs"] = len(SEEDS)
    res.to_csv(pred_out / "external_perturbation_predictions_cyp3a4s.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(pred_out / "official_replay_seed_metrics.csv", index=False)
    return res


def predict_pgp(candidates: pd.DataFrame, out: Path, python: str) -> pd.DataFrame:
    task = "pgp_broccatelli"
    sub = candidates[(candidates["task"] == task) & (candidates["perturbation_valid"])].copy().reset_index(drop=True)
    if sub.empty:
        return pd.DataFrame()
    PGP_RECON_OUT.mkdir(parents=True, exist_ok=True)
    # Existing PGP reconstructor caches external embeddings by path only, so clear
    # them whenever the candidate set changes.
    for p in [
        PGP_RECON_OUT / "external_embeddings",
        PGP_RECON_OUT / "external_ept",
        PGP_RECON_OUT / "external_ept_task",
        PGP_RECON_OUT / "external_ept_processed",
    ]:
        if p.exists():
            shutil.rmtree(p)
    pgp_cand = sub.drop(columns=["smiles", "canonical_smiles"], errors="ignore").rename(columns={"smiles_for_prediction": "smiles"}).copy()
    pgp_cand["canonical_smiles"] = [canonicalize(s) for s in pgp_cand["smiles"].astype(str)]
    pgp_cand["source_note"] = "external perturbation attribution candidate"
    cols = ["drug", "task_sheet", "task", "external_label", "smiles", "canonical_smiles", "source_note", "perturbation_id"]
    pgp_cand[cols].to_csv(PGP_RECON_OUT / "pgp_external_candidates.csv", index=False)
    script = ROOT / "reconstruct_pgp_exact_external_v36.py"
    if not script.exists():
        raise FileNotFoundError(f"missing {script}")
    subprocess.run([python, str(script)], cwd=str(ROOT), check=True)
    pred = pd.read_csv(PGP_RECON_OUT / "external_predictions_pgp_exact_reconstructed.csv")
    merged = sub.merge(
        pred[["perturbation_id", "trimole_exact_reconstructed_pred_mean", "trimole_exact_reconstructed_pred_std", "n_runs", "prediction_status", "endpoint_config", "exact_v36_external_status"]],
        on="perturbation_id",
        how="left",
    )
    merged["model"] = "Trimole-Hybrid"
    merged["prediction_source"] = "exact_selected_endpoint_reconstruction"
    merged = merged.rename(columns={"trimole_exact_reconstructed_pred_mean": "pred_mean", "trimole_exact_reconstructed_pred_std": "pred_std"})
    pgp_out = out / "pgp_exact_reconstructed"
    pgp_out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(pgp_out / "external_perturbation_predictions_pgp.csv", index=False)
    return merged


def summarize(pred: pd.DataFrame, out: Path) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    rows = []
    for (case_id, task), g in pred.groupby(["case_id", "task"]):
        original = g[g["substructure_type"] == "original"]
        if original.empty:
            continue
        base_pred = float(original["pred_mean"].iloc[0])
        for _, r in g.iterrows():
            delta = abs(float(r["pred_mean"]) - base_pred)
            rows.append(
                {
                    "case_id": case_id,
                    "drug": r["drug"],
                    "task": task,
                    "external_label": r["external_label"],
                    "perturbation_id": r["perturbation_id"],
                    "substructure_type": r["substructure_type"],
                    "substructure_name": r["substructure_name"],
                    "atom_indices": r["atom_indices"],
                    "substructure_smarts_or_smiles": r["substructure_smarts_or_smiles"],
                    "perturbed_smiles": r["smiles_for_prediction"],
                    "original_pred_mean": base_pred,
                    "perturbed_pred_mean": r["pred_mean"],
                    "prediction_delta_abs": delta,
                    "pred_std": r.get("pred_std", np.nan),
                    "attribution_source": r["prediction_source"],
                    "endpoint_config": r.get("endpoint_config", ""),
                    "note": "prediction-supported perturbation sensitivity; not causal chemical proof",
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = summary.sort_values(["case_id", "prediction_delta_abs"], ascending=[True, False])
    summary.to_csv(out / "attribution_delta_summary.csv", index=False)
    top = summary[summary["substructure_type"] != "original"].groupby("case_id", group_keys=False).head(5)
    top.to_csv(out / "top_perturbation_hits.csv", index=False)
    return summary


def write_readme(out: Path, args: argparse.Namespace, cand: pd.DataFrame, pred: pd.DataFrame, summary: pd.DataFrame) -> None:
    status = {
        "n_candidates_total": int(len(cand)),
        "n_valid_perturbations_including_original": int(cand["perturbation_valid"].sum()),
        "ran_cyp3a4_exact": bool(args.run_cyp3a4),
        "ran_pgp_exact": bool(args.run_pgp),
        "n_predictions": int(len(pred)),
        "n_summary_rows": int(len(summary)),
    }
    text = (
        "# External case perturbation attribution v1\n\n"
        "This directory tests whether the two external MiniMol-vs-Trimole cases can support model-based functional-group perturbation analysis.\n\n"
        "Rules used here:\n"
        "- No retraining for endpoint selection.\n"
        "- No data_new.\n"
        "- Perturbations are generated from RDKit-detected substructures, not hand-drawn after looking at predictions.\n"
        "- A highlighted group is only reportable if a valid perturbation changes the frozen selected endpoint prediction.\n"
        "- The output should be described as prediction-supported perturbation sensitivity, not causal chemical attribution.\n\n"
        "Attribution source:\n"
        "- CYP3A4-S uses exact selected endpoint reconstruction: pooled CYP substrate family XGB fingerprint endpoint, seeds 1-5.\n"
        "- P-gp uses exact selected endpoint reconstruction if the PGP reconstructor passes replay and external embedding generation succeeds.\n"
        "- MiniMol perturbation attribution is not run in this script; this script first validates Trimole-side feasibility.\n\n"
        f"```json\n{json.dumps(status, indent=2)}\n```\n"
    )
    out.joinpath("README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(MAIN_OUT))
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--run-cyp3a4", action="store_true")
    parser.add_argument("--run-pgp", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cand = generate_candidates(out)
    if args.generate_only or (not args.run_cyp3a4 and not args.run_pgp):
        write_readme(out, args, cand, pd.DataFrame(), pd.DataFrame())
        print(f"[generate] wrote {out / 'perturbation_candidates.csv'}")
        print(cand.groupby(["drug", "perturbation_valid"]).size().to_string())
        return

    preds = []
    if args.run_cyp3a4:
        preds.append(predict_cyp3a4(cand, out))
    if args.run_pgp:
        preds.append(predict_pgp(cand, out, args.python))
    pred = pd.concat([p for p in preds if p is not None and not p.empty], ignore_index=True) if preds else pd.DataFrame()
    pred.to_csv(out / "trimole_perturbation_predictions.csv", index=False)
    summary = summarize(pred, out)
    write_readme(out, args, cand, pred, summary)
    print(f"[done] out={out}")
    if not summary.empty:
        cols = ["case_id", "drug", "task", "substructure_name", "original_pred_mean", "perturbed_pred_mean", "prediction_delta_abs"]
        print(summary[summary["substructure_type"] != "original"][cols].groupby("case_id", group_keys=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
