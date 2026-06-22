#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path("<PROJECT_ROOT>/trimole_hybrid")
EPT_REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
OUT = ROOT / "results_strict" / "alyftrek_minimol_trimole_case_v1" / "exact_reconstructed_cyp3a4s_v36"
SHEET_FLAT = ROOT / "results_strict" / "alyftrek_minimol_trimole_case_v1" / "sheet_candidates_flat.csv"
MAIN_OUT = ROOT / "results_strict" / "alyftrek_minimol_trimole_case_v1"
TASK = "cyp3a4_substrate_carbonmangels"
TAG = f"{TASK}__fp__cfg02"
SEEDS = [1, 2, 3, 4, 5]


def add_paths() -> None:
    for p in [EPT_REPO / "tools", EPT_REPO, EPT_REPO / "results_strict", ROOT / "tools", ROOT]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def canonical_smiles(s: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(s))
    return "" if mol is None else Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def main() -> None:
    add_paths()
    import run_cyp_substrate_pooled_family_xgb_v1 as cypxgb
    import descriptor_sidecar_official_v1 as sidecar

    OUT.mkdir(parents=True, exist_ok=True)
    data_root = EPT_REPO / "data" / "data_benchmark_official_v1"
    features, meta = cypxgb.build_task_features(EPT_REPO, data_root)
    feat_name = "fp"
    metric = "auroc"

    x_pool_train, y_pool_train = cypxgb.pooled_xy(features, meta, feat_name, include_valid=False)
    x_pool_trainvalid, y_pool_trainvalid = cypxgb.pooled_xy(features, meta, feat_name, include_valid=True)
    cfg = cypxgb.xgb_grid(y_pool_train)[2]
    x_valid = features[TASK][feat_name][1]
    y_valid = meta[TASK]["y_valid"]
    x_test = features[TASK][feat_name][2]
    y_test = meta[TASK]["y_test"]

    sheet = pd.read_csv(SHEET_FLAT)
    ext = sheet[sheet["task"] == TASK].copy().reset_index(drop=True)
    if ext.empty:
        raise SystemExit("No CYP3A4-S external candidates in sheet_candidates_flat.csv")
    task_idx = cypxgb.TASKS.index(TASK)
    n_tasks = len(cypxgb.TASKS)
    fp_ext = sidecar.get_fingerprints(ext["smiles"].astype(str))
    x_ext = cypxgb.add_task_onehot(fp_ext.astype(np.float32), task_idx, n_tasks)

    seed_rows = []
    ext_preds = []
    test_preds = []
    valid_preds = []
    for seed in SEEDS:
        model = cypxgb.make_model(seed, cfg, n_jobs=8)
        model.fit(x_pool_trainvalid, y_pool_trainvalid, eval_set=[(x_valid, y_valid)], verbose=False)
        vp = model.predict_proba(x_valid)[:, 1]
        tp = model.predict_proba(x_test)[:, 1]
        ep = model.predict_proba(x_ext)[:, 1]
        valid_preds.append(vp)
        test_preds.append(tp)
        ext_preds.append(ep)
        seed_rows.append(
            {
                "task": TASK,
                "seed": seed,
                "valid_auroc": float(roc_auc_score(y_valid, vp)),
                "test_auroc": float(roc_auc_score(y_test, tp)),
            }
        )
        pd.DataFrame({"sample_idx": np.arange(len(y_test)), "y_true": y_test, "y_prob": tp}).to_csv(
            OUT / f"official_test_reconstructed_seed_{seed}.csv", index=False
        )
        seed_ext = ext[["drug", "task_sheet", "task", "external_label", "smiles", "canonical_smiles"]].copy()
        seed_ext["seed"] = seed
        seed_ext["trimole_exact_reconstructed_pred"] = ep
        seed_ext.to_csv(OUT / f"external_predictions_seed_{seed}.csv", index=False)

    test_mean = np.mean(np.vstack(test_preds), axis=0)
    ext_mean = np.mean(np.vstack(ext_preds), axis=0)
    ext_std = np.std(np.vstack(ext_preds), axis=0, ddof=1)
    official_score = float(roc_auc_score(y_test, test_mean))

    materialized = pd.read_csv(
        ROOT
        / "results_strict"
        / "case_study_v36_exact_full"
        / "full_predictions"
        / TASK
        / "test_predictions_full_seed_mean.csv"
    )
    mat_pred_col = "y_prob" if "y_prob" in materialized.columns else "y_pred"
    mat = materialized[mat_pred_col].to_numpy(dtype=float)
    max_abs_diff = float(np.max(np.abs(test_mean - mat)))
    mean_abs_diff = float(np.mean(np.abs(test_mean - mat)))
    mat_score = float(roc_auc_score(materialized["y_true"].to_numpy(dtype=float), mat))

    ext_out = ext[["drug", "task_sheet", "task", "external_label", "smiles", "canonical_smiles"]].copy()
    ext_out["trimole_exact_reconstructed_pred_mean"] = ext_mean
    ext_out["trimole_exact_reconstructed_pred_std"] = ext_std
    ext_out["n_runs"] = len(SEEDS)
    ext_out["prediction_status"] = "exact_v36_selected_endpoint_reconstructed_external_prediction"
    ext_out["endpoint_config"] = "cyp_substrate_pooled_family_xgb_v1; fp; cfg02; trainvalid pooled; seeds 1-5"
    ext_out["official_test_replay_max_abs_diff_vs_materialized"] = max_abs_diff
    ext_out["official_test_replay_mean_abs_diff_vs_materialized"] = mean_abs_diff
    ext_out.to_csv(OUT / "external_predictions_cyp3a4s_exact_reconstructed.csv", index=False)

    pd.DataFrame(seed_rows).to_csv(OUT / "official_replay_seed_metrics.csv", index=False)
    pd.DataFrame({"sample_idx": np.arange(len(y_test)), "y_true": y_test, "y_prob": test_mean}).to_csv(
        OUT / "official_test_reconstructed_seed_mean.csv", index=False
    )

    replay_tol = 1e-6
    summary = {
        "task": TASK,
        "endpoint_config": "cyp_substrate_pooled_family_xgb_v1; fp; cfg02; trainvalid pooled; seeds 1-5",
        "reconstructed_test_auroc": official_score,
        "materialized_test_auroc": mat_score,
        "max_abs_diff_vs_materialized_seed_mean": max_abs_diff,
        "mean_abs_diff_vs_materialized_seed_mean": mean_abs_diff,
        "replay_tolerance": replay_tol,
        "status": "exact_reconstruction_pass" if max_abs_diff < replay_tol else "reconstruction_numeric_mismatch_review_needed",
    }
    (OUT / "README.md").write_text(
        "# CYP3A4-S v36 selected endpoint external reconstruction\n\n"
        "This reconstructs the fixed v36 endpoint `cyp_substrate_pooled_family_xgb_v1; fp; cfg02` "
        "from official train+valid pooled CYP substrate family data and predicts external sheet molecules. "
        "Endpoint selection is not repeated; official test is used only to replay/check materialized v36 predictions.\n\n"
        f"```json\n{json.dumps(summary, indent=2)}\n```\n"
    )
    (OUT / "reconstruction_summary.json").write_text(json.dumps(summary, indent=2))

    # Patch the broader comparison table with exact reconstructed CYP3A4-S predictions.
    comp_path = MAIN_OUT / "comparison_summary.csv"
    comp = pd.read_csv(comp_path)
    for _, r in ext_out.iterrows():
        mask = (comp["drug"] == r["drug"]) & (comp["task"] == TASK)
        comp.loc[mask, "trimole_pred_mean"] = r["trimole_exact_reconstructed_pred_mean"]
        comp.loc[mask, "trimole_pred_std"] = r["trimole_exact_reconstructed_pred_std"]
        comp.loc[mask, "trimole_prediction_status"] = r["prediction_status"]
        comp.loc[mask, "trimole_prediction_source"] = str((OUT / "external_predictions_cyp3a4s_exact_reconstructed.csv").relative_to(ROOT))
        comp.loc[mask, "trimole_exact_v36_external_status"] = summary["status"]
        y = float(r["external_label"])
        pred = float(r["trimole_exact_reconstructed_pred_mean"])
        comp.loc[mask, "trimole_correct"] = (pred >= 0.5) if y >= 0.5 else (pred < 0.5)
        comp.loc[mask, "trimole_margin_to_boundary"] = (pred - 0.5) if y >= 0.5 else (0.5 - pred)
        # Keep original selection logic readable.
        mini_correct = comp.loc[mask, "minimol_correct"].astype(str).iloc[0] == "True"
        tri_correct = bool(comp.loc[mask, "trimole_correct"].iloc[0])
        mini_margin = float(comp.loc[mask, "minimol_margin_to_boundary"].iloc[0])
        tri_margin = float(comp.loc[mask, "trimole_margin_to_boundary"].iloc[0])
        comp.loc[mask, "winner"] = "Trimole" if tri_correct and not mini_correct else "unclear"
        comp.loc[mask, "separation_score"] = (100.0 if tri_correct and not mini_correct else 0.0) + tri_margin - mini_margin
        comp.loc[mask, "main_text_ready"] = True
        comp.loc[mask, "proxy_labeled_candidate"] = False
        comp.loc[mask, "passes_case_threshold"] = bool(tri_correct and not mini_correct)

    patched = MAIN_OUT / "comparison_summary_with_cyp3a4s_exact_reconstructed.csv"
    comp.to_csv(patched, index=False)
    with pd.ExcelWriter(MAIN_OUT / "comparison_summary_with_cyp3a4s_exact_reconstructed.xlsx") as writer:
        comp.to_excel(writer, index=False, sheet_name="comparison")
        ext_out.to_excel(writer, index=False, sheet_name="cyp3a4s_exact_external")
        pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="reconstruction_summary")

    selected = comp[(comp["main_text_ready"] == True) & (comp["passes_case_threshold"] == True)].copy()
    selected.to_csv(MAIN_OUT / "selected_cases_with_cyp3a4s_exact_reconstructed.csv", index=False)
    print(json.dumps(summary, indent=2))
    print("\nCYP3A4-S external exact reconstructed predictions")
    print(ext_out[["drug", "external_label", "trimole_exact_reconstructed_pred_mean", "trimole_exact_reconstructed_pred_std"]].to_string(index=False))
    print("\nSelected exact cases")
    if selected.empty:
        print("none")
    else:
        print(selected[["drug", "task_sheet", "external_label", "minimol_pred_mean", "trimole_pred_mean", "winner", "separation_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
