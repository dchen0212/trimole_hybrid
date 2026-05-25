#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole_hybrid")
EPT_SWAP = Path("/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1")
EPT_REPO = Path("/mnt/afs/250010150/zhensheng/EPT_try_v1")
EPT_PY = Path("/mnt/afs/250010150/zhensheng/EPT_env_v1/bin/python")
LOCAL_CHEMBERTA = Path("/mnt/afs/250010150/zhensheng/local_hf_models/ChemBERTa-zinc-base-v1")
KPGT_PYTHON = Path("/mnt/afs/250010150/envs/kpgt/bin/python")
KPGT_LIB = KPGT_PYTHON.parent.parent / "lib"
TASK = "pgp_broccatelli"
MAIN_OUT = ROOT / "results_strict" / "alyftrek_minimol_trimole_case_v1"
OUT = MAIN_OUT / "exact_reconstructed_pgp_v36"
DATA_ROOT = EPT_SWAP / "data" / "data_benchmark_official_v1"
MATERIALIZED = EPT_SWAP / "results_strict" / "official_layerwise_selected_5seed_v1" / "_materialized_data" / TASK
FULL_MAT = ROOT / "results_strict" / "case_study_v36_exact_full" / "full_predictions" / TASK / "test_predictions_full_seed_mean.csv"
WEIGHTS = np.array([0.650, 0.250, 0.100], dtype=np.float64)
SEEDS = [1, 2, 3, 4, 5]
REPLAY_ATOL = 2e-6
GROUP_PATTERNS = {
    "layerwise_selected_5seed": "results_strict/official_layerwise_selected_5seed_v1/pgp_broccatelli__chemberta_kpgt_ept_gated__final__seed_{seed}/run_*/pgp_broccatelli",
    "aux_kpgt_ept_5seed": "results_strict/pgp_auxiliary_5seed_routes_v1/pgp_broccatelli__kpgt_ept__seed_{seed}/run_*/pgp_broccatelli",
    "aux_kpgt_5seed": "results_strict/pgp_auxiliary_5seed_routes_v1/pgp_broccatelli__kpgt__seed_{seed}/run_*/pgp_broccatelli",
}
DEFAULT_CANDIDATES = [
    {
        "drug": "Tezacaftor",
        "task_sheet": "TDC.Pgp",
        "task": TASK,
        "external_label": 1.0,
        "smiles": "CC(C)(CO)c1cc2cc(NC(=O)C3(c4ccc5c(c4)OC(F)(F)O5)CC3)c(F)cc2n1C[C@@H](O)CO",
        "source_note": "PDF/FDA candidate; label requires manual confirmation for PGP task semantics",
    },
    {
        "drug": "Deutivacaftor",
        "task_sheet": "TDC.Pgp",
        "task": TASK,
        "external_label": 1.0,
        "smiles": "[2H]C([2H])([2H])C(C1=C(C=C(C(=C1)C(C)(C)C)NC(=O)C2=CNC3=CC=CC=C3C2=O)O)(C([2H])([2H])[2H])C([2H])([2H])[2H]",
        "source_note": "PDF/FDA candidate; label requires manual confirmation for PGP task semantics",
    },
    {
        "drug": "Vanzacaftor",
        "task_sheet": "TDC.Pgp",
        "task": TASK,
        "external_label": 1.0,
        "smiles": "CC1(C)C[C@H]2CN1C1=NC(=CC=C1C(=O)NS(=O)(=O)C1=CC=CC(NCCC2)=N1)N1C=CC(OCCC2C3(CC3)C22CC2)=N1",
        "source_note": "PDF/FDA candidate; label requires manual confirmation for PGP task semantics",
    },
]


def add_paths() -> None:
    # Keep EPT_SWAP ahead of ROOT so replay uses the exact model code family
    # that produced the frozen layerwise-selected artifacts.
    for p in [ROOT / "tools", ROOT, EPT_SWAP / "tools", EPT_SWAP / "results_strict", EPT_SWAP]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def find_run(group: str, seed: int) -> Path:
    paths = sorted(EPT_SWAP.glob(GROUP_PATTERNS[group].format(seed=seed)))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one run for {group} seed={seed}, got {len(paths)}")
    return paths[0]


def load_split_features(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr = pd.read_csv(MATERIALIZED / "train.csv")
    va = pd.read_csv(MATERIALIZED / "valid.csv")
    te = pd.read_csv(MATERIALIZED / "test.csv")
    n_tr, n_va, n_te = len(tr), len(va), len(te)
    chem_all = np.load(MATERIALIZED / "embeddings" / "chemberta.npy").astype(np.float32)
    kpgt_all = np.load(MATERIALIZED / "embeddings" / "kpgt.npy").astype(np.float32)
    ept_map = {
        "train": np.load(MATERIALIZED / "embeddings_ept" / "train_ept.npy").astype(np.float32),
        "valid": np.load(MATERIALIZED / "embeddings_ept" / "valid_ept.npy").astype(np.float32),
        "test": np.load(MATERIALIZED / "embeddings_ept" / "test_ept.npy").astype(np.float32),
    }
    offsets = {"train": (0, n_tr), "valid": (n_tr, n_va), "test": (n_tr + n_va, n_te)}
    off, n = offsets[split]
    return chem_all[off:off+n], ept_map[split], kpgt_all[off:off+n]


def ept_normalizer() -> tuple[np.ndarray, np.ndarray, float]:
    train = np.load(MATERIALIZED / "embeddings_ept" / "train_ept.npy").astype(np.float32)
    train = np.nan_to_num(train, nan=0.0, posinf=0.0, neginf=0.0)
    finite = np.abs(train[np.isfinite(train)])
    clip_value = float(np.percentile(finite, 99.9)) if finite.size else 50.0
    clip_value = float(min(max(clip_value, 50.0), 1e4))
    train = np.clip(train, -clip_value, clip_value)
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std, clip_value


def normalize_ept(arr: np.ndarray, mean: np.ndarray, std: np.ndarray, clip_value: float) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, -clip_value, clip_value)
    arr = (arr - mean) / std
    return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def predict_run(run_dir: Path, chem: np.ndarray, ept: np.ndarray, kpgt: np.ndarray) -> np.ndarray:
    from trimole.models.model import MultiModalFusionMLP

    meta = json.loads((run_dir / "meta.json").read_text())
    cfg = meta["config"]
    dims = meta["dims"]
    model = MultiModalFusionMLP(
        dim_smiles=int(dims["chemberta"]),
        dim_3d=int(dims["ept"]),
        dim_graph=int(dims["kpgt"]),
        out_dim=2,
        hidden_dim=int(cfg["hidden_dim"]),
        dropout_proj=float(cfg["dropout_proj"]),
        dropout_head=float(cfg["dropout_head"]),
        fusion_type=str(cfg["fusion_type"]),
        task_context_dim=int(cfg.get("task_context_dim", 3)),
    )
    state = torch.load(run_dir / "best_model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    modalities = str(cfg.get("modalities", "all"))
    mask_map = {
        "all": None,
        "chemberta": (True, False, False),
        "kpgt": (False, True, False),
        "ept": (False, False, True),
        "chemberta_kpgt": (True, True, False),
        "ept_kpgt": (False, True, True),
        "chemberta_ept": (True, False, True),
    }
    modality_mask = mask_map[modalities]
    outs = []
    with torch.no_grad():
        task_context = None
        if meta.get("task_context_vector") is not None:
            task_context = torch.tensor(meta["task_context_vector"], dtype=torch.float32)
        for start in range(0, len(chem), 256):
            c = torch.from_numpy(chem[start:start+256].astype(np.float32))
            e = torch.from_numpy(ept[start:start+256].astype(np.float32))
            k = torch.from_numpy(kpgt[start:start+256].astype(np.float32))
            logits = model(c, e, k, modality_mask=modality_mask, task_context=task_context)
            prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float64)
            outs.append(prob)
    return np.concatenate(outs, axis=0)


def blend_seed_predictions(seed_parts: list[np.ndarray]) -> np.ndarray:
    return sigmoid(np.tensordot(WEIGHTS, np.stack([logit(x) for x in seed_parts], axis=0), axes=(0, 0)))


def canonicalize(smiles: str) -> str:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(str(smiles))
    return "" if mol is None else Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def load_candidates() -> pd.DataFrame:
    cand_path = OUT / "pgp_external_candidates.csv"
    if cand_path.exists():
        df = pd.read_csv(cand_path)
    else:
        df = pd.DataFrame(DEFAULT_CANDIDATES)
        df["canonical_smiles"] = [canonicalize(s) for s in df["smiles"]]
        cand_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cand_path, index=False)
    if "canonical_smiles" not in df.columns:
        df["canonical_smiles"] = [canonicalize(s) for s in df["smiles"]]
    return df


def leakage_audit(ext: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ["train", "valid", "test"]:
        d = pd.read_csv(DATA_ROOT / TASK / f"{split}.csv")
        canon_set = set(canonicalize(s) for s in d["smiles"].astype(str))
        for _, r in ext.iterrows():
            rows.append({"drug": r["drug"], "split": split, "overlap": r["canonical_smiles"] in canon_set})
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "leakage_audit_pgp.csv", index=False)
    return audit


def ensure_external_chemberta(ext: pd.DataFrame) -> np.ndarray:
    p = OUT / "external_embeddings" / "chemberta.npy"
    if p.exists():
        return np.load(p).astype(np.float32)
    from trimole.embeddings.chemberta import build_chemberta
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = build_chemberta(
        ext["smiles"].astype(str).tolist(),
        device="cuda",
        batch_size=8,
        model_name_or_path=str(LOCAL_CHEMBERTA),
        local_files_only=True,
    )
    np.save(p, arr.astype(np.float32))
    return arr.astype(np.float32)


def ensure_external_kpgt(ext: pd.DataFrame) -> np.ndarray:
    p = OUT / "external_embeddings" / "kpgt.npy"
    if p.exists():
        return np.load(p).astype(np.float32)
    from trimole.embeddings.kpgt import build_kpgt_from_smiles_list
    p.parent.mkdir(parents=True, exist_ok=True)
    ld_parts = [str(KPGT_LIB)]
    if os.environ.get("LD_LIBRARY_PATH"):
        ld_parts.append(os.environ["LD_LIBRARY_PATH"])
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    build_kpgt_from_smiles_list(
        ext["smiles"].astype(str).tolist(),
        p,
        n_jobs=1,
        num_workers=0,
        batch_size=32,
        kpgt_python=str(KPGT_PYTHON),
        log_path=OUT / "external_kpgt.log",
        dataset_name="pgp_external",
    )
    return np.load(p).astype(np.float32)


def ensure_external_ept(ext: pd.DataFrame) -> np.ndarray:
    emb_path = OUT / "external_ept" / "test_ept.npy"
    if emb_path.exists():
        return np.load(emb_path).astype(np.float32)
    task_tmp = OUT / "external_ept_task"
    processed = OUT / "external_ept_processed"
    emb_dir = OUT / "external_ept"
    for p in [task_tmp, processed, emb_dir]:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
    task_df = pd.DataFrame({"smiles": ext["smiles"].astype(str).tolist(), "label": ext["external_label"].astype(float).tolist()})
    for split in ["train", "valid", "test"]:
        task_df.to_csv(task_tmp / f"{split}.csv", index=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{EPT_REPO}:{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [str(EPT_PY), str(EPT_REPO / "scripts" / "process_tdc_smiles.py"), "--task-dir", str(task_tmp), "--out-dir", str(processed), "--fast-mode"],
        cwd=str(EPT_REPO), env=env, check=True,
        stdout=(OUT / "external_ept_process.log").open("w"), stderr=subprocess.STDOUT,
    )
    ckpt = EPT_REPO / "ckpts_tdc" / "pgp_broccatelli_full" / "version_0" / "checkpoint" / "epoch37_step1026.ckpt"
    subprocess.run(
        [str(EPT_PY), str(EPT_REPO / "scripts" / "ept_export_embeddings_tdc.py"), "--repo", str(EPT_REPO), "--processed-dir", str(processed), "--out-dir", str(emb_dir), "--ckpt", str(ckpt), "--task-type", "classification", "--batch-size", "16", "--device", "cpu"],
        cwd=str(EPT_REPO), env=env, check=True,
        stdout=(OUT / "external_ept_export.log").open("w"), stderr=subprocess.STDOUT,
    )
    return np.load(emb_path).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()

    add_paths()
    OUT.mkdir(parents=True, exist_ok=True)

    y_test = pd.read_csv(DATA_ROOT / TASK / "test.csv")["label"].to_numpy(dtype=int)
    chem_te, ept_te_raw, kpgt_te = load_split_features("test")
    ept_mean, ept_std, ept_clip = ept_normalizer()
    ept_te = normalize_ept(ept_te_raw, ept_mean, ept_std, ept_clip)
    seed_rows = []
    test_seed_blends = []
    source_replay_rows = []
    for seed in SEEDS:
        group_preds = []
        for group in GROUP_PATTERNS:
            run_dir = find_run(group, seed)
            pred = predict_run(run_dir, chem_te, ept_te, kpgt_te)
            src = pd.read_csv(run_dir / "test_predictions.csv").sort_values("sample_idx").reset_index(drop=True)
            src_pred = src["y_prob"].to_numpy(dtype=float)
            source_replay_rows.append({
                "seed": seed, "group": group, "source_run_dir": str(run_dir),
                "max_abs_diff_vs_source_test_predictions": float(np.max(np.abs(pred - src_pred))),
                "mean_abs_diff_vs_source_test_predictions": float(np.mean(np.abs(pred - src_pred))),
            })
            group_preds.append(pred)
        blend = blend_seed_predictions(group_preds)
        auc = float(roc_auc_score(y_test, blend))
        seed_rows.append({"seed": seed, "test_auroc": auc})
        test_seed_blends.append(blend)
        pd.DataFrame({"sample_idx": np.arange(len(y_test)), "y_true": y_test, "y_prob": blend}).to_csv(OUT / f"official_test_reconstructed_seed_{seed}.csv", index=False)

    test_mean = np.mean(np.vstack(test_seed_blends), axis=0)
    score = float(roc_auc_score(y_test, test_mean))
    mat = pd.read_csv(FULL_MAT).sort_values("sample_idx").reset_index(drop=True)
    mat_pred = mat["y_prob"].to_numpy(dtype=float)
    max_diff = float(np.max(np.abs(test_mean - mat_pred)))
    mean_diff = float(np.mean(np.abs(test_mean - mat_pred)))
    mat_score = float(roc_auc_score(mat["y_true"].to_numpy(dtype=int), mat_pred))
    pd.DataFrame(source_replay_rows).to_csv(OUT / "source_component_replay_audit.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(OUT / "official_replay_seed_metrics.csv", index=False)
    pd.DataFrame({"sample_idx": np.arange(len(y_test)), "y_true": y_test, "y_prob": test_mean}).to_csv(OUT / "official_test_reconstructed_seed_mean.csv", index=False)

    replay_pass = (
        max_diff < REPLAY_ATOL
        and max(r["max_abs_diff_vs_source_test_predictions"] for r in source_replay_rows) < REPLAY_ATOL
    )

    if args.replay_only:
        summary = {
            "task": TASK,
            "endpoint_config": "pgp_frozen_blend_audit_v1; logit blend 0.65*layerwise C+K+E gated + 0.25*K+E + 0.10*K; seeds 1-5",
            "reconstructed_test_auroc": score,
            "materialized_test_auroc": mat_score,
            "max_abs_diff_vs_materialized_seed_mean": max_diff,
            "mean_abs_diff_vs_materialized_seed_mean": mean_diff,
            "max_component_source_diff": max(r["max_abs_diff_vs_source_test_predictions"] for r in source_replay_rows),
            "status": "exact_reconstruction_pass" if replay_pass else "reconstruction_numeric_mismatch_review_needed",
            "mode": "replay_only",
        }
        (OUT / "replay_only_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    ext = load_candidates()
    audit = leakage_audit(ext)
    if audit[(audit["split"] == "train") & (audit["overlap"] == True)].any().any():
        raise RuntimeError("PGP external candidates include train overlap; refusing prediction")

    chem_ext = ensure_external_chemberta(ext)
    kpgt_ext = ensure_external_kpgt(ext)
    ept_ext = normalize_ept(ensure_external_ept(ext), ept_mean, ept_std, ept_clip)
    ext_seed_blends = []
    for seed in SEEDS:
        group_preds = []
        for group in GROUP_PATTERNS:
            run_dir = find_run(group, seed)
            group_preds.append(predict_run(run_dir, chem_ext, ept_ext, kpgt_ext))
        seed_blend = blend_seed_predictions(group_preds)
        ext_seed_blends.append(seed_blend)
        seed_out = ext.copy()
        seed_out["seed"] = seed
        seed_out["trimole_exact_reconstructed_pred"] = seed_blend
        seed_out.to_csv(OUT / f"external_predictions_seed_{seed}.csv", index=False)

    ext_mean = np.mean(np.vstack(ext_seed_blends), axis=0)
    ext_std = np.std(np.vstack(ext_seed_blends), axis=0, ddof=1)
    status = "exact_reconstruction_pass" if replay_pass else "reconstruction_numeric_mismatch_review_needed"
    ext_out = ext.copy()
    ext_out["trimole_exact_reconstructed_pred_mean"] = ext_mean
    ext_out["trimole_exact_reconstructed_pred_std"] = ext_std
    ext_out["n_runs"] = len(SEEDS)
    ext_out["prediction_status"] = "exact_v36_selected_endpoint_reconstructed_external_prediction" if status == "exact_reconstruction_pass" else "reconstructed_external_prediction_but_official_replay_mismatch"
    ext_out["endpoint_config"] = "pgp_frozen_blend_audit_v1; logit blend 0.65*layerwise C+K+E gated + 0.25*K+E + 0.10*K; seeds 1-5"
    ext_out["official_test_replay_max_abs_diff_vs_materialized"] = max_diff
    ext_out["official_test_replay_mean_abs_diff_vs_materialized"] = mean_diff
    ext_out["exact_v36_external_status"] = status
    ext_out.to_csv(OUT / "external_predictions_pgp_exact_reconstructed.csv", index=False)

    summary = {
        "task": TASK,
        "endpoint_config": "pgp_frozen_blend_audit_v1; logit blend 0.65*layerwise C+K+E gated + 0.25*K+E + 0.10*K; seeds 1-5",
        "reconstructed_test_auroc": score,
        "materialized_test_auroc": mat_score,
        "max_abs_diff_vs_materialized_seed_mean": max_diff,
        "mean_abs_diff_vs_materialized_seed_mean": mean_diff,
        "max_component_source_diff": max(r["max_abs_diff_vs_source_test_predictions"] for r in source_replay_rows),
        "status": status,
        "n_external_candidates": int(len(ext_out)),
        "train_overlap_count": int(audit[(audit["split"] == "train") & (audit["overlap"] == True)].shape[0]),
    }
    (OUT / "reconstruction_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "README.md").write_text(
        "# PGP v36 exact external reconstruction\n\n"
        "This reconstructs the frozen v36 PGP endpoint from saved component models and regenerates ChemBERTa/KPGT/EPT embeddings for external SMILES. "
        "Endpoint selection is not repeated. Official test labels are used only to verify replay against materialized v36 predictions.\n\n"
        f"```json\n{json.dumps(summary, indent=2)}\n```\n"
        "\nExternal labels in the default candidate file are placeholders from the FDA/PDF case-study queue and must be manually confirmed against the exact PGP task semantics before manuscript use.\n"
    )
    print(json.dumps(summary, indent=2))
    print("\nExternal PGP predictions")
    print(ext_out[["drug", "external_label", "trimole_exact_reconstructed_pred_mean", "trimole_exact_reconstructed_pred_std", "exact_v36_external_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
