from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, RDKFingerprint, rdMolDescriptors

import descriptor_sidecar_official_v1 as base
import official_sidecar_bagged_blend_v1 as bagged
import official_sidecar_nested_refit_v1 as nested
import paper_main_multimodal_prior_taskwise_v1 as v1


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
OUT_ROOT = REPO / "results_strict" / "paper_main_chemical_prior_v2"
FOCUS_TASKS = [
    "clearance_microsome_az",
    "ames",
    "hia_hou",
    "pgp_broccatelli",
    "cyp3a4_substrate_carbonmangels",
    "bbb_martins",
    "cyp2d6_substrate_carbonmangels",
    "herg",
    "caco2_wang",
    "cyp3a4_veith",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", default=str(v1.MASTER))
    p.add_argument("--data-root", default=str(v1.DATA_ROOT))
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--base-5seed-roots", nargs="*", default=[str(x) for x in v1.DEFAULT_BASE_ROOTS])
    p.add_argument("--tasks", nargs="*", default=FOCUS_TASKS)
    p.add_argument("--seeds", nargs="*", type=int, default=v1.DEFAULT_SEEDS)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260426)
    p.add_argument("--lambda-std", type=float, default=1.0)
    p.add_argument("--weight-step", type=float, default=0.1)
    p.add_argument("--xgb-estimators", type=int, default=250)
    p.add_argument(
        "--chemical-blocks",
        nargs="*",
        default=["core_maccs_fcfp", "core_pair_torsion", "wide_chem"],
    )
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def fp_to_array(fp) -> np.ndarray:
    if hasattr(fp, "GetLength"):
        n = int(fp.GetLength())
    else:
        n = int(fp.GetNumBits())
    arr = np.zeros((n,), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def mols_from_smiles(smiles: pd.Series) -> list[Chem.Mol]:
    empty = Chem.MolFromSmiles("")
    mols: list[Chem.Mol] = []
    for smi in smiles.astype(str):
        mol = Chem.MolFromSmiles(smi)
        mols.append(mol if mol is not None else empty)
    return mols


def stack_fps(mols: list[Chem.Mol], fn) -> np.ndarray:
    return np.stack([fp_to_array(fn(mol)) for mol in mols], axis=0)


def extra_chemical_blocks(smiles: pd.Series) -> dict[str, np.ndarray]:
    RDLogger.DisableLog("rdApp.*")
    mols = mols_from_smiles(smiles)
    core = base.get_fingerprints(smiles)
    maccs = stack_fps(mols, MACCSkeys.GenMACCSKeys)
    morgan3 = stack_fps(
        mols,
        lambda mol: rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=3, nBits=1024),
    )
    fcfp2 = stack_fps(
        mols,
        lambda mol: rdMolDescriptors.GetHashedMorganFingerprint(
            mol, radius=2, nBits=1024, useFeatures=True
        ),
    )
    atom_pair = stack_fps(
        mols,
        lambda mol: rdMolDescriptors.GetHashedAtomPairFingerprint(mol, nBits=1024),
    )
    torsion = stack_fps(
        mols,
        lambda mol: rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(mol, nBits=1024),
    )
    rdk_path = stack_fps(mols, lambda mol: RDKFingerprint(mol, fpSize=1024))

    blocks = {
        "core": core,
        "core_maccs_fcfp": np.concatenate([core, maccs, fcfp2], axis=1),
        "core_pair_torsion": np.concatenate([core, atom_pair, torsion], axis=1),
        "core_path": np.concatenate([core, rdk_path], axis=1),
        "wide_chem": np.concatenate([core, maccs, morgan3, fcfp2, atom_pair, torsion, rdk_path], axis=1),
    }
    return {name: base.sanitize_features(value.astype(np.float32)) for name, value in blocks.items()}


def build_v2_variants(
    emb_tr: np.ndarray,
    emb_va: np.ndarray,
    emb_te: np.ndarray,
    chem_tr: dict[str, np.ndarray],
    chem_va: dict[str, np.ndarray],
    chem_te: dict[str, np.ndarray],
    pred_tr: np.ndarray | None,
    pred_va: np.ndarray | None,
    pred_te: np.ndarray | None,
    wanted_blocks: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    variants: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for block in wanted_blocks:
        if block not in chem_tr:
            raise KeyError(f"unknown chemical block {block}; available={sorted(chem_tr)}")
        tr, va, te = chem_tr[block], chem_va[block], chem_te[block]
        variants[f"chem_{block}"] = (tr, va, te)
        variants[f"embed_chem_{block}"] = (
            np.concatenate([emb_tr, tr], axis=1),
            np.concatenate([emb_va, va], axis=1),
            np.concatenate([emb_te, te], axis=1),
        )
        if pred_tr is not None and pred_va is not None and pred_te is not None:
            variants[f"chem_{block}_base_pred"] = (
                np.concatenate([tr, pred_tr.reshape(-1, 1)], axis=1),
                np.concatenate([va, pred_va.reshape(-1, 1)], axis=1),
                np.concatenate([te, pred_te.reshape(-1, 1)], axis=1),
            )
            variants[f"embed_chem_{block}_base_pred"] = (
                np.concatenate([emb_tr, tr, pred_tr.reshape(-1, 1)], axis=1),
                np.concatenate([emb_va, va, pred_va.reshape(-1, 1)], axis=1),
                np.concatenate([emb_te, te, pred_te.reshape(-1, 1)], axis=1),
            )
    return {k: tuple(base.sanitize_features(x.astype(np.float32)) for x in mats) for k, mats in variants.items()}


def run_one(row: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    task = row["task"]
    out_dir = Path(args.out_root) / task
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.exists() and not args.force:
        return json.loads(result_path.read_text())

    task_dir = Path(args.data_root) / task
    train_df = pd.read_csv(task_dir / "train.csv")
    valid_df = pd.read_csv(task_dir / "valid.csv")
    test_df = pd.read_csv(task_dir / "test.csv")
    smiles_col = nested.get_smiles_col(train_df)
    label_col = base.find_label_col(train_df)

    n_tr, n_va, n_te = len(train_df), len(valid_df), len(test_df)
    emb_tr, emb_va, emb_te = base.build_embedding_features(task_dir, row["candidate"], n_tr, n_va, n_te)
    chem_tr = extra_chemical_blocks(train_df[smiles_col])
    chem_va = extra_chemical_blocks(valid_df[smiles_col])
    chem_te = extra_chemical_blocks(test_df[smiles_col])

    train_pred_files, valid_pred_files, test_pred_files, pred_source = v1.find_seed_predictions_optional(
        [Path(x) for x in args.base_5seed_roots], task, row["candidate"], [int(x) for x in args.seeds]
    )
    pred_tr = v1.average_optional(train_pred_files)
    pred_va = v1.average_optional(valid_pred_files)
    pred_te = v1.average_optional(test_pred_files)
    has_base_blend = pred_tr is not None and pred_va is not None and pred_te is not None

    variants = build_v2_variants(
        emb_tr,
        emb_va,
        emb_te,
        chem_tr,
        chem_va,
        chem_te,
        pred_tr,
        pred_va,
        pred_te,
        args.chemical_blocks,
    )

    y_tr = train_df[label_col].to_numpy()
    y_va = valid_df[label_col].to_numpy()
    y_te = test_df[label_col].to_numpy()
    y_tv = np.concatenate([y_tr, y_va], axis=0)
    smiles_tv = pd.concat([train_df[smiles_col], valid_df[smiles_col]], ignore_index=True).astype(str).tolist()
    folds = nested.build_scaffold_folds(smiles_tv, args.folds, args.seed)
    task_type = base.infer_task_type(y_tv)
    metric = row["tdc_metric"]
    direction = row["metric_direction"]

    base_tv = base_te = None
    if has_base_blend:
        base_tv = np.concatenate([pred_tr, pred_va], axis=0).astype(np.float32)
        base_te = pred_te.astype(np.float32)

    weight_values = [1.0] if not has_base_blend else list(
        np.round(np.arange(0.0, 1.0 + args.weight_step / 2.0, args.weight_step), 6)
    )
    blend_modes = ["raw"]
    if has_base_blend and str(metric).upper() == "SPEARMAN":
        blend_modes.append("rank")

    cv_rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_adj = -math.inf if direction == "max" else math.inf
    best_mean = -math.inf if direction == "max" else math.inf

    for variant_name, (X_tr, X_va, X_te) in variants.items():
        print(f"[variant] {task}::{variant_name}", flush=True)
        X_tv = np.concatenate([X_tr, X_va], axis=0)
        oof = np.zeros(len(y_tv), dtype=np.float32)
        test_preds: list[np.ndarray] = []
        backend = ""
        for fold_idx, valid_idx in enumerate(folds):
            train_mask = np.ones(len(y_tv), dtype=bool)
            train_mask[valid_idx] = False
            train_idx = np.where(train_mask)[0]
            model, backend = bagged.fit_fold_model(
                X_tv,
                y_tv,
                train_idx,
                valid_idx,
                task_type,
                metric,
                seed=args.seed + fold_idx,
                n_estimators=args.xgb_estimators,
            )
            oof[valid_idx] = base.predict_model(model, X_tv[valid_idx], task_type)
            test_preds.append(base.predict_model(model, X_te, task_type).astype(np.float32))
        bag_test = np.stack(test_preds, axis=0).mean(axis=0).astype(np.float32)

        for mode in blend_modes:
            for weight in weight_values:
                if has_base_blend and base_tv is not None and base_te is not None:
                    oof_eval = v1.blend_prediction(oof, base_tv, float(weight), mode)
                    test_eval = v1.blend_prediction(bag_test, base_te, float(weight), mode)
                else:
                    oof_eval = oof
                    test_eval = bag_test
                fold_scores = v1.score_foldwise(metric, y_tv, oof_eval, folds)
                mean = float(np.mean(fold_scores))
                std = float(np.std(fold_scores, ddof=0))
                adj = v1.adjusted_score(mean, std, direction, args.lambda_std)
                oof_score = float(base.score_metric(metric, y_tv, oof_eval))
                test_score = float(base.score_metric(metric, y_te, test_eval))
                row_out = {
                    "task": task,
                    "variant": variant_name,
                    "blend_mode": mode,
                    "weight_sidecar": float(weight),
                    "tdc_metric": metric,
                    "metric_direction": direction,
                    "cv_mean": mean,
                    "cv_std": std,
                    "cv_adjusted_score": adj,
                    "cv_oof_score": oof_score,
                    "test_tdc_score": test_score,
                    "backend": backend,
                }
                cv_rows.append(row_out)
                if best is None or v1.better_adjusted(adj, best_adj, mean, best_mean, direction):
                    best = dict(row_out)
                    best["trainval_predictions"] = oof_eval
                    best["test_predictions"] = test_eval
                    best_adj = adj
                    best_mean = mean

    if best is None:
        raise RuntimeError(f"no v2 variants evaluated for {task}")

    trainval_pred = np.asarray(best.pop("trainval_predictions"), dtype=np.float32)
    test_pred = np.asarray(best.pop("test_predictions"), dtype=np.float32)
    base.write_predictions(out_dir / "trainval_predictions.csv", y_tv, trainval_pred, task_type)
    base.write_predictions(out_dir / "test_predictions.csv", y_te, test_pred, task_type)
    with (out_dir / "cv_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for x in cv_rows for k in x}))
        writer.writeheader()
        writer.writerows(cv_rows)

    incumbent_test = float(row.get("test_tdc_score_mean") or row.get("test_tdc_score") or row.get("test_score"))
    top1_ref = float(row["tdc_top1_ref"])
    test_score = float(best["test_tdc_score"])
    result = {
        "task": task,
        "candidate": row["candidate"],
        "head": row.get("head", ""),
        "tdc_metric": metric,
        "metric_direction": direction,
        "selected_variant": best["variant"],
        "blend_mode": best["blend_mode"],
        "weight_sidecar": best["weight_sidecar"],
        "cv_mean": best["cv_mean"],
        "cv_std": best["cv_std"],
        "cv_adjusted_score": best["cv_adjusted_score"],
        "cv_oof_score": best["cv_oof_score"],
        "test_tdc_score": test_score,
        "incumbent_test_tdc_score": incumbent_test,
        "improved_test": base.direction_better(test_score, incumbent_test, direction),
        "tdc_top1_ref": top1_ref,
        "is_top1_level": (test_score >= top1_ref if direction == "max" else test_score <= top1_ref),
        "gap_vs_top1_ref": abs(test_score - top1_ref),
        "backend": best["backend"],
        "has_base_blend": has_base_blend,
        "base_pred_source_root": pred_source,
        "endpoint": "chemical_prior_v2_multiblock_scaffold_cv_bagging",
        "chemical_blocks": ",".join(args.chemical_blocks),
        "trainval_pred_file": str(out_dir / "trainval_predictions.csv"),
        "test_pred_file": str(out_dir / "test_predictions.csv"),
        "cv_rows_file": str(out_dir / "cv_rows.csv"),
    }
    result_path.write_text(json.dumps(result, indent=2))
    return result


def write_method_note(out_root: Path, args: argparse.Namespace) -> None:
    text = f"""# Paper-main chemical prior v2

This run keeps the frozen EPT-family multimodal backbone and expands the chemistry-prior branch.

New chemistry blocks:
- core: previous Morgan r2 + Avalon + ErG + RDKit descriptors/fragments
- core_maccs_fcfp: core + MACCS keys + feature Morgan/FCFP
- core_pair_torsion: core + atom-pair + topological torsion fingerprints
- wide_chem: all above plus Morgan r3 and RDKit path fingerprint

Selection:
- scaffold-CV fold bagging
- variant/blend selected by CV mean - lambda * CV std
- official test is not used for model selection

Run config:
- tasks: {args.tasks}
- chemical_blocks: {args.chemical_blocks}
- folds: {args.folds}
- seed: {args.seed}
- lambda_std: {args.lambda_std}
- xgb_estimators: {args.xgb_estimators}
"""
    (out_root / "METHOD.md").write_text(text)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    write_method_note(out_root, args)

    rows = v1.load_master(Path(args.master), args.tasks)
    results: list[dict[str, object]] = []
    for row in rows:
        print(f"[task] {row['task']} candidate={row['candidate']}", flush=True)
        try:
            results.append(run_one(row, args))
        except Exception as exc:
            results.append({"task": row.get("task", ""), "status": "error", "error": str(exc)})
            print(f"[error] {row.get('task', '')}: {exc}", flush=True)

    with (out_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for x in results for k in x}))
        writer.writeheader()
        writer.writerows(results)
    (out_root / "meta.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
