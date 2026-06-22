from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, RDKFingerprint, rdMolDescriptors
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import descriptor_sidecar_official_v1 as base
import paper_main_chem_select_multibackend_v3 as v3


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
OUT_ROOT = REPO / "results_strict" / "paper_main_chemical_prior_xl_v4"
XL_FOCUS_TASKS = [
    "ames",
    "clearance_microsome_az",
    "cyp2c9_substrate_carbonmangels",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "pgp_broccatelli",
    "hia_hou",
    "bbb_martins",
    "solubility_aqsoldb",
    "caco2_wang",
]
XL_BLOCKS = [
    "xl_morgan_family",
    "xl_topology_family",
    "xl_full_chemical_prior",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", default=str(v3.v1.MASTER))
    p.add_argument("--data-root", default=str(v3.v1.DATA_ROOT))
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--base-5seed-roots", nargs="*", default=[str(x) for x in v3.v1.DEFAULT_BASE_ROOTS])
    p.add_argument("--tasks", nargs="*", default=[])
    p.add_argument("--seeds", nargs="*", type=int, default=v3.v1.DEFAULT_SEEDS)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260426)
    p.add_argument("--lambda-std", type=float, default=1.0)
    p.add_argument("--weight-step", type=float, default=0.1)
    p.add_argument("--xgb-estimators", type=int, default=600)
    p.add_argument("--tree-estimators", type=int, default=900)
    p.add_argument("--cat-estimators", type=int, default=700)
    p.add_argument("--n-jobs", type=int, default=32)
    p.add_argument("--fp-bits", type=int, default=2048)
    p.add_argument("--topk", nargs="*", type=int, default=[256, 512, 1024, 2048, 4096, 8192])
    p.add_argument("--backends", nargs="*", default=["xgb", "catboost", "extratrees", "rf", "linear"])
    p.add_argument("--chemical-blocks", nargs="*", default=XL_BLOCKS)
    p.add_argument("--variants", nargs="*", default=["chem", "embed_chem", "chem_base_pred", "embed_chem_base_pred"])
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def fp_to_array(fp) -> np.ndarray:
    n = int(fp.GetLength()) if hasattr(fp, "GetLength") else int(fp.GetNumBits())
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


def extra_chemical_blocks_xl(smiles: pd.Series) -> dict[str, np.ndarray]:
    RDLogger.DisableLog("rdApp.*")
    n_bits = int(getattr(v3, "XL_FP_BITS", 2048))
    mols = mols_from_smiles(smiles)

    core = base.get_fingerprints(smiles)
    maccs = stack_fps(mols, MACCSkeys.GenMACCSKeys)
    morgan1 = stack_fps(mols, lambda m: rdMolDescriptors.GetHashedMorganFingerprint(m, radius=1, nBits=n_bits))
    morgan2 = stack_fps(mols, lambda m: rdMolDescriptors.GetHashedMorganFingerprint(m, radius=2, nBits=n_bits))
    morgan3 = stack_fps(mols, lambda m: rdMolDescriptors.GetHashedMorganFingerprint(m, radius=3, nBits=n_bits))
    fcfp2 = stack_fps(
        mols,
        lambda m: rdMolDescriptors.GetHashedMorganFingerprint(m, radius=2, nBits=n_bits, useFeatures=True),
    )
    fcfp3 = stack_fps(
        mols,
        lambda m: rdMolDescriptors.GetHashedMorganFingerprint(m, radius=3, nBits=n_bits, useFeatures=True),
    )
    atom_pair = stack_fps(mols, lambda m: rdMolDescriptors.GetHashedAtomPairFingerprint(m, nBits=n_bits))
    torsion = stack_fps(mols, lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(m, nBits=n_bits))
    rdk_path = stack_fps(mols, lambda m: RDKFingerprint(m, fpSize=n_bits))
    layered = stack_fps(mols, lambda m: Chem.LayeredFingerprint(m, fpSize=n_bits))
    pattern = stack_fps(mols, lambda m: Chem.PatternFingerprint(m, fpSize=n_bits))

    morgan_family = np.concatenate([core, maccs, morgan1, morgan2, morgan3, fcfp2, fcfp3], axis=1)
    topology_family = np.concatenate([core, maccs, atom_pair, torsion, rdk_path, layered, pattern], axis=1)
    full = np.concatenate(
        [
            core,
            maccs,
            morgan1,
            morgan2,
            morgan3,
            fcfp2,
            fcfp3,
            atom_pair,
            torsion,
            rdk_path,
            layered,
            pattern,
        ],
        axis=1,
    )
    return {
        "xl_morgan_family": base.sanitize_features(morgan_family.astype(np.float32)),
        "xl_topology_family": base.sanitize_features(topology_family.astype(np.float32)),
        "xl_full_chemical_prior": base.sanitize_features(full.astype(np.float32)),
    }


def fit_backend_xl(
    backend: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    metric: str,
    seed: int,
    args: argparse.Namespace,
):
    n_jobs = max(1, int(getattr(args, "n_jobs", 1)))
    if backend == "xgb":
        if not base.HAS_XGB:
            raise RuntimeError("xgboost unavailable")
        if task_type == "classification":
            pos = float(np.sum(np.asarray(y_train) == 1))
            neg = float(len(y_train) - pos)
            model = base.XGBClassifier(
                n_estimators=args.xgb_estimators,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.75,
                min_child_weight=2,
                reg_lambda=2.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cpu",
                random_state=seed,
                n_jobs=n_jobs,
                scale_pos_weight=max(1.0, neg / max(pos, 1.0)),
            )
        else:
            model = base.XGBRegressor(
                n_estimators=args.xgb_estimators,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.75,
                min_child_weight=2,
                reg_lambda=2.0,
                objective="reg:squarederror",
                tree_method="hist",
                device="cpu",
                random_state=seed,
                n_jobs=n_jobs,
            )
        model.fit(X_train, y_train, verbose=False)
        return model

    if backend == "catboost":
        if not v3.HAS_CATBOOST:
            raise RuntimeError("catboost unavailable")
        common = dict(
            iterations=args.cat_estimators,
            depth=6,
            learning_rate=0.03,
            random_seed=seed,
            verbose=False,
            thread_count=n_jobs,
            allow_writing_files=False,
        )
        if task_type == "classification":
            model = v3.CatBoostClassifier(loss_function="Logloss", eval_metric="Logloss", **common)
        else:
            model = v3.CatBoostRegressor(loss_function="RMSE", **common)
        model.fit(X_train, y_train)
        return model

    if backend == "extratrees":
        cls = ExtraTreesClassifier if task_type == "classification" else ExtraTreesRegressor
        kwargs = dict(
            n_estimators=args.tree_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=n_jobs,
        )
        if task_type == "classification":
            kwargs["class_weight"] = "balanced"
        model = cls(**kwargs)
        model.fit(X_train, y_train)
        return model

    if backend == "rf":
        cls = RandomForestClassifier if task_type == "classification" else RandomForestRegressor
        kwargs = dict(
            n_estimators=args.tree_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=n_jobs,
        )
        if task_type == "classification":
            kwargs["class_weight"] = "balanced"
        model = cls(**kwargs)
        model.fit(X_train, y_train)
        return model

    if backend == "linear":
        if task_type == "classification":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.5,
                    max_iter=3000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=seed,
                ),
            )
        else:
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=seed))
        model.fit(X_train, y_train)
        return model

    raise ValueError(f"unknown backend: {backend}")


def main() -> None:
    args = parse_args()
    v3.XL_FP_BITS = int(args.fp_bits)
    v3.parse_args = lambda: args
    v3.chemv2.extra_chemical_blocks = extra_chemical_blocks_xl
    v3.fit_backend = fit_backend_xl
    v3.main()


if __name__ == "__main__":
    main()
