"""Fixed fitting primitives extracted from the original model implementations."""
from __future__ import annotations
from types import SimpleNamespace
import argparse
try:
    import xgboost as _xgb
    base = SimpleNamespace(HAS_XGB=True, XGBClassifier=_xgb.XGBClassifier, XGBRegressor=_xgb.XGBRegressor)
except ImportError:
    base = SimpleNamespace(HAS_XGB=False)
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
v3 = SimpleNamespace(HAS_CATBOOST=HAS_CATBOOST)
if HAS_CATBOOST:
    v3.CatBoostClassifier, v3.CatBoostRegressor = CatBoostClassifier, CatBoostRegressor
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif,f_regression
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
def fit_selector(X: np.ndarray, y: np.ndarray, task_type: str, k: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n_features = X.shape[1]
    if k <= 0 or k >= n_features:
        return np.arange(n_features, dtype=np.int64)

    var = np.nan_to_num(np.var(X, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.flatnonzero(var > 1e-12)
    if len(keep) <= k:
        return keep.astype(np.int64)

    Xk = X[:, keep]
    try:
        if task_type == "classification":
            scores, _ = f_classif(Xk, y)
        else:
            scores, _ = f_regression(Xk, y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        scores = var[keep]
    top_local = np.argsort(scores)[-k:]
    return np.sort(keep[top_local]).astype(np.int64)

def fit_backend(
    backend: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    metric: str,
    seed: int,
    args: argparse.Namespace,
):
    if backend == "xgb":
        if not base.HAS_XGB:
            raise RuntimeError("xgboost unavailable")
        if task_type == "classification":
            pos = float(np.sum(np.asarray(y_train) == 1))
            neg = float(len(y_train) - pos)
            model = base.XGBClassifier(
                n_estimators=args.xgb_estimators,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=2,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cpu",
                random_state=seed,
                n_jobs=1,
                scale_pos_weight=max(1.0, neg / max(pos, 1.0)),
            )
        else:
            model = base.XGBRegressor(
                n_estimators=args.xgb_estimators,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=2,
                objective="reg:squarederror",
                tree_method="hist",
                device="cpu",
                random_state=seed,
                n_jobs=1,
            )
        model.fit(X_train, y_train, verbose=False)
        return model

    if backend == "catboost":
        if not HAS_CATBOOST:
            raise RuntimeError("catboost unavailable")
        common = dict(
            iterations=args.cat_estimators,
            depth=6,
            learning_rate=0.04,
            random_seed=seed,
            verbose=False,
            thread_count=1,
            allow_writing_files=False,
        )
        if task_type == "classification":
            model = CatBoostClassifier(loss_function="Logloss", eval_metric="Logloss", **common)
        else:
            model = CatBoostRegressor(loss_function="RMSE", **common)
        model.fit(X_train, y_train)
        return model

    if backend == "extratrees":
        if task_type == "classification":
            model = ExtraTreesClassifier(
                n_estimators=args.tree_estimators,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            )
        else:
            model = ExtraTreesRegressor(
                n_estimators=args.tree_estimators,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=1,
            )
        model.fit(X_train, y_train)
        return model

    if backend == "rf":
        if task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=args.tree_estimators,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=args.tree_estimators,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=1,
            )
        model.fit(X_train, y_train)
        return model

    if backend == "linear":
        if task_type == "classification":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.5,
                    max_iter=2000,
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

def scaffold_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"invalid::{smiles}"
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

def build_scaffold_folds(smiles: list[str], n_folds: int, seed: int) -> list[np.ndarray]:
    scaffolds: dict[str, list[int]] = {}
    for idx, smi in enumerate(smiles):
        scaffolds.setdefault(scaffold_from_smiles(smi), []).append(idx)

    rng = np.random.default_rng(seed)
    groups = list(scaffolds.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    fold_bins = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    for g in groups:
        tgt = min(range(n_folds), key=lambda i: fold_sizes[i])
        fold_bins[tgt].extend(g)
        fold_sizes[tgt] += len(g)
    return [np.array(sorted(x), dtype=int) for x in fold_bins]
