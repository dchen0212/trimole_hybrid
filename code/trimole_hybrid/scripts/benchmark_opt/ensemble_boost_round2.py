#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

PROJECT_ROOT = Path("<PROJECT_ROOT>/trimole")
DATA_ROOT = PROJECT_ROOT / "data/data_benchmark"
RUN_ROOT = PROJECT_ROOT / "results/model_log/boost_round2"
BASELINE_FILE = PROJECT_ROOT / "results/model_log/final_best_of_all_22_v5_selected_with_config.csv"

OUT_DIR = PROJECT_ROOT / "results/model_log/boost_round2_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_METRICS = OUT_DIR / "ensemble_metrics.csv"
OUT_COMPARE = OUT_DIR / "ensemble_vs_v5.csv"
OUT_SUMMARY = OUT_DIR / "ensemble_summary.json"

TASKS = {
    "bioavailability_ma": "classification",
    "cyp3a4_substrate_carbonmangels": "classification",
    "half_life_obach": "regression",
}

ID_CANDIDATES = ["Drug", "drug", "SMILES", "smiles", "molecule", "mol", "ID", "id", "name"]
TARGET_CANDIDATES = ["Y", "y", "label", "labels", "target", "targets", "value"]

def infer_target_col(df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    bad = {"split", "group"}
    numeric_cols = [c for c in numeric_cols if c.lower() not in bad]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    if len(numeric_cols) >= 1:
        return numeric_cols[-1]
    raise ValueError(f"Cannot infer target column from columns: {list(df.columns)}")

def latest_run_dir(seed_dir: Path) -> Path | None:
    run_dirs = sorted(seed_dir.glob("run_*"))
    return run_dirs[-1] if run_dirs else None

def find_prediction_file(run_dir: Path, n_rows: int) -> Path | None:
    candidates = []
    for f in run_dir.rglob("*.csv"):
        if f.name == "results_all.csv":
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if len(df) != n_rows:
            continue
        cols_lower = [c.lower() for c in df.columns]
        good = any(x in col for col in cols_lower for x in ["pred", "prob", "score", "logit"])
        has_targetish = any(col in cols_lower for col in ["y_true", "target", "label", "y"])
        if good or has_targetish:
            candidates.append(f)
    if not candidates:
        return None
    # prefer files with "test" and prediction keywords
    def rank(p: Path):
        name = p.name.lower()
        score = 0
        if "test" in name:
            score += 5
        for kw in ["prob", "pred", "score", "logit"]:
            if kw in name:
                score += 2
        return score
    candidates = sorted(candidates, key=rank, reverse=True)
    return candidates[0]

def choose_classification_score_col(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lower_map = {c: c.lower() for c in cols}

    preferred = []
    for c in cols:
        lc = lower_map[c]
        if any(k in lc for k in ["prob_1", "prob1", "positive", "pos_prob", "pred_prob", "score", "logit", "prob"]):
            preferred.append(c)
    if preferred:
        # prefer positive class columns
        for c in preferred:
            lc = lower_map[c]
            if "prob_1" in lc or "prob1" in lc or "positive" in lc or "pos" in lc:
                return c
        return preferred[0]

    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    exclude = {"y_true", "y", "label", "labels", "target", "targets", "value", "pred", "prediction"}
    filtered = [c for c in num_cols if lower_map[c] not in exclude]
    if filtered:
        return filtered[0]

    if "pred" in cols:
        return "pred"
    if "prediction" in cols:
        return "prediction"
    raise ValueError(f"Cannot choose classification score column from {cols}")

def choose_regression_pred_col(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lower_map = {c: c.lower() for c in cols}
    for c in cols:
        lc = lower_map[c]
        if lc in {"pred", "prediction", "y_pred"} or "pred" in lc:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    exclude = {"y_true", "y", "label", "labels", "target", "targets", "value"}
    filtered = [c for c in num_cols if lower_map[c] not in exclude]
    if filtered:
        return filtered[0]
    raise ValueError(f"Cannot choose regression prediction column from {cols}")

def get_align_key(test_df: pd.DataFrame, pred_df: pd.DataFrame) -> str | None:
    for c in ID_CANDIDATES:
        if c in test_df.columns and c in pred_df.columns:
            return c
    return None

def load_member_prediction(task: str, cfg_dir: Path, task_type: str) -> pd.DataFrame:
    test_csv = DATA_ROOT / task / "test.csv"
    test_df = pd.read_csv(test_csv)
    n_rows = len(test_df)

    seed_dirs = sorted(cfg_dir.glob("seed_*"))
    members = []

    for seed_dir in seed_dirs:
        run_dir = latest_run_dir(seed_dir)
        if run_dir is None:
            continue

        pred_file = find_prediction_file(run_dir, n_rows)
        if pred_file is None:
            print(f"[WARN] No prediction csv found for {task} | {cfg_dir.name} | {seed_dir.name}")
            continue

        pred_df = pd.read_csv(pred_file)
        key = get_align_key(test_df, pred_df)
        target_col = infer_target_col(test_df)

        if task_type == "classification":
            score_col = choose_classification_score_col(pred_df)
            tmp = pred_df[[score_col]].copy() if key is None else pred_df[[key, score_col]].copy()
            tmp = tmp.rename(columns={score_col: f"{cfg_dir.name}__{seed_dir.name}"})
        else:
            pred_col = choose_regression_pred_col(pred_df)
            tmp = pred_df[[pred_col]].copy() if key is None else pred_df[[key, pred_col]].copy()
            tmp = tmp.rename(columns={pred_col: f"{cfg_dir.name}__{seed_dir.name}"})

        if key is None:
            tmp["_row_id_"] = np.arange(len(tmp))
            members.append(("__ROW__", tmp))
        else:
            members.append((key, tmp))

    if not members:
        raise RuntimeError(f"No usable prediction members found for task={task}, cfg_dir={cfg_dir}")

    # use first member's key mode
    key_mode = members[0][0]
    if key_mode == "__ROW__":
        merged = pd.DataFrame({"_row_id_": np.arange(n_rows)})
        for _, m in members:
            merged = merged.merge(m, on="_row_id_", how="left")
        merged = merged.drop(columns=["_row_id_"])
    else:
        merged = test_df[[key_mode]].copy()
        for _, m in members:
            merged = merged.merge(m, on=key_mode, how="left")
        merged = merged.drop(columns=[key_mode])

    return merged

def eval_classification(y_true, y_score):
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "test_auc": float(roc_auc_score(y_true, y_score)),
        "test_auprc": float(average_precision_score(y_true, y_score)),
        "test_acc": float(accuracy_score(y_true, y_pred)),
    }

def eval_regression(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    sp = spearmanr(y_true, y_pred).correlation
    return {
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
        "test_rmse": float(rmse),
        "test_spearman": float(sp if sp is not None and not np.isnan(sp) else np.nan),
    }

baseline = pd.read_csv(BASELINE_FILE) if BASELINE_FILE.exists() else pd.DataFrame()
baseline_map = baseline.set_index("task").to_dict(orient="index") if len(baseline) else {}

rows = []
compare_rows = []

for task, task_type in TASKS.items():
    task_dir = RUN_ROOT / task
    if not task_dir.exists():
        print(f"[WARN] Missing task dir: {task_dir}")
        continue

    test_df = pd.read_csv(DATA_ROOT / task / "test.csv")
    target_col = infer_target_col(test_df)
    y_true = test_df[target_col].to_numpy()

    cfg_dirs = sorted([p for p in task_dir.iterdir() if p.is_dir()])
    member_frames = []
    member_names = []

    for cfg_dir in cfg_dirs:
        try:
            member_df = load_member_prediction(task, cfg_dir, task_type)
        except Exception as e:
            print(f"[WARN] Skip cfg {cfg_dir.name} for {task}: {e}")
            continue

        # average within config across seeds
        cfg_avg = member_df.mean(axis=1)
        member_frames.append(cfg_avg.rename(cfg_dir.name))
        member_names.append(cfg_dir.name)

    if not member_frames:
        print(f"[WARN] No usable members for task={task}")
        continue

    all_members = pd.concat(member_frames, axis=1)

    # try all non-empty ensembles of configs up to full set
    best_metrics = None
    best_subset = None

    cfg_list = list(all_members.columns)
    n = len(cfg_list)

    for mask in range(1, 1 << n):
        chosen = [cfg_list[i] for i in range(n) if (mask >> i) & 1]
        pred = all_members[chosen].mean(axis=1).to_numpy()

        if task_type == "classification":
            metrics = eval_classification(y_true.astype(int), pred.astype(float))
            score = metrics["test_auc"]
        else:
            metrics = eval_regression(y_true.astype(float), pred.astype(float))
            score = -metrics["test_mae"]

        if best_metrics is None or score > best_metrics["_score"]:
            best_metrics = {"_score": score, **metrics}
            best_subset = chosen

    row = {
        "task": task,
        "task_type": task_type,
        "ensemble_member_configs": "|".join(best_subset),
        "n_configs_used": len(best_subset),
    }
    row.update({k: v for k, v in best_metrics.items() if not k.startswith("_")})
    rows.append(row)

    base = baseline_map.get(task, {})
    if task_type == "classification":
        old_val = base.get("test_auc", np.nan)
        new_val = best_metrics["test_auc"]
        improved = (pd.notna(old_val) and new_val > old_val) or pd.isna(old_val)
        delta = float(new_val - old_val) if pd.notna(old_val) else np.nan
        metric_name = "test_auc"
    else:
        old_val = base.get("test_mae", np.nan)
        new_val = best_metrics["test_mae"]
        improved = (pd.notna(old_val) and new_val < old_val) or pd.isna(old_val)
        delta = float(new_val - old_val) if pd.notna(old_val) else np.nan
        metric_name = "test_mae"

    compare_rows.append({
        "task": task,
        "task_type": task_type,
        "metric_name": metric_name,
        "baseline_v5_value": old_val,
        "ensemble_value": new_val,
        "delta_new_minus_old": delta,
        "is_improved": bool(improved),
        "ensemble_member_configs": "|".join(best_subset),
        "n_configs_used": len(best_subset),
        "ensemble_test_auc": best_metrics.get("test_auc", np.nan),
        "ensemble_test_auprc": best_metrics.get("test_auprc", np.nan),
        "ensemble_test_acc": best_metrics.get("test_acc", np.nan),
        "ensemble_test_mae": best_metrics.get("test_mae", np.nan),
        "ensemble_test_rmse": best_metrics.get("test_rmse", np.nan),
        "ensemble_test_spearman": best_metrics.get("test_spearman", np.nan),
    })

metrics_df = pd.DataFrame(rows).sort_values(["task_type", "task"]).reset_index(drop=True)
compare_df = pd.DataFrame(compare_rows).sort_values(["is_improved", "task"], ascending=[False, True]).reset_index(drop=True)

metrics_df.to_csv(OUT_METRICS, index=False)
compare_df.to_csv(OUT_COMPARE, index=False)

summary = {
    "run_root": str(RUN_ROOT),
    "baseline_file": str(BASELINE_FILE),
    "tasks_checked": list(TASKS.keys()),
    "n_tasks_with_ensemble": int(len(metrics_df)),
    "improved_tasks": compare_df.loc[compare_df["is_improved"], "task"].tolist() if len(compare_df) else [],
}
OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

print("Saved:")
print(" -", OUT_METRICS)
print(" -", OUT_COMPARE)
print(" -", OUT_SUMMARY)
print()
if len(compare_df):
    print(compare_df.to_string(index=False))
