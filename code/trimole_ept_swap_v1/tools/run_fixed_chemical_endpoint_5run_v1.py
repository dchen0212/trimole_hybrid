from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


REPO = Path("/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1")
for _path in (REPO, REPO / "tools", REPO / "results_strict"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import descriptor_sidecar_official_v1 as base
import official_sidecar_nested_refit_v1 as nested
import paper_main_chem_select_multibackend_v3 as v3
import paper_main_chemical_prior_xl_v4 as v4
import paper_main_multimodal_prior_taskwise_v1 as v1


DATA_ROOT = REPO / "data" / "data_benchmark_official_v1"
RESULTS = REPO / "results_strict"
OUT = RESULTS / "fixed_chemical_endpoint_5run_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--out-root", default=str(OUT))
    p.add_argument("--summary", required=True, help="Path relative to results_strict or absolute summary.csv.")
    p.add_argument("--tasks", nargs="*", default=[])
    p.add_argument("--formal-seeds", nargs="*", type=int, default=[101, 202, 303, 404, 505])
    p.add_argument("--base-seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--fp-bits", type=int, default=2048)
    p.add_argument("--xgb-estimators", type=int, default=600)
    p.add_argument("--tree-estimators", type=int, default=900)
    p.add_argument("--cat-estimators", type=int, default=700)
    p.add_argument("--n-jobs", type=int, default=16)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def as_path(repo: Path, summary: str) -> Path:
    p = Path(summary)
    if p.is_absolute():
        return p
    return repo / "results_strict" / summary


def norm_topk(x: object) -> int:
    s = str(x)
    if s.lower() == "all" or s == "0":
        return 0
    return int(float(s))


def load_rows(summary_path: Path, tasks: list[str]) -> list[dict[str, str]]:
    rows = list(csv.DictReader(summary_path.open()))
    if tasks:
        wanted = set(tasks)
        rows = [r for r in rows if r.get("task") in wanted]
    return rows


def is_xl_row(row: dict[str, str], summary_path: Path) -> bool:
    return "xl_v4" in str(summary_path) or "xl_" in row.get("selected_variant", "")


def read_task_frames(task: str):
    task_dir = DATA_ROOT / task
    train_df = pd.read_csv(task_dir / "train.csv")
    valid_df = pd.read_csv(task_dir / "valid.csv")
    test_df = pd.read_csv(task_dir / "test.csv")
    return task_dir, train_df, valid_df, test_df


def build_variant_for_row(row: dict[str, str], xl: bool, args: argparse.Namespace):
    task = row["task"]
    task_dir, train_df, valid_df, test_df = read_task_frames(task)
    smiles_col = nested.get_smiles_col(train_df)
    label_col = base.find_label_col(train_df)
    n_tr, n_va, n_te = len(train_df), len(valid_df), len(test_df)

    selected_variant = row["selected_variant"]
    candidate = row["candidate"]
    if candidate.startswith("xl_v4_"):
        candidate = candidate[len("xl_v4_") :]
    block = selected_variant
    for prefix in ("embed_chem_", "chem_"):
        if block.startswith(prefix):
            block = block[len(prefix) :]
    if block.endswith("_base_pred"):
        block = block[: -len("_base_pred")]

    if xl:
        v3.XL_FP_BITS = int(args.fp_bits)
        chem_func = v4.extra_chemical_blocks_xl
        fit_backend = v4.fit_backend_xl
    else:
        chem_func = v3.chemv2.extra_chemical_blocks
        fit_backend = v3.fit_backend

    chem_tr = chem_func(train_df[smiles_col])
    chem_va = chem_func(valid_df[smiles_col])
    chem_te = chem_func(test_df[smiles_col])
    if block not in chem_tr:
        raise KeyError(f"{task}: block {block!r} not in chemical blocks {sorted(chem_tr)}")

    needs_embedding = selected_variant.startswith("embed_chem_")
    emb_tr = emb_va = emb_te = None
    if needs_embedding:
        emb_tr, emb_va, emb_te = base.build_embedding_features(task_dir, candidate, n_tr, n_va, n_te)

    base_pred_roots = [RESULTS / "ept_family_official_v1_5seed_runs"]
    source_root = row.get("base_pred_source_root")
    if source_root:
        base_pred_roots.insert(0, Path(source_root))
    train_pred_files, valid_pred_files, test_pred_files, _ = v1.find_seed_predictions_optional(
        base_pred_roots,
        task,
        row["candidate"],
        [int(x) for x in args.base_seeds],
    )
    pred_tr = v1.average_optional(train_pred_files)
    pred_va = v1.average_optional(valid_pred_files)
    pred_te = v1.average_optional(test_pred_files)
    weight_sidecar = float(row.get("weight_sidecar") or 1.0)
    blend_mode = row.get("blend_mode") or "raw"
    has_base_blend = pred_tr is not None and pred_va is not None and pred_te is not None and weight_sidecar < 1.0

    tr, va, te = chem_tr[block], chem_va[block], chem_te[block]
    if needs_embedding:
        tr = np.concatenate([emb_tr, tr], axis=1)
        va = np.concatenate([emb_va, va], axis=1)
        te = np.concatenate([emb_te, te], axis=1)
    if selected_variant.endswith("_base_pred"):
        if pred_tr is None or pred_va is None or pred_te is None:
            raise RuntimeError(f"{task}: selected variant needs base predictions but none were found")
        tr = np.concatenate([tr, pred_tr.reshape(-1, 1)], axis=1)
        va = np.concatenate([va, pred_va.reshape(-1, 1)], axis=1)
        te = np.concatenate([te, pred_te.reshape(-1, 1)], axis=1)

    X_tr = base.sanitize_features(tr.astype(np.float32))
    X_va = base.sanitize_features(va.astype(np.float32))
    X_te = base.sanitize_features(te.astype(np.float32))
    y_tr = train_df[label_col].to_numpy()
    y_va = valid_df[label_col].to_numpy()
    y_te = test_df[label_col].to_numpy()
    y_tv = np.concatenate([y_tr, y_va], axis=0)
    X_tv = np.concatenate([X_tr, X_va], axis=0)
    smiles_tv = pd.concat([train_df[smiles_col], valid_df[smiles_col]], ignore_index=True).astype(str).tolist()
    base_tv = np.concatenate([pred_tr, pred_va], axis=0).astype(np.float32) if pred_tr is not None and pred_va is not None else None
    base_te = pred_te.astype(np.float32) if pred_te is not None else None
    task_type = base.infer_task_type(y_tv)
    return {
        "task_dir": task_dir,
        "X_tv": X_tv,
        "X_te": X_te,
        "y_tv": y_tv,
        "y_te": y_te,
        "smiles_tv": smiles_tv,
        "task_type": task_type,
        "base_tv": base_tv,
        "base_te": base_te,
        "has_base_blend": has_base_blend,
        "blend_mode": blend_mode,
        "weight_sidecar": weight_sidecar,
        "fit_backend": fit_backend,
    }


def read_prediction_score(pred_path: Path, metric: str) -> float:
    pred_df = pd.read_csv(pred_path)
    if "y_prob" in pred_df.columns:
        y_pred = pred_df["y_prob"].to_numpy()
    elif "y_pred" in pred_df.columns:
        y_pred = pred_df["y_pred"].to_numpy()
    else:
        raise KeyError(f"{pred_path}: expected y_prob or y_pred column")
    return float(base.score_metric(metric, pred_df["y_true"].to_numpy(), y_pred))


def read_seed_result(out_dir: Path, formal_seed: int, metric: str, force: bool) -> dict[str, object] | None:
    if force:
        return None
    pred_path = out_dir / f"test_predictions_formal_seed_{formal_seed}.csv"
    if not pred_path.exists():
        return None
    seed_path = out_dir / f"summary_formal_seed_{formal_seed}.json"
    valid_score = None
    if seed_path.exists():
        try:
            payload = json.loads(seed_path.read_text())
            valid_score = payload.get("valid_score")
            test_score = payload.get("test_score")
            if test_score is not None:
                return {"valid_score": valid_score, "test_score": float(test_score), "reused": True}
        except Exception:
            pass
    return {"valid_score": valid_score, "test_score": read_prediction_score(pred_path, metric), "reused": True}


def fmt_score_list(values: list[float | None]) -> str:
    out = []
    for value in values:
        if value is None:
            out.append("nan")
        else:
            out.append(f"{float(value):.12f}")
    return ";".join(out)


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    finite = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    if not finite:
        return None, None
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return mean, std


def run_fixed_row(row: dict[str, str], summary_path: Path, args: argparse.Namespace) -> dict[str, object]:
    task = row["task"]
    out_dir = Path(args.out_root) / summary_path.parent.name / task
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "summary.json"
    metric = row.get("tdc_metric") or row.get("metric")
    direction = row.get("metric_direction") or row.get("direction")
    backend = row["selected_backend"]
    topk = norm_topk(row["selected_topk"])
    top1_ref = float(row.get("tdc_top1_ref") or row.get("top1_ref"))
    existing = {seed: read_seed_result(out_dir, int(seed), metric, args.force) for seed in args.formal_seeds}
    missing_seeds = [int(seed) for seed, payload in existing.items() if payload is None]
    if result_path.exists() and not args.force and not missing_seeds:
        result = json.loads(result_path.read_text())
        existing_seeds = set(str(x) for x in result.get("formal_seeds", "").split(",") if str(x))
        requested_seeds = set(str(x) for x in args.formal_seeds)
        if requested_seeds.issubset(existing_seeds):
            return result

    xl = is_xl_row(row, summary_path)
    pack = build_variant_for_row(row, xl, args) if missing_seeds else None
    test_scores: list[float | None] = []
    valid_scores: list[float | None] = []

    fit_args = SimpleNamespace(
        xgb_estimators=args.xgb_estimators,
        tree_estimators=args.tree_estimators,
        cat_estimators=args.cat_estimators,
        n_jobs=args.n_jobs,
    )

    for formal_seed in args.formal_seeds:
        cached = existing.get(int(formal_seed))
        if cached is not None:
            valid_scores.append(cached.get("valid_score"))
            test_scores.append(float(cached["test_score"]))
            print(f"[reuse] {task} seed={formal_seed} test={float(cached['test_score']):.6f}", flush=True)
            continue
        assert pack is not None
        folds = nested.build_scaffold_folds(pack["smiles_tv"], int(args.folds), int(formal_seed))
        oof = np.zeros(len(pack["y_tv"]), dtype=np.float32)
        test_preds: list[np.ndarray] = []
        for fold_idx, valid_idx in enumerate(folds):
            train_mask = np.ones(len(pack["y_tv"]), dtype=bool)
            train_mask[valid_idx] = False
            train_idx = np.where(train_mask)[0]
            selected = v3.fit_selector(pack["X_tv"][train_idx], pack["y_tv"][train_idx], pack["task_type"], topk)
            model = pack["fit_backend"](
                backend,
                pack["X_tv"][train_idx][:, selected],
                pack["y_tv"][train_idx],
                pack["task_type"],
                metric,
                int(formal_seed) + fold_idx,
                fit_args,
            )
            oof[valid_idx] = v3.predict_backend(model, pack["X_tv"][valid_idx][:, selected], pack["task_type"])
            test_preds.append(v3.predict_backend(model, pack["X_te"][:, selected], pack["task_type"]))
        test_pred = np.stack(test_preds, axis=0).mean(axis=0).astype(np.float32)
        oof_eval = oof
        test_eval = test_pred
        if pack["has_base_blend"] and pack["base_tv"] is not None and pack["base_te"] is not None:
            oof_eval = v1.blend_prediction(oof, pack["base_tv"], float(pack["weight_sidecar"]), pack["blend_mode"])
            test_eval = v1.blend_prediction(test_pred, pack["base_te"], float(pack["weight_sidecar"]), pack["blend_mode"])
        valid_score = float(base.score_metric(metric, pack["y_tv"], oof_eval))
        test_score = float(base.score_metric(metric, pack["y_te"], test_eval))
        valid_scores.append(valid_score)
        test_scores.append(test_score)
        pred_path = out_dir / f"test_predictions_formal_seed_{formal_seed}.csv"
        base.write_predictions(pred_path, pack["y_te"], test_eval, pack["task_type"])
        seed_result = {
            "task": task,
            "formal_seed": int(formal_seed),
            "metric": metric,
            "valid_score": valid_score,
            "test_score": test_score,
            "prediction_file": str(pred_path),
        }
        (out_dir / f"summary_formal_seed_{formal_seed}.json").write_text(json.dumps(seed_result, indent=2))
        print(f"[seed] {task} seed={formal_seed} valid={valid_score:.6f} test={test_score:.6f}", flush=True)

    test_mean, test_std = mean_std(test_scores)
    valid_mean, valid_std = mean_std(valid_scores)
    if test_mean is None:
        raise RuntimeError(f"{task}: no formal seed test scores were available")
    is_top1 = test_mean >= top1_ref if direction == "max" else test_mean <= top1_ref
    result = {
        "task": task,
        "metric": metric,
        "direction": direction,
        "source_summary": str(summary_path),
        "candidate": row.get("candidate", ""),
        "selected_variant": row.get("selected_variant", ""),
        "selected_topk": row.get("selected_topk", ""),
        "selected_backend": backend,
        "blend_mode": row.get("blend_mode", ""),
        "weight_sidecar": row.get("weight_sidecar", ""),
        "formal_seeds": ",".join(str(x) for x in args.formal_seeds),
        "valid_scores": fmt_score_list(valid_scores),
        "valid_mean": valid_mean,
        "valid_std": valid_std,
        "test_scores": fmt_score_list(test_scores),
        "test_mean": test_mean,
        "test_std": test_std,
        "top1_ref": top1_ref,
        "tdc_rule_top1": bool(is_top1),
        "endpoint": "fixed_selected_chemical_scaffold_foldbag_5run",
    }
    result_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    global REPO, DATA_ROOT, RESULTS
    REPO = Path(args.repo)
    DATA_ROOT = REPO / "data" / "data_benchmark_official_v1"
    RESULTS = REPO / "results_strict"
    summary_path = as_path(REPO, args.summary)
    rows = load_rows(summary_path, args.tasks)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        print("[task]", row.get("task"), "from", summary_path, flush=True)
        try:
            result = run_fixed_row(row, summary_path, args)
            results.append(result)
            print(result["task"], result["test_mean"], result["test_std"], flush=True)
        except Exception as exc:
            results.append({"task": row.get("task", ""), "status": "error", "error": str(exc), "source_summary": str(summary_path)})
            print("[error]", row.get("task"), exc, flush=True)
    fields = sorted({k for r in results for k in r})
    out_file = out_root / f"{summary_path.parent.name}_summary.csv"
    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(out_file, flush=True)


if __name__ == "__main__":
    main()
