from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


REPO = Path("/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

import cv_selected_prediction_ensemble_builder_fast_v2 as base
import run_xl_metric_calibrated_blend_all22_v2 as xl


OUT_ROOT = REPO / "results_strict" / "neartop_valid_bootstrap_selector_v1"
DEFAULT_TASKS = [
    "clearance_hepatocyte_az",
    "cyp2c9_substrate_carbonmangels",
    "bbb_martins",
    "hia_hou",
]


@dataclass
class Candidate:
    task: str
    models: str
    mode: str
    weights: str
    valid_y: np.ndarray
    valid_pred: np.ndarray
    test_y: np.ndarray
    test_pred: np.ndarray
    source: str
    diversity: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    p.add_argument("--max-streams", type=int, default=14)
    p.add_argument("--weight-step", type=float, default=0.1)
    p.add_argument("--n-subsets", type=int, default=200)
    p.add_argument("--subset-fracs", nargs="*", type=float, default=[0.65, 0.75, 0.85])
    p.add_argument("--lambdas", nargs="*", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0])
    p.add_argument("--diversity-bonuses", nargs="*", type=float, default=[0.0])
    p.add_argument("--seed", type=int, default=20260429)
    return p.parse_args()


def smiles_col(df: pd.DataFrame) -> str:
    for col in ("smiles", "Drug", "SMILES", "drug"):
        if col in df.columns:
            return col
    raise KeyError("SMILES column not found")


def label_col(df: pd.DataFrame) -> str:
    skip = {"smiles", "drug", "drug_id", "mol", "id", "sample_idx"}
    for col in df.columns:
        if col.lower() not in skip:
            return col
    raise KeyError("label column not found")


def scaffold_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return f"invalid::{smiles}"
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaf or f"empty::{smiles}"


def valid_scaffold_groups(task: str) -> tuple[list[np.ndarray], np.ndarray]:
    df = pd.read_csv(base.DATA / task / "valid.csv")
    y = df[label_col(df)].to_numpy(dtype=np.float64)
    groups: dict[str, list[int]] = {}
    for idx, smi in enumerate(df[smiles_col(df)].astype(str).tolist()):
        groups.setdefault(scaffold_key(smi), []).append(idx)
    return [np.asarray(v, dtype=int) for v in groups.values()], y


def make_subsets(task: str, n_subsets: int, frac: float, seed: int) -> list[np.ndarray]:
    groups, y = valid_scaffold_groups(task)
    rng = np.random.default_rng(seed)
    subsets: list[np.ndarray] = []
    n_groups = len(groups)
    k = max(1, int(round(frac * n_groups)))
    for _ in range(n_subsets * 4):
        chosen = rng.choice(n_groups, size=k, replace=False)
        idx = np.concatenate([groups[i] for i in chosen])
        if len(idx) < 5:
            continue
        yy = y[idx]
        metric = base.TASKS[task]["metric"]
        if metric in {"AUROC", "AUPRC"} and (np.sum(yy == 1) < 2 or np.sum(yy == 0) < 2):
            continue
        subsets.append(np.sort(idx))
        if len(subsets) >= n_subsets:
            break
    if not subsets:
        subsets = [np.arange(len(y), dtype=int)]
    return subsets


def direction(task: str) -> str:
    return base.TASKS[task]["direction"]


def better(task: str, a: float, b: float) -> bool:
    return a > b if direction(task) == "max" else a < b


def sort_rows(task: str, rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    reverse = direction(task) == "max"
    bad = -1e99 if reverse else 1e99
    return sorted(rows, key=lambda r: float(r[key]) if r[key] == r[key] else bad, reverse=reverse)


def candidate_modes(task: str) -> list[str]:
    metric = base.TASKS[task]["metric"]
    if metric == "MAE":
        return ["raw"]
    if metric == "Spearman":
        return ["rank", "zscore", "raw"]
    return ["logit", "rank", "zscore", "raw"]


def weight_vectors(n: int, step: float):
    units = int(round(1.0 / step))
    if n == 1:
        yield np.array([1.0], dtype=float)
        return
    if n == 2:
        for a in range(units + 1):
            yield np.array([a / units, 1.0 - a / units], dtype=float)
        return
    if n == 3:
        for a in range(units + 1):
            for b in range(units + 1 - a):
                yield np.array([a / units, b / units, (units - a - b) / units], dtype=float)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rank_diversity(parts: list[np.ndarray]) -> float:
    if len(parts) <= 1:
        return 0.0
    ranked = np.vstack([base.transform(x, "rank") for x in parts])
    return float(np.nanmean(np.nanstd(ranked, axis=0)))


def build_candidate_objects(task: str, streams: list[base.Stream], max_streams: int, step: float) -> list[Candidate]:
    singles = base.single_rows(task, streams)
    selected_names = {r["models"] for r in sort_rows(task, singles, "valid_score")[:max_streams]}

    # Keep top test-beating diagnostic streams out of the selector rule, but include
    # all top valid streams so this script can discover a valid-only signal quickly.
    pool = [s for s in streams if s.name in selected_names]
    candidates: list[Candidate] = []
    for s in pool:
        candidates.append(
            Candidate(
                task=task,
                models=s.name,
                mode="single",
                weights="1.000",
                valid_y=s.valid_y,
                valid_pred=s.valid_pred,
                test_y=s.test_y,
                test_pred=s.test_pred,
                source=s.source,
                diversity=0.0,
            )
        )

    for n in (2, 3):
        if len(pool) < n:
            continue
        for combo in itertools.combinations(pool, n):
            names = " + ".join(s.name for s in combo)
            source = " | ".join(s.source for s in combo)
            for mode in candidate_modes(task):
                valid_parts = [base.transform(s.valid_pred, mode) for s in combo]
                test_parts = [base.transform(s.test_pred, mode) for s in combo]
                diversity = rank_diversity(valid_parts)
                for w in weight_vectors(n, step):
                    vp = sum(w[i] * valid_parts[i] for i in range(n))
                    tp = sum(w[i] * test_parts[i] for i in range(n))
                    candidates.append(
                        Candidate(
                            task=task,
                            models=names,
                            mode=mode,
                            weights=",".join(f"{float(x):.3f}" for x in w),
                            valid_y=combo[0].valid_y,
                            valid_pred=vp,
                            test_y=combo[0].test_y,
                            test_pred=tp,
                            source=source,
                            diversity=diversity,
                        )
                    )
    return candidates


def subset_stats(task: str, cand: Candidate, subsets: list[np.ndarray]) -> tuple[float, float, float]:
    vals = []
    for idx in subsets:
        vals.append(base.score(task, cand.valid_y[idx], cand.valid_pred[idx]))
    arr = np.asarray([x for x in vals if not math.isnan(float(x))], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=0)), float(np.min(arr) if direction(task) == "max" else np.max(arr))


def prepare_prediction_zoo() -> None:
    for directory in [
        base.RESULTS / "paper_main_chemical_prior_xl_v4_all22_32core",
        base.RESULTS / "paper_main_chemical_prior_xl_v4_remaining4_32core",
    ]:
        xl.write_xl_summary(directory)
    base.TASKS.update(xl.EXTRA_TASKS)
    for summary in xl.EXTRA_SUMMARIES:
        if summary not in base.PRED_SUMMARIES:
            base.PRED_SUMMARIES.append(summary)


def run_task(task: str, out_root: Path, args: argparse.Namespace) -> dict[str, object]:
    streams = base.build_streams(task)
    candidates = build_candidate_objects(task, streams, args.max_streams, args.weight_step)
    print(task, "streams", len(streams), "candidates", len(candidates), flush=True)
    task_dir = out_root / task
    subsets_by_frac = {
        frac: make_subsets(task, args.n_subsets, frac, args.seed + int(1000 * frac))
        for frac in args.subset_fracs
    }
    rows: list[dict[str, object]] = []
    for cand in candidates:
        full_valid = base.score(task, cand.valid_y, cand.valid_pred)
        test = base.score(task, cand.test_y, cand.test_pred)
        base_row = {
            "task": task,
            "metric": base.TASKS[task]["metric"],
            "top1_ref": base.TASKS[task]["top1_ref"],
            "models": cand.models,
            "mode": cand.mode,
            "weights": cand.weights,
            "full_valid_score": full_valid,
            "test_score": test,
            "beats_top1": base.beats(task, test),
            "source": cand.source,
            "diversity": cand.diversity,
        }
        for frac in args.subset_fracs:
            mean, std, worst = subset_stats(task, cand, subsets_by_frac[frac])
            for lam in args.lambdas:
                for bonus in args.diversity_bonuses:
                    if direction(task) == "max":
                        selector = mean - lam * std + bonus * cand.diversity
                    else:
                        selector = mean + lam * std - bonus * cand.diversity
                    row = dict(base_row)
                    row.update(
                        {
                            "subset_frac": frac,
                            "subset_mean": mean,
                            "subset_std": std,
                            "subset_worst": worst,
                            "lambda_std": lam,
                            "diversity_bonus": bonus,
                            "selector_score": selector,
                        }
                    )
                    rows.append(row)

    by_selector = sort_rows(task, rows, "selector_score")
    by_full_valid = sort_rows(task, rows, "full_valid_score")
    by_test = sort_rows(task, rows, "test_score")
    write_csv(task_dir / "best_by_bootstrap_selector.csv", by_selector[:200])
    write_csv(task_dir / "best_by_full_valid.csv", by_full_valid[:200])
    write_csv(task_dir / "best_by_test_diagnostic.csv", by_test[:200])

    selected = by_selector[0]
    diagnostic = by_test[0]
    return {
        "task": task,
        "metric": base.TASKS[task]["metric"],
        "top1_ref": base.TASKS[task]["top1_ref"],
        "n_streams": len(streams),
        "n_candidates_expanded": len(rows),
        "selected_models": selected["models"],
        "selected_mode": selected["mode"],
        "selected_weights": selected["weights"],
        "selected_subset_frac": selected["subset_frac"],
        "selected_lambda_std": selected["lambda_std"],
        "selected_diversity_bonus": selected["diversity_bonus"],
        "selected_diversity": selected["diversity"],
        "selected_full_valid_score": selected["full_valid_score"],
        "selected_subset_mean": selected["subset_mean"],
        "selected_subset_std": selected["subset_std"],
        "selected_test_score": selected["test_score"],
        "selected_beats_top1": selected["beats_top1"],
        "best_test_models": diagnostic["models"],
        "best_test_mode": diagnostic["mode"],
        "best_test_weights": diagnostic["weights"],
        "best_test_full_valid_score": diagnostic["full_valid_score"],
        "best_test_selector_score": diagnostic["selector_score"],
        "best_test_score": diagnostic["test_score"],
        "best_test_beats_top1": diagnostic["beats_top1"],
    }


def main() -> None:
    args = parse_args()
    global REPO
    REPO = Path(args.repo)
    base.REPO = REPO
    base.RESULTS = REPO / "results_strict"
    base.DATA = REPO / "data" / "data_benchmark_official_v1"
    prepare_prediction_zoo()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for task in args.tasks:
        row = run_task(task, out_root, args)
        summary.append(row)
        print(
            task,
            "selected_test",
            row["selected_test_score"],
            "best_test",
            row["best_test_score"],
            flush=True,
        )
    write_csv(out_root / "summary.csv", summary)
    print(out_root / "summary.csv", flush=True)


if __name__ == "__main__":
    main()
