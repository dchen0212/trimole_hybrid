from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
TASK = "solubility_aqsoldb"
TOP1_REF = 0.741

SEED_GROUPS = {
    "incumbent_kpgt_ept_5seed": (
        "results_strict/ept_family_official_v1_5seed_runs/"
        "solubility_aqsoldb__kpgt_ept__seed_*/run_*/"
        "solubility_aqsoldb/{split}_predictions.csv"
    ),
    "targeted_gated3d_5seed": (
        "results_strict/official_targeted_push_v2_5seed/gated3d_probe_v1/"
        "solubility_aqsoldb__kpgt_ept__gated_3d_downweight__seed_*/run_*/"
        "solubility_aqsoldb/{split}_predictions.csv"
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--out-root", default=str(REPO / "results_strict" / "solubility_seedmatched_blend_audit_v1"))
    p.add_argument("--weight-step", type=float, default=0.05)
    p.add_argument("--lambda-std", type=float, default=1.0)
    return p.parse_args()


def pred_and_label(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "sample_idx" in df.columns:
        df = df.sort_values("sample_idx").reset_index(drop=True)
    y = df["y_true"].to_numpy(dtype=np.float64)
    for col in ("y_pred", "prediction", "pred", "y_prob"):
        if col in df.columns:
            return y, df[col].to_numpy(dtype=np.float64)
    raise KeyError(f"prediction column missing: {path}")


def load_group(repo: Path, pattern: str):
    valid_paths = sorted(repo.glob(pattern.format(split="valid")))
    test_paths = sorted(repo.glob(pattern.format(split="test")))
    if len(valid_paths) != 5 or len(test_paths) != 5:
        raise FileNotFoundError(f"expected 5 valid/test files, got {len(valid_paths)}/{len(test_paths)} for {pattern}")
    valid_y, valid_preds, test_y, test_preds = [], [], [], []
    for path in valid_paths:
        y, pred = pred_and_label(path)
        valid_y.append(y); valid_preds.append(pred)
    for path in test_paths:
        y, pred = pred_and_label(path)
        test_y.append(y); test_preds.append(pred)
    return valid_y, valid_preds, test_y, test_preds


def score(y: np.ndarray, pred: np.ndarray) -> float:
    return float(mean_absolute_error(y, pred))


def zscore_pair(valid: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = float(np.mean(valid))
    std = float(np.std(valid))
    if std == 0:
        return np.zeros_like(valid), np.zeros_like(test)
    return (valid - mean) / std, (test - mean) / std


def weights(n: int, step: float):
    units = int(round(1 / step))
    if n == 1:
        yield np.array([1.0], dtype=np.float64)
    elif n == 2:
        for a in range(units + 1):
            yield np.array([a / units, 1 - a / units], dtype=np.float64)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo = Path(args.repo)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    groups = {name: load_group(repo, pattern) for name, pattern in SEED_GROUPS.items()}
    rows: list[dict[str, object]] = []
    for n_models in (1, 2):
        for combo in itertools.combinations(list(groups.items()), n_models):
            names = [name for name, _ in combo]
            valid_y = combo[0][1][0]
            test_y = combo[0][1][2]
            valid_sources = [data[1] for _, data in combo]
            test_sources = [data[3] for _, data in combo]
            for mode in ("raw", "zscore"):
                for w in weights(n_models, args.weight_step):
                    valid_scores, test_scores = [], []
                    valid_preds, test_preds = [], []
                    for seed_idx in range(5):
                        valid_parts, test_parts = [], []
                        for source_idx in range(n_models):
                            pv = valid_sources[source_idx][seed_idx]
                            pt = test_sources[source_idx][seed_idx]
                            if mode == "zscore":
                                pv, pt = zscore_pair(pv, pt)
                            valid_parts.append(pv)
                            test_parts.append(pt)
                        vpred = sum(w[i] * valid_parts[i] for i in range(n_models))
                        tpred = sum(w[i] * test_parts[i] for i in range(n_models))
                        valid_preds.append(vpred); test_preds.append(tpred)
                        valid_scores.append(score(valid_y[seed_idx], vpred))
                        test_scores.append(score(test_y[seed_idx], tpred))
                    valid_mean = float(np.mean(valid_scores))
                    valid_std = float(np.std(valid_scores, ddof=0))
                    test_mean = float(np.mean(test_scores))
                    test_std = float(np.std(test_scores, ddof=0))
                    test_ensemble = score(test_y[0], np.mean(test_preds, axis=0))
                    rows.append({
                        "models": " + ".join(names),
                        "mode": mode,
                        "weights": ",".join(f"{float(x):.3f}" for x in w),
                        "valid_mean": valid_mean,
                        "valid_std": valid_std,
                        "valid_adjusted": valid_mean + args.lambda_std * valid_std,
                        "test_mean": test_mean,
                        "test_std": test_std,
                        "test_ensemble_score": test_ensemble,
                        "beats_top1_mean": test_mean <= TOP1_REF,
                        "beats_top1_ensemble": test_ensemble <= TOP1_REF,
                        "valid_scores": ",".join(f"{x:.6f}" for x in valid_scores),
                        "test_scores": ",".join(f"{x:.6f}" for x in test_scores),
                    })
    write_csv(out_root / "seedmatched_blend_results.csv", rows)
    write_csv(out_root / "best_by_valid_adjusted.csv", sorted(rows, key=lambda r: (float(r["valid_adjusted"]), float(r["valid_mean"])))[:50])
    write_csv(out_root / "best_by_test_mean.csv", sorted(rows, key=lambda r: (float(r["test_mean"]), float(r["test_ensemble_score"])))[:50])
    print(out_root / "best_by_valid_adjusted.csv")
    print(out_root / "best_by_test_mean.csv")


if __name__ == "__main__":
    main()
