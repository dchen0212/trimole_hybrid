from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_cyp3a4_substrate_seedmatched_prediction_zoo_probe_v1 as v1


REPO = v1.REPO
TASK = v1.TASK
TOP1_REF = v1.TOP1_REF
OUT = REPO / "results_strict" / "cyp3a4_substrate_seedbag_endpoint_selector_v2"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score(y: np.ndarray, pred: np.ndarray) -> float:
    return float(roc_auc_score(y, pred)) if len(np.unique(y)) >= 2 else float("nan")


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    streams = {}
    for name, pattern in v1.SEED_GROUPS.items():
        try:
            streams[name] = v1.load_group(pattern)
        except FileNotFoundError as exc:
            print(f"skip_group {name}: {exc}", flush=True)
    for name, (valid_rel, test_rel, trainval) in v1.SINGLETONS.items():
        if (REPO / valid_rel).exists() and (REPO / test_rel).exists():
            streams[name] = v1.load_singleton(valid_rel, test_rel, trainval)

    rows: list[dict[str, object]] = []
    items = list(streams.items())
    for n_models in range(1, min(3, len(items)) + 1):
        for combo in itertools.combinations(items, n_models):
            names = [name for name, _ in combo]
            seed_group_count = sum(1 for _, data in combo if int(data["n"]) >= 5)
            singleton_count = len(combo) - seed_group_count
            for mode in ("prob", "logit", "zscore", "rank"):
                valid_parts = [[v1.transform(pred, mode) for pred in data["vp"]] for _, data in combo]
                test_parts = [[v1.transform(pred, mode) for pred in data["tp"]] for _, data in combo]
                vy = combo[0][1]["vy"]
                ty = combo[0][1]["ty"]
                for w in v1.weight_vectors(n_models):
                    valid_scores, test_scores, valid_preds, test_preds = [], [], [], []
                    for seed_idx in range(5):
                        vpred = sum(w[i] * valid_parts[i][seed_idx] for i in range(n_models))
                        tpred = sum(w[i] * test_parts[i][seed_idx] for i in range(n_models))
                        valid_scores.append(score(vy[seed_idx], vpred))
                        test_scores.append(score(ty[seed_idx], tpred))
                        valid_preds.append(vpred)
                        test_preds.append(tpred)
                    valid_ensemble = score(vy[0], np.mean(valid_preds, axis=0))
                    test_ensemble = score(ty[0], np.mean(test_preds, axis=0))
                    rows.append(
                        {
                            "models": " + ".join(names),
                            "mode": mode,
                            "weights": ",".join(f"{x:.2f}" for x in w),
                            "seed_group_count": seed_group_count,
                            "singleton_count": singleton_count,
                            "valid_mean": float(np.nanmean(valid_scores)),
                            "valid_std": float(np.nanstd(valid_scores)),
                            "valid_adjusted": float(np.nanmean(valid_scores) - np.nanstd(valid_scores)),
                            "valid_ensemble": valid_ensemble,
                            "test_mean": float(np.nanmean(test_scores)),
                            "test_std": float(np.nanstd(test_scores)),
                            "test_ensemble": test_ensemble,
                            "beats_mean": bool(np.nanmean(test_scores) >= TOP1_REF),
                            "beats_ensemble": bool(test_ensemble >= TOP1_REF),
                            "test_scores": ";".join(f"{x:.6f}" for x in test_scores),
                            "valid_scores": ";".join(f"{x:.6f}" for x in valid_scores),
                        }
                    )

    endpoint_rows = [r for r in rows if int(r["seed_group_count"]) >= 1]
    no_xl_rows = [r for r in endpoint_rows if "xl_v4" not in str(r["models"]) and "rank_tabular" not in str(r["models"])]

    outputs = [
        ("all_results.csv", rows),
        ("endpoint_results.csv", endpoint_rows),
        ("endpoint_no_xl_results.csv", no_xl_rows),
    ]
    for filename, data in outputs:
        write_csv(OUT / filename, data)
        write_csv(OUT / filename.replace(".csv", "_best_by_valid_ensemble.csv"), sorted(data, key=lambda r: r["valid_ensemble"], reverse=True)[:100])
        write_csv(OUT / filename.replace(".csv", "_best_by_valid_adjusted.csv"), sorted(data, key=lambda r: r["valid_adjusted"], reverse=True)[:100])
        write_csv(OUT / filename.replace(".csv", "_best_by_test_ensemble_diagnostic.csv"), sorted(data, key=lambda r: r["test_ensemble"], reverse=True)[:100])

    selected = sorted(endpoint_rows, key=lambda r: r["valid_ensemble"], reverse=True)[0]
    selected_no_xl = sorted(no_xl_rows, key=lambda r: r["valid_ensemble"], reverse=True)[0]
    best_test = sorted(endpoint_rows, key=lambda r: r["test_ensemble"], reverse=True)[0]
    summary = [
        {"selector": "endpoint_valid_ensemble", **selected},
        {"selector": "endpoint_no_xl_valid_ensemble", **selected_no_xl},
        {"selector": "diagnostic_best_test_ensemble", **best_test},
    ]
    write_csv(OUT / "summary.csv", summary)
    for row in summary:
        print(row, flush=True)


if __name__ == "__main__":
    run()
