#!/usr/bin/env bash
set -euo pipefail
PY=<ENV_ROOT>/trimole_bench310/bin/python
ROOT=results_strict/formal_chemical_prior_v2_microsome_5seed
mkdir -p "$ROOT"
for seed in 101 202 303 404 505; do
  echo "[formal_seed $seed]"
  "$PY" paper_main_chemical_prior_v2.py \
    --out-root "$ROOT/seed_$seed" \
    --tasks clearance_microsome_az \
    --folds 3 \
    --seed "$seed" \
    --xgb-estimators 250 \
    --chemical-blocks core_pair_torsion \
    --force
 done
"$PY" - <<"PY"
import csv,json,statistics
from pathlib import Path
root=Path("results_strict/formal_chemical_prior_v2_microsome_5seed")
rows=[]
for d in sorted(root.glob("seed_*")):
    p=d/"clearance_microsome_az"/"result.json"
    if p.exists():
        r=json.loads(p.read_text())
        rows.append({"formal_seed":d.name.replace("seed_",""), **r})
if not rows:
    raise SystemExit("no seed results")
scores=[float(r["test_tdc_score"]) for r in rows]
summary={
    "task":"clearance_microsome_az",
    "metric":"Spearman",
    "top1_ref":0.630,
    "formal_seeds": ",".join(r["formal_seed"] for r in rows),
    "formal_run_scores": ",".join(f"{s:.12f}" for s in scores),
    "test_mean": statistics.mean(scores),
    "test_std": statistics.pstdev(scores),
    "beats_top1_mean": statistics.mean(scores) >= 0.630,
    "best_single_run": max(scores),
    "n_runs": len(rows),
    "selection_note":"For each formal run, variants/weights are selected only by train+valid scaffold-CV; test is evaluated once after selection.",
}
with (root/"formal_seed_results.csv").open("w", newline="") as f:
    fields=sorted({k for r in rows for k in r})
    w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
(root/"summary.json").write_text(json.dumps(summary, indent=2))
with (root/"summary.csv").open("w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(summary)); w.writeheader(); w.writerow(summary)
print(json.dumps(summary, indent=2))
PY
