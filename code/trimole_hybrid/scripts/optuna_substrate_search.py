import argparse
import json
import subprocess
from pathlib import Path

import optuna
import pandas as pd


TASKS = [
    "cyp2c9_substrate_carbonmangels",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
]


def load_score(results_csv: Path) -> float:
    df = pd.read_csv(results_csv)
    df = df[df["task"].isin(TASKS)].copy()
    # Optimize mean AUPRC across substrate tasks
    return float(df["test_auprc"].mean())


def objective(trial: optuna.Trial) -> float:
    gamma = trial.suggest_float("focal_gamma", 1.0, 2.5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 192, 256])
    dropout_head = trial.suggest_float("dropout_head", 0.15, 0.40)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 3e-3, log=True)
    sampler_on = trial.suggest_categorical("use_weighted_sampler", [False, True])
    sampler_pos_weight = trial.suggest_float("sampler_pos_weight", 1.0, 3.0)

    out_dir = Path("results/model_log/optuna_substrate") / f"trial_{trial.number:03d}"
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "trimole.pipelines.batch_run_data_new",
        "--data-new", "./data/data_benchmark",
        "--out", str(out_dir),
        "--tasks", *TASKS,
        "--loss-type", "focal",
        "--focal-gamma", str(gamma),
        "--hidden-dim", str(hidden_dim),
        "--dropout-head", str(dropout_head),
        "--weight-decay", str(weight_decay),
    ]

    if sampler_on:
        cmd += ["--use-weighted-sampler", "--sampler-pos-weight", str(sampler_pos_weight)]

    subprocess.run(cmd, check=True)

    run_files = sorted(out_dir.glob("run_*/results_all.csv"), reverse=True)
    if not run_files:
        raise RuntimeError(f"No results_all.csv found under {out_dir}")

    score = load_score(run_files[0])
    trial.report(score, step=0)

    if trial.should_prune():
        raise optuna.TrialPruned()

    meta = {
        "trial": trial.number,
        "score_mean_auprc": score,
        "params": {
            "focal_gamma": gamma,
            "hidden_dim": hidden_dim,
            "dropout_head": dropout_head,
            "weight_decay": weight_decay,
            "use_weighted_sampler": sampler_on,
            "sampler_pos_weight": sampler_pos_weight,
        },
        "results_csv": str(run_files[0]),
    }
    (out_dir / "trial_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--study-name", type=str, default="substrate_search")
    args = parser.parse_args()

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
    )
    study.optimize(objective, n_trials=args.n_trials)

    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
    }
    out_path = Path("results/model_log/optuna_substrate_best.json")
    out_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
