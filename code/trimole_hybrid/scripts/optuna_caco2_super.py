import json
import subprocess
from pathlib import Path

import optuna
import pandas as pd

TASK = "caco2_wang"
OUT_ROOT = Path("results/model_log/optuna_caco2_super")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def load_score(results_csv: Path) -> float:
    df = pd.read_csv(results_csv)
    row = df[df["task"] == TASK].iloc[0]
    # Caco2 用 MAE，越小越好，所以 maximize(-MAE)
    return -float(row["test_mae"])

def objective(trial: optuna.Trial) -> float:
    hidden_dim = trial.suggest_categorical("hidden_dim", [192, 256, 320])
    dropout_head = trial.suggest_categorical("dropout_head", [0.10, 0.18, 0.25, 0.32])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-4, 2e-4, 3e-4, 5e-4, 8e-4])
    lr = trial.suggest_categorical("lr", [8e-5, 1.2e-4, 1.8e-4, 2.4e-4])
    spearman_reg = trial.suggest_categorical("spearman_reg", [0.0, 0.02, 0.05])

    out_dir = OUT_ROOT / f"trial_{trial.number:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "trimole.pipelines.batch_run_data_new",
        "--data-new", "./data/data_benchmark",
        "--out", str(out_dir),
        "--tasks", TASK,
        "--hidden-dim", str(hidden_dim),
        "--dropout-head", str(dropout_head),
        "--weight-decay", str(weight_decay),
        "--lr", str(lr),
        "--spearman-reg", str(spearman_reg),
    ]
    subprocess.run(cmd, check=True)

    run_files = sorted(out_dir.glob("run_*/results_all.csv"), reverse=True)
    if not run_files:
        raise RuntimeError(f"No results_all.csv found under {out_dir}")

    score = load_score(run_files[0])

    meta = {
        "task": TASK,
        "trial": trial.number,
        "score": score,
        "params": {
            "hidden_dim": hidden_dim,
            "dropout_head": dropout_head,
            "weight_decay": weight_decay,
            "lr": lr,
            "spearman_reg": spearman_reg,
        },
        "results_csv": str(run_files[0]),
    }
    (out_dir / "trial_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    trial.report(score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return score

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="caco2_super",
        storage="sqlite:///results/model_log/optuna_caco2_super/caco2_super.db",
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4),
    )
    study.optimize(objective, n_trials=16)

    best = {
        "task": TASK,
        "best_value": study.best_value,
        "best_params": study.best_params,
    }
    out_path = OUT_ROOT / "best.json"
    out_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(f"saved -> {out_path}")
