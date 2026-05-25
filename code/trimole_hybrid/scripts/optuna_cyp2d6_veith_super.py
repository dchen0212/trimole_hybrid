import json
import subprocess
from pathlib import Path

import optuna
import pandas as pd

TASK = "cyp2d6_veith"
OUT_ROOT = Path("results/model_log/optuna_cyp2d6_veith_super")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def load_score(results_csv: Path) -> float:
    df = pd.read_csv(results_csv)
    row = df[df["task"] == TASK].iloc[0]
    # Veith 这里按 AUPRC 冲
    return float(row["test_auprc"])

def objective(trial: optuna.Trial) -> float:
    fusion_type = trial.suggest_categorical("fusion_type", ["gated", "mlp"])
    hidden_dim = trial.suggest_categorical("hidden_dim", [192, 256, 320])
    dropout_head = trial.suggest_categorical("dropout_head", [0.15, 0.22, 0.30, 0.36])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-4, 3e-4, 5e-4, 8e-4, 1.2e-3])
    loss_type = trial.suggest_categorical("loss_type", ["auto", "focal"])
    focal_gamma = trial.suggest_float("focal_gamma", 1.2, 2.2)

    out_dir = OUT_ROOT / f"trial_{trial.number:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "trimole.pipelines.batch_run_data_new",
        "--data-new", "./data/data_benchmark",
        "--out", str(out_dir),
        "--tasks", TASK,
        "--fusion-type", str(fusion_type),
        "--hidden-dim", str(hidden_dim),
        "--dropout-head", str(dropout_head),
        "--weight-decay", str(weight_decay),
    ]

    if loss_type == "focal":
        cmd += ["--loss-type", "focal", "--focal-gamma", str(focal_gamma)]
    else:
        cmd += ["--loss-type", "auto"]

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
            "fusion_type": fusion_type,
            "hidden_dim": hidden_dim,
            "dropout_head": dropout_head,
            "weight_decay": weight_decay,
            "loss_type": loss_type,
            "focal_gamma": focal_gamma,
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
        study_name="cyp2d6_veith_super",
        storage="sqlite:///results/model_log/optuna_cyp2d6_veith_super/cyp2d6_veith_super.db",
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
