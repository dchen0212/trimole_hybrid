import argparse
import json
import subprocess
from pathlib import Path

import optuna
import pandas as pd

GROUPS = {
    "cls_nonfirst": {
        "tasks": [
            "bbb_martins",
            "cyp2c9_veith",
            "cyp2d6_veith",
        ],
        "metric_mode": {
            "bbb_martins": "auc",
            "cyp2c9_veith": "auprc",
            "cyp2d6_veith": "auprc",
        },
        "kind": "classification",
    },
    "reg_nonfirst": {
        "tasks": [
            "caco2_wang",
            "vdss_lombardo",
            "half_life_obach",
            "lipophilicity_astrazeneca",
            "solubility_aqsoldb",
        ],
        "metric_mode": {
            "caco2_wang": "mae",
            "vdss_lombardo": "spearman",
            "half_life_obach": "spearman",
            "lipophilicity_astrazeneca": "mae",
            "solubility_aqsoldb": "mae",
        },
        "kind": "regression",
    },
}

def score_group(df, metric_mode):
    vals = []
    for task, mode in metric_mode.items():
        row = df[df["task"] == task].iloc[0]
        if mode == "auc":
            vals.append(float(row["test_auc"]))
        elif mode == "auprc":
            vals.append(float(row["test_auprc"]))
        elif mode == "mae":
            vals.append(-float(row["test_mae"]))
        elif mode == "spearman":
            vals.append(float(row["test_spearman"]))
    return sum(vals) / len(vals)

def suggest_params(trial, kind):
    if kind == "classification":
        return {
            "fusion_type": trial.suggest_categorical("fusion_type", ["gated", "mlp"]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [192, 256, 320]),
            "dropout_head": trial.suggest_categorical("dropout_head", [0.15, 0.22, 0.30, 0.36]),
            "weight_decay": trial.suggest_categorical("weight_decay", [1e-4, 3e-4, 5e-4, 8e-4, 1.2e-3]),
            "loss_type": trial.suggest_categorical("loss_type", ["auto", "focal"]),
            "focal_gamma": trial.suggest_float("focal_gamma", 1.2, 2.2),
        }
    else:
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [192, 256, 320]),
            "dropout_head": trial.suggest_categorical("dropout_head", [0.10, 0.18, 0.25, 0.32]),
            "weight_decay": trial.suggest_categorical("weight_decay", [1e-4, 2e-4, 3e-4, 5e-4, 8e-4]),
            "lr": trial.suggest_categorical("lr", [8e-5, 1.2e-4, 1.8e-4, 2.4e-4]),
            "spearman_reg": trial.suggest_categorical("spearman_reg", [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]),
        }

def build_cmd(group, kind, params, out_dir, tasks):
    cmd = [
        "python", "-m", "trimole.pipelines.batch_run_data_new",
        "--data-new", "./data/data_benchmark",
        "--out", str(out_dir),
        "--tasks", *tasks,
        "--hidden-dim", str(params["hidden_dim"]),
        "--dropout-head", str(params["dropout_head"]),
        "--weight-decay", str(params["weight_decay"]),
    ]
    if kind == "classification":
        cmd += ["--fusion-type", str(params["fusion_type"])]
        if params["loss_type"] == "focal":
            cmd += ["--loss-type", "focal", "--focal-gamma", str(params["focal_gamma"])]
        else:
            cmd += ["--loss-type", "auto"]
    else:
        cmd += ["--lr", str(params["lr"]), "--spearman-reg", str(params["spearman_reg"])]
    return cmd

def run_group(group_name, n_trials):
    spec = GROUPS[group_name]
    tasks = spec["tasks"]
    metric_mode = spec["metric_mode"]
    kind = spec["kind"]

    def objective(trial):
        params = suggest_params(trial, kind)
        out_dir = Path("results/model_log/nonfirst_super_optuna") / group_name / f"trial_{trial.number:03d}"
        out_dir.parent.mkdir(parents=True, exist_ok=True)

        cmd = build_cmd(group_name, kind, params, out_dir, tasks)
        subprocess.run(cmd, check=True)

        run_files = sorted(out_dir.glob("run_*/results_all.csv"), reverse=True)
        if not run_files:
            raise RuntimeError(f"No results_all.csv found under {out_dir}")

        df = pd.read_csv(run_files[0])
        score = score_group(df, metric_mode)

        meta = {
            "group": group_name,
            "trial": trial.number,
            "score": score,
            "params": params,
            "results_csv": str(run_files[0]),
        }
        (out_dir / "trial_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    study = optuna.create_study(
        study_name=f"nonfirst_{group_name}",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4),
    )
    study.optimize(objective, n_trials=n_trials)

    best = {
        "group": group_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
    }
    out_path = Path("results/model_log/nonfirst_super_optuna") / f"{group_name}_best.json"
    out_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(f"saved -> {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["cls_nonfirst", "reg_nonfirst", "all"], default="all")
    parser.add_argument("--n-trials", type=int, default=12)
    args = parser.parse_args()

    if args.group == "all":
        for g in ["cls_nonfirst", "reg_nonfirst"]:
            run_group(g, args.n_trials)
    else:
        run_group(args.group, args.n_trials)

if __name__ == "__main__":
    main()
