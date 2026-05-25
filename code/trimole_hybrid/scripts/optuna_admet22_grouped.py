import argparse
import json
import subprocess
from pathlib import Path

import optuna
import pandas as pd

GROUPS = {
    "substrate": {
        "tasks": [
            "cyp2c9_substrate_carbonmangels",
            "cyp2d6_substrate_carbonmangels",
            "cyp3a4_substrate_carbonmangels",
        ],
        "metric_mode": {
            "cyp2c9_substrate_carbonmangels": "auprc",
            "cyp2d6_substrate_carbonmangels": "auprc",
            "cyp3a4_substrate_carbonmangels": "auc",
        },
        "search_space": "substrate",
    },
    "veith": {
        "tasks": [
            "cyp2c9_veith",
            "cyp2d6_veith",
            "cyp3a4_veith",
        ],
        "metric_mode": {
            "cyp2c9_veith": "auprc",
            "cyp2d6_veith": "auprc",
            "cyp3a4_veith": "auprc",
        },
        "search_space": "classification",
    },
    "classification": {
        "tasks": [
            "ames",
            "bbb_martins",
            "bioavailability_ma",
            "dili",
            "herg",
            "hia_hou",
            "pgp_broccatelli",
        ],
        "metric_mode": {
            "ames": "auc",
            "bbb_martins": "auc",
            "bioavailability_ma": "auc",
            "dili": "auc",
            "herg": "auc",
            "hia_hou": "auc",
            "pgp_broccatelli": "auc",
        },
        "search_space": "classification",
    },
    "regression": {
        "tasks": [
            "caco2_wang",
            "clearance_hepatocyte_az",
            "clearance_microsome_az",
            "half_life_obach",
            "ld50_zhu",
            "lipophilicity_astrazeneca",
            "ppbr_az",
            "solubility_aqsoldb",
            "vdss_lombardo",
        ],
        "metric_mode": {
            "caco2_wang": "mae",
            "clearance_hepatocyte_az": "spearman",
            "clearance_microsome_az": "spearman",
            "half_life_obach": "spearman",
            "ld50_zhu": "mae",
            "lipophilicity_astrazeneca": "mae",
            "ppbr_az": "mae",
            "solubility_aqsoldb": "mae",
            "vdss_lombardo": "spearman",
        },
        "search_space": "regression",
    },
}

def score_group(df: pd.DataFrame, metric_mode: dict) -> float:
    vals = []
    for task, mode in metric_mode.items():
        row = df[df["task"] == task]
        if row.empty:
            raise ValueError(f"Missing task in results: {task}")
        row = row.iloc[0]
        if mode == "auc":
            vals.append(float(row["test_auc"]))
        elif mode == "auprc":
            vals.append(float(row["test_auprc"]))
        elif mode == "mae":
            # maximize negative MAE
            vals.append(-float(row["test_mae"]))
        elif mode == "spearman":
            vals.append(float(row["test_spearman"]))
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return sum(vals) / len(vals)

def build_cmd(group_name: str, params: dict, out_dir: Path, tasks: list[str]) -> list[str]:
    cmd = [
        "python", "-m", "trimole.pipelines.batch_run_data_new",
        "--data-new", "./data/data_benchmark",
        "--out", str(out_dir),
        "--tasks", *tasks,
        "--hidden-dim", str(params["hidden_dim"]),
        "--dropout-head", str(params["dropout_head"]),
        "--weight-decay", str(params["weight_decay"]),
    ]

    if group_name == "substrate":
        cmd += [
            "--loss-type", "focal",
            "--focal-gamma", str(params["focal_gamma"]),
        ]
        if params["use_weighted_sampler"]:
            cmd += [
                "--use-weighted-sampler",
                "--sampler-pos-weight", str(params["sampler_pos_weight"]),
            ]

    elif group_name in {"veith", "classification"}:
        cmd += [
            "--fusion-type", params["fusion_type"],
        ]
        if params["loss_type"] == "focal":
            cmd += [
                "--loss-type", "focal",
                "--focal-gamma", str(params["focal_gamma"]),
            ]
        else:
            cmd += ["--loss-type", "auto"]

    elif group_name == "regression":
        cmd += [
            "--lr", str(params["lr"]),
            "--spearman-reg", str(params["spearman_reg"]),
        ]

    return cmd

def suggest_params(trial: optuna.Trial, group_name: str) -> dict:
    common = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 192, 256]),
        "dropout_head": trial.suggest_float("dropout_head", 0.20, 0.40),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 3e-3, log=True),
    }

    if group_name == "substrate":
        common.update({
            "focal_gamma": trial.suggest_float("focal_gamma", 1.2, 1.8),
            "use_weighted_sampler": trial.suggest_categorical("use_weighted_sampler", [False, True]),
            "sampler_pos_weight": trial.suggest_float("sampler_pos_weight", 1.0, 3.0),
        })
    elif group_name in {"veith", "classification"}:
        common.update({
            "fusion_type": trial.suggest_categorical("fusion_type", ["gated", "mlp"]),
            "loss_type": trial.suggest_categorical("loss_type", ["auto", "focal"]),
            "focal_gamma": trial.suggest_float("focal_gamma", 1.5, 2.5),
        })
    elif group_name == "regression":
        common.update({
            "lr": trial.suggest_float("lr", 1e-4, 3e-4, log=True),
            "spearman_reg": trial.suggest_float("spearman_reg", 0.0, 0.2),
        })
    return common

def run_study(group_name: str, n_trials: int) -> None:
    spec = GROUPS[group_name]
    tasks = spec["tasks"]
    metric_mode = spec["metric_mode"]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, group_name)
        out_dir = Path("results/model_log/admet22_grouped_optuna") / group_name / f"trial_{trial.number:03d}"
        out_dir.parent.mkdir(parents=True, exist_ok=True)

        cmd = build_cmd(group_name, params, out_dir, tasks)
        subprocess.run(cmd, check=True)

        run_files = sorted(out_dir.glob("run_*/results_all.csv"), reverse=True)
        if not run_files:
            raise RuntimeError(f"No results_all.csv under {out_dir}")

        df = pd.read_csv(run_files[0])
        score = score_group(df, metric_mode)

        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        meta = {
            "group": group_name,
            "trial": trial.number,
            "score": score,
            "params": params,
            "results_csv": str(run_files[0]),
        }
        (out_dir / "trial_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return score

    study = optuna.create_study(
        study_name=f"admet22_{group_name}",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
    )
    study.optimize(objective, n_trials=n_trials)

    best = {
        "group": group_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
    }
    out_path = Path("results/model_log/admet22_grouped_optuna") / f"{group_name}_best.json"
    out_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(f"saved -> {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=list(GROUPS.keys()) + ["all"], default="all")
    parser.add_argument("--n-trials", type=int, default=8)
    args = parser.parse_args()

    if args.group == "all":
        for g in ["substrate", "veith", "classification", "regression"]:
            run_study(g, args.n_trials)
    else:
        run_study(args.group, args.n_trials)

if __name__ == "__main__":
    main()
