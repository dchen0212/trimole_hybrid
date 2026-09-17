from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--overlap-summary", type=Path, required=True)
    parser.add_argument("--safe-run-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    return parser.parse_args()


def label_column(frame: pd.DataFrame) -> str:
    return "Y" if "Y" in frame.columns else "label"


def split_summary(data_root: Path, task: str) -> dict[str, object]:
    frames = {
        split: pd.read_csv(data_root / task / f"{split}.csv")
        for split in ("train", "valid", "test")
    }
    labels = pd.concat(
        [frame[label_column(frame)] for frame in frames.values()], ignore_index=True
    ).to_numpy(dtype=np.float64)
    unique = np.unique(labels[np.isfinite(labels)])
    binary = len(unique) <= 2 and set(unique).issubset({0.0, 1.0})
    return {
        "n_total": sum(len(frame) for frame in frames.values()),
        "n_train": len(frames["train"]),
        "n_valid": len(frames["valid"]),
        "n_test": len(frames["test"]),
        "n_unique_smiles": pd.concat(
            [frame.smiles for frame in frames.values()], ignore_index=True
        ).nunique(),
        "positive_n": int(np.sum(labels == 1.0)) if binary else pd.NA,
        "positive_prevalence": float(np.mean(labels == 1.0)) if binary else pd.NA,
        "label_min": float(np.min(labels)),
        "label_max": float(np.max(labels)),
    }


def build_s19(args: argparse.Namespace) -> pd.DataFrame:
    catalog = pd.read_csv(args.catalog)
    rows = []
    for record in catalog.to_dict("records"):
        rows.append(
            {
                **record,
                **split_summary(args.data_root, str(record["task"])),
                "split_protocol": "frozen TDC ADMET scaffold split",
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 22:
        raise ValueError(f"expected 22 tasks, found {len(result)}")
    return result


def build_s20(args: argparse.Namespace) -> pd.DataFrame:
    overlap = pd.read_csv(args.overlap_summary)
    overlap = overlap[overlap.source_task != overlap.target_task].copy()
    overlap["identity_key"] = "RDKit canonical parent InChIKey connectivity layer"
    overlap["audit_interpretation"] = np.where(
        overlap.overlap_unique_molecules > 0,
        "cross-task molecular identity overlap detected",
        "no cross-task molecular identity overlap detected",
    )

    correction_rows = []
    for run_root in sorted(args.safe_run_root.glob("revision_20260917_*_leakage_safe_v2")):
        audit_path = run_root / "holdout_overlap_filter_audit.csv"
        score_path = run_root / "final_selected_test_summary.csv"
        provenance_path = run_root / "provenance.json"
        if not (audit_path.exists() and score_path.exists() and provenance_path.exists()):
            continue
        audit = pd.read_csv(audit_path)
        score = pd.read_csv(score_path).iloc[0]
        for stage, stage_rows in audit.groupby("stage", sort=True):
            correction_rows.append(
                {
                    "record_type": "filter_and_corrected_score",
                    "group": "clearance" if "clearance" in score.target_task else "cyp_substrate",
                    "target_task": score.target_task,
                    "target_split": "valid+test" if stage == "selection" else "test",
                    "source_task": ";".join(sorted(stage_rows.source_task.unique())),
                    "source_split": ";".join(sorted(stage_rows.source_split.unique())),
                    "source_rows": int(stage_rows.rows_before.sum()),
                    "source_unique_molecules": pd.NA,
                    "holdout_rows": pd.NA,
                    "holdout_unique_molecules": pd.NA,
                    "overlap_source_rows": int(
                        stage_rows.rows_removed_for_holdout_overlap.sum()
                    ),
                    "overlap_unique_molecules": pd.NA,
                    "identity_key": "RDKit canonical parent InChIKey connectivity layer",
                    "audit_interpretation": (
                        f"{stage} stage removed all rows matching forbidden holdout identities; "
                        f"corrected {score.metric}={score.test_mean:.6f}+/-{score.test_std:.6f}"
                    ),
                    "provenance_file": str(provenance_path),
                }
            )
    overlap.insert(0, "record_type", "pairwise_overlap_audit")
    overlap["provenance_file"] = str(args.overlap_summary)
    return pd.concat([overlap, pd.DataFrame(correction_rows)], ignore_index=True, sort=False)


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    build_s19(args).to_csv(args.out_root / "Table_S19_dataset_provenance.csv", index=False)
    build_s20(args).to_csv(args.out_root / "Table_S20_cross_task_leakage_audit.csv", index=False)


if __name__ == "__main__":
    main()
