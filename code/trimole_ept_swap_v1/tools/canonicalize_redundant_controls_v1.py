#!/usr/bin/env python3
"""Canonicalize mathematically redundant top-3 and uniform predictions.

When the candidate pool contains exactly three modalities, validation top-3
and uniform averaging have the same definition.  Summing arrays in different
orders can nevertheless perturb tied classification scores at machine epsilon
and change rank-based metrics.  This script removes that numerical artifact
only after strict set, shape, label and tolerance checks.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


MODALITIES = {"chemberta", "kpgt", "ept"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args()


def prediction_column(frame: pd.DataFrame) -> str:
    matches = [name for name in ("prediction", "y_pred", "y_prob", "pred") if name in frame.columns]
    if len(matches) != 1:
        raise ValueError(f"expected one prediction column, found {matches}")
    return matches[0]


def main() -> None:
    args = parse_args()
    selection_path = args.result_root / "selection" / "selected_controls.json"
    manifest_path = args.result_root / "final" / "prediction_manifest.csv"
    selection = json.loads(selection_path.read_text())
    manifest = pd.read_csv(manifest_path)
    records: list[dict[str, object]] = []

    for (task, seed), group in manifest.groupby(["task", "seed"], sort=True):
        ranking = selection["per_task_validation_ranking"][task]
        if set(ranking[:3]) != MODALITIES or len(ranking[:3]) != 3:
            raise ValueError(f"{task}: top-3 is not exactly the complete modality set")
        top3_row = group[group["control"] == "validation_top3_average"]
        uniform_row = group[group["control"] == "uniform_average"]
        if len(top3_row) != 1 or len(uniform_row) != 1:
            raise ValueError(f"{task} seed {seed}: missing unique redundant controls")
        top3_path = Path(top3_row.iloc[0]["prediction_file"])
        uniform_path = Path(uniform_row.iloc[0]["prediction_file"])
        top3 = pd.read_csv(top3_path)
        uniform = pd.read_csv(uniform_path)
        if list(top3.columns) != list(uniform.columns) or len(top3) != len(uniform):
            raise ValueError(f"{task} seed {seed}: prediction schema mismatch")
        column = prediction_column(top3)
        max_abs_diff = float(np.max(np.abs(top3[column].to_numpy() - uniform[column].to_numpy())))
        if not np.isfinite(max_abs_diff) or max_abs_diff > args.tolerance:
            raise ValueError(
                f"{task} seed {seed}: max difference {max_abs_diff} exceeds {args.tolerance}"
            )
        temporary = top3_path.with_suffix(top3_path.suffix + ".tmp")
        # Preserve the exact canonical byte representation. Re-serializing the
        # parsed frame can change the final decimal digit of binary floats.
        shutil.copyfile(uniform_path, temporary)
        temporary.replace(top3_path)
        records.append(
            {
                "task": task,
                "seed": int(seed),
                "max_abs_diff_before": max_abs_diff,
                "top3_prediction_file": str(top3_path),
                "uniform_prediction_file": str(uniform_path),
                "action": "top3_replaced_with_identical_canonical_uniform_array",
            }
        )

    audit = pd.DataFrame(records)
    audit_path = args.result_root / "final" / "redundant_control_canonicalization.csv"
    audit.to_csv(audit_path, index=False)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_file": str(selection_path),
        "manifest_file": str(manifest_path),
        "tolerance": args.tolerance,
        "rows": len(audit),
        "max_abs_diff_before": float(audit["max_abs_diff_before"].max()),
        "reason": "prevent machine-epsilon summation order from changing tied rank metrics",
    }
    (args.result_root / "final" / "redundant_control_canonicalization.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
