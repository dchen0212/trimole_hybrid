from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-bootstrap", type=Path, required=True)
    parser.add_argument("--subgroup-uncertainty", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_nonempty(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty input table: {path}")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return frame


def main() -> None:
    args = parse_args()
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")

    paired = load_nonempty(
        args.paired_bootstrap,
        {
            "task",
            "metric",
            "reference_model",
            "comparator_model",
            "improvement",
            "ci95_lower",
            "ci95_upper",
            "p_value_two_sided",
            "p_value_bh_fdr",
        },
    )
    subgroup = load_nonempty(
        args.subgroup_uncertainty,
        {
            "task",
            "metric",
            "descriptor",
            "group",
            "n_samples",
            "metric_valid",
            "reference_model",
            "comparator_model",
        },
    )

    paired.insert(0, "record_type", "task_paired_bootstrap")
    subgroup.insert(0, "record_type", "property_subgroup")
    combined = pd.concat([paired, subgroup], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["record_type", "task", "comparator_model"], kind="stable"
    ).reset_index(drop=True)

    args.out_root.mkdir(parents=True)
    output_path = args.out_root / "Table_S22_uncertainty_and_subgroups.csv"
    combined.to_csv(output_path, index=False)
    (args.out_root / "provenance.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "operation": "row-wise union without statistical recomputation",
                "paired_bootstrap": {
                    "path": str(args.paired_bootstrap),
                    "sha256": sha256(args.paired_bootstrap),
                    "rows": len(paired),
                },
                "subgroup_uncertainty": {
                    "path": str(args.subgroup_uncertainty),
                    "sha256": sha256(args.subgroup_uncertainty),
                    "rows": len(subgroup),
                },
                "output": {
                    "path": str(output_path),
                    "sha256": sha256(output_path),
                    "rows": len(combined),
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
