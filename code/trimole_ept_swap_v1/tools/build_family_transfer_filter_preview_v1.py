#!/usr/bin/env python3
"""Preview exact row removals imposed by the leakage-safe family policy."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


FAMILIES = {
    "cyp_substrate": [
        "cyp2c9_substrate_carbonmangels",
        "cyp2d6_substrate_carbonmangels",
        "cyp3a4_substrate_carbonmangels",
    ],
    "clearance": ["clearance_hepatocyte_az", "clearance_microsome_az"],
}
FORMAL_TARGETS = {
    "cyp3a4_substrate_carbonmangels",
    "clearance_hepatocyte_az",
}


def read_smiles(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"missing header: {path}")
        lookup = {name.lower(): name for name in reader.fieldnames}
        column = next((lookup[name] for name in ("smiles", "drug") if name in lookup), None)
        if column is None:
            raise KeyError(f"SMILES column not found: {path}")
        return [row[column] for row in reader]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--tools-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.tools_root))
    from family_transfer_safety import (  # noqa: PLC0415
        canonical_connectivity_keys,
        forbidden_keys,
        keep_mask,
    )

    rows: list[dict[str, object]] = []
    for family, tasks in FAMILIES.items():
        keys = {
            (task, split): canonical_connectivity_keys(
                read_smiles(args.data_root / task / f"{split}.csv")
            )
            for task in tasks
            for split in ("train", "valid", "test")
        }
        for target in tasks:
            if target not in FORMAL_TARGETS:
                continue
            for stage, source_splits in (
                ("selection", ("train",)),
                ("final", ("train", "valid")),
            ):
                blocked = forbidden_keys(
                    keys[(target, "valid")], keys[(target, "test")], stage
                )
                for source in tasks:
                    for source_split in source_splits:
                        source_keys = keys[(source, source_split)]
                        mask = keep_mask(source_keys, blocked)
                        removed_keys = {
                            key for key, keep in zip(source_keys, mask, strict=True) if not keep
                        }
                        rows.append(
                            {
                                "family": family,
                                "target_task": target,
                                "stage": stage,
                                "source_task": source,
                                "source_split": source_split,
                                "rows_before": len(mask),
                                "rows_removed": mask.count(False),
                                "unique_identities_removed": len(removed_keys),
                                "rows_after": mask.count(True),
                            }
                        )

    write_rows(args.out, rows)
    for target in sorted(FORMAL_TARGETS):
        for stage in ("selection", "final"):
            subset = [
                row
                for row in rows
                if row["target_task"] == target and row["stage"] == stage
            ]
            print(
                target,
                stage,
                "rows_removed=",
                sum(int(row["rows_removed"]) for row in subset),
                "of",
                sum(int(row["rows_before"]) for row in subset),
            )


if __name__ == "__main__":
    main()
