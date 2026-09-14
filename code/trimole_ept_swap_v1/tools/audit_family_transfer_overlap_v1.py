"""Audit exact molecular overlap in the family-transfer task groups."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from family_transfer_safety import (
    anonymized_key,
    canonical_connectivity_keys,
    overlap_stats,
)


GROUPS = {
    "cyp_substrate": [
        "cyp2c9_substrate_carbonmangels",
        "cyp2d6_substrate_carbonmangels",
        "cyp3a4_substrate_carbonmangels",
    ],
    "clearance": [
        "clearance_hepatocyte_az",
        "clearance_microsome_az",
    ],
}
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--group", choices=("all", *GROUPS), default="all")
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--fail-on-overlap", action="store_true")
    return parser.parse_args()


def read_smiles(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"missing header: {path}")
        lookup = {name.lower(): name for name in reader.fieldnames}
        column = next((lookup[name] for name in ("smiles", "drug") if name in lookup), None)
        if column is None:
            raise KeyError(f"SMILES column not found in {path}")
        return [row[column] for row in reader]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    groups = GROUPS if args.group == "all" else {args.group: GROUPS[args.group]}
    summary: list[dict[str, object]] = []
    details: list[dict[str, object]] = []

    for group, tasks in groups.items():
        keys: dict[tuple[str, str], list[str]] = {}
        for task in tasks:
            for split in SPLITS:
                keys[(task, split)] = canonical_connectivity_keys(
                    read_smiles(args.data_root / task / f"{split}.csv")
                )

        for target in tasks:
            for target_split in ("valid", "test"):
                holdout = keys[(target, target_split)]
                holdout_set = set(holdout)
                for source in tasks:
                    for source_split in ("train", "valid"):
                        source_keys = keys[(source, source_split)]
                        stats = overlap_stats(source_keys, holdout)
                        summary.append(
                            {
                                "group": group,
                                "target_task": target,
                                "target_split": target_split,
                                "source_task": source,
                                "source_split": source_split,
                                **stats,
                            }
                        )
                        for key in sorted(set(source_keys) & holdout_set):
                            details.append(
                                {
                                    "group": group,
                                    "target_task": target,
                                    "target_split": target_split,
                                    "source_task": source,
                                    "source_split": source_split,
                                    "molecule_identity_sha256": anonymized_key(key),
                                }
                            )

    write_csv(args.out_root / "family_transfer_overlap_summary.csv", summary)
    write_csv(args.out_root / "family_transfer_overlap_details_hashed.csv", details)
    overlap_total = sum(int(row["overlap_unique_molecules"]) for row in summary)
    print(f"wrote {len(summary)} comparisons; overlap count across comparisons={overlap_total}")
    if args.fail_on_overlap and overlap_total:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
