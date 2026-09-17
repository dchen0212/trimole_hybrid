#!/usr/bin/env python3
"""Create reviewer-facing overlap summaries from the immutable raw audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


FORMAL_TARGETS = {
    "cyp3a4_substrate_carbonmangels",
    "clearance_hepatocyte_az",
}
INTEGER_FIELDS = {
    "holdout_rows",
    "holdout_unique_molecules",
    "overlap_source_rows",
    "overlap_unique_molecules",
    "source_rows",
    "source_unique_molecules",
}


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows: list[dict[str, object]] = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    key: int(value) if key in INTEGER_FIELDS else value
                    for key, value in row.items()
                }
            )
        return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--details", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    args = parser.parse_args()

    rows = read_rows(args.summary)
    formal = [row for row in rows if row["target_task"] in FORMAL_TARGETS]
    formal.sort(
        key=lambda row: (
            str(row["target_task"]),
            str(row["target_split"]),
            str(row["source_task"]),
            str(row["source_split"]),
        )
    )
    write_rows(args.out_root / "formal_endpoint_overlap_all_comparisons.csv", formal)

    cross_task_test = [
        row
        for row in rows
        if row["target_split"] == "test"
        and row["source_task"] != row["target_task"]
        and int(row["overlap_unique_molecules"]) > 0
    ]
    cross_task_test.sort(
        key=lambda row: (
            str(row["target_task"]),
            str(row["source_task"]),
            str(row["source_split"]),
        )
    )
    write_rows(args.out_root / "cross_task_to_target_test_nonzero.csv", cross_task_test)

    identities: dict[tuple[str, str], set[str]] = defaultdict(set)
    with args.details.open(newline="") as handle:
        for row in csv.DictReader(handle):
            identities[(row["target_task"], row["target_split"])].add(
                row["molecule_identity_sha256"]
            )

    compact: list[dict[str, object]] = []
    for target in sorted(FORMAL_TARGETS):
        for split in ("valid", "test"):
            comparisons = [
                row
                for row in formal
                if row["target_task"] == target and row["target_split"] == split
            ]
            external = [row for row in comparisons if row["source_task"] != target]
            own = [row for row in comparisons if row["source_task"] == target]
            compact.append(
                {
                    "target_task": target,
                    "target_split": split,
                    "unique_overlapping_holdout_identities_any_source": len(
                        identities[(target, split)]
                    ),
                    "external_comparison_overlap_unique_sum": sum(
                        int(row["overlap_unique_molecules"]) for row in external
                    ),
                    "own_comparison_overlap_unique_sum": sum(
                        int(row["overlap_unique_molecules"]) for row in own
                    ),
                    "note": "comparison sums can double-count identities across sources",
                }
            )
    write_rows(args.out_root / "formal_endpoint_overlap_compact.csv", compact)

    payload = {
        "raw_comparisons": len(rows),
        "raw_comparison_overlap_unique_sum": sum(
            int(row["overlap_unique_molecules"]) for row in rows
        ),
        "nonzero_cross_task_to_target_test_comparisons": len(cross_task_test),
        "formal_endpoint_holdout_identity_counts": compact,
    }
    (args.out_root / "overlap_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
