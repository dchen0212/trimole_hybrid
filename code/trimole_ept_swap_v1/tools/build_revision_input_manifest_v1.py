#!/usr/bin/env python3
"""Freeze official split metadata and hashes without loading model code."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SPLITS = ("train", "valid", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_csv(path: Path) -> dict[str, object]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty CSV: {path}") from exc
        rows = sum(1 for _ in reader)
    return {
        "path": str(path.resolve()),
        "rows": rows,
        "columns": len(header),
        "column_names": json.dumps(header, ensure_ascii=True),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    tasks = sorted(
        path.name
        for path in args.data_root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )
    records: list[dict[str, object]] = []
    for task in tasks:
        for split in SPLITS:
            path = args.data_root / task / f"{split}.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append({"task": task, "split": split, **inspect_csv(path)})

    if len(tasks) != 22:
        raise ValueError(f"expected 22 tasks, found {len(tasks)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "split",
        "rows",
        "columns",
        "column_names",
        "bytes",
        "sha256",
        "path",
    ]
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} split records for {len(tasks)} tasks to {args.out}")


if __name__ == "__main__":
    main()
