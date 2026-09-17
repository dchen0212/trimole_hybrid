from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "valid", "test")
JOINT_MODALITIES = ("chemberta", "kpgt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def finite_count(array: np.ndarray, chunk_size: int = 2048) -> int:
    count = 0
    for start in range(0, len(array), chunk_size):
        count += int(np.size(array[start : start + chunk_size]) - np.isfinite(array[start : start + chunk_size]).sum())
    return count


def main() -> None:
    args = parse_args()
    task_dirs = sorted(
        path
        for path in args.data_root.iterdir()
        if path.is_dir() and all((path / f"{split}.csv").exists() for split in SPLITS)
    )
    rows: list[dict[str, object]] = []
    failures: list[str] = []

    for task_dir in task_dirs:
        split_frames = {
            split: pd.read_csv(task_dir / f"{split}.csv") for split in SPLITS
        }
        split_counts = {split: len(frame) for split, frame in split_frames.items()}
        total = sum(split_counts.values())

        for modality in JOINT_MODALITIES:
            path = task_dir / "embeddings" / f"{modality}.npy"
            if not path.exists():
                failures.append(f"{task_dir.name}:{modality}:missing")
                continue
            array = np.load(path, mmap_mode="r")
            nonfinite = finite_count(array)
            aligned = array.ndim == 2 and array.shape[0] == total
            if not aligned or nonfinite:
                failures.append(
                    f"{task_dir.name}:{modality}:shape={array.shape}:nonfinite={nonfinite}"
                )
            rows.append(
                {
                    "task": task_dir.name,
                    "modality": modality,
                    "storage": "joint_train_valid_test",
                    "split": "all",
                    "expected_rows": total,
                    "actual_rows": array.shape[0],
                    "feature_dimension": array.shape[1] if array.ndim == 2 else "",
                    "dtype": str(array.dtype),
                    "nonfinite_values": nonfinite,
                    "aligned": aligned and nonfinite == 0,
                    "path": str(path.resolve()),
                }
            )

        ept_dimensions: set[int] = set()
        for split in SPLITS:
            path = task_dir / "embeddings_ept" / f"{split}_ept.npy"
            if not path.exists():
                failures.append(f"{task_dir.name}:ept:{split}:missing")
                continue
            array = np.load(path, mmap_mode="r")
            nonfinite = finite_count(array)
            aligned = array.ndim == 2 and array.shape[0] == split_counts[split]
            if array.ndim == 2:
                ept_dimensions.add(int(array.shape[1]))
            if not aligned or nonfinite:
                failures.append(
                    f"{task_dir.name}:ept:{split}:shape={array.shape}:nonfinite={nonfinite}"
                )
            rows.append(
                {
                    "task": task_dir.name,
                    "modality": "ept",
                    "storage": "split_specific",
                    "split": split,
                    "expected_rows": split_counts[split],
                    "actual_rows": array.shape[0],
                    "feature_dimension": array.shape[1] if array.ndim == 2 else "",
                    "dtype": str(array.dtype),
                    "nonfinite_values": nonfinite,
                    "aligned": aligned and nonfinite == 0,
                    "path": str(path.resolve()),
                }
            )
        if len(ept_dimensions) > 1:
            failures.append(f"{task_dir.name}:ept:inconsistent_dimensions={ept_dimensions}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "status": "ok" if not failures else "failed",
        "data_root": str(args.data_root.resolve()),
        "task_count": len(task_dirs),
        "audit_row_count": len(rows),
        "modalities": ["chemberta", "kpgt", "ept"],
        "failures": failures,
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
