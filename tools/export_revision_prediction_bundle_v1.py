"""Archive label-free common-pool predictions with source checksums."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prediction_csv(path: Path, prediction_column: str) -> tuple[bytes, int]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or prediction_column not in reader.fieldnames:
            raise ValueError(f"missing {prediction_column}: {path}")
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("sample_idx", "prediction"))
        count = 0
        for count, row in enumerate(reader, start=1):
            if "sample_idx" in reader.fieldnames and int(row["sample_idx"]) != count - 1:
                raise ValueError(f"nonconsecutive sample index: {path}")
            writer.writerow((count - 1, row[prediction_column]))
    return output.getvalue().encode(), count


def build_bundle(
    candidate_root: Path,
    score_root: Path,
    selection_manifest: Path,
    destination: Path,
) -> dict[str, object]:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    frozen = json.loads(selection_manifest.read_text())
    families = list(frozen["candidate_pool"])
    tasks = list(frozen["tasks"])
    seeds = [int(seed) for seed in frozen["seeds"]]
    entries: list[dict[str, object]] = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for family in families:
            for task in tasks:
                for seed in seeds:
                    for split in ("valid", "test"):
                        source = candidate_root / family / task / f"seed_{seed}" / f"{split}_predictions.csv"
                        payload, rows = prediction_csv(source, "y_pred")
                        member = f"candidate/{family}/{task}/seed_{seed}/{split}.csv"
                        archive.writestr(member, payload)
                        entries.append({"member": member, "rows": rows, "source_sha256": digest(source), "prediction_sha256": hashlib.sha256(payload).hexdigest()})
        methods = sorted(path.name for path in (score_root / "predictions" / tasks[0]).glob("*_seed_*.csv"))
        method_names = sorted({name.rsplit("_seed_", 1)[0] for name in methods})
        for method in method_names:
            for task in tasks:
                for seed in seeds:
                    source = score_root / "predictions" / task / f"{method}_seed_{seed}.csv"
                    payload, rows = prediction_csv(source, "prediction")
                    member = f"method/{method}/{task}/seed_{seed}/test.csv"
                    archive.writestr(member, payload)
                    entries.append({"member": member, "rows": rows, "source_sha256": digest(source), "prediction_sha256": hashlib.sha256(payload).hexdigest()})
        manifest = {
            "selection_manifest_sha256": digest(selection_manifest),
            "candidate_families": families,
            "tasks": tasks,
            "seeds": seeds,
            "methods": method_names,
            "entries": entries,
            "policy": "All exported CSVs contain only sample_idx and prediction; no labels, SMILES or model weights are included.",
            "limitation": "Official TDC split labels and exact candidate-generation environments are required for independent score reproduction. The AqSolDB candidate test array contains two legacy rows absent from the aligned method outputs; use the published alignment audit.",
        }
        archive.writestr("MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    return {"files": len(entries), "methods": len(method_names), "sha256": digest(destination)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.candidate_root, args.score_root, args.selection_manifest, args.destination), indent=2))


if __name__ == "__main__":
    main()
