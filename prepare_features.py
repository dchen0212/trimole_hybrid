"""Generate fixed fingerprint/descriptor inputs without fitting or searching."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV; row order is preserved.")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--task", help="Read the fixed chemical profile from an existing task record.")
    choice.add_argument("--block", help="Explicit archived chemical block for a component.")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--label-column", help="Optional numeric label column, exported unchanged as y.")
    parser.add_argument("--id-column", help="Optional row identifier column.")
    parser.add_argument("--embeddings", type=Path, help="NPZ with aligned C/K/E embedding blocks.")
    parser.add_argument("--candidate", help="Embedding order for --block; task profiles already fix this.")
    parser.add_argument("--base-predictions", type=Path, help="Aligned .npy vector, only for a recorded base-prediction variant.")
    parser.add_argument("--output", required=True, type=Path, help="New .npz output file.")
    args = parser.parse_args()
    try:
        import numpy as np
        import pandas as pd
        from rdkit import rdBase
        from trimole_hybrid.chemical.features import build_features, task_profile, embedding_features
        from trimole_hybrid.registry import load_task
        if args.output.suffix != ".npz":
            parser.error("Use a .npz output path.")
        meta_path = args.output.with_suffix(".features.json")
        if args.output.exists() or meta_path.exists():
            parser.error("Output or feature schema already exists; choose a new path.")
        data = pd.read_csv(args.input)
        if args.smiles_column not in data or data[args.smiles_column].isna().any():
            parser.error("The SMILES column must exist and contain no missing rows.")
        if args.task:
            if args.candidate:
                parser.error("A task profile already fixes its embedding candidate.")
            settings = task_profile(load_task(args.task))
        else:
            settings = {"profile": args.block, "candidate": args.candidate,
                        "base_prediction": bool(args.base_predictions)}
        embedding = None
        if settings["candidate"]:
            if args.embeddings is None:
                parser.error("The fixed profile requires --embeddings.")
            with np.load(args.embeddings, allow_pickle=False) as blocks:
                embedding = embedding_features(blocks, settings["candidate"], len(data))
                if args.id_column and "row_ids" in blocks:
                    if not np.array_equal(data[args.id_column].astype(str).to_numpy(), blocks["row_ids"].astype(str)):
                        parser.error("Embedding row identifiers differ from the input CSV.")
        elif args.embeddings:
            parser.error("This fixed profile uses chemical features only.")
        if bool(args.base_predictions) != settings["base_prediction"]:
            parser.error("Base-prediction input does not match the fixed feature variant.")
        base = np.load(args.base_predictions, allow_pickle=False) if args.base_predictions else None
        smiles = data[args.smiles_column].to_numpy(dtype=str)
        matrix, columns = build_features(smiles, settings["profile"], embedding, base)
        payload = {"X": matrix, "smiles": smiles,
                   "row_ids": data[args.id_column].to_numpy(dtype=str) if args.id_column else np.arange(len(data))}
        if args.label_column:
            payload["y"] = data[args.label_column].to_numpy(dtype=float)
        schema = {"profile": settings, "task": args.task, "rows": len(data),
                  "feature_count": len(columns), "columns": columns,
                  "rdkit_version": rdBase.rdkitVersion,
                  "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  "feature_sha256": hashlib.sha256(matrix.tobytes()).hexdigest()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as target:
            np.savez_compressed(target, **payload)
        with meta_path.open("x") as target:
            json.dump(schema, target, indent=2)
        print("Wrote", len(data), "rows and", len(columns), "fixed feature columns.")
    except (ValueError, KeyError, FileNotFoundError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
