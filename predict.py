"""Predict with a fixed component or apply a task's recorded combination."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from trimole_hybrid.registry import load_task, checkpoint_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=Path)
    mode.add_argument("--component-predictions", type=Path, help="NPZ of aligned, per-run component predictions.")
    p.add_argument("--features", type=Path)
    p.add_argument("--metadata", type=Path, help="Feature-head meta.json, when not bound by --artifact-id.")
    p.add_argument("--artifact-id", help="Artifact ID in checkpoints/manifest.json for this task.")
    p.add_argument("--sha256", help="Expected checksum for a checkpoint without a training sidecar.")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    try:
        import numpy as np
        from trimole_hybrid.prediction import predict_classical, predict_head
        from trimole_hybrid.composition import task_prediction
        config = load_task(args.task)
        if args.output.suffix != ".npy":
            p.error("Use a .npy output path.")
        if args.output.exists():
            p.error("Output already exists; choose a new path.")
        if args.component_predictions:
            with np.load(args.component_predictions, allow_pickle=False) as inputs:
                result = task_prediction(config, inputs)
        else:
            if args.features is None:
                p.error("Checkpoint prediction requires --features.")
            checksum = args.sha256
            meta = json.loads(args.metadata.read_text()) if args.metadata else None
            if args.artifact_id:
                record = next((r for r in checkpoint_manifest()["records"]
                               if r["id"] == args.artifact_id and r["task"] == args.task), None)
                if record is None:
                    p.error("Artifact is not registered for this task.")
                checksum = record["sha256"]
                if meta is None:
                    meta = next((r for c in config.get("components_bound", [])
                                 for r in c.get("fitted_runs", [])
                                 if r.get("checkpoint_id") == args.artifact_id), None)
            if meta is None and args.checkpoint.suffix in {".pth", ".pt"}:
                sidecar = args.checkpoint.parent / "meta.json"
                if sidecar.is_file():
                    meta = json.loads(sidecar.read_text())
            if meta is not None:
                if not checksum:
                    p.error("Feature-head prediction requires --sha256 or a registered --artifact-id.")
                result = predict_head(args.checkpoint, args.features,
                                      {"sha256": checksum, "model_metadata": meta})
            else:
                result = predict_classical(args.checkpoint, args.features, config, checksum)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as output:
            np.save(output, result)
    except (ValueError, KeyError, FileNotFoundError, ImportError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
