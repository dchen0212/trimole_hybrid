"""Train a recorded final component without searching model configurations."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from trimole_hybrid.registry import load_task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--inspect", action="store_true", help="Print the fixed task configuration and exit.")
    p.add_argument("--component", help="A recorded feature-head component ID.")
    p.add_argument("--seed", type=int, help="A seed listed in the task configuration.")
    p.add_argument("--run-index", type=int, default=1, help="AMES formal run group, 1 through 5.")
    p.add_argument("--features", type=Path, help="Classical fitting NPZ on the recorded feature basis.")
    p.add_argument("--data-dir", type=Path, help="Prepared feature-head task directory.")
    p.add_argument("--output", type=Path, help="New checkpoint file or feature-head output directory.")
    args = p.parse_args()
    try:
        config = load_task(args.task)
        if args.inspect:
            print(json.dumps(config, indent=2))
            return
        if args.output is None or args.seed is None:
            p.error("Training requires --seed and a new --output path. Use --inspect for the recorded settings.")
        if args.output.exists():
            p.error("Output already exists; choose a new path.")
        from trimole_hybrid.training import train, train_neural
        if args.component:
            if args.data_dir is None:
                p.error("Feature-head training requires --data-dir.")
            train_neural(config, args.component, args.seed, args.data_dir, args.output)
        else:
            train(config, args.features, args.output, args.seed, args.run_index)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
