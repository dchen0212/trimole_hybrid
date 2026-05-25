from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_import_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))


def main() -> None:
    _bootstrap_import_path()
    from trimole.analysis.compare_baselines import main as _main

    _main()


if __name__ == "__main__":
    main()
