from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the trimole project root (folder that contains `data/`, `trimole/`, etc.)."""
    return Path(__file__).resolve().parents[2]
