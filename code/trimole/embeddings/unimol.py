from __future__ import annotations

import os
from contextlib import contextmanager
from typing import List

import numpy as np


@contextmanager
def _temporary_cwd(path: str):
    old = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def build_unimol(
    smiles_list: List[str],
    fallback_dim: int = 512,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Build UniMol molecular representations for a list of SMILES.
    
    Args:
        smiles_list: List of SMILES strings.
        fallback_dim: Fallback embedding dimension if model fails.
        batch_size: Batch size for UniMol inference. Default 256 is good for H100.
                    Increase to 512 for larger GPU memory, decrease if OOM.
    """
    # `unimol_tools` writes logs to `<cwd>/logs/unimol_tools_*.log` at import-time.
    # To avoid polluting this repo with a top-level `logs/` folder, we temporarily
    # switch CWD during import/initialization.
    unimol_tools_workdir = os.environ.get("TRIMOLE_UNIMOL_TOOLS_WORKDIR", "/tmp/trimole_unimol_tools")
    with _temporary_cwd(unimol_tools_workdir):
        from unimol_tools import UniMolRepr

        clf = UniMolRepr(data_type="molecule", remove_hs=False, batch_size=batch_size)

    reprs = clf.get_repr(smiles_list, return_atomic_reprs=False)
    embs = np.asarray(reprs, dtype=np.float32)
    return embs
