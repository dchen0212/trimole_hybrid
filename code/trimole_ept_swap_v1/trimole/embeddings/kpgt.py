from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _pick_first_existing(*candidates: str) -> str:
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return candidates[0] if candidates else ""


def _default_kpgt_python() -> str:
    env_override = os.environ.get("KPGT_PYTHON", "")
    return _pick_first_existing(
        env_override,
        "/mnt/afs/250010150/envs/kpgt/bin/python",
    )


def _default_kpgt_path() -> str:
    return os.environ.get("KPGT_PATH", "/mnt/afs/250010150/zhensheng/KPGT")


DEFAULT_KPGT_PYTHON = _default_kpgt_python()
DEFAULT_KPGT_PATH = _default_kpgt_path()
DEFAULT_KPGT_MODEL_PATH = os.environ.get("KPGT_MODEL_PATH", os.path.join(DEFAULT_KPGT_PATH, "models", "base.pth"))


def build_kpgt_from_smiles_list(
    smiles_list: List[str],
    output_npy: str | Path,
    *,
    config: str = "base",
    model_path: str = DEFAULT_KPGT_MODEL_PATH,
    n_jobs: int = 32,
    path_length: int = 5,
    batch_size: int = 128,
    num_workers: int = 8,
    kpgt_python: str = DEFAULT_KPGT_PYTHON,
    kpgt_path: str = DEFAULT_KPGT_PATH,
    log_path: str | Path | None = None,
    dataset_name: str = "data_new",
) -> None:
    """
    Export KPGT (LiGhT) latent features for a list of SMILES.

    This wrapper calls KPGT's official scripts in a separate conda env:
      1) preprocess_downstream_dataset.py
      2) extract_features.py

    It then saves the extracted features into output_npy as a float32 matrix [N, D].
    """

    output_npy = Path(output_npy)
    output_npy.parent.mkdir(parents=True, exist_ok=True)

    kpgt_path = Path(kpgt_path)
    scripts_dir = kpgt_path / "scripts"
    preprocess_py = scripts_dir / "preprocess_downstream_dataset.py"
    extract_py = scripts_dir / "extract_features.py"

    if not os.path.exists(kpgt_python):
        raise FileNotFoundError(f"KPGT python not found: {kpgt_python}")
    if not preprocess_py.exists():
        raise FileNotFoundError(f"KPGT preprocess script not found: {preprocess_py}")
    if not extract_py.exists():
        raise FileNotFoundError(f"KPGT extract_features script not found: {extract_py}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"KPGT model checkpoint not found: {model_path}")

    log_path = str(log_path) if log_path else f"{output_npy}.kpgt.log"

    def _tail(path: str, max_lines: int = 200) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return "".join(lines[-max_lines:])
        except Exception:
            return ""

    def _run(cmd: list[str], cwd: str) -> None:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log_f:
            log_f.write("\n===== KPGT run =====\n")
            log_f.write("CWD: " + cwd + "\n")
            log_f.write("CMD: " + " ".join(cmd) + "\n")
            log_f.flush()
            p = subprocess.run(
                cmd,
                check=False,
                cwd=cwd,
                env=os.environ.copy(),
                text=True,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
        if p.returncode != 0:
            msg = _tail(log_path)
            raise RuntimeError(
                f"KPGT scripts failed (exit={p.returncode}). See log: {log_path}\n--- log tail ---\n{msg}"
            )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ds_dir = root / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        ds_csv = ds_dir / f"{dataset_name}.csv"

        # KPGT scripts expect a <data_path>/<dataset>/<dataset>.csv with at least a 'smiles' column.
        # We add a dummy label column for robustness; it is not used by extract_features.py.
        pd.DataFrame({"smiles": [str(s) for s in smiles_list], "label": np.zeros(len(smiles_list), dtype=np.int64)}).to_csv(
            ds_csv, index=False
        )

        _run(
            [
                kpgt_python,
                str(preprocess_py),
                "--data_path",
                str(root),
                "--dataset",
                dataset_name,
                "--path_length",
                str(int(path_length)),
                "--n_jobs",
                str(int(n_jobs)),
            ],
            cwd=str(scripts_dir),
        )

        _run(
            [
                kpgt_python,
                str(extract_py),
                "--config",
                str(config),
                "--model_path",
                str(model_path),
                "--data_path",
                str(root),
                "--dataset",
                dataset_name,
                "--batch_size",
                str(int(batch_size)),
                "--num_workers",
                str(int(num_workers)),
            ],
            cwd=str(scripts_dir),
        )

        npz_path = ds_dir / f"kpgt_{config}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"KPGT output not found: {npz_path} (see log: {log_path})")

        arr = np.load(npz_path, allow_pickle=False)["fps"].astype(np.float32)
        if np.isnan(arr).any() or np.isinf(arr).any():
            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            with open(log_path, "a", encoding="utf-8") as log_f:
                log_f.write(f"\n[sanitize] replaced NaN/Inf with 0. nan={nan_count} inf={inf_count}\n")

        np.save(output_npy, arr)


def build_kpgt_from_csv(
    data_csv: str | Path,
    output_npy: str | Path,
    *,
    smiles_col: str = "smiles",
    config: str = "base",
    model_path: str = DEFAULT_KPGT_MODEL_PATH,
    n_jobs: int = 32,
    path_length: int = 5,
    batch_size: int = 128,
    num_workers: int = 8,
    kpgt_python: str = DEFAULT_KPGT_PYTHON,
    kpgt_path: str = DEFAULT_KPGT_PATH,
    log_path: str | Path | None = None,
) -> None:
    df = pd.read_csv(str(data_csv))
    if smiles_col not in df.columns:
        raise ValueError(f"{data_csv} must contain column: {smiles_col}")
    smiles_list = df[smiles_col].astype(str).tolist()
    build_kpgt_from_smiles_list(
        smiles_list,
        output_npy=output_npy,
        config=config,
        model_path=model_path,
        n_jobs=n_jobs,
        path_length=path_length,
        batch_size=batch_size,
        num_workers=num_workers,
        kpgt_python=kpgt_python,
        kpgt_path=kpgt_path,
        log_path=log_path,
        dataset_name=Path(data_csv).stem,
    )

