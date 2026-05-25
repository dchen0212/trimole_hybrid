#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

SETTINGS = {
    "full": set(),
    "drop_smiles": {"smiles"},
    "drop_graph": {"graph"},
    "drop_3d": {"3d"},
}

# 按你当前项目里三种 embedding 维度识别模态
MODALITY_BY_DIM = {
    768: "smiles",   # ChemBERTa
    512: "3d",       # UniMol
    2304: "graph",   # KPGT
}

def infer_modality_from_npy(path: Path) -> str | None:
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception:
        return None

    if arr.ndim == 1:
        dim = int(arr.shape[0])
    elif arr.ndim >= 2:
        dim = int(arr.shape[-1])
    else:
        return None

    return MODALITY_BY_DIM.get(dim)

def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def zero_out_file(path: Path) -> dict:
    arr = np.load(path)
    arr_zero = np.zeros_like(arr)
    np.save(path, arr_zero)
    return {
        "file": str(path),
        "shape": list(arr.shape),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=str, default="data/data_benchmark")
    parser.add_argument("--dst-root", type=str, default="data/data_benchmark_ablation")
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    if args.tasks:
        tasks = args.tasks
    else:
        tasks = sorted([p.name for p in src_root.iterdir() if p.is_dir()])

    manifest = {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "tasks": tasks,
        "settings": {},
    }

    for setting, dropped in SETTINGS.items():
        setting_root = dst_root / setting
        setting_root.mkdir(parents=True, exist_ok=True)
        manifest["settings"][setting] = {
            "dropped_modalities": sorted(list(dropped)),
            "tasks": {},
        }

        for task in tasks:
            src_task_dir = src_root / task
            dst_task_dir = setting_root / task

            copy_tree(src_task_dir, dst_task_dir)

            emb_dir = dst_task_dir / "embeddings"
            task_info = {
                "zeroed_files": [],
                "kept_files": [],
                "unknown_files": [],
            }

            if emb_dir.exists():
                for npy in sorted(emb_dir.rglob("*.npy")):
                    modality = infer_modality_from_npy(npy)
                    rel = str(npy.relative_to(dst_task_dir))

                    if modality is None:
                        task_info["unknown_files"].append(rel)
                        continue

                    if modality in dropped:
                        meta = zero_out_file(npy)
                        meta["relative_path"] = rel
                        meta["modality"] = modality
                        task_info["zeroed_files"].append(meta)
                    else:
                        task_info["kept_files"].append({
                            "relative_path": rel,
                            "modality": modality,
                        })

            manifest["settings"][setting]["tasks"][task] = task_info

    out_manifest = dst_root / "ablation_manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Saved manifest: {out_manifest}")

if __name__ == "__main__":
    main()
