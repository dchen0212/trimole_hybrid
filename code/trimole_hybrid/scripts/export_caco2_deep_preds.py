from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from trimole.training.trainer import (
    dump_regression_predictions,
    MultiModalDataset,
    _read_split,
    _load_concat_embeddings,
    _slice,
)
from trimole.models.model import MultiModalFusionMLP


RUN_DIR = Path("results/model_log/admet17_ex_top1_optuna/regression/trial_004/run_20260410_1807/caco2_wang")
TASK_DIR = Path("data/data_benchmark/caco2_wang")
OUT_DIR = Path("results/stacking_inputs/caco2_wang")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = json.loads((RUN_DIR / "meta.json").read_text())
    cfg = meta["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = TASK_DIR / "train.csv"
    valid_csv = TASK_DIR / "valid.csv"
    test_csv = TASK_DIR / "test.csv"
    embeddings_root = TASK_DIR / "embeddings"

    n_tr, y_tr = _read_split(train_csv)
    n_va, y_va = _read_split(valid_csv)
    n_te, y_te = _read_split(test_csv)

    y_tr = np.asarray(y_tr).astype(np.float32)
    y_va = np.asarray(y_va).astype(np.float32)
    y_te = np.asarray(y_te).astype(np.float32)

    label_mean = float(meta["label_mean"])
    label_std = float(meta["label_std"])

    # trainer.py 训练时：train/valid 做 z-score，test 保持原尺度
    y_va_norm = (y_va - label_mean) / label_std

    emb_all_s, emb_all_3d, emb_all_g = _load_concat_embeddings(embeddings_root)

    off_tr = 0
    off_va = off_tr + n_tr
    off_te = off_va + n_va

    emb_va_s = _slice(emb_all_s, off_va, n_va, "chemberta", embeddings_root)
    emb_va_3d = _slice(emb_all_3d, off_va, n_va, "unimol", embeddings_root)
    emb_va_g = _slice(emb_all_g, off_va, n_va, "kpgt", embeddings_root)

    emb_te_s = _slice(emb_all_s, off_te, n_te, "chemberta", embeddings_root)
    emb_te_3d = _slice(emb_all_3d, off_te, n_te, "unimol", embeddings_root)
    emb_te_g = _slice(emb_all_g, off_te, n_te, "kpgt", embeddings_root)

    valid_loader = DataLoader(
        MultiModalDataset(emb_va_s, emb_va_3d, emb_va_g, y_va_norm, task_type="regression"),
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
    )
    test_loader = DataLoader(
        MultiModalDataset(emb_te_s, emb_te_3d, emb_te_g, y_te, task_type="regression"),
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
    )

    model = MultiModalFusionMLP(
        dim_smiles=int(meta["dims"]["chemberta"]),
        dim_3d=int(meta["dims"]["unimol"]),
        dim_graph=int(meta["dims"]["kpgt"]),
        out_dim=1,
        hidden_dim=int(cfg["hidden_dim"]),
        dropout_proj=float(cfg["dropout_proj"]),
        dropout_head=float(cfg["dropout_head"]),
        fusion_type=str(cfg.get("fusion_type", "mlp")),
    ).to(device)

    best_path = RUN_DIR / "best_model.pth"
    model.load_state_dict(torch.load(best_path, map_location=device))

    valid_out = OUT_DIR / "valid_deep_raw.csv"
    test_out = OUT_DIR / "test_deep_raw.csv"

    dump_regression_predictions(
        model=model,
        loader=valid_loader,
        device=device,
        out_csv=valid_out,
        label_mean=label_mean,
        label_std=label_std,
        modality_mask=None,
    )
    dump_regression_predictions(
        model=model,
        loader=test_loader,
        device=device,
        out_csv=test_out,
        label_mean=label_mean,
        label_std=label_std,
        modality_mask=None,
    )

    # 转成 stacking_oneclick.py 需要的列名：pred
    valid_df = pd.read_csv(valid_out).rename(columns={"y_pred": "pred"})
    test_df = pd.read_csv(test_out).rename(columns={"y_pred": "pred"})

    valid_df.to_csv(OUT_DIR / "valid_deep.csv", index=False)
    test_df.to_csv(OUT_DIR / "test_deep.csv", index=False)

    print("saved ->", OUT_DIR / "valid_deep.csv")
    print("saved ->", OUT_DIR / "test_deep.csv")
    print()
    print("valid preview:")
    print(valid_df.head().to_string(index=False))
    print()
    print("test preview:")
    print(test_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
