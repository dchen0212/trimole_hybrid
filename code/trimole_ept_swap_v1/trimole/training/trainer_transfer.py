from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset

from trimole.models.model import MultiModalFusionMLP
from trimole.training.losses import FocalLoss, SpearmanLoss, SmoothL1LossWrapper


@dataclass
class TrainConfig:
    hidden_dim: int = 128
    batch_size: int = 64
    lr: float = 3e-4
    max_epochs: int = 100
    max_patience: int = 15
    seed: int = 42
    weight_decay: float = 0.0
    dropout_proj: float = 0.2
    dropout_head: float = 0.3
    task_type: str = "auto"  # auto | classification | regression
    primary_metric_name: str = "auto"  # auto | AUROC | AUPRC | MAE | Spearman
    modalities: str = "all"  # all | chemberta | unimol | kpgt
    # Task-specific loss parameters
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    spearman_reg: float = 0.1
    # Classification imbalance / loss selection
    loss_type: str = "auto"  # auto | focal | weighted_ce
    pos_weight: Optional[float] = None
    neg_weight: Optional[float] = None
    # Fusion type for MultiModalFusionMLP
    fusion_type: str = "gated"  # mlp | gated
    # Transfer learning: initialize from a saved best_model.pth
    init_from: Optional[str] = None


class MultiModalDataset(Dataset):
    def __init__(
        self,
        emb_smiles: np.ndarray,
        emb_3d: np.ndarray,
        emb_graph: np.ndarray,
        labels: np.ndarray,
        task_type: str,
    ):
        if not (len(emb_smiles) == len(emb_3d) == len(emb_graph) == len(labels)):
            raise ValueError(
                "Length mismatch: "
                f"smiles={len(emb_smiles)}, 3d={len(emb_3d)}, graph={len(emb_graph)}, labels={len(labels)}"
            )
        self.emb_smiles = emb_smiles
        self.emb_3d = emb_3d
        self.emb_graph = emb_graph
        self.labels = labels
        self.task_type = task_type

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.task_type == "classification":
            label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        else:
            label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return {
            "emb1": torch.from_numpy(self.emb_smiles[idx]).float(),
            "emb2": torch.from_numpy(self.emb_3d[idx]).float(),
            "emb3": torch.from_numpy(self.emb_graph[idx]).float(),
            "label": label,
        }


def infer_task_type(labels: np.ndarray) -> str:
    labels = np.asarray(labels)
    if labels.size == 0:
        return "classification"
    if np.all(np.isfinite(labels)):
        rounded = np.rint(labels)
        if np.allclose(labels, rounded):
            unique = set(int(x) for x in np.unique(rounded))
            if unique.issubset({0, 1}):
                return "classification"
    return "regression"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    labels = labels.astype(int)
    class_counts = np.bincount(labels)
    if len(class_counts) < 2:
        raise ValueError(f"Need 2 classes for classification, got bincount={class_counts}")
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.sum() * 2
    return torch.tensor(class_weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    modality_mask: Optional[Tuple[bool, bool, bool]] = None,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        emb1 = batch["emb1"].to(device)
        emb2 = batch["emb2"].to(device)
        emb3 = batch["emb3"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(emb1, emb2, emb3, modality_mask=modality_mask)
        if labels.dtype.is_floating_point and outputs.ndim == 2 and outputs.shape[1] == 1 and labels.ndim == 1:
            outputs = outputs.view(-1)
            labels = labels.view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality_mask: Optional[Tuple[bool, bool, bool]] = None,
) -> Dict[str, float]:
    model.eval()
    all_probs = []
    all_labels = []
    for batch in loader:
        emb1 = batch["emb1"].to(device)
        emb2 = batch["emb2"].to(device)
        emb3 = batch["emb3"].to(device)
        labels = batch["label"].to(device)
        logits = model(emb1, emb2, emb3, modality_mask=modality_mask)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    if len(set(all_labels)) < 2:
        auc = float("nan")
        auprc = float("nan")
    else:
        auc = float(roc_auc_score(all_labels, all_probs))
        auprc = float(average_precision_score(all_labels, all_probs))
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc = float(accuracy_score(all_labels, preds))
    return {"acc": acc, "auc": auc, "auprc": auprc}


@torch.no_grad()
def evaluate_regression(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_mean: Optional[float] = None,
    label_std: Optional[float] = None,
    modality_mask: Optional[Tuple[bool, bool, bool]] = None,
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []
    for batch in loader:
        emb1 = batch["emb1"].to(device)
        emb2 = batch["emb2"].to(device)
        emb3 = batch["emb3"].to(device)
        labels = batch["label"].to(device)
        outputs = model(emb1, emb2, emb3, modality_mask=modality_mask)
        preds = outputs.view(-1)
        preds_np = preds.detach().cpu().numpy()
        if label_mean is not None and label_std is not None:
            preds_np = preds_np * float(label_std) + float(label_mean)
        all_preds.extend(preds_np.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    mae = float(mean_absolute_error(all_labels, all_preds)) if all_labels else float("nan")
    if all_labels:
        mse = float(mean_squared_error(all_labels, all_preds))
        rmse = float(math.sqrt(mse)) if np.isfinite(mse) else float("nan")
    else:
        rmse = float("nan")

    if all_labels and len(all_labels) >= 2:
        y_true = np.asarray(all_labels, dtype=np.float64)
        y_pred = np.asarray(all_preds, dtype=np.float64)
        if np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred)):
            r_true = pd.Series(y_true).rank(method="average").to_numpy(dtype=np.float64)
            r_pred = pd.Series(y_pred).rank(method="average").to_numpy(dtype=np.float64)
            std_true = float(np.std(r_true))
            std_pred = float(np.std(r_pred))
            spearman = float(np.corrcoef(r_true, r_pred)[0, 1]) if (std_true > 0 and std_pred > 0) else float("nan")
        else:
            spearman = float("nan")
    else:
        spearman = float("nan")

    return {"mae": mae, "rmse": rmse, "spearman": spearman}


def _read_split(split_csv: Path) -> Tuple[int, np.ndarray]:
    df = pd.read_csv(split_csv)
    if "label" not in df.columns:
        raise ValueError(f"{split_csv} must contain column: label")
    labels = df["label"].to_numpy()
    return int(len(df)), labels


def _load_concat_embeddings(embeddings_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def load_one(name: str) -> np.ndarray:
        p = embeddings_root / f"{name}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing embedding file: {p}")
        arr = np.load(p).astype(np.float32)
        if np.isnan(arr).any() or np.isinf(arr).any():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    graph = load_one("kpgt")
    return load_one("chemberta"), load_one("unimol"), graph


def _slice(arr: np.ndarray, start: int, length: int, name: str, embeddings_root: Path) -> np.ndarray:
    end = start + length
    if arr.shape[0] < end:
        raise ValueError(
            f"Embedding rows ({arr.shape[0]}) < required end index ({end}) for {name} at {embeddings_root}. "
            "This trainer expects embeddings built as train+valid+test concatenation."
        )
    return arr[start:end]


def _maybe_load_init_weights(model: nn.Module, init_from: Optional[str], device: torch.device) -> Optional[str]:
    if not init_from:
        return None
    p = Path(init_from).expanduser().resolve()
    if p.is_dir():
        p = p / "best_model.pth"
    if not p.exists():
        raise FileNotFoundError(f"init_from not found: {p}")
    state = torch.load(str(p), map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    msg = f"loaded_init_from={p.name} missing={len(missing)} unexpected={len(unexpected)}"
    return msg


def fit_on_task(
    task_dir: Path,
    out_dir: Path,
    config: TrainConfig,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    task_name = task_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = task_dir / "train.csv"
    valid_csv = task_dir / "valid.csv"
    test_csv = task_dir / "test.csv"
    embeddings_root = task_dir / "embeddings"

    n_tr, y_tr = _read_split(train_csv)
    n_va, y_va = _read_split(valid_csv)
    n_te, y_te = _read_split(test_csv)

    task_type = config.task_type
    if task_type == "auto":
        task_type = infer_task_type(np.asarray(y_tr))
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Invalid task_type={config.task_type}")

    label_mean: Optional[float] = None
    label_std: Optional[float] = None

    if task_type == "classification":
        y_tr = np.asarray(y_tr).astype(int)
        y_va = np.asarray(y_va).astype(int)
        y_te = np.asarray(y_te).astype(int)
    else:
        y_tr = np.asarray(y_tr).astype(np.float32)
        y_va = np.asarray(y_va).astype(np.float32)
        y_te = np.asarray(y_te).astype(np.float32)

        y_tr_finite = y_tr[np.isfinite(y_tr)]
        if y_tr_finite.size == 0:
            label_mean = 0.0
            label_std = 1.0
        else:
            label_mean = float(np.mean(y_tr_finite))
            label_std = float(np.std(y_tr_finite)) + 1e-8

        y_tr = (y_tr - label_mean) / label_std
        y_va = (y_va - label_mean) / label_std

    emb_all_s, emb_all_3d, emb_all_g = _load_concat_embeddings(embeddings_root)
    off_tr = 0
    off_va = off_tr + n_tr
    off_te = off_va + n_va

    emb_tr_s = _slice(emb_all_s, off_tr, n_tr, "chemberta", embeddings_root)
    emb_tr_3d = _slice(emb_all_3d, off_tr, n_tr, "unimol", embeddings_root)
    emb_tr_g = _slice(emb_all_g, off_tr, n_tr, "kpgt", embeddings_root)

    emb_va_s = _slice(emb_all_s, off_va, n_va, "chemberta", embeddings_root)
    emb_va_3d = _slice(emb_all_3d, off_va, n_va, "unimol", embeddings_root)
    emb_va_g = _slice(emb_all_g, off_va, n_va, "kpgt", embeddings_root)

    emb_te_s = _slice(emb_all_s, off_te, n_te, "chemberta", embeddings_root)
    emb_te_3d = _slice(emb_all_3d, off_te, n_te, "unimol", embeddings_root)
    emb_te_g = _slice(emb_all_g, off_te, n_te, "kpgt", embeddings_root)

    modalities = str(config.modalities or "all").lower().strip()
    modality_map = {
        "all": None,
        "chemberta": (True, False, False),
        "kpgt": (False, True, False),
        "unimol": (False, False, True),
        "chemberta_kpgt": (True, True, False),
        "unimol_kpgt": (False, True, True),
        "chemberta_unimol": (True, False, True),
    }
    if modalities not in modality_map:
        raise ValueError(
            f"Invalid config.modalities={config.modalities!r}. "
            "Expected one of: all|chemberta|unimol|kpgt|chemberta_kpgt|unimol_kpgt|chemberta_unimol."
        )
    modality_mask: Optional[Tuple[bool, bool, bool]] = modality_map[modalities]

    train_loader = DataLoader(
        MultiModalDataset(emb_tr_s, emb_tr_3d, emb_tr_g, y_tr, task_type=task_type),
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        MultiModalDataset(emb_va_s, emb_va_3d, emb_va_g, y_va, task_type=task_type),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        MultiModalDataset(emb_te_s, emb_te_3d, emb_te_g, y_te, task_type=task_type),
        batch_size=config.batch_size,
        shuffle=False,
    )

    out_dim = 2 if task_type == "classification" else 1
    model = MultiModalFusionMLP(
        dim_smiles=int(emb_tr_s.shape[1]),
        dim_3d=int(emb_tr_3d.shape[1]),
        dim_graph=int(emb_tr_g.shape[1]),
        out_dim=out_dim,
        hidden_dim=config.hidden_dim,
        dropout_proj=config.dropout_proj,
        dropout_head=config.dropout_head,
        fusion_type=str(getattr(config, "fusion_type", "mlp")),
    ).to(device)

    init_msg = _maybe_load_init_weights(model, config.init_from, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_epoch = -1

    primary_metric_name_cfg = str(config.primary_metric_name or "auto")
    if task_type == "regression":
        m = primary_metric_name_cfg.upper()
        if m == "SPEARMAN":
            primary_metric_name = "Spearman"
        elif m == "RMSE":
            primary_metric_name = "RMSE"
        else:
            primary_metric_name = "MAE"
    else:
        primary_metric_name = "AUPRC" if primary_metric_name_cfg.upper() in {"AUPRC", "AUCPR"} else "AUROC"

    if task_type == "classification":
        if config.pos_weight is not None and config.neg_weight is not None:
            raise ValueError("Specify at most one of pos_weight or neg_weight (not both).")
        if config.pos_weight is not None:
            class_weights = torch.tensor([1.0, float(config.pos_weight)], dtype=torch.float32, device=device)
        elif config.neg_weight is not None:
            class_weights = torch.tensor([float(config.neg_weight), 1.0], dtype=torch.float32, device=device)
        else:
            class_weights = compute_class_weights(y_tr, device)

        loss_type = str(config.loss_type or "auto").lower().strip()
        if loss_type in {"focal", "focalloss"}:
            criterion = FocalLoss(alpha=class_weights, gamma=float(config.focal_gamma))
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(config.label_smoothing))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    else:
        if primary_metric_name == "Spearman":
            criterion = SpearmanLoss(regularization=config.spearman_reg)
        elif primary_metric_name == "MAE":
            criterion = SmoothL1LossWrapper(beta=1.0)
        else:
            criterion = nn.MSELoss()
        sched_mode = "max" if primary_metric_name == "Spearman" else "min"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=sched_mode, factor=0.5, patience=5)

    best_score = float("inf") if primary_metric_name in {"MAE"} else -1.0
    patience_counter = 0
    best_path = out_dir / "best_model.pth"
    history = []

    for epoch in range(config.max_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, modality_mask=modality_mask)
        if task_type == "classification":
            valid_metrics = evaluate_classification(model, valid_loader, device, modality_mask=modality_mask)
            if primary_metric_name == "AUPRC":
                scheduler.step(valid_metrics["auprc"] if not np.isnan(valid_metrics["auprc"]) else 0.0)
            else:
                scheduler.step(valid_metrics["auc"] if not np.isnan(valid_metrics["auc"]) else 0.0)
        else:
            valid_metrics = evaluate_regression(model, valid_loader, device, modality_mask=modality_mask)
            if primary_metric_name == "Spearman":
                v = float(valid_metrics.get("spearman", float("nan")))
                scheduler.step(v if not np.isnan(v) else 0.0)
            elif primary_metric_name == "RMSE":
                v = float(valid_metrics.get("rmse", float("nan")))
                scheduler.step(v if not np.isnan(v) else 0.0)
            else:
                v = float(valid_metrics.get("mae", float("nan")))
                scheduler.step(v if not np.isnan(v) else 0.0)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "valid_acc": float(valid_metrics.get("acc", float("nan"))),
                "valid_auc": float(valid_metrics.get("auc", float("nan"))),
                "valid_auprc": float(valid_metrics.get("auprc", float("nan"))),
                "valid_mae": float(valid_metrics.get("mae", float("nan"))),
                "valid_rmse": float(valid_metrics.get("rmse", float("nan"))),
                "valid_spearman": float(valid_metrics.get("spearman", float("nan"))),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if primary_metric_name == "MAE":
            current = float(valid_metrics.get("mae", float("nan")))
            improved = (not np.isnan(current)) and (current < best_score)
        elif primary_metric_name == "RMSE":
            current = float(valid_metrics.get("rmse", float("nan")))
            improved = (not np.isnan(current)) and (current < best_score)
        elif primary_metric_name == "Spearman":
            current = float(valid_metrics.get("spearman", float("nan")))
            improved = (not np.isnan(current)) and (current > best_score)
        elif primary_metric_name == "AUPRC":
            current = float(valid_metrics.get("auprc", float("nan")))
            improved = (not np.isnan(current)) and (current > best_score)
        else:
            current = float(valid_metrics.get("auc", float("nan")))
            improved = (not np.isnan(current)) and (current > best_score)

        if improved:
            best_score = float(current)
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= config.max_patience:
                break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    if task_type == "classification":
        test_metrics = evaluate_classification(model, test_loader, device, modality_mask=modality_mask)
    else:
        test_metrics = evaluate_regression(
            model,
            test_loader,
            device,
            label_mean=label_mean,
            label_std=label_std,
            modality_mask=modality_mask,
        )

    if primary_metric_name == "AUPRC":
        primary_metric = float(test_metrics.get("auprc", float("nan")))
    elif primary_metric_name == "MAE":
        primary_metric = float(test_metrics.get("mae", float("nan")))
    elif primary_metric_name == "RMSE":
        primary_metric = float(test_metrics.get("rmse", float("nan")))
    elif primary_metric_name == "Spearman":
        primary_metric = float(test_metrics.get("spearman", float("nan")))
    else:
        primary_metric = float(test_metrics.get("auc", float("nan")))

    best_valid_primary = float(best_score) if best_epoch > 0 else float("nan")

    if task_type == "classification":
        lt = str(config.loss_type or "auto").lower().strip()
        loss_type = "FocalLoss" if lt in {"focal", "focalloss"} else "CrossEntropyLoss"
    else:
        if primary_metric_name == "Spearman":
            loss_type = "SpearmanLoss"
        elif primary_metric_name == "MAE":
            loss_type = "SmoothL1Loss"
        else:
            loss_type = "MSELoss"

    meta = {
        "task": task_name,
        "task_type": task_type,
        "device": str(device),
        "seed": config.seed,
        "dims": {"chemberta": int(emb_tr_s.shape[1]), "unimol": int(emb_tr_3d.shape[1]), "kpgt": int(emb_tr_g.shape[1])},
        "config": asdict(config),
        "modalities": modalities,
        "best_epoch": int(best_epoch),
        "primary_metric_name": primary_metric_name,
        "primary_metric": primary_metric,
        "best_valid_primary": best_valid_primary,
        "loss_type": loss_type,
        "test_acc": float(test_metrics.get("acc", float("nan"))),
        "test_auc": float(test_metrics.get("auc", float("nan"))),
        "test_auprc": float(test_metrics.get("auprc", float("nan"))),
        "test_mae": float(test_metrics.get("mae", float("nan"))),
        "test_rmse": float(test_metrics.get("rmse", float("nan"))),
        "test_spearman": float(test_metrics.get("spearman", float("nan"))),
    }
    if init_msg is not None:
        meta["init_from_status"] = init_msg
    if task_type == "regression":
        meta["label_mean"] = float(label_mean) if label_mean is not None else float("nan")
        meta["label_std"] = float(label_std) if label_std is not None else float("nan")

    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2))
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    return meta


__all__ = ["TrainConfig", "fit_on_task"]

