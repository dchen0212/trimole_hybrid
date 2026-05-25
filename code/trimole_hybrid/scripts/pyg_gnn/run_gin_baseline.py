#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

TASK_METRIC = {
    "bioavailability_ma": "AUROC",
    "bbb_martins": "AUROC",
    "cyp2c9_veith": "AUPRC",
    "cyp2d6_veith": "AUPRC",
    "cyp3a4_veith": "AUPRC",
    "pgp_broccatelli": "AUROC",
    "herg": "AUROC",
}

ATOM_LIST = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
HYBRID_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

def one_hot_with_unknown(x, choices):
    out = [0] * (len(choices) + 1)
    try:
        idx = choices.index(x)
    except ValueError:
        idx = len(choices)
    out[idx] = 1
    return out

def atom_features(atom: Chem.Atom) -> List[float]:
    feats = []
    feats += one_hot_with_unknown(atom.GetAtomicNum(), ATOM_LIST)
    feats += one_hot_with_unknown(atom.GetHybridization(), HYBRID_LIST)
    feats += one_hot_with_unknown(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
    feats += one_hot_with_unknown(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    feats += one_hot_with_unknown(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    feats += [int(atom.GetIsAromatic())]
    feats += [float(atom.GetMass()) / 200.0]
    return feats

def bond_features(bond: Chem.Bond) -> List[float]:
    feats = []
    feats += one_hot_with_unknown(bond.GetBondType(), BOND_LIST)
    feats += [int(bond.GetIsConjugated())]
    feats += [int(bond.IsInRing())]
    return feats

def smiles_to_data(smiles: str, y: float) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edges = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edges.append([i, j])
        edges.append([j, i])
        edge_attrs.append(bf)
        edge_attrs.append(bf)

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    if edge_attr.dim() != 2 or edge_attr.size(-1) != 7:
        raise ValueError(f"edge_attr shape mismatch for {smiles}: {tuple(edge_attr.shape)}")

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([float(y)], dtype=torch.float),
    )
    return data

def detect_cols(df: pd.DataFrame):
    cols_lower = {c.lower(): c for c in df.columns}
    smiles_col = cols_lower.get("smiles") or cols_lower.get("drug")
    y_col = cols_lower.get("label") or cols_lower.get("y") or cols_lower.get("target")
    if smiles_col is None or y_col is None:
        raise KeyError(f"Cannot detect smiles/label columns: {list(df.columns)}")
    return smiles_col, y_col

def load_dataset(csv_path: Path) -> List[Data]:
    df = pd.read_csv(csv_path)
    smiles_col, y_col = detect_cols(df)
    out = []
    bad = 0
    for smi, y in zip(df[smiles_col].astype(str), df[y_col].values):
        data = smiles_to_data(smi, y)
        if data is None:
            bad += 1
            continue
        out.append(data)
    if not out:
        raise ValueError(f"No valid molecules loaded from {csv_path}")
    if bad:
        print(f"[WARN] {csv_path.name}: skipped {bad} invalid SMILES")
    return out

class GINGraphClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        last_dim = in_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(last_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            last_dim = hidden_dim

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        g = global_add_pool(x, batch)
        logits = self.head(g).view(-1)
        return logits

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        prob = torch.sigmoid(logits)
        ys.append(batch.y.view(-1).cpu().numpy())
        ps.append(prob.cpu().numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    return y_true, y_prob

def eval_cls(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "test_auc": float(roc_auc_score(y_true, y_prob)),
        "test_auprc": float(average_precision_score(y_true, y_prob)),
        "test_acc": float(accuracy_score(y_true, y_pred)),
    }

def fit_one_task(task_dir: Path, task: str, seed: int, epochs: int, batch_size: int, lr: float, hidden_dim: int, num_layers: int, dropout: float):
    metric_name = TASK_METRIC[task]

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = load_dataset(task_dir / "train.csv")
    valid_ds = load_dataset(task_dir / "valid.csv")
    test_ds  = load_dataset(task_dir / "test.csv")

    in_dim = train_ds[0].x.size(-1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GINGraphClassifier(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_valid_primary = -1.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * batch.num_graphs
            total_n += batch.num_graphs

        y_val, p_val = predict(model, valid_loader, device)
        valid_auc = roc_auc_score(y_val, p_val)
        valid_auprc = average_precision_score(y_val, p_val)
        valid_primary = valid_auc if metric_name == "AUROC" else valid_auprc

        if valid_primary > best_valid_primary:
            best_valid_primary = float(valid_primary)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)

    y_test, p_test = predict(model, test_loader, device)
    row = {
        "task": task,
        "task_type": "classification",
        "primary_metric_name": metric_name,
        "best_valid_primary": float(best_valid_primary),
        "best_epoch": int(best_epoch),
        "loss_type": "GIN",
        "seed": seed,
    }
    row.update(eval_cls(y_test, p_test))
    row["primary_metric"] = row["test_auc"] if metric_name == "AUROC" else row["test_auprc"]
    return row, y_test, p_test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / "run_gin_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = {}

    for task in args.tasks:
        try:
            row, y_true, y_pred = fit_one_task(
                data_root / task, task,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
            pd.DataFrame({
                "task": task,
                "y_true": y_true,
                "y_pred": y_pred,
            }).to_csv(run_dir / f"{task}_test_predictions.csv", index=False)

            rows.append(row)
            print(f"[{task}] {row['primary_metric_name']}={row['primary_metric']:.6f}")
        except Exception as e:
            errors[task] = str(e)
            print(f"[{task}] FAILED: {e}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(run_dir / "results_all.csv", index=False)
        print(f"\nDone. Summary: {run_dir / 'results_all.csv'}")

    if errors:
        (run_dir / "errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False))
        print(f"Failures: {len(errors)} (see {run_dir / 'errors.json'})")

if __name__ == "__main__":
    main()
