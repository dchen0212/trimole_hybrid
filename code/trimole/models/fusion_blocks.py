import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDynamicGatedFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        pair_hidden_dim: int | None = None,
        dropout: float = 0.1,
        pair_scale: float = 0.5,
        residual_scale: float = 0.5,
    ):
        super().__init__()
        pair_hidden_dim = pair_hidden_dim or hidden_dim

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, pair_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.pair_scale = pair_scale
        self.residual_scale = residual_scale

    def forward(self, h1: torch.Tensor, h2: torch.Tensor, h3: torch.Tensor):
        gate_in = torch.cat([h1, h2, h3], dim=-1)
        gate_logits = self.gate(gate_in)
        weights = F.softmax(gate_logits, dim=-1)

        h_gate = (
            weights[:, 0:1] * h1 +
            weights[:, 1:2] * h2 +
            weights[:, 2:3] * h3
        )

        pair12 = h1 * h2
        pair13 = h1 * h3
        pair23 = h2 * h3
        pair_in = torch.cat([pair12, pair13, pair23], dim=-1)
        h_pair = self.pair_mlp(pair_in)

        h_res = (h1 + h2 + h3) / 3.0

        h_final = h_gate + self.pair_scale * h_pair + self.residual_scale * h_res
        h_final = self.norm(h_final)

        return h_final, weights

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDynamicGatedFusion3DDownweight(nn.Module):
    """
    轻量版 3D-downweight fusion
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        init_3d_bias: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        in_dim = hidden_dim * 3

        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.weight_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.modality_3d_penalty = nn.Parameter(torch.tensor(float(init_3d_bias)))

    def forward(
        self,
        h_smiles: torch.Tensor,
        h_graph: torch.Tensor,
        h_3d: torch.Tensor,
        return_aux: bool = False,
    ):
        x = torch.cat([h_smiles, h_graph, h_3d], dim=-1)

        g_3d = torch.sigmoid(self.gate_mlp(x))
        h_3d_tilde = g_3d * h_3d

        x2 = torch.cat([h_smiles, h_graph, h_3d_tilde], dim=-1)

        logits = self.weight_mlp(x2)
        penalty = F.softplus(self.modality_3d_penalty)
        logits[..., 2] = logits[..., 2] - penalty

        weights = torch.softmax(logits, dim=-1)

        fused = (
            weights[..., 0:1] * h_smiles
            + weights[..., 1:2] * h_graph
            + weights[..., 2:3] * h_3d_tilde
        )

        residual = (h_smiles + h_graph + h_3d_tilde) / 3.0
        fused = fused + self.out_proj(residual)

        if not return_aux:
            return fused, weights

        aux = {
            "g_3d": g_3d,
            "weights": weights,
            "penalty_3d": penalty.detach(),
            "h_3d_norm": h_3d.norm(dim=-1).mean().detach(),
            "h_3d_tilde_norm": h_3d_tilde.norm(dim=-1).mean().detach(),
        }
        return fused, weights, aux
