from __future__ import annotations

import torch
import torch.nn as nn

from trimole.models.fusion_blocks import ResidualDynamicGatedFusion, ResidualDynamicGatedFusion3DDownweight


class MultiModalFusionMLP(nn.Module):
    def __init__(
        self,
        dim_smiles: int,
        dim_3d: int,
        dim_graph: int,
        out_dim: int = 2,
        hidden_dim: int = 128,
        dropout_proj: float = 0.2,
        dropout_head: float = 0.3,
        fusion_type: str = "mlp",  # mlp | gated | residual_dynamic | gated_3d_downweight
        task_context_dim: int = 0,
    ):
        super().__init__()
        self.fusion_type = str(fusion_type).lower().strip()
        self.task_context_dim = max(int(task_context_dim), 0)
        self.latest_fusion_aux = None
        self.dynamic_alpha = nn.Parameter(torch.tensor(1.0))
        self.dynamic_alpha_scale = 0.0
        self.multi_head_types = {
            "dual_head",
            "task_conditional_dual_head",
            "tri_head",
            "task_conditional_tri_head",
            "task_conditional",
        }
        self.tri_head_types = {"tri_head", "task_conditional_tri_head"}
        self.task_conditioned_types = {"task_conditional_dual_head", "task_conditional_tri_head", "task_conditional"}
        self.enable_dual_head = self.fusion_type in self.multi_head_types
        self.enable_tri_head = self.fusion_type in self.tri_head_types
        self.enable_task_condition = self.fusion_type in self.task_conditioned_types
        self.aux_head_names = ["mlp", "gated"] + (["residual"] if self.enable_tri_head else [])

        self.proj_smiles = nn.Sequential(
            nn.Linear(dim_smiles, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_proj),
        )
        self.proj_3d = nn.Sequential(
            nn.Linear(dim_3d, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_proj),
        )
        self.proj_graph = nn.Sequential(
            nn.Linear(dim_graph, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_proj),
        )

        # Original gated branch
        if self.fusion_type in {"gated", "gate", "gated_concat"} or self.enable_dual_head:
            gate_in_dim = hidden_dim * 3 + (self.task_context_dim if self.enable_task_condition else 0)
            self.gate = nn.Sequential(
                nn.Linear(gate_in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_head),
                nn.Linear(hidden_dim, 3),
            )
        else:
            self.gate = None

        # Residual dynamic refinement branch
        if self.fusion_type in {"gated_3d_downweight", "gated3d", "gated_3d"}:
            self.residual_dynamic_fusion = ResidualDynamicGatedFusion3DDownweight(
                hidden_dim=hidden_dim,
                dropout=dropout_head,
                init_3d_bias=0.5,
            )
        elif self.fusion_type in {"residual_dynamic", "rdg", "resdyn"} or self.enable_tri_head:

            self.residual_dynamic_fusion = ResidualDynamicGatedFusion(
                hidden_dim=hidden_dim,
                pair_hidden_dim=hidden_dim,
                dropout=dropout_head,
                pair_scale=0.3,
                residual_scale=0.3,
            )
            self.dynamic_alpha_scale = 0.0 if self.enable_tri_head else 0.2
        else:
            self.residual_dynamic_fusion = None
            self.dynamic_alpha_scale = 0.0

        # Keep original concat -> MLP backbone
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_head),
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_head),
            nn.Linear(hidden_dim // 2, int(out_dim)),
        )
        if self.enable_dual_head:
            self.gated_fusion_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_head),
            )
            self.predictor_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_head),
                nn.Linear(hidden_dim // 2, int(out_dim)),
            )
            self.predictor_gated = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_head),
                nn.Linear(hidden_dim // 2, int(out_dim)),
            )
            if self.enable_tri_head:
                self.predictor_residual = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout_head),
                    nn.Linear(hidden_dim // 2, int(out_dim)),
                )
            else:
                self.predictor_residual = None
            mixer_in_dim = hidden_dim * (3 if self.enable_tri_head else 2) + (self.task_context_dim if self.enable_task_condition else 0)
            self.head_mixer = nn.Sequential(
                nn.Linear(mixer_in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_head),
                nn.Linear(hidden_dim, 3 if self.enable_tri_head else 1),
            )
        else:
            self.gated_fusion_mlp = None
            self.predictor_mlp = None
            self.predictor_gated = None
            self.predictor_residual = None
            self.head_mixer = None

    def _expand_task_context(self, batch_size: int, device: torch.device, task_context: torch.Tensor | None) -> torch.Tensor | None:
        if not self.enable_task_condition or self.task_context_dim <= 0:
            return None
        if task_context is None:
            return torch.zeros(batch_size, self.task_context_dim, device=device)
        if task_context.ndim == 1:
            task_context = task_context.unsqueeze(0).expand(batch_size, -1)
        if task_context.shape[0] != batch_size:
            task_context = task_context.expand(batch_size, -1)
        return task_context.to(device=device, dtype=torch.float32)

    def forward(
        self,
        emb_smiles: torch.Tensor,
        emb_3d: torch.Tensor,
        emb_graph: torch.Tensor,
        modality_mask: tuple[bool, bool, bool] | None = None,
        task_context: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        h_text = self.proj_smiles(emb_smiles)
        h_graph = self.proj_graph(emb_graph)
        h_3d = self.proj_3d(emb_3d)

        use_smiles, use_graph, use_3d = (True, True, True) if modality_mask is None else tuple(bool(x) for x in modality_mask)

        if not (use_smiles or use_graph or use_3d):
            raise ValueError("modality_mask disables all modalities; expected at least one modality to be enabled.")

        if not use_smiles:
            h_text = torch.zeros_like(h_text)
        if not use_graph:
            h_graph = torch.zeros_like(h_graph)
        if not use_3d:
            h_3d = torch.zeros_like(h_3d)

        base_text = h_text
        base_graph = h_graph
        base_3d = h_3d
        task_ctx = self._expand_task_context(batch_size=base_text.shape[0], device=base_text.device, task_context=task_context)

        if self.enable_dual_head:
            plain_in = torch.cat([base_text, base_graph, base_3d], dim=1)
            fused_plain = self.fusion_mlp(plain_in)

            gate_in = torch.cat([base_text, base_graph, base_3d], dim=1)
            if task_ctx is not None:
                gate_in = torch.cat([gate_in, task_ctx], dim=1)
            gate_logits = self.gate(gate_in)
            if not use_smiles:
                gate_logits[:, 0] = -1e9
            if not use_graph:
                gate_logits[:, 1] = -1e9
            if not use_3d:
                gate_logits[:, 2] = -1e9
            gate = torch.softmax(gate_logits, dim=1)

            gated_text = base_text * gate[:, 0:1]
            gated_graph = base_graph * gate[:, 1:2]
            gated_3d = base_3d * gate[:, 2:3]
            fused_gated = self.gated_fusion_mlp(torch.cat([gated_text, gated_graph, gated_3d], dim=1))

            mlp_logits = self.predictor_mlp(fused_plain)
            gated_logits = self.predictor_gated(fused_gated)

            residual_logits = None
            mixer_parts = [fused_plain, fused_gated]
            if self.enable_tri_head:
                fused_residual, _ = self.residual_dynamic_fusion(base_text, base_graph, base_3d)
                residual_logits = self.predictor_residual(fused_residual)
                mixer_parts.append(fused_residual)

            mixer_in = torch.cat(mixer_parts, dim=1)
            if task_ctx is not None:
                mixer_in = torch.cat([mixer_in, task_ctx], dim=1)
            if self.enable_tri_head:
                mix_weights = torch.softmax(self.head_mixer(mixer_in), dim=1)
                logits = (
                    mix_weights[:, 0:1] * mlp_logits
                    + mix_weights[:, 1:2] * gated_logits
                    + mix_weights[:, 2:3] * residual_logits
                )
            else:
                mix_alpha = torch.sigmoid(self.head_mixer(mixer_in))
                mix_weights = torch.cat([1.0 - mix_alpha, mix_alpha], dim=1)
                logits = (1.0 - mix_alpha) * mlp_logits + mix_alpha * gated_logits

            self.latest_fusion_aux = {
                "mlp_logits": mlp_logits,
                "gated_logits": gated_logits,
                "mix_alpha": mix_weights[:, 1:2] if not self.enable_tri_head else None,
                "mix_weights": mix_weights,
                "gate": gate,
            }
            if residual_logits is not None:
                self.latest_fusion_aux["residual_logits"] = residual_logits
            if return_aux:
                out = {
                    "logits": logits,
                    "mlp_logits": mlp_logits,
                    "gated_logits": gated_logits,
                    "mix_alpha": mix_weights[:, 1:2] if not self.enable_tri_head else None,
                    "mix_weights": mix_weights,
                    "gate": gate,
                }
                if residual_logits is not None:
                    out["residual_logits"] = residual_logits
                return out
            return logits

        # Original gated weighting if selected
        if self.gate is not None:
            fused_for_gate = torch.cat([h_text, h_graph, h_3d], dim=1)
            if task_ctx is not None:
                fused_for_gate = torch.cat([fused_for_gate, task_ctx], dim=1)
            gate_logits = self.gate(fused_for_gate)

            if not use_smiles:
                gate_logits[:, 0] = -1e9
            if not use_graph:
                gate_logits[:, 1] = -1e9
            if not use_3d:
                gate_logits[:, 2] = -1e9

            gate = torch.softmax(gate_logits, dim=1)
            h_text = h_text * gate[:, 0:1]
            h_graph = h_graph * gate[:, 1:2]
            h_3d = h_3d * gate[:, 2:3]

        # Original strong backbone
        fused_in = torch.cat([h_text, h_graph, h_3d], dim=1)
        fused_base = self.fusion_mlp(fused_in)

        # Conservative residual enhancement
        if self.residual_dynamic_fusion is not None:
            fused_dyn, _weights = self.residual_dynamic_fusion(h_text, h_graph, h_3d)
            alpha = self.dynamic_alpha_scale * torch.sigmoid(self.dynamic_alpha)
            fused = fused_base + alpha * fused_dyn
        else:
            fused = fused_base

        logits = self.predictor(fused)
        self.latest_fusion_aux = None
        return logits


__all__ = ["MultiModalFusionMLP"]
