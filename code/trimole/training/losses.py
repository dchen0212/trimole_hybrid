"""Custom loss functions for task-specific optimization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in classification tasks.

    Particularly useful for AUPRC-focused tasks where minority class performance
    is critical. The focusing parameter gamma down-weights easy examples and
    focuses training on hard negatives.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class AUPRCSurrogateLoss(nn.Module):
    """
    Practical AUPRC-oriented surrogate.

    This is not a literal SOAP implementation. It combines FocalLoss with a
    pairwise ranking term that pushes positive examples above negatives, with
    extra weight on hard negatives. The intent is to better align training with
    AUPRC-focused leaderboard tasks while staying stable in the existing stack.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        rank_weight: float = 0.35,
        margin: float = 0.5,
        hard_negative_temperature: float = 1.0,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.rank_weight = float(rank_weight)
        self.margin = float(margin)
        self.hard_negative_temperature = float(hard_negative_temperature)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.view(-1).long()
        base_loss = self.focal(logits, targets)

        if logits.ndim == 2 and logits.shape[1] == 2:
            score = logits[:, 1] - logits[:, 0]
        else:
            score = logits.view(-1)

        pos_mask = targets == 1
        neg_mask = targets == 0
        if int(pos_mask.sum()) == 0 or int(neg_mask.sum()) == 0:
            return base_loss

        pos_scores = score[pos_mask]
        neg_scores = score[neg_mask]

        pair_margin = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        rank_loss = F.softplus(self.margin - pair_margin)

        temp = max(self.hard_negative_temperature, 1e-6)
        with torch.no_grad():
            neg_weights = torch.softmax(neg_scores / temp, dim=0)
        rank_loss = (rank_loss * neg_weights.unsqueeze(0)).mean()

        return (1.0 - self.rank_weight) * base_loss + self.rank_weight * rank_loss


class AUROCMarginLoss(nn.Module):
    """
    Practical AUROC-oriented surrogate.

    Combines a weighted classification term with a pairwise ranking objective
    over positive-negative pairs. Unlike the AUPRC surrogate, it uses uniform
    pair weighting because AUROC does not privilege early retrieval in the same
    way AUPRC does.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        margin: float = 0.5,
        rank_weight: float = 0.25,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.margin = float(margin)
        self.rank_weight = float(rank_weight)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.view(-1).long()
        base_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="mean",
            label_smoothing=self.label_smoothing,
        )

        if logits.ndim == 2 and logits.shape[1] == 2:
            score = logits[:, 1] - logits[:, 0]
        else:
            score = logits.view(-1)

        pos_mask = targets == 1
        neg_mask = targets == 0
        if int(pos_mask.sum()) == 0 or int(neg_mask.sum()) == 0:
            return base_loss

        pos_scores = score[pos_mask]
        neg_scores = score[neg_mask]
        pair_margin = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        rank_loss = F.softplus(self.margin - pair_margin).mean()
        return (1.0 - self.rank_weight) * base_loss + self.rank_weight * rank_loss


class SpearmanLoss(nn.Module):
    """
    Differentiable approximation to Spearman correlation loss for regression tasks.

    Spearman correlation measures the monotonic relationship between predictions
    and targets based on their ranks. This loss uses a soft ranking approximation
    to make the ranking operation differentiable.

    Combines ranking loss with MSE regularization for training stability.
    """

    def __init__(self, regularization: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.reg = regularization
        self.temperature = temperature

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)

        if pred.shape[0] < 2:
            return F.mse_loss(pred, target)

        pred_rank = self._soft_rank(pred)
        target_rank = self._soft_rank(target)

        pred_centered = pred_rank - pred_rank.mean()
        target_centered = target_rank - target_rank.mean()

        pred_norm = pred_centered.norm()
        target_norm = target_centered.norm()

        if pred_norm < 1e-8 or target_norm < 1e-8:
            corr = torch.tensor(0.0, device=pred.device)
        else:
            corr = (pred_centered * target_centered).sum() / (pred_norm * target_norm)

        mse = F.mse_loss(pred, target)
        return -corr + self.reg * mse

    def _soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        x_expand = x.unsqueeze(1).expand(-1, n)
        diff = x_expand - x_expand.t()
        scale = 10.0 / self.temperature
        return torch.sigmoid(diff * scale).sum(dim=1)


class RankingMSELoss(nn.Module):
    """
    Combined loss that optimizes both absolute values (MSE) and relative ordering.
    """

    def __init__(self, rank_weight: float = 0.5, margin: float = 0.1):
        super().__init__()
        self.rank_weight = rank_weight
        self.margin = margin

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)

        mse_loss = F.mse_loss(pred, target)

        if pred.shape[0] < 2:
            return mse_loss

        n = pred.shape[0]
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
        target_diff = target.unsqueeze(1) - target.unsqueeze(0)
        target_sign = torch.sign(target_diff)
        rank_loss = F.relu(-target_sign * pred_diff + self.margin)
        mask = torch.triu(torch.ones(n, n, device=pred.device), diagonal=1)
        rank_loss = (rank_loss * mask).sum() / (mask.sum() + 1e-8)

        return (1 - self.rank_weight) * mse_loss + self.rank_weight * rank_loss


class SmoothL1LossWrapper(nn.Module):
    """
    Wrapper for SmoothL1Loss that handles shape mismatches.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        return F.smooth_l1_loss(pred, target, beta=self.beta)
