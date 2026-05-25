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
        """
        Args:
            alpha: Class weights tensor. If None, uses uniform weights.
            gamma: Focusing parameter. Higher values focus more on hard examples.
                   gamma=0 is equivalent to CrossEntropyLoss.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits of shape [B, C] where C is num_classes
            targets: Ground truth labels of shape [B]
        
        Returns:
            Scalar focal loss
        """
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class SpearmanLoss(nn.Module):
    """
    Differentiable approximation to Spearman correlation loss for regression tasks.
    
    Spearman correlation measures the monotonic relationship between predictions
    and targets based on their ranks. This loss uses a soft ranking approximation
    to make the ranking operation differentiable.
    
    Combines ranking loss with MSE regularization for training stability.
    """
    
    def __init__(self, regularization: float = 0.1, temperature: float = 1.0):
        """
        Args:
            regularization: Weight for MSE regularization term (for stability)
            temperature: Temperature for soft ranking (higher = softer ranking)
        """
        super().__init__()
        self.reg = regularization
        self.temperature = temperature
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape [B] or [B, 1]
            target: Targets of shape [B] or [B, 1]
        
        Returns:
            Scalar loss (negative correlation + MSE regularization)
        """
        # Ensure 1D tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        if pred.shape[0] < 2:
            # Cannot compute correlation with fewer than 2 samples
            return F.mse_loss(pred, target)
        
        # Compute soft ranks
        pred_rank = self._soft_rank(pred)
        target_rank = self._soft_rank(target)
        
        # Center the ranks
        pred_centered = pred_rank - pred_rank.mean()
        target_centered = target_rank - target_rank.mean()
        
        # Compute Pearson correlation on ranks (= Spearman)
        pred_norm = pred_centered.norm()
        target_norm = target_centered.norm()
        
        if pred_norm < 1e-8 or target_norm < 1e-8:
            # Degenerate case: constant predictions or targets
            corr = torch.tensor(0.0, device=pred.device)
        else:
            corr = (pred_centered * target_centered).sum() / (pred_norm * target_norm)
        
        # MSE for stability
        mse = F.mse_loss(pred, target)
        
        # Negative correlation (we want to maximize correlation, so minimize -corr)
        return -corr + self.reg * mse
    
    def _soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable soft ranks using sigmoid.
        
        For each element, count how many other elements are smaller than it
        using a soft comparison (sigmoid of difference).
        
        Args:
            x: Input tensor of shape [N]
        
        Returns:
            Soft ranks of shape [N]
        """
        n = x.shape[0]
        # Expand for pairwise comparison: [N, N]
        x_expand = x.unsqueeze(1).expand(-1, n)
        # diff[i,j] = x[i] - x[j]
        diff = x_expand - x_expand.t()
        # Soft count of elements smaller than x[i]
        # sigmoid(diff * scale) approaches 1 if x[i] > x[j], 0 otherwise
        scale = 10.0 / self.temperature
        return torch.sigmoid(diff * scale).sum(dim=1)


class RankingMSELoss(nn.Module):
    """
    Combined loss that optimizes both absolute values (MSE) and relative ordering.
    
    Useful for tasks where both the actual predictions and their ranking matter.
    """
    
    def __init__(self, rank_weight: float = 0.5, margin: float = 0.1):
        """
        Args:
            rank_weight: Weight for ranking loss component (0 to 1)
            margin: Margin for ranking pairs
        """
        super().__init__()
        self.rank_weight = rank_weight
        self.margin = margin
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        
        # MSE loss
        mse_loss = F.mse_loss(pred, target)
        
        if pred.shape[0] < 2:
            return mse_loss
        
        # Pairwise ranking loss
        n = pred.shape[0]
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)  # [N, N]
        target_diff = target.unsqueeze(1) - target.unsqueeze(0)  # [N, N]
        
        # We want: if target[i] > target[j], then pred[i] > pred[j]
        # Sign of target_diff indicates desired ordering
        target_sign = torch.sign(target_diff)
        
        # Margin ranking loss: max(0, -sign(t_i - t_j) * (p_i - p_j) + margin)
        rank_loss = F.relu(-target_sign * pred_diff + self.margin)
        
        # Only consider upper triangle (avoid double counting)
        mask = torch.triu(torch.ones(n, n, device=pred.device), diagonal=1)
        rank_loss = (rank_loss * mask).sum() / (mask.sum() + 1e-8)
        
        return (1 - self.rank_weight) * mse_loss + self.rank_weight * rank_loss


class SmoothL1LossWrapper(nn.Module):
    """
    Wrapper for SmoothL1Loss that handles shape mismatches.
    
    SmoothL1Loss (Huber Loss) is less sensitive to outliers than MSE,
    making it suitable for MAE-focused tasks.
    """
    
    def __init__(self, beta: float = 1.0):
        """
        Args:
            beta: Threshold at which to change from L2 to L1 loss
        """
        super().__init__()
        self.beta = beta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        return F.smooth_l1_loss(pred, target, beta=self.beta)
