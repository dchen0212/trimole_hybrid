"""
Adaptive configuration system based on comprehensive analysis.

This module implements intelligent configuration selection based on:
1. Dataset size (small/medium/large)
2. Class imbalance severity
3. Task type (classification/regression)
4. Primary metric (AUROC/AUPRC/MAE/Spearman)

Based on analysis report: results/analysis/comprehensive/综合分析报告.md

Key findings:
- Focal Loss has 100% failure rate (5/5 tasks failed)
- Small datasets (<500 samples) fail with aggressive config (4/5 failed)
- Early stopping (best_epoch <= 3) indicates config problems
- One-size-fits-all approach leads to only 14% success rate

Solution:
- Disable Focal Loss, use weighted CrossEntropy instead
- Conservative config for small datasets
- Adaptive learning rate and regularization based on dataset size
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np


# =============================================================================
# Base Configurations by Dataset Size
# =============================================================================

LARGE_DATASET_CONFIG = {
    "description": "For datasets with >2000 training samples",
    "hidden_dim": 256,
    "batch_size": 128,
    "lr": 3e-4,
    "dropout_proj": 0.15,
    "dropout_head": 0.2,
    "weight_decay": 0.01,
    "max_epochs": 60,
    "max_patience": 12,
    "focal_gamma": 0.0,  # Disabled - Focal Loss failed in all cases
    "label_smoothing": 0.05,
}

MEDIUM_DATASET_CONFIG = {
    "description": "For datasets with 500-2000 training samples",
    "hidden_dim": 128,
    "batch_size": 64,
    "lr": 1e-4,
    "dropout_proj": 0.1,
    "dropout_head": 0.15,
    "weight_decay": 0.03,
    "max_epochs": 80,
    "max_patience": 15,
    "focal_gamma": 0.0,  # Disabled
    "label_smoothing": 0.0,
}

SMALL_DATASET_CONFIG = {
    "description": "For datasets with <500 training samples",
    "hidden_dim": 128,
    "batch_size": 32,
    "lr": 5e-5,  # 6x lower than large datasets
    "dropout_proj": 0.05,  # 4x lower to preserve information
    "dropout_head": 0.1,  # 3x lower
    "weight_decay": 0.08,  # 8x higher for regularization
    "max_epochs": 100,
    "max_patience": 20,
    "focal_gamma": 0.0,  # Disabled - especially harmful on small datasets
    "label_smoothing": 0.0,  # Disabled
}


# =============================================================================
# Extreme Imbalance Handling
# =============================================================================

def compute_class_weight(pos_ratio: float, for_negative_class: bool = False) -> float:
    """
    Compute class weight for imbalanced datasets.
    
    Args:
        pos_ratio: Proportion of positive samples (0-1)
        for_negative_class: If True, compute weight for negative class
        
    Returns:
        Class weight
    """
    if for_negative_class:
        # For reverse imbalance (>75% positive), weight negative class more
        return pos_ratio / (1 - pos_ratio)
    else:
        # For normal imbalance (<25% positive), weight positive class more
        return (1 - pos_ratio) / pos_ratio


# =============================================================================
# Task-Specific Configuration Overrides
# =============================================================================

TASK_SPECIFIC_OVERRIDES = {
    # Failed tasks - Priority 1 fixes
    "cyp2c9_substrate_carbonmangels": {
        "pos_ratio": 0.21,
        "use_pos_weight": True,
        "note": "Small dataset + severe imbalance + early stop at epoch 1"
    },
    "cyp2d6_substrate_carbonmangels": {
        "pos_ratio": 0.30,
        "use_pos_weight": True,
        "note": "Small dataset + imbalance + early stop at epoch 2"
    },
    "bioavailability_ma": {
        "pos_ratio": 0.78,
        "use_neg_weight": True,  # Reverse imbalance!
        "note": "Small dataset + reverse imbalance (78% positive)"
    },
    
    # Poor performing tasks - Priority 2
    "cyp3a4_veith": {
        "pos_ratio": 0.41,
        "note": "Failed with Focal Loss, needs weighted CE"
    },
    "cyp2c9_veith": {
        "pos_ratio": 0.33,
        "note": "Failed with Focal Loss, needs weighted CE"
    },
    "cyp2d6_veith": {
        "pos_ratio": 0.19,
        "note": "Failed with Focal Loss + severe imbalance"
    },
    "cyp3a4_substrate_carbonmangels": {
        "pos_ratio": 0.53,
        "note": "Small dataset + early stop at epoch 3"
    },
    "dili": {
        "pos_ratio": 0.48,
        "note": "Small dataset + early stop at epoch 4"
    },
    
    # Reverse imbalance tasks
    "hia_hou": {
        "pos_ratio": 0.87,
        "use_neg_weight": True,
        "note": "Extreme reverse imbalance"
    },
    "bbb_martins": {
        "pos_ratio": 0.77,
        "use_neg_weight": True,
        "note": "Reverse imbalance + early stop at epoch 2"
    },
    "herg": {
        "pos_ratio": 0.68,
        "note": "Moderate reverse imbalance"
    },
}


# =============================================================================
# Adaptive Configuration Selection
# =============================================================================

def get_adaptive_config(
    task_name: str,
    train_size: int,
    task_type: str = "auto",
    primary_metric: str = "auto",
    pos_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Get adaptive configuration based on task characteristics.
    
    Args:
        task_name: Name of the task
        train_size: Number of training samples
        task_type: Task type (classification/regression)
        primary_metric: Primary evaluation metric
        pos_ratio: Positive class ratio for classification (0-1)
        
    Returns:
        Configuration dictionary
    """
    # Step 1: Select base config by dataset size
    if train_size < 500:
        config = SMALL_DATASET_CONFIG.copy()
        size_category = "small"
    elif train_size < 2000:
        config = MEDIUM_DATASET_CONFIG.copy()
        size_category = "medium"
    else:
        config = LARGE_DATASET_CONFIG.copy()
        size_category = "large"
    
    # Step 2: Set task type and metric
    config["task_type"] = task_type
    config["primary_metric_name"] = primary_metric
    
    # Step 3: Apply task-specific overrides
    if task_name in TASK_SPECIFIC_OVERRIDES:
        override = TASK_SPECIFIC_OVERRIDES[task_name]
        
        # Get pos_ratio from override if not provided
        if pos_ratio is None and "pos_ratio" in override:
            pos_ratio = override["pos_ratio"]
        
        # Apply class weights for imbalanced tasks
        if override.get("use_pos_weight") and pos_ratio is not None:
            config["pos_weight"] = compute_class_weight(pos_ratio, for_negative_class=False)
            config["loss_type"] = "weighted_ce"
        
        if override.get("use_neg_weight") and pos_ratio is not None:
            config["neg_weight"] = compute_class_weight(pos_ratio, for_negative_class=True)
            config["loss_type"] = "weighted_ce"
    
    # Step 4: Handle extreme imbalance (even if not in specific overrides)
    if pos_ratio is not None:
        if pos_ratio < 0.25:
            # Severe minority positive class
            if "pos_weight" not in config:
                config["pos_weight"] = compute_class_weight(pos_ratio, for_negative_class=False)
                config["loss_type"] = "weighted_ce"
        elif pos_ratio > 0.75:
            # Severe minority negative class (reverse imbalance)
            if "neg_weight" not in config:
                config["neg_weight"] = compute_class_weight(pos_ratio, for_negative_class=True)
                config["loss_type"] = "weighted_ce"
    
    # Step 5: Force disable Focal Loss
    config["focal_gamma"] = 0.0
    
    # Step 6: Add metadata
    config["_metadata"] = {
        "task_name": task_name,
        "train_size": train_size,
        "size_category": size_category,
        "pos_ratio": pos_ratio,
        "adaptive_config_version": "1.0",
    }
    
    return config


if __name__ == "__main__":
    # Example: Problem tasks from analysis
    problem_tasks = {
        "cyp2c9_substrate_carbonmangels": {
            "train_size": 468,
            "task_type": "classification",
            "primary_metric": "AUPRC",
            "pos_ratio": 0.21,
        },
        "bioavailability_ma": {
            "train_size": 448,
            "task_type": "classification",
            "primary_metric": "AUROC",
            "pos_ratio": 0.78,
        },
        "ames": {
            "train_size": 5094,
            "task_type": "classification",
            "primary_metric": "AUROC",
            "pos_ratio": 0.54,
        },
    }
    
    print("=" * 80)
    print("ADAPTIVE CONFIGURATION EXAMPLES")
    print("=" * 80)
    
    for task_name, chars in problem_tasks.items():
        config = get_adaptive_config(**chars, task_name=task_name)
        print(f"\n{task_name}:")
        print(f"  Size: {chars['train_size']} samples")
        print(f"  LR: {config.get('lr')}")
        print(f"  Batch: {config.get('batch_size')}")
        print(f"  Focal: {config.get('focal_gamma')}")
