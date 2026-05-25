"""
Task-specific hyperparameter configurations.

Tasks are grouped by dataset size and characteristics:
- Group A: Large datasets (N > 3000) - can use larger models
- Group B: Medium datasets (1000 <= N < 3000) - balanced settings
- Group C: Small datasets (500 <= N < 1000) - need regularization
- Group D: Very small datasets (N < 500) - strong regularization required
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


# =============================================================================
# Group Configurations
# =============================================================================

GROUP_A: Dict[str, Any] = {
    "group": "A",
    "hidden_dim": 256,
    "batch_size": 128,
    "lr": 3e-4,
    "dropout_proj": 0.2,
    "dropout_head": 0.3,
    "weight_decay": 0.01,
    "max_epochs": 60,
    "max_patience": 12,
}

GROUP_B: Dict[str, Any] = {
    "group": "B",
    "hidden_dim": 128,
    "batch_size": 64,
    "lr": 2e-4,
    "dropout_proj": 0.25,
    "dropout_head": 0.35,
    "weight_decay": 0.02,
    "max_epochs": 80,
    "max_patience": 15,
}

GROUP_C: Dict[str, Any] = {
    "group": "C",
    "hidden_dim": 64,
    "batch_size": 32,
    "lr": 1e-4,
    "dropout_proj": 0.35,
    "dropout_head": 0.4,
    "weight_decay": 0.05,
    "max_epochs": 100,
    "max_patience": 20,
}

GROUP_D: Dict[str, Any] = {
    "group": "D",
    "hidden_dim": 64,
    "batch_size": 16,
    "lr": 5e-5,
    "dropout_proj": 0.4,
    "dropout_head": 0.5,
    "weight_decay": 0.1,
    "max_epochs": 150,
    "max_patience": 25,
}


# =============================================================================
# Task-to-Group Mapping with Task-Specific Overrides
# =============================================================================

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # Group A: Large datasets (N > 3000)
    # -------------------------------------------------------------------------
    
    # cyp2d6_veith: 9191 samples, cls, AUPRC, imbalanced (19% pos)
    "cyp2d6_veith": {
        **GROUP_A,
        "primary_metric_name": "AUPRC",
        "focal_gamma": 2.5,  # Higher gamma for severe imbalance
    },
    
    # cyp3a4_veith: 8629 samples, cls, AUPRC, moderate balance (41% pos)
    "cyp3a4_veith": {
        **GROUP_A,
        "primary_metric_name": "AUPRC",
        "focal_gamma": 2.0,
    },
    
    # cyp2c9_veith: 8465 samples, cls, AUPRC, imbalanced (33% pos)
    "cyp2c9_veith": {
        **GROUP_A,
        "primary_metric_name": "AUPRC",
        "focal_gamma": 2.0,
    },
    
    # solubility_aqsoldb: 6986 samples, reg, MAE
    "solubility_aqsoldb": {
        **GROUP_A,
        "primary_metric_name": "MAE",
    },
    
    # ld50_zhu: 5170 samples, reg, MAE
    "ld50_zhu": {
        **GROUP_A,
        "primary_metric_name": "MAE",
    },
    
    # ames: 5094 samples, cls, AUROC, balanced (54% pos)
    "ames": {
        **GROUP_A,
        "primary_metric_name": "AUROC",
    },
    
    # lipophilicity_astrazeneca: 2940 samples, reg, MAE
    # Note: On the boundary of Group A/B, using Group A for consistency
    "lipophilicity_astrazeneca": {
        **GROUP_A,
        "primary_metric_name": "MAE",
        # Slightly smaller capacity since it's on the smaller end
        "hidden_dim": 192,
    },
    
    # -------------------------------------------------------------------------
    # Group B: Medium datasets (1000 <= N < 3000)
    # -------------------------------------------------------------------------
    
    # bbb_martins: 1421 samples, cls, AUROC, imbalanced (77% pos)
    "bbb_martins": {
        **GROUP_B,
        "primary_metric_name": "AUROC",
    },
    
    # ppbr_az: 1130 samples, reg, MAE
    "ppbr_az": {
        **GROUP_B,
        "primary_metric_name": "MAE",
    },
    
    # -------------------------------------------------------------------------
    # Group C: Small datasets (500 <= N < 1000)
    # -------------------------------------------------------------------------
    
    # pgp_broccatelli: 852 samples, cls, AUROC, moderate (43% pos)
    "pgp_broccatelli": {
        **GROUP_C,
        "primary_metric_name": "AUROC",
    },
    
    # clearance_hepatocyte_az: 849 samples, reg, Spearman
    "clearance_hepatocyte_az": {
        **GROUP_C,
        "primary_metric_name": "Spearman",
    },
    
    # vdss_lombardo: 791 samples, reg, Spearman
    "vdss_lombardo": {
        **GROUP_C,
        "primary_metric_name": "Spearman",
    },
    
    # clearance_microsome_az: 772 samples, reg, Spearman
    "clearance_microsome_az": {
        **GROUP_C,
        "primary_metric_name": "Spearman",
    },
    
    # caco2_wang: 637 samples, reg, MAE
    "caco2_wang": {
        **GROUP_C,
        "primary_metric_name": "MAE",
    },
    
    # -------------------------------------------------------------------------
    # Group D: Very small datasets (N < 500)
    # -------------------------------------------------------------------------
    
    # cyp2c9_substrate_carbonmangels: 468 samples, cls, AUPRC, imbalanced (21% pos)
    "cyp2c9_substrate_carbonmangels": {
        **GROUP_D,
        "primary_metric_name": "AUPRC",
        "focal_gamma": 2.5,  # Stronger focal for imbalance
    },
    
    # cyp2d6_substrate_carbonmangels: 467 samples, cls, AUPRC, imbalanced (30% pos)
    "cyp2d6_substrate_carbonmangels": {
        **GROUP_D,
        "primary_metric_name": "AUPRC",
        "focal_gamma": 2.5,
    },
    
    # cyp3a4_substrate_carbonmangels: 469 samples, cls, AUROC, balanced (53% pos)
    "cyp3a4_substrate_carbonmangels": {
        **GROUP_D,
        "primary_metric_name": "AUROC",
    },
    
    # half_life_obach: 467 samples, reg, Spearman
    "half_life_obach": {
        **GROUP_D,
        "primary_metric_name": "Spearman",
    },
    
    # herg: 458 samples, cls, AUROC, imbalanced (68% pos)
    "herg": {
        **GROUP_D,
        "primary_metric_name": "AUROC",
    },
    
    # bioavailability_ma: 448 samples, cls, AUROC, imbalanced (78% pos)
    "bioavailability_ma": {
        **GROUP_D,
        "primary_metric_name": "AUROC",
    },
    
    # hia_hou: 404 samples, cls, AUROC, severely imbalanced (87% pos)
    "hia_hou": {
        **GROUP_D,
        "primary_metric_name": "AUROC",
        "focal_gamma": 3.0,  # Strong focal for severe imbalance
    },
    
    # dili: 332 samples, cls, AUROC, balanced (48% pos)
    "dili": {
        **GROUP_D,
        "primary_metric_name": "AUROC",
    },
}


# =============================================================================
# Special Configurations for Imbalanced Tasks
# =============================================================================

IMBALANCED_TASK_SETTINGS: Dict[str, Dict[str, Any]] = {
    # Tasks with severe class imbalance that need special handling
    "hia_hou": {"focal_gamma": 3.0, "pos_ratio": 0.87},
    "bioavailability_ma": {"focal_gamma": 2.5, "pos_ratio": 0.78},
    "bbb_martins": {"focal_gamma": 2.0, "pos_ratio": 0.77},
    "herg": {"focal_gamma": 2.0, "pos_ratio": 0.68},
    "cyp2d6_veith": {"focal_gamma": 2.5, "pos_ratio": 0.19},
    "cyp2c9_substrate_carbonmangels": {"focal_gamma": 2.5, "pos_ratio": 0.21},
    "cyp2d6_substrate_carbonmangels": {"focal_gamma": 2.5, "pos_ratio": 0.30},
    "cyp2c9_veith": {"focal_gamma": 2.0, "pos_ratio": 0.33},
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_task_config(task_name: str) -> Dict[str, Any]:
    """
    Get the configuration for a specific task.
    
    Args:
        task_name: Name of the task (e.g., 'ames', 'bioavailability_ma')
    
    Returns:
        Dictionary with hyperparameters for the task.
        Falls back to GROUP_B (medium) settings if task is unknown.
    """
    if task_name in TASK_CONFIGS:
        return TASK_CONFIGS[task_name].copy()
    
    # Default to medium configuration
    return {**GROUP_B, "primary_metric_name": "auto"}


def get_focal_gamma(task_name: str, default: float = 2.0) -> float:
    """
    Get the focal loss gamma for a specific task.
    
    Args:
        task_name: Name of the task
        default: Default gamma if not specified
    
    Returns:
        Focal loss gamma parameter
    """
    config = TASK_CONFIGS.get(task_name, {})
    return config.get("focal_gamma", default)


def get_group_for_dataset_size(n_train: int) -> Dict[str, Any]:
    """
    Determine appropriate group based on training set size.
    
    Args:
        n_train: Number of training samples
    
    Returns:
        Configuration dictionary for the appropriate group
    """
    if n_train >= 3000:
        return GROUP_A.copy()
    elif n_train >= 1000:
        return GROUP_B.copy()
    elif n_train >= 500:
        return GROUP_C.copy()
    else:
        return GROUP_D.copy()


def list_tasks_by_group() -> Dict[str, list]:
    """
    Get a mapping of group names to task lists.
    
    Returns:
        Dictionary mapping group names to lists of task names
    """
    groups: Dict[str, list] = {"A": [], "B": [], "C": [], "D": []}
    
    for task_name, config in TASK_CONFIGS.items():
        group = config.get("group", "B")
        groups[group].append(task_name)
    
    return groups


# Print summary when module is run directly
if __name__ == "__main__":
    groups = list_tasks_by_group()
    print("Task Configuration Summary")
    print("=" * 60)
    
    for group_name in ["A", "B", "C", "D"]:
        tasks = groups[group_name]
        print(f"\nGroup {group_name} ({len(tasks)} tasks):")
        for task in tasks:
            config = TASK_CONFIGS[task]
            metric = config.get("primary_metric_name", "auto")
            gamma = config.get("focal_gamma", "-")
            print(f"  - {task}: metric={metric}, focal_gamma={gamma}")
