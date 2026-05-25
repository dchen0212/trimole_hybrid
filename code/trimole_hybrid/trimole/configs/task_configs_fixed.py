"""
Fixed configurations for problematic small datasets.

Based on deep analysis in results/analysis/poor_performance/ANALYSIS_REPORT.md
These configs address early stopping, overfitting, and poor generalization issues.
"""

# ==============================================================================
# Small Dataset Configuration (for datasets with <500 training samples)
# ==============================================================================

SMALL_DATASET_BASE = {
    "hidden_dim": 128,
    "batch_size": 32,  # Reduced from 64 for better gradient estimation
    "lr": 1e-4,  # Reduced from 3e-4 to prevent immediate overfitting
    "max_epochs": 100,  # Increased from 60 for more training opportunity
    "max_patience": 20,  # Increased from 12 for more patience
    "weight_decay": 0.05,  # Increased from 0.01 for stronger regularization
    "dropout_proj": 0.1,  # Reduced from 0.2 to preserve more information
    "dropout_head": 0.15,  # Reduced from 0.3 to preserve more information
    "task_type": "auto",
    "modalities": "all",
    "focal_gamma": 0.0,  # Disabled - use simple CrossEntropy instead
    "label_smoothing": 0.0,  # Disabled - conflicts with focal loss
    "spearman_reg": 0.1,
}

# ==============================================================================
# Problem Dataset Specific Configurations
# ==============================================================================

FIXED_CONFIGS = {
    # cyp2c9_substrate_carbonmangels: 468 samples, 21% positive
    # Issue: AUPRC 0.269 vs baseline 0.474, best_epoch=1, test_acc=0.201
    "cyp2c9_substrate_carbonmangels": {
        **SMALL_DATASET_BASE,
        "primary_metric_name": "AUPRC",
        # Use class weights instead of focal loss
        # exact from current split: pos_ratio=0.2073 => (1-pos)/pos ≈ 3.82
        "pos_weight": 3.82,
        "loss_type": "weighted_ce",  # Weighted CrossEntropy
        # Extra conservative: reduce capacity + stronger regularization
        "hidden_dim": 64,
        "lr": 5e-5,  # Even lower learning rate
        "dropout_proj": 0.1,
        "dropout_head": 0.2,
        "weight_decay": 0.12,
    },
    
    # cyp2d6_substrate_carbonmangels: 467 samples, 30% positive
    # Issue: AUPRC 0.535 vs baseline 0.736, best_epoch=2, test_acc=0.489
    "cyp2d6_substrate_carbonmangels": {
        **SMALL_DATASET_BASE,
        "primary_metric_name": "AUPRC",
        # exact from current split: pos_ratio=0.2955 => (1-pos)/pos ≈ 2.38
        "pos_weight": 2.38,
        "loss_type": "weighted_ce",
        # Conservative: reduce capacity + stronger regularization to reduce valid-test gap
        "hidden_dim": 64,
        "lr": 5e-5,
        "dropout_proj": 0.1,
        "dropout_head": 0.2,
        "weight_decay": 0.12,
    },
    
    # bioavailability_ma: 448 samples, 78% positive (reverse imbalance!)
    # Issue: AUROC 0.728 vs baseline 0.942
    "bioavailability_ma": {
        **SMALL_DATASET_BASE,
        "primary_metric_name": "AUROC",
        # Reverse imbalance (pos_ratio ~0.783): rely on default inverse-frequency weights (more stable than a hard neg_weight).
        "loss_type": "weighted_ce",
        "hidden_dim": 64,
        "label_smoothing": 0.05,
        "lr": 8e-5,
        "dropout_proj": 0.1,
        "dropout_head": 0.2,
        "weight_decay": 0.1,
    },
}

# ==============================================================================
# Alternative Experimental Configurations
# ==============================================================================

# Experiment 1: Try mild focal loss with proper settings
EXPERIMENT_1_FOCAL = {
    "cyp2c9_substrate_carbonmangels": {
        **SMALL_DATASET_BASE,
        "primary_metric_name": "AUPRC",
        "focal_gamma": 1.0,  # Much milder than 2.5
        "label_smoothing": 0.0,  # No smoothing with focal
        "lr": 5e-5,
        "loss_type": "focal",
    },
    "cyp2d6_substrate_carbonmangels": {
        **SMALL_DATASET_BASE,
        "primary_metric_name": "AUPRC",
        "focal_gamma": 1.0,
        "label_smoothing": 0.0,
        "lr": 5e-5,
        "loss_type": "focal",
    },
}

# Experiment 2: Very conservative settings with heavy regularization
EXPERIMENT_2_CONSERVATIVE = {
    "cyp2c9_substrate_carbonmangels": {
        **SMALL_DATASET_BASE,
        "primary_metric_name": "AUPRC",
        "lr": 3e-5,  # Very low
        "batch_size": 16,  # Very small
        "dropout_proj": 0.0,  # No dropout in projection
        "dropout_head": 0.05,  # Minimal dropout in head
        "weight_decay": 0.1,  # Strong L2 regularization
        "pos_weight": 3.76,
        "loss_type": "weighted_ce",
    },
}

# ==============================================================================
# Helper Functions
# ==============================================================================

def is_small_dataset(task_name: str, train_size: int) -> bool:
    """
    Determine if a dataset should use small dataset configuration.
    
    Args:
        task_name: Name of the task
        train_size: Number of training samples
        
    Returns:
        True if should use small dataset config
    """
    # Explicit problem datasets
    if task_name in FIXED_CONFIGS:
        return True
    
    # Any dataset with <500 training samples
    if train_size < 500:
        return True
    
    return False

def get_fixed_config(task_name: str, experiment: str = "default"):
    """
    Get the fixed configuration for a problematic task.
    
    Args:
        task_name: Name of the task
        experiment: Which experiment config to use
            - "default": Use FIXED_CONFIGS (recommended)
            - "focal": Use mild focal loss
            - "conservative": Very conservative settings
    
    Returns:
        Configuration dictionary
    """
    if experiment == "focal":
        return EXPERIMENT_1_FOCAL.get(task_name)
    elif experiment == "conservative":
        return EXPERIMENT_2_CONSERVATIVE.get(task_name)
    else:
        return FIXED_CONFIGS.get(task_name)

def apply_small_dataset_fixes(task_configs: dict, task_sizes: dict):
    """
    Apply small dataset fixes to task configurations.
    
    Args:
        task_configs: Original task configuration dictionary
        task_sizes: Dictionary mapping task names to training sizes
        
    Returns:
        Updated task configuration dictionary
    """
    updated_configs = task_configs.copy()
    
    for task_name, train_size in task_sizes.items():
        if is_small_dataset(task_name, train_size):
            # Get fixed config if available, otherwise use base small dataset config
            fixed_config = get_fixed_config(task_name)
            if fixed_config:
                print(f"[CONFIG] Applying fixed config for {task_name} (train_size={train_size})")
                updated_configs[task_name] = fixed_config
            else:
                print(f"[CONFIG] Applying small dataset base config for {task_name} (train_size={train_size})")
                # Merge with existing config, preferring small dataset settings
                existing = updated_configs.get(task_name, {})
                updated_configs[task_name] = {**existing, **SMALL_DATASET_BASE}
    
    return updated_configs

# ==============================================================================
# Usage Example
# ==============================================================================

if __name__ == "__main__":
    # Example: Check if a task needs fixes
    task_sizes = {
        "cyp2c9_substrate_carbonmangels": 468,
        "bioavailability_ma": 448,
        "cyp2d6_substrate_carbonmangels": 467,
        "ames": 5094,
    }
    
    for task_name, size in task_sizes.items():
        if is_small_dataset(task_name, size):
            config = get_fixed_config(task_name)
            print(f"\n{task_name} (size={size}): NEEDS FIXES")
            print(f"  Learning rate: {config.get('lr', 'N/A')}")
            print(f"  Focal gamma: {config.get('focal_gamma', 'N/A')}")
            print(f"  Dropout: proj={config.get('dropout_proj', 'N/A')}, head={config.get('dropout_head', 'N/A')}")
            print(f"  Weight: pos={config.get('pos_weight', 'N/A')}, neg={config.get('neg_weight', 'N/A')}")
        else:
            print(f"\n{task_name} (size={size}): OK (uses standard config)")
