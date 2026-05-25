#!/usr/bin/env python3
"""
Deep analysis of datasets with poor performance.
Investigates cyp2c9_substrate_carbonmangels, bioavailability_ma, and cyp2d6_substrate_carbonmangels.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tdc.single_pred import ADME

def load_task_data(task_name, data_root):
    """Load dataset using TDC"""
    print(f"\n{'='*80}")
    print(f"Loading dataset: {task_name}")
    print(f"{'='*80}")
    
    # Get dataset
    data = ADME(name=task_name.upper())
    split = data.get_split()
    
    # Combine all splits to see overall distribution
    train_df = split['train']
    valid_df = split['valid']
    test_df = split['test']
    
    all_data = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(all_data)}")
    print(f"  Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    # Check label distribution
    if 'Y' in all_data.columns:
        label_counts = all_data['Y'].value_counts()
        print(f"\nLabel Distribution (overall):")
        for label, count in sorted(label_counts.items()):
            pct = count / len(all_data) * 100
            print(f"    Label {label}: {count} ({pct:.1f}%)")
        
        # Check each split
        for split_name, split_df in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
            if len(split_df) > 0:
                split_counts = split_df['Y'].value_counts()
                pos_ratio = split_counts.get(1, 0) / len(split_df)
                print(f"  {split_name}: pos_ratio={pos_ratio:.3f}")
    
    return {
        'train': train_df,
        'valid': valid_df,
        'test': test_df,
        'all': all_data
    }

def analyze_model_predictions(task_name, run_dir):
    """Analyze model predictions and performance"""
    print(f"\n{'='*80}")
    print(f"Model Analysis: {task_name}")
    print(f"{'='*80}")
    
    task_dir = run_dir / task_name
    
    # Load metadata
    meta_file = task_dir / 'meta.json'
    if not meta_file.exists():
        print(f"ERROR: meta.json not found at {meta_file}")
        return None
    
    with open(meta_file) as f:
        meta = json.load(f)
    
    print(f"\nModel Performance:")
    print(f"  Task Type: {meta['task_type']}")
    print(f"  Primary Metric: {meta['primary_metric_name']} = {meta['primary_metric']:.4f}")
    print(f"  Best Epoch: {meta['best_epoch']}")
    print(f"  Best Valid {meta['primary_metric_name']}: {meta['best_valid_primary']:.4f}")
    print(f"  Loss Type: {meta['loss_type']}")
    
    if meta['task_type'] == 'classification':
        print(f"  Test Accuracy: {meta.get('test_acc', 'N/A'):.4f}")
        print(f"  Test AUROC: {meta.get('test_auc', 'N/A'):.4f}")
        print(f"  Test AUPRC: {meta.get('test_auprc', 'N/A'):.4f}")
    
    # Load training history
    history_file = task_dir / 'history.json'
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        
        print(f"\nTraining History:")
        print(f"  Total Epochs: {len(history)}")
        
        # Check for overfitting
        best_epoch_data = history[meta['best_epoch'] - 1]
        last_epoch_data = history[-1]
        
        print(f"\n  Best Epoch ({meta['best_epoch']}):")
        print(f"    Train Loss: {best_epoch_data['train_loss']:.4f}")
        metric_key = f"valid_{meta['primary_metric_name'].lower()}"
        print(f"    Valid {meta['primary_metric_name']}: {best_epoch_data[metric_key]:.4f}")
        
        print(f"\n  Last Epoch ({len(history)}):")
        print(f"    Train Loss: {last_epoch_data['train_loss']:.4f}")
        
        # Plot training curve
        plot_training_curves(history, meta, task_name)
        
        # Check for early convergence or instability
        valid_metric_key = f"valid_{meta['primary_metric_name'].lower()}"
        valid_metrics = [epoch[valid_metric_key] for epoch in history if valid_metric_key in epoch and not pd.isna(epoch[valid_metric_key])]
        
        if len(valid_metrics) > 3:
            # Check variance in last few epochs
            last_n = min(5, len(valid_metrics))
            recent_var = np.var(valid_metrics[-last_n:])
            overall_var = np.var(valid_metrics)
            
            print(f"\n  Stability Analysis:")
            print(f"    Overall variance: {overall_var:.6f}")
            print(f"    Recent variance (last {last_n} epochs): {recent_var:.6f}")
            
            # Check if stuck at early stage
            if meta['best_epoch'] <= 3:
                print(f"  ⚠️  WARNING: Best epoch is very early ({meta['best_epoch']}), possible issues:")
                print(f"      - Model might be overfitting immediately")
                print(f"      - Learning rate might be too high")
                print(f"      - Data might be too noisy")
    
    return meta

def plot_training_curves(history, meta, task_name):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    
    # Plot loss
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss')
    axes[0].axvline(x=meta['best_epoch'], color='r', linestyle='--', label=f'Best Epoch ({meta["best_epoch"]})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{task_name} - Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot validation metric
    metric_name = meta['primary_metric_name'].lower()
    valid_metric_key = f'valid_{metric_name}'
    valid_metrics = [h.get(valid_metric_key, np.nan) for h in history]
    valid_metrics = [m if not pd.isna(m) else None for m in valid_metrics]
    
    axes[1].plot(epochs, valid_metrics, 'g-', label=f'Valid {meta["primary_metric_name"]}')
    axes[1].axvline(x=meta['best_epoch'], color='r', linestyle='--', label=f'Best Epoch')
    axes[1].axhline(y=meta['best_valid_primary'], color='orange', linestyle=':', label=f'Best Valid ({meta["best_valid_primary"]:.4f})')
    axes[1].axhline(y=meta['primary_metric'], color='purple', linestyle=':', label=f'Test ({meta["primary_metric"]:.4f})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel(meta['primary_metric_name'])
    axes[1].set_title(f'{task_name} - Validation Performance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = PROJECT_ROOT / 'results' / 'analysis' / 'poor_performance'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{task_name}_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved training curves to: {output_dir / f'{task_name}_training_curves.png'}")
    plt.close()

def check_embedding_quality(task_name, data_root):
    """Check if embeddings exist and are reasonable"""
    print(f"\n{'='*80}")
    print(f"Embedding Quality Check: {task_name}")
    print(f"{'='*80}")
    
    emb_dir = data_root / 'data_new' / task_name
    
    for emb_type in ['chemberta', 'unimol', 'kpgt']:
        emb_file = emb_dir / f'{emb_type}_embeddings.pt'
        
        if emb_file.exists():
            try:
                embeddings = torch.load(emb_file, map_location='cpu')
                
                # Check if it's a dict or tensor
                if isinstance(embeddings, dict):
                    for split_name, emb_tensor in embeddings.items():
                        if isinstance(emb_tensor, torch.Tensor):
                            mean_val = emb_tensor.mean().item()
                            std_val = emb_tensor.std().item()
                            has_nan = torch.isnan(emb_tensor).any().item()
                            has_inf = torch.isinf(emb_tensor).any().item()
                            
                            print(f"\n  {emb_type} ({split_name}): shape={emb_tensor.shape}")
                            print(f"    Mean: {mean_val:.4f}, Std: {std_val:.4f}")
                            print(f"    Has NaN: {has_nan}, Has Inf: {has_inf}")
                            
                            if has_nan or has_inf:
                                print(f"    ⚠️  WARNING: Embeddings contain NaN or Inf values!")
                            
                            if abs(mean_val) < 1e-6 and std_val < 1e-6:
                                print(f"    ⚠️  WARNING: Embeddings are nearly zero - possible generation failure!")
                
                elif isinstance(embeddings, torch.Tensor):
                    mean_val = embeddings.mean().item()
                    std_val = embeddings.std().item()
                    has_nan = torch.isnan(embeddings).any().item()
                    has_inf = torch.isinf(embeddings).any().item()
                    
                    print(f"\n  {emb_type}: shape={embeddings.shape}")
                    print(f"    Mean: {mean_val:.4f}, Std: {std_val:.4f}")
                    print(f"    Has NaN: {has_nan}, Has Inf: {has_inf}")
                    
                    if has_nan or has_inf:
                        print(f"    ⚠️  WARNING: Embeddings contain NaN or Inf values!")
                    
                    if abs(mean_val) < 1e-6 and std_val < 1e-6:
                        print(f"    ⚠️  WARNING: Embeddings are nearly zero - possible generation failure!")
            
            except Exception as e:
                print(f"  ⚠️  ERROR loading {emb_type}: {e}")
        else:
            print(f"  ⚠️  {emb_type}: File not found at {emb_file}")

def main():
    """Main analysis function"""
    print("="*80)
    print("DEEP ANALYSIS OF POOR-PERFORMING DATASETS")
    print("="*80)
    
    # Configuration
    run_dir = PROJECT_ROOT / 'results' / 'model_log' / 'run_20260123_2200'
    data_root = PROJECT_ROOT
    
    # Problem datasets
    problem_tasks = [
        'cyp2c9_substrate_carbonmangels',  # AUPRC 0.269 vs baseline 0.474
        'bioavailability_ma',               # AUROC 0.728 vs baseline 0.942
        'cyp2d6_substrate_carbonmangels',  # AUPRC 0.535 vs baseline 0.736
    ]
    
    results = {}
    
    for task in problem_tasks:
        try:
            # Load dataset
            data_splits = load_task_data(task, data_root)
            
            # Analyze model performance
            meta = analyze_model_predictions(task, run_dir)
            
            # Check embedding quality
            check_embedding_quality(task, data_root)
            
            results[task] = {
                'data': data_splits,
                'meta': meta
            }
            
        except Exception as e:
            print(f"\n⚠️  ERROR analyzing {task}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("SUMMARY REPORT")
    print(f"{'='*80}")
    
    for task, result in results.items():
        if result['meta'] is not None:
            meta = result['meta']
            print(f"\n{task}:")
            print(f"  Best Epoch: {meta['best_epoch']}")
            print(f"  Test {meta['primary_metric_name']}: {meta['primary_metric']:.4f}")
            print(f"  Valid {meta['primary_metric_name']}: {meta['best_valid_primary']:.4f}")
            print(f"  Gap (Test - Valid): {meta['primary_metric'] - meta['best_valid_primary']:.4f}")
            
            # Diagnose issues
            print(f"\n  Potential Issues:")
            if meta['best_epoch'] <= 3:
                print(f"    ❌ Very early stopping - possible overfitting or bad hyperparameters")
            if abs(meta['primary_metric'] - meta['best_valid_primary']) > 0.15:
                print(f"    ❌ Large test-valid gap - possible overfitting or distribution shift")
            if meta['primary_metric'] < 0.6:
                print(f"    ❌ Very poor performance - model might be predicting random or constant")
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
