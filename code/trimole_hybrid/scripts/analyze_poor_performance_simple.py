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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_csv_data(task_name, data_root):
    """Load dataset from CSV files"""
    print(f"\n{'='*80}")
    print(f"Loading dataset: {task_name}")
    print(f"{'='*80}")
    
    task_dir = data_root / 'data' / 'data_new' / task_name
    
    # Load CSV files
    train_df = pd.read_csv(task_dir / 'train.csv')
    valid_df = pd.read_csv(task_dir / 'valid.csv')
    test_df = pd.read_csv(task_dir / 'test.csv')
    
    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(all_df)}")
    print(f"  Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    # Check label distribution
    if 'label' in all_df.columns:
        print(f"\nLabel Distribution (overall):")
        label_counts = all_df['label'].value_counts()
        for label in sorted(label_counts.index):
            count = label_counts[label]
            pct = count / len(all_df) * 100
            print(f"    Label {label}: {count:4d} ({pct:5.1f}%)")
        
        # Check each split
        for split_name, split_df in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
            if len(split_df) > 0 and 'label' in split_df.columns:
                split_counts = split_df['label'].value_counts()
                pos_count = split_counts.get(1, 0)
                pos_ratio = pos_count / len(split_df)
                neg_count = split_counts.get(0, 0)
                print(f"  {split_name:5s}: pos={pos_count:3d} ({pos_ratio:.3f}), neg={neg_count:3d}")
    
    return {
        'train': train_df,
        'valid': valid_df,
        'test': test_df,
        'all': all_df
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
        print(f"  Test Accuracy: {meta.get('test_acc', 0):.4f}")
        print(f"  Test AUROC: {meta.get('test_auc', 0):.4f}")
        print(f"  Test AUPRC: {meta.get('test_auprc', 0):.4f}")
    
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
        if metric_key in best_epoch_data:
            print(f"    Valid {meta['primary_metric_name']}: {best_epoch_data[metric_key]:.4f}")
        
        print(f"\n  Last Epoch ({len(history)}):")
        print(f"    Train Loss: {last_epoch_data['train_loss']:.4f}")
        
        # Plot training curve
        plot_training_curves(history, meta, task_name)
        
        # Check for early convergence or instability
        valid_metric_key = f"valid_{meta['primary_metric_name'].lower()}"
        valid_metrics = []
        for epoch in history:
            if valid_metric_key in epoch:
                val = epoch[valid_metric_key]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    valid_metrics.append(val)
        
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
                print(f"      - Class imbalance handling might be too aggressive")
    
    return meta

def plot_training_curves(history, meta, task_name):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    
    # Plot loss
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0].axvline(x=meta['best_epoch'], color='r', linestyle='--', linewidth=2, label=f'Best Epoch ({meta["best_epoch"]})')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{task_name} - Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot validation metric
    metric_name = meta['primary_metric_name'].lower()
    valid_metric_key = f'valid_{metric_name}'
    valid_metrics = []
    valid_epochs = []
    
    for h in history:
        if valid_metric_key in h:
            val = h.get(valid_metric_key)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                valid_metrics.append(val)
                valid_epochs.append(h['epoch'])
    
    if valid_metrics:
        axes[1].plot(valid_epochs, valid_metrics, 'g-', label=f'Valid {meta["primary_metric_name"]}', linewidth=2)
        axes[1].axvline(x=meta['best_epoch'], color='r', linestyle='--', linewidth=2, label=f'Best Epoch')
        axes[1].axhline(y=meta['best_valid_primary'], color='orange', linestyle=':', linewidth=2, label=f'Best Valid ({meta["best_valid_primary"]:.4f})')
        axes[1].axhline(y=meta['primary_metric'], color='purple', linestyle=':', linewidth=2, label=f'Test ({meta["primary_metric"]:.4f})')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel(meta['primary_metric_name'], fontsize=12)
        axes[1].set_title(f'{task_name} - Validation Performance', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = PROJECT_ROOT / 'results' / 'analysis' / 'poor_performance'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{task_name}_training_curves.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved training curves to: {output_file}")
    plt.close()

def check_embedding_quality(task_name, data_root):
    """Check if embeddings exist and are reasonable"""
    print(f"\n{'='*80}")
    print(f"Embedding Quality Check: {task_name}")
    print(f"{'='*80}")
    
    emb_dir = data_root / 'data' / 'data_new' / task_name / 'embeddings'
    
    for emb_type in ['chemberta', 'unimol', 'kpgt']:
        emb_file = emb_dir / f'{emb_type}.npy'
        
        if emb_file.exists():
            try:
                embeddings = np.load(emb_file)
                
                mean_val = embeddings.mean()
                std_val = embeddings.std()
                has_nan = np.isnan(embeddings).any()
                has_inf = np.isinf(embeddings).any()
                
                print(f"\n  {emb_type}: shape={embeddings.shape}")
                print(f"    Mean: {mean_val:.4f}, Std: {std_val:.4f}")
                print(f"    Min: {embeddings.min():.4f}, Max: {embeddings.max():.4f}")
                print(f"    Has NaN: {has_nan}, Has Inf: {has_inf}")
                
                if has_nan or has_inf:
                    print(f"    ⚠️  WARNING: Embeddings contain NaN or Inf values!")
                
                if abs(mean_val) < 1e-6 and std_val < 1e-6:
                    print(f"    ⚠️  WARNING: Embeddings are nearly zero - possible generation failure!")
                
                # Check variance per feature
                feature_stds = embeddings.std(axis=0)
                zero_var_features = (feature_stds < 1e-6).sum()
                if zero_var_features > embeddings.shape[1] * 0.1:  # More than 10% features have zero variance
                    print(f"    ⚠️  WARNING: {zero_var_features}/{embeddings.shape[1]} features have near-zero variance!")
            
            except Exception as e:
                print(f"  ⚠️  ERROR loading {emb_type}: {e}")
        else:
            print(f"  ⚠️  {emb_type}: File not found at {emb_file}")

def compare_with_good_task(problem_task, good_task, run_dir, data_root):
    """Compare a problem task with a well-performing task"""
    print(f"\n{'='*80}")
    print(f"Comparison: {problem_task} vs {good_task}")
    print(f"{'='*80}")
    
    # Load both datasets
    problem_data = load_csv_data(problem_task, data_root)
    good_data = load_csv_data(good_task, data_root)
    
    # Load both model results
    problem_meta = json.load(open(run_dir / problem_task / 'meta.json'))
    good_meta = json.load(open(run_dir / good_task / 'meta.json'))
    
    print(f"\nSize Comparison:")
    print(f"  {problem_task}: {len(problem_data['all'])} samples")
    print(f"  {good_task}: {len(good_data['all'])} samples")
    
    print(f"\nPerformance Comparison:")
    print(f"  {problem_task}: {problem_meta['primary_metric_name']}={problem_meta['primary_metric']:.4f}, best_epoch={problem_meta['best_epoch']}")
    print(f"  {good_task}: {good_meta['primary_metric_name']}={good_meta['primary_metric']:.4f}, best_epoch={good_meta['best_epoch']}")

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
        ('cyp2c9_substrate_carbonmangels', 'AUPRC 0.269 vs baseline 0.474'),
        ('bioavailability_ma', 'AUROC 0.728 vs baseline 0.942'),
        ('cyp2d6_substrate_carbonmangels', 'AUPRC 0.535 vs baseline 0.736'),
    ]
    
    # Good performing task for comparison
    good_task = 'ames'  # AUROC 0.904, beats baseline
    
    results = {}
    
    for task, desc in problem_tasks:
        print(f"\n\n{'#'*80}")
        print(f"# ANALYZING: {task}")
        print(f"# {desc}")
        print(f"{'#'*80}")
        
        try:
            # Load dataset
            data_splits = load_csv_data(task, data_root)
            
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
    
    # Compare with good task
    print(f"\n\n{'#'*80}")
    print(f"# COMPARISON WITH GOOD PERFORMING TASK: {good_task}")
    print(f"{'#'*80}")
    
    for task, desc in problem_tasks:
        if task in results:
            try:
                compare_with_good_task(task, good_task, run_dir, data_root)
            except Exception as e:
                print(f"Error comparing {task}: {e}")
    
    # Generate summary report
    print(f"\n\n{'='*80}")
    print("SUMMARY REPORT & DIAGNOSIS")
    print(f"{'='*80}")
    
    for task, result in results.items():
        if result['meta'] is not None:
            meta = result['meta']
            data = result['data']
            
            print(f"\n{'='*80}")
            print(f"{task}")
            print(f"{'='*80}")
            
            print(f"\nPerformance:")
            print(f"  Test {meta['primary_metric_name']}: {meta['primary_metric']:.4f}")
            print(f"  Valid {meta['primary_metric_name']}: {meta['best_valid_primary']:.4f}")
            print(f"  Gap (Test - Valid): {meta['primary_metric'] - meta['best_valid_primary']:.4f}")
            print(f"  Best Epoch: {meta['best_epoch']}")
            
            # Count labels
            test_labels = data['test']['label'].value_counts()
            train_labels = data['train']['label'].value_counts()
            
            print(f"\nData Distribution:")
            print(f"  Train size: {len(data['train'])}, pos_ratio: {train_labels.get(1,0)/len(data['train']):.3f}")
            print(f"  Test size: {len(data['test'])}, pos_ratio: {test_labels.get(1,0)/len(data['test']):.3f}")
            
            # Diagnose issues
            print(f"\n🔍 DIAGNOSIS:")
            issues = []
            
            if meta['best_epoch'] <= 3:
                issues.append("❌ EARLY STOPPING: Best epoch ≤ 3 suggests immediate overfitting or poor initialization")
                issues.append("   → Possible causes:")
                issues.append("     • Learning rate too high")
                issues.append("     • Focal loss too aggressive for this imbalance")
                issues.append("     • Label smoothing conflicting with focal loss")
                issues.append("     • High dropout causing underfitting")
            
            gap = meta['primary_metric'] - meta['best_valid_primary']
            if abs(gap) > 0.15:
                issues.append(f"❌ LARGE TEST-VALID GAP: {gap:.4f} indicates distribution shift or overfitting")
            
            if meta['primary_metric'] < 0.6:
                issues.append(f"❌ VERY POOR PERFORMANCE: {meta['primary_metric']:.4f} < 0.6")
                issues.append("   → Model might be:")
                issues.append("     • Predicting random or constant values")
                issues.append("     • Unable to learn useful features")
                issues.append("     • Suffering from severe class imbalance handling")
            
            if meta.get('test_acc', 0) < 0.5:
                issues.append(f"❌ ACCURACY < 0.5: {meta.get('test_acc', 0):.4f} - worse than random!")
            
            # Check for extreme imbalance
            train_pos_ratio = train_labels.get(1, 0) / len(data['train'])
            if train_pos_ratio < 0.25 or train_pos_ratio > 0.75:
                issues.append(f"⚠️  CLASS IMBALANCE: pos_ratio={train_pos_ratio:.3f}")
                issues.append("   → Current settings use Focal Loss with strong gamma")
                issues.append("     • This might be TOO aggressive, preventing learning")
            
            if issues:
                for issue in issues:
                    print(f"  {issue}")
            else:
                print("  ✓ No obvious issues detected")
            
            # Recommendations
            print(f"\n💡 RECOMMENDATIONS:")
            
            if meta['best_epoch'] <= 3:
                print("  1. Try reducing learning rate (e.g., 1e-4 instead of 3e-4)")
                print("  2. Try simpler loss function (CrossEntropy instead of Focal)")
                print("  3. Remove label smoothing when using Focal Loss")
                print("  4. Reduce dropout (e.g., 0.1 instead of 0.2/0.3)")
            
            if train_pos_ratio < 0.25 or train_pos_ratio > 0.75:
                print("  5. Try class weights instead of Focal Loss")
                print("  6. Try SMOTE or other resampling techniques")
                print(f"  7. Current focal_gamma=2.5 might be too high, try 1.0-1.5")
            
            if abs(gap) > 0.15:
                print("  8. Increase regularization (weight_decay)")
                print("  9. Use more aggressive data augmentation")
    
    print(f"\n{'='*80}")
    print("Analysis complete! Check results/analysis/poor_performance/ for plots")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
