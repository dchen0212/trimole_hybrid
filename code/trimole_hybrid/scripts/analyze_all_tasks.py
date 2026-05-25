#!/usr/bin/env python3
"""
Comprehensive analysis of all 22 tasks.
Identify patterns in successful vs failing tasks.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_all_results(run_dir, data_root):
    """Load results for all tasks"""
    results = {}
    
    # Load baseline comparison
    baseline_file = run_dir / 'compare' / 'baseline_comparison.csv'
    baseline_df = pd.read_csv(baseline_file)
    
    for _, row in baseline_df.iterrows():
        task = row['task']
        
        # Load task metadata
        meta_file = run_dir / task / 'meta.json'
        if not meta_file.exists():
            continue
            
        with open(meta_file) as f:
            meta = json.load(f)
        
        # Load training history
        history_file = run_dir / task / 'history.json'
        history = None
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)
        
        # Load data info
        train_file = data_root / 'data' / 'data_new' / task / 'train.csv'
        test_file = data_root / 'data' / 'data_new' / task / 'test.csv'
        
        train_df = pd.read_csv(train_file) if train_file.exists() else None
        test_df = pd.read_csv(test_file) if test_file.exists() else None
        
        results[task] = {
            'baseline_info': row.to_dict(),
            'meta': meta,
            'history': history,
            'train_df': train_df,
            'test_df': test_df,
        }
    
    return results

def categorize_tasks(results):
    """Categorize tasks by performance"""
    categories = {
        'excellent': [],      # improvement > 3%
        'good': [],          # 0% < improvement <= 3%
        'marginal': [],      # -5% < improvement <= 0%
        'poor': [],          # -15% < improvement <= -5%
        'failed': [],        # improvement <= -15%
    }
    
    for task, data in results.items():
        improvement = data['baseline_info']['improvement']
        direction = data['baseline_info']['direction']
        
        # Normalize improvement (positive = better)
        if direction == 'lower_better':
            improvement = -improvement
        
        if improvement > 0.03:
            categories['excellent'].append(task)
        elif improvement > 0:
            categories['good'].append(task)
        elif improvement > -0.05:
            categories['marginal'].append(task)
        elif improvement > -0.15:
            categories['poor'].append(task)
        else:
            categories['failed'].append(task)
    
    return categories

def analyze_task_characteristics(results):
    """Analyze characteristics of each task"""
    characteristics = {}
    
    for task, data in results.items():
        meta = data['meta']
        train_df = data['train_df']
        test_df = data['test_df']
        history = data['history']
        baseline_info = data['baseline_info']
        
        char = {
            'task': task,
            'task_type': meta['task_type'],
            'metric': meta['primary_metric_name'],
            'our_score': meta['primary_metric'],
            'baseline_score': baseline_info['baseline_best'],
            'improvement': baseline_info['improvement'],
            'direction': baseline_info['direction'],
        }
        
        # Data characteristics
        if train_df is not None:
            char['train_size'] = len(train_df)
            char['test_size'] = len(test_df) if test_df is not None else 0
            
            if 'label' in train_df.columns:
                label_counts = train_df['label'].value_counts()
                pos_count = label_counts.get(1, 0)
                char['pos_ratio'] = pos_count / len(train_df)
                char['imbalance_ratio'] = max(label_counts) / min(label_counts) if len(label_counts) > 1 else 1.0
            else:
                char['pos_ratio'] = None
                char['imbalance_ratio'] = None
        
        # Training characteristics
        char['best_epoch'] = meta['best_epoch']
        char['loss_type'] = meta['loss_type']
        
        if history:
            char['total_epochs'] = len(history)
            char['early_stop_ratio'] = meta['best_epoch'] / len(history)
            
            # Check training stability
            train_losses = [h['train_loss'] for h in history]
            char['final_train_loss'] = train_losses[-1]
            char['loss_reduction'] = (train_losses[0] - train_losses[-1]) / train_losses[0]
        
        # Performance gap
        if meta['task_type'] == 'classification':
            char['test_acc'] = meta.get('test_acc', None)
            char['valid_test_gap'] = meta['primary_metric'] - meta['best_valid_primary']
        
        characteristics[task] = char
    
    return characteristics

def create_comprehensive_analysis(characteristics, categories):
    """Create comprehensive analysis report"""
    
    print("="*80)
    print("COMPREHENSIVE ANALYSIS OF ALL 22 TASKS")
    print("="*80)
    
    # Overall statistics
    print(f"\n{'='*80}")
    print("OVERALL PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    for category, tasks in categories.items():
        print(f"\n{category.upper()}: {len(tasks)} tasks")
        for task in tasks:
            char = characteristics[task]
            improvement = char['improvement']
            if char['direction'] == 'lower_better':
                improvement = -improvement
            print(f"  - {task}: {char['metric']}={char['our_score']:.4f} "
                  f"(baseline={char['baseline_score']:.4f}, Δ={improvement:+.1%})")
    
    # Analyze patterns
    print(f"\n\n{'='*80}")
    print("PATTERN ANALYSIS")
    print(f"{'='*80}")
    
    # Pattern 1: Dataset size
    print(f"\n{'─'*80}")
    print("1. DATASET SIZE ANALYSIS")
    print(f"{'─'*80}")
    
    for category in ['excellent', 'good', 'marginal', 'poor', 'failed']:
        tasks = categories[category]
        if not tasks:
            continue
        
        sizes = [characteristics[t]['train_size'] for t in tasks]
        avg_size = np.mean(sizes)
        print(f"\n{category.upper()} ({len(tasks)} tasks):")
        print(f"  Average train size: {avg_size:.0f}")
        print(f"  Range: {min(sizes)} - {max(sizes)}")
        print(f"  Tasks: {', '.join(tasks[:3])}{'...' if len(tasks) > 3 else ''}")
    
    # Pattern 2: Class imbalance
    print(f"\n{'─'*80}")
    print("2. CLASS IMBALANCE ANALYSIS (Classification tasks only)")
    print(f"{'─'*80}")
    
    for category in ['excellent', 'good', 'marginal', 'poor', 'failed']:
        tasks = [t for t in categories[category] if characteristics[t]['pos_ratio'] is not None]
        if not tasks:
            continue
        
        pos_ratios = [characteristics[t]['pos_ratio'] for t in tasks]
        imbalance_ratios = [characteristics[t]['imbalance_ratio'] for t in tasks]
        
        print(f"\n{category.upper()} ({len(tasks)} classification tasks):")
        print(f"  Average pos_ratio: {np.mean(pos_ratios):.3f}")
        print(f"  Average imbalance_ratio: {np.mean(imbalance_ratios):.2f}")
        
        # Find extreme imbalance
        extreme = [t for t in tasks if characteristics[t]['pos_ratio'] < 0.3 or characteristics[t]['pos_ratio'] > 0.7]
        if extreme:
            print(f"  Extreme imbalance: {', '.join(extreme)}")
    
    # Pattern 3: Early stopping
    print(f"\n{'─'*80}")
    print("3. EARLY STOPPING ANALYSIS")
    print(f"{'─'*80}")
    
    for category in ['excellent', 'good', 'marginal', 'poor', 'failed']:
        tasks = categories[category]
        if not tasks:
            continue
        
        best_epochs = [characteristics[t]['best_epoch'] for t in tasks]
        early_stops = [t for t in tasks if characteristics[t]['best_epoch'] <= 3]
        
        print(f"\n{category.upper()}:")
        print(f"  Average best_epoch: {np.mean(best_epochs):.1f}")
        print(f"  Very early stops (≤3): {len(early_stops)}/{len(tasks)}")
        if early_stops:
            print(f"    Tasks: {', '.join(early_stops)}")
    
    # Pattern 4: Loss type
    print(f"\n{'─'*80}")
    print("4. LOSS TYPE ANALYSIS")
    print(f"{'─'*80}")
    
    loss_performance = defaultdict(list)
    for task, char in characteristics.items():
        improvement = char['improvement']
        if char['direction'] == 'lower_better':
            improvement = -improvement
        loss_performance[char['loss_type']].append(improvement)
    
    for loss_type, improvements in loss_performance.items():
        avg_improvement = np.mean(improvements)
        print(f"\n{loss_type}:")
        print(f"  Average improvement: {avg_improvement:+.1%}")
        print(f"  Tasks: {len(improvements)}")
        print(f"  Success rate: {sum(1 for x in improvements if x > 0) / len(improvements):.1%}")
    
    # Pattern 5: Generalization gap
    print(f"\n{'─'*80}")
    print("5. GENERALIZATION GAP ANALYSIS (Classification)")
    print(f"{'─'*80}")
    
    for category in ['excellent', 'good', 'marginal', 'poor', 'failed']:
        tasks = [t for t in categories[category] if characteristics[t].get('valid_test_gap') is not None]
        if not tasks:
            continue
        
        gaps = [characteristics[t]['valid_test_gap'] for t in tasks]
        large_gaps = [t for t in tasks if abs(characteristics[t]['valid_test_gap']) > 0.1]
        
        print(f"\n{category.upper()}:")
        print(f"  Average valid-test gap: {np.mean(gaps):+.4f}")
        print(f"  Large gaps (|gap|>0.1): {len(large_gaps)}/{len(tasks)}")
        if large_gaps:
            for t in large_gaps:
                gap = characteristics[t]['valid_test_gap']
                print(f"    {t}: {gap:+.4f}")
    
    return characteristics

def identify_common_issues(characteristics, categories):
    """Identify common issues in failing tasks"""
    
    print(f"\n\n{'='*80}")
    print("COMMON ISSUES IN FAILING TASKS")
    print(f"{'='*80}")
    
    failing_tasks = categories['poor'] + categories['failed']
    
    issues = {
        'small_dataset': [],
        'extreme_imbalance': [],
        'early_stopping': [],
        'large_gap': [],
        'focal_loss_issue': [],
    }
    
    for task in failing_tasks:
        char = characteristics[task]
        
        # Small dataset
        if char['train_size'] < 500:
            issues['small_dataset'].append(task)
        
        # Extreme imbalance
        if char['pos_ratio'] is not None:
            if char['pos_ratio'] < 0.25 or char['pos_ratio'] > 0.75:
                issues['extreme_imbalance'].append(task)
        
        # Early stopping
        if char['best_epoch'] <= 3:
            issues['early_stopping'].append(task)
        
        # Large generalization gap
        if char.get('valid_test_gap') is not None:
            if abs(char['valid_test_gap']) > 0.15:
                issues['large_gap'].append(task)
        
        # Focal loss issue
        if char['loss_type'] == 'FocalLoss' and char['best_epoch'] <= 3:
            issues['focal_loss_issue'].append(task)
    
    print(f"\nTotal failing tasks: {len(failing_tasks)}")
    print(f"\nIssue breakdown:")
    for issue_type, tasks in issues.items():
        if tasks:
            print(f"\n{issue_type.upper().replace('_', ' ')}: {len(tasks)} tasks")
            for task in tasks:
                char = characteristics[task]
                print(f"  - {task}")
                print(f"      Size: {char['train_size']}, Best epoch: {char['best_epoch']}, "
                      f"Loss: {char['loss_type']}")
                if char['pos_ratio'] is not None:
                    print(f"      Pos ratio: {char['pos_ratio']:.3f}")
    
    # Find tasks with multiple issues
    print(f"\n{'─'*80}")
    print("TASKS WITH MULTIPLE ISSUES")
    print(f"{'─'*80}")
    
    issue_counts = defaultdict(int)
    task_issues = defaultdict(list)
    
    for issue_type, tasks in issues.items():
        for task in tasks:
            issue_counts[task] += 1
            task_issues[task].append(issue_type)
    
    for task in sorted(issue_counts.keys(), key=lambda x: issue_counts[x], reverse=True):
        count = issue_counts[task]
        if count >= 2:
            char = characteristics[task]
            improvement = char['improvement']
            if char['direction'] == 'lower_better':
                improvement = -improvement
            
            print(f"\n{task} ({count} issues): {improvement:+.1%}")
            for issue in task_issues[task]:
                print(f"  ✗ {issue.replace('_', ' ')}")

def create_visualizations(characteristics, categories, output_dir):
    """Create visualization plots"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Performance vs Dataset Size
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for category, color in [('excellent', 'green'), ('good', 'lightgreen'), 
                            ('marginal', 'yellow'), ('poor', 'orange'), ('failed', 'red')]:
        tasks = categories[category]
        if not tasks:
            continue
        
        sizes = [characteristics[t]['train_size'] for t in tasks]
        improvements = [characteristics[t]['improvement'] if characteristics[t]['direction'] == 'higher_better' 
                       else -characteristics[t]['improvement'] for t in tasks]
        
        ax.scatter(sizes, improvements, c=color, label=category, s=100, alpha=0.6)
        
        # Annotate failed tasks
        if category in ['poor', 'failed']:
            for task, size, imp in zip(tasks, sizes, improvements):
                ax.annotate(task, (size, imp), fontsize=8, alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Improvement over Baseline', fontsize=12)
    ax.set_title('Performance vs Dataset Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_vs_size.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'performance_vs_size.png'}")
    plt.close()
    
    # Plot 2: Best Epoch Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # All tasks
    all_best_epochs = [characteristics[t]['best_epoch'] for t in characteristics.keys()]
    axes[0].hist(all_best_epochs, bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=3, color='red', linestyle='--', linewidth=2, label='Early stop threshold (≤3)')
    axes[0].set_xlabel('Best Epoch', fontsize=12)
    axes[0].set_ylabel('Number of Tasks', fontsize=12)
    axes[0].set_title('Distribution of Best Epochs (All Tasks)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # By category
    category_colors = {'excellent': 'green', 'good': 'lightgreen', 'marginal': 'yellow', 
                      'poor': 'orange', 'failed': 'red'}
    
    for category, color in category_colors.items():
        tasks = categories[category]
        if not tasks:
            continue
        best_epochs = [characteristics[t]['best_epoch'] for t in tasks]
        axes[1].hist(best_epochs, bins=10, alpha=0.5, label=category, color=color, edgecolor='black')
    
    axes[1].axvline(x=3, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Best Epoch', fontsize=12)
    axes[1].set_ylabel('Number of Tasks', fontsize=12)
    axes[1].set_title('Best Epochs by Performance Category', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_epoch_distribution.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'best_epoch_distribution.png'}")
    plt.close()
    
    # Plot 3: Class Imbalance vs Performance
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for category, color in category_colors.items():
        tasks = [t for t in categories[category] if characteristics[t]['pos_ratio'] is not None]
        if not tasks:
            continue
        
        pos_ratios = [characteristics[t]['pos_ratio'] for t in tasks]
        improvements = [characteristics[t]['improvement'] if characteristics[t]['direction'] == 'higher_better' 
                       else -characteristics[t]['improvement'] for t in tasks]
        
        ax.scatter(pos_ratios, improvements, c=color, label=category, s=100, alpha=0.6)
        
        # Annotate problematic tasks
        if category in ['poor', 'failed']:
            for task, ratio, imp in zip(tasks, pos_ratios, improvements):
                ax.annotate(task, (ratio, imp), fontsize=8, alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.axvline(x=0.3, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=0.7, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Positive Class Ratio', fontsize=12)
    ax.set_ylabel('Improvement over Baseline', fontsize=12)
    ax.set_title('Performance vs Class Imbalance (Classification Tasks)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_vs_imbalance.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'performance_vs_imbalance.png'}")
    plt.close()

def generate_recommendations(characteristics, categories):
    """Generate specific recommendations"""
    
    print(f"\n\n{'='*80}")
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print(f"{'='*80}")
    
    failing_tasks = categories['poor'] + categories['failed']
    marginal_tasks = categories['marginal']
    
    print(f"\n{'─'*80}")
    print("PRIORITY 1: CRITICAL FIXES (Failed & Poor Tasks)")
    print(f"{'─'*80}")
    
    for task in failing_tasks:
        char = characteristics[task]
        improvement = char['improvement']
        if char['direction'] == 'lower_better':
            improvement = -improvement
        
        print(f"\n{task}: {improvement:+.1%}")
        print(f"  Current: {char['metric']}={char['our_score']:.4f}, Baseline={char['baseline_score']:.4f}")
        print(f"  Issues:")
        
        recommendations = []
        
        if char['train_size'] < 500:
            print(f"    ✗ Small dataset ({char['train_size']} samples)")
            recommendations.append("Use small dataset config (lr=5e-5, dropout=0.05/0.1)")
        
        if char['best_epoch'] <= 3:
            print(f"    ✗ Very early stopping (epoch {char['best_epoch']})")
            recommendations.append("Reduce learning rate by 5-10x")
            recommendations.append("Reduce dropout")
        
        if char['loss_type'] == 'FocalLoss':
            print(f"    ✗ Using Focal Loss")
            recommendations.append("Switch to CrossEntropy + class weights")
        
        if char['pos_ratio'] is not None:
            if char['pos_ratio'] < 0.25:
                print(f"    ✗ Severe class imbalance (pos={char['pos_ratio']:.1%})")
                recommendations.append(f"Use pos_weight={1-char['pos_ratio']:.2f}/{char['pos_ratio']:.2f}")
            elif char['pos_ratio'] > 0.75:
                print(f"    ✗ Reverse class imbalance (pos={char['pos_ratio']:.1%})")
                recommendations.append(f"Use neg_weight={char['pos_ratio']:.2f}/{1-char['pos_ratio']:.2f}")
        
        if char.get('valid_test_gap') is not None and abs(char['valid_test_gap']) > 0.15:
            print(f"    ✗ Large generalization gap ({char['valid_test_gap']:+.4f})")
            recommendations.append("Increase weight_decay")
            recommendations.append("Increase max_patience")
        
        if recommendations:
            print(f"  Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"    {i}. {rec}")
    
    print(f"\n{'─'*80}")
    print("PRIORITY 2: OPTIMIZATION (Marginal Tasks)")
    print(f"{'─'*80}")
    
    for task in marginal_tasks[:5]:  # Show top 5
        char = characteristics[task]
        improvement = char['improvement']
        if char['direction'] == 'lower_better':
            improvement = -improvement
        
        print(f"\n{task}: {improvement:+.1%} (close to baseline)")
        print(f"  Potential improvements:")
        
        if char['best_epoch'] < 10:
            print(f"    • Early stopping at epoch {char['best_epoch']} - try longer training")
        
        if char['train_size'] < 1000:
            print(f"    • Small dataset ({char['train_size']}) - try data augmentation")
        
        print(f"    • Fine-tune learning rate and regularization")

def main():
    """Main analysis function"""
    
    print("="*80)
    print("COMPREHENSIVE ANALYSIS OF ALL TASKS")
    print("="*80)
    
    run_dir = PROJECT_ROOT / 'results' / 'model_log' / 'run_20260123_2200'
    data_root = PROJECT_ROOT
    output_dir = PROJECT_ROOT / 'results' / 'analysis' / 'comprehensive'
    
    # Load all results
    print("\nLoading results...")
    results = load_all_results(run_dir, data_root)
    print(f"Loaded {len(results)} tasks")
    
    # Categorize tasks
    categories = categorize_tasks(results)
    
    # Analyze characteristics
    characteristics = analyze_task_characteristics(results)
    
    # Create comprehensive analysis
    create_comprehensive_analysis(characteristics, categories)
    
    # Identify common issues
    identify_common_issues(characteristics, categories)
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    create_visualizations(characteristics, categories, output_dir)
    
    # Generate recommendations
    generate_recommendations(characteristics, categories)
    
    # Save detailed data
    char_df = pd.DataFrame.from_dict(characteristics, orient='index')
    char_df = char_df.sort_values('improvement', ascending=False)
    output_file = output_dir / 'task_characteristics.csv'
    char_df.to_csv(output_file)
    print(f"\n{'='*80}")
    print(f"Saved detailed characteristics to: {output_file}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
