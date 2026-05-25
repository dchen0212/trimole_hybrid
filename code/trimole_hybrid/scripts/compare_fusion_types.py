#!/usr/bin/env python3
"""Compare MLP fusion vs Gated fusion performance."""

import pandas as pd
import sys
from pathlib import Path

def compare_runs(mlp_csv: Path, gated_csv: Path):
    """Compare two runs and generate analysis."""
    
    mlp_df = pd.read_csv(mlp_csv)
    gated_df = pd.read_csv(gated_csv)
    
    # Merge on task
    merged = mlp_df.merge(gated_df, on='task', suffixes=('_mlp', '_gated'))
    
    # Calculate improvement
    results = []
    for _, row in merged.iterrows():
        task = row['task']
        metric = row['primary_metric_name_mlp']
        mlp_score = row['primary_metric_mlp']
        gated_score = row['primary_metric_gated']
        
        # For MAE/RMSE, lower is better
        if metric.upper() in ['MAE', 'RMSE']:
            improvement = mlp_score - gated_score  # positive = gated better
            pct_change = (improvement / mlp_score) * 100
        else:
            improvement = gated_score - mlp_score  # positive = gated better
            pct_change = (improvement / mlp_score) * 100
        
        results.append({
            'task': task,
            'metric': metric,
            'mlp': mlp_score,
            'gated': gated_score,
            'improvement': improvement,
            'pct_change': pct_change,
            'winner': 'gated' if improvement > 0 else 'mlp' if improvement < 0 else 'tie'
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('improvement', ascending=False)
    
    # Summary statistics
    wins_gated = (result_df['improvement'] > 0).sum()
    wins_mlp = (result_df['improvement'] < 0).sum()
    ties = (result_df['improvement'] == 0).sum()
    
    avg_improvement = result_df['improvement'].mean()
    avg_pct_change = result_df['pct_change'].mean()
    
    print("=" * 80)
    print("GATED FUSION vs MLP FUSION COMPARISON")
    print("=" * 80)
    print()
    print(f"Total tasks: {len(result_df)}")
    print(f"Gated wins: {wins_gated} ({wins_gated/len(result_df)*100:.1f}%)")
    print(f"MLP wins: {wins_mlp} ({wins_mlp/len(result_df)*100:.1f}%)")
    print(f"Ties: {ties}")
    print()
    print(f"Average improvement: {avg_improvement:+.4f}")
    print(f"Average % change: {avg_pct_change:+.2f}%")
    print()
    
    print("=" * 80)
    print("TOP 10 IMPROVEMENTS (Gated better)")
    print("=" * 80)
    top10 = result_df.head(10)
    for _, row in top10.iterrows():
        print(f"{row['task']:40s} {row['metric']:10s} {row['improvement']:+.4f} ({row['pct_change']:+.2f}%)")
    print()
    
    print("=" * 80)
    print("TOP 10 REGRESSIONS (MLP better)")
    print("=" * 80)
    bottom10 = result_df.tail(10)
    for _, row in bottom10.iterrows():
        print(f"{row['task']:40s} {row['metric']:10s} {row['improvement']:+.4f} ({row['pct_change']:+.2f}%)")
    print()
    
    return result_df

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python compare_fusion_types.py <mlp_results_all.csv> <gated_results_all.csv>")
        sys.exit(1)
    
    mlp_csv = Path(sys.argv[1])
    gated_csv = Path(sys.argv[2])
    
    result_df = compare_runs(mlp_csv, gated_csv)
    
    # Save detailed comparison
    out_csv = Path("fusion_comparison.csv")
    result_df.to_csv(out_csv, index=False, float_format='%.6f')
    print(f"Detailed comparison saved to: {out_csv}")
