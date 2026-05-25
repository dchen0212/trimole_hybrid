#!/usr/bin/env python3
"""
Analyze and visualize single-modality ablation study results.

Compares the performance of each modality (ChemBERTa, KPGT, Uni-Mol) 
to identify which contributes most to model performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(run_dir: Path) -> pd.DataFrame:
    """Load results_all.csv from a run directory."""
    results_csv = run_dir / "results_all.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Results file not found: {results_csv}")
    
    df = pd.read_csv(results_csv)
    return df


def normalize_metric_for_comparison(value: float, metric_name: str) -> float:
    """
    Normalize metric values for comparison.
    For MAE/RMSE (lower is better), use negative value.
    """
    metric_upper = str(metric_name).upper()
    if metric_upper in {"MAE", "RMSE", "MSE"}:
        return -float(value)
    return float(value)


def compute_improvement(ablation_value: float, baseline_value: float, metric_name: str) -> float:
    """
    Compute improvement over baseline.
    For lower-is-better metrics (MAE, RMSE), improvement = baseline - ablation.
    For higher-is-better metrics (AUROC, AUPRC, Spearman), improvement = ablation - baseline.
    """
    metric_upper = str(metric_name).upper()
    if metric_upper in {"MAE", "RMSE", "MSE"}:
        # Lower is better: improvement is positive when ablation < baseline
        return float(baseline_value) - float(ablation_value)
    else:
        # Higher is better: improvement is positive when ablation > baseline
        return float(ablation_value) - float(baseline_value)


def load_baselines(baselines_dir: Path) -> Dict[str, tuple[str, float]]:
    """
    Load baseline best scores from TDCommons leaderboard CSVs.
    Returns dict: task_name -> (metric_name, best_score)
    """
    import re
    
    def normalize_task_name(name: str) -> str:
        s = name.strip().lower()
        s = re.sub(r"leaderboard", "", s)
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s
    
    def parse_score(x) -> float:
        if x is None or pd.isna(x):
            return float("nan")
        s = str(x).strip()
        if not s:
            return float("nan")
        s = s.replace("±", "+-").split("+-", 1)[0].replace(",", "")
        try:
            return float(s)
        except Exception:
            return float("nan")
    
    baselines = {}
    metric_candidates = ["AUROC", "AUC", "AUPRC", "AUCPR", "MAE", "RMSE", "MSE", "Spearman", "ACC"]
    
    for csv_path in baselines_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            
            # Find metric column
            metric_col = None
            for col in df.columns:
                if str(col).strip() in metric_candidates:
                    metric_col = str(col).strip()
                    break
            
            if metric_col is None:
                continue
            
            # Get dataset_id
            if "dataset_id" in df.columns:
                dataset_id = str(df.iloc[0]["dataset_id"])
            else:
                stem = csv_path.stem
                dataset_id = re.sub(r"^tdcommons_", "", stem, flags=re.IGNORECASE)
                dataset_id = re.sub(r"_?leaderboard(_leaderboard)?$", "", dataset_id, flags=re.IGNORECASE)
            
            task_norm = normalize_task_name(dataset_id)
            
            # Get best score (first row, typically)
            scores = [parse_score(x) for x in df[metric_col]]
            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                best_score = valid_scores[0]  # Assuming first row is best
                baselines[task_norm] = (metric_col, best_score)
        
        except Exception as e:
            print(f"Warning: Failed to load baseline from {csv_path.name}: {e}")
    
    return baselines


def create_comparison_table(
    chemberta_df: pd.DataFrame,
    kpgt_df: pd.DataFrame,
    unimol_df: pd.DataFrame,
    baselines: Dict[str, tuple[str, float]],
) -> pd.DataFrame:
    """
    Create a comprehensive comparison table across all modalities.
    """
    
    def normalize_task(name: str) -> str:
        import re
        s = name.strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s
    
    # Merge results from all three modalities
    records = []
    
    for _, row in chemberta_df.iterrows():
        task = str(row["task"])
        task_norm = normalize_task(task)
        metric_name = str(row["primary_metric_name"])
        
        # Get scores from each modality
        chemberta_score = float(row["primary_metric"])
        
        kpgt_row = kpgt_df[kpgt_df["task"] == task]
        kpgt_score = float(kpgt_row["primary_metric"].iloc[0]) if not kpgt_row.empty else np.nan
        
        unimol_row = unimol_df[unimol_df["task"] == task]
        unimol_score = float(unimol_row["primary_metric"].iloc[0]) if not unimol_row.empty else np.nan
        
        # Get baseline
        baseline_score = np.nan
        if task_norm in baselines:
            _, baseline_score = baselines[task_norm]
        
        # Compute improvements over baseline
        chemberta_imp = compute_improvement(chemberta_score, baseline_score, metric_name) if not np.isnan(baseline_score) else np.nan
        kpgt_imp = compute_improvement(kpgt_score, baseline_score, metric_name) if not np.isnan(baseline_score) else np.nan
        unimol_imp = compute_improvement(unimol_score, baseline_score, metric_name) if not np.isnan(baseline_score) else np.nan
        
        # Identify best modality (highest score for higher-is-better, lowest for lower-is-better)
        metric_upper = metric_name.upper()
        if metric_upper in {"MAE", "RMSE", "MSE"}:
            # Lower is better
            scores = {
                "chemberta": chemberta_score,
                "kpgt": kpgt_score,
                "unimol": unimol_score,
            }
            valid_scores = {k: v for k, v in scores.items() if not np.isnan(v)}
            best_modality = min(valid_scores, key=valid_scores.get) if valid_scores else "none"
        else:
            # Higher is better
            scores = {
                "chemberta": chemberta_score,
                "kpgt": kpgt_score,
                "unimol": unimol_score,
            }
            valid_scores = {k: v for k, v in scores.items() if not np.isnan(v)}
            best_modality = max(valid_scores, key=valid_scores.get) if valid_scores else "none"
        
        records.append({
            "task": task,
            "task_type": row["task_type"],
            "metric": metric_name,
            "baseline": baseline_score,
            "chemberta": chemberta_score,
            "kpgt": kpgt_score,
            "unimol": unimol_score,
            "chemberta_vs_baseline": chemberta_imp,
            "kpgt_vs_baseline": kpgt_imp,
            "unimol_vs_baseline": unimol_imp,
            "best_modality": best_modality,
        })
    
    df = pd.DataFrame(records)
    return df


def plot_ablation_comparison(
    df: pd.DataFrame,
    out_path: Path,
    top_k: int = 30,
):
    """
    Create visualization comparing modality performance.
    """
    
    # Prepare data for plotting
    plot_df = df.copy()
    
    # Normalize scores for fair comparison (all metrics higher=better after normalization)
    for col in ["chemberta", "kpgt", "unimol", "baseline"]:
        plot_df[f"{col}_norm"] = plot_df.apply(
            lambda row: normalize_metric_for_comparison(row[col], row["metric"]),
            axis=1
        )
    
    # Compute relative performance (percentage of baseline)
    for modality in ["chemberta", "kpgt", "unimol"]:
        plot_df[f"{modality}_pct"] = (plot_df[f"{modality}_norm"] / plot_df["baseline_norm"]) * 100
    
    # Sort by average performance across modalities
    plot_df["avg_pct"] = plot_df[["chemberta_pct", "kpgt_pct", "unimol_pct"]].mean(axis=1)
    plot_df = plot_df.sort_values("avg_pct", ascending=False)
    
    # Limit to top_k tasks
    if len(plot_df) > top_k:
        plot_df = plot_df.head(top_k)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, max(12, len(plot_df) * 0.4)))
    
    # Plot 1: Absolute performance
    ax1 = axes[0]
    x = np.arange(len(plot_df))
    width = 0.2
    
    ax1.barh(x - width*1.5, plot_df["chemberta_norm"], width, label="ChemBERTa (SMILES)", alpha=0.8, color="#1f77b4")
    ax1.barh(x - width*0.5, plot_df["kpgt_norm"], width, label="KPGT (Graph)", alpha=0.8, color="#ff7f0e")
    ax1.barh(x + width*0.5, plot_df["unimol_norm"], width, label="Uni-Mol (3D)", alpha=0.8, color="#2ca02c")
    ax1.barh(x + width*1.5, plot_df["baseline_norm"], width, label="TDCommons Baseline", alpha=0.5, color="#d62728")
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(plot_df["task"], fontsize=9)
    ax1.set_xlabel("Normalized Performance (higher = better)", fontsize=11)
    ax1.set_title("Single-Modality Ablation: Absolute Performance", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(axis="x", alpha=0.3)
    ax1.axvline(0, color="black", linewidth=0.8)
    
    # Plot 2: Relative performance (% of baseline)
    ax2 = axes[1]
    
    ax2.barh(x - width, plot_df["chemberta_pct"], width, label="ChemBERTa (SMILES)", alpha=0.8, color="#1f77b4")
    ax2.barh(x, plot_df["kpgt_pct"], width, label="KPGT (Graph)", alpha=0.8, color="#ff7f0e")
    ax2.barh(x + width, plot_df["unimol_pct"], width, label="Uni-Mol (3D)", alpha=0.8, color="#2ca02c")
    
    ax2.axvline(100, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Baseline (100%)")
    
    ax2.set_yticks(x)
    ax2.set_yticklabels(plot_df["task"], fontsize=9)
    ax2.set_xlabel("Performance (% of TDCommons Baseline)", fontsize=11)
    ax2.set_title("Single-Modality Ablation: Relative to Baseline", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved comparison plot: {out_path}")


def generate_summary_stats(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics across all tasks.
    """
    
    # Count best modality
    best_counts = df["best_modality"].value_counts().to_dict()
    
    # Average performance vs baseline
    avg_vs_baseline = {
        "chemberta": df["chemberta_vs_baseline"].mean(),
        "kpgt": df["kpgt_vs_baseline"].mean(),
        "unimol": df["unimol_vs_baseline"].mean(),
    }
    
    # Win rate (tasks where modality beats baseline)
    win_rate = {
        "chemberta": (df["chemberta_vs_baseline"] > 0).sum() / len(df),
        "kpgt": (df["kpgt_vs_baseline"] > 0).sum() / len(df),
        "unimol": (df["unimol_vs_baseline"] > 0).sum() / len(df),
    }
    
    return {
        "total_tasks": len(df),
        "best_modality_counts": best_counts,
        "avg_improvement_vs_baseline": avg_vs_baseline,
        "win_rate_vs_baseline": win_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze single-modality ablation study results")
    parser.add_argument("--chemberta-run", type=str, required=True, help="Path to ChemBERTa-only run directory")
    parser.add_argument("--kpgt-run", type=str, required=True, help="Path to KPGT-only run directory")
    parser.add_argument("--unimol-run", type=str, required=True, help="Path to Uni-Mol-only run directory")
    parser.add_argument("--baselines-dir", type=str, required=True, help="Path to baselines directory")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for comparison results")
    parser.add_argument("--top-k", type=int, default=30, help="Number of top tasks to show in visualization")
    
    args = parser.parse_args()
    
    chemberta_dir = Path(args.chemberta_run)
    kpgt_dir = Path(args.kpgt_run)
    unimol_dir = Path(args.unimol_run)
    baselines_dir = Path(args.baselines_dir)
    out_dir = Path(args.out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Single-Modality Ablation Analysis")
    print("=" * 60)
    print()
    
    # Load results
    print("Loading results...")
    chemberta_df = load_results(chemberta_dir)
    kpgt_df = load_results(kpgt_dir)
    unimol_df = load_results(unimol_dir)
    print(f"  ChemBERTa: {len(chemberta_df)} tasks")
    print(f"  KPGT:      {len(kpgt_df)} tasks")
    print(f"  Uni-Mol:   {len(unimol_df)} tasks")
    print()
    
    # Load baselines
    print("Loading baselines...")
    baselines = load_baselines(baselines_dir)
    print(f"  Loaded {len(baselines)} baseline entries")
    print()
    
    # Create comparison table
    print("Creating comparison table...")
    comparison_df = create_comparison_table(chemberta_df, kpgt_df, unimol_df, baselines)
    
    # Save comparison table
    comparison_csv = out_dir / "ablation_summary.csv"
    comparison_df.to_csv(comparison_csv, index=False, float_format="%.4f")
    print(f"✓ Saved comparison table: {comparison_csv}")
    print()
    
    # Generate summary statistics
    print("Computing summary statistics...")
    summary_stats = generate_summary_stats(comparison_df)
    
    summary_json = out_dir / "ablation_stats.json"
    with open(summary_json, "w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"✓ Saved summary stats: {summary_json}")
    print()
    
    # Print summary
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total tasks analyzed: {summary_stats['total_tasks']}")
    print()
    print("Best modality counts:")
    for modality, count in summary_stats['best_modality_counts'].items():
        print(f"  {modality:12s}: {count:2d} tasks ({count/summary_stats['total_tasks']*100:.1f}%)")
    print()
    print("Average improvement over baseline:")
    for modality, imp in summary_stats['avg_improvement_vs_baseline'].items():
        print(f"  {modality:12s}: {imp:+.4f}")
    print()
    print("Win rate vs baseline (% tasks beating baseline):")
    for modality, rate in summary_stats['win_rate_vs_baseline'].items():
        print(f"  {modality:12s}: {rate*100:.1f}%")
    print()
    
    # Create visualization
    print("Creating visualization...")
    plot_path = out_dir / "ablation_comparison.png"
    plot_ablation_comparison(comparison_df, plot_path, top_k=args.top_k)
    print()
    
    # Create detailed comparison by best modality
    print("Creating per-modality breakdown...")
    for modality in ["chemberta", "kpgt", "unimol"]:
        modality_tasks = comparison_df[comparison_df["best_modality"] == modality]
        if not modality_tasks.empty:
            modality_csv = out_dir / f"tasks_best_on_{modality}.csv"
            modality_tasks.to_csv(modality_csv, index=False, float_format="%.4f")
            print(f"✓ Saved {modality} best tasks: {modality_csv}")
    print()
    
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
