#!/bin/bash
# Single-modality ablation study: run each modality separately to identify which is most important

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Setup environment
export UNIMOL_WEIGHT_DIR="$PROJECT_ROOT/data/weights"
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/mnt/afs/250010150/envs/kpgt/lib:$LD_LIBRARY_PATH

PYTHON=/mnt/afs/250010150/envs/trimole/bin/python
TIMESTAMP=$(date +%Y%m%d_%H%M)

# Output directory for ablation experiments
OUT_ROOT="$PROJECT_ROOT/results/ablation_single_modality"
mkdir -p "$OUT_ROOT"

# Common training parameters - using conservative settings for fair comparison
COMMON_ARGS="
  --data-new $PROJECT_ROOT/data/data_new
  --max-epochs 80
  --patience 15
  --batch-size 64
  --lr 2e-4
  --hidden-dim 128
  --weight-decay 0.01
  --dropout-proj 0.25
  --dropout-head 0.35
  --use-task-configs
  --task-config-variant adaptive
  --baselines-dir $PROJECT_ROOT/results/baselines
"

echo "=========================================="
echo "Single-Modality Ablation Study"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Project root: $PROJECT_ROOT"
echo "Output root: $OUT_ROOT"
echo ""

# Run 1: ChemBERTa only (SMILES)
echo "----------------------------------------"
echo "Run 1/3: ChemBERTa only (SMILES)"
echo "----------------------------------------"
RUN_DIR_CHEMBERTA="$OUT_ROOT/run_${TIMESTAMP}_chemberta_only"
$PYTHON -m trimole.pipelines.batch_run_data_new \
  --out "$RUN_DIR_CHEMBERTA" \
  --modalities chemberta \
  $COMMON_ARGS

echo ""
echo "✓ ChemBERTa run completed: $RUN_DIR_CHEMBERTA"
echo ""

# Run 2: KPGT only (Graph)
echo "----------------------------------------"
echo "Run 2/3: KPGT only (Graph)"
echo "----------------------------------------"
RUN_DIR_KPGT="$OUT_ROOT/run_${TIMESTAMP}_kpgt_only"
$PYTHON -m trimole.pipelines.batch_run_data_new \
  --out "$RUN_DIR_KPGT" \
  --modalities kpgt \
  $COMMON_ARGS

echo ""
echo "✓ KPGT run completed: $RUN_DIR_KPGT"
echo ""

# Run 3: Uni-Mol only (3D)
echo "----------------------------------------"
echo "Run 3/3: Uni-Mol only (3D)"
echo "----------------------------------------"
RUN_DIR_UNIMOL="$OUT_ROOT/run_${TIMESTAMP}_unimol_only"
$PYTHON -m trimole.pipelines.batch_run_data_new \
  --out "$RUN_DIR_UNIMOL" \
  --modalities unimol \
  $COMMON_ARGS

echo ""
echo "✓ Uni-Mol run completed: $RUN_DIR_UNIMOL"
echo ""

# Generate comparison report
echo "=========================================="
echo "Generating Ablation Comparison Report"
echo "=========================================="

COMPARISON_DIR="$OUT_ROOT/comparison_${TIMESTAMP}"
mkdir -p "$COMPARISON_DIR"

$PYTHON "$SCRIPT_DIR/analyze_ablation_results.py" \
  --chemberta-run "$RUN_DIR_CHEMBERTA" \
  --kpgt-run "$RUN_DIR_KPGT" \
  --unimol-run "$RUN_DIR_UNIMOL" \
  --baselines-dir "$PROJECT_ROOT/results/baselines" \
  --out-dir "$COMPARISON_DIR"

echo ""
echo "=========================================="
echo "Ablation Study Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - ChemBERTa: $RUN_DIR_CHEMBERTA"
echo "  - KPGT:      $RUN_DIR_KPGT"
echo "  - Uni-Mol:   $RUN_DIR_UNIMOL"
echo "  - Comparison: $COMPARISON_DIR"
echo ""
echo "Key files:"
echo "  - Summary: $COMPARISON_DIR/ablation_summary.csv"
echo "  - Visualization: $COMPARISON_DIR/ablation_comparison.png"
echo ""
