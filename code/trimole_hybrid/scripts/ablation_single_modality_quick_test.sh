#!/bin/bash
# Quick test version of single-modality ablation study
# Only runs on 3 tasks for rapid validation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Setup environment
export UNIMOL_WEIGHT_DIR="$PROJECT_ROOT/data/weights"
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=<ENV_ROOT>/kpgt/lib:$LD_LIBRARY_PATH

PYTHON=<ENV_ROOT>/trimole/bin/python
TIMESTAMP=$(date +%Y%m%d_%H%M)

# Output directory for ablation experiments
OUT_ROOT="$PROJECT_ROOT/results/ablation_single_modality_test"
mkdir -p "$OUT_ROOT"

# Select 3 representative tasks for quick test:
# - ames: large classification (5094 samples)
# - bbb_martins: medium classification (1421 samples)  
# - solubility_aqsoldb: large regression (6986 samples)
TEST_TASKS="ames bbb_martins solubility_aqsoldb"

# Common training parameters - using faster settings for testing
COMMON_ARGS="
  --data-new $PROJECT_ROOT/data/data_new
  --max-epochs 20
  --patience 5
  --batch-size 64
  --lr 2e-4
  --hidden-dim 128
  --weight-decay 0.01
  --dropout-proj 0.25
  --dropout-head 0.35
  --use-task-configs
  --task-config-variant adaptive
  --baselines-dir $PROJECT_ROOT/results/baselines
  --tasks $TEST_TASKS
"

echo "=========================================="
echo "Single-Modality Ablation Study (QUICK TEST)"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Test tasks: $TEST_TASKS"
echo "Project root: $PROJECT_ROOT"
echo "Output root: $OUT_ROOT"
echo ""
echo "NOTE: This is a quick test with only 3 tasks and reduced epochs."
echo "      For full experiment, use: ablation_single_modality.sh"
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
  --out-dir "$COMPARISON_DIR" \
  --top-k 10

echo ""
echo "=========================================="
echo "Quick Test Complete!"
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
echo "  - Stats: $COMPARISON_DIR/ablation_stats.json"
echo "  - Visualization: $COMPARISON_DIR/ablation_comparison.png"
echo ""
echo "If test looks good, run full experiment with:"
echo "  bash scripts/ablation_single_modality.sh"
echo ""
