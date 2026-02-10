#!/bin/bash
# DiT4SR Evaluation Script - Per-Dataset Processing
# Evaluates SR results for each dataset folder individually

# Default values
BASE_DIR="${1:-results/eval_latest}"
OUTPUT_CSV="${2:-${BASE_DIR}/metrics.csv}"

echo "=========================================="
echo "DiT4SR Evaluation - Per Dataset"
echo "=========================================="
echo "Base Directory: $BASE_DIR"
echo "Output: $OUTPUT_CSV"
echo "=========================================="

# Use the Python script's --base_dir mode which handles:
# - Discovering dataset folders (DrealSR_CenterCrop, RealSR_CenterCrop, etc.)
# - Mapping folder names to dataset names
# - Running evaluation for each dataset
# - Writing consistent CSV with all columns

python3 eval/evaluate_dit4sr.py \
    --base_dir "$BASE_DIR" \
    --output_csv "$OUTPUT_CSV" \
    --device cuda

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_CSV"
echo "=========================================="
