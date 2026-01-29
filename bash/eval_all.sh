#!/bin/bash
# DiT4SR Evaluation Script
# Evaluates SR results on all test datasets using IQA metrics

# Default values
SR_DIR="${1:-results/sample00}"
DATASET="${2:-all}"
OUTPUT_CSV="${3:-results/metrics.csv}"

echo "=========================================="
echo "DiT4SR Evaluation"
echo "=========================================="
echo "SR Directory: $SR_DIR"
echo "Dataset(s): $DATASET"
echo "Output: $OUTPUT_CSV"
echo "=========================================="

python3 eval/evaluate_dit4sr.py \
    --sr_dir "$SR_DIR" \
    --dataset "$DATASET" \
    --output_csv "$OUTPUT_CSV" \
    --device cuda

echo ""
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_CSV"
