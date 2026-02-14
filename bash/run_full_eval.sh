#!/bin/bash
# DiT4SR Full Pipeline: Inference + Evaluation
# Runs inference on test datasets and computes metrics
# Supports resume: will skip already-processed images automatically

set -e  # Exit on error

# Configuration
CHECKPOINT="${1:-experiments/dit4sr-replication-2/checkpoint-145000}"
PRETRAINED_MODEL="preset/models/stable-diffusion-3.5-medium"

# Use fixed output directory for resume capability
# Pass custom output dir as second argument, or use default "eval_latest"
OUTPUT_BASE="${2:-results/eval_dit4sr-replication-2}"

echo "=========================================="
echo "DiT4SR Full Pipeline (with Resume Support)"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Output Base: $OUTPUT_BASE"
echo "Resume: Enabled (will skip existing images)"
echo "=========================================="

# Datasets to evaluate
DATASETS=("DrealSR_CenterCrop" "RealSR_CenterCrop" "RealLR200" "RealLQ250")
PROMPT_DIRS=("DrealSRVal_crop128" "RealSRVal_crop128" "RealLR200" "RealLQ250")
# DATASETS=("RealLR200")
# PROMPT_DIRS=("RealLR200")

# Run inference on each dataset
for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    PROMPT_DIR="${PROMPT_DIRS[$i]}"
    
    echo ""
    echo "Processing: $DATASET"
    echo "----------------------------------------"
    
    OUTPUT_DIR="$OUTPUT_BASE/$DATASET"
    mkdir -p "$OUTPUT_DIR"
    
    python3 test/test_wollava.py \
        --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
        --transformer_model_name_or_path="$CHECKPOINT" \
        --image_path "preset/datasets/test_datasets/$DATASET/lq" \
        --output_dir "$OUTPUT_DIR" \
        --prompt_path "preset/prompts/$PROMPT_DIR" \
        --mixed_precision "fp16"
    
    echo "✓ Inference complete for $DATASET"
done

echo ""
echo "=========================================="
echo "Running Evaluation"
echo "=========================================="

# Run evaluation
python3 eval/evaluate_dit4sr.py \
    --sr_dir "$OUTPUT_BASE" \
    --dataset all \
    --output_csv "$OUTPUT_BASE/metrics.csv"

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "Results: $OUTPUT_BASE/metrics.csv"
echo "=========================================="
