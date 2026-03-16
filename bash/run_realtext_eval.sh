#!/bin/bash
# DiT4SR RealText Pipeline: prompt generation + OCR + inference + evaluation
#
# Prerequisites:
# - LLaVA checkpoints configured in CKPT_PTH.py
# - dots.ocr vLLM server running if OCR prompts need to be appended
#
# Usage:
#   bash bash/run_realtext_eval.sh [checkpoint] [output_base]

set -e

CHECKPOINT="${1:-experiments/dit4sr-replication/checkpoint-150000}"
OUTPUT_BASE="${2:-results/eval_realtext_baseline}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-preset/models/stable-diffusion-3.5-medium}"
INPUT_DIR="${INPUT_DIR:-preset/datasets/novelty_test_dataset/RealTextFiltered/lr}"
PROMPT_DIR="${PROMPT_DIR:-preset/prompts/RealTextFiltered}"
DATASET_NAME="RealText"
DATASET_OUTPUT="$OUTPUT_BASE/$DATASET_NAME"
OCR_PORT="${OCR_PORT:-8000}"
OCR_MODEL_NAME="${OCR_MODEL_NAME:-rednote-hilab/dots.ocr}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"

echo "=========================================="
echo "DiT4SR RealText Evaluation Pipeline"
echo "=========================================="
echo "Checkpoint:   $CHECKPOINT"
echo "Input LR Dir: $INPUT_DIR"
echo "Prompt Dir:   $PROMPT_DIR"
echo "Output Base:  $OUTPUT_BASE"
echo "OCR Port:     $OCR_PORT"
echo "=========================================="

# mkdir -p "$PROMPT_DIR"
mkdir -p "$DATASET_OUTPUT"

# echo ""
# echo "=========================================="
# echo "Step 1/4: Generating captions"
# echo "=========================================="
# python3 utils_data/make_prompt.py \
#     --img_dir "$INPUT_DIR" \
#     --save_dir "$PROMPT_DIR"

# echo ""
# echo "=========================================="
# echo "Step 2/4: Appending dots.ocr text"
# echo "=========================================="
# python3 utils_data/append_ocr_to_prompts.py \
#     --img_dir "$INPUT_DIR" \
#     --prompt_dir "$PROMPT_DIR" \
#     --port "$OCR_PORT" \
#     --model_name "$OCR_MODEL_NAME"

# echo ""
# echo "=========================================="
# echo "Step 3/4: Running inference"
# echo "=========================================="
# python3 test/test_wollava.py \
#     --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
#     --transformer_model_name_or_path "$CHECKPOINT" \
#     --image_path "$INPUT_DIR" \
#     --output_dir "$DATASET_OUTPUT" \
#     --prompt_path "$PROMPT_DIR" \
#     --mixed_precision "$MIXED_PRECISION"

echo ""
echo "=========================================="
echo "Step 4/4: Computing metrics"
echo "=========================================="
python3 eval/evaluate_dit4sr.py \
    --sr_dir "$DATASET_OUTPUT/sample00" \
    --dataset "$DATASET_NAME" \
    --output_csv "$OUTPUT_BASE/${DATASET_NAME}_metrics.csv"

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "SR Output: $DATASET_OUTPUT/sample00"
echo "Metrics:   $OUTPUT_BASE/${DATASET_NAME}_metrics.csv"
echo "=========================================="
