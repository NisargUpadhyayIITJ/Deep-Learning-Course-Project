#!/bin/bash
# DiT4SR Training with WandB Monitoring
# This script launches training with comprehensive logging

# Set GPU (modify as needed)
export CUDA_VISIBLE_DEVICES=0

# WandB configuration (optional - uncomment and set your key)
# export WANDB_API_KEY="your_key_here"
# export WANDB_PROJECT="dit4sr-training"

# Training configuration
PRETRAINED_MODEL="preset/models/stable-diffusion-3.5-medium"
OUTPUT_DIR="./experiments/dit4sr_wandb"
TRAIN_DATA="preset/datasets/train_datasets/merge_train"

# Optional: Resume from checkpoint
# RESUME_ARG="--resume_from_checkpoint latest"
RESUME_ARG="--resume_from_checkpoint latest"

accelerate launch train/train_dit4sr_wandb.py \
    --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
    --output_dir="$OUTPUT_DIR" \
    --root_folders="$TRAIN_DATA" \
    --report_to="wandb" \
    --tracker_project_name="dit4sr-training" \
    --mixed_precision="fp16" \
    --learning_rate=5e-5 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --null_text_ratio=0.2 \
    --dataloader_num_workers=0 \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=5 \
    --num_train_epochs=1000 \
    --log_every_n_steps=10 \
    --log_grad_norm \
    --seed=42 \
    $RESUME_ARG
