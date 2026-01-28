# CUDA_VISIBLE_DEVICES=0 python utils_data/make_prompt.py \
# --img_dir 'preset/datasets/train_datasets/gt'  \
# --save_dir 'preset/datasets/train_datasets/prompt' \
# --stop_num -1 \
# --start_num 0

# vLLM batch inference version
# Note: vLLM requires HuggingFace-format LLaVA (llava-hf/llava-1.5-13b-hf)
# The original liuhaotian format (llava-v1.5-13b) is NOT supported
# source /home/b23cs1037/DN/nisarg/DL-Project/dit_venv/bin/activate

CUDA_VISIBLE_DEVICES=3 python utils_data/make_prompt_vllm.py \
    --img_dir preset/datasets/train_datasets/FFHQ/gt \
    --save_dir preset/datasets/train_datasets/FFHQ/prompt \
    --batch_size 16 \
    --model_path llava-hf/llava-1.5-13b-hf
    # --tensor_parallel_size 2 \