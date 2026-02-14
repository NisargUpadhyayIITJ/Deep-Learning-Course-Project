# Append OCR text to SA_Text prompts using dots.ocr vLLM server
#
# Prerequisites: dots.ocr vLLM server must be running on the specified port
# Start it with:
#   source /home/b23cs1037/DN/nisarg/DL-Project/ocr_venv/bin/activate
#   CUDA_VISIBLE_DEVICES=1 vllm serve rednote-hilab/dots.ocr \
#       --trust-remote-code --gpu-memory-utilization 0.45

python utils_data/append_ocr_to_prompts.py \
    --img_dir preset/datasets/novelty_train_dataset/SA_Text/gt \
    --prompt_dir preset/datasets/novelty_train_dataset/SA_Text/prompt \
    --port 8000 \
    --model_name rednote-hilab/dots.ocr \
    --num_threads 16 \
    --temperature 0.1 \
    --max_tokens 4096
