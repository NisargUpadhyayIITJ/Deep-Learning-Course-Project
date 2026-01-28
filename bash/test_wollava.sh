CUDA_VISIBLE_DEVICES=0,2 python test/test_wollava.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
--transformer_model_name_or_path="experiments/dit4sr/checkpoint-6200/transformer" \
--image_path preset/datasets/test_datasets/RealLQ250/lq \
--output_dir results/ \
--prompt_path preset/prompts/RealLQ250