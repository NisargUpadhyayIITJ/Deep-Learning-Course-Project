CUDA_VISIBLE_DEVICES=0,3 python test/test_wllava.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
--transformer_model_name_or_path="preset/models/dit4sr_q" \
--image_path preset/datasets/test_datasets/DrealSR_CenterCrop/test_LR \
--output_dir results/ \