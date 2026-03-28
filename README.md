# DiT4SR Course Project: Reproduction, Custom Data Training, and OCR-Augmented Prompting

This repository is a deep learning course project built on top of the DiT4SR paper:

> **DiT4SR: Taming Diffusion Transformer for Real-World Image Super-Resolution**  
> arXiv: https://arxiv.org/abs/2503.23580

The project has three tracks:

1. Reproducing the base DiT4SR training and evaluation pipeline.
2. Training the same architecture on a custom merged dataset.
3. Extending the method with OCR-aware prompt conditioning for text-heavy image restoration.

Unlike the original upstream repository, this repo is documented as a project workspace: it contains experiment scripts, report assets, Hugging Face upload helpers, OCR tooling, and evaluation utilities used for the course submission.

## Links

- Paper: https://arxiv.org/abs/2503.23580
- GitHub repo: https://github.com/NisargUpadhyayIITJ/Deep-Learning-Course-Project
- Reproduction W&B: https://wandb.ai/b23cs1075-indian-institute-of-technology-/dit4sr-training
- Custom dataset W&B: https://wandb.ai/b23cs1075-indian-institute-of-technology-/dit4sr-custom-dataset
- OCR novelty W&B: https://wandb.ai/b23cs1075-indian-institute-of-technology-/dit4sr-novelty
- Hugging Face repo: https://huggingface.co/datasets/NisargUpadhyay/ImageSuperResolution

## Project Summary

This project keeps the core DiT4SR idea intact: use an SD3-based diffusion transformer for real-world super-resolution, with explicit low-resolution conditioning inside the transformer.

On top of that, this repo adds:

- a full reproduction workflow for the DiT4SR training pipeline,
- a second training track on a smaller custom dataset mixture,
- an OCR-aware prompt appender using `dots.ocr`,
- a filtered OCR-focused training subset,
- a RealText evaluation workflow for text-containing images,
- a second Gradio demo that exposes OCR-aware prompting,
- utilities to publish checkpoints and datasets to Hugging Face.

## What Is In This Repository

### Core model and pipeline

- `model_dit4sr/`: DiT4SR transformer modifications and attention logic.
- `pipelines/pipeline_dit4sr.py`: SD3-based DiT4SR inference pipeline.
- `train/train_dit4sr.py`: base training script.
- `train/train_dit4sr_wandb.py`: training script with richer W&B logging.
- `test/test_wllava.py`: inference with on-the-fly LLaVA caption generation.
- `test/test_wollava.py`: inference using pre-generated prompt files.

### Data preparation

- `utils_data/make_paired_data.py`: create paired GT/LR data.
- `utils_data/make_prompt.py`: sequential prompt generation.
- `utils_data/make_prompt_vllm.py`: faster batch prompt generation with vLLM.
- `utils_data/make_latents.py`: create SD3 latent caches.
- `utils_data/make_embedding.py`: create SD3 prompt embedding caches.
- `utils_data/append_ocr_to_prompts.py`: append OCR text to prompts with `dots.ocr`.
- `utils_data/filter_ocr_dataset.py`: keep only samples with non-empty OCR text.

### Evaluation and reporting

- `eval/evaluate_dit4sr.py`: compute LPIPS, MUSIQ, MANIQA, CLIPIQA, and LIQE.
- `eval/dataset_config.py`: evaluation dataset path mapping.
- `bash/run_full_eval.sh`: inference plus benchmark evaluation.
- `bash/run_realtext_eval.sh`: RealText evaluation helper.
- `report/`: LaTeX report, generated figures, and report PDF.

### Demos and tooling

- `gradio_dit4sr.py`: base DiT4SR demo.
- `gradio_dit4sr_ocr.py`: OCR-aware DiT4SR demo.
- `utils/upload_experiments_to_hf.py`: upload experiment checkpoints to Hugging Face model repos.
- `utils/upload_dataset_subset_to_hf.py`: upload paired dataset subsets to a Hugging Face dataset repo.
- `utils/upload_folder_tree_to_hf.py`: upload arbitrary folder trees such as evaluation assets and results.

### Vendored dependencies used by the project

- `llava/`: local LLaVA code used for prompt generation.
- `basicsr/`: utility code inherited from the restoration stack used by DiT4SR.

## Environment Setup

Create the main Conda environment:

```bash
conda env create -f environment.yaml --name dit4sr
conda activate dit4sr
```

If Conda complains about the machine-specific `prefix:` entry near the end of `environment.yaml`, remove that line locally and rerun the command above.

The main environment file covers the core DiT4SR training and inference stack. A few optional workflows require extra packages that are not pinned in `environment.yaml`:

```bash
pip install openai pyiqa
```

Install `vllm` only if you want fast batch prompt generation or a `dots.ocr` server:

```bash
pip install vllm
```

## External Assets You Need Locally

Heavy model weights and datasets are not committed to the repo. The scripts expect you to place them in local workspace directories such as `preset/`, `llava_ckpt/`, and `experiments/`.

### Required model assets

- Stable Diffusion 3.5 Medium under `preset/models/stable-diffusion-3.5-medium`
- LLaVA CLIP weights under `llava_ckpt/clip-vit-large-patch14-336`
- LLaVA model weights under `llava_ckpt/llava-v1.5-13b`
- DiT4SR checkpoints either under `preset/models/...` or under experiment checkpoint folders such as `experiments/<run>/checkpoint-*/transformer`
- null prompt embeddings for training caches:
  - `NULL_prompt_embeds.pt`
  - `NULL_pooled_prompt_embeds.pt`

`CKPT_PTH.py` currently points to:

```python
LLAVA_CLIP_PATH = 'llava_ckpt/clip-vit-large-patch14-336'
LLAVA_MODEL_PATH = 'llava_ckpt/llava-v1.5-13b'
```

Update those paths if your local layout is different.

## Expected Data Layout

The training scripts use a cached SD3-latent dataset structure. A prepared training root typically looks like this:

```text
<train_root>/
├── gt/
├── lr/ or sr_bicubic/
├── prompt/
├── prompt_embeds/
├── pooled_prompt_embeds/
├── latent_hr/
└── latent_lr/
```

For evaluation, the repo expects prompt folders under `preset/prompts/` and benchmark datasets under `preset/datasets/test_datasets/`.

The evaluation code currently recognizes:

- `DrealSR`
- `RealSR`
- `RealLR200`
- `RealLQ250`
- `RealText`

## Project Datasets Used In This Repo

The course project uses three training tracks:

### 1. Reproduction track

- training root: `preset/datasets/train_datasets/merge_train`
- source mixture: DIV2K, DIV8K, FFHQ, Flickr2K, Flickr8K, NKUSR8K
- local merged size used in the report: `23,141` samples

### 2. Custom dataset track

- training root: `preset/datasets/custom_train_datasets/merge_train`
- source mixture: `ISRDataset`, `ISRUnsplash`, `unsplashLite5K`
- local merged size used in the report: `14,314` samples

### 3. OCR novelty track

- source root: `preset/datasets/novelty_train_dataset/SA-Text`
- OCR-filtered root: `preset/datasets/novelty_train_dataset/SA_Text_OCR` or similar filtered copy
- local OCR-focused size used in the report/W&B logs: about `23.7k` samples

## End-to-End Workflows

### 1. Prepare training data

The helper scripts under `bash_data/` show the intended preprocessing order:

```bash
# 1) make LR/GT pairs
bash bash_data/make_pairs.sh

# 2) generate prompts
bash bash_data/make_prompt.sh

# 3) compute latents
bash bash_data/make_latent.sh

# 4) compute SD3 prompt embeddings
bash bash_data/make_embedding.sh
```

Notes:

- These scripts are examples, not universal launchers. Several of them contain hard-coded local paths and should be edited before use.
- `bash_data/make_prompt.sh` currently points at the RealText novelty test set and uses `utils_data/make_prompt_vllm.py`.
- `bash_data/make_pairs.sh`, `bash_data/make_latent.sh`, and `bash_data/make_embedding.sh` currently point to the novelty track.

### 2. Train the reproduction model

The default training launcher is:

```bash
bash bash/train.sh
```

That script runs:

```bash
accelerate launch train/train_dit4sr.py \
  --pretrained_model_name_or_path preset/models/stable-diffusion-3.5-medium \
  --output_dir ./experiments/dit4sr \
  --root_folders preset/datasets/train_datasets/merge_train \
  --mixed_precision fp16 \
  --learning_rate 5e-5 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --null_text_ratio 0.2 \
  --checkpointing_steps 1000 \
  --num_train_epochs 1000 \
  --checkpoints_total_limit 5 \
  --resume_from_checkpoint latest
```

To run on a different dataset, keep the same command and change `--output_dir` and `--root_folders`.

### 3. Train the OCR novelty model with W&B

The W&B launcher in this repo is:

```bash
bash bash/train_wandb.sh
```

It is preconfigured for the novelty run:

- output dir: `./experiments/dit4sr_novel`
- training data: `preset/datasets/novelty_train_dataset/SA_Text_OCR/`
- tracker project: `dit4sr-novelty`

Before running it, check the hard-coded `CUDA_VISIBLE_DEVICES`, `OUTPUT_DIR`, and `TRAIN_DATA` values in the script.

### 4. Generate inference results

There are two inference entry points.

### A. Inference with LLaVA-generated prompts

Use `test/test_wllava.py` if you want prompts generated automatically from images:

```bash
bash bash/test_wllava.sh
```

This mode depends on local LLaVA checkpoints configured in `CKPT_PTH.py`.

### B. Inference with pre-generated prompt files

Use `test/test_wollava.py` if you already have prompt `.txt` files:

```bash
bash bash/test_wollava.sh
```

This is the lighter-weight evaluation path used for benchmark runs and RealText experiments.

### 5. Evaluate benchmark results

Run evaluation on a single dataset directory:

```bash
python eval/evaluate_dit4sr.py \
  --sr_dir results/eval_replication_50eps/DrealSR_CenterCrop/sample00 \
  --dataset DrealSR \
  --output_csv results/eval_replication_50eps/metrics.csv
```

Run evaluation on a benchmark result folder containing multiple datasets:

```bash
python eval/evaluate_dit4sr.py \
  --base_dir results/eval_replication_50eps \
  --output_csv results/eval_replication_50eps/metrics.csv
```

Or use the shell wrappers:

```bash
bash bash/eval_all.sh results/eval_replication_50eps
```

```bash
bash bash/run_full_eval.sh \
  experiments/dit4sr-replication/checkpoint-150000 \
  results/eval_replication_50eps
```

### 6. OCR-aware novelty workflow

### Append OCR text to prompts

Start a `dots.ocr` server first, then append OCR text to prompt files:

```bash
python utils_data/append_ocr_to_prompts.py \
  --img_dir preset/datasets/novelty_train_dataset/SA-Text/lr \
  --prompt_dir preset/datasets/novelty_train_dataset/SA-Text/prompt \
  --port 8000 \
  --model_name rednote-hilab/dots.ocr
```

The appended format is:

```text
<base caption> OCR Text: "recognized text here"
```

### Filter to OCR-positive samples

```bash
python utils_data/filter_ocr_dataset.py \
  --src_dir preset/datasets/novelty_train_dataset/SA-Text \
  --dst_dir preset/datasets/novelty_train_dataset/SA_Text_OCR
```

### Evaluate on RealText

The helper script is:

```bash
bash bash/run_realtext_eval.sh \
  experiments/dit4sr-replication/checkpoint-150000 \
  results/eval_realtext_baseline
```

Important note:

- `bash/run_realtext_eval.sh` currently has prompt generation and inference steps commented out.
- In its current state, it computes metrics for an already-generated RealText output folder.
- If you want a full end-to-end RealText pipeline, uncomment or restore Steps 1 to 3 in that script.

## Gradio Demos

### Base demo

```bash
python gradio_dit4sr.py \
  --pretrained_model_name_or_path preset/models/stable-diffusion-3.5-medium \
  --transformer_model_name_or_path preset/models/dit4sr_f
```

### OCR-aware demo

```bash
python gradio_dit4sr_ocr.py \
  --pretrained_model_name_or_path preset/models/stable-diffusion-3.5-medium \
  --transformer_model_name_or_path /path/to/transformer/checkpoint
```

The OCR demo additionally requires:

- `pip install openai`
- a `dots.ocr` server reachable at the configured host and port

## Current Project Results

These are the main results currently committed under `results/` and summarized in the course report.

### Benchmark averages from local evaluation

| Model                          | CLIPIQA |   LIQE |  LPIPS | MANIQA |   MUSIQ |
| ------------------------------ | ------: | -----: | -----: | -----: | ------: |
| Reproduction checkpoint-150k   |  0.7024 | 4.2563 | 0.3789 | 0.4779 | 69.3684 |
| Custom dataset checkpoint-190k |  0.7062 | 4.1765 | 0.3730 | 0.4743 | 68.9963 |

### RealText OCR ablation

| Method          | CLIPIQA |   LIQE |  LPIPS | MANIQA |   MUSIQ |
| --------------- | ------: | -----: | -----: | -----: | ------: |
| Baseline prompt |  0.6727 | 4.4907 | 0.3810 | 0.5339 | 68.2013 |
| OCR prompt      |  0.6797 | 4.4217 | 0.3644 | 0.5433 | 66.6734 |

Interpretation:

- OCR-aware prompting improved `LPIPS`, `CLIPIQA`, and `MANIQA` on RealText.
- The custom dataset run slightly improved average `LPIPS` over the reproduction checkpoint.
- The OCR novelty behaves like a targeted text-restoration enhancement rather than a universal boost on all no-reference metrics.

## Hugging Face Assets

### Model repos

- `NisargUpadhyay/ImageSuperResolution-replication`
- `NisargUpadhyay/ImageSuperResolution-custom-dataset`
- `NisargUpadhyay/ImageSuperResolution-novel-text-conditioned`

### Dataset repo

- `NisargUpadhyay/ImageSuperResolution`

## Acknowledgements

This project builds on:

- the DiT4SR paper and official implementation idea,
- Stable Diffusion 3.5 Medium,
- LLaVA for prompt generation,
- `dots.ocr` for OCR-conditioned prompting,
- `pyiqa` for evaluation metrics.

## License

This repository keeps the original project license:

- [LICENSE](LICENSE)

## Citation

If you use the DiT4SR method in academic work, please cite the original paper:

```bibtex
@article{duan2025dit4sr,
  title   = {DiT4SR: Taming Diffusion Transformer for Real-World Image Super-Resolution},
  author  = {Duan, Zheng-Peng and Zhang, Jiawei and Jin, Xin and Zhang, Ziheng and Xiong, Zheng and Zou, Dongqing and Ren, Jimmy S. and Guo, Chunle and Li, Chongyi},
  journal = {arXiv preprint arXiv:2503.23580},
  year    = {2025}
}
```
