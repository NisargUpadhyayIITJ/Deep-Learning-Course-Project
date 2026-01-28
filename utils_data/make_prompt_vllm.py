"""
make_prompt_vllm.py

This script provides the same functionality as make_prompt.py but uses vLLM 
for efficient batch inference instead of sequential HuggingFace inference.

Key differences from make_prompt.py:
1. Uses vLLM's LLM class with batch inference for significantly faster processing
2. Processes all images in batches rather than one at a time
3. Supports configurable batch sizes for memory management
"""

import torch
from PIL import Image
import os
from tqdm import tqdm
import re
import sys
import argparse

sys.path.append(os.getcwd())

# Default model path - use HuggingFace-format LLaVA for vLLM compatibility
# Original liuhaotian format (llava-v1.5-13b) is NOT supported by vLLM
DEFAULT_MODEL_PATH = "llava-hf/llava-1.5-13b-hf"

# Try to import vLLM
try:
    from vllm import LLM, SamplingParams
    from vllm.multimodal import MultiModalDataBuiltins
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please install it with: pip install vllm"
    )


def remove_focus_sentences(text):
    """
    Remove sentences containing prohibited words related to focus, blur, etc.
    Uses regex to split by sentence-ending punctuation while preserving them.
    """
    prohibited_words = [
        'focus', 'focal', 'prominent', 'close-up', 'black and white', 
        'blur', 'depth', 'dense', 'locate', 'position'
    ]
    parts = re.split(r'([.?!])', text)
    
    filtered_sentences = []
    i = 0
    while i < len(parts):
        sentence = parts[i]
        punctuation = parts[i+1] if (i+1 < len(parts)) else ''
        full_sentence = sentence + punctuation
        
        full_sentence_lower = full_sentence.lower()
        skip = False
        for word in prohibited_words:
            if word.lower() in full_sentence_lower:
                skip = True
                break
        
        if not skip:
            filtered_sentences.append(full_sentence)
        
        i += 2
    
    return "".join(filtered_sentences).strip()


def get_prompt_template():
    """
    Returns the prompt template for image captioning.
    This matches the prompt used in the original LLavaAgent.
    """
    prompt = (
        "Please describe the actual objects in the image in a very detailed manner. "
        "Please do not include descriptions related to the focus and bokeh of this image. "
        "Please do not include descriptions like the background is blurred."
    )
    return prompt


def create_vllm_engine(model_path, tensor_parallel_size=1, gpu_memory_utilization=0.9):
    """
    Create a vLLM engine for LLaVA model inference.
    
    Args:
        model_path: Path to the LLaVA model (must be HuggingFace-compatible format)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
    
    Returns:
        vLLM LLM engine
    
    Note:
        vLLM requires the HuggingFace-format LLaVA models (e.g., llava-hf/llava-1.5-13b-hf)
        not the original liuhaotian format (llava-v1.5-13b).
    """
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
        # Limit number of multimodal items per prompt
        limit_mm_per_prompt={"image": 1},
    )
    return llm


def prepare_batch_inputs(image_paths, prompt):
    """
    Prepare batch inputs for vLLM inference.
    
    Args:
        image_paths: List of paths to images
        prompt: Text prompt for image captioning
    
    Returns:
        List of prompt strings and list of multi-modal data
    """
    prompts = []
    multi_modal_data_list = []
    
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Format prompt with image token for LLaVA
            # vLLM LLaVA uses <image> placeholder
            formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            
            prompts.append(formatted_prompt)
            multi_modal_data_list.append({"image": image})
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
    
    return prompts, multi_modal_data_list


def batch_inference(llm, prompts, multi_modal_data_list, sampling_params):
    """
    Perform batch inference using vLLM.
    
    Args:
        llm: vLLM LLM engine
        prompts: List of formatted prompts
        multi_modal_data_list: List of multi-modal data dicts
        sampling_params: vLLM SamplingParams
    
    Returns:
        List of generated text outputs
    """
    # Create inputs with multi-modal data
    inputs = []
    for prompt, mm_data in zip(prompts, multi_modal_data_list):
        inputs.append({
            "prompt": prompt,
            "multi_modal_data": mm_data,
        })
    
    # Run batch inference
    outputs = llm.generate(inputs, sampling_params)
    
    # Extract text from outputs
    results = []
    for output in outputs:
        generated_text = output.outputs[0].text
        # Clean up the output
        generated_text = generated_text.strip()
        results.append(generated_text)
    
    return results


def process_images_batch(llm, image_paths, prompt, sampling_params, batch_size=8):
    """
    Process images in batches for memory efficiency.
    
    Args:
        llm: vLLM LLM engine
        image_paths: List of all image paths to process
        prompt: Text prompt for captioning
        sampling_params: vLLM SamplingParams
        batch_size: Number of images per batch
    
    Returns:
        Dict mapping image paths to generated captions
    """
    results = {}
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + batch_size]
        
        # Prepare batch inputs
        prompts, mm_data_list = prepare_batch_inputs(batch_paths, prompt)
        
        if not prompts:
            continue
        
        # Run batch inference
        batch_outputs = batch_inference(llm, prompts, mm_data_list, sampling_params)
        
        # Map results back to image paths
        valid_idx = 0
        for img_path in batch_paths:
            if valid_idx < len(batch_outputs):
                # Apply post-processing to remove focus sentences
                caption = remove_focus_sentences(batch_outputs[valid_idx])
                results[img_path] = caption
                valid_idx += 1
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate image captions using vLLM batch inference"
    )
    parser.add_argument(
        "--img_dir", 
        type=str, 
        default='preset/datasets/train_datasets/training_for',
        help='Directory containing images to process'
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default='preset/datasets/train_datasets/training_for',
        help='Directory to save generated prompts/captions'
    )
    parser.add_argument(
        "--stop_num", 
        type=int, 
        default=-1,
        help='Stop after processing this many images (-1 for all)'
    )
    parser.add_argument(
        "--start_num", 
        type=int, 
        default=0,
        help='Start processing from this image index'
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help='Number of images to process in each batch'
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help='Number of GPUs for tensor parallelism'
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help='Fraction of GPU memory to use (0.0-1.0)'
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help='Sampling temperature'
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.7,
        help='Top-p (nucleus) sampling parameter'
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help='Path to LLaVA model (must be HuggingFace-format, e.g., llava-hf/llava-1.5-13b-hf)'
    )
    args = parser.parse_args()

    # Setup directories
    img_folder = args.img_dir
    prompt_save_folder = args.save_dir
    os.makedirs(prompt_save_folder, exist_ok=True)

    # Get list of images to process
    img_name_list = os.listdir(img_folder)
    # img_name_list = img_name_list[args.start_num:args.stop_num if args.stop_num > 0 else None]

    # Filter out already processed images
    images_to_process = []
    for img_name in img_name_list:
        save_path = os.path.join(prompt_save_folder, img_name.replace('png', 'txt'))
        if not os.path.exists(save_path):
            images_to_process.append(os.path.join(img_folder, img_name))

    if not images_to_process:
        print("All images have already been processed!")
        return

    print(f"Found {len(images_to_process)} images to process")

    # Initialize vLLM engine
    print("Initializing vLLM engine...")
    print(f"Using model: {args.model_path}")
    llm = create_vllm_engine(
        args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Get the prompt
    prompt = get_prompt_template()

    # Process images in batches
    print("Processing images with vLLM batch inference...")
    results = process_images_batch(
        llm, 
        images_to_process, 
        prompt, 
        sampling_params, 
        batch_size=args.batch_size
    )

    # Save results
    print("Saving captions...")
    for img_path, caption in results.items():
        img_name = os.path.basename(img_path)
        save_path = os.path.join(prompt_save_folder, img_name.replace('png', 'txt'))
        with open(save_path, 'w', encoding="utf-8") as f:
            f.write(caption)

    print(f"Done! Processed {len(results)} images.")


if __name__ == "__main__":
    main()
