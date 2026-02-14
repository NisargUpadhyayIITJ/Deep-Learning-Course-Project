"""
append_ocr_to_prompts.py

Uses dots.ocr (served via vLLM) to extract OCR text from GT images and
appends it to existing LLaVA-generated prompt .txt files.

This enriches the text conditioning for DiT4SR with actual OCR content,
improving text restoration quality during super-resolution training.

Requirements:
    - dots.ocr vLLM server running (e.g., `vllm serve rednote-hilab/dots.ocr --trust-remote-code`)
    - openai Python package (`pip install openai`)
    - PIL/Pillow

Usage:
    python utils_data/append_ocr_to_prompts.py \
        --img_dir preset/datasets/novelty_train_dataset/SA_Text/gt \
        --prompt_dir preset/datasets/novelty_train_dataset/SA_Text/prompt \
        --port 8000 \
        --model_name rednote-hilab/dots.ocr
"""

import os
import sys
import base64
import argparse
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.getcwd())

# Sentinel marker used to detect already-processed prompts
OCR_SEPARATOR = " OCR Text: "


def pil_image_to_base64(image, fmt="PNG"):
    """Convert a PIL image to a base64 data URI string."""
    buffered = BytesIO()
    image.save(buffered, format=fmt)
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def ocr_single_image(image_path, client, model_name, temperature, top_p, max_tokens):
    """
    Send a single image to the dots.ocr vLLM server and return extracted text.

    Returns:
        str or None: The extracted OCR text, or None on failure.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Failed to open image {image_path}: {e}")
        return None

    b64_uri = pil_image_to_base64(image)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": b64_uri},
                },
                {
                    "type": "text",
                    "text": "<|img|><|imgpad|><|endofimg|>Extract the text content from this image.",
                },
            ],
        }
    ]

    try:
        response = client.chat.completions.create(
            messages=messages,
            model=model_name,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] OCR request failed for {image_path}: {e}")
        return None


def process_single(img_name, img_dir, prompt_dir, client, model_name, temperature, top_p, max_tokens):
    """
    Process a single image: run OCR and append result to its prompt file.

    Returns:
        tuple: (img_name, status_str)
    """
    basename = os.path.splitext(img_name)[0]
    prompt_path = os.path.join(prompt_dir, basename + ".txt")
    img_path = os.path.join(img_dir, img_name)

    # Check prompt file exists
    if not os.path.exists(prompt_path):
        return (img_name, "skipped: no prompt file")

    # Read existing prompt
    with open(prompt_path, "r", encoding="utf-8") as f:
        existing_prompt = f.read().strip()

    # Skip if already processed
    if OCR_SEPARATOR in existing_prompt:
        return (img_name, "skipped: already has OCR")

    # Run OCR
    ocr_text = ocr_single_image(img_path, client, model_name, temperature, top_p, max_tokens)

    if ocr_text is None or ocr_text.strip() == "":
        return (img_name, "skipped: no OCR text extracted")

    # Clean OCR text: collapse to single line, strip
    ocr_text_clean = " ".join(ocr_text.strip().split())

    # Append OCR text to existing prompt
    updated_prompt = existing_prompt + OCR_SEPARATOR + '"' + ocr_text_clean + '"'

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(updated_prompt)

    return (img_name, "done")


def main():
    parser = argparse.ArgumentParser(
        description="Append OCR-extracted text to DiT4SR prompt files using dots.ocr vLLM server"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Directory containing GT images (e.g., .../SA_Text/gt)",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        required=True,
        help="Directory containing prompt .txt files (e.g., .../SA_Text/prompt)",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="http",
        choices=["http", "https"],
    )
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model_name",
        type=str,
        default="rednote-hilab/dots.ocr",
        help="Model name as served by vLLM",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Max tokens for OCR output",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=16,
        help="Number of concurrent threads for vLLM requests",
    )
    parser.add_argument(
        "--stop_num",
        type=int,
        default=-1,
        help="Process only this many images (-1 for all)",
    )
    parser.add_argument(
        "--start_num",
        type=int,
        default=0,
        help="Start processing from this image index",
    )
    args = parser.parse_args()

    # Import OpenAI client
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")

    # Create OpenAI client pointing to vLLM server
    addr = f"{args.protocol}://{args.ip}:{args.port}/v1"
    client = OpenAI(
        api_key=os.environ.get("API_KEY", "0"),
        base_url=addr,
    )

    # Gather image files
    supported_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    all_images = sorted(
        [f for f in os.listdir(args.img_dir) if os.path.splitext(f)[1].lower() in supported_exts]
    )

    # Apply start/stop slicing
    end = args.stop_num if args.stop_num > 0 else None
    images_to_process = all_images[args.start_num:end]

    print(f"Total images found: {len(all_images)}")
    print(f"Images to process (after slicing): {len(images_to_process)}")
    print(f"vLLM server: {addr}")
    print(f"Model: {args.model_name}")
    print(f"Threads: {args.num_threads}")

    # Process with thread pool
    stats = {"done": 0, "skipped": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {
            executor.submit(
                process_single,
                img_name,
                args.img_dir,
                args.prompt_dir,
                client,
                args.model_name,
                args.temperature,
                args.top_p,
                args.max_tokens,
            ): img_name
            for img_name in images_to_process
        }

        with tqdm(total=len(futures), desc="Appending OCR text") as pbar:
            for future in as_completed(futures):
                img_name, status = future.result()
                if status == "done":
                    stats["done"] += 1
                elif "skipped" in status:
                    stats["skipped"] += 1
                else:
                    stats["error"] += 1
                pbar.set_postfix(done=stats["done"], skip=stats["skipped"], err=stats["error"])
                pbar.update(1)

    print(f"\nFinished! Done: {stats['done']}, Skipped: {stats['skipped']}, Errors: {stats['error']}")


if __name__ == "__main__":
    main()
