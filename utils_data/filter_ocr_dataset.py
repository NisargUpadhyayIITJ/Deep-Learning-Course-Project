"""
filter_ocr_dataset.py

Filters the SA_Text dataset to keep only images whose prompt files
contain non-empty OCR text (i.e., have " OCR Text: " followed by content).

Creates a NEW filtered dataset by symlinking files from the original dataset,
keeping all subdirectories (gt, lr, latent_hr, latent_lr, prompt, sr_bicubic)
in sync.

Usage:
    python utils_data/filter_ocr_dataset.py \
        --src_dir preset/datasets/novelty_train_dataset/SA_Text \
        --dst_dir preset/datasets/novelty_train_dataset/SA_Text_OCR
"""

import os
import shutil
import argparse
from tqdm import tqdm

OCR_SEPARATOR = " OCR Text: "
SUBDIRS = ["gt", "lr", "latent_hr", "latent_lr", "prompt", "sr_bicubic",
           "prompt_embeds", "pooled_prompt_embeds"]


def has_nonempty_ocr(prompt_path):
    """Check if a prompt file contains non-empty OCR text."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()
    if OCR_SEPARATOR not in content:
        return False
    ocr_part = content.split(OCR_SEPARATOR, 1)[1].strip()
    # Check it's not just empty quotes
    cleaned = ocr_part.strip('"').strip()
    return len(cleaned) > 0


def main():
    parser = argparse.ArgumentParser(
        description="Filter dataset to keep only images with non-empty OCR text"
    )
    parser.add_argument(
        "--src_dir", type=str, required=True,
        help="Source dataset directory (e.g., .../SA_Text)"
    )
    parser.add_argument(
        "--dst_dir", type=str, required=True,
        help="Destination directory for filtered dataset (e.g., .../SA_Text_OCR)"
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of symlinking (uses more disk space)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only report counts, don't create files"
    )
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src_dir)
    dst_dir = os.path.abspath(args.dst_dir)
    prompt_dir = os.path.join(src_dir, "prompt")

    if not os.path.isdir(prompt_dir):
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")

    # Find all prompt files with non-empty OCR text
    prompt_files = sorted([f for f in os.listdir(prompt_dir) if f.endswith(".txt")])
    print(f"Total prompt files: {len(prompt_files)}")

    # Filter
    basenames_with_ocr = []
    basenames_without_ocr = []
    basenames_no_separator = []

    for pf in tqdm(prompt_files, desc="Scanning prompts"):
        basename = os.path.splitext(pf)[0]
        prompt_path = os.path.join(prompt_dir, pf)
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()
        if OCR_SEPARATOR not in content:
            basenames_no_separator.append(basename)
        elif has_nonempty_ocr(prompt_path):
            basenames_with_ocr.append(basename)
        else:
            basenames_without_ocr.append(basename)

    print(f"\nResults:")
    print(f"  With OCR text:      {len(basenames_with_ocr)}")
    print(f"  Empty OCR text:     {len(basenames_without_ocr)}")
    print(f"  No OCR separator:   {len(basenames_no_separator)} (not yet processed)")
    print(f"  Total kept:         {len(basenames_with_ocr)}")

    if args.dry_run:
        print("\n[DRY RUN] No files created.")
        return

    # Create filtered dataset
    # Detect which subdirs exist in source
    existing_subdirs = [d for d in SUBDIRS if os.path.isdir(os.path.join(src_dir, d))]
    print(f"\nSubdirectories to sync: {existing_subdirs}")

    for subdir in existing_subdirs:
        os.makedirs(os.path.join(dst_dir, subdir), exist_ok=True)

    # Determine file extensions per subdir
    subdir_extensions = {}
    for subdir in existing_subdirs:
        subdir_path = os.path.join(src_dir, subdir)
        sample_files = os.listdir(subdir_path)[:10]
        exts = set(os.path.splitext(f)[1] for f in sample_files if f)
        subdir_extensions[subdir] = exts
        print(f"  {subdir}: extensions = {exts}")

    # Link/copy files
    linked = 0
    missing = 0
    for basename in tqdm(basenames_with_ocr, desc="Creating filtered dataset"):
        for subdir in existing_subdirs:
            subdir_src = os.path.join(src_dir, subdir)
            subdir_dst = os.path.join(dst_dir, subdir)

            # Try each known extension
            found = False
            for ext in subdir_extensions[subdir]:
                src_file = os.path.join(subdir_src, basename + ext)
                dst_file = os.path.join(subdir_dst, basename + ext)
                if os.path.exists(src_file):
                    if not os.path.exists(dst_file):
                        if args.copy:
                            shutil.copy2(src_file, dst_file)
                        else:
                            os.symlink(src_file, dst_file)
                    linked += 1
                    found = True
                    break
            if not found:
                missing += 1

    print(f"\nDone! Created filtered dataset at: {dst_dir}")
    print(f"  Files linked/copied: {linked}")
    print(f"  Missing files: {missing}")
    print(f"  Images kept: {len(basenames_with_ocr)} / {len(prompt_files)}")


if __name__ == "__main__":
    main()
