#!/usr/bin/env python3
"""
Filter out small images from a dataset.
Images smaller than the minimum size are moved to a separate folder.
"""

import os
import shutil
import argparse
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def get_image_size(image_path):
    """Get image dimensions without loading full image into memory."""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def filter_small_images(
    input_dir: str,
    min_size: int = 512,
    move_to_dir: str = None,
    dry_run: bool = False
):
    """
    Filter out images smaller than min_size.
    
    Args:
        input_dir: Directory containing images
        min_size: Minimum dimension (both width and height must be >= min_size)
        move_to_dir: Directory to move small images to. If None, creates 'small_images' subfolder.
        dry_run: If True, only print what would be done without actually moving files
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: {input_dir} does not exist")
        return
    
    # Default move directory
    if move_to_dir is None:
        move_to_dir = input_path.parent / f"{input_path.name}_small"
    else:
        move_to_dir = Path(move_to_dir)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    
    # Find all images
    images = []
    for ext in image_extensions:
        images.extend(input_path.glob(f"*{ext}"))
        images.extend(input_path.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(images)} images in {input_dir}")
    print(f"Minimum size threshold: {min_size}x{min_size}")
    print(f"Small images will be moved to: {move_to_dir}")
    if dry_run:
        print("DRY RUN - no files will be moved")
    print("-" * 60)
    
    small_images = []
    valid_images = []
    failed_images = []
    
    for img_path in tqdm(images, desc="Checking images"):
        size = get_image_size(img_path)
        
        if size is None:
            failed_images.append(img_path)
            continue
        
        width, height = size
        
        if width < min_size or height < min_size:
            small_images.append((img_path, width, height))
        else:
            valid_images.append((img_path, width, height))
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images scanned: {len(images)}")
    print(f"Valid images (>= {min_size}x{min_size}): {len(valid_images)}")
    print(f"Small images (< {min_size}x{min_size}): {len(small_images)}")
    print(f"Failed to read: {len(failed_images)}")
    
    if small_images:
        print(f"\nSmall images to be moved:")
        for img_path, w, h in small_images[:10]:  # Show first 10
            print(f"  {img_path.name}: {w}x{h}")
        if len(small_images) > 10:
            print(f"  ... and {len(small_images) - 10} more")
    
    if failed_images:
        print(f"\nFailed images:")
        for img_path in failed_images[:5]:
            print(f"  {img_path.name}")
        if len(failed_images) > 5:
            print(f"  ... and {len(failed_images) - 5} more")
    
    # Move small images
    if small_images and not dry_run:
        print(f"\nMoving {len(small_images)} small images...")
        move_to_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path, w, h in tqdm(small_images, desc="Moving"):
            dest = move_to_dir / img_path.name
            shutil.move(str(img_path), str(dest))
        
        print(f"Done! Small images moved to: {move_to_dir}")
    
    # Also move failed images
    if failed_images and not dry_run:
        failed_dir = move_to_dir.parent / f"{input_path.name}_failed"
        failed_dir.mkdir(parents=True, exist_ok=True)
        for img_path in failed_images:
            dest = failed_dir / img_path.name
            shutil.move(str(img_path), str(dest))
        print(f"Failed images moved to: {failed_dir}")
    
    print("\n" + "=" * 60)
    print(f"Remaining valid images in {input_dir}: {len(valid_images)}")
    print("=" * 60)
    
    return {
        'valid': len(valid_images),
        'small': len(small_images),
        'failed': len(failed_images)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter out small images from a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Filter images smaller than 512x512 (default)
    python filter_small_images.py /path/to/dataset
    
    # Filter with custom minimum size
    python filter_small_images.py /path/to/dataset --min_size 256
    
    # Dry run (see what would be done without actually moving)
    python filter_small_images.py /path/to/dataset --dry_run
        """
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing images to filter"
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=512,
        help="Minimum dimension in pixels (default: 512)"
    )
    parser.add_argument(
        "--move_to",
        type=str,
        default=None,
        help="Directory to move small images to (default: <input_dir>_small)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print what would be done, don't actually move files"
    )
    
    args = parser.parse_args()
    
    filter_small_images(
        args.input_dir,
        min_size=args.min_size,
        move_to_dir=args.move_to,
        dry_run=args.dry_run
    )
