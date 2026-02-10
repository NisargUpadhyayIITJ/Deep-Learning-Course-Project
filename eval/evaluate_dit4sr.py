#!/usr/bin/env python
"""
DiT4SR Evaluation Script

Computes image quality metrics on super-resolved images.

Metrics:
- LPIPS (lower is better) - requires ground truth
- MUSIQ (higher is better) - no-reference
- MANIQA (higher is better) - no-reference  
- ClipIQA (higher is better) - no-reference
- LIQE (higher is better) - no-reference

Usage:
    python eval/evaluate_dit4sr.py \
        --sr_dir results/sample00 \
        --dataset DrealSR \
        --output_csv results/metrics.csv

    # Evaluate on all datasets
    python eval/evaluate_dit4sr.py \
        --sr_dir results/sample00 \
        --dataset all \
        --output_csv results/metrics.csv
"""

import os
import sys
import glob
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.dataset_config import (
    get_dataset_config, 
    get_all_dataset_names,
    DATASET_CONFIGS
)
from eval.metrics_utils import IQAMetrics, format_results_table


def get_image_pairs(
    sr_dir: str,
    lq_dir: str,
    gt_dir: Optional[str] = None
) -> Tuple[List[str], List[str], Optional[List[str]]]:
    """
    Match SR images with LQ/GT images by filename.
    
    Args:
        sr_dir: Directory containing super-resolved images
        lq_dir: Directory containing low-quality input images
        gt_dir: Directory containing ground truth images (optional)
    
    Returns:
        Tuple of (sr_paths, lq_paths, gt_paths or None)
    """
    # Get SR images
    sr_patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    sr_images = []
    for pattern in sr_patterns:
        sr_images.extend(glob.glob(os.path.join(sr_dir, pattern)))
    sr_images = sorted(sr_images)
    
    if not sr_images:
        raise ValueError(f"No images found in {sr_dir}")
    
    # Match with LQ/GT by basename
    sr_paths = []
    lq_paths = []
    gt_paths = [] if gt_dir else None
    
    for sr_path in sr_images:
        basename = os.path.splitext(os.path.basename(sr_path))[0]
        
        # Find matching LQ
        lq_candidates = glob.glob(os.path.join(lq_dir, f"{basename}.*"))
        if not lq_candidates:
            print(f"Warning: No LQ match for {basename}, skipping")
            continue
        
        sr_paths.append(sr_path)
        lq_paths.append(lq_candidates[0])
        
        # Find matching GT if available
        if gt_dir:
            gt_candidates = glob.glob(os.path.join(gt_dir, f"{basename}.*"))
            if gt_candidates:
                gt_paths.append(gt_candidates[0])
            else:
                gt_paths.append(None)
    
    return sr_paths, lq_paths, gt_paths


def evaluate_dataset(
    sr_dir: str,
    dataset_name: str,
    metrics_calculator: IQAMetrics,
    max_images: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate SR results on a specific dataset.
    
    Args:
        sr_dir: Directory containing SR results
        dataset_name: Name of the dataset (DrealSR, RealSR, etc.)
        metrics_calculator: IQAMetrics instance
        max_images: Limit number of images to evaluate
        verbose: Print progress
    
    Returns:
        Dictionary of metric name -> mean value
    """
    config = get_dataset_config(dataset_name)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name}")
        print(f"  SR dir: {sr_dir}")
        print(f"  LQ dir: {config.lq_dir}")
        if config.gt_dir:
            print(f"  GT dir: {config.gt_dir}")
        print(f"  Has GT: {config.has_gt}")
        print(f"{'='*60}")
    
    # Get image pairs
    try:
        sr_paths, lq_paths, gt_paths = get_image_pairs(
            sr_dir, 
            config.lq_dir, 
            config.gt_dir if config.has_gt else None
        )
    except ValueError as e:
        print(f"Error: {e}")
        return {}
    
    # Limit images if specified
    if max_images is not None and max_images < len(sr_paths):
        sr_paths = sr_paths[:max_images]
        lq_paths = lq_paths[:max_images]
        if gt_paths:
            gt_paths = gt_paths[:max_images]
    
    if verbose:
        print(f"Found {len(sr_paths)} image pairs")
    
    # Compute metrics
    _, mean_results = metrics_calculator.compute_batch(
        sr_paths,
        gt_paths if config.has_gt else None,
        show_progress=verbose
    )
    
    if verbose:
        print(f"\nResults for {dataset_name}:")
        for metric, value in mean_results.items():
            arrow = "↓" if metric == "lpips" else "↑"
            print(f"  {metric.upper()}{arrow}: {value:.4f}")
    
    return mean_results


def save_results(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    format: str = 'csv'
):
    """Save results to file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    if format == 'csv':
        import csv
        
        # Get all metrics
        all_metrics = set()
        for dataset_results in results.values():
            all_metrics.update(dataset_results.keys())
        all_metrics = sorted(all_metrics)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Dataset'] + [m.upper() for m in all_metrics])
            
            for dataset, metrics in results.items():
                row = [dataset]
                for m in all_metrics:
                    if m in metrics:
                        row.append(f"{metrics[m]:.4f}")
                    else:
                        row.append("N/A")
                writer.writerow(row)
        
        print(f"\nResults saved to {output_path}")
    
    elif format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="DiT4SR Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate on DrealSR dataset
    python eval/evaluate_dit4sr.py --sr_dir results/sample00 --dataset DrealSR

    # Evaluate on all datasets
    python eval/evaluate_dit4sr.py --sr_dir results/sample00 --dataset all

    # Quick test with limited images
    python eval/evaluate_dit4sr.py --sr_dir results/sample00 --dataset DrealSR --max_images 5
        """
    )
    
    parser.add_argument(
        "--sr_dir",
        type=str,
        default=None,
        help="Directory containing super-resolved images (use with --dataset for single dataset)"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Base directory containing per-dataset folders (e.g., results/eval_latest)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=get_all_dataset_names() + ["all"],
        help="Dataset to evaluate on (DrealSR, RealSR, RealLR200, RealLQ250, or 'all')"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save CSV results"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save JSON results"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit number of images to evaluate (for quick testing)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for metric computation"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Specific metrics to compute (default: all)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.base_dir is None and args.sr_dir is None:
        parser.error("Either --sr_dir or --base_dir must be specified")
    
    # Folder name to dataset name mapping
    FOLDER_TO_DATASET = {
        "DrealSR_CenterCrop": "DrealSR",
        "RealSR_CenterCrop": "RealSR",
        "RealLR200": "RealLR200",
        "RealLQ250": "RealLQ250",
    }
    
    # Determine datasets and their SR directories
    if args.base_dir:
        # Discover datasets from base_dir
        datasets = []
        sr_dirs = {}
        for folder, dataset in FOLDER_TO_DATASET.items():
            sr_path = os.path.join(args.base_dir, folder, "sample00")
            if os.path.isdir(sr_path):
                datasets.append(dataset)
                sr_dirs[dataset] = sr_path
                print(f"Found: {folder} -> {dataset}")
            else:
                print(f"Skipping: {folder} (not found at {sr_path})")
        
        if not datasets:
            print("Error: No dataset folders found in base_dir")
            return
    else:
        # Single sr_dir mode (original behavior)
        if args.dataset == "all":
            datasets = get_all_dataset_names()
        else:
            datasets = [args.dataset]
        sr_dirs = {d: args.sr_dir for d in datasets}
    
    print(f"\nDiT4SR Evaluation")
    print(f"=" * 60)
    if args.base_dir:
        print(f"Base Directory: {args.base_dir}")
    else:
        print(f"SR Directory: {args.sr_dir}")
    print(f"Datasets: {datasets}")
    print(f"Device: {args.device}")
    if args.max_images:
        print(f"Max images per dataset: {args.max_images}")
    print(f"=" * 60)
    
    # Check if any dataset has GT for LPIPS
    any_has_gt = any(DATASET_CONFIGS[d].has_gt for d in datasets)
    
    # Initialize metrics calculator
    metrics_calc = IQAMetrics(
        device=args.device,
        metrics=args.metrics,
        include_lpips=any_has_gt  # Only include LPIPS if some datasets have GT
    )
    
    # Evaluate each dataset
    all_results = {}
    for dataset in datasets:
        config = DATASET_CONFIGS[dataset]
        
        # For datasets without GT, skip LPIPS
        if not config.has_gt and 'lpips' in metrics_calc.models:
            # Temporarily remove LPIPS model
            lpips_model = metrics_calc.models.pop('lpips', None)
        else:
            lpips_model = None
        
        try:
            # Get the SR directory for this dataset
            dataset_sr_dir = sr_dirs[dataset]
            results = evaluate_dataset(
                dataset_sr_dir,
                dataset,
                metrics_calc,
                max_images=args.max_images,
                verbose=not args.quiet
            )
            all_results[dataset] = results
        except Exception as e:
            print(f"Error evaluating {dataset}: {e}")
            all_results[dataset] = {}
        
        # Restore LPIPS model if it was removed
        if lpips_model is not None:
            metrics_calc.models['lpips'] = lpips_model
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(format_results_table(all_results, datasets))
    
    # Save results
    if args.output_csv:
        save_results(all_results, args.output_csv, format='csv')
    
    if args.output_json:
        save_results(all_results, args.output_json, format='json')
    
    # Also save with timestamp if no output specified
    if not args.output_csv and not args.output_json:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output = f"eval_results_{timestamp}.csv"
        save_results(all_results, default_output, format='csv')


if __name__ == "__main__":
    main()
