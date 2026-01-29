"""
Metrics utilities for DiT4SR evaluation.
Provides wrappers around pyiqa metrics library.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
import os

try:
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    PYIQA_AVAILABLE = False
    print("Warning: pyiqa not installed. Install with: pip install pyiqa")


class IQAMetrics:
    """
    Image Quality Assessment metrics calculator using pyiqa.
    
    Supports:
    - LPIPS (lower is better, requires GT)
    - MUSIQ (higher is better, no-reference)
    - MANIQA (higher is better, no-reference)
    - ClipIQA (higher is better, no-reference)
    - LIQE (higher is better, no-reference)
    """
    
    # Metric configurations
    METRIC_CONFIGS = {
        'lpips': {
            'pyiqa_name': 'lpips',
            'requires_gt': True,
            'higher_better': False,
            'description': 'Learned Perceptual Image Patch Similarity'
        },
        'musiq': {
            'pyiqa_name': 'musiq',
            'requires_gt': False,
            'higher_better': True,
            'description': 'Multi-scale Image Quality Transformer'
        },
        'maniqa': {
            'pyiqa_name': 'maniqa',
            'requires_gt': False,
            'higher_better': True,
            'description': 'Multi-dimension Attention Network for No-Reference IQA'
        },
        'clipiqa': {
            'pyiqa_name': 'clipiqa',
            'requires_gt': False,
            'higher_better': True,
            'description': 'CLIP-based Image Quality Assessment'
        },
        'liqe': {
            'pyiqa_name': 'liqe',
            'requires_gt': False,
            'higher_better': True,
            'description': 'Language Image Quality Evaluator'
        },
    }
    
    def __init__(
        self, 
        device: str = 'cuda',
        metrics: Optional[List[str]] = None,
        include_lpips: bool = True
    ):
        """
        Initialize metrics calculators.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
            metrics: List of metric names to compute. If None, uses all.
            include_lpips: Whether to include LPIPS (requires GT images)
        """
        if not PYIQA_AVAILABLE:
            raise ImportError("pyiqa is required. Install with: pip install pyiqa")
        
        self.device = device
        
        # Determine which metrics to use
        if metrics is None:
            metrics = list(self.METRIC_CONFIGS.keys())
        
        if not include_lpips and 'lpips' in metrics:
            metrics.remove('lpips')
        
        self.metric_names = metrics
        self.models = {}
        
        # Load metric models
        print("Loading IQA models...")
        for metric_name in self.metric_names:
            if metric_name not in self.METRIC_CONFIGS:
                print(f"Warning: Unknown metric '{metric_name}', skipping.")
                continue
            
            config = self.METRIC_CONFIGS[metric_name]
            try:
                self.models[metric_name] = pyiqa.create_metric(
                    config['pyiqa_name'], 
                    device=device
                )
                print(f"  Loaded {metric_name}")
            except Exception as e:
                print(f"  Failed to load {metric_name}: {e}")
        
        print(f"Loaded {len(self.models)} metrics: {list(self.models.keys())}")
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load image and convert to tensor."""
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)
    
    def compute_single(
        self, 
        sr_image: str, 
        gt_image: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute metrics for a single SR image.
        
        Args:
            sr_image: Path to super-resolved image
            gt_image: Path to ground truth image (optional, required for LPIPS)
        
        Returns:
            Dictionary of metric name -> value
        """
        results = {}
        
        sr_tensor = self._load_image(sr_image)
        gt_tensor = self._load_image(gt_image) if gt_image else None
        
        for metric_name, model in self.models.items():
            config = self.METRIC_CONFIGS[metric_name]
            
            try:
                if config['requires_gt']:
                    if gt_tensor is None:
                        # Skip LPIPS if no GT available
                        continue
                    score = model(sr_tensor, gt_tensor)
                else:
                    score = model(sr_tensor)
                
                results[metric_name] = score.item()
            except Exception as e:
                print(f"Error computing {metric_name} for {sr_image}: {e}")
                results[metric_name] = float('nan')
        
        return results
    
    def compute_batch(
        self,
        sr_images: List[str],
        gt_images: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
        """
        Compute metrics for multiple images.
        
        Args:
            sr_images: List of SR image paths
            gt_images: List of GT image paths (optional)
            show_progress: Show progress bar
        
        Returns:
            Tuple of (per_image_results, mean_results)
        """
        if gt_images is not None:
            assert len(sr_images) == len(gt_images), \
                "Number of SR and GT images must match"
        
        all_results = {name: [] for name in self.models.keys()}
        
        iterator = enumerate(sr_images)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Computing metrics")
            except ImportError:
                pass
        
        for i, sr_path in iterator:
            gt_path = gt_images[i] if gt_images else None
            
            results = self.compute_single(sr_path, gt_path)
            
            for metric_name, value in results.items():
                all_results[metric_name].append(value)
        
        # Compute means
        mean_results = {}
        for metric_name, values in all_results.items():
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                mean_results[metric_name] = np.mean(valid_values)
            else:
                mean_results[metric_name] = float('nan')
        
        return all_results, mean_results
    
    @staticmethod
    def get_metric_info() -> Dict[str, Dict]:
        """Get information about available metrics."""
        return IQAMetrics.METRIC_CONFIGS.copy()
    
    @staticmethod
    def is_higher_better(metric_name: str) -> bool:
        """Check if higher values are better for a metric."""
        if metric_name in IQAMetrics.METRIC_CONFIGS:
            return IQAMetrics.METRIC_CONFIGS[metric_name]['higher_better']
        raise ValueError(f"Unknown metric: {metric_name}")


def format_results_table(
    results: Dict[str, Dict[str, float]],
    datasets: List[str]
) -> str:
    """
    Format results as a markdown table matching paper format.
    
    Args:
        results: {dataset_name: {metric_name: value}}
        datasets: List of dataset names in order
    
    Returns:
        Formatted markdown table string
    """
    metrics = ['lpips', 'musiq', 'maniqa', 'clipiqa', 'liqe']
    metric_arrows = {'lpips': '↓', 'musiq': '↑', 'maniqa': '↑', 'clipiqa': '↑', 'liqe': '↑'}
    
    # Header
    header = "| Dataset |"
    for m in metrics:
        header += f" {m.upper()}{metric_arrows.get(m, '')} |"
    
    separator = "|---------|" + "--------|" * len(metrics)
    
    # Data rows
    rows = []
    for dataset in datasets:
        if dataset not in results:
            continue
        row = f"| {dataset} |"
        for m in metrics:
            if m in results[dataset]:
                value = results[dataset][m]
                if np.isnan(value):
                    row += " N/A |"
                elif m == 'lpips':
                    row += f" {value:.3f} |"
                elif m in ['musiq']:
                    row += f" {value:.3f} |"
                else:
                    row += f" {value:.3f} |"
            else:
                row += " - |"
        rows.append(row)
    
    return "\n".join([header, separator] + rows)
