"""
Dataset configuration for DiT4SR evaluation.
Defines paths and metadata for all test datasets.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict

# Base paths - adjust these according to your setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESET_DIR = os.path.join(PROJECT_ROOT, "preset")
DATASETS_DIR = os.path.join(PRESET_DIR, "datasets", "test_datasets")
PROMPTS_DIR = os.path.join(PRESET_DIR, "prompts")


@dataclass
class DatasetConfig:
    """Configuration for a test dataset."""
    name: str
    lq_dir: str
    gt_dir: Optional[str]  # None if no ground truth available
    prompt_dir: str
    has_gt: bool
    
    def validate(self) -> bool:
        """Check if paths exist."""
        if not os.path.exists(self.lq_dir):
            return False
        if self.has_gt and self.gt_dir and not os.path.exists(self.gt_dir):
            return False
        if not os.path.exists(self.prompt_dir):
            return False
        return True


# Dataset configurations
DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "DrealSR": DatasetConfig(
        name="DrealSR",
        lq_dir=os.path.join(DATASETS_DIR, "DrealSR_CenterCrop", "lq"),
        gt_dir=os.path.join(DATASETS_DIR, "DrealSR_CenterCrop", "gt"),
        prompt_dir=os.path.join(PROMPTS_DIR, "DrealSRVal_crop128"),
        has_gt=True,
    ),
    "RealSR": DatasetConfig(
        name="RealSR",
        lq_dir=os.path.join(DATASETS_DIR, "RealSR_CenterCrop", "lq"),
        gt_dir=os.path.join(DATASETS_DIR, "RealSR_CenterCrop", "gt"),
        prompt_dir=os.path.join(PROMPTS_DIR, "RealSRVal_crop128"),
        has_gt=True,
    ),
    "RealLR200": DatasetConfig(
        name="RealLR200",
        lq_dir=os.path.join(DATASETS_DIR, "RealLR200", "lq"),
        gt_dir=None,
        prompt_dir=os.path.join(PROMPTS_DIR, "RealLR200"),
        has_gt=False,
    ),
    "RealLQ250": DatasetConfig(
        name="RealLQ250",
        lq_dir=os.path.join(DATASETS_DIR, "RealLQ250", "lq"),
        gt_dir=None,
        prompt_dir=os.path.join(PROMPTS_DIR, "RealLQ250"),
        has_gt=False,
    ),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get configuration for a specific dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_name]


def get_all_dataset_names() -> list:
    """Get list of all available dataset names."""
    return list(DATASET_CONFIGS.keys())


def get_datasets_with_gt() -> list:
    """Get list of datasets that have ground truth images."""
    return [name for name, config in DATASET_CONFIGS.items() if config.has_gt]


def get_datasets_without_gt() -> list:
    """Get list of datasets without ground truth images."""
    return [name for name, config in DATASET_CONFIGS.items() if not config.has_gt]
