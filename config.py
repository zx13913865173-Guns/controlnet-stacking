"""
Configuration for the multi-Condition ControlNet stacking experiment.
All parameters match those specified in the paper.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class GenerationConfig:
    """Unified generation parameters."""
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    cfg_scale: float = 7.5
    num_inference_steps: int = 30
    sampler: str = "dpmpp_2m_karras"
    resolution: Tuple[int, int] = (1024, 1024)
    seed: int = 42
    negative_prompt: str = "low quality, blurry, distorted, ugly, bad anatomy"
    fixed_a_weight: float = 0.5
    b_weight_levels: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])


@dataclass
class MetricConfig:
    """Metric computation parameters."""
    # Canny Edge IoU
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200
    # Depth SSIM
    depth_model: str = "Intel/dpt-large"  # MiDaS DPT-Large
    ssim_data_range: float = 1.0
    ssim_win_size: int = 7
    ssim_sigma: float = 1.5
    # OpenPose keypoint matching
    pose_threshold_ratio: float = 0.05  # 5% of image diagonal
    pose_confidence: float = 0.1
    # Bootstrap
    n_bootstrap: int = 1000
    bootstrap_alpha: float = 0.05


# Condition pairs studied in the main experiment
CONDITION_PAIRS = {
    "canny_depth": {
        "a_condition": "depth",
        "b_condition": "canny",
        "a_metric": "depth_ssim",
        "b_metric": "canny_iou",
        "prompt_source": "coco",
    },
    "openpose_canny": {
        "a_condition": "canny",
        "b_condition": "openpose",
        "a_metric": "canny_iou",
        "b_metric": "openpose_keypoint",
        "prompt_source": "human36m",
    },
    "scribble_depth": {
        "a_condition": "depth",
        "b_condition": "scribble",
        "a_metric": "depth_ssim",
        "b_metric": "scribble_iou",
        "prompt_source": "coco",
    },
}

# Content categories for prompts (20 each, 100 total)
PROMPT_CATEGORIES = {
    "landscape": 20,
    "portrait": 20,
    "architecture": 20,
    "still_life": 20,
    "street_scene": 20,
}

# ControlNet model IDs for SDXL
CONTROLNET_MODELS_SDXL = {
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
    "openpose": "xinsir/controlnet-openpose-sdxl-1.0",
    "scribble": "diffusers/controlnet-scribble-sdxl-1.0",
}

# Prespecified heuristic defaults (only measured cells)
HEURISTIC_DEFAULTS = {
    ("canny_depth", 7.5, "sdxl"): {"canny": 0.38, "depth": 0.62},
}

# Mean single-condition baselines at weight 1.0 (descriptive only)
MEAN_BASELINES_CFG75 = {
    "canny": 0.91,
    "depth": 0.84,
    "openpose": 0.88,
    "scribble": 0.85,
}
