#!/usr/bin/env python
"""
Run the main fixed-A non-normalized experiment.
3 condition pairs × 5 weight ratios × 100 prompts = 1,500 images (main design)
+ 3 pairs × 2 directions × 5 B levels × 20 prompts = 600 images (fixed-A single-B subset with 20 prompts)
"""
import sys
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation import MultiControlNetGenerator, prepare_condition_images
from src.config import GenerationConfig, CONDITION_PAIRS


def load_prompts(n: int = 100) -> list:
    """Load or generate prompts (replace with actual COCO/Human3.6M captions)."""
    categories = {
        "landscape": "a landscape with mountains and a lake",
        "portrait": "a portrait of a person in natural lighting",
        "architecture": "a building with architectural details",
        "still_life": "a still life arrangement on a table",
        "street_scene": "a street scene with people and buildings",
    }
    prompts = []
    per_cat = n // 5
    for cat, base in categories.items():
        for i in range(per_cat):
            prompts.append(f"{base}, variation {i+1}")
    return prompts[:n]


def load_reference_images(pair_key: str, n: int = 100) -> list:
    """Load reference images from COCO/Human3.6M (placeholder - user provides actual images)."""
    from PIL import Image
    ref_dir = Path(f"references/{pair_key}")
    if ref_dir.exists():
        images = []
        for i in range(n):
            img_path = ref_dir / f"ref_{i:03d}.png"
            if img_path.exists():
                images.append(Image.open(img_path).convert("RGB"))
            else:
                # Placeholder white image if reference not found
                images.append(Image.new("RGB", (1024, 1024), (128, 128, 128)))
        return images
    else:
        return [Image.new("RGB", (1024, 1024), (128, 128, 128)) for _ in range(n)]


def main():
    config = GenerationConfig()
    generator = MultiControlNetGenerator(config=config)
    output_base = Path("data/main_experiment")
    output_base.mkdir(parents=True, exist_ok=True)

    prompts_100 = load_prompts(100)

    for pair_key, pair_config in CONDITION_PAIRS.items():
        print(f"\n{'='*60}")
        print(f"Processing pair: {pair_key}")
        print(f"A={pair_config['a_condition']} (fixed at {config.fixed_a_weight}), "
              f"B={pair_config['b_condition']} (varying)")
        print(f"{'='*60}")

        n_prompts = 100
        refs = load_reference_images(pair_key, n_prompts)

        # Prepare condition maps for A and B
        cond_b_images = prepare_condition_images(refs, pair_config["b_condition"])
        cond_a_images = prepare_condition_images(refs, pair_config["a_condition"])

        # Generate fixed-A series: B varies over 5 levels
        pair_dir = output_base / pair_key / "forward"
        generator.generate_fixed_a_series(
            prompts=prompts_100,
            cond_a_images=cond_a_images,
            cond_b_images=cond_b_images,
            cond_a_type=pair_config["a_condition"],
            cond_b_type=pair_config["b_condition"],
            w_a=config.fixed_a_weight,
            output_dir=pair_dir,
        )

        # Also generate reverse direction (swap A and B roles)
        # For reverse: A is now b_condition, B is now a_condition
        pair_dir_rev = output_base / pair_key / "reverse"
        generator.generate_fixed_a_series(
            prompts=prompts_100,
            cond_a_images=cond_b_images,
            cond_b_images=cond_a_images,
            cond_a_type=pair_config["b_condition"],
            cond_b_type=pair_config["a_condition"],
            w_a=config.fixed_a_weight,
            output_dir=pair_dir_rev,
        )

        # Generate single-condition baselines (weight=1.0) for both conditions
        baseline_dir_a = output_base / pair_key / "baseline_A"
        generator.generate_single_condition_baseline(
            prompts=prompts_100[:20],  # 20 prompts for fixed-A single-B analysis
            cond_images=cond_a_images[:20],
            cond_type=pair_config["a_condition"],
            weight=1.0,
            output_dir=baseline_dir_a,
        )
        baseline_dir_b = output_base / pair_key / "baseline_B"
        generator.generate_single_condition_baseline(
            prompts=prompts_100[:20],
            cond_images=cond_b_images[:20],
            cond_type=pair_config["b_condition"],
            weight=1.0,
            output_dir=baseline_dir_b,
        )

    print("\nMain experiment generation complete.")
    print(f"Images saved to: {output_base.resolve()}")


if __name__ == "__main__":
    main()
