#!/usr/bin/env python
"""
Run CFG 15 forward-direction probe (Canny → Depth).
1 pair × 5 B levels × 20 prompts = 100 images
+ CFG15-specific Depth-only weight-1.0 baseline (20 images)
= 120 images total.
Uses CFG15-specific per-prompt baselines; NO comparison with CFG7.5.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation import MultiControlNetGenerator, prepare_condition_images
from src.config import GenerationConfig


def load_prompts(n: int = 20) -> list:
    return [f"a scene with varied content, CFG15 probe {i+1}" for i in range(n)]


def load_reference_images(n: int = 20) -> list:
    from PIL import Image
    ref_dir = Path("references/canny_depth")
    if ref_dir.exists():
        images = []
        for i in range(n):
            p = ref_dir / f"ref_{i:03d}.png"
            images.append(Image.open(p).convert("RGB") if p.exists()
                         else Image.new("RGB", (1024, 1024), (128, 128, 128)))
        return images
    return [Image.new("RGB", (1024, 1024), (128, 128, 128)) for _ in range(n)]


def main():
    config = GenerationConfig()
    config.cfg_scale = 15.0  # CFG 15
    generator = MultiControlNetGenerator(config=config)
    output_base = Path("data/cfg15_probe")
    output_base.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(20)
    refs = load_reference_images(20)

    # Forward direction only: A=Depth (fixed 0.5), B=Canny varying
    depth_imgs = prepare_condition_images(refs, "depth")
    canny_imgs = prepare_condition_images(refs, "canny")

    forward_dir = output_base / "canny_to_depth"
    generator.generate_fixed_a_series(
        prompts=prompts,
        cond_a_images=depth_imgs,
        cond_b_images=canny_imgs,
        cond_a_type="depth",
        cond_b_type="canny",
        w_a=0.5,
        output_dir=forward_dir,
    )

    # CFG15-specific Depth-only weight-1.0 baseline
    baseline_dir = output_base / "depth_baseline_cfg15"
    generator.generate_single_condition_baseline(
        prompts=prompts,
        cond_images=depth_imgs,
        cond_type="depth",
        weight=1.0,
        output_dir=baseline_dir,
    )

    print("\nCFG15 forward probe complete.")
    print(f"Expected signed slope ≈ -0.52, SII ≈ 0.52 (per paper).")


if __name__ == "__main__":
    main()
