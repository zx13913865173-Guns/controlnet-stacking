#!/usr/bin/env python
"""
Run A-only multi-level ablation.
4 conditions × 5 levels × 20 prompts = 400 images.
B absent; A varies over {0.1, 0.3, 0.5, 0.7, 0.9}.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation import MultiControlNetGenerator, prepare_condition_images
from src.config import GenerationConfig, CONTROLNET_MODELS_SDXL


def load_prompts(n: int = 20) -> list:
    """Load 20 prompts for the ablation."""
    return [f"a scene with varied content, variation {i+1}" for i in range(n)]


def load_reference_images(cond_type: str, n: int = 20) -> list:
    from PIL import Image
    ref_dir = Path(f"references/{cond_type}")
    if ref_dir.exists():
        from PIL import Image
        images = []
        for i in range(n):
            p = ref_dir / f"ref_{i:03d}.png"
            images.append(Image.open(p).convert("RGB") if p.exists()
                         else Image.new("RGB", (1024, 1024), (128, 128, 128)))
        return images
    return [Image.new("RGB", (1024, 1024), (128, 128, 128)) for _ in range(n)]


def main():
    config = GenerationConfig()
    generator = MultiControlNetGenerator(config=config)
    output_base = Path("data/a_only_ablation")
    output_base.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(20)
    conditions = ["canny", "depth", "openpose", "scribble"]

    for cond_type in conditions:
        print(f"\nA-only ablation for condition: {cond_type}")
        refs = load_reference_images(cond_type, 20)
        cond_images = prepare_condition_images(refs, cond_type)

        cond_dir = output_base / cond_type
        generator.generate_a_only_ablation(
            prompts=prompts,
            cond_images=cond_images,
            cond_type=cond_type,
            a_levels=config.b_weight_levels,
            output_dir=cond_dir,
        )

    # Independent A = 0.7 fixed-A cross-check (Canny+Depth only, 100 images)
    print("\nA=0.7 fixed-A cross-check (Canny+Depth)...")
    refs = load_reference_images("canny_depth", 20)
    canny_imgs = prepare_condition_images(refs, "canny")
    depth_imgs = prepare_condition_images(refs, "depth")

    crosscheck_dir = output_base / "A0.7_crosscheck"
    generator.reload_pipeline_for_pair("depth", "canny")
    generator.generate_fixed_a_series(
        prompts=prompts,
        cond_a_images=depth_imgs,
        cond_b_images=canny_imgs,
        cond_a_type="depth",
        cond_b_type="canny",
        w_a=0.7,
        output_dir=crosscheck_dir,
    )

    print("\nA-only ablation complete.")


if __name__ == "__main__":
    main()
