#!/usr/bin/env python
"""
Run cross-model image generation (SD1.5 and Flux.1-dev).
Each: 3 condition pairs × 5 normalized weight ratios × 20 prompts = 300 images,
total 600 images. NORMALIZED-RATIO DESIGN (not fixed-A), so SII is NOT
identifiable by design. These are descriptive compliance coverage only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GenerationConfig, CONDITION_PAIRS


def generate_normalized_ratio(
    model_name: str,
    prompts: list,
    cond_images_pair: list,
    ratios: list,
    output_dir: Path,
):
    """
    Generate with normalized weight ratios (wA + wB = 1.0).
    SD1.5 and Flux do NOT have fixed-A single-B runs; SII not identifiable.
    """
    import torch
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        DPMSolverMultistepScheduler,
        FluxControlNetPipeline,
    )
    from PIL import Image
    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    # ControlNet models for each base
    cn_models = {
        "sd15": {
            "canny": "lllyasviel/sd-controlnet-canny",
            "depth": "lllyasviel/sd-controlnet-depth",
            "openpose": "lllyasviel/sd-controlnet-openpose",
            "scribble": "lllyasviel/sd-controlnet-scribble",
        },
        "flux": {
            "canny": "InstantX/FLUX.1-dev-Controlnet-Canny",
            "depth": "InstantX/FLUX.1-dev-Controlnet-Depth",
        },
    }

    for pair_key, pair_cfg in CONDITION_PAIRS.items():
        if model_name == "flux" and pair_key == "openpose_canny":
            print(f"Skipping {pair_key} for Flux (model not available)")
            continue

        print(f"\n{model_name} - {pair_key}")
        cn_a = ControlNetModel.from_pretrained(
            cn_models[model_name][pair_cfg["a_condition"]], torch_dtype=torch.float16
        )
        cn_b = ControlNetModel.from_pretrained(
            cn_models[model_name][pair_cfg["b_condition"]], torch_dtype=torch.float16
        )

        if model_name == "sd15":
            from diffusers import StableDiffusionControlNetPipeline
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=[cn_a, cn_b],
                torch_dtype=torch.float16,
            )
        else:  # flux
            from diffusers import FluxControlNetPipeline
            pipe = FluxControlNetPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                controlnet=[cn_a, cn_b],
                torch_dtype=torch.float16,
            )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
        )
        pipe.to(device)
        pipe.enable_model_cpu_offload()

        pair_dir = output_dir / pair_key
        pair_dir.mkdir(exist_ok=True)

        # Normalized ratios: wA + wB = 1.0
        for w_b_norm in ratios:
            w_a_norm = 1.0 - w_b_norm
            wb_dir = pair_dir / f"ratio_{w_a_norm:.1f}_{w_b_norm:.1f}"
            wb_dir.mkdir(exist_ok=True)

            for i, prompt in enumerate(tqdm(prompts, desc=f"wA={w_a_norm:.1f} wB={w_b_norm:.1f}")):
                generator = torch.Generator(device=device).manual_seed(42 + i)
                img_a = cond_images_pair[pair_key][0][i]
                img_b = cond_images_pair[pair_key][1][i]
                gen = pipe(
                    prompt=prompt,
                    negative_prompt="low quality, blurry, distorted, ugly, bad anatomy",
                    image=[img_a, img_b],
                    controlnet_conditioning_scale=[w_a_norm, w_b_norm],
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=generator,
                    width=1024, height=1024,
                ).images[0]
                gen.save(wb_dir / f"prompt_{i:03d}.png")

        del pipe
        torch.cuda.empty_cache()


def main():
    config = GenerationConfig()
    normalized_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]  # wB in normalized design
    output_base = Path("data/cross_model")

    prompts = [f"a scene with varied content, cross-model {i+1}" for i in range(20)]

    # Load condition images
    from src.generation import prepare_condition_images
    from PIL import Image
    cond_images_pair = {}
    for pair_key, pair_cfg in CONDITION_PAIRS.items():
        ref_dir = Path(f"references/{pair_key}")
        refs = []
        for i in range(20):
            p = ref_dir / f"ref_{i:03d}.png"
            refs.append(Image.open(p).convert("RGB") if p.exists()
                       else Image.new("RGB", (1024, 1024), (128, 128, 128)))
        cond_a = prepare_condition_images(refs, pair_cfg["a_condition"])
        cond_b = prepare_condition_images(refs, pair_cfg["b_condition"])
        cond_images_pair[pair_key] = (cond_a, cond_b)

    for model_name in ["sd15", "flux"]:
        print(f"\n{'='*60}")
        print(f"Cross-model generation: {model_name}")
        print(f"{'='*60}")
        generate_normalized_ratio(
            model_name=model_name,
            prompts=prompts,
            cond_images_pair=cond_images_pair,
            ratios=normalized_ratios,
            output_dir=output_base / model_name,
        )

    print("\nCross-model generation complete (SII NOT identifiable by design).")


if __name__ == "__main__":
    main()
