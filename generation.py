"""
Image generation pipeline for multi-Condition ControlNet stacking.
Implements fixed-A non-normalized weight variation as described in the paper.
"""
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from PIL import Image
from tqdm import tqdm

from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    FluxControlNetPipeline,
)

from .config import GenerationConfig, CONTROLNET_MODELS_SDXL


class MultiControlNetGenerator:
    """Generator supporting dual ControlNet stacking with fixed-A, varying-B."""

    def __init__(self, config: Optional[GenerationConfig] = None, device: str = "cuda"):
        self.config = config or GenerationConfig()
        self.device = device
        self.pipe = None
        self.controlnets = {}
        self._load_pipeline()

    def _load_pipeline(self):
        """Load SDXL Base 1.0 with two ControlNet units."""
        print(f"Loading SDXL pipeline from {self.config.model_id}...")
        
        # Default: load canny + depth (most used pair); other CNs loaded lazily
        controlnet_canny = ControlNetModel.from_pretrained(
            CONTROLNET_MODELS_SDXL["canny"], torch_dtype=torch.float16
        )
        controlnet_depth = ControlNetModel.from_pretrained(
            CONTROLNET_MODELS_SDXL["depth"], torch_dtype=torch.float16
        )
        
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.config.model_id,
            controlnet=[controlnet_canny, controlnet_depth],
            torch_dtype=torch.float16,
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload()  # memory efficiency
        self.controlnets = {"canny": controlnet_canny, "depth": controlnet_depth}
        print("Pipeline loaded successfully.")

    def _load_controlnet(self, cond_type: str) -> ControlNetModel:
        """Lazy-load additional ControlNet models."""
        if cond_type not in self.controlnets:
            if cond_type not in CONTROLNET_MODELS_SDXL:
                raise ValueError(f"Unknown condition type: {cond_type}")
            print(f"Loading ControlNet for {cond_type}...")
            self.controlnets[cond_type] = ControlNetModel.from_pretrained(
                CONTROLNET_MODELS_SDXL[cond_type], torch_dtype=torch.float16
            )
        return self.controlnets[cond_type]

    def reload_pipeline_for_pair(self, cond_a: str, cond_b: str):
        """Reload pipeline with the two required ControlNets for a given pair."""
        cn_a = self._load_controlnet(cond_a)
        cn_b = self._load_controlnet(cond_b)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.config.model_id,
            controlnet=[cn_a, cn_b],
            torch_dtype=torch.float16,
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload()

    def generate_single(
        self,
        prompt: str,
        cond_a_image: Image.Image,
        cond_b_image: Image.Image,
        w_a: float,
        w_b: float,
        cond_a_type: str = "depth",
        cond_b_type: str = "canny",
        seed_offset: int = 0,
    ) -> Image.Image:
        """
        Generate a single image with fixed A weight and given B weight (no renormalization).
        Total conditioning weight = w_a + w_b (non-normalized design).
        """
        generator = torch.Generator(device=self.device).manual_seed(
            self.config.seed + seed_offset
        )
        image = self.pipe(
            prompt=prompt,
            negative_prompt=self.config.negative_prompt,
            image=[cond_a_image, cond_b_image],
            controlnet_conditioning_scale=[w_a, w_b],
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.cfg_scale,
            generator=generator,
            width=self.config.resolution[0],
            height=self.config.resolution[1],
        ).images[0]
        return image

    def generate_fixed_a_series(
        self,
        prompts: List[str],
        cond_a_images: List[Image.Image],
        cond_b_images: List[Image.Image],
        cond_a_type: str = "depth",
        cond_b_type: str = "canny",
        w_a: float = 0.5,
        output_dir: Optional[Path] = None,
    ) -> Dict[float, List[Image.Image]]:
        """
        Fixed-A design: hold wA fixed, vary wB over five levels without renormalization.
        Returns dict mapping wB -> list of generated images (one per prompt).
        """
        self.reload_pipeline_for_pair(cond_a_type, cond_b_type)
        results = {w: [] for w in self.config.b_weight_levels}
        
        for w_b in self.config.b_weight_levels:
            wb_dir = output_dir / f"wB_{w_b:.1f}" if output_dir else None
            if wb_dir:
                wb_dir.mkdir(parents=True, exist_ok=True)
            
            for i, (prompt, img_a, img_b) in enumerate(tqdm(
                zip(prompts, cond_a_images, cond_b_images),
                total=len(prompts),
                desc=f"wB={w_b:.1f}",
            )):
                gen_img = self.generate_single(
                    prompt=prompt,
                    cond_a_image=img_a,
                    cond_b_image=img_b,
                    w_a=w_a,
                    w_b=w_b,
                    cond_a_type=cond_a_type,
                    cond_b_type=cond_b_type,
                    seed_offset=i,
                )
                results[w_b].append(gen_img)
                if wb_dir:
                    gen_img.save(wb_dir / f"prompt_{i:03d}.png")
        
        return results

    def generate_single_condition_baseline(
        self,
        prompts: List[str],
        cond_images: List[Image.Image],
        cond_type: str,
        weight: float = 1.0,
        output_dir: Optional[Path] = None,
    ) -> List[Image.Image]:
        """
        Generate single-condition baseline images (weight=1.0).
        These serve as per-prompt denominators b_i for retention ratio R_A,i.
        """
        cn = self._load_controlnet(cond_type)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.config.model_id,
            controlnet=cn,
            torch_dtype=torch.float16,
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload()

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, (prompt, cond_img) in enumerate(tqdm(
            zip(prompts, cond_images),
            total=len(prompts),
            desc=f"Single-condition baseline ({cond_type})",
        )):
            generator = torch.Generator(device=self.device).manual_seed(
                self.config.seed + i
            )
            gen_img = self.pipe(
                prompt=prompt,
                negative_prompt=self.config.negative_prompt,
                image=cond_img,
                controlnet_conditioning_scale=weight,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.cfg_scale,
                generator=generator,
                width=self.config.resolution[0],
                height=self.config.resolution[1],
            ).images[0]
            results.append(gen_img)
            if output_dir:
                gen_img.save(output_dir / f"prompt_{i:03d}.png")

        return results

    def generate_a_only_ablation(
        self,
        prompts: List[str],
        cond_images: List[Image.Image],
        cond_type: str,
        a_levels: List[float] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[float, List[Image.Image]]:
        """
        A-only multi-level ablation: B absent, A varies over {0.1, 0.3, 0.5, 0.7, 0.9}.
        4 conditions × 5 levels × 20 prompts = 400 images total.
        """
        if a_levels is None:
            a_levels = self.config.b_weight_levels  # same five levels
        
        cn = self._load_controlnet(cond_type)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.config.model_id,
            controlnet=cn,
            torch_dtype=torch.float16,
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload()

        results = {w: [] for w in a_levels}
        for w_a in a_levels:
            wa_dir = output_dir / f"wA_{w_a:.1f}" if output_dir else None
            if wa_dir:
                wa_dir.mkdir(parents=True, exist_ok=True)
            
            for i, (prompt, cond_img) in enumerate(tqdm(
                zip(prompts, cond_images),
                total=len(prompts),
                desc=f"A-only wA={w_a:.1f}",
            )):
                generator = torch.Generator(device=self.device).manual_seed(
                    self.config.seed + i
                )
                gen_img = self.pipe(
                    prompt=prompt,
                    negative_prompt=self.config.negative_prompt,
                    image=cond_img,
                    controlnet_conditioning_scale=w_a,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.cfg_scale,
                    generator=generator,
                    width=self.config.resolution[0],
                    height=self.config.resolution[1],
                ).images[0]
                results[w_a].append(gen_img)
                if wa_dir:
                    gen_img.save(wa_dir / f"prompt_{i:03d}.png")

        return results


def prepare_condition_images(
    reference_images: List[Image.Image],
    cond_type: str,
) -> List[Image.Image]:
    """
    Extract/prepare condition maps from reference images.
    Applies the same preprocessing for both generated and reference sides.
    """
    condition_images = []
    
    if cond_type == "canny":
        import cv2
        for img in reference_images:
            img_np = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            condition_images.append(Image.fromarray(edges_rgb))
    
    elif cond_type == "depth":
        from transformers import pipeline
        depth_estimator = pipeline(
            "depth-estimation", model="Intel/dpt-large", device=0
        )
        for img in tqdm(reference_images, desc="Computing depth maps"):
            depth_result = depth_estimator(img)
            depth_map = depth_result["depth"]
            # min-max normalize to [0,1]
            depth_np = np.array(depth_map).astype(np.float32)
            depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
            depth_np = (depth_np * 255).astype(np.uint8)
            condition_images.append(Image.fromarray(depth_np, mode="L").convert("RGB"))
    
    elif cond_type == "openpose":
        from controlnet_aux import OpenposeDetector
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        for img in tqdm(reference_images, desc="Computing OpenPose skeletons"):
            pose_img = openpose(img, hand_and_face=False)
            condition_images.append(pose_img)
    
    elif cond_type == "scribble":
        import cv2
        for img in reference_images:
            img_np = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            # HED-like simplified scribble via Canny + dilation for approximation
            edges = cv2.Canny(gray, 100, 200)
            kernel = np.ones((2, 2), np.uint8)
            scribble = cv2.dilate(edges, kernel, iterations=1)
            scribble_rgb = cv2.cvtColor(scribble, cv2.COLOR_GRAY2RGB)
            condition_images.append(Image.fromarray(scribble_rgb))
    
    else:
        raise ValueError(f"Unknown condition type: {cond_type}")

    return condition_images
