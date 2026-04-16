"""Compatibility wrapper around the extracted image generation engine."""

from __future__ import annotations

import os
from typing import Any, Optional

from src.core.image_gen import ImageGenerator
from src.core.pipeline_manager import PipelineManager


def generate_image_wrapper(
    prompt,
    negative_prompt,
    height,
    width,
    steps,
    seed,
    guidance,
    device,
    model_choice,
    input_images,
    downscale_factor,
    img2img_strength,
    lora_file,
    lora_strength,
    enable_multi_character,
    character_input_folder,
    character_description,
    enable_faceswap,
    faceswap_source_image,
    faceswap_target_index,
    optimization_profile,
    enable_windows_compile_probe,
    enable_cuda_graphs,
    enable_optional_accelerators,
    enable_pose_preservation,
    pose_detector_type,
    pose_mode,
    controlnet_strength,
    show_pose_skeleton,
    enable_gender_preservation,
    gender_strength,
    enable_prompt_upsampling,
    enable_klein_anatomy_fix,
    *character_references,
    pipeline_manager: Optional[PipelineManager] = None,
    image_generator: Optional[ImageGenerator] = None,
    progress=None,
):
    """Delegate image generation to the extracted engine without importing app.py."""
    _ = character_input_folder

    manager = pipeline_manager or PipelineManager(os.getcwd())
    generator = image_generator or ImageGenerator(manager)
    return generator.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        steps=steps,
        seed=seed,
        guidance=guidance,
        device=device,
        model_choice=model_choice,
        input_images=input_images,
        downscale_factor=downscale_factor,
        img2img_strength=img2img_strength,
        lora_file=lora_file,
        lora_strength=lora_strength,
        enable_multi_character=enable_multi_character,
        character_references=list(character_references),
        character_description=character_description,
        enable_faceswap=enable_faceswap,
        faceswap_source_image=faceswap_source_image,
        faceswap_target_index=faceswap_target_index,
        enable_pose_preservation=enable_pose_preservation,
        pose_detector_type=pose_detector_type,
        pose_mode=pose_mode,
        controlnet_strength=controlnet_strength,
        show_pose_skeleton=show_pose_skeleton,
        enable_gender_preservation=enable_gender_preservation,
        gender_strength=gender_strength,
        enable_prompt_upsampling=enable_prompt_upsampling,
        enable_klein_anatomy_fix=enable_klein_anatomy_fix,
        optimization_profile=optimization_profile,
        enable_windows_compile_probe=enable_windows_compile_probe,
        enable_cuda_graphs=enable_cuda_graphs,
        enable_optional_accelerators=enable_optional_accelerators,
        progress_callback=progress,
    )
