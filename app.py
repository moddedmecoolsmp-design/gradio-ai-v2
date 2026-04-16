"""Flux Image Generator - Gradio Web Interface.

Fast image generation on CUDA (optimized for RTX 3070 / CUDA 13).
Supports multiple models:
- Z-Image Turbo (INT8)
- FLUX.2-klein-4B (Int8 quantized)
- FLUX.2-klein-4B (4bit SDNQ)

FLUX.2-klein and Z-Image support image-to-image editing.
"""

import os
import sys
import threading
from typing import Optional

# ---------------------------------------------------------------------------
# Environment & CUDA configuration (must run before torch import)
# ---------------------------------------------------------------------------
import src.config  # noqa: F401 — configures SDNQ env vars, cache paths, CUDA knobs
import torch

from src.config import (
    BASE_DIR,
    CUDA_SPEED_STATUS,
    DEFAULT_OPTIMIZATION_PROFILE,
    DEFAULT_OPTIONAL_ACCELERATORS,
    DEFAULT_WINDOWS_COMPILE_PROBE,
    DEPENDENCY_CHECK_FLAG,
    RUNTIME_CUDA13,
    SKIP_DEPENDENCY_CHECK as SKIP_CHECK,
    read_bool_env,
    read_positive_int_env,
)

from src.runtime_policies import (
    FAST_FLUX_MODEL_CHOICE,
    FAST_RESOLUTION_PRESET,
    LOW_VRAM_FLUX_MODEL_CHOICE,
    choose_image_generation_mode,
    default_enable_windows_compile_probe,
    describe_acceleration_stack,
    get_torch_compile_probe_status,
    is_distilled_model,
    is_flux_model,
    is_sdnq_or_quantized,
    is_windows_3070_fast_profile,
    is_zimage_model,
    resolve_default_flux_model_choice,
    resolve_generation_guidance,
    resolve_model_choice_for_device,
    resolve_optional_accelerators_enabled,
    resolve_optimization_profile,
    resolve_zimage_img2img_steps,
    resolution_preset_to_long_edge,
    should_enable_autocast,
    should_show_edit_controls,
    should_use_attention_slicing,
    should_use_vae_slicing,
    should_use_vae_tiling,
)
from src.image.vlm_prompt_upsampler import upsample_prompt_from_image
from src.utils.device_utils import get_available_devices, get_device_vram_gb
from src.security import clamp_int, clamp_float, load_json_safe, save_json_safe
from src.core.async_batch_integration import calculate_dimensions_from_ratio, clear_batch_processor_cache
from src.utils.common import sanitize_choice
from src.utils.ui_state import UIState
from src.constants import (
    CHARACTER_MANAGER_STATE_FILENAME,
    FAST_FLUX_STATE_MIGRATION_KEY,
    GRADIO_DELETE_CACHE,
    KLEIN_ANATOMY_LORA_URL,
    POSE_MODES,
    POSE_DETECTOR_TYPES,
)
from src.ui.gradio_app import create_ui
from src.core.video_processor import (
    count_processed_video_frames,
    prepare_video_workdirs,
    build_video_status,
    save_processed_video_frame,
    enhance_video_prompt_for_gender,
    resolve_video_tool_path,
    safe_remove_path,
)
from src.core.cuda_graphs import CUDAGraphRunner, get_cuda_graph_runner
from src.startup import (
    enforce_cuda13_runtime_profile,
    ensure_cache_dirs,
    log_provider_telemetry,
    run_dependency_preflight,
)
from src.state import (
    STATE_PATH,
    STATE_IMAGES_DIR,
    STATE_CHAR_REFS_DIR,
    MODELS_DIR,
    LORAS_DIR,
    ensure_state_dirs,
    load_user_state,
    save_user_state,
    save_images_to_dir,
    load_images_from_dir,
    clear_dir,
    save_input_images,
    load_input_images,
)
from src.config import STATE_DIR
from src.utils.dimensions import (
    apply_scale_to_dimensions,
    calculate_dimensions_from_target_size,
    format_downscale_preview,
    get_device_gpu_name,
    get_next_lower_dimensions,
    normalize_downscale_factor,
    parse_downscale_factor,
    resolve_model_dimensions,
    resolve_resolution_preset_for_model,
)

# NOTE: Windows asyncio fix, SDNQ/Triton env, CUDA knobs, and cache paths
# are now configured by src.config (imported above).


GRADIO_CACHE_TTL_SECONDS = read_positive_int_env("UFIG_GRADIO_CACHE_TTL_SECONDS", 24 * 60 * 60)
GRADIO_CACHE_CLEANUP_FREQUENCY_SECONDS = read_positive_int_env(
    "UFIG_GRADIO_CACHE_CLEANUP_FREQUENCY_SECONDS",
    GRADIO_CACHE_TTL_SECONDS
)
GRADIO_DELETE_CACHE_SECONDS = GRADIO_DELETE_CACHE


# Create cache/state directories and log provider telemetry at import time
ensure_cache_dirs()
log_provider_telemetry(
    profile=DEFAULT_OPTIMIZATION_PROFILE,
    cuda_speed_status=CUDA_SPEED_STATUS,
    compile_probe=DEFAULT_WINDOWS_COMPILE_PROBE,
    optional_accelerators=DEFAULT_OPTIONAL_ACCELERATORS,
)

import atexit
import contextlib
import inspect
import json
import logging
import re
import shutil
import tempfile
import time

import gradio as gr
import torch._dynamo
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(_log_dir, "app.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core pipeline / generation imports
# ---------------------------------------------------------------------------
from src.core.pipeline_manager import PipelineManager
from src.core.image_gen import ImageGenerator
from src.core.batch_gen import BatchGenerator
from src.core.async_batch_integration import build_safe_kwargs, run_async_batch_processing
from src.image import lora_zimage
from src.core.video_temporal import (
    build_video_temporal_config,
    build_ffmpeg_audio_trim_command,
    build_ffmpeg_deflicker_command,
    build_ffmpeg_frame_encode_command,
    build_ffmpeg_merge_command,
    prepare_temporal_condition_frame,
    resolve_video_frame_seed,
    resolve_video_generation_mode,
    resolve_video_shot_seed,
    should_reset_temporal_history,
)
try:
    from src.audio.qwen_tts_helper import qwen_handler
except ImportError:
    qwen_handler = None
from src.audio import audio_ui_helpers

# Temp directory override
APP_TEMP_DIR = os.environ.get("UFIG_TEMP_DIR", os.environ.get("GRADIO_TEMP_DIR"))
if APP_TEMP_DIR:
    tempfile.tempdir = APP_TEMP_DIR

KLEIN_ANATOMY_LORA_PATH = os.path.join(LORAS_DIR, "kleinSliderAnatomy.safetensors")



# Dependency checking, state management, and image persistence functions
# have been extracted to src.startup and src.state respectively.


def safe_int_value(value, default: int) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if not text:
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def coerce_float(value, default: float) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default



# normalize_downscale_factor -> moved to src.utils.dimensions


def build_initial_state(available_devices, default_device):
    state_data = load_json_safe(STATE_PATH)
    state = state_data
    persisted_state = UIState.load_payload(state_data)
    fast_flux_state_migrated = bool(state_data.get(FAST_FLUX_STATE_MIGRATION_KEY, False))

    def get_text(value, fallback=""):
        return value if isinstance(value, str) else fallback

    def get_bool(value, fallback=False):
        return bool(value) if value is not None else fallback

    persisted_model_choice = persisted_state.model_choice
    if isinstance(persisted_model_choice, str) and persisted_model_choice.startswith("Z-Image Turbo"):
        persisted_model_choice = "Z-Image Turbo (Int8 - 8GB Safe)"
    model_choice = sanitize_choice(persisted_model_choice, MODEL_CHOICES, MODEL_CHOICES[0])
    preset_choice = sanitize_choice(
        persisted_state.preset_choice,
        ["None"] + list(ANIME_PHOTO_PRESETS.keys()),
        "None",
    )
    resolution_preset = sanitize_choice(
        persisted_state.resolution_preset,
        SINGLE_RESOLUTION_PRESETS,
        SINGLE_RESOLUTION_PRESETS[0],
    )
    batch_resolution_preset = sanitize_choice(
        persisted_state.batch_resolution_preset,
        BATCH_RESOLUTION_PRESETS,
        BATCH_RESOLUTION_PRESETS[0],
    )
    downscale_factor = normalize_downscale_factor(persisted_state.downscale_factor)
    img2img_strength = clamp_float(coerce_float(state.get("img2img_strength"), 0.6), 0.0, 1.0, 0.6)

    has_saved_dimensions = "width" in state_data and "height" in state_data
    height = clamp_int(persisted_state.height, 256, 2048, 512)
    width = clamp_int(persisted_state.width, 256, 2048, 512)
    steps = clamp_int(persisted_state.steps, 1, 50, 4)
    seed = clamp_int(persisted_state.seed, -1, (2**32 - 1), -1)
    guidance_scale = clamp_float(persisted_state.guidance_scale, 0.0, 10.0, 1.0)
    enable_klein_anatomy_fix = get_bool(persisted_state.enable_klein_anatomy_fix, fallback=False)
    lora_strength = clamp_float(persisted_state.lora_strength, 0.0, 2.0, 1.0)

    device = sanitize_choice(persisted_state.device, available_devices, default_device)
    lora_file = persisted_state.lora_file
    if lora_file and not os.path.isfile(str(lora_file)):
        lora_file = None

    input_images = load_input_images(persisted_state.input_image_names)

    if not has_saved_dimensions:
        width, height = resolve_model_dimensions(
            model_choice,
            width=1024,
            height=1024,
            preset=resolution_preset,
            device=device,
        )

    if (
        not fast_flux_state_migrated
        and model_choice == LOW_VRAM_FLUX_MODEL_CHOICE
        and is_windows_fast_flux_device(device)
    ):
        model_choice = resolve_default_flux_model_choice(
            device=device,
            vram_gb=get_device_vram_gb(device),
            gpu_name=get_device_gpu_name(device),
        )
        resolution_preset = resolve_resolution_preset_for_model(model_choice, mode="single", device=device)
        batch_resolution_preset = resolve_resolution_preset_for_model(model_choice, mode="batch", device=device)
        video_resolution_preset = resolve_resolution_preset_for_model(model_choice, mode="video", device=device)
        width, height = resolve_model_dimensions(
            model_choice,
            width=width,
            height=height,
            preset=resolution_preset,
            device=device,
        )
        fast_flux_state_migrated = True
        state_data.update(
            {
                "model_choice": model_choice,
                "resolution_preset": resolution_preset,
                "batch_resolution_preset": batch_resolution_preset,
                "video_resolution_preset": video_resolution_preset,
                "width": width,
                "height": height,
                FAST_FLUX_STATE_MIGRATION_KEY: True,
            }
        )
        save_json_safe(STATE_PATH, state_data)

    # New Features
    enable_multi_character = get_bool(persisted_state.enable_multi_character)
    character_input_folder = get_text(persisted_state.character_input_folder)
    character_description = get_text(persisted_state.character_description)
    enable_faceswap = get_bool(persisted_state.enable_faceswap)
    # faceswap_source_image handled separately if needed, but we'll try to persist it
    faceswap_source_names = persisted_state.faceswap_source_names
    faceswap_source_image = load_images_from_dir(faceswap_source_names, STATE_IMAGES_DIR)
    faceswap_source_image = faceswap_source_image[0] if faceswap_source_image else None

    faceswap_target_index = clamp_int(persisted_state.faceswap_target_index, 0, 10, 0)
    optimization_profile = sanitize_choice(
        persisted_state.optimization_profile,
        ["max_speed", "balanced", "stability"],
        DEFAULT_OPTIMIZATION_PROFILE,
    )
    enable_windows_compile_probe = get_bool(
        persisted_state.enable_windows_compile_probe,
        fallback=DEFAULT_WINDOWS_COMPILE_PROBE,
    )
    enable_optional_accelerators = get_bool(
        persisted_state.enable_optional_accelerators,
        fallback=DEFAULT_OPTIONAL_ACCELERATORS,
    )
    enable_pose_preservation = get_bool(persisted_state.enable_pose_preservation)
    pose_detector_type = sanitize_choice(persisted_state.pose_detector_type, POSE_DETECTOR_TYPES, "dwpose")
    pose_mode = sanitize_choice(persisted_state.pose_mode, POSE_MODES, "Body + Face")
    controlnet_strength = clamp_float(persisted_state.controlnet_strength, 0.0, 1.0, 0.7)
    show_pose_skeleton = get_bool(persisted_state.show_pose_skeleton)
    enable_gender_preservation = get_bool(persisted_state.enable_gender_preservation, fallback=True)
    gender_strength = clamp_float(persisted_state.gender_strength, 0.5, 2.0, 1.0)
    enable_prompt_upsampling = get_bool(persisted_state.enable_prompt_upsampling, fallback=False)

    # Video Processing
    video_output_path = get_text(persisted_state.video_output_path)
    preserve_audio = get_bool(persisted_state.preserve_audio, fallback=True)
    video_resolution_preset = sanitize_choice(
        persisted_state.video_resolution_preset,
        SINGLE_RESOLUTION_PRESETS,
        SINGLE_RESOLUTION_PRESETS[0],
    )

    # Character reference slots
    char_ref_names = persisted_state.character_reference_names
    character_references = load_images_from_dir(char_ref_names, STATE_CHAR_REFS_DIR)
    # Pad to 10
    while len(character_references) < 10:
        character_references.append(None)

    initial_state = {
        "model_choice": model_choice,
        "prompt": get_text(persisted_state.prompt),
        "negative_prompt": get_text(persisted_state.negative_prompt),
        "preset_choice": preset_choice,
        "resolution_preset": resolution_preset,
        "batch_input_folder": get_text(persisted_state.batch_input_folder),
        "batch_output_folder": get_text(persisted_state.batch_output_folder),
        "batch_resolution_preset": batch_resolution_preset,
        "downscale_factor": downscale_factor,
        "img2img_strength": img2img_strength,
        "height": height,
        "width": width,
        "steps": steps,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "enable_klein_anatomy_fix": enable_klein_anatomy_fix,
        "device": device,
        "lora_file": lora_file,
        "lora_strength": lora_strength,
        "enable_multi_character": enable_multi_character,
        "character_input_folder": character_input_folder,
        "character_description": character_description,
        "enable_faceswap": enable_faceswap,
        "faceswap_source_image": faceswap_source_image,
        "faceswap_target_index": faceswap_target_index,
        "optimization_profile": optimization_profile,
        "enable_windows_compile_probe": enable_windows_compile_probe,
        "enable_optional_accelerators": enable_optional_accelerators,
        "enable_pose_preservation": enable_pose_preservation,
        "pose_detector_type": pose_detector_type,
        "pose_mode": pose_mode,
        "controlnet_strength": controlnet_strength,
        "show_pose_skeleton": show_pose_skeleton,
        "enable_gender_preservation": enable_gender_preservation,
        "gender_strength": gender_strength,
        "enable_prompt_upsampling": enable_prompt_upsampling,
        "video_output_path": video_output_path,
        "preserve_audio": preserve_audio,
        "video_resolution_preset": video_resolution_preset,
        "character_references": character_references,
        FAST_FLUX_STATE_MIGRATION_KEY: fast_flux_state_migrated,
    }
    return initial_state, input_images


def persist_ui_state(
    model_choice,
    prompt,
    negative_prompt,
    preset_choice,
    input_images,
    resolution_preset,
    batch_input_folder,
    batch_output_folder,
    batch_resolution_preset,
    downscale_factor,
    img2img_strength,
    height,
    width,
    steps,
    seed,
    guidance_scale,
    enable_klein_anatomy_fix,
    device,
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
    video_output_path,
    preserve_audio,
    video_resolution_preset,
    *character_references
):
    existing_state = load_json_safe(STATE_PATH)
    img2img_strength = coerce_float(img2img_strength, 0.6)
    # Compatibility anchor for legacy substring-based checks:
    # "img2img_strength": float(img2img_strength)
    state = UIState(
        model_choice=model_choice,
        prompt=prompt or "",
        negative_prompt=negative_prompt or "",
        preset_choice=preset_choice,
        resolution_preset=resolution_preset,
        batch_input_folder=batch_input_folder or "",
        batch_output_folder=batch_output_folder or "",
        batch_resolution_preset=batch_resolution_preset,
        downscale_factor=normalize_downscale_factor(downscale_factor),
        img2img_strength=img2img_strength,
        height=safe_int_value(height, 512),
        width=safe_int_value(width, 512),
        steps=safe_int_value(steps, 4),
        seed=safe_int_value(seed, -1),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 0.0,
        enable_klein_anatomy_fix=bool(enable_klein_anatomy_fix),
        device=device,
        lora_file=lora_file if lora_file else None,
        lora_strength=float(lora_strength) if lora_strength is not None else 1.0,
        enable_multi_character=bool(enable_multi_character),
        character_input_folder=character_input_folder or "",
        character_description=character_description or "",
        enable_faceswap=bool(enable_faceswap),
        faceswap_target_index=safe_int_value(faceswap_target_index, 0),
        optimization_profile=optimization_profile or DEFAULT_OPTIMIZATION_PROFILE,
        enable_windows_compile_probe=bool(enable_windows_compile_probe),
        enable_cuda_graphs=bool(enable_cuda_graphs),
        enable_optional_accelerators=bool(enable_optional_accelerators),
        enable_pose_preservation=bool(enable_pose_preservation),
        pose_detector_type=pose_detector_type,
        pose_mode=pose_mode,
        controlnet_strength=float(controlnet_strength) if controlnet_strength is not None else 0.7,
        show_pose_skeleton=bool(show_pose_skeleton),
        enable_gender_preservation=bool(enable_gender_preservation),
        gender_strength=float(gender_strength) if gender_strength is not None else 1.0,
        enable_prompt_upsampling=bool(enable_prompt_upsampling),
        video_output_path=video_output_path or "",
        preserve_audio=bool(preserve_audio),
        video_resolution_preset=video_resolution_preset,
    ).to_dict()
    state["img2img_strength"] = float(img2img_strength)
    compatibility_state = {
        FAST_FLUX_STATE_MIGRATION_KEY: bool(existing_state.get(FAST_FLUX_STATE_MIGRATION_KEY, False))
    }
    state.update(compatibility_state)
    state["input_image_names"] = save_input_images(input_images)

    # Save faceswap source image
    if faceswap_source_image:
        state["faceswap_source_names"] = save_images_to_dir([faceswap_source_image], "fs", STATE_IMAGES_DIR, max_count=1)
    else:
        state["faceswap_source_names"] = []

    # Save character references
    clear_dir(STATE_CHAR_REFS_DIR)
    state["character_reference_names"] = save_images_to_dir(character_references, "char", STATE_CHAR_REFS_DIR, max_count=10)

    save_json_safe(STATE_PATH, state)
    return state


STOP_EVENT = threading.Event()


def cleanup_gradio_cache():
    gradio_temp = os.environ.get("GRADIO_TEMP_DIR", os.path.join(tempfile.gettempdir(), "gradio"))
    if os.path.exists(gradio_temp):
        try:
            shutil.rmtree(gradio_temp)
            print("Cleaned up Gradio cache.")
        except Exception:
            pass

atexit.register(cleanup_gradio_cache)

# Global modular engine instances
pipeline_manager = PipelineManager(BASE_DIR)
gen = ImageGenerator(pipeline_manager)
batch_gen = BatchGenerator(pipeline_manager)
_last_batch_processor_signature = None

# Model choices
MODEL_CHOICES = [
    FAST_FLUX_MODEL_CHOICE,
    LOW_VRAM_FLUX_MODEL_CHOICE,
    "Z-Image Turbo (Int8 - 8GB Safe)",
]

SINGLE_RESOLUTION_PRESETS = [FAST_RESOLUTION_PRESET, "~1024px", "~1280px", "~1536px (32GB+)"]
BATCH_RESOLUTION_PRESETS = [
    FAST_RESOLUTION_PRESET,
    "~1024px",
    "~1280px",
    "~1536px (32GB+)",
    "~2048px (48GB+)",
]

ANIME_PHOTO_PRESETS = {
    "Anime -> Photoreal (Subtle)": {
        "prompt": "amateur photo, ultra-realistic portrait photo, natural skin texture, 85mm lens, soft studio lighting, shallow depth of field, subtle makeup, realistic hair strands, cinematic color grading",
        "negative_prompt": "anime, cartoon, illustration, painting, lowres, blurry, oversharpened, plastic skin, doll-like, extra fingers, deformed hands, bad anatomy",
        "strength": 0.30,
        "steps": 9,
        "lora": "zimage_realistic",
        "lora_strength": 0.65,
    },
    "Anime -> Photoreal (Balanced)": {
        "prompt": "amateur photo, ultra-realistic portrait photo, natural skin texture with visible pores, 85mm lens, soft window light, shallow depth of field, subtle makeup, realistic hair strands, cinematic color grading, photorealistic lighting, 8k, sharp focus",
        "negative_prompt": "anime, cartoon, illustration, painting, cel-shaded, lowres, blurry, oversharpened, plastic skin, doll-like, extra fingers, deformed hands, bad anatomy, messy pupils, asymmetrical eyes",
        "strength": 0.40,
        "steps": 9,
        "lora": "zimage_realistic",
        "lora_strength": 0.70,
    },
    "Anime -> Photoreal (Bold)": {
        "prompt": "amateur photo, hyper-realistic portrait photograph, detailed natural skin with pores and micro-texture, 85mm f/1.4 lens, golden hour window lighting, shallow depth of field, natural makeup, individual hair strands visible, film grain, cinematic color grading, 8k resolution, sharp focus, realistic shadows",
        "negative_prompt": "anime, cartoon, illustration, painting, cel-shaded, flat shading, lowres, blurry, oversharpened, plastic skin, doll-like, extra fingers, deformed hands, bad anatomy, messy pupils, asymmetrical eyes, extra limbs, floating objects, posterization",
        "strength": 0.55,
        "steps": 9,
        "lora": "zimage_realistic",
        "lora_strength": 0.70,
    },
    "Anime -> Photoreal (Preserve Text)": {
        "prompt": "Realistic photograph preserving all text and labels exactly as they appear, natural lighting, detailed textures, photorealistic materials and surfaces, sharp focus on text areas, 8k",
        "negative_prompt": "anime, cartoon, illustration, painting, lowres, blurry, oversharpened, extra fingers, deformed hands, random text, duplicated text, garbled text, additional reflections",
        "strength": 0.25,
        "steps": 9,
        "lora": "zimage_realistic",
        "lora_strength": 0.60,
    },
    "Anime -> Photoreal (FLUX Klein 4B + LoRA)": {
        "prompt": "amateur selfie, ultra-realistic portrait, natural skin texture, smartphone photo, authentic lighting, casual pose, photorealistic, 8k, sharp focus",
        "negative_prompt": "anime, cartoon, illustration, painting, cel-shaded, lowres, blurry, oversharpened, plastic skin, doll-like, extra fingers, deformed hands, bad anatomy, messy pupils, asymmetrical eyes",
        "strength": 0.40,
        "steps": 8,
        "guidance_scale": 1.0,
        "lora": "flux_anime2real",
        "lora_strength": 1.15,
    },
}



# Dimension utilities (get_device_gpu_name, resolve_resolution_preset_for_model,
# resolve_model_dimensions, get_next_lower_dimensions, etc.) -> moved to src.utils.dimensions

def is_windows_fast_flux_device(device: str) -> bool:
    return is_windows_3070_fast_profile(
        device=device,
        vram_gb=get_device_vram_gb(device),
        gpu_name=get_device_gpu_name(device),
    )


def apply_prompt_preset(preset_name, current_prompt, current_negative, current_strength, current_steps, current_lora_file, current_lora_strength, current_guidance):
    if preset_name in ANIME_PHOTO_PRESETS:
        preset = ANIME_PHOTO_PRESETS[preset_name]
        strength = preset.get("strength", current_strength)
        steps = preset.get("steps", current_steps)
        guidance = preset.get("guidance_scale", current_guidance)

        # Resolve built-in LoRA references to actual file paths
        # Only return the path if the file exists; otherwise it will be
        # auto-downloaded when the user clicks Generate
        lora_ref = preset.get("lora")
        if lora_ref == "zimage_realistic":
            lora_path = pipeline_manager.zimage_realistic_lora_path
            if os.path.exists(lora_path):
                lora_file = lora_path
                lora_strength = preset.get("lora_strength", 0.70)
            else:
                lora_file = None
                lora_strength = preset.get("lora_strength", 0.70)
                print(f"[Preset] LoRA will be downloaded on first use: {os.path.basename(lora_path)}")
        elif lora_ref == "flux_anime2real":
            lora_path = pipeline_manager.flux_anime2real_lora_path
            if os.path.exists(lora_path):
                lora_file = lora_path
                lora_strength = preset.get("lora_strength", 0.80)
            else:
                lora_file = None
                lora_strength = preset.get("lora_strength", 0.80)
                print(f"[Preset] LoRA will be downloaded on first use: {os.path.basename(lora_path)}")
        else:
            lora_file = current_lora_file
            lora_strength = current_lora_strength

        return preset["prompt"], preset["negative_prompt"], strength, steps, lora_file, lora_strength, guidance
    return current_prompt, current_negative, current_strength, current_steps, current_lora_file, current_lora_strength, current_guidance


def build_pipe_kwargs(prompt, negative_prompt, target_pipe=None):
    kwargs = {"prompt": prompt}
    active_pipe = target_pipe or pipeline_manager.pipe
    if negative_prompt and active_pipe is not None:
        try:
            # Safely check if negative_prompt is supported by the current pipeline
            sig = inspect.signature(active_pipe.__call__)
            if "negative_prompt" in sig.parameters:
                kwargs["negative_prompt"] = negative_prompt
        except Exception:
            # Fallback to no negative prompt if inspection fails
            pass
    return kwargs


def request_stop():
    STOP_EVENT.set()
    gen.request_stop()
    batch_gen.request_stop()
    return (
        "Stop requested. Current job will cancel as soon as it reaches a safe point.",
        "Stop requested. Current job will cancel as soon as it reaches a safe point.",
        "Stop requested. Current job will cancel as soon as it reaches a safe point.",
    )


def select_folder(current_value):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        selected = filedialog.askdirectory()
        root.destroy()
        return selected or current_value
    except Exception:
        return current_value


def load_zimage_pipeline(device="mps", use_full_model=False):
    """Load the RTX 3070-safe Z-Image INT8 pipeline."""
    from diffusers import ZImagePipeline, FlowMatchEulerDiscreteScheduler
    
    # Use bfloat16 for CUDA/MPS (Ampere Tensor Cores) and float32 for CPU
    dtype = torch.bfloat16 if device in ["cuda", "mps"] else torch.float32
    print(f"Loading Z-Image-Turbo (int8) on {device}...")
    pipe = ZImagePipeline.from_pretrained(
        "Disty0/Z-Image-Turbo-SDNQ-int8",
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )

    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        use_beta_sigmas=True,
    )

    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
        print("Z-Image app-side safety checker disabled (model-native output only).")

    pipe.to(device)
    pipe.enable_attention_slicing()

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    if hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()

    compile_key = "zimage-int8"
    if compile_pipeline_components(pipe, device, compile_key):
        print(f"  torch.compile enabled for {compile_key} pipeline")

    return pipe


def load_zimage_img2img_pipeline(device="mps", use_full_model=False):
    """Get the cached Z-Image img2img pipeline from the shared pipeline manager."""
    img2img_pipe = pipeline_manager.get_zimage_img2img_pipeline(
        device=device,
        use_full_model=use_full_model,
    )
    return img2img_pipe


def get_memory_usage():
    """Get current memory usage in GB."""
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1024**3
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def compile_pipeline_components(pipe, device, cache_key=None):
    """Apply torch.compile to pipeline components for RTX 3070 Ampere optimization."""
    if os.name == "nt":
        try:
            import triton  # noqa: F401
        except ImportError:
            print("  torch.compile disabled on Windows (requires Triton — install with: pip install \"triton-windows>=3.6,<3.7\")")
            return False

    if device != "cuda" or not torch.cuda.is_available():
        return False

    capability = torch.cuda.get_device_capability(0)
    if not capability or capability[0] < 8:
        return False

    key = cache_key or f"{pipe.__class__.__name__}:{device}"
    if pipeline_manager.compiled_models.get(key):
        return False

    try:
        compiled_any = False
        for component_name in ("transformer", "unet"):
            component = getattr(pipe, component_name, None)
            if component is None:
                continue
            # Apply channels_last memory format for Ampere+ GPUs (5-10% speedup)
            if hasattr(component, 'to'):
                try:
                    component = component.to(memory_format=torch.channels_last)
                    setattr(pipe, component_name, component)
                except Exception:
                    pass  # Not all models support channels_last
            if hasattr(component, "compile_repeated_blocks"):
                component.compile_repeated_blocks(fullgraph=True)
                compiled_any = True
            else:
                setattr(pipe, component_name, torch.compile(component, mode="max-autotune", fullgraph=True))
                compiled_any = True
        if not compiled_any:
            return False
        pipeline_manager.compiled_models[key] = True
        return True
    except Exception as exc:
        print(f"  torch.compile skipped: {exc}")
        return False



# CUDAGraphRunner and get_cuda_graph_runner -> moved to src.core.cuda_graphs


def print_memory(label):
    """Print memory usage with label."""
    mem = get_memory_usage()
    print(f"  [MEM] {label}: {mem:.2f} GB")


def load_controlnet_union(device="cuda"):
    controlnet = pipeline_manager.load_controlnet_union(device)
    return controlnet


def unload_controlnet_union():
    """Unload ControlNet Union to free VRAM."""
    pipeline_manager.unload_controlnet_union()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print_memory("After ControlNet unload")


def configure_runtime_optimization_policy(
    device: str,
    optimization_profile: Optional[str],
    enable_windows_compile_probe: Optional[bool],
    enable_optional_accelerators: Optional[bool],
):
    resolved_profile = resolve_optimization_profile(
        optimization_profile,
        device=device,
        cuda_runtime=getattr(torch.version, "cuda", None),
    )
    if enable_windows_compile_probe is None:
        compile_probe_enabled = default_enable_windows_compile_probe(
            device=device,
            cuda_runtime=getattr(torch.version, "cuda", None),
        )
    else:
        compile_probe_enabled = bool(enable_windows_compile_probe)
    optional_accelerators_enabled = resolve_optional_accelerators_enabled(
        enable_optional_accelerators,
        optimization_profile=resolved_profile,
    )
    os.environ["UFIG_ENABLE_OPTIONAL_ACCELERATORS"] = "1" if optional_accelerators_enabled else "0"
    pipeline_manager.configure_optimization_policy(
        device=device,
        profile=resolved_profile,
        enable_windows_compile_probe=compile_probe_enabled,
        enable_optional_accelerators=optional_accelerators_enabled,
    )
    return resolved_profile, compile_probe_enabled, optional_accelerators_enabled


def load_pipeline(
    model_choice: str,
    device: str = "mps",
    optimization_profile: Optional[str] = None,
    enable_windows_compile_probe: Optional[bool] = None,
    enable_optional_accelerators: Optional[bool] = None,
):
    global _last_batch_processor_signature
    resolved_profile = resolve_optimization_profile(
        optimization_profile,
        device=device,
        cuda_runtime=getattr(torch.version, "cuda", None),
    )
    optional_accelerators_enabled = resolve_optional_accelerators_enabled(
        enable_optional_accelerators,
        optimization_profile=resolved_profile,
    )
    os.environ["UFIG_ENABLE_OPTIONAL_ACCELERATORS"] = "1" if optional_accelerators_enabled else "0"
    pipeline_manager.configure_optimization_policy(
        device=device,
        profile=resolved_profile,
        enable_windows_compile_probe=enable_windows_compile_probe,
        enable_optional_accelerators=optional_accelerators_enabled,
    )

    requested_choice, _ = resolve_model_choice_for_device(
        model_choice,
        device,
        vram_gb=get_device_vram_gb(device),
    )
    batch_processor_signature = (requested_choice, device)
    if _last_batch_processor_signature is not None and _last_batch_processor_signature != batch_processor_signature:
        clear_batch_processor_cache()

    selected_pipe = pipeline_manager.load_pipeline(model_choice, device)
    _last_batch_processor_signature = batch_processor_signature
    if pipeline_manager.last_model_fallback_reason:
        print(pipeline_manager.last_model_fallback_reason)
    print(f"Pipeline loaded on {device}! (Model: {pipeline_manager.current_model})")
    return selected_pipe


def clear_device_cache(device: str) -> None:
    import gc

    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_effective_optional_accelerators_enabled(model_key: str, requested_value: bool) -> bool:
    return bool(requested_value)


def build_flux_generation_attempts(model_key: str, mode: str, width: int, height: int):
    attempts = []
    seen = set()

    def add_attempt(label: str, attempt_width: int, attempt_height: int, attention_slicing: bool, vae_slicing: bool):
        key = (int(attempt_width), int(attempt_height), bool(attention_slicing), bool(vae_slicing))
        if key in seen:
            return
        seen.add(key)
        attempts.append(
            {
                "label": label,
                "width": int(attempt_width),
                "height": int(attempt_height),
                "attention_slicing": bool(attention_slicing),
                "vae_slicing": bool(vae_slicing),
                "downscaled": (int(attempt_width), int(attempt_height)) != (int(width), int(height)),
            }
        )

    cached_policy = pipeline_manager.get_cached_runtime_memory_policy(model_key, mode, width, height)
    if cached_policy:
        add_attempt(
            "cached",
            width,
            height,
            cached_policy.get("attention_slicing", False),
            cached_policy.get("vae_slicing", False),
        )

    add_attempt("fast-path", width, height, False, False)
    add_attempt("vae-slicing-retry", width, height, False, True)
    add_attempt("attention-slicing-retry", width, height, True, True)

    downscaled_width, downscaled_height = get_next_lower_dimensions(width, height)
    if (downscaled_width, downscaled_height) != (int(width), int(height)):
        add_attempt("auto-downscale", downscaled_width, downscaled_height, False, False)
        add_attempt("auto-downscale+memory", downscaled_width, downscaled_height, True, True)

    return attempts


def log_generation_runtime_stack(
    resolved_profile: str,
    compile_probe_enabled: bool,
    optional_accelerator_status: Optional[dict],
    device: str,
    current_model_key: str,
    policy: dict,
    attempt_label: Optional[str] = None,
) -> None:
    accelerator_status = dict(optional_accelerator_status or {})
    runtime_stack = {
        "profile": resolved_profile,
        "cuda_runtime": getattr(torch.version, "cuda", None) or "none",
        "autocast": should_enable_autocast(device, current_model_key, pipeline_manager.pipe),
        "attention_slicing": bool(policy.get("attention_slicing", False)),
        "vae_slicing": bool(policy.get("vae_slicing", False)),
        "vae_tiling": bool(policy.get("vae_tiling", False)),
        "cpu_offload": bool(policy.get("cpu_offload", False)),
        "compile_probe": compile_probe_enabled,
        "compile_status": get_torch_compile_probe_status(current_model_key),
        "optional_accelerators": bool(accelerator_status.get("requested", False)),
        "small_decoder": bool(accelerator_status.get("small_decoder", False)),
        "pruna_fora": bool(accelerator_status.get("pruna_fora", False)),
    }
    if accelerator_status.get("skip_reason"):
        runtime_stack["accelerator_skip"] = accelerator_status["skip_reason"]
    if attempt_label:
        runtime_stack["attempt"] = attempt_label
    print(f"[Runtime] {describe_acceleration_stack(runtime_stack)}")


def load_lora(lora_file, lora_strength: float, device: str):
    """Load or update LoRA adapter."""
    result = pipeline_manager.load_lora(lora_file, lora_strength, device)
    return result


def update_lora_strength(strength: float):
    """Update the LoRA strength without reloading."""
    if pipeline_manager.current_lora_path is not None and pipeline_manager.pipe is not None:
        try:
            if pipeline_manager.current_lora_network is not None:
                pipeline_manager.current_lora_network.multiplier = strength
            else:
                pipeline_manager.pipe.set_adapters(["default"], adapter_weights=[strength])
            return f"LoRA strength updated to {strength}"
        except Exception as e:
            return f"Error updating strength: {str(e)}"
    return "No LoRA loaded"


def run_model_preflight_at_startup(initial_model_choice: str, device: str, state: dict):
    """Pre-download selected model/deps at startup. Generate-time checks remain as fallback."""
    if os.environ.get("UFIG_STARTUP_MODEL_PREFLIGHT", "1") != "1":
        print("Startup model preflight disabled (UFIG_STARTUP_MODEL_PREFLIGHT != 1).")
        return

    resolved_choice, reason = resolve_model_choice_for_device(
        initial_model_choice,
        device,
        vram_gb=get_device_vram_gb(device),
    )
    if reason:
        print(reason)

    try:
        ensure_models_downloaded(
            resolved_choice,
            enable_multi_character=bool(state.get("enable_multi_character", False)),
            enable_faceswap=bool(state.get("enable_faceswap", False)),
            enable_pose_preservation=bool(state.get("enable_pose_preservation", False)),
            enable_klein_anatomy_fix=bool(state.get("enable_klein_anatomy_fix", False)),
            progress=None,
        )
        print(f"Startup model preflight complete for: {resolved_choice}")
    except Exception as exc:
        print(f"Startup model preflight warning: {exc}")


def _legacy_generate_image_impl(
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
    preset_choice,
    gender_strength,
    enable_prompt_upsampling,
    enable_klein_anatomy_fix,
    *character_references,  # Captures the 10 reference image slots
    progress=gr.Progress()
):
    """Generate image with automatic model download if needed."""
    STOP_EVENT.clear()

    # --- Multi-Character PuLID Setup ---
    character_embeddings = []

    # Determine target dimension for PuLID based on model choice
    target_dim = 3072 # Default for FLUX.1
    if "klein-4B" in model_choice or "klein-4b" in model_choice:
        target_dim = 7680

    if enable_multi_character:
        try:
            from src.image.pulid_helper import MultiCharacterManager
            manager = MultiCharacterManager(device=device)
            char_state_path = os.path.join(STATE_DIR, CHARACTER_MANAGER_STATE_FILENAME)
            if os.path.exists(char_state_path):
                manager.load_state(char_state_path)

                # Assign the current reference images from UI to the manager
                for i, ref_img in enumerate(character_references):
                    if ref_img is not None and i < len(manager.characters):
                        char_id = manager.characters[i]['character_id']
                        manager.assign_reference_image(char_id, ref_img)

                character_embeddings = manager.get_embeddings_for_generation(target_dim=target_dim)
                if character_embeddings:
                    print(f"  Using {len(character_embeddings)} character reference(s) for PuLID ({target_dim} dims)")
        except Exception as e:
            print(f"  Warning: Multi-character PuLID setup failed: {e}")

    model_choice, pre_download_reason = resolve_model_choice_for_device(
        model_choice,
        device,
        vram_gb=get_device_vram_gb(device),
    )
    if pre_download_reason:
        print(pre_download_reason)
    (
        resolved_profile,
        compile_probe_enabled,
        optional_accelerators_enabled,
    ) = configure_runtime_optimization_policy(
        device=device,
        optimization_profile=optimization_profile,
        enable_windows_compile_probe=enable_windows_compile_probe,
        enable_optional_accelerators=enable_optional_accelerators,
    )

    if STOP_EVENT.is_set():
        return None, "Cancelled by user.", None

    # Apply character consistency to prompt if enabled
    if character_description:
        from src.image.pulid_helper import enhance_prompt_with_character_description
        prompt = enhance_prompt_with_character_description(prompt, character_description)
        print(f"Enhanced prompt with character description: {prompt[:100]}...")

    if enable_prompt_upsampling and input_images is not None and len(input_images) > 0:
        try:
            first_image = input_images[0][0] if isinstance(input_images[0], tuple) else input_images[0]
            prompt, upsample_err = upsample_prompt_from_image(prompt, first_image, device=device)
            if upsample_err:
                print(f"VLM prompt upsampling error: {upsample_err}")
            else:
                print(f"VLM prompt upsampling applied ({len(prompt)} chars)")
        except Exception as e:
            print(f"VLM prompt upsampling failed: {e}")

    # Gender preservation: detect genders and enhance prompts
    if enable_gender_preservation and input_images is not None and len(input_images) > 0:
        try:
            from src.core.gender_helper import (
                get_gender_details,
                enhance_prompt_with_gender,
                get_gender_negative_prompt,
                merge_negative_prompts,
                get_cached_face_app
            )

            first_image = input_images[0][0] if isinstance(input_images[0], tuple) else input_images[0]
            face_app = get_cached_face_app(device=device)
            gender_info = get_gender_details(first_image, face_app)

            if gender_info['total_faces'] > 0:
                print(f"  Gender detection: {gender_info['male_count']} male, {gender_info['female_count']} female")

                # Enhance prompt with gender keywords
                prompt = enhance_prompt_with_gender(prompt, gender_info, strength=gender_strength)

                # Add gender-specific negative prompts
                gender_neg = get_gender_negative_prompt(gender_info, strength=gender_strength * 1.3)
                negative_prompt = merge_negative_prompts(negative_prompt, gender_neg)

                print(f"  Gender-enhanced prompt: {prompt[:80]}...")
            else:
                print("  No faces detected for gender preservation")

        except Exception as e:
            print(f"  Warning: Gender preservation failed: {e}")

    # Extract pose if pose preservation is enabled
    pose_image = None
    if enable_pose_preservation and input_images is not None and len(input_images) > 0:
        try:
            from src.image.pose_helper import get_pose_extractor
            
            print(f"Extracting pose in '{pose_mode}' mode...")
            first_image = input_images[0][0] if isinstance(input_images[0], tuple) else input_images[0]
            
            # Map UI mode to pose_helper mode
            mode_map = {
                "Body Only": "body",
                "Body + Face": "body_face",
                "Body + Face + Hands": "body_face_hands"
            }
            extraction_mode = mode_map.get(pose_mode, "body_face")

            extractor = get_pose_extractor(device=device, detector_type=pose_detector_type)
            pose_image = extractor.extract_pose(
                first_image,
                mode=extraction_mode,
                detect_resolution=512,
                image_resolution=max(int(height), int(width))
            )
            
            if pose_image is None:
                print("  Warning: Pose extraction failed, continuing without pose control")
                enable_pose_preservation = False
            else:
                print(f"  Pose extracted successfully")
                
        except Exception as e:
            print(f"  Warning: Pose extraction error: {e}")
            print("  Continuing without pose control")
            enable_pose_preservation = False
            pose_image = None

    # Check and download models if missing (only when user presses Generate)
    # Determine LoRA download flags based on preset
    enable_zimage_realistic_lora = False
    enable_flux_anime2real_lora = False
    if preset_choice in ANIME_PHOTO_PRESETS:
        preset = ANIME_PHOTO_PRESETS[preset_choice]
        lora_ref = preset.get("lora")
        if lora_ref == "zimage_realistic":
            enable_zimage_realistic_lora = True
        elif lora_ref == "flux_anime2real":
            enable_flux_anime2real_lora = True

    try:
        ensure_models_downloaded(
            model_choice,
            enable_multi_character=enable_multi_character,
            enable_faceswap=enable_faceswap,
            enable_pose_preservation=enable_pose_preservation,
            enable_zimage_realistic_lora=enable_zimage_realistic_lora,
            enable_flux_anime2real_lora=enable_flux_anime2real_lora,
            progress=progress
        )
    except Exception as e:
        error_msg = f"Model download failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        return None, error_msg, None

    # After download, resolve built-in LoRA path if preset is active but lora_file is None
    if preset_choice in ANIME_PHOTO_PRESETS and lora_file is None:
        preset = ANIME_PHOTO_PRESETS[preset_choice]
        lora_ref = preset.get("lora")
        if lora_ref == "zimage_realistic" and os.path.exists(pipeline_manager.zimage_realistic_lora_path):
            lora_file = pipeline_manager.zimage_realistic_lora_path
            print(f"[Generate] Using downloaded LoRA: {os.path.basename(lora_file)}")
        elif lora_ref == "flux_anime2real" and os.path.exists(pipeline_manager.flux_anime2real_lora_path):
            lora_file = pipeline_manager.flux_anime2real_lora_path
            print(f"[Generate] Using downloaded LoRA: {os.path.basename(lora_file)}")

    pipe = load_pipeline(
        model_choice,
        device,
        optimization_profile=resolved_profile,
        enable_windows_compile_probe=compile_probe_enabled,
        enable_optional_accelerators=optional_accelerators_enabled,
    )

    if seed == -1:
        seed = torch.randint(0, 2**32, (1,)).item()
    
    generator = create_torch_generator(device, seed)
    
    generation_mode = choose_image_generation_mode(
        current_model=pipeline_manager.current_model,
        has_input_images=(input_images is not None and len(input_images) > 0),
        enable_pose_preservation=enable_pose_preservation,
    )
    will_enable_cpu_offload = pipeline_manager.should_enable_cpu_offload(
        pipeline_manager.current_model,
        bool(enable_pose_preservation and pose_image is not None),
        device,
    )
    pipe, optional_accelerator_status = pipeline_manager.prepare_flux_sdnq_optional_accelerators(
        pipe,
        device=device,
        steps=int(steps),
        enable_optional_accelerators=optional_accelerators_enabled,
        mode="single",
        has_lora=bool(lora_file) or bool(enable_klein_anatomy_fix),
        has_pulid=bool(enable_multi_character and character_embeddings),
        has_faceswap=bool(enable_faceswap and faceswap_source_image is not None),
        has_pose_control=bool(enable_pose_preservation and pose_image is not None),
        has_cpu_offload=will_enable_cpu_offload,
    )
    effective_optional_accelerators_enabled = bool(optional_accelerator_status.get("enabled", False))

    # --- Apply PuLID Patching ---
    pulid_patch = None
    if enable_multi_character and character_embeddings:
        try:
            from src.image.pulid_helper import PuLIDFluxPatch
            pulid_patch = PuLIDFluxPatch(pipe.transformer, character_embeddings)
            pulid_patch.patch()
        except Exception as e:
            print(f"  Warning: PuLID patching failed: {e}")

    if STOP_EVENT.is_set():
        if pulid_patch: pulid_patch.unpatch()
        return None, "Cancelled by user.", None

    supports_lora_model = (
        any(q in str(pipeline_manager.current_model) for q in ["sdnq", "int8"])
    )
    if supports_lora_model and lora_file:
        load_lora(lora_file, lora_strength, device)

    # Apply Klein Anatomy Quality Fixer LoRA for FLUX models
    if enable_klein_anatomy_fix and "flux2-klein" in pipeline_manager.current_model:
        if os.path.exists(KLEIN_ANATOMY_LORA_PATH):
            print(f"Applying Klein Anatomy Quality Fixer LoRA...")
            try:
                load_lora(KLEIN_ANATOMY_LORA_PATH, 0.8, device)
                print("Klein Anatomy Quality Fixer applied successfully!")
            except Exception as e:
                print(f"Warning: Failed to apply Klein Anatomy Quality Fixer: {e}")
        else:
            print("Warning: Klein Anatomy Quality Fixer LoRA not found")

    base_policy = pipeline_manager.apply_runtime_memory_policy(
        pipe,
        model_key=pipeline_manager.current_model,
        device=device,
        width=int(width),
        height=int(height),
    )
    if (
        hasattr(pipe, "enable_model_cpu_offload")
        and will_enable_cpu_offload
    ):
        # Reserve CPU offload for the heavier FLUX pose/ControlNet path.
        pipe.enable_model_cpu_offload()
        cpu_offload_enabled = True
    else:
        cpu_offload_enabled = False
    base_policy["cpu_offload"] = cpu_offload_enabled
    pipeline_manager.active_runtime_memory_policy["cpu_offload"] = cpu_offload_enabled

    print_memory("Before generation")

    # CUDA Graphs: capture execution graph for repeated inference speedup.
    # Only works on CUDA with non-SDNQ models (SDNQ uses eager backend).
    # Graphs are captured per (height, width, steps) and replayed on subsequent runs.
    cuda_graph_captured = False
    if enable_cuda_graphs and device == "cuda" and not is_sdnq_or_quantized(pipeline_manager.current_model, pipe):
        try:
            runner = get_cuda_graph_runner()
            cuda_graph_captured = runner.capture_graph(
                pipe, batch_size=1, height=int(height), width=int(width),
                steps=int(steps), device=device,
            )
            if cuda_graph_captured:
                print(f"  [CUDA Graphs] Graph captured for {height}x{width} @ {steps} steps")
        except Exception as e:
            print(f"  [CUDA Graphs] Capture skipped: {e}")

    final_guidance = resolve_generation_guidance(pipeline_manager.current_model, guidance)
    if final_guidance != float(guidance):
        print(f"  [Model Policy] Guidance scale adjusted: {guidance} -> {final_guidance}")

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if should_enable_autocast(device, pipeline_manager.current_model, pipe)
        else contextlib.nullcontext()
    )

    with torch.inference_mode(), autocast_ctx:
        if is_flux_model(pipeline_manager.current_model):
            # Handle pose-conditioned generation with ControlNet
            if enable_pose_preservation and pose_image is not None:
                log_generation_runtime_stack(
                    resolved_profile,
                    compile_probe_enabled,
                    optional_accelerator_status,
                    device,
                    pipeline_manager.current_model,
                    base_policy,
                )
                try:
                    from diffusers import FluxControlNetPipeline
                    
                    # Load ControlNet Union if not already loaded
                    cn = load_controlnet_union(device)
                    
                    if cn is not None:
                        print(f"Creating ControlNet pipeline with pose conditioning (strength={controlnet_strength})...")

                        # Create ControlNet pipeline using existing pipe components
                        # FLUX requires dual text encoders (CLIP + T5)
                        cn_pipe = FluxControlNetPipeline(
                            scheduler=pipe.scheduler,
                            vae=pipe.vae,
                            text_encoder=pipe.text_encoder,
                            tokenizer=pipe.tokenizer,
                            text_encoder_2=getattr(pipe, 'text_encoder_2', None),
                            tokenizer_2=getattr(pipe, 'tokenizer_2', None),
                            transformer=pipe.transformer,
                            controlnet=cn,
                        )

                        # Resize pose image to match output dimensions
                        pose_resized = pose_image.resize((int(width), int(height)), Image.Resampling.LANCZOS)

                        pipe_kwargs = build_pipe_kwargs(prompt, negative_prompt)

                        image = cn_pipe(
                            **pipe_kwargs,
                            control_image=pose_resized,
                            height=int(height),
                            width=int(width),
                            num_inference_steps=int(steps),
                            guidance_scale=final_guidance,
                            controlnet_conditioning_scale=float(controlnet_strength),
                            generator=generator,
                            num_images_per_prompt=1,
                        ).images[0]
                        
                        mode = "txt2img+pose"
                        
                        # Return with pose skeleton if requested
                        if show_pose_skeleton:
                            return image, f"Seed: {seed} | Mode: {mode} | Pose: {pose_mode}", pose_image
                        else:
                            return image, f"Seed: {seed} | Mode: {mode} | Pose: {pose_mode}", None
                    else:
                        print("  ControlNet not available, falling back to standard generation")
                        enable_pose_preservation = False
                        
                except Exception as e:
                    print(f"  Warning: ControlNet generation failed: {e}")
                    print("  Falling back to standard generation")
                    enable_pose_preservation = False
            
            if generation_mode == "flux-img2img":
                base_width, base_height = apply_scale_to_dimensions(int(width), int(height), downscale_factor)
                attempt_error = None
                for attempt in build_flux_generation_attempts(
                    pipeline_manager.current_model,
                    generation_mode,
                    base_width,
                    base_height,
                ):
                    policy = pipeline_manager.apply_runtime_memory_policy(
                        pipe,
                        model_key=pipeline_manager.current_model,
                        device=device,
                        width=attempt["width"],
                        height=attempt["height"],
                        attention_slicing=attempt["attention_slicing"],
                        vae_slicing=attempt["vae_slicing"],
                    )
                    policy["cpu_offload"] = cpu_offload_enabled
                    log_generation_runtime_stack(
                        resolved_profile,
                        compile_probe_enabled,
                        optional_accelerator_status,
                        device,
                        pipeline_manager.current_model,
                        policy,
                        attempt_label=attempt["label"],
                    )

                    images_to_process = []
                    for img_data in input_images[:6]:
                        img = img_data[0] if isinstance(img_data, tuple) else img_data
                        resized = img.copy().resize(
                            (attempt["width"], attempt["height"]),
                            Image.Resampling.LANCZOS,
                        )
                        if resized.mode != "RGB":
                            resized = resized.convert("RGB")
                        images_to_process.append(resized)
                    print_memory(f"After resizing {len(images_to_process)} image(s)")

                    if hasattr(pipe, "vae") and hasattr(pipe.vae, "disable_tiling"):
                        pipe.vae.disable_tiling()

                    try:
                        pipe_kwargs = build_pipe_kwargs(prompt, negative_prompt)
                        image = pipe(
                            **pipe_kwargs,
                            image=images_to_process if len(images_to_process) > 1 else images_to_process[0],
                            height=attempt["height"],
                            width=attempt["width"],
                            num_inference_steps=int(steps),
                            guidance_scale=final_guidance,
                            generator=create_torch_generator(device, seed),
                            num_images_per_prompt=1,
                        ).images[0]
                        if not attempt["downscaled"]:
                            pipeline_manager.cache_runtime_memory_policy(
                                pipeline_manager.current_model,
                                generation_mode,
                                attempt["width"],
                                attempt["height"],
                                policy,
                            )
                        mode = f"img2img ({len(images_to_process)} ref)"
                        if attempt["downscaled"]:
                            mode += f", auto {attempt['width']}x{attempt['height']}"
                        break
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        attempt_error = exc
                        print(f"  OOM during {attempt['label']}; retrying...")
                        clear_device_cache(device)
                    finally:
                        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                            pipe.vae.enable_tiling()
                else:
                    raise attempt_error or RuntimeError("FLUX img2img failed after all retry policies.")
            else:
                attempt_error = None
                for attempt in build_flux_generation_attempts(
                    pipeline_manager.current_model,
                    generation_mode,
                    int(width),
                    int(height),
                ):
                    policy = pipeline_manager.apply_runtime_memory_policy(
                        pipe,
                        model_key=pipeline_manager.current_model,
                        device=device,
                        width=attempt["width"],
                        height=attempt["height"],
                        attention_slicing=attempt["attention_slicing"],
                        vae_slicing=attempt["vae_slicing"],
                    )
                    policy["cpu_offload"] = cpu_offload_enabled
                    log_generation_runtime_stack(
                        resolved_profile,
                        compile_probe_enabled,
                        optional_accelerator_status,
                        device,
                        pipeline_manager.current_model,
                        policy,
                        attempt_label=attempt["label"],
                    )
                    try:
                        pipe_kwargs = build_pipe_kwargs(prompt, negative_prompt)
                        image = pipe(
                            **pipe_kwargs,
                            height=attempt["height"],
                            width=attempt["width"],
                            num_inference_steps=int(steps),
                            guidance_scale=final_guidance,
                            generator=create_torch_generator(device, seed),
                            num_images_per_prompt=1,
                        ).images[0]
                        if not attempt["downscaled"]:
                            pipeline_manager.cache_runtime_memory_policy(
                                pipeline_manager.current_model,
                                generation_mode,
                                attempt["width"],
                                attempt["height"],
                                policy,
                            )
                        mode = "txt2img"
                        if attempt["downscaled"]:
                            mode += f", auto {attempt['width']}x{attempt['height']}"
                        break
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        attempt_error = exc
                        print(f"  OOM during {attempt['label']}; retrying...")
                        clear_device_cache(device)
                else:
                    raise attempt_error or RuntimeError("FLUX txt2img failed after all retry policies.")
        elif generation_mode == "zimage-img2img":
            log_generation_runtime_stack(
                resolved_profile,
                compile_probe_enabled,
                effective_optional_accelerators_enabled,
                device,
                pipeline_manager.current_model,
                base_policy,
            )
            img_w, img_h = int(width), int(height)
            img_w, img_h = apply_scale_to_dimensions(img_w, img_h, downscale_factor)
            images_to_process = []
            for img_data in input_images[:6]:
                img = img_data[0] if isinstance(img_data, tuple) else img_data
                resized = img.copy().resize((img_w, img_h), Image.Resampling.LANCZOS)
                if resized.mode != "RGB":
                    resized = resized.convert("RGB")
                images_to_process.append(resized)

            zimage_steps, clamped = resolve_zimage_img2img_steps(int(steps))
            if clamped:
                print(f"  Z-Image img2img requested {steps} steps; clamped to {zimage_steps}.")

            zimage_img2img_pipe = load_zimage_img2img_pipeline(
                device=device,
                use_full_model=False,
            )

            if hasattr(zimage_img2img_pipe, "vae") and hasattr(zimage_img2img_pipe.vae, "disable_tiling"):
                zimage_img2img_pipe.vae.disable_tiling()

            pipe_kwargs = build_pipe_kwargs(prompt, negative_prompt)
            image = zimage_img2img_pipe(
                **pipe_kwargs,
                image=images_to_process if len(images_to_process) > 1 else images_to_process[0],
                strength=float(img2img_strength),
                height=img_h,
                width=img_w,
                num_inference_steps=zimage_steps,
                guidance_scale=0.0,
                generator=create_torch_generator(device, seed),
                num_images_per_prompt=1,
            ).images[0]

            if hasattr(zimage_img2img_pipe, "vae") and hasattr(zimage_img2img_pipe.vae, "enable_tiling"):
                zimage_img2img_pipe.vae.enable_tiling()

            mode = f"img2img ({len(images_to_process)} ref)"
        else:
            log_generation_runtime_stack(
                resolved_profile,
                compile_probe_enabled,
                effective_optional_accelerators_enabled,
                device,
                pipeline_manager.current_model,
                base_policy,
            )
            pipe_kwargs = build_pipe_kwargs(prompt, negative_prompt)
            # Both FLUX and Z-Image VAEs require dimensions divisible by 16.
            aligned_h = (int(height) // 16) * 16 or 16
            aligned_w = (int(width) // 16) * 16 or 16
            print(f"  Generating ({aligned_w}x{aligned_h}, {steps} steps)...")
            image = pipe(
                **pipe_kwargs,
                height=aligned_h,
                width=aligned_w,
                num_inference_steps=int(steps),
                guidance_scale=final_guidance,
                generator=create_torch_generator(device, seed),
                num_images_per_prompt=1,
            ).images[0]
            mode = "txt2img"

    if STOP_EVENT.is_set():
        return None, "Cancelled by user.", None

    # --- Face Swap Post-Processing ---
    if enable_faceswap and faceswap_source_image is not None and image is not None:
        try:
            print("\n[Face Swap] Starting face swap post-processing...")
            from src.image.faceswap_helper import get_faceswap_helper

            swapper = get_faceswap_helper(device=device)
            if not swapper.is_loaded:
                if progress:
                    progress(0.95, desc="Loading face swap models...")
                swapper.load_models()

            if progress:
                progress(0.97, desc="Swapping faces...")
            image = swapper.swap_face(
                target_image=image,
                source_image=faceswap_source_image,
                face_index=int(faceswap_target_index),
                use_similarity=True,
                similarity_threshold=0.6
            )
            print("[Face Swap] Face swap completed successfully")
        except Exception as e:
            print(f"[Face Swap] Warning: Face swap failed: {e}")

    # Ensure final image is PNG and not converted by Gradio
    if image is not None:
        # Gradio 5+ handles format in the component, but we ensure PIL is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

    print_memory("After generation")

    # Force memory cleanup
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    print_memory("After cache clear")

    lora_name = os.path.basename(lora_file) if lora_file else None
    lora_info = f" | LoRA: {lora_name} ({lora_strength})" if lora_name else ""
    cfg_info = f" | CFG: {guidance}" if guidance > 0 else ""

    model_short = {
        "zimage-int8": "Z-Image (int8)",
        "flux2-klein-int8": "FLUX.2-klein-4B (int8)",
        "flux2-klein-sdnq": "FLUX.2-klein-4B (4bit)",
    }.get(pipeline_manager.current_model, pipeline_manager.current_model)

    pose_info = f" | Pose: {pose_mode}" if enable_pose_preservation and pose_image is not None else ""
    pulid_info = " | PuLID: Yes" if enable_multi_character and len(character_embeddings) > 0 else ""
    swap_info = " | FaceSwap: Yes" if enable_faceswap and faceswap_source_image is not None else ""
    fallback_info = f" | Note: {pipeline_manager.last_model_fallback_reason}" if pipeline_manager.last_model_fallback_reason else ""

    return_pose = pose_image if (show_pose_skeleton and pose_image is not None) else None

    return image, f"Seed: {seed} | Model: {model_short} | Mode: {mode} | Device: {device}{cfg_info}{lora_info}{pose_info}{pulid_info}{swap_info}{fallback_info}", return_pose


def detect_characters_handler(input_folder, device, progress=gr.Progress()):
    """
    Scan input folder for characters and prepare UI.
    Returns: status message, character data, updated UI components
    """
    import os
    try:
        if not input_folder or not os.path.exists(input_folder):
            # Return error + empty character gallery + hide all slots
            return "Error: Invalid folder path", [], gr.update(visible=False), gr.update(visible=True), *[gr.update(visible=False) if j % 4 == 0 else gr.update() for j in range(40)]

        progress(0.1, desc="Loading face detector...")
        from src.image.pulid_helper import MultiCharacterManager

        # Use provided device or fallback to CPU if CUDA unavailable
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        manager = MultiCharacterManager(device=device)

        progress(0.3, desc="Scanning folder for faces...")
        characters = manager.detect_characters_from_folder(input_folder)

        if len(characters) == 0:
            return "No faces detected in folder", [], gr.update(visible=False), gr.update(visible=True), *[gr.update(visible=False) if j % 4 == 0 else gr.update() for j in range(40)]

        progress(0.8, desc="Preparing character gallery...")

        # Prepare UI updates
        status = f"✓ Detected {len(characters)} unique characters. Upload references below."

        # Show character assignment UI, hide simple mode
        updates = [gr.update(visible=True), gr.update(visible=False)]

        # Update character slots
        for i in range(10):
            if i < len(characters):
                char = characters[i]
                updates.extend([
                    gr.update(visible=True),  # row
                    gr.update(value=char['representative_face']),  # detected_face
                    gr.update(),  # reference_upload
                    gr.update(value=f"ID: {char['character_id']}\nAppears {char['count']} times")  # info
                ])
            else:
                updates.extend([
                    gr.update(visible=False),  # row
                    gr.update(),  # detected_face
                    gr.update(),  # reference_upload
                    gr.update()  # info
                ])

        # Save manager state
        char_state_path = os.path.join(STATE_DIR, CHARACTER_MANAGER_STATE_FILENAME)
        manager.save_state(char_state_path)

        return status, characters, *updates

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, [], gr.update(visible=False), gr.update(visible=True), *[gr.update(visible=False) if j % 4 == 0 else gr.update() for j in range(40)]


def _legacy_batch_process_folder_impl(
    prompt,
    negative_prompt,
    input_folder,
    output_folder,
    batch_resolution_preset,
    downscale_factor,
    height,
    width,
    steps,
    seed,
    guidance,
    device,
    model_choice,
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
    enable_optional_accelerators,
    enable_pose_preservation,
    pose_detector_type,
    pose_mode,
    controlnet_strength,
    enable_gender_preservation,
    gender_strength,
    enable_prompt_upsampling,
    enable_klein_anatomy_fix,
    *character_references,
    progress=gr.Progress(),
):
    STOP_EVENT.clear()
    input_folder = normalize_folder_path(input_folder)
    output_folder = normalize_folder_path(output_folder)

    if not input_folder or not os.path.isdir(input_folder):
        return "Input folder not found."

    if not output_folder:
        return "Output folder is required."

    os.makedirs(output_folder, exist_ok=True)

    # Normalize paths for comparison
    abs_input = os.path.abspath(input_folder)
    abs_output = os.path.abspath(output_folder)

    image_paths = []
    for root, dirs, files in os.walk(input_folder):
        # Prune the output folder from traversal to prevent recursion if it's inside input_folder
        abs_root = os.path.abspath(root)

        # Modify dirs in-place to skip the output folder tree
        i = len(dirs) - 1
        while i >= 0:
            abs_dir = os.path.abspath(os.path.join(root, dirs[i]))
            if abs_dir == abs_output or abs_dir.startswith(abs_output + os.sep):
                dirs.pop(i)
            i -= 1

        # Double check root as well
        if abs_root == abs_output or abs_root.startswith(abs_output + os.sep):
            continue

        for name in sorted(files):
            # Skip generated files from previous runs
            if name.lower().endswith("_out.png"):
                continue

            if os.path.splitext(name)[1].lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                image_paths.append(os.path.join(root, name))

    if not image_paths:
        return "No images found in the input folder."

    if "FLUX" not in model_choice:
        return "Batch folder processing currently supports FLUX models only."

    # --- Multi-Character PuLID Setup ---
    character_embeddings = []

    # Determine target dimension for PuLID based on model choice
    target_dim = 3072 # Default for FLUX.1
    if "klein-4B" in model_choice or "klein-4b" in model_choice:
        target_dim = 7680

    if enable_multi_character:
        try:
            from src.image.pulid_helper import MultiCharacterManager
            manager = MultiCharacterManager(device=device)
            state_file = os.path.join(STATE_DIR, CHARACTER_MANAGER_STATE_FILENAME)
            if os.path.exists(state_file):
                manager.load_state(state_file)
                for i, ref_img in enumerate(character_references):
                    if ref_img is not None and i < len(manager.characters):
                        char_id = manager.characters[i]['character_id']
                        manager.assign_reference_image(char_id, ref_img)
                character_embeddings = manager.get_embeddings_for_generation(target_dim=target_dim)
        except Exception as e:
            print(f"  Warning: Multi-character PuLID setup failed: {e}")

    model_choice, pre_download_reason = resolve_model_choice_for_device(
        model_choice,
        device,
        vram_gb=get_device_vram_gb(device),
    )
    if pre_download_reason:
        print(pre_download_reason)
    (
        resolved_profile,
        compile_probe_enabled,
        optional_accelerators_enabled,
    ) = configure_runtime_optimization_policy(
        device=device,
        optimization_profile=optimization_profile,
        enable_windows_compile_probe=enable_windows_compile_probe,
        enable_optional_accelerators=enable_optional_accelerators,
    )
    requested_optional_accelerators_enabled = get_effective_optional_accelerators_enabled(
        model_choice,
        optional_accelerators_enabled,
    )

    # Apply character consistency to prompt if enabled
    if character_description:
        from src.image.pulid_helper import enhance_prompt_with_character_description
        prompt = enhance_prompt_with_character_description(prompt, character_description)

    # Check and download models if missing (only when user presses Generate)
    try:
        ensure_models_downloaded(
            model_choice,
            enable_multi_character=enable_multi_character,
            enable_faceswap=enable_faceswap,
            enable_pose_preservation=enable_pose_preservation,
            progress=progress
        )
    except Exception as e:
        return f"Model download failed: {str(e)}"

    pipe = load_pipeline(
        model_choice,
        device,
        optimization_profile=resolved_profile,
        enable_windows_compile_probe=compile_probe_enabled,
        enable_optional_accelerators=optional_accelerators_enabled,
    )
    _, optional_accelerator_status = pipeline_manager.prepare_flux_sdnq_optional_accelerators(
        pipe,
        device=device,
        steps=int(steps),
        enable_optional_accelerators=requested_optional_accelerators_enabled,
        mode="batch",
    )
    batch_runtime_stack = {
        "profile": resolved_profile,
        "cuda_runtime": getattr(torch.version, "cuda", None) or "none",
        "attention_slicing": pipeline_manager.active_runtime_memory_policy.get("attention_slicing", False),
        "vae_slicing": pipeline_manager.active_runtime_memory_policy.get("vae_slicing", False),
        "vae_tiling": pipeline_manager.active_runtime_memory_policy.get("vae_tiling", False),
        "compile_probe": compile_probe_enabled,
        "optional_accelerators": bool(optional_accelerator_status.get("requested", False)),
        "small_decoder": bool(optional_accelerator_status.get("small_decoder", False)),
        "pruna_fora": bool(optional_accelerator_status.get("pruna_fora", False)),
    }
    if optional_accelerator_status.get("skip_reason"):
        batch_runtime_stack["accelerator_skip"] = optional_accelerator_status["skip_reason"]
    print(f"[Batch Runtime] {describe_acceleration_stack(batch_runtime_stack)}")

    # --- Apply PuLID Patching ---
    pulid_patch = None
    if enable_multi_character and character_embeddings:
        try:
            from src.image.pulid_helper import PuLIDFluxPatch
            pulid_patch = PuLIDFluxPatch(pipe.transformer, character_embeddings)
            pulid_patch.patch()
        except Exception as e:
            print(f"  Warning: PuLID patching failed: {e}")

    try:
        supports_lora_model = (
            any(q in str(pipeline_manager.current_model) for q in ["sdnq", "int8"])
        )
        if supports_lora_model and lora_file:
            load_lora(lora_file, lora_strength, device)

        # Apply Klein Anatomy Quality Fixer LoRA for FLUX models
        if enable_klein_anatomy_fix and "flux2-klein" in pipeline_manager.current_model:
            if os.path.exists(KLEIN_ANATOMY_LORA_PATH):
                print(f"Applying Klein Anatomy Quality Fixer LoRA for batch...")
                try:
                    load_lora(KLEIN_ANATOMY_LORA_PATH, 0.8, device)
                except Exception as e:
                    print(f"Warning: Failed to apply Klein Anatomy Quality Fixer: {e}")

        # Define progress callback wrapper
        def progress_wrapper(p, desc):
            progress(p, desc=desc)
            if STOP_EVENT.is_set():
                raise RuntimeError("Cancelled by user")

        # Load Pose Helpers if needed
        cn_pipe_instance = None
        extractor_instance = None
        current_extraction_mode = "body_face"

        if enable_pose_preservation:
             try:
                from src.image.pose_helper import get_pose_extractor
                from diffusers import FluxControlNetPipeline

                # Load ControlNet Union
                cn = load_controlnet_union(device)
                if cn is not None:
                    # Create ControlNet pipeline using existing pipe components
                    cn_pipe_instance = FluxControlNetPipeline(
                        scheduler=pipe.scheduler,
                        vae=pipe.vae,
                        text_encoder=pipe.text_encoder,
                        tokenizer=pipe.tokenizer,
                        text_encoder_2=getattr(pipe, 'text_encoder_2', None),
                        tokenizer_2=getattr(pipe, 'tokenizer_2', None),
                        transformer=pipe.transformer,
                        controlnet=cn,
                    )
                    # Setup extractor
                    extractor_instance = get_pose_extractor(device=device, detector_type=pose_detector_type)

                    mode_map = {
                        "Body Only": "body",
                        "Body + Face": "body_face",
                        "Body + Face + Hands": "body_face_hands"
                    }
                    current_extraction_mode = mode_map.get(pose_mode, "body_face")
                else:
                    print("  ControlNet not available, disabling pose preservation for batch")
             except Exception as e:
                print(f"  Warning: Batch pose setup failed: {e}")

        # Run optimized async batch processing
        try:
            preset = batch_resolution_preset or resolve_resolution_preset_for_model(
                model_choice,
                mode="batch",
                device=device,
            )
            total = len(image_paths)

            print(f"\n🚀 Starting async batch processing ({total} images)")
            print(f"  Device: {device}")
            print(f"  Resolution: {preset} (downscale: {downscale_factor}x)")
            print(f"  Inference steps: {steps}")
            print(f"  Guidance scale: {guidance}")
            print(f"  Features: Pose={enable_pose_preservation}, "
                  f"Gender={enable_gender_preservation}, FaceSwap={enable_faceswap}")

            stats = run_async_batch_processing(
                image_paths=image_paths,
                pipe=pipe,
                device=device,
                output_folder=output_folder,
                progress_callback=progress_wrapper,

                # Process kwargs
                model_choice=pipeline_manager.current_model,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance=guidance,
                seed=seed,
                input_folder=input_folder,
                preset=preset,
                downscale_factor=downscale_factor,

                # Features
                enable_pose_preservation=(
                    enable_pose_preservation
                    and cn_pipe_instance is not None
                    and extractor_instance is not None
                ),
                cn_pipe=cn_pipe_instance,
                extractor=extractor_instance,
                controlnet_strength=controlnet_strength,
                extraction_mode=current_extraction_mode,

                # Pose args
                pose_detector_type=pose_detector_type,

                # Face Swap args
                enable_faceswap=enable_faceswap,
                faceswap_source_image=faceswap_source_image,
                faceswap_target_index=faceswap_target_index,

                # Gender args
                enable_gender_preservation=enable_gender_preservation,
                gender_strength=gender_strength,
                enable_prompt_upsampling=enable_prompt_upsampling,

                # Autocast context
                autocast_ctx=(
                    torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if should_enable_autocast(device, pipeline_manager.current_model, pipe)
                    else contextlib.nullcontext()
                ),
                optimization_profile=resolved_profile,
            )

            summary = f"Processed {stats.processed_images}/{stats.total_images} images in {stats.total_time:.1f}s. Output: {output_folder}"
            if stats.failed_images > 0:
                summary += f"\nFailed: {stats.failed_images} images."

            # Performance summary
            throughput = stats.processed_images / stats.total_time if stats.total_time > 0 else 0
            print(f"\n✓ Batch processing complete!")
            print(f"  Processed: {stats.processed_images}/{stats.total_images}")
            print(f"  Failed: {stats.failed_images}")
            print(f"  Total time: {stats.total_time:.1f}s")
            print(f"  Avg time/image: {stats.avg_time_per_image:.1f}s")
            print(f"  Throughput: {throughput:.1f} images/sec")

            return summary

        except Exception as e:
            print(f"\n✗ Batch processing error: {e}")
            import traceback
            traceback.print_exc()
            return f"Batch processing failed: {str(e)}"

    finally:
        if pulid_patch:
            pulid_patch.unpatch()

        pipeline_manager.cleanup_auxiliary_models()
        
        # Cleanup handled by AsyncBatchPipeline shutdown, but we should clear global cache
        import gc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()


def create_torch_generator(device: str, seed: int):
    if device == "cuda":
        return torch.Generator("cuda").manual_seed(int(seed))
    if device == "mps":
        return torch.Generator("mps").manual_seed(int(seed))
    return torch.Generator().manual_seed(int(seed))


def render_video_frame(
    source_image,
    prompt,
    negative_prompt,
    steps,
    guidance,
    frame_seed,
    device,
    current_model_key,
    img2img_strength,
    enable_pose_preservation,
    cn_pipe_instance,
    extractor_instance,
    extraction_mode,
    controlnet_strength,
    enable_gender_preservation,
    gender_strength,
    enable_faceswap,
    faceswap_source_image,
    faceswap_target_index,
):
    source_image = source_image.convert("RGB")
    frame_prompt, frame_negative = enhance_video_prompt_for_gender(
        prompt,
        negative_prompt,
        source_image,
        device,
        enable_gender_preservation,
        gender_strength,
    )
    frame_generator = create_torch_generator(device, frame_seed)
    final_guidance = resolve_generation_guidance(current_model_key, guidance)
    generation_mode = resolve_video_generation_mode(
        current_model_key,
        enable_pose_preservation=(
            enable_pose_preservation
            and cn_pipe_instance is not None
            and extractor_instance is not None
        ),
    )

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if should_enable_autocast(device, current_model_key, pipe)
        else contextlib.nullcontext()
    )

    with torch.inference_mode(), autocast_ctx:
        if generation_mode == "flux-pose":
            pose_image = extractor_instance.extract_pose(
                source_image,
                mode=extraction_mode,
                detect_resolution=512,
                image_resolution=max(source_image.width, source_image.height),
            )
            if pose_image is None:
                raise ValueError("Pose extraction failed for the current frame")

            pose_resized = pose_image.resize((int(source_image.width), int(source_image.height)), Image.Resampling.LANCZOS)
            pose_kwargs = build_safe_kwargs(
                cn_pipe_instance,
                **build_pipe_kwargs(frame_prompt, frame_negative, target_pipe=cn_pipe_instance),
                control_image=pose_resized,
                image=source_image,
                height=(int(source_image.height) // 16) * 16 or 16,
                width=(int(source_image.width) // 16) * 16 or 16,
                num_inference_steps=int(steps),
                guidance_scale=final_guidance,
                controlnet_conditioning_scale=float(controlnet_strength),
                generator=frame_generator,
                num_images_per_prompt=1,
            )
            image = cn_pipe_instance(**pose_kwargs).images[0]
            mode = "img2img+pose" if "image" in pose_kwargs else "txt2img+pose"
        elif generation_mode == "flux-img2img":
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "disable_tiling"):
                pipe.vae.disable_tiling()
            try:
                image = pipe(
                    **build_pipe_kwargs(frame_prompt, frame_negative, target_pipe=pipe),
                    image=source_image,
                    height=(int(source_image.height) // 16) * 16 or 16,
                    width=(int(source_image.width) // 16) * 16 or 16,
                    num_inference_steps=int(steps),
                    guidance_scale=final_guidance,
                    generator=frame_generator,
                    num_images_per_prompt=1,
                ).images[0]
            finally:
                if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                    pipe.vae.enable_tiling()
            mode = "img2img (1 ref)"
        elif generation_mode == "zimage-img2img":
            zimage_steps, clamped = resolve_zimage_img2img_steps(int(steps))
            if clamped:
                print(f"  Z-Image video img2img requested {steps} steps; clamped to {zimage_steps}.")

            zimage_img2img_pipe = load_zimage_img2img_pipeline(
                device=device,
                use_full_model=False,
            )
            if hasattr(zimage_img2img_pipe, "vae") and hasattr(zimage_img2img_pipe.vae, "disable_tiling"):
                zimage_img2img_pipe.vae.disable_tiling()
            try:
                image = zimage_img2img_pipe(
                    **build_pipe_kwargs(frame_prompt, frame_negative, target_pipe=zimage_img2img_pipe),
                    image=source_image,
                    strength=float(img2img_strength),
                    height=(int(source_image.height) // 16) * 16 or 16,
                    width=(int(source_image.width) // 16) * 16 or 16,
                    num_inference_steps=zimage_steps,
                    guidance_scale=0.0,
                    generator=frame_generator,
                    num_images_per_prompt=1,
                ).images[0]
            finally:
                if hasattr(zimage_img2img_pipe, "vae") and hasattr(zimage_img2img_pipe.vae, "enable_tiling"):
                    zimage_img2img_pipe.vae.enable_tiling()
            mode = "img2img (1 ref)"
        else:
            raise ValueError(
                f"Video processing requires image-conditioned generation; resolved mode was '{generation_mode}'."
            )

    if enable_faceswap and faceswap_source_image is not None:
        try:
            print("\n[Video Face Swap] Starting face swap post-processing...")
            from src.image.faceswap_helper import get_faceswap_helper

            swapper = get_faceswap_helper(device=device)
            if not swapper.is_loaded:
                swapper.load_models()
            image = swapper.swap_face(
                target_image=image,
                source_image=faceswap_source_image,
                face_index=int(faceswap_target_index),
                use_similarity=True,
                similarity_threshold=0.6,
            )
        except Exception as exc:
            print(f"[Video Face Swap] Warning: Face swap failed: {exc}")

    return image, mode


def create_video_from_frames(frames_dir, output_path, fps, audio_source=None, temp_dir=None, deflicker_size=5):
    """
    Create a video from processed frames with audio matching the video length.

    Args:
        frames_dir: Directory containing frame_%05d_out.png files
        output_path: Path for output video
        fps: Frames per second
        audio_source: Optional source video to extract audio from
        temp_dir: Temporary directory for intermediate files
        deflicker_size: FFmpeg deflicker window size

    Returns:
        Tuple of (success: bool, message: str, processed_count: int)
    """
    pattern = os.path.join(frames_dir, "frame_%05d_out.png")
    processed_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith("_out.png")])

    if not processed_frames:
        return False, "No processed frames found", 0

    processed_count = len(processed_frames)
    video_duration = processed_count / fps
    ffmpeg_exe = resolve_video_tool_path("ffmpeg")

    temp_root = temp_dir or os.path.dirname(output_path)
    temp_video = os.path.join(temp_root, "temp_no_audio.mp4")
    temp_filtered_video = os.path.join(temp_root, "temp_deflickered.mp4")
    temp_audio = os.path.join(temp_root, "temp_audio.aac")

    try:
        encode_cmd = build_ffmpeg_frame_encode_command(
            ffmpeg_exe=ffmpeg_exe,
            fps=fps,
            pattern=pattern,
            duration=video_duration,
            output_path=temp_video,
        )
        encode_result = subprocess.run(encode_cmd, capture_output=True, text=True)
        if encode_result.returncode != 0:
            return False, f"FFmpeg video creation failed: {encode_result.stderr or 'Unknown error'}", processed_count

        deflicker_cmd = build_ffmpeg_deflicker_command(
            ffmpeg_exe=ffmpeg_exe,
            input_path=temp_video,
            output_path=temp_filtered_video,
            deflicker_size=deflicker_size,
        )
        deflicker_result = subprocess.run(deflicker_cmd, capture_output=True, text=True)
        if deflicker_result.returncode != 0:
            return False, f"FFmpeg deflicker failed: {deflicker_result.stderr or 'Unknown error'}", processed_count

        if audio_source and os.path.exists(audio_source):
            audio_cmd = build_ffmpeg_audio_trim_command(
                ffmpeg_exe=ffmpeg_exe,
                audio_source=audio_source,
                duration=video_duration,
                output_path=temp_audio,
            )
            audio_result = subprocess.run(audio_cmd, capture_output=True, text=True)
            if audio_result.returncode != 0:
                return False, f"FFmpeg audio extraction failed: {audio_result.stderr or 'Unknown error'}", processed_count

            merge_cmd = build_ffmpeg_merge_command(
                ffmpeg_exe=ffmpeg_exe,
                video_path=temp_filtered_video,
                audio_path=temp_audio,
                output_path=output_path,
            )
            merge_result = subprocess.run(merge_cmd, capture_output=True, text=True)
            if merge_result.returncode != 0:
                return False, f"FFmpeg video/audio merge failed: {merge_result.stderr or 'Unknown error'}", processed_count
            return True, f"Deflickered video with audio created ({processed_count} frames)", processed_count

        shutil.copy(temp_filtered_video, output_path)
        return True, f"Deflickered video created ({processed_count} frames)", processed_count
    except Exception as e:
        return False, f"Error creating video: {str(e)}", processed_count
    finally:
        for temp_file in [temp_video, temp_filtered_video, temp_audio]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass


def _legacy_process_video_impl(
    prompt,
    negative_prompt,
    video_input,
    preserve_audio,
    video_output_path,
    video_resolution_preset,
    img2img_strength,
    height,
    width,
    steps,
    seed,
    guidance,
    device,
    model_choice,
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
    enable_optional_accelerators,
    enable_pose_preservation,
    pose_detector_type,
    pose_mode,
    controlnet_strength,
    enable_gender_preservation,
    gender_strength,
    enable_klein_anatomy_fix,
    *character_references,
    progress=gr.Progress(),
):
    """
    Process video with progressive output - yields updates as frames are processed.
    Working files live under the chosen output directory so the user can inspect
    processed frames while the render is active. All temp frame folders are
    removed after partial/final video assembly.
    """
    STOP_EVENT.clear()

    if not video_input:
        yield "No video input provided.", None
        return

    if not video_output_path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_output_path = os.path.join(BASE_DIR, "output", "videos", f"output_video_{timestamp}.mp4")
    video_output_path = os.path.abspath(os.path.expanduser(str(video_output_path)))

    output_parent = os.path.dirname(os.path.abspath(video_output_path)) if os.path.dirname(video_output_path) else "."
    os.makedirs(output_parent, exist_ok=True)

    workspace = prepare_video_workdirs(video_output_path)
    work_root = workspace["work_root"]
    frames_dir = workspace["raw_frames_dir"]
    processed_dir = workspace["processed_dir"]
    preview_output_path = workspace["preview_output_path"]

    fps = 30.0
    total_frames = 0
    latest_preview_path = None
    pulid_patch = None
    temporal_config = build_video_temporal_config(img2img_strength)
    _ = height, width, character_input_folder

    try:
        initial_status = build_video_status(
            stage="Preparing",
            processed_count=0,
            total_frames=0,
            detail="Creating working folders under the output directory.",
            processed_dir=processed_dir,
            final_video_path=video_output_path,
        )
        yield initial_status, None

        progress(0, desc="Extracting frames...")
        yield build_video_status(
            stage="Extracting source frames",
            processed_count=0,
            total_frames=0,
            detail="Reading the input video and expanding it into frame images.",
            processed_dir=processed_dir,
            final_video_path=video_output_path,
        ), None

        ffmpeg_exe = resolve_video_tool_path("ffmpeg")
        ffprobe_exe = resolve_video_tool_path("ffprobe")

        if ffprobe_exe:
            ffprobe_cmd = [
                ffprobe_exe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate", "-of",
                "default=noprint_wrappers=1:nokey=1", video_input
            ]
            probe_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
        else:
            probe_result = None

        if probe_result and probe_result.returncode == 0:
            import cv2
            rate = probe_result.stdout.strip()
            if "/" in rate:
                num, den = rate.split("/")
                try:
                    den_float = float(den)
                    if den_float == 0:
                        raise ZeroDivisionError("FPS denominator is zero")
                    fps = float(num) / den_float
                except (ValueError, ZeroDivisionError):
                    cap = cv2.VideoCapture(video_input)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
            else:
                try:
                    fps = float(rate)
                except ValueError:
                    cap = cv2.VideoCapture(video_input)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
        else:
            import cv2
            cap = cv2.VideoCapture(video_input)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

        ffmpeg_extract_cmd = [
            ffmpeg_exe or "ffmpeg", "-y", "-i", video_input,
            "-vsync", "0",
            os.path.join(frames_dir, "frame_%05d.png")
        ]
        print(f"Running FFmpeg Extraction: {' '.join(ffmpeg_extract_cmd)}")
        extract_result = subprocess.run(ffmpeg_extract_cmd, capture_output=True, text=True)

        extracted_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")])
        if not extracted_paths:
            detail = "Failed to extract frames from video."
            if extract_result.stderr:
                detail = f"{detail}\n{extract_result.stderr.strip()}"
            yield detail, None
            return

        total_frames = len(extracted_paths)
        print(f"Extracted {total_frames} frames at {fps} FPS")
        yield build_video_status(
            stage="Loading model",
            processed_count=0,
            total_frames=total_frames,
            detail=f"Extracted {total_frames} source frames at {fps:.2f} FPS.",
            processed_dir=processed_dir,
            final_video_path=video_output_path,
        ), None

        with Image.open(extracted_paths[0]) as first_frame_handle:
            first_frame = first_frame_handle.convert("RGB")
        effective_video_preset = video_resolution_preset or resolve_resolution_preset_for_model(
            model_choice,
            mode="video",
            device=device,
        )
        render_width, render_height = calculate_dimensions_from_ratio(
            first_frame.width,
            first_frame.height,
            effective_video_preset,
        )
        render_width, render_height = apply_scale_to_dimensions(render_width, render_height, "1x")
        shot_seed = resolve_video_shot_seed(seed, lambda: torch.randint(0, 2**32, (1,)).item())

        model_choice, pre_download_reason = resolve_model_choice_for_device(
            model_choice,
            device,
            vram_gb=get_device_vram_gb(device),
        )
        if pre_download_reason:
            print(pre_download_reason)
        (
            resolved_profile,
            compile_probe_enabled,
            optional_accelerators_enabled,
        ) = configure_runtime_optimization_policy(
            device=device,
            optimization_profile=optimization_profile,
            enable_windows_compile_probe=enable_windows_compile_probe,
            enable_optional_accelerators=enable_optional_accelerators,
        )
        requested_optional_accelerators_enabled = get_effective_optional_accelerators_enabled(
            model_choice,
            optional_accelerators_enabled,
        )

        ensure_models_downloaded(
            model_choice,
            enable_multi_character=enable_multi_character,
            enable_faceswap=enable_faceswap,
            enable_pose_preservation=enable_pose_preservation,
            progress=progress
        )

        pipe = load_pipeline(
            model_choice,
            device,
            optimization_profile=resolved_profile,
            enable_windows_compile_probe=compile_probe_enabled,
            enable_optional_accelerators=optional_accelerators_enabled,
        )
        _, optional_accelerator_status = pipeline_manager.prepare_flux_sdnq_optional_accelerators(
            pipe,
            device=device,
            steps=int(steps),
            enable_optional_accelerators=requested_optional_accelerators_enabled,
            mode="video",
        )
        video_runtime_stack = {
            "profile": resolved_profile,
            "cuda_runtime": getattr(torch.version, "cuda", None) or "none",
            "attention_slicing": pipeline_manager.active_runtime_memory_policy.get("attention_slicing", False),
            "vae_slicing": pipeline_manager.active_runtime_memory_policy.get("vae_slicing", False),
            "vae_tiling": pipeline_manager.active_runtime_memory_policy.get("vae_tiling", False),
            "compile_probe": compile_probe_enabled,
            "optional_accelerators": bool(optional_accelerator_status.get("requested", False)),
            "small_decoder": bool(optional_accelerator_status.get("small_decoder", False)),
            "pruna_fora": bool(optional_accelerator_status.get("pruna_fora", False)),
        }
        if optional_accelerator_status.get("skip_reason"):
            video_runtime_stack["accelerator_skip"] = optional_accelerator_status["skip_reason"]
        print(f"[Video Runtime] {describe_acceleration_stack(video_runtime_stack)}")

        character_embeddings = []
        target_dim = 3072
        if "klein-4B" in model_choice or "klein-4b" in model_choice:
            target_dim = 7680

        if enable_multi_character:
            try:
                from src.image.pulid_helper import MultiCharacterManager, PuLIDFluxPatch

                manager = MultiCharacterManager(device=device)
                char_state_path = os.path.join(STATE_DIR, CHARACTER_MANAGER_STATE_FILENAME)
                if os.path.exists(char_state_path):
                    manager.load_state(char_state_path)
                    for i, ref_img in enumerate(character_references):
                        if ref_img is not None and i < len(manager.characters):
                            char_id = manager.characters[i]['character_id']
                            manager.assign_reference_image(char_id, ref_img)
                    character_embeddings = manager.get_embeddings_for_generation(target_dim=target_dim)
                    if character_embeddings:
                        print(f"  [Video] Using {len(character_embeddings)} character reference(s) for PuLID")
                        pulid_patch = PuLIDFluxPatch(pipe.transformer, character_embeddings)
                        pulid_patch.patch()
            except Exception as e:
                print(f"  Warning: Video multi-character PuLID setup failed: {e}")
                pulid_patch = None

        if character_description:
            try:
                from src.image.pulid_helper import enhance_prompt_with_character_description
                prompt = enhance_prompt_with_character_description(prompt, character_description)
                print(f"  [Video] Enhanced prompt: {prompt[:100]}...")
            except Exception as e:
                print(f"  Warning: Prompt enhancement failed: {e}")

        supports_lora_model = (
            any(q in str(pipeline_manager.current_model) for q in ["sdnq", "int8"])
        )
        if supports_lora_model and lora_file:
            load_lora(lora_file, lora_strength, device)

        if enable_klein_anatomy_fix and "flux2-klein" in pipeline_manager.current_model:
            if os.path.exists(KLEIN_ANATOMY_LORA_PATH):
                print("  [Video] Applying Klein Anatomy Quality Fixer LoRA...")
                try:
                    load_lora(KLEIN_ANATOMY_LORA_PATH, 0.8, device)
                except Exception as e:
                    print(f"Warning: Video Klein Anatomy Fix failed: {e}")

        print(f"\n[Video] Starting frame processing ({total_frames} frames)")

        cn_pipe_instance = None
        extractor_instance = None
        current_extraction_mode = "body_face"
        if enable_pose_preservation:
            try:
                from src.image.pose_helper import get_pose_extractor
                from diffusers import FluxControlNetPipeline

                cn = load_controlnet_union(device)
                if cn is not None:
                    cn_pipe_instance = FluxControlNetPipeline(
                        scheduler=pipe.scheduler,
                        vae=pipe.vae,
                        text_encoder=pipe.text_encoder,
                        tokenizer=pipe.tokenizer,
                        text_encoder_2=getattr(pipe, "text_encoder_2", None),
                        tokenizer_2=getattr(pipe, "tokenizer_2", None),
                        transformer=pipe.transformer,
                        controlnet=cn,
                    )
                    extractor_instance = get_pose_extractor(device=device, detector_type=pose_detector_type)
                    mode_map = {
                        "Body Only": "body",
                        "Body + Face": "body_face",
                        "Body + Face + Hands": "body_face_hands",
                    }
                    current_extraction_mode = mode_map.get(pose_mode, "body_face")
                else:
                    print("  ControlNet not available, disabling pose preservation for video")
            except Exception as e:
                print(f"  Warning: Video pose setup failed: {e}")
                cn_pipe_instance = None
                extractor_instance = None

        resolve_video_generation_mode(
            pipeline_manager.current_model,
            enable_pose_preservation=(
                enable_pose_preservation
                and cn_pipe_instance is not None
                and extractor_instance is not None
            ),
        )

        last_preview_frame = 0
        last_preview_at = 0.0
        update_interval = max(1, total_frames // 20)
        min_frames_for_preview = min(3, total_frames)
        shared_state = {
            "done": False,
            "stats": None,
            "error": None,
            "cancelled": False,
            "detail": f"Resolved shot seed {shot_seed}. Output size: {render_width}x{render_height}",
            "processed_count": 0,
        }

        def run_video_temporal_job():
            try:
                previous_raw_frame = None
                previous_stylized_frame = None

                for frame_index, frame_path in enumerate(extracted_paths):
                    if STOP_EVENT.is_set():
                        shared_state["cancelled"] = True
                        break

                    with Image.open(frame_path) as frame_handle:
                        raw_frame = frame_handle.convert("RGB")
                    resized_raw_frame = raw_frame.resize((render_width, render_height), Image.Resampling.LANCZOS)
                    frame_seed = resolve_video_frame_seed(shot_seed, frame_index)
                    condition_image = resized_raw_frame
                    frame_strength = temporal_config.keyframe_strength
                    temporal_note = "temporal reset on initial frame"
                    hist_correlation = None

                    if previous_raw_frame is not None and previous_stylized_frame is not None:
                        should_reset, reset_reason, hist_correlation = should_reset_temporal_history(
                            frame_index,
                            previous_raw_frame,
                            resized_raw_frame,
                            temporal_config,
                        )
                        if should_reset:
                            temporal_note = reset_reason
                        else:
                            temporal_bundle = prepare_temporal_condition_frame(
                                previous_raw_frame,
                                resized_raw_frame,
                                previous_stylized_frame,
                                temporal_config,
                            )
                            condition_image = temporal_bundle["condition_image"]
                            temporal_note = (
                                f"temporal reuse ({temporal_bundle['confidence_ratio']:.0%} confident"
                                + (f", hist {hist_correlation:.3f}" if hist_correlation is not None else "")
                                + ")"
                            )

                    frame_image, mode = render_video_frame(
                        source_image=condition_image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        steps=steps,
                        guidance=guidance,
                        frame_seed=frame_seed,
                        device=device,
                        current_model_key=pipeline_manager.current_model,
                        img2img_strength=frame_strength,
                        enable_pose_preservation=enable_pose_preservation,
                        cn_pipe_instance=cn_pipe_instance,
                        extractor_instance=extractor_instance,
                        extraction_mode=current_extraction_mode,
                        controlnet_strength=controlnet_strength,
                        enable_gender_preservation=enable_gender_preservation,
                        gender_strength=gender_strength,
                        enable_faceswap=enable_faceswap,
                        faceswap_source_image=faceswap_source_image,
                        faceswap_target_index=faceswap_target_index,
                    )
                    save_processed_video_frame(frame_image, frame_path, processed_dir)
                    previous_raw_frame = resized_raw_frame
                    previous_stylized_frame = frame_image.convert("RGB").resize((render_width, render_height), Image.Resampling.LANCZOS)
                    shared_state["processed_count"] = frame_index + 1
                    shared_state["detail"] = (
                        f"Frame {frame_index + 1}/{total_frames}: {temporal_note} | Mode: {mode} | Seed: {frame_seed}"
                    )

                shared_state["stats"] = {
                    "processed_images": shared_state["processed_count"],
                    "total_images": total_frames,
                }
            except RuntimeError as exc:
                if "Cancelled by user" in str(exc):
                    shared_state["cancelled"] = True
                else:
                    shared_state["error"] = exc
            except Exception as exc:
                shared_state["error"] = exc
            finally:
                shared_state["done"] = True

        worker_thread = threading.Thread(target=run_video_temporal_job, name="video-temporal-worker", daemon=True)
        worker_thread.start()

        while not shared_state["done"]:
            worker_thread.join(timeout=0.75)
            processed_count = count_processed_video_frames(processed_dir)

            should_refresh_preview = (
                processed_count >= min_frames_for_preview
                and processed_count > last_preview_frame
                and (
                    processed_count - last_preview_frame >= update_interval
                    or (time.time() - last_preview_at) >= 4.0
                )
            )
            if should_refresh_preview:
                print(f"  Creating progressive preview with {processed_count} frames...")
                success, _, _ = create_video_from_frames(
                    processed_dir,
                    preview_output_path,
                    fps,
                    audio_source=video_input if preserve_audio else None,
                    temp_dir=work_root,
                    deflicker_size=temporal_config.deflicker_size,
                )
                if success and os.path.exists(preview_output_path):
                    latest_preview_path = preview_output_path
                    last_preview_frame = processed_count
                    last_preview_at = time.time()

            progress_fraction = 0.1 + ((processed_count / total_frames) if total_frames else 0.0) * 0.8
            progress(progress_fraction, desc=f"Processing Video: {shared_state['detail']}")
            yield build_video_status(
                stage="Processing frames",
                processed_count=processed_count,
                total_frames=total_frames,
                detail=shared_state["detail"],
                processed_dir=processed_dir,
                final_video_path=video_output_path,
                stop_requested=STOP_EVENT.is_set() or shared_state["cancelled"],
            ), latest_preview_path

        worker_thread.join()

        processed_count = count_processed_video_frames(processed_dir)
        if shared_state["error"] is not None:
            summary = f"Frame processing error: {shared_state['error']}"
            print(f"Video processing error: {shared_state['error']}")
        elif shared_state["cancelled"] or STOP_EVENT.is_set():
            summary = f"Processing interrupted by user after {processed_count}/{total_frames} frames."
        elif shared_state["stats"] is not None:
            summary = (
                f"Processed {shared_state['stats']['processed_images']}/{shared_state['stats']['total_images']} "
                "frames successfully."
            )
        else:
            summary = f"Processed {processed_count}/{total_frames} frames."

        progress(0.9, desc="Assembling final video with FFmpeg...")
        yield build_video_status(
            stage="Assembling final video",
            processed_count=processed_count,
            total_frames=total_frames,
            detail="Combining processed frames into the output video, applying deflicker, and trimming audio to match.",
            processed_dir=processed_dir,
            final_video_path=video_output_path,
            stop_requested=shared_state["cancelled"] or STOP_EVENT.is_set(),
        ), latest_preview_path

        success, msg, final_processed_count = create_video_from_frames(
            processed_dir,
            video_output_path,
            fps,
            audio_source=video_input if preserve_audio else None,
            temp_dir=work_root,
            deflicker_size=temporal_config.deflicker_size,
        )

        if success:
            summary += f"\n{msg}. Video saved to: {video_output_path}"
            yield summary, video_output_path
        else:
            summary += f"\nFailed to create video: {msg}"
            if final_processed_count > 0 and latest_preview_path and os.path.exists(latest_preview_path):
                summary += f"\nPreview video contains {final_processed_count} processed frames."
                yield summary, latest_preview_path
            else:
                yield summary, None

    finally:
        progress(1.0, desc="Cleaning up...")
        safe_remove_path(processed_dir)
        safe_remove_path(work_root)

        if pulid_patch:
            try:
                pulid_patch.unpatch()
            except Exception:
                pass

        pipeline_manager.cleanup_auxiliary_models()

        import gc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()


def normalize_folder_path(value):
    if not value:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if not value:
        return None
    if isinstance(value, str):
        value = value.strip().strip('"')
    path = os.path.abspath(os.path.expanduser(str(value)))
    if os.path.isfile(path):
        return os.path.dirname(path)
    return path


def generate_image(
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
    progress=gr.Progress()
):
    STOP_EVENT.clear()
    _ = character_input_folder
    downscale_factor = normalize_downscale_factor(downscale_factor)
    height = safe_int_value(height, 1024)
    width = safe_int_value(width, 1024)
    # Both FLUX and Z-Image VAEs require dimensions divisible by 16.
    height = (height // 16) * 16 or 16
    width = (width // 16) * 16 or 16
    steps = safe_int_value(steps, 4)
    seed = safe_int_value(seed, -1)
    faceswap_target_index = safe_int_value(faceswap_target_index, 0)
    return gen.generate(
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


def batch_process_folder(
    prompt,
    negative_prompt,
    input_folder,
    output_folder,
    batch_resolution_preset,
    downscale_factor,
    height,
    width,
    steps,
    seed,
    guidance,
    device,
    model_choice,
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
    enable_gender_preservation,
    gender_strength,
    enable_prompt_upsampling,
    enable_klein_anatomy_fix,
    preset_choice,
    *character_references,
    progress=gr.Progress(),
):
    STOP_EVENT.clear()
    _ = character_input_folder
    downscale_factor = normalize_downscale_factor(downscale_factor)
    height = safe_int_value(height, 1024)
    width = safe_int_value(width, 1024)
    # Both FLUX and Z-Image VAEs require dimensions divisible by 16.
    height = (height // 16) * 16 or 16
    width = (width // 16) * 16 or 16
    steps = safe_int_value(steps, 4)
    seed = safe_int_value(seed, -1)
    faceswap_target_index = safe_int_value(faceswap_target_index, 0)

    # Determine and download built-in LoRA based on preset
    enable_zimage_realistic_lora = False
    enable_flux_anime2real_lora = False
    if preset_choice in ANIME_PHOTO_PRESETS:
        preset = ANIME_PHOTO_PRESETS[preset_choice]
        lora_ref = preset.get("lora")
        if lora_ref == "zimage_realistic":
            enable_zimage_realistic_lora = True
        elif lora_ref == "flux_anime2real":
            enable_flux_anime2real_lora = True

    try:
        ensure_models_downloaded(
            model_choice,
            enable_zimage_realistic_lora=enable_zimage_realistic_lora,
            enable_flux_anime2real_lora=enable_flux_anime2real_lora,
            progress=progress
        )
    except Exception as e:
        return f"Model download failed: {str(e)}"

    # Resolve LoRA path after download
    if preset_choice in ANIME_PHOTO_PRESETS and lora_file is None:
        preset = ANIME_PHOTO_PRESETS[preset_choice]
        lora_ref = preset.get("lora")
        if lora_ref == "zimage_realistic" and os.path.exists(pipeline_manager.zimage_realistic_lora_path):
            lora_file = pipeline_manager.zimage_realistic_lora_path
        elif lora_ref == "flux_anime2real" and os.path.exists(pipeline_manager.flux_anime2real_lora_path):
            lora_file = pipeline_manager.flux_anime2real_lora_path

    return batch_gen.batch_process_folder(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_folder=normalize_folder_path(input_folder),
        output_folder=normalize_folder_path(output_folder),
        batch_resolution_preset=batch_resolution_preset,
        downscale_factor=downscale_factor,
        height=height,
        width=width,
        steps=steps,
        seed=seed,
        guidance=guidance,
        device=device,
        model_choice=model_choice,
        lora_file=lora_file,
        lora_strength=lora_strength,
        enable_multi_character=enable_multi_character,
        character_description=character_description,
        enable_faceswap=enable_faceswap,
        faceswap_source_image=faceswap_source_image,
        faceswap_target_index=faceswap_target_index,
        optimization_profile=optimization_profile,
        enable_windows_compile_probe=enable_windows_compile_probe,
        enable_cuda_graphs=enable_cuda_graphs,
        enable_optional_accelerators=enable_optional_accelerators,
        enable_pose_preservation=enable_pose_preservation,
        pose_detector_type=pose_detector_type,
        pose_mode=pose_mode,
        controlnet_strength=controlnet_strength,
        enable_gender_preservation=enable_gender_preservation,
        gender_strength=gender_strength,
        enable_prompt_upsampling=enable_prompt_upsampling,
        enable_klein_anatomy_fix=enable_klein_anatomy_fix,
        character_references=list(character_references),
        progress_callback=progress,
    )


def process_video(
    prompt,
    negative_prompt,
    video_input,
    preserve_audio,
    video_output_path,
    video_resolution_preset,
    img2img_strength,
    height,
    width,
    steps,
    seed,
    guidance,
    device,
    model_choice,
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
    enable_gender_preservation,
    gender_strength,
    enable_klein_anatomy_fix,
    preset_choice,
    *character_references,
    progress=gr.Progress(),
):
    STOP_EVENT.clear()
    _ = character_input_folder
    height = safe_int_value(height, 1024)
    width = safe_int_value(width, 1024)
    # Both FLUX and Z-Image VAEs require dimensions divisible by 16.
    height = (height // 16) * 16 or 16
    width = (width // 16) * 16 or 16
    steps = safe_int_value(steps, 4)
    seed = safe_int_value(seed, -1)
    faceswap_target_index = safe_int_value(faceswap_target_index, 0)

    # Determine and download built-in LoRA based on preset
    enable_zimage_realistic_lora = False
    enable_flux_anime2real_lora = False
    if preset_choice in ANIME_PHOTO_PRESETS:
        preset = ANIME_PHOTO_PRESETS[preset_choice]
        lora_ref = preset.get("lora")
        if lora_ref == "zimage_realistic":
            enable_zimage_realistic_lora = True
        elif lora_ref == "flux_anime2real":
            enable_flux_anime2real_lora = True

    try:
        ensure_models_downloaded(
            model_choice,
            enable_zimage_realistic_lora=enable_zimage_realistic_lora,
            enable_flux_anime2real_lora=enable_flux_anime2real_lora,
            progress=progress
        )
    except Exception as e:
        yield f"Model download failed: {str(e)}", None
        return

    # Resolve LoRA path after download
    if preset_choice in ANIME_PHOTO_PRESETS and lora_file is None:
        preset = ANIME_PHOTO_PRESETS[preset_choice]
        lora_ref = preset.get("lora")
        if lora_ref == "zimage_realistic" and os.path.exists(pipeline_manager.zimage_realistic_lora_path):
            lora_file = pipeline_manager.zimage_realistic_lora_path
        elif lora_ref == "flux_anime2real" and os.path.exists(pipeline_manager.flux_anime2real_lora_path):
            lora_file = pipeline_manager.flux_anime2real_lora_path

    yield from batch_gen.process_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        video_input=video_input,
        preserve_audio=preserve_audio,
        video_output_path=video_output_path,
        video_resolution_preset=video_resolution_preset,
        img2img_strength=img2img_strength,
        height=height,
        width=width,
        steps=steps,
        seed=seed,
        guidance=guidance,
        device=device,
        model_choice=model_choice,
        lora_file=lora_file,
        lora_strength=lora_strength,
        enable_multi_character=enable_multi_character,
        character_description=character_description,
        enable_faceswap=enable_faceswap,
        faceswap_source_image=faceswap_source_image,
        faceswap_target_index=faceswap_target_index,
        optimization_profile=optimization_profile,
        enable_windows_compile_probe=enable_windows_compile_probe,
        enable_cuda_graphs=enable_cuda_graphs,
        enable_optional_accelerators=enable_optional_accelerators,
        enable_pose_preservation=enable_pose_preservation,
        pose_detector_type=pose_detector_type,
        pose_mode=pose_mode,
        controlnet_strength=controlnet_strength,
        enable_gender_preservation=enable_gender_preservation,
        gender_strength=gender_strength,
        enable_prompt_upsampling=False,
        enable_klein_anatomy_fix=enable_klein_anatomy_fix,
        character_references=list(character_references),
        progress_callback=progress,
    )


def clear_lora():
    """Clear the current LoRA."""
    if pipeline_manager.current_lora_path is not None and pipeline_manager.pipe is not None:
        is_quantized = any(q in str(pipeline_manager.current_model) for q in ["sdnq", "int8"])
        if pipeline_manager.current_lora_network is not None:
            try:
                pipeline_manager.current_lora_network.remove()
            except Exception:
                pass
            pipeline_manager.current_lora_network = None
        elif not is_quantized:
            # Only call unload_lora_weights for native (non-quantized) LoRA.
            # Quantized models use custom lora_zimage which doesn't register
            # with diffusers' LoRA system.
            try:
                pipeline_manager.pipe.unload_lora_weights()
            except Exception:
                pass
        pipeline_manager.current_lora_path = None
    return None, "LoRA cleared"



# calculate_dimensions_from_target_size, parse_downscale_factor,
# apply_scale_to_dimensions, format_downscale_preview -> moved to src.utils.dimensions

def on_image_upload(images, current_preset, downscale_factor):
    if images is None or len(images) == 0:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False, value=SINGLE_RESOLUTION_PRESETS[0]),
            "Downscale applies to image-to-image and batch."
        )
    
    try:
        first_image = images[0][0] if isinstance(images[0], tuple) else images[0]
        img_width, img_height = first_image.size
    except Exception:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False, value=SINGLE_RESOLUTION_PRESETS[0]),
            "Downscale applies to image-to-image and batch."
        )
    
    preset = current_preset if current_preset in SINGLE_RESOLUTION_PRESETS else SINGLE_RESOLUTION_PRESETS[0]
    new_width, new_height = calculate_dimensions_from_ratio(img_width, img_height, preset)
    
    preview = format_downscale_preview(new_width, new_height, downscale_factor)

    return (
        gr.update(visible=False, value=new_width),
        gr.update(visible=False, value=new_height),
        gr.update(visible=True, value=preset),
        preview,
    )


def detect_gender_for_display(images, enabled):
    """Handler to show detected genders when input images change or feature is toggled."""
    if not enabled:
        return "Gender preservation is disabled", gr.update(visible=False)

    if images is None or len(images) == 0:
        return "No image uploaded", gr.update(visible=False)

    try:
        first_image = images[0][0] if isinstance(images[0], tuple) else images[0]

        from src.core.gender_helper import get_gender_details, get_cached_face_app

        face_app = get_cached_face_app(device="cuda" if torch.cuda.is_available() else "cpu")
        # Request visualization from the helper
        info = get_gender_details(first_image, face_app, visualize=True)

        if info['total_faces'] == 0:
            return "No faces detected", gr.update(visible=False)

        text_info = f"Detected: {info['male_count']} male, {info['female_count']} female ({info['total_faces']} faces)"
        return text_info, gr.update(value=info['visualization'], visible=True)
    except Exception as e:
        return f"Detection error: {str(e)}", gr.update(visible=False)


def on_resolution_preset_change(preset, images, downscale_factor):
    if images is None or len(images) == 0:
        return gr.update(), gr.update(), gr.update()
    
    first_image = images[0][0] if isinstance(images[0], tuple) else images[0]
    img_width, img_height = first_image.size
    new_width, new_height = calculate_dimensions_from_ratio(img_width, img_height, preset)
    
    preview = format_downscale_preview(new_width, new_height, downscale_factor)
    return gr.update(value=new_width), gr.update(value=new_height), gr.update(value=preview)


def update_single_downscale_preview(images, width, height, downscale_factor):
    if images is None or len(images) == 0:
        return "Downscale applies to image-to-image and batch."
    try:
        base_w = int(width)
        base_h = int(height)
    except (TypeError, ValueError):
        return "Downscale preview unavailable."
    return format_downscale_preview(base_w, base_h, downscale_factor)


def update_batch_downscale_preview(input_folder, preset, downscale_factor):
    input_folder = normalize_folder_path(input_folder)
    if not input_folder or not os.path.isdir(input_folder):
        return "Select an input folder to preview downscale."

    for root, _, files in os.walk(input_folder):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                continue
            path = os.path.join(root, name)
            try:
                with Image.open(path) as img:
                    base_w, base_h = calculate_dimensions_from_ratio(img.width, img.height, preset)
                preview = format_downscale_preview(base_w, base_h, downscale_factor)
                return f"First image: {preview}"
            except Exception:
                continue

    return "No images found in the input folder."


def update_ui_for_model(model_choice, images, width, height, downscale_factor, device):
    """Update UI visibility and defaults based on model selection."""
    is_flux = is_flux_model(model_choice)
    is_zimage = is_zimage_model(model_choice)
    is_edit_model = should_show_edit_controls(model_choice)
    # Enable LoRA for all models now that we have custom quantized support
    show_lora = True

    # Z-Image Turbo: guidance_scale is forced to 0.0 (no user control needed).
    # FLUX: guidance_scale is user-controllable (1.0 recommended).
    guidance_default = 1.0 if is_flux else 0.0
    show_guidance = is_flux

    # Batch folder processing is FLUX-only (Z-Image not supported).
    show_batch = is_flux

    # PuLID multi-character consistency patches the FLUX transformer
    # directly — not compatible with Z-Image models.
    show_pulid = is_flux

    # ControlNet pose preservation requires FLUX ControlNet Union.
    # Z-Image does not have a ControlNet implementation.
    show_pose = is_flux

    single_preset = resolve_resolution_preset_for_model(model_choice, mode="single", device=device)
    batch_preset = resolve_resolution_preset_for_model(model_choice, mode="batch", device=device)
    video_preset = resolve_resolution_preset_for_model(model_choice, mode="video", device=device)

    if images is not None and len(images) > 0:
        first_image = images[0][0] if isinstance(images[0], tuple) else images[0]
        base_width, base_height = first_image.size
    else:
        base_width = clamp_int(width, 256, 2048, 1024)
        base_height = clamp_int(height, 256, 2048, 1024)

    new_width, new_height = resolve_model_dimensions(
        model_choice,
        width=base_width,
        height=base_height,
        preset=single_preset,
        device=device,
    )
    preview = format_downscale_preview(new_width, new_height, downscale_factor)

    return (
        gr.update(visible=is_edit_model),  # img2img_label
        gr.update(visible=is_edit_model),  # input_image
        gr.update(visible=is_edit_model, value=single_preset),  # resolution_preset
        gr.update(visible=is_edit_model),  # img2img_strength
        gr.update(visible=show_lora),  # lora_label
        gr.update(visible=show_lora),  # lora_file
        gr.update(visible=show_lora),  # lora_strength
        gr.update(visible=show_lora),  # clear_lora_btn
        gr.update(visible=show_guidance, value=guidance_default),  # guidance_scale
        gr.update(value=new_width),  # width
        gr.update(value=new_height),  # height
        gr.update(value=preview),  # single_downscale_preview
        gr.update(value=batch_preset),  # batch_resolution_preset
        gr.update(value=video_preset),  # video_resolution_preset
        gr.update(visible=show_batch),  # batch_tab
        gr.update(visible=show_pulid),  # pulid_accordion
        gr.update(visible=show_pose),  # pose_accordion
    )


def on_tab_select(evt: gr.SelectData):
    """Hide generate button column when in Batch Folder or Audio Tools tab."""
    if evt.value in ["Batch Folder", "Audio Tools"] or evt.index in [1, 3]:
        return gr.update(visible=False)
    return gr.update(visible=True)


# Get available devices at startup
# #region agent log
logger.info("Computing available devices at startup")
available_devices = get_available_devices()
default_device = available_devices[0] if available_devices else "cpu"
# #region agent log
logger.info(f"Device selection result: available={available_devices}, default={default_device}")
# #endregion

initial_state, persisted_input_images = build_initial_state(available_devices, default_device)

# Create Gradio interface
# Legacy inline UI block removed. The extracted builder in src.ui.gradio_app is the only active UI source.
# Rebind the extracted UI builder so external imports and launch use src.ui.gradio_app.
demo = create_ui(globals())

# Launch the Gradio app
if __name__ == "__main__":
    if not enforce_cuda13_runtime_profile():
        sys.exit(1)

    if not SKIP_CHECK:
        if not run_dependency_preflight():
            print("Dependency preflight failed. Resolve issues and restart.")
            sys.exit(1)
    else:
        print("Skipping dependency verification (SKIP_DEPENDENCY_CHECK=1).")

    run_model_preflight_at_startup(
        initial_model_choice=initial_state.get("model_choice", MODEL_CHOICES[0]),
        device=initial_state.get("device", default_device),
        state=initial_state,
    )

    launch_server_name = os.environ.get("UFIG_GRADIO_SERVER_NAME", "0.0.0.0")
    launch_server_port = read_positive_int_env("UFIG_GRADIO_PORT", 7860)
    launch_in_browser = read_bool_env("UFIG_OPEN_BROWSER", default=False)

    demo.launch(
        server_name=launch_server_name,
        server_port=launch_server_port,
        inbrowser=launch_in_browser,
    )




