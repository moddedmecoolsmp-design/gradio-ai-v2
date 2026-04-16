"""
Integration module for async batch pipeline in app.py.

This module provides wrapper functions to integrate the async batch pipeline
with the existing batch_process_folder function.
"""

import asyncio
import os
import contextlib
from typing import Any, Optional, Callable, List, Dict, Tuple
from PIL import Image
import torch
from concurrent.futures import ThreadPoolExecutor

# Thread pool for blocking I/O operations
_io_executor = ThreadPoolExecutor(max_workers=4)

# Cache for batch processors
_batch_processor_cache: Dict[str, Any] = {}

# Ampere optimization flag
_ampere_optimizations_enabled = False

from src.runtime_policies import (
    apply_global_cuda_speed_knobs,
    resolution_preset_to_long_edge,
    resolve_generation_guidance,
    should_enable_autocast,
)
from src.core.async_batch_pipeline import (
    create_optimized_batch_processor,
    ProcessingStats
)
from src.image.vlm_prompt_upsampler import upsample_prompt_from_image


def _get_processor_cache_key(
    max_batch_size: int,
    prefetch_depth: int,
    device: str,
    **kwargs
) -> str:
    """Generate a cache key for the batch processor configuration."""
    key_parts = [
        f"batch_size={max_batch_size}",
        f"prefetch={prefetch_depth}",
        f"device={device}",
    ]
    # Add relevant configuration parameters
    for k, v in sorted(kwargs.items()):
        if k in ['enable_cuda_streams', 'warmup_steps', 'dynamic_batching', 'vram_threshold_mb']:
            key_parts.append(f"{k}={v}")
    return "|".join(key_parts)


def clear_batch_processor_cache():
    """Clear the batch processor cache. Call when model/device changes."""
    global _batch_processor_cache
    for processor in _batch_processor_cache.values():
        try:
            processor.shutdown()
        except Exception:
            pass
    _batch_processor_cache.clear()
    print("[Cache] Batch processor cache cleared")


def _gpu_total_vram_gb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
    return 0.0


def enable_ampere_optimizations() -> None:
    """
    Enable Ampere-specific optimizations for RTX 30xx/40xx GPUs.
    Call this once before batch processing starts.

    Enables:
    - TF32 for matmul/conv (10%+ speedup on Ampere)
    - cuDNN benchmark/autotune for fixed shapes
    - cuDNN deterministic=False for faster kernels
    - Maximize fragmentation cache for better allocator reuse
    """
    global _ampere_optimizations_enabled
    if _ampere_optimizations_enabled:
        return
    
    if not torch.cuda.is_available():
        return
    
    apply_global_cuda_speed_knobs(torch)
    _ampere_optimizations_enabled = True


def calculate_dimensions_from_ratio(width: int, height: int, target_resolution: str) -> tuple:
    """Calculate output dimensions maintaining aspect ratio for target resolution."""
    target_size = resolution_preset_to_long_edge(target_resolution)

    aspect_ratio = width / height

    if aspect_ratio >= 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

    # Round to nearest multiple of 64 for model compatibility
    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64

    new_width = max(256, min(2048, new_width))
    new_height = max(256, min(2048, new_height))

    return new_width, new_height


def apply_scale_to_dimensions(width: int, height: int, downscale_factor: Any) -> tuple:
    """Apply scale factor to dimensions."""
    try:
        if isinstance(downscale_factor, str):
            factor = float(downscale_factor.lower().replace("x", ""))
        else:
            factor = float(downscale_factor)
    except (ValueError, TypeError):
        factor = 1.0

    if factor <= 1.0:
        # FLUX and Z-Image VAEs require dimensions divisible by 16.
        # Align even when no downscale is applied to avoid pipeline crashes.
        new_width = (width // 16) * 16 or 16
        new_height = (height // 16) * 16 or 16
        return max(256, new_width), max(256, new_height)

    # FLUX VAE requires dimensions divisible by vae_scale_factor*2 = 16.
    # Round down to nearest 16 to avoid latent shape mismatches.
    new_width = max(256, int(width / factor))
    new_height = max(256, int(height / factor))

    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16

    new_width = max(256, min(2048, new_width))
    new_height = max(256, min(2048, new_height))

    return new_width, new_height


import inspect

def build_safe_kwargs(pipe: Any, **kwargs):
    """Build kwargs for pipeline call, only including supported parameters."""
    try:
        # Always allow num_images_per_prompt to ensure we don't get duplicates
        allowed_overrides = {"num_images_per_prompt"}

        sig = inspect.signature(pipe.__call__)
        supported_params = sig.parameters
        return {k: v for k, v in kwargs.items() if k in supported_params or k in allowed_overrides}
    except (TypeError, ValueError):
        # Fallback to a minimal set if signature cannot be inspected
        return {"prompt": kwargs.get("prompt", ""), "num_images_per_prompt": 1}

def create_batch_processor_func(
    pipe: Any,
    device: str = "cuda",
    autocast_ctx: Any = None,
    **fixed_kwargs
) -> Callable:
    """
    Create a processor function for a batch of images.
    """
    # Extract fixed parameters that don't change per batch
    prompt = fixed_kwargs.get("prompt", "")
    negative_prompt = fixed_kwargs.get("negative_prompt", "")
    steps = fixed_kwargs.get("steps", 4)
    guidance = fixed_kwargs.get("guidance", 3.5)
    seed = fixed_kwargs.get("seed", -1)
    input_folder = fixed_kwargs.get("input_folder", "")
    preset = fixed_kwargs.get("preset", "~1024px")
    downscale_factor = fixed_kwargs.get("downscale_factor", 1.0)
    img2img_strength = fixed_kwargs.get("img2img_strength", 1.0)
    enable_pose_preservation = fixed_kwargs.get("enable_pose_preservation", False)
    cn_pipe = fixed_kwargs.get("cn_pipe")
    extractor = fixed_kwargs.get("extractor")
    extraction_mode = fixed_kwargs.get("extraction_mode", "body_face")
    controlnet_strength = fixed_kwargs.get("controlnet_strength", 0.5)
    enable_faceswap = fixed_kwargs.get("enable_faceswap", False)
    faceswap_source_image = fixed_kwargs.get("faceswap_source_image")
    faceswap_target_index = fixed_kwargs.get("faceswap_target_index", 0)
    enable_gender_preservation = fixed_kwargs.get("enable_gender_preservation", False)
    gender_strength = fixed_kwargs.get("gender_strength", 1.0)
    enable_prompt_upsampling = fixed_kwargs.get("enable_prompt_upsampling", False)
    character_embeddings = fixed_kwargs.get("character_embeddings", [])
    enable_multi_character = fixed_kwargs.get("enable_multi_character", False)

    if enable_pose_preservation and (cn_pipe is None or extractor is None):
        print("  Pose preservation requested but cn_pipe/extractor not wired; disabling pose path for this run.")
        enable_pose_preservation = False

    async def process_batch(
        paths: List[str],
        images: List[Image.Image],
        pipe: Any,
        device: str,
        output_folder: str,
        **kwargs
    ) -> List[Image.Image]:
        """Process a batch of images."""
        batch_size = len(images)
        if batch_size == 0:
            return []

        # Determine if we should override guidance scale for distilled models
        # (SDNQ, INT8, etc. work best with 0.0 to avoid warnings)
        final_guidance = resolve_generation_guidance(str(type(pipe)), guidance)

        # 1. Preprocess all images (Resize & Dimensions)
        resized_images = []
        target_sizes = []
        
        for img in images:
            new_w, new_h = calculate_dimensions_from_ratio(
                img.width, img.height, preset
            )
            new_w, new_h = apply_scale_to_dimensions(new_w, new_h, downscale_factor)
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized_images.append(resized)
            target_sizes.append((new_w, new_h))

        # 2. Prepare Prompts & Seeds
        batch_prompts = []
        batch_negatives = []
        seeds = []
        
        # Base seed
        base_seed_val = int(seed) if seed != -1 else torch.randint(0, 2**32, (1,)).item()
        batch_indices = kwargs.get('indices')
        if not batch_indices:
            batch_indices = list(range(batch_size))

        for i in range(batch_size):
            # Seed
            current_seed = base_seed_val + int(batch_indices[i])
            seeds.append(current_seed)
            
            # Prompts (Gender Preservation)
            current_prompt = prompt
            current_negative = negative_prompt
            
            if enable_prompt_upsampling:
                try:
                    current_prompt, upsample_err = upsample_prompt_from_image(
                        current_prompt,
                        resized_images[i],
                        device=device,
                    )
                    if upsample_err:
                        print(f"  VLM prompt upsampling error (img {i}): {upsample_err}")
                except Exception as exc:
                    print(f"  VLM prompt upsampling failed (img {i}): {exc}")

            if enable_gender_preservation:
                try:
                    from src.core.gender_helper import (
                        get_gender_details,
                        enhance_prompt_with_gender,
                        get_gender_negative_prompt,
                        merge_negative_prompts,
                        get_cached_face_app
                    )
                    face_app = get_cached_face_app(device=device)
                    # Note: get_gender_details might be slow, could be optimized further
                    gender_info = get_gender_details(resized_images[i], face_app)
                    if gender_info['total_faces'] > 0:
                        current_prompt = enhance_prompt_with_gender(
                            current_prompt, gender_info, strength=gender_strength
                        )
                        gender_neg = get_gender_negative_prompt(
                            gender_info, strength=gender_strength * 1.3
                        )
                        current_negative = merge_negative_prompts(current_negative, gender_neg)
                except Exception as e:
                    print(f"  Gender preservation error (img {i}): {e}")
            
            batch_prompts.append(current_prompt)
            batch_negatives.append(current_negative)

        # 3. Prepare Generators
        generators = []
        if device == "cuda":
            gen_device = "cuda"
        elif device == "mps":
            gen_device = "mps"
        else:
            gen_device = None
            
        for s in seeds:
            g = torch.Generator(gen_device).manual_seed(s) if gen_device else torch.Generator().manual_seed(s)
            generators.append(g)

        # 4. Run Inference (Batch)
        # Disable autocast for SDNQ models to avoid "Unexpected floating ScalarType" error
        autocast_context = (
            autocast_ctx
            if (autocast_ctx and should_enable_autocast(device, str(type(pipe)), pipe))
            else contextlib.nullcontext()
        )

        results = []

        # Apply PuLID patching for the batch if enabled
        pulid_patch = None
        if enable_multi_character and character_embeddings:
            try:
                from src.image.pulid_helper import PuLIDFluxPatch
                # character_embeddings should be a list of embeddings
                # PuLIDFluxPatch handles the list
                pulid_patch = PuLIDFluxPatch(pipe.transformer, character_embeddings)
                pulid_patch.patch()
                print(f"  [PuLID] Patched batch with {len(character_embeddings)} character(s)")
            except Exception as e:
                print(f"  Warning: PuLID patching failed for batch: {e}")

        try:
            with torch.inference_mode(), autocast_context:
                # Handle ControlNet / Pose
                if enable_pose_preservation and cn_pipe is not None and extractor is not None:
                    batch_poses = []
                    for i, img in enumerate(resized_images):
                        w, h = target_sizes[i]
                        pose_img = extractor.extract_pose(
                            img,
                            mode=extraction_mode,
                            detect_resolution=512,
                            image_resolution=max(int(h), int(w))
                        )
                        batch_poses.append(pose_img if pose_img is not None else img) # Fallback? 

                    # Flux pipeline with ControlNet usually expects corresponding lists
                    # Note: We must ensure all images in batch have same size for stacking 
                    # OR the pipeline handles list of images with different sizes (unlikely for efficient batching)
                    # For max efficiency, we force same size if batching is enabled, 
                    # but here we might have variable sizes. 
                    # If sizes differ, we CANNOT batch efficiently in one forward pass without padding.
                    # For now, let's assume they are roughly same or process supports it. 
                    # Actually, standard Diffusers pipelines generally require same-size batching.
                    
                    # Check sizes
                    first_size = target_sizes[0]
                    all_same_size = all(s == first_size for s in target_sizes)
                    
                    if all_same_size:
                        # True Batch
                        kwargs = build_safe_kwargs(
                            cn_pipe,
                            prompt=batch_prompts,
                            negative_prompt=batch_negatives,
                            control_image=batch_poses,
                            image=resized_images if "img2img" in str(type(cn_pipe)).lower() else None,
                            height=first_size[1],
                            width=first_size[0],
                            num_inference_steps=int(steps),
                            guidance_scale=final_guidance,
                            controlnet_conditioning_scale=float(controlnet_strength),
                            generator=generators,
                            num_images_per_prompt=1,
                        )
                        print(f"  Running batch inference (ControlNet): {len(batch_prompts)} prompts...")
                        batch_results = cn_pipe(**kwargs).images
                        print(f"  Inference complete. Got {len(batch_results)} images.")
                        results = batch_results
                    else:
                        # Fallback to serial for this batch if sizes differ
                        # (This shouldn't happen if dynamic batcher groups by aspect ratio, but we don't have that yet)
                        print("  Warning: Batch images have different sizes. Processing serially.")
                        for i in range(batch_size):
                            kwargs = build_safe_kwargs(
                                cn_pipe,
                                prompt=batch_prompts[i],
                                negative_prompt=batch_negatives[i],
                                control_image=batch_poses[i],
                                height=target_sizes[i][1],
                                width=target_sizes[i][0],
                                num_inference_steps=int(steps),
                                guidance_scale=final_guidance,
                                controlnet_conditioning_scale=float(controlnet_strength),
                                generator=generators[i],
                                num_images_per_prompt=1,
                            )
                            res = cn_pipe(**kwargs).images[0]
                            results.append(res)
                else:
                    # Standard Img2Img / Txt2Img
                    # Check sizes
                    first_size = target_sizes[0]
                    all_same_size = all(s == first_size for s in target_sizes)

                    if all_same_size:
                        kwargs = build_safe_kwargs(
                            pipe,
                            prompt=batch_prompts,
                            negative_prompt=batch_negatives,
                            image=resized_images,
                            strength=float(img2img_strength),
                            height=first_size[1],
                            width=first_size[0],
                            num_inference_steps=int(steps),
                            guidance_scale=final_guidance,
                            generator=generators,
                            num_images_per_prompt=1,
                        )
                        print(f"  Running batch inference: {len(batch_prompts)} prompts...")
                        batch_results = pipe(**kwargs).images
                        print(f"  Inference complete. Got {len(batch_results)} images.")
                        results = batch_results
                    else:
                         print("  Warning: Batch images have different sizes. Processing serially.")
                         for i in range(batch_size):
                            kwargs = build_safe_kwargs(
                                pipe,
                                prompt=batch_prompts[i],
                                negative_prompt=batch_negatives[i],
                                image=resized_images[i],
                                strength=float(img2img_strength),
                                height=target_sizes[i][1],
                                width=target_sizes[i][0],
                                num_inference_steps=int(steps),
                                guidance_scale=final_guidance,
                                generator=generators[i],
                                num_images_per_prompt=1,
                            )
                            res = pipe(**kwargs).images[0]
                            results.append(res)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise e # Propagate to dynamic sizer
            print(f"  Batch inference error: {e}")
            return [] # Fail batch

        finally:
            # Unpatch PuLID after the batch is finished
            if pulid_patch:
                try:
                    pulid_patch.unpatch()
                    print("  [PuLID] Unpatched batch")
                except Exception as e:
                    print(f"  Warning: PuLID unpatching failed for batch: {e}")

        # 5. Post-processing & Saving
        final_results = []
        # Ensure we don't process more results than we have input paths
        process_limit = min(len(results), len(paths))
        if len(results) != len(paths):
            print(f"  Warning: Batch results mismatch. Paths: {len(paths)}, Results: {len(results)}. Using limit: {process_limit}")

        for i in range(process_limit):
            res = results[i]
            path = paths[i]
            
            # Face Swap
            if enable_faceswap and faceswap_source_image is not None:
                try:
                    from src.image.faceswap_helper import get_faceswap_helper
                    swapper = get_faceswap_helper(device=device)
                    if not swapper.is_loaded:
                        swapper.load_models()
                    res = swapper.swap_face(
                        target_image=res,
                        source_image=faceswap_source_image,
                        face_index=int(faceswap_target_index),
                        use_similarity=True,
                        similarity_threshold=0.6
                    )
                except Exception as e:
                    print(f"  Warning: Face swap failed for {path}: {e}")

            final_results.append(res)
            
            # Save
            base_name = os.path.splitext(os.path.basename(path))[0]

            # Normalize paths for Windows compatibility
            norm_path = os.path.abspath(os.path.normpath(path))
            norm_input = os.path.abspath(os.path.normpath(input_folder))

            try:
                rel_dir = os.path.relpath(os.path.dirname(norm_path), norm_input)
            except ValueError:
                # Paths on different drives or other relpath error
                rel_dir = "."

            output_dir = (
                output_folder if rel_dir == "."
                else os.path.join(output_folder, rel_dir)
            )
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{base_name}_out.png")
            await asyncio.to_thread(res.save, output_path)
            print(f"  [SAVED] {output_path}")

        return final_results

    return process_batch



async def process_batch_folder_async(
    image_paths: List[str],
    pipe: Any,
    device: str,
    output_folder: str,
    progress_callback: Optional[Callable] = None,
    max_batch_size: int = 4,
    prefetch_depth: int = 3,
    **process_kwargs
) -> ProcessingStats:
    """
    Process a folder of images asynchronously with optimizations.

    Args:
        image_paths: List of image file paths
        pipe: Pipeline object
        device: Device to use
        output_folder: Output directory
        progress_callback: Gradio progress callback
        max_batch_size: Maximum batch size
        prefetch_depth: Prefetch queue depth
        **process_kwargs: Additional processing parameters

    Returns:
        ProcessingStats with processing statistics
    """
    # Enable Ampere optimizations once per session
    if device == "cuda":
        enable_ampere_optimizations()

    # Determine optimal batch size based on device
    if device == "cuda":
        # Check VRAM
        if torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # 3070 has 8GB; bias toward stability.
            if total_vram_gb <= 8:
                max_batch_size = min(max_batch_size, 2)
            elif total_vram_gb <= 12:
                max_batch_size = min(max_batch_size, 3)
            else:
                max_batch_size = min(max_batch_size, 4)
    elif device == "mps":
        max_batch_size = 2  # Conservative for MPS
    else:
        max_batch_size = 1  # CPU

    # Bucket images by target size to keep true batching
    preset = process_kwargs.get("preset", "~1024px")
    downscale_factor = process_kwargs.get("downscale_factor", 1.0)
    buckets = {}
    fallback = []
    
    async def get_image_dimensions(path):
        """Get image dimensions asynchronously."""
        with Image.open(path) as img:
            new_w, new_h = calculate_dimensions_from_ratio(img.width, img.height, preset)
            new_w, new_h = apply_scale_to_dimensions(new_w, new_h, downscale_factor)
        return new_w, new_h
    
    for path in image_paths:
        try:
            new_w, new_h = await asyncio.to_thread(get_image_dimensions, path)
            buckets.setdefault((new_w, new_h), []).append(path)
        except Exception:
            fallback.append(path)

    if buckets:
        bucket_list = [buckets[key] for key in sorted(buckets.keys())]
        if fallback:
            bucket_list.append(fallback)
        processing_paths = bucket_list
    else:
        processing_paths = image_paths

    # Create or reuse processor
    cache_key = _get_processor_cache_key(
        max_batch_size=max_batch_size,
        prefetch_depth=prefetch_depth,
        device=device,
        enable_cuda_streams=(device == "cuda"),
        warmup_steps=1,
        dynamic_batching=True,
        vram_threshold_mb=1536 if device == "cuda" else 1024
    )
    
    if cache_key in _batch_processor_cache:
        processor = _batch_processor_cache[cache_key]
        print(f"[Cache] Reusing batch processor for config: {cache_key}")
    else:
        processor = create_optimized_batch_processor(
            max_batch_size=max_batch_size,
            prefetch_depth=prefetch_depth,
            num_workers=4,
            enable_cuda_streams=(device == "cuda"),
            warmup_steps=1,
            dynamic_batching=True,
            vram_threshold_mb=1536 if device == "cuda" else 1024
        )
        _batch_processor_cache[cache_key] = processor
        print(f"[Cache] Created new batch processor for config: {cache_key}")


    # Create process function
    process_fn = create_batch_processor_func(
        pipe=pipe,
        device=device,
        **process_kwargs
    )

    # Add index to kwargs for seed calculation
    indexed_paths = {p: i for i, p in enumerate(image_paths)}

    # Wrapper to inject index for batch
    def process_with_index(paths, images, pipe, device, output_folder, **kwargs):
        batch_indices = [indexed_paths.get(p, 0) for p in paths]

        return process_fn(
            paths=paths,
            images=images,
            pipe=pipe,
            device=device,
            output_folder=output_folder,
            indices=batch_indices,
            **kwargs
        )

    # Cached processors stay alive across runs and are released only when
    # clear_batch_processor_cache() is called for a model/device change.
    stats = await processor.process_batch_async(
        image_paths=processing_paths,
        pipe=pipe,
        device=device,
        process_fn=process_with_index,
        output_folder=output_folder,
        progress_callback=progress_callback,
        **process_kwargs
    )

    return stats


def run_async_batch_processing(
    image_paths: List[str],
    pipe: Any,
    device: str,
    output_folder: str,
    progress_callback: Optional[Callable] = None,
    **process_kwargs
) -> ProcessingStats:
    """
    Synchronous wrapper for async batch processing.

    This function can be called from the existing synchronous code.
    """
    # Check if event loop exists
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create new loop in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    process_batch_folder_async(
                        image_paths=image_paths,
                        pipe=pipe,
                        device=device,
                        output_folder=output_folder,
                        progress_callback=progress_callback,
                        **process_kwargs
                    )
                )
                return future.result()
        else:
            # Use existing loop
            return loop.run_until_complete(
                process_batch_folder_async(
                    image_paths=image_paths,
                    pipe=pipe,
                    device=device,
                    output_folder=output_folder,
                    progress_callback=progress_callback,
                    **process_kwargs
                )
            )
    except RuntimeError:
        # No event loop, create new one
        return asyncio.run(
            process_batch_folder_async(
                image_paths=image_paths,
                pipe=pipe,
                device=device,
                output_folder=output_folder,
                progress_callback=progress_callback,
                **process_kwargs
            )
        )
