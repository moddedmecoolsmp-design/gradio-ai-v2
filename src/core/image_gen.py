import os
import time
import torch
import contextlib
import gc
import re
import inspect
from PIL import Image, Image as PILImage
from typing import Optional, List, Dict, Any, Union, Callable

from src.constants import CHARACTER_MANAGER_STATE_FILENAME
from src.runtime_policies import (
    is_flux_model,
    resolve_generation_guidance,
    should_enable_autocast,
    should_use_attention_slicing,
    should_use_vae_slicing,
    should_use_vae_tiling,
)
from src.utils.device_utils import get_memory_usage, print_memory, get_device_vram_gb

class ImageGenerator:
    """Core image generation engine decoupled from UI."""

    def __init__(self, pipeline_manager):
        self.pm = pipeline_manager
        self.stop_requested = False
        self._cached_pipe_params = None  # Cache inspect.signature result
        self._cached_pipe_class = None   # Track which pipe class was cached
        self._last_config_signature = None  # Skip redundant configure_optimization_policy calls

    def request_stop(self):
        self.stop_requested = True

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        steps: int = 20,
        seed: int = -1,
        guidance: float = 3.5,
        device: str = "cuda",
        model_choice: str = "Z-Image Turbo (Int8 - 8GB Safe)",
        input_images: Optional[List[Any]] = None,
        downscale_factor: float = 1.0,
        img2img_strength: float = 0.8,
        lora_file: Optional[str] = None,
        lora_strength: float = 1.0,
        enable_multi_character: bool = False,
        character_references: Optional[List[Any]] = None,
        character_description: Optional[str] = None,
        enable_faceswap: bool = False,
        faceswap_source_image: Optional[Any] = None,
        faceswap_target_index: int = 0,
        enable_pose_preservation: bool = False,
        pose_detector_type: str = "dwpose",
        pose_mode: str = "Body + Face",
        controlnet_strength: float = 0.6,
        show_pose_skeleton: bool = False,
        enable_gender_preservation: bool = False,
        gender_strength: float = 0.5,
        enable_klein_anatomy_fix: bool = False,
        optimization_profile: Optional[str] = None,
        enable_windows_compile_probe: Optional[bool] = None,
        enable_cuda_graphs: Optional[bool] = None,
        enable_optional_accelerators: Optional[bool] = None,
        enable_prompt_upsampling: bool = False,
        progress_callback: Optional[Callable] = None,
    ):
        self.stop_requested = False
        height = self._safe_int(height, 512)
        width = self._safe_int(width, 512)
        # Both FLUX and Z-Image VAEs require dimensions divisible by
        # vae_scale_factor*2=16. Align down to avoid pipeline crashes.
        height = (height // 16) * 16 or 16
        width = (width // 16) * 16 or 16
        steps = self._safe_int(steps, 20)
        seed = self._safe_int(seed, -1)
        faceswap_target_index = self._safe_int(faceswap_target_index, 0)
        downscale_factor = self._normalize_downscale_factor(downscale_factor)
        # Skip redundant configure_optimization_policy calls when parameters haven't changed
        config_signature = (device, optimization_profile, enable_windows_compile_probe, enable_cuda_graphs, enable_optional_accelerators)
        if config_signature != self._last_config_signature:
            self.pm.configure_optimization_policy(
                device=device,
                profile=optimization_profile,
                enable_windows_compile_probe=enable_windows_compile_probe,
                enable_optional_accelerators=enable_optional_accelerators,
            )
            self._last_config_signature = config_signature

        # 1. Model & Pipeline Preparation
        try:
            self.pm.ensure_models_downloaded(
                model_choice,
                enable_multi_character=enable_multi_character,
                enable_faceswap=enable_faceswap,
                enable_pose_preservation=enable_pose_preservation,
                enable_klein_anatomy_fix=enable_klein_anatomy_fix,
                progress=progress_callback
            )
        except Exception as e:
            return None, f"Model download failed: {e}", None

        pipe = self.pm.load_pipeline(model_choice, device)
        current_model = self.pm.current_model

        # 2. Multi-Character PuLID Setup
        character_embeddings = []
        pulid_patch = None
        if enable_multi_character and character_references:
            try:
                from src.image.pulid_helper import MultiCharacterManager, PuLIDFluxPatch
                target_dim = 3072
                if "klein-4B" in current_model:
                    target_dim = 7680

                manager = MultiCharacterManager(device=device)
                char_state_path = os.path.join(self.pm.state_dir, CHARACTER_MANAGER_STATE_FILENAME)
                if os.path.exists(char_state_path):
                    manager.load_state(char_state_path)
                    for i, ref_img in enumerate(character_references):
                        if ref_img is not None and i < len(manager.characters):
                            manager.assign_reference_image(manager.characters[i]['character_id'], ref_img)

                    character_embeddings = manager.get_embeddings_for_generation(target_dim=target_dim)
                    if character_embeddings:
                        pulid_patch = PuLIDFluxPatch(pipe.transformer, character_embeddings)
                        pulid_patch.patch()
            except Exception as e:
                print(f"  Warning: PuLID setup failed: {e}")

        # 3. Prompt Enhancements (Character & Gender)
        if character_description:
            try:
                from src.image.pulid_helper import enhance_prompt_with_character_description
                prompt = enhance_prompt_with_character_description(prompt, character_description)
            except ImportError: pass

        if enable_prompt_upsampling and input_images:
            try:
                from src.image.vlm_prompt_upsampler import upsample_prompt_from_image
                first_image = input_images[0][0] if isinstance(input_images[0], tuple) else input_images[0]
                prompt, upsample_err = upsample_prompt_from_image(prompt, first_image, device=device)
                if upsample_err:
                    print(f"  VLM prompt upsampling error: {upsample_err}")
            except Exception as e:
                print(f"  Warning: Prompt upsampling failed: {e}")

        if enable_gender_preservation and input_images:
            try:
                from src.core.gender_helper import get_gender_details, enhance_prompt_with_gender, get_gender_negative_prompt, merge_negative_prompts, get_cached_face_app
                first_image = input_images[0][0] if isinstance(input_images[0], tuple) else input_images[0]
                face_app = get_cached_face_app(device=device)
                gender_info = get_gender_details(first_image, face_app)
                if gender_info['total_faces'] > 0:
                    prompt = enhance_prompt_with_gender(prompt, gender_info, strength=gender_strength)
                    gender_neg = get_gender_negative_prompt(gender_info, strength=gender_strength * 1.3)
                    negative_prompt = merge_negative_prompts(negative_prompt, gender_neg)
            except Exception as e:
                print(f"  Warning: Gender preservation failed: {e}")

        # 4. Pose Extraction
        pose_image = None
        if enable_pose_preservation and input_images:
            try:
                from src.image.pose_helper import get_pose_extractor
                first_image = input_images[0][0] if isinstance(input_images[0], tuple) else input_images[0]
                mode_map = {"Body Only": "body", "Body + Face": "body_face", "Body + Face + Hands": "body_face_hands"}
                extractor = get_pose_extractor(device=device, detector_type=pose_detector_type)
                pose_image = extractor.extract_pose(
                    first_image,
                    mode=mode_map.get(pose_mode, "body_face"),
                    image_resolution=max(int(height), int(width))
                )
            except Exception as e:
                print(f"  Warning: Pose extraction failed: {e}")

        if self.stop_requested:
            if pulid_patch: pulid_patch.unpatch()
            return None, "Cancelled by user.", None

        # 5. LoRA Loading
        if lora_file:
            self.pm.load_lora(lora_file, lora_strength, device)

        if enable_klein_anatomy_fix and "flux2-klein" in current_model:
            if os.path.exists(self.pm.klein_anatomy_lora_path):
                self.pm.load_lora(self.pm.klein_anatomy_lora_path, 0.8, device)

        # 6. Generation Parameters
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()

        generator = torch.Generator(device if device != "cpu" else None).manual_seed(int(seed))

        resolved_profile = getattr(self.pm, "optimization_profile", "balanced")
        vram_gb = get_device_vram_gb(device)
        use_attention_slicing = should_use_attention_slicing(
            device=device,
            model_key=current_model,
            pipe=pipe,
            vram_gb=vram_gb,
            optimization_profile=resolved_profile,
        )
        if use_attention_slicing:
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
        elif hasattr(pipe, "disable_attention_slicing"):
            try:
                pipe.disable_attention_slicing()
            except Exception:
                pass

        if should_use_vae_slicing(device=device, vram_gb=vram_gb, optimization_profile=resolved_profile):
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            elif hasattr(getattr(pipe, "vae", None), "enable_slicing"):
                try:
                    pipe.vae.enable_slicing()
                except Exception:
                    pass
        if should_use_vae_tiling(
            device=device,
            model_key=current_model,
            pipe=pipe,
            vram_gb=vram_gb,
            width=int(width),
            height=int(height),
            optimization_profile=resolved_profile,
        ):
            if hasattr(pipe, "enable_vae_tiling"):
                try:
                    pipe.enable_vae_tiling()
                except ValueError:
                    pass
            elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
                try:
                    pipe.vae.enable_tiling()
                except ValueError:
                    pass
        if hasattr(pipe, "enable_model_cpu_offload") and self.pm.should_enable_cpu_offload(
            current_model,
            enable_pose_preservation,
            device,
        ):
            pipe.enable_model_cpu_offload()
            print("  CPU offload enabled (model components will move between CPU/GPU per step).")
            print("  Note: This significantly slows generation. Disable pose preservation for faster inference.")

        final_guidance = resolve_generation_guidance(current_model, guidance)

        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if should_enable_autocast(device, current_model, pipe)
            else contextlib.nullcontext()
        )

        # 7. Core Generation Logic
        try:
            with torch.inference_mode(), autocast_ctx:
                # Decide generation mode
                has_input = input_images is not None and len(input_images) > 0
                is_flux = is_flux_model(current_model)

                # Helper to build common kwargs
                def get_pipe_kwargs(target_pipe):
                    kwargs = {"prompt": prompt}
                    # Cache inspect.signature result per pipe class to avoid
                    # repeated introspection on every generation call.
                    pipe_class = target_pipe.__class__
                    if self._cached_pipe_class is pipe_class and self._cached_pipe_params is not None:
                        supported_params = self._cached_pipe_params
                    else:
                        try:
                            supported_params = inspect.signature(target_pipe.__call__).parameters
                        except (TypeError, ValueError):
                            supported_params = {}
                        self._cached_pipe_class = pipe_class
                        self._cached_pipe_params = supported_params

                    if "prompt_2" in supported_params:
                        kwargs["prompt_2"] = prompt
                    if negative_prompt and "negative_prompt" in supported_params:
                        kwargs["negative_prompt"] = negative_prompt
                    if negative_prompt and "negative_prompt_2" in supported_params:
                        kwargs["negative_prompt_2"] = negative_prompt
                    return kwargs

                image = None
                mode_str = "txt2img"

                # A. ControlNet Path
                if is_flux and enable_pose_preservation and pose_image:
                    from diffusers import FluxControlNetPipeline
                    cn = self.pm.load_controlnet_union(device)
                    if cn:
                        cn_pipe = FluxControlNetPipeline(
                            scheduler=pipe.scheduler, vae=pipe.vae,
                            text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer,
                            text_encoder_2=getattr(pipe, 'text_encoder_2', None),
                            tokenizer_2=getattr(pipe, 'tokenizer_2', None),
                            transformer=pipe.transformer, controlnet=cn
                        )
                        pose_resized = pose_image.resize((int(width), int(height)), PILImage.Resampling.LANCZOS)
                        image = cn_pipe(
                            **get_pipe_kwargs(cn_pipe),
                            control_image=pose_resized,
                            height=int(height), width=int(width),
                            num_inference_steps=int(steps),
                            guidance_scale=final_guidance,
                            controlnet_conditioning_scale=float(controlnet_strength),
                            generator=generator,
                            num_images_per_prompt=1,
                        ).images[0]
                        mode_str = "txt2img+pose"

                # B. Img2Img Path
                if image is None and has_input:
                    img_w, img_h = self._scale_dims(width, height, downscale_factor)
                    processed_images = []
                    for img_data in input_images[:6]:
                        img = img_data[0] if isinstance(img_data, tuple) else img_data
                        resized = img.copy().resize((img_w, img_h), PILImage.Resampling.LANCZOS).convert("RGB")
                        processed_images.append(resized)

                    input_arg = processed_images if len(processed_images) > 1 else processed_images[0]

                    if is_flux:
                        # Flux2KleinPipeline auto-aligns image dims to multiples
                        # of vae_scale_factor*2=16 internally. Pass height/width
                        # only when they match the downscaled image to avoid
                        # latent shape mismatches between noise and image latents.
                        flux_kwargs = get_pipe_kwargs(pipe)
                        flux_kwargs["image"] = input_arg
                        flux_kwargs["height"] = img_h
                        flux_kwargs["width"] = img_w
                        flux_kwargs["num_inference_steps"] = int(steps)
                        flux_kwargs["guidance_scale"] = final_guidance
                        flux_kwargs["generator"] = generator
                        flux_kwargs["num_images_per_prompt"] = 1
                        # Only pass height/width if they match the image dims
                        # after pipeline's internal alignment (multiples of 16)
                        aligned_w = (img_w // 16) * 16
                        aligned_h = (img_h // 16) * 16
                        if aligned_w != img_w or aligned_h != img_h:
                            # Dimensions weren't aligned — let pipeline derive
                            # height/width from the image itself
                            flux_kwargs.pop("height", None)
                            flux_kwargs.pop("width", None)
                        image = pipe(**flux_kwargs).images[0]
                    elif "zimage" in current_model:
                        zimage_img2img_pipe = self.pm.get_zimage_img2img_pipeline(
                            device=device,
                            use_full_model=False,
                        )
                        image = zimage_img2img_pipe(
                            **get_pipe_kwargs(zimage_img2img_pipe),
                            image=input_arg,
                            strength=float(img2img_strength),
                            height=img_h, width=img_w,
                            num_inference_steps=int(steps),
                            guidance_scale=0.0,
                            generator=generator,
                            num_images_per_prompt=1,
                        ).images[0]
                    mode_str = f"img2img ({len(processed_images)} ref)"

                # C. Txt2Img Path (Fallback)
                if image is None:
                    image = pipe(
                        **get_pipe_kwargs(pipe),
                        height=int(height), width=int(width),
                        num_inference_steps=int(steps),
                        guidance_scale=final_guidance,
                        generator=generator,
                        num_images_per_prompt=1,
                    ).images[0]
                    mode_str = "txt2img"

        except Exception as e:
            if pulid_patch: pulid_patch.unpatch()
            return None, f"Generation failed: {e}", None

        if pulid_patch: pulid_patch.unpatch()
        if self.stop_requested: return None, "Cancelled.", None

        # 8. Post-processing (FaceSwap)
        if enable_faceswap and faceswap_source_image and image:
            try:
                from src.image.faceswap_helper import get_faceswap_helper
                swapper = get_faceswap_helper(device=device)
                if not swapper.is_loaded: swapper.load_models()
                image = swapper.swap_face(
                    target_image=image, source_image=faceswap_source_image,
                    face_index=int(faceswap_target_index), use_similarity=True
                )
            except Exception as e:
                print(f"  Warning: Face swap failed: {e}")

        # 9. Cleanup & Return
        self._cleanup(device)

        fallback_info = f" | Note: {self.pm.last_model_fallback_reason}" if self.pm.last_model_fallback_reason else ""
        status = f"Seed: {seed} | Mode: {mode_str} | Model: {current_model} | Device: {device}{fallback_info}"
        return_pose = pose_image if (show_pose_skeleton and pose_image) else None

        return image, status, return_pose

    def _scale_dims(self, w, h, factor):
        base_w = self._safe_int(w, 512)
        base_h = self._safe_int(h, 512)
        scale = self._parse_downscale_factor(factor)
        if scale <= 1.0:
            # Still align to 16 even without downscale — Z-Image VAE
            # requires dimensions divisible by vae_scale_factor*2 = 16.
            return max(256, (base_w // 16) * 16), max(256, (base_h // 16) * 16)
        # FLUX VAE requires dimensions divisible by vae_scale_factor*2 = 16.
        # Round down to nearest 16 to avoid latent shape mismatches.
        align = 16
        new_w = int((base_w / scale) // align) * align
        new_h = int((base_h / scale) // align) * align
        return max(256, new_w), max(256, new_h)

    def _safe_int(self, value, default: int) -> int:
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

    def _parse_downscale_factor(self, value) -> float:
        if value is None:
            return 1.0
        if isinstance(value, (int, float)):
            factor = float(value)
        else:
            text = str(value).strip().lower()
            if not text:
                return 1.0
            if text.endswith("x"):
                text = text[:-1].strip()
            match = re.search(r"\d+(?:\.\d+)?", text)
            if match:
                text = match.group(0)
            try:
                factor = float(text)
            except ValueError:
                return 1.0
        return factor if factor > 0 else 1.0

    def _normalize_downscale_factor(self, value) -> str:
        factor = self._parse_downscale_factor(value)
        if factor <= 1.0:
            return "1x"
        return f"{factor:g}x"

    def _cleanup(self, device):
        # Lightweight cleanup: avoid gc.collect() and empty_cache() after every
        # generation as they force CUDA to release cached blocks, causing
        # re-allocation overhead on the next generation. Heavy cleanup is
        # handled during model switches in PipelineManager.load_pipeline().
        pass
