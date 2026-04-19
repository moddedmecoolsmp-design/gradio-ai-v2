"""
Gradio handlers for the Upscale tab.

The UI lives in ``src/ui/gradio_app.py``; this module contains the pure
event handlers (no Gradio imports) so they can be tested independently of
the UI framework.
"""

from __future__ import annotations

import logging
import __main__
from typing import Optional, Tuple

from PIL import Image

from src.image.upscaler import (
    DEFAULT_MODEL,
    KLEIN_HIRES_LORA_MODEL_KEY,
    MODELS,
    get_upscaler,
)

logger = logging.getLogger(__name__)


def _run_klein_hires_lora_upscale(
    image: Image.Image,
    target_scale: float,
    device: str,
    progress=None,
) -> Tuple[Optional[Image.Image], str]:
    """
    Quality-first upscaler path: run the Klein High-Resolution LoRA
    (Civitai v2739957) through a FLUX.2-klein img2img pass.

    This is intentionally slower than ESRGAN (~5 s vs ~1 s on a 3070)
    but "preserves everything" (the LoRA's trained property — it barely
    deviates from the input composition). The weights auto-download on
    first call via ``PipelineManager.ensure_builtin_lora_downloaded``.

    Run via the global :class:`ImageGenerator` instance because that's
    where the FLUX.2-klein pipeline warmup, device placement and
    LoRA-stacking logic already live. Lazy-imported to keep
    ``upscaler_ui`` importable at app boot (app.py imports this module
    before instantiating the generator).
    """
    _app = __main__

    gen = getattr(_app, "gen", None)
    pm = getattr(_app, "pipeline_manager", None)
    if gen is None or pm is None:
        return None, (
            "Klein Hi-Res LoRA upscale requires the main generator; run "
            "it through the inline 'Upscale after generation' accordion "
            "instead."
        )

    try:
        from src.constants import (
            KLEIN_HIRES_LORA_STRENGTH,
            KLEIN_HIRES_LORA_TRIGGER,
        )
        # Route through the canonical 4-bit SDNQ model_choice constant.
        # A previous revision hard-coded the string "FLUX.2 [klein] 4B
        # (SDNQ Int4)" which doesn't match any entry in
        # ``runtime_policies.MODEL_CHOICES`` — ``load_pipeline`` then
        # fell through the dispatch chain and loaded the Int8 build
        # while ``get_model_repos_for_choice`` downloaded SDNQ repos,
        # a mismatch that OOMs low-VRAM users. Using the constant
        # keeps them in lock-step.
        from src.runtime_policies import LOW_VRAM_FLUX_MODEL_CHOICE

        lora_path = pm.ensure_builtin_lora_downloaded("klein_hires", progress)
        if not lora_path:
            return None, (
                "Klein Hi-Res LoRA download failed. Check connectivity "
                "and try again."
            )

        # Scale the target resolution by ``target_scale``; the LoRA is a
        # refiner, so the effective upscale comes from the resolution
        # bump, not from the model multiplying dimensions itself.
        scale = float(target_scale) if target_scale else 4.0
        new_w = int(image.width * scale)
        new_h = int(image.height * scale)
        # VAE requires dims divisible by 16. Align down.
        new_w = (new_w // 16) * 16 or 16
        new_h = (new_h // 16) * 16 or 16

        upscaled_input = image.resize((new_w, new_h), Image.LANCZOS)

        # Run the short img2img refine. We reuse the generator's
        # ``generate`` entry point because it already handles
        # quantization, device offload, attention cascade, LoRA
        # stacking and preservation gates.
        prompt = f"{KLEIN_HIRES_LORA_TRIGGER}. highly detailed, sharp focus"
        result = gen.generate(
            prompt=prompt,
            negative_prompt="",
            height=new_h,
            width=new_w,
            steps=4,
            seed=-1,
            guidance=1.0,
            device=device,
            model_choice=LOW_VRAM_FLUX_MODEL_CHOICE,
            input_images=[upscaled_input],
            img2img_strength=0.35,  # light refine — preserve content
            lora_file=lora_path,
            lora_strength=KLEIN_HIRES_LORA_STRENGTH,
            # Suppress the inner generate()'s library-driven face swap.
            # Two reasons:
            #   - Inline path (Generate tab): the outer generate() already
            #     ran post_process_with_library; running it again would
            #     double-swap the same faces, degrading identity + adding
            #     ~1 s/face.
            #   - Standalone path (Upscale tab): the user clicked "Run
            #     Upscale" on an arbitrary image — they never asked for a
            #     face swap, so silently swapping faces because
            #     auto-swap-from-library is globally enabled would be a
            #     surprise behaviour change on their upload.
            skip_face_swap=True,
            progress_callback=progress,
        )
        # Strip the Hi-Res LoRA from the shared pipeline before returning.
        # Without this, ``pm.current_lora_path`` still references the Klein
        # Hi-Res LoRA (strength 0.9, trigger word "High resolution") and
        # leaks into the *next* outer ``generate()`` call whenever the
        # user doesn't pass ``lora_file=`` / ``enable_klein_anatomy_fix=`` /
        # ``enable_expression_transfer=`` (both ``image_gen.generate`` and
        # ``pipeline_manager.load_pipeline`` short-circuit when the model
        # doesn't change, so no implicit cleanup happens). Unload here so
        # the upscaler is a hermetic post-processing step.
        try:
            pm.unload_lora()
        except Exception:
            # Best-effort — never fail the upscale because cleanup hiccuped.
            pass
        # ``generate`` returns (image, status_or_seed_info, …) — peel.
        if isinstance(result, tuple) and len(result) >= 1:
            out_img = result[0]
        else:
            out_img = result
        if not isinstance(out_img, Image.Image):
            return None, (
                "Klein Hi-Res LoRA upscale did not produce an image. "
                "See console for generator status."
            )
        return out_img, (
            f"Upscaled with Klein High-Resolution LoRA "
            f"({image.width}x{image.height} -> {out_img.width}x{out_img.height}) "
            f"on {device}."
        )
    except Exception as exc:
        logger.exception("Klein Hi-Res LoRA upscale failed")
        return None, f"Klein Hi-Res LoRA upscale failed: {exc}"


def run_upscale(
    image: Optional[Image.Image],
    model_key: str,
    target_scale: float,
    tile: int,
    device: str,
    progress=None,
) -> Tuple[Optional[Image.Image], str]:
    """
    Run ``model_key`` on ``image`` and return (upscaled_image, status).

    ``target_scale`` lets the user ask for a non-native scale (e.g. a 4x
    model scaled to 2x). ``tile`` caps VRAM by running the model on
    overlapping crops. Failures are reported in the status message rather
    than raised so the Gradio call chain doesn't 500.

    Dispatches to the Klein High-Resolution LoRA img2img path when
    ``model_key`` matches the sentinel — that path runs through the
    main FLUX.2-klein generator (slower, quality-first) and ignores the
    ``tile`` parameter.
    """
    if image is None:
        return None, "Upload or send an image to the Upscaler first."

    if model_key == KLEIN_HIRES_LORA_MODEL_KEY:
        return _run_klein_hires_lora_upscale(image, target_scale, device, progress)

    if model_key not in MODELS:
        return None, f"Unknown upscaler model: {model_key}."

    try:
        helper = get_upscaler(device=device)
        out = helper.upscale(
            image,
            model_key=model_key,
            target_scale=float(target_scale) if target_scale else None,
            tile=int(tile) if tile else 512,
            progress=progress,
        )
    except Exception as exc:
        logger.exception("Upscale failed")
        return None, f"Upscale failed: {exc}"

    status = (
        f"Upscaled with {model_key} "
        f"({image.width}x{image.height} -> {out.width}x{out.height}) on {device}."
    )
    return out, status


def send_generated_to_upscaler(generated: Optional[Image.Image]) -> Optional[Image.Image]:
    """Pass-through used by the Generate tab's 'Send to Upscaler' button."""
    return generated


def update_upscale_resolution_info(image, target_scale):
    """
    Update the resolution display for the upscaler.
    Returns (input_resolution_text, output_preview_text).
    """
    if image is None:
        return (
            "**Input Resolution:** No image uploaded",
            "**Output Preview:** Upload an image to see preview"
        )
    
    try:
        img_width, img_height = image.size
        scale = float(target_scale) if target_scale else 4.0
        
        # Calculate output dimensions
        out_width = int(img_width * scale)
        out_height = int(img_height * scale)
        
        input_text = f"**Input Resolution:** {img_width}×{img_height}"
        output_text = f"**Output Preview:** {out_width}×{out_height} ({scale:g}x upscale)"
        
        return input_text, output_text
    except Exception:
        return (
            "**Input Resolution:** Unable to determine",
            "**Output Preview:** Unable to calculate"
        )


__all__ = [
    "run_upscale",
    "send_generated_to_upscaler",
    "update_upscale_resolution_info",
    "MODELS",
    "DEFAULT_MODEL",
]
