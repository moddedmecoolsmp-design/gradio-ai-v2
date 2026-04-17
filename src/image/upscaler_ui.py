"""
Gradio handlers for the Upscale tab.

The UI lives in ``src/ui/gradio_app.py``; this module contains the pure
event handlers (no Gradio imports) so they can be tested independently of
the UI framework.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from PIL import Image

from src.image.upscaler import DEFAULT_MODEL, MODELS, get_upscaler

logger = logging.getLogger(__name__)


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
    """
    if image is None:
        return None, "Upload or send an image to the Upscaler first."

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


__all__ = [
    "run_upscale",
    "send_generated_to_upscaler",
    "MODELS",
    "DEFAULT_MODEL",
]
