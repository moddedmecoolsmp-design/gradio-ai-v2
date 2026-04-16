"""Dimension calculation utilities for image generation.

Centralizes resolution scaling, downscale factor parsing, and aspect-ratio
helpers that were previously inline in app.py.
"""

import re
from typing import Tuple

from src.core.async_batch_integration import calculate_dimensions_from_ratio
from src.runtime_policies import (
    resolve_default_resolution_preset,
)
from src.utils.device_utils import get_device_vram_gb


def get_device_gpu_name(device: str):
    """Get the GPU name for the given device."""
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def resolve_resolution_preset_for_model(
    model_choice: str, mode: str = "single", device: str = "cuda"
) -> str:
    """Resolve the default resolution preset for a model/device combination."""
    return resolve_default_resolution_preset(
        model_choice=model_choice,
        mode=mode,
        device=device,
        vram_gb=get_device_vram_gb(device),
        gpu_name=get_device_gpu_name(device),
    )


def calculate_dimensions_from_base(
    width: int, height: int, preset: str
) -> Tuple[int, int]:
    """Calculate dimensions from base width/height and a resolution preset."""
    safe_width = max(256, int(width or 1024))
    safe_height = max(256, int(height or 1024))
    return calculate_dimensions_from_ratio(safe_width, safe_height, preset)


def resolve_model_dimensions(
    model_choice: str,
    width: int,
    height: int,
    preset: str = None,
    device: str = "cuda",
) -> Tuple[int, int]:
    """Resolve output dimensions for a model, applying the appropriate preset."""
    target_preset = preset or resolve_resolution_preset_for_model(
        model_choice, mode="single", device=device
    )
    return calculate_dimensions_from_base(width, height, target_preset)


def get_next_lower_dimensions(width: int, height: int) -> Tuple[int, int]:
    """Step down to the next lower standard resolution tier."""
    long_edge = max(int(width), int(height))
    if long_edge > 1280:
        target_edge = 1024
    elif long_edge > 1024:
        target_edge = 768
    elif long_edge > 768:
        target_edge = 640
    else:
        target_edge = long_edge

    if target_edge == long_edge:
        return int(width), int(height)

    aspect_ratio = max(1e-6, float(width) / max(1, int(height)))
    if aspect_ratio >= 1:
        new_width = target_edge
        new_height = int(target_edge / aspect_ratio)
    else:
        new_height = target_edge
        new_width = int(target_edge * aspect_ratio)

    new_width = max(256, (new_width // 64) * 64)
    new_height = max(256, (new_height // 64) * 64)
    return new_width, new_height


def parse_downscale_factor(value) -> float:
    """Parse a downscale factor from various input formats (e.g. '2x', 2, '2.0')."""
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


def normalize_downscale_factor(value) -> str:
    """Normalize a downscale factor to a standard string representation."""
    factor = parse_downscale_factor(value)
    if factor <= 1:
        return "1x"
    return f"{factor:g}x"


def apply_scale_to_dimensions(
    width: int, height: int, downscale_factor
) -> Tuple[int, int]:
    """Apply a downscale factor to dimensions, aligning to 16px (VAE requirement)."""
    factor = parse_downscale_factor(downscale_factor)

    if factor <= 1:
        new_width = (width // 16) * 16 or 16
        new_height = (height // 16) * 16 or 16
        return max(256, new_width), max(256, new_height)

    new_width = max(256, int(width / factor))
    new_height = max(256, int(height / factor))

    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16

    new_width = max(256, min(2048, new_width))
    new_height = max(256, min(2048, new_height))

    return new_width, new_height


def format_downscale_preview(width: int, height: int, downscale_factor) -> str:
    """Format a human-readable downscale preview string."""
    factor = parse_downscale_factor(downscale_factor)
    scaled_w, scaled_h = apply_scale_to_dimensions(width, height, factor)
    if factor <= 1:
        return f"{width}x{height} (no downscale)"
    return f"{width}x{height} → {scaled_w}x{scaled_h} ({factor:g}x downscale)"


def calculate_dimensions_from_target_size(
    width: int, height: int, target_size: int
) -> Tuple[int, int]:
    """Calculate output dimensions maintaining aspect ratio for a target longest side."""
    aspect_ratio = width / height

    if aspect_ratio >= 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64

    new_width = max(256, min(2048, new_width))
    new_height = max(256, min(2048, new_height))

    return new_width, new_height
