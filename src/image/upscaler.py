"""
Image upscaling helper.

Wraps ESRGAN-family upscalers (Real-ESRGAN x4plus, x2plus, the anime
variant, and 4x-UltraSharp) behind a single ``UpscalerHelper`` interface.
Weights are auto-downloaded to a per-user cache on first use and the model
is kept resident until explicitly unloaded.

Backend strategy
~~~~~~~~~~~~~~~~
1. **Primary**: `spandrel <https://github.com/chaiNNer-org/spandrel>`_.
   Spandrel auto-detects the architecture from the ``.pth`` checkpoint
   and returns a ready-to-run model without the user having to pick an
   arch. It's the canonical loader used by ChaiNNer / ComfyUI nodes.
2. **Fallback**: bare ``RRDBNet`` from ``basicsr.archs.rrdbnet_arch`` if
   spandrel isn't available. Real-ESRGAN / UltraSharp are all RRDBNet
   variants so this covers every shipped model.

Tiled inference is implemented in pure torch (no extra deps) so we don't
OOM on RTX 3070 when upscaling 1024x1024 → 4096x4096.

All CUDA work happens in FP16 when the device is CUDA (ESRGAN networks
tolerate FP16 inference with zero visible quality loss and ~2× speedup
on Ampere).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry. Each entry is a (url, scale, arch_hint) tuple.
# arch_hint is only used by the fallback loader (spandrel auto-detects).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UpscalerModel:
    name: str
    url: str
    scale: int
    # ``num_block`` is the only RRDBNet hyperparameter that varies across
    # the Real-ESRGAN checkpoints (x4plus = 23, anime_6B = 6).
    num_block: int = 23
    # HuggingFace coordinates are the *primary* download path when
    # available — ``huggingface_hub.hf_hub_download`` is resumable,
    # CDN-cached and shares the cache with every other model the app
    # already pulls from HF (one ``HF_HOME`` rules them all). ``url`` is
    # retained as the offline fallback (e.g. corp networks that block
    # huggingface.co but allow github.com releases). The Real-ESRGAN
    # xinntao checkpoints have no stable HF mirror, so they keep the
    # GitHub URL only. 4x-UltraSharp is on HF (lokCX/4x-Ultrasharp) so
    # it uses hf_hub_download by default.
    hf_repo: Optional[str] = None
    hf_filename: Optional[str] = None


# Special key for the FLUX.2-klein High-Resolution LoRA upscaler path.
# Dispatched through ``src.image.lora_upscaler`` which runs a full
# FLUX.2 img2img pass; ~5x slower than ESRGAN but preserves content
# essentially pixel-for-pixel.
KLEIN_HIRES_LORA_MODEL_KEY = "Klein High-Resolution LoRA (img2img, quality)"


MODELS: Dict[str, UpscalerModel] = {
    "Real-ESRGAN x4plus": UpscalerModel(
        name="RealESRGAN_x4plus",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        scale=4,
        num_block=23,
    ),
    "Real-ESRGAN x2plus": UpscalerModel(
        name="RealESRGAN_x2plus",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        scale=2,
        num_block=23,
    ),
    "Real-ESRGAN x4plus anime": UpscalerModel(
        name="RealESRGAN_x4plus_anime_6B",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        scale=4,
        num_block=6,
    ),
    "4x-UltraSharp": UpscalerModel(
        name="4x-UltraSharp",
        hf_repo="lokCX/4x-Ultrasharp",
        hf_filename="4x-UltraSharp.pth",
        url="https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth",
        scale=4,
        num_block=23,
    ),
}

DEFAULT_MODEL = "Real-ESRGAN x4plus"


def get_all_model_choices() -> list[str]:
    """
    Return every upscaler choice for the UI dropdown, including the
    FLUX.2-klein High-Resolution LoRA virtual entry. The LoRA path is
    handled by ``upscaler_ui`` (not by :class:`UpscalerHelper`) so we
    don't put it in ``MODELS`` — it doesn't have a ``.pth`` and isn't
    an RRDBNet.
    """
    return list(MODELS.keys()) + [KLEIN_HIRES_LORA_MODEL_KEY]


def get_models() -> list[str]:
    """Return the list of upscaler display names for the UI dropdown."""
    return list(MODELS.keys())


def _cache_dir() -> Path:
    """Return the per-user weights cache directory, creating it if needed."""
    override = os.environ.get("UFIG_UPSCALER_CACHE_DIR")
    if override:
        root = Path(override).expanduser()
    else:
        root = Path.home() / ".cache" / "ufig-upscalers"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _download_weights(model: UpscalerModel) -> Path:
    """
    Download ``model`` into the cache dir if not already present.

    HuggingFace is the preferred path when ``model.hf_repo`` is set — it's
    resumable, shares ``HF_HOME`` with the FLUX.2-klein base model (so
    users who already run FLUX don't have to warm a second cache), and
    is much faster than raw GitHub releases on Windows (HF CDN +
    connection pooling vs single-connection urllib on github.com). We
    fall back to urllib on ``model.url`` if ``huggingface_hub`` isn't
    installed or the HF download fails (HF_HUB_OFFLINE, rate-limit,
    corporate proxy, etc.).
    """
    target = _cache_dir() / f"{model.name}.pth"
    if target.exists() and target.stat().st_size > 0:
        return target

    # --- Primary path: huggingface_hub.hf_hub_download (when available).
    if model.hf_repo and model.hf_filename:
        try:
            from huggingface_hub import hf_hub_download  # type: ignore

            logger.info(
                "Upscaler: downloading %s from hf://%s/%s",
                model.name,
                model.hf_repo,
                model.hf_filename,
            )
            hf_path = hf_hub_download(
                repo_id=model.hf_repo,
                filename=model.hf_filename,
            )
            # ``hf_hub_download`` returns a path in the HF cache; copy it
            # into our own cache so callers don't have to care whether
            # HF_HOME is set. ``Path.replace`` across filesystems is
            # unsafe, so we fall back to a tiny copy loop.
            import shutil

            shutil.copyfile(hf_path, target)
            return target
        except Exception as exc:
            logger.warning(
                "Upscaler: hf_hub_download failed for %s (%s); falling back to urllib.",
                model.name,
                exc,
            )

    # --- Fallback: plain urllib download from ``model.url``.
    logger.info("Upscaler: downloading %s from %s", model.name, model.url)
    tmp = target.with_suffix(".pth.partial")
    try:
        # Simple urllib download — weights files are 60-70 MB, no need for
        # anything fancier, and we avoid adding requests / tqdm deps.
        with urllib.request.urlopen(model.url, timeout=300) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        tmp.rename(target)
    except Exception:
        # Clean up partial file so the next attempt redownloads cleanly.
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return target


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def _load_with_spandrel(weights_path: Path, device: str):
    """Return a callable (tensor -> tensor) using spandrel, or None on failure."""
    try:
        from spandrel import ModelLoader  # type: ignore
    except Exception as exc:
        logger.debug("spandrel unavailable (%s); will try fallback loader.", exc)
        return None

    try:
        loaded = ModelLoader().load_from_file(str(weights_path))
        model = loaded.model
        model.eval()
        if device.startswith("cuda"):
            model = model.to(device=device, dtype=torch.float16)
        else:
            model = model.to(device=device)
        return model
    except Exception as exc:
        logger.warning("spandrel failed to load %s: %s", weights_path.name, exc)
        return None


def _load_with_rrdbnet(weights_path: Path, device: str, num_block: int, scale: int):
    """Fallback: bare RRDBNet from basicsr (the arch Real-ESRGAN uses)."""
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
    except Exception as exc:
        logger.warning("basicsr unavailable (%s); upscaler cannot load.", exc)
        return None

    net = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=num_block,
        num_grow_ch=32,
        scale=scale,
    )
    state = torch.load(str(weights_path), map_location="cpu")
    # Real-ESRGAN checkpoints wrap the state dict under "params_ema" or "params".
    for key in ("params_ema", "params"):
        if key in state:
            state = state[key]
            break
    net.load_state_dict(state, strict=True)
    net.eval()
    if device.startswith("cuda"):
        net = net.to(device=device, dtype=torch.float16)
    else:
        net = net.to(device=device)
    return net


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------
def _to_tensor(image: Image.Image, device: str) -> torch.Tensor:
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if device.startswith("cuda"):
        tensor = tensor.to(device=device, dtype=torch.float16)
    else:
        tensor = tensor.to(device=device)
    return tensor


def _from_tensor(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.squeeze(0).clamp(0, 1).float().cpu().numpy()
    arr = (arr.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _tiled_forward(
    model,
    tensor: torch.Tensor,
    scale: int,
    tile: int = 512,
    pad: int = 16,
) -> torch.Tensor:
    """
    Split ``tensor`` into overlapping tiles, run ``model`` on each, stitch.

    Overlap = 2*``pad`` in each direction (trimmed after upscale) to hide
    seam artifacts. Memory footprint scales with (tile+2*pad)² * scale²
    rather than the full image resolution — this is what lets RTX 3070
    upscale a 1024² image to 4096² without OOM.
    """
    _, _, h, w = tensor.shape
    if h <= tile and w <= tile:
        with torch.inference_mode():
            return model(tensor)

    out = torch.zeros(
        (tensor.shape[0], tensor.shape[1], h * scale, w * scale),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y1 = max(y - pad, 0)
            x1 = max(x - pad, 0)
            y2 = min(y + tile + pad, h)
            x2 = min(x + tile + pad, w)
            chunk = tensor[:, :, y1:y2, x1:x2]
            with torch.inference_mode():
                up = model(chunk)
            # Figure out the slice of ``up`` that corresponds to the
            # non-padded region of the original tile (strip the pad).
            crop_y1 = (y - y1) * scale
            crop_x1 = (x - x1) * scale
            tile_h = min(tile, h - y) * scale
            tile_w = min(tile, w - x) * scale
            crop_y2 = crop_y1 + tile_h
            crop_x2 = crop_x1 + tile_w
            out[:, :, y * scale : y * scale + tile_h, x * scale : x * scale + tile_w] = (
                up[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
            )
    return out


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------
class UpscalerHelper:
    """Load and run ESRGAN-family upscalers on a single image."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._loaded_key: Optional[str] = None
        self._model = None
        self._scale = 1
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def load(self, model_key: str) -> None:
        """Ensure the requested model is resident on ``self.device``."""
        if model_key not in MODELS:
            raise ValueError(
                f"Unknown upscaler '{model_key}'. Choices: {list(MODELS)}"
            )

        with self._lock:
            if self._loaded_key == model_key and self._model is not None:
                return

            if self._model is not None:
                # Free the previous model before loading a different one.
                # Must call the unlocked helper — ``self._lock`` is a plain
                # (non-reentrant) ``threading.Lock`` and re-entering it via
                # ``self.unload()`` would deadlock.
                self._unload_unlocked()

            spec = MODELS[model_key]
            weights_path = _download_weights(spec)

            model = _load_with_spandrel(weights_path, self.device)
            if model is None:
                model = _load_with_rrdbnet(
                    weights_path,
                    self.device,
                    num_block=spec.num_block,
                    scale=spec.scale,
                )
            if model is None:
                raise RuntimeError(
                    f"Failed to load upscaler '{model_key}' — "
                    "install spandrel or basicsr to enable upscaling."
                )

            self._model = model
            self._scale = spec.scale
            self._loaded_key = model_key
            logger.info(
                "Upscaler ready: %s (scale=%dx) on %s",
                model_key,
                spec.scale,
                self.device,
            )

    # ------------------------------------------------------------------
    def _unload_unlocked(self) -> None:
        """Inner cleanup shared by ``unload()`` and ``load()``.

        Assumes ``self._lock`` is already held by the caller. Kept as a
        private helper so ``load()`` can reuse the cleanup path without
        re-acquiring the non-reentrant ``threading.Lock`` (which would
        deadlock).
        """
        if self._model is not None:
            try:
                self._model.to("cpu")
            except Exception:
                pass
            del self._model
        self._model = None
        self._loaded_key = None
        if self.device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def unload(self) -> None:
        """Release the model and any CUDA memory it held."""
        with self._lock:
            self._unload_unlocked()

    # ------------------------------------------------------------------
    def upscale(
        self,
        image: Image.Image,
        model_key: str = DEFAULT_MODEL,
        target_scale: Optional[float] = None,
        tile: int = 512,
        pad: int = 16,
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> Image.Image:
        """
        Upscale ``image`` with ``model_key``.

        If ``target_scale`` is supplied and differs from the model's native
        scale, the output is LANCZOS-resized to match (e.g. 4x model +
        target_scale=2 → output resized to 2x).
        """
        if image is None:
            raise ValueError("Upscale received an empty image.")

        if progress is not None:
            progress(0.05, f"Loading {model_key}...")
        self.load(model_key)
        assert self._model is not None

        if progress is not None:
            progress(0.25, "Upscaling (tiled inference)...")

        tensor = _to_tensor(image, self.device)
        result = _tiled_forward(self._model, tensor, self._scale, tile=tile, pad=pad)
        out = _from_tensor(result)

        # Adjust to target_scale if asked.
        if target_scale is not None and abs(target_scale - self._scale) > 1e-3:
            new_w = int(round(image.width * target_scale))
            new_h = int(round(image.height * target_scale))
            out = out.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if progress is not None:
            progress(1.0, "Done.")
        return out


# ---------------------------------------------------------------------------
# Module-level singleton + entry point for gradio handlers
# ---------------------------------------------------------------------------
_SINGLETON: Optional[UpscalerHelper] = None
_SINGLETON_LOCK = threading.Lock()


def get_upscaler(device: str = "cuda") -> UpscalerHelper:
    """Thread-safe accessor; recreates if device changes."""
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None or _SINGLETON.device != device:
            if _SINGLETON is not None:
                _SINGLETON.unload()
            _SINGLETON = UpscalerHelper(device=device)
        return _SINGLETON


def unload_upscaler() -> None:
    """Drop the cached helper and free its VRAM."""
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is not None:
            _SINGLETON.unload()
            _SINGLETON = None


__all__ = [
    "DEFAULT_MODEL",
    "MODELS",
    "UpscalerHelper",
    "UpscalerModel",
    "get_models",
    "get_upscaler",
    "unload_upscaler",
]
