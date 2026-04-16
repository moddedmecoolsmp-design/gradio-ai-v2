"""
Z-Image Turbo UINT4 - Fast Image Generation on Mac

Uses the quantized uint4 model (only 3.5GB!) for fast inference on Apple Silicon.
Now with LoRA support!
"""

import os
import argparse
import contextlib

# Enable fast-math for MPS
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

import torch
from diffusers import ZImagePipeline, FlowMatchEulerDiscreteScheduler


compiled_zimage = False

# Flash Attention backend preference order: flash_attention_2 > sdpa > eager
_FLASH_ATTN_BACKEND = None


def _detect_flash_attention_backend():
    """Detect best available attention backend for this runtime."""
    global _FLASH_ATTN_BACKEND
    if _FLASH_ATTN_BACKEND is not None:
        return _FLASH_ATTN_BACKEND

    if torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            _FLASH_ATTN_BACKEND = "flash_attention_2"
            return _FLASH_ATTN_BACKEND
        except ImportError:
            pass
        # SDPA (scaled dot-product attention) is built into PyTorch >= 2.0
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            _FLASH_ATTN_BACKEND = "sdpa"
            return _FLASH_ATTN_BACKEND

    _FLASH_ATTN_BACKEND = "eager"
    return _FLASH_ATTN_BACKEND


def _apply_flash_attention(pipe, device):
    """
    Apply Flash Attention to transformer/unet components where supported.

    flash_attention_2 delivers 2.49x-7.47x speedup over standard attention on CUDA.
    SDPA (PyTorch built-in) delivers ~1.5-2x speedup as a fallback.
    On MPS/CPU we skip Flash Attention (not supported) and rely on attention slicing.
    """
    if device not in ("cuda",):
        return False

    backend = _detect_flash_attention_backend()
    if backend == "eager":
        return False

    applied = False
    for component_name in ("transformer", "unet"):
        component = getattr(pipe, component_name, None)
        if component is None:
            continue
        try:
            if backend == "flash_attention_2":
                component.enable_flash_attention()
                applied = True
                print(f"  Flash Attention 2 enabled on {component_name} (2.49x-7.47x speedup)")
            elif backend == "sdpa":
                if hasattr(component, "set_attn_processor"):
                    from diffusers.models.attention_processor import AttnProcessor2_0
                    component.set_attn_processor(AttnProcessor2_0())
                    applied = True
                    print(f"  SDPA attention enabled on {component_name}")
        except Exception as exc:
            print(f"  Flash Attention ({backend}) skipped for {component_name}: {exc}")

    return applied


def compile_pipeline_components(pipe, device):
    """Apply torch.compile to pipeline components for RTX 3070 Ampere optimization."""
    global compiled_zimage
    if os.name == "nt":
        print("torch.compile disabled on Windows (requires Triton which is not natively supported)")
        return False
    if compiled_zimage or device != "cuda" or not torch.cuda.is_available():
        return False

    capability = torch.cuda.get_device_capability(0)
    if not capability or capability[0] < 8:
        return False

    try:
        if getattr(pipe, "transformer", None) is not None:
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        if getattr(pipe, "unet", None) is not None:
            pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
        compiled_zimage = True
        return True
    except Exception as exc:
        print(f"torch.compile skipped: {exc}")
        return False


def load_pipeline(device="mps"):
    """Load the full-precision Z-Image pipeline."""
    print("Loading Z-Image-Turbo (full precision)...")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"PyTorch version: {torch.__version__}")

    # Use bfloat16 for CUDA/MPS (Ampere Tensor Cores) and float32 for CPU
    dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # Use Euler with beta sigmas for cleaner images
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        use_beta_sigmas=True,
    )

    pipe.to(device)

    # Memory optimizations
    pipe.enable_attention_slicing()

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
        print("VAE slicing enabled")

    if hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
        print("VAE tiling enabled")

    # Flash Attention: 2.49x-7.47x speedup on CUDA with Flash Attention 2,
    # or ~1.5-2x speedup via PyTorch SDPA fallback.
    _apply_flash_attention(pipe, device)

    if compile_pipeline_components(pipe, device):
        print("torch.compile enabled for Z-Image pipeline")

    print("Pipeline loaded!")
    return pipe


def generate(
    pipe,
    prompt: str,
    height: int = 512,
    width: int = 512,
    steps: int = 5,
    seed: int = None,
    device: str = "mps",
):
    """Generate an image from a prompt."""
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    print(f"Generating with seed {seed}...")

    # Both FLUX and Z-Image VAEs require dimensions divisible by 16.
    height = (int(height) // 16) * 16 or 16
    width = (int(width) // 16) * 16 or 16

    # Use appropriate generator for device
    if device == "cuda":
        generator = torch.Generator("cuda").manual_seed(seed)
    elif device == "mps":
        generator = torch.Generator("mps").manual_seed(seed)
    else:
        generator = torch.Generator().manual_seed(seed)

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Disable autocast for SDNQ models which are incompatible with bfloat16 autocast kernels
    is_sdnq = "sdnq" in str(type(pipe)).lower() or "sdnq" in str(getattr(pipe, "transformer", "")).lower()
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if device == "cuda" and not is_sdnq
        else contextlib.nullcontext()
    )

    with torch.inference_mode(), autocast_ctx:
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
            num_images_per_prompt=1,
        )
        image = result.images[0]
        # Release pipeline output tensors immediately to free VRAM before returning
        del result

    # Lightweight post-generation cleanup (non-blocking)
    import gc
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return image, seed


def main():
    parser = argparse.ArgumentParser(description="Generate images with Z-Image Turbo UINT4")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--height", type=int, default=512, help="Image height (default: 512)")
    parser.add_argument("--width", type=int, default=512, help="Image width (default: 512)")
    parser.add_argument("--steps", type=int, default=5, help="Inference steps (default: 5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="output.png", help="Output path")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps, cuda, cpu)")

    # LoRA arguments
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA safetensors file")
    parser.add_argument("--lora-strength", type=float, default=1.0, help="LoRA strength (default: 1.0)")

    args = parser.parse_args()

    print("Environment optimizations:")
    print(f"  TORCHINDUCTOR_MAX_AUTOTUNE={os.getenv('TORCHINDUCTOR_MAX_AUTOTUNE', 'unset')}")
    print(f"  TORCHINDUCTOR_FREEZING={os.getenv('TORCHINDUCTOR_FREEZING', 'unset')}")
    print(f"  PYTORCH_CUDA_ALLOC_CONF={os.getenv('PYTORCH_CUDA_ALLOC_CONF', 'unset')}")

    # Determine device
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    pipe = load_pipeline(device)

    # Load LoRA if specified (using native diffusers support)
    if args.lora:
        if not os.path.exists(args.lora):
            print(f"Error: LoRA file not found: {args.lora}")
            return

        print(f"Loading LoRA: {args.lora} (strength={args.lora_strength})")
        try:
            pipe.load_lora_weights(args.lora, adapter_name="default")
            pipe.set_adapters(["default"], adapter_weights=[args.lora_strength])
            print("LoRA loaded successfully!")
        except Exception as e:
            print(f"Error loading LoRA: {e}")
            return

    image, seed = generate(
        pipe,
        args.prompt,
        args.height,
        args.width,
        args.steps,
        args.seed,
        device,
    )

    image.save(args.output)
    lora_info = f", LoRA: {os.path.basename(args.lora)}" if args.lora else ""
    print(f"Saved to {args.output} (seed: {seed}{lora_info})")


if __name__ == "__main__":
    main()
