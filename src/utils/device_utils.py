import torch
import os
import sys

def get_memory_usage():
    """Get current memory usage in GB."""
    if torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / 1024**3
        except Exception:
            return 0
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0

def print_memory(label):
    """Print memory usage with label."""
    mem = get_memory_usage()
    print(f"  [MEM] {label}: {mem:.2f} GB")

def get_available_devices():
    """Get list of available devices. Prioritizes CUDA for NVIDIA GPUs."""
    devices = []

    # Check CUDA first (for NVIDIA GPUs like RTX 3070)
    if torch.cuda.is_available():
        devices.append("cuda")
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"CUDA detected: {device_name} (VRAM: {vram_gb:.1f} GB)")
    else:
        is_cpu_only = "+cpu" in torch.__version__
        if is_cpu_only:
            print("WARNING: PyTorch CPU-only version detected. CUDA support requires CUDA-enabled PyTorch.")
        elif not torch.cuda.is_available():
            print("CUDA not available. Check NVIDIA drivers and CUDA installation.")

    # MPS for Apple Silicon (Mac)
    if torch.backends.mps.is_available():
        devices.append("mps")

    # CPU always available as fallback
    devices.append("cpu")
    return devices

def get_device_vram_gb(device: str):
    """Get total VRAM in GB for the given device."""
    if device != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        return None

def is_cuda13_runtime(runtime_version: str) -> bool:
    """Check if the provided runtime version string matches CUDA 13.x."""
    if not runtime_version:
        return False
    return runtime_version.startswith("13.")
