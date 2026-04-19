"""
First-run auto-installer for optional inference accelerators.

Targets the Windows + RTX 3070 fast path: on the first generation, this
module probes for `sageattention` (SDPA drop-in, INT8 QK + FP16 PV) and
`xformers` (memory-efficient attention), and pip-installs the appropriate
wheel pinned to the user's existing PyTorch build if either is missing.

Note: For CUDA 13.0+, xformers is installed from HuggingFace community wheels
(Wildminder/AI-windows-whl) since official PyTorch wheels don't exist for cu130.
SageAttention is used as a fallback if xformers installation fails.

Design notes:
  * Gated by the `UFIG_AUTO_INSTALL_ACCELERATORS` env var. Default is "1"
    (enabled) on Windows CUDA, "0" elsewhere. Users can always opt out
    with `UFIG_AUTO_INSTALL_ACCELERATORS=0`.
  * Installs run at most once per process (cached) — subsequent calls
    are no-ops regardless of outcome. This avoids surprise pip activity
    mid-session and guarantees idempotency under Gradio's threaded
    request handlers.
  * pip output is captured so the caller can surface a single concise
    status line per package rather than a 200-line progress dump.
  * Never raises: all failures degrade to "skipped" and the caller
    continues with the default SDPA math kernel.
"""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
import threading
from typing import Dict, List, Optional, Tuple

__all__ = [
    "auto_install_accelerators",
    "accelerator_install_status",
    "reset_auto_install_state",
]

_INSTALL_LOCK = threading.Lock()
_INSTALL_STATE: Dict[str, str] = {}  # package_name -> "installed" | "present" | "skipped" | "failed"
_ATTEMPTED = False


def _is_windows() -> bool:
    return platform.system().lower().startswith("win")


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


def _is_cuda_available() -> bool:
    try:
        import torch  # noqa: WPS433 (local import: lazy; torch may be heavy)
        return bool(getattr(torch, "cuda", None)) and torch.cuda.is_available()
    except Exception:
        return False


def _should_attempt(device: Optional[str]) -> bool:
    """Env gate + platform check. Windows CUDA → default on, else default off."""
    default_enabled = _is_windows() and _is_cuda_available()
    if device is not None and str(device).lower() != "cuda":
        return False
    return _bool_env("UFIG_AUTO_INSTALL_ACCELERATORS", default_enabled)


def _module_importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False
    except Exception:
        # Importable but broken install (e.g. dll mismatch). Treat as unavailable
        # so we can reinstall a working version.
        return False


def _xformers_is_compatible() -> bool:
    """
    Check if xformers is installed AND compatible with current PyTorch CUDA version.
    
    Returns True only if xformers can load its C++/CUDA extensions successfully.
    A broken xformers (wrong CUDA version) will return False.
    """
    try:
        import xformers
        import torch
        
        # Get xformers build info
        xformers_version = getattr(xformers, "__version__", "unknown")
        
        # Try to load the C++ extension - this will fail if CUDA versions mismatch
        from xformers import ops
        
        # Check if we can actually use memory_efficient_attention
        # This verifies CUDA extensions are loaded properly by running a minimal test
        if hasattr(ops, "memory_efficient_attention"):
            try:
                # Create minimal test tensors and try to run attention
                # This will actually exercise the CUDA code path and fail if incompatible
                test_q = torch.randn(1, 8, 16, 64, dtype=torch.float16, device="cuda")
                test_k = torch.randn(1, 8, 16, 64, dtype=torch.float16, device="cuda")
                test_v = torch.randn(1, 8, 16, 64, dtype=torch.float16, device="cuda")
                
                # Try to run memory_efficient_attention - this will fail with CUDA mismatch
                _ = ops.memory_efficient_attention(test_q, test_k, test_v)
                
                # Clean up
                del test_q, test_k, test_v
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                return True
            except Exception as e:
                error_msg = str(e)
                if "CUDA" in error_msg or "xFormers" in error_msg:
                    print(f"  [accelerator-installer] xformers CUDA incompatible: {error_msg[:100]}")
                return False
        
        return True
    except ImportError:
        return False
    except Exception as e:
        # xformers is installed but broken (CUDA mismatch)
        print(f"  [accelerator-installer] xformers compatibility check failed: {e}")
        return False


def _pip_install(args: List[str]) -> Tuple[bool, str]:
    """Run `python -m pip install ...` quietly and return (ok, message)."""
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", *args]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return False, f"pip invocation failed: {exc}"

    if result.returncode == 0:
        return True, (result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "ok")

    tail = (result.stderr.strip() or result.stdout.strip()).splitlines()
    message = tail[-1] if tail else f"pip returned {result.returncode}"
    return False, message


def _pip_uninstall(args: List[str]) -> Tuple[bool, str]:
    """Run `python -m pip uninstall ...` quietly and return (ok, message)."""
    cmd = [sys.executable, "-m", "pip", "uninstall", "--disable-pip-version-check", "--no-input", *args]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return False, f"pip invocation failed: {exc}"

    if result.returncode == 0:
        return True, "uninstalled"

    tail = (result.stderr.strip() or result.stdout.strip()).splitlines()
    message = tail[-1] if tail else f"pip returned {result.returncode}"
    return False, message


def _install_sageattention() -> str:
    """
    Install the SageAttention wheel for the current platform.

    The upstream `sageattention` wheels on PyPI require manual CUDA toolkit
    setup; on Windows we prefer the prebuilt `sageattention-windows` wheel
    which ships matched binaries for CUDA 12.x/13.x + PyTorch 2.x.
    """
    if _module_importable("sageattention"):
        return "present"

    pip_target = "sageattention-windows" if _is_windows() else "sageattention"
    ok, message = _pip_install([pip_target])
    if ok:
        return "installed" if _module_importable("sageattention") else "failed"
    # Surface the failure but don't raise — caller falls back to xFormers/SDPA.
    print(f"  [accelerator-installer] {pip_target} install skipped: {message}")
    return "failed"


def _install_xformers() -> str:
    """
    Install xFormers without disturbing the user's existing torch build.

    xFormers wheels on PyPI declare a strict `torch==X.Y.Z` dependency. If
    we let pip resolve normally, it will silently downgrade (or replace
    with a CPU-only wheel) the user's CUDA-enabled torch to satisfy the
    xformers requirement — breaking the rest of the app.

    Strategy:
      1. Check if xformers is installed but incompatible (CUDA version mismatch).
         If so, uninstall it first to prepare for clean installation.
      2. CUDA 13.0+: Install from HuggingFace community wheels
         (Wildminder/AI-windows-whl) since official PyTorch wheels don't
         exist for cu130. These are ABI3 wheels compatible with Python 3.9+.
      3. `--no-deps`: install the xformers wheel without letting pip touch
         torch. If the ABI happens to line up with the user's torch, this
         works. If not, the import will fail and we report "failed" — the
         pipeline already falls back to SDPA/sageattention in that case.
      4. As a conservative second attempt we try the normal resolution,
         which is fine for users with a plain PyPI torch build but may
         shuffle torch on CUDA-pinned environments. We only take this
         second path if the user's torch is a vanilla release (no "+cuXXX"
         local version suffix) so we don't clobber CUDA-enabled builds.
    """
    # The current Windows + Python 3.13 runtime is a known bad fit for the
    # available xformers wheels in this project. Importing the package can
    # hard-crash the interpreter during extension registration, so we skip
    # probing entirely and rely on the safer SDPA/SageAttention path.
    if sys.version_info >= (3, 13):
        return "skipped"

    # Check if xformers is already installed and working
    if _xformers_is_compatible():
        return "present"
    
    # If xformers is installed but incompatible, uninstall it first
    if _module_importable("xformers"):
        print("  [accelerator-installer] xformers installed but incompatible, uninstalling...")
        _pip_uninstall(["xformers", "-y"])

    torch_tag: Optional[str] = None
    torch_cuda_version: Optional[str] = None
    try:
        import torch  # noqa: WPS433
        torch_tag = str(torch.__version__)
        # Extract CUDA version from torch (e.g., "2.10.0+cu130" → "13.0")
        if "+" in torch_tag:
            cuda_part = torch_tag.split("+")[1].replace("cu", "")
            # Convert compact format (e.g., "130") to dotted format (e.g., "13.0")
            if len(cuda_part) == 3 and cuda_part.isdigit():
                torch_cuda_version = f"{cuda_part[:-1]}.{cuda_part[-1]}"
            else:
                torch_cuda_version = cuda_part
    except Exception:
        torch_tag = None
        torch_cuda_version = None

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    # CUDA 13.0+: Use HuggingFace community wheels (no official PyTorch wheels exist)
    # ABI3 wheels work with Python 3.9-3.13+
    if torch_cuda_version and float(torch_cuda_version) >= 13.0:
        # Map PyTorch version to compatible wheel
        torch_ver_short = torch_tag.split("+")[0] if torch_tag and "+" in torch_tag else None
        if torch_ver_short:
            torch_major_minor = ".".join(torch_ver_short.split(".")[:2])  # "2.10.0" -> "2.10"
        else:
            torch_major_minor = "2.10"  # Default fallback

        # Community wheel URLs from Wildminder/AI-windows-whl (HuggingFace)
        # Correct format: xformers-0.0.33+cu130torch2.10-cp39-abi3-win_amd64.whl
        HF_WHEEL_URLS = {
            "2.10": "https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/xformers-0.0.33+cu130torch2.10-cp39-abi3-win_amd64.whl",
            "2.11": "https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/xformers-0.0.33+cu130torch2.11-cp39-abi3-win_amd64.whl",
        }

        wheel_url = HF_WHEEL_URLS.get(torch_major_minor, HF_WHEEL_URLS["2.10"])
        print(f"  [accelerator-installer] Installing xformers for CUDA {torch_cuda_version} from HuggingFace...")
        ok, message = _pip_install([wheel_url, "--no-deps"])
        if ok and _xformers_is_compatible():
            return "installed"
        print(f"  [accelerator-installer] HuggingFace wheel failed: {message}")
        # Fall through to standard install as last resort

    candidates: List[List[str]] = [["xformers", "--no-deps"]]
    # Only allow the unconstrained install on plain PyPI torch builds.
    # Builds with a local version suffix (e.g. "2.10.0+cu130") come from
    # the CUDA-specific index and would be replaced by pip's normal
    # resolver.
    if torch_tag is not None and "+" not in torch_tag:
        candidates.append(["xformers"])

    for args in candidates:
        ok, message = _pip_install(args)
        if ok and _xformers_is_compatible():
            return "installed"
        print(f"  [accelerator-installer] xformers ({' '.join(args)}) skipped: {message}")
    return "failed"


def _install_preservation_stack() -> Dict[str, str]:
    """
    Install ``controlnet_aux`` for DWPose/OpenPose-based pose preservation.

    ``controlnet_aux`` ships the DWposeDetector / OpenposeDetector classes
    that ``src.image.pose_helper`` relies on. The package auto-downloads
    its ONNX weights to the HuggingFace cache on first use, so once the
    pip install completes no further bootstrap is needed.
    """
    results: Dict[str, str] = {}
    if _module_importable("controlnet_aux"):
        results["controlnet_aux"] = "present"
        return results

    ok, message = _pip_install(["controlnet_aux"])
    results["controlnet_aux"] = (
        "installed" if ok and _module_importable("controlnet_aux") else "failed"
    )
    if not ok:
        print(f"  [accelerator-installer] controlnet_aux install skipped: {message}")
    return results


def _install_upscaler_stack() -> Dict[str, str]:
    """
    Install upscaler dependencies — prefer ``spandrel`` (universal arch
    loader used by ChaiNNer / ComfyUI), fall back to ``basicsr`` which
    ships the bare RRDBNet architecture Real-ESRGAN uses.

    We try spandrel first because:
      * it auto-detects the arch from a ``.pth`` file, so users can drop
        in any ESRGAN / SwinIR / HAT checkpoint without a code change;
      * it's a pure Python package with no CUDA build step.

    If spandrel fails (rare — it's pure Python), we fall back to basicsr,
    which is a heavier dependency but covers every Real-ESRGAN variant.
    """
    results: Dict[str, str] = {}
    if _module_importable("spandrel"):
        results["spandrel"] = "present"
        return results

    ok, message = _pip_install(["spandrel"])
    if ok and _module_importable("spandrel"):
        results["spandrel"] = "installed"
        return results

    print(f"  [accelerator-installer] spandrel install skipped: {message}")
    results["spandrel"] = "failed"

    # Fallback: basicsr (RRDBNet only, but battle-tested).
    if _module_importable("basicsr"):
        results["basicsr"] = "present"
        return results
    ok, message = _pip_install(["basicsr"])
    results["basicsr"] = (
        "installed" if ok and _module_importable("basicsr") else "failed"
    )
    if not ok:
        print(f"  [accelerator-installer] basicsr install skipped: {message}")
    return results


def _install_face_swap_stack() -> Dict[str, str]:
    """
    Install the face-swap runtime stack if the user has toggled face swap
    (or any code path that imports ``src.image.faceswap_helper``) and the
    required dependencies aren't present.

    The stack is:
      * ``opencv-python`` (headless fallback) — for image color conversion.
      * ``insightface`` — ships ``FaceAnalysis`` + ``get_model('inswapper_128.onnx')``.
      * ``onnxruntime-gpu`` — CUDA execution provider for inswapper.
        Falls back to ``onnxruntime`` (CPU) if the GPU wheel install fails
        so that the feature at least works degradedly without CUDA.

    We deliberately do **not** pin versions: the project's ``requirements.txt``
    is the canonical source of truth. This installer only exists as a
    safety net for users who have a partial environment (e.g. installed
    from a trimmed lock file, or are on a fresh Windows box).
    """
    results: Dict[str, str] = {}

    if not _module_importable("cv2"):
        ok, message = _pip_install(["opencv-python"])
        if not ok:
            ok, message = _pip_install(["opencv-python-headless"])
        results["opencv-python"] = (
            "installed" if ok and _module_importable("cv2") else "failed"
        )
        if not ok:
            print(f"  [accelerator-installer] opencv install skipped: {message}")
    else:
        results["opencv-python"] = "present"

    if not _module_importable("insightface"):
        ok, message = _pip_install(["insightface"])
        results["insightface"] = (
            "installed" if ok and _module_importable("insightface") else "failed"
        )
        if not ok:
            print(f"  [accelerator-installer] insightface install skipped: {message}")
    else:
        results["insightface"] = "present"

    # onnxruntime: prefer GPU wheel on CUDA, fall back to CPU wheel.
    if not _module_importable("onnxruntime"):
        prefer_gpu = _is_cuda_available()
        order = ["onnxruntime-gpu", "onnxruntime"] if prefer_gpu else ["onnxruntime"]
        last_message = ""
        for pkg in order:
            ok, message = _pip_install([pkg])
            last_message = message
            if ok and _module_importable("onnxruntime"):
                results["onnxruntime"] = "installed"
                break
        else:
            results["onnxruntime"] = "failed"
            print(
                f"  [accelerator-installer] onnxruntime install skipped: {last_message}"
            )
    else:
        results["onnxruntime"] = "present"

    return results


def auto_install_accelerators(device: Optional[str] = None) -> Dict[str, str]:
    """
    One-shot, thread-safe installer. Returns {pkg: status} where status is one of
    "present" | "installed" | "failed" | "skipped".

    Safe to call from any code path; subsequent calls return the cached result.
    Covers both the attention accelerators (SageAttention / xFormers) and
    the face-swap runtime stack (OpenCV / InsightFace / onnxruntime).
    """
    global _ATTEMPTED

    with _INSTALL_LOCK:
        if _ATTEMPTED:
            return dict(_INSTALL_STATE)

        # Face-swap stack always runs — it's required by the feature whether
        # we're on the Windows+RTX 3070 fast path or not. It'll no-op quickly
        # when everything is already present.
        print("  [accelerator-installer] first-run probe: checking face-swap stack...")
        _INSTALL_STATE.update(_install_face_swap_stack())

        # Preservation (DWPose/OpenPose) + Upscaler (spandrel / basicsr). Both
        # are small, pure-Python packages and their install never clobbers
        # the user's torch build, so they run alongside the face-swap stack
        # as part of the first-run probe regardless of platform.
        print("  [accelerator-installer] first-run probe: checking preservation stack...")
        _INSTALL_STATE.update(_install_preservation_stack())
        print("  [accelerator-installer] first-run probe: checking upscaler stack...")
        _INSTALL_STATE.update(_install_upscaler_stack())

        if not _should_attempt(device):
            _INSTALL_STATE.setdefault("sageattention", "skipped")
            _INSTALL_STATE.setdefault("xformers", "skipped")
            _ATTEMPTED = True
            summary = ", ".join(f"{pkg}={status}" for pkg, status in _INSTALL_STATE.items())
            print(f"  [accelerator-installer] done ({summary}).")
            return dict(_INSTALL_STATE)

        print("  [accelerator-installer] first-run probe: checking sageattention / xformers...")
        _INSTALL_STATE["sageattention"] = _install_sageattention()
        _INSTALL_STATE["xformers"] = _install_xformers()
        _ATTEMPTED = True

        summary = ", ".join(f"{pkg}={status}" for pkg, status in _INSTALL_STATE.items())
        print(f"  [accelerator-installer] done ({summary}).")
        return dict(_INSTALL_STATE)


def accelerator_install_status() -> Dict[str, str]:
    """Read-only snapshot of install state (no side effects)."""
    with _INSTALL_LOCK:
        return dict(_INSTALL_STATE)


def reset_auto_install_state() -> None:
    """Test/debug hook — clears the one-shot cache so the next call re-runs."""
    global _ATTEMPTED
    with _INSTALL_LOCK:
        _INSTALL_STATE.clear()
        _ATTEMPTED = False
