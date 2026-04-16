"""
First-run auto-installer for optional inference accelerators.

Targets the Windows + RTX 3070 fast path: on the first generation, this
module probes for `sageattention` (SDPA drop-in, INT8 QK + FP16 PV) and
`xformers` (memory-efficient attention fallback), and pip-installs the
appropriate wheel pinned to the user's existing PyTorch build if either
is missing.

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
        # Importable but broken install (e.g. dll mismatch). Treat as present
        # so we don't blindly overwrite a user's pinned build.
        return True


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
      1. `--no-deps`: install the xformers wheel without letting pip touch
         torch. If the ABI happens to line up with the user's torch, this
         works. If not, the import will fail and we report "failed" — the
         pipeline already falls back to SDPA in that case.
      2. As a conservative second attempt we try the normal resolution,
         which is fine for users with a plain PyPI torch build but may
         shuffle torch on CUDA-pinned environments. We only take this
         second path if the user's torch is a vanilla release (no "+cuXXX"
         local version suffix) so we don't clobber CUDA-enabled builds.
    """
    if _module_importable("xformers"):
        return "present"

    torch_tag: Optional[str] = None
    try:
        import torch  # noqa: WPS433
        torch_tag = str(torch.__version__)
    except Exception:
        torch_tag = None

    candidates: List[List[str]] = [["xformers", "--no-deps"]]
    # Only allow the unconstrained install on plain PyPI torch builds.
    # Builds with a local version suffix (e.g. "2.10.0+cu130") come from
    # the CUDA-specific index and would be replaced by pip's normal
    # resolver.
    if torch_tag is not None and "+" not in torch_tag:
        candidates.append(["xformers"])

    for args in candidates:
        ok, message = _pip_install(args)
        if ok and _module_importable("xformers"):
            return "installed"
        print(f"  [accelerator-installer] xformers ({' '.join(args)}) skipped: {message}")
    return "failed"


def auto_install_accelerators(device: Optional[str] = None) -> Dict[str, str]:
    """
    One-shot, thread-safe installer. Returns {pkg: status} where status is one of
    "present" | "installed" | "failed" | "skipped".

    Safe to call from any code path; subsequent calls return the cached result.
    """
    global _ATTEMPTED

    with _INSTALL_LOCK:
        if _ATTEMPTED:
            return dict(_INSTALL_STATE)

        if not _should_attempt(device):
            _INSTALL_STATE.update({"sageattention": "skipped", "xformers": "skipped"})
            _ATTEMPTED = True
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
