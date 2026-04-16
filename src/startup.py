"""Startup preflight checks: dependency verification, CUDA 13 enforcement,
and optional dependency probing.

Extracted from app.py to keep the main entry point lean.
"""

import importlib
import importlib.util
import json
import os
from typing import List, Tuple

import torch

from src.config import BASE_DIR, DEPENDENCY_CHECK_FLAG
from src.runtime_policies import (
    build_dependency_profile_metadata,
    compute_file_sha256,
    describe_acceleration_stack,
    is_cuda13_runtime,
    is_dependency_metadata_current,
    select_requirements_file,
)


# ---------------------------------------------------------------------------
# Module availability helpers
# ---------------------------------------------------------------------------

def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _probe_optional_dependency_statuses() -> List[Tuple[str, str, str]]:
    """Probe optional packages and return (level, name, detail) tuples."""
    statuses: List[Tuple[str, str, str]] = []

    for import_name, display_name in [
        ("insightface", "InsightFace"),
        ("facexlib", "FaceXLib"),
        ("timm", "PyTorch Image Models"),
        ("einops", "Einops"),
        ("ftfy", "FTFY"),
        ("filterpy", "FilterPy"),
        ("pydub", "PyDub"),
        ("librosa", "Librosa"),
        ("pruna", "Pruna"),
    ]:
        if _module_available(import_name):
            statuses.append(("OK", display_name, "available"))
        else:
            statuses.append(("WARN", display_name, "package not installed"))

    if _module_available("qwen_tts"):
        statuses.append(("OK", "Qwen3 TTS", "available"))
    else:
        statuses.append(("WARN", "Qwen3 TTS", "package not installed"))

    # MediaPipe / ControlNet Aux
    mediapipe_available = _module_available("mediapipe")
    mediapipe_has_solutions = False
    mediapipe_error = None
    if mediapipe_available:
        try:
            import mediapipe  # type: ignore
            mediapipe_has_solutions = hasattr(mediapipe, "solutions")
        except Exception as exc:
            mediapipe_error = str(exc)

    if _module_available("controlnet_aux"):
        if mediapipe_error:
            statuses.append(("WARN", "ControlNet Aux", f"degraded: {mediapipe_error}"))
        elif not mediapipe_available:
            statuses.append(("WARN", "ControlNet Aux", "degraded: MediaPipe not installed"))
        elif not mediapipe_has_solutions:
            statuses.append(("WARN", "ControlNet Aux", "degraded: MediaPipe legacy solutions API unavailable"))
        else:
            try:
                importlib.import_module("controlnet_aux")
                statuses.append(("OK", "ControlNet Aux", "available"))
            except Exception as exc:
                statuses.append(("WARN", "ControlNet Aux", f"unavailable: {exc}"))
    else:
        statuses.append(("WARN", "ControlNet Aux", "package not installed"))

    if mediapipe_available and mediapipe_has_solutions:
        statuses.append(("OK", "MediaPipe", "available"))
    elif mediapipe_available:
        statuses.append(("WARN", "MediaPipe", "installed without legacy solutions API"))
    elif mediapipe_error:
        statuses.append(("WARN", "MediaPipe", mediapipe_error))
    else:
        statuses.append(("WARN", "MediaPipe", "package not installed"))

    # Torchaudio-dependent packages
    torchaudio_module = None
    torchaudio_error = None
    if _module_available("torchaudio"):
        try:
            import torchaudio  # type: ignore
            torchaudio_module = torchaudio
        except Exception as exc:
            torchaudio_error = str(exc)

    if torchaudio_error:
        statuses.append(("WARN", "PyAnnote Audio", f"unavailable: torchaudio import failed ({torchaudio_error})"))
        statuses.append(("WARN", "SpeechBrain", f"unavailable: torchaudio import failed ({torchaudio_error})"))
    else:
        missing_pyannote = []
        missing_speechbrain = []
        if torchaudio_module is None:
            missing_pyannote.append("torchaudio")
            missing_speechbrain.append("torchaudio")
        else:
            if not hasattr(torchaudio_module, "AudioMetaData"):
                missing_pyannote.append("AudioMetaData")
            if not hasattr(torchaudio_module, "list_audio_backends"):
                missing_speechbrain.append("list_audio_backends")

        if _module_available("pyannote.audio") and not missing_pyannote:
            statuses.append(("OK", "PyAnnote Audio", "available"))
        elif _module_available("pyannote.audio"):
            statuses.append(("WARN", "PyAnnote Audio", f"unavailable: torchaudio missing {', '.join(missing_pyannote)}"))
        else:
            statuses.append(("WARN", "PyAnnote Audio", "package not installed"))

        if _module_available("speechbrain") and not missing_speechbrain:
            statuses.append(("OK", "SpeechBrain", "available"))
        elif _module_available("speechbrain"):
            statuses.append(("WARN", "SpeechBrain", f"unavailable: torchaudio missing {', '.join(missing_speechbrain)}"))
        else:
            statuses.append(("WARN", "SpeechBrain", "package not installed"))

    return statuses


# ---------------------------------------------------------------------------
# Core startup checks
# ---------------------------------------------------------------------------

def check_required_dependencies() -> bool:
    """Verify critical imports without mutating the environment."""
    print("Checking required dependencies...")

    required_checks = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("diffusers", "Diffusers"),
        ("accelerate", "Accelerate"),
        ("safetensors", "SafeTensors"),
        ("sentencepiece", "SentencePiece"),
        ("google.protobuf", "Protobuf"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("gradio", "Gradio"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-learn"),
        ("peft", "PEFT"),
        ("optimum.quanto", "Optimum Quanto"),
        ("requests", "Requests"),
    ]

    failed_imports = []
    for import_name, display_name in required_checks:
        try:
            if "." in import_name:
                importlib.import_module(import_name)
            else:
                __import__(import_name)
            print(f"[OK] {display_name} available")
        except Exception as exc:
            print(f"[FAIL] {display_name}: {exc}")
            failed_imports.append((import_name, str(exc)))

    for level, display_name, detail in _probe_optional_dependency_statuses():
        print(f"[{level}] {display_name} {detail}")

    if not failed_imports:
        print("All required dependencies are available.")
        return True

    print(f"\nDependency verification failed for {len(failed_imports)} module(s).")
    print("Run Install.bat --repair, then relaunch.")
    return False


def _load_dependency_metadata(flag_path: str) -> dict:
    if not os.path.exists(flag_path):
        return {}
    try:
        with open(flag_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        try:
            with open(flag_path, "r", encoding="utf-8") as handle:
                raw_value = handle.read().strip()
            if raw_value:
                return {"_legacy_metadata_text": raw_value}
        except Exception:
            pass
        return {}


def run_dependency_preflight() -> bool:
    """Verify dependency profile and imports without mutating the environment."""
    requirements_file = select_requirements_file(
        base_dir=BASE_DIR,
        is_windows=(os.name == "nt"),
        cuda_available=torch.cuda.is_available(),
    )
    requirements_path = os.path.join(BASE_DIR, requirements_file)
    requirements_hash = compute_file_sha256(requirements_path)
    target_metadata = build_dependency_profile_metadata(requirements_file, requirements_hash)
    current_metadata = _load_dependency_metadata(DEPENDENCY_CHECK_FLAG)
    is_current = is_dependency_metadata_current(current_metadata, target_metadata)

    if not os.path.exists(requirements_path):
        print(f"Dependency preflight failed: missing requirements file {requirements_path}")
        return False

    if not is_current:
        if current_metadata.get("_legacy_metadata_text"):
            print("Dependency metadata is in legacy format and must be regenerated.")
        print(f"Dependency profile is out of date for {requirements_file}.")
        print("Run Install.bat --repair, then relaunch.")
        return False

    print("Dependency profile is current.")
    if not check_required_dependencies():
        print("Dependency preflight failed during import verification.")
        print("Run Install.bat --repair, then relaunch.")
        return False

    return True


def enforce_cuda13_runtime_profile() -> bool:
    """Enforce CUDA 13 runtime profile on Windows when CUDA is available."""
    if os.name != "nt":
        return True
    if os.environ.get("UFIG_ENFORCE_CUDA13", "1") != "1":
        return True
    if not torch.cuda.is_available():
        print("CUDA 13 profile check failed: CUDA device not available.")
        print("This workflow is configured for CUDA 13 + NVIDIA GPU.")
        print("Re-run Install.bat --repair and verify NVIDIA drivers/runtime.")
        return False

    cuda_runtime = getattr(torch.version, "cuda", None)
    if is_cuda13_runtime(cuda_runtime):
        print(f"CUDA runtime profile validated: {cuda_runtime}")
        return True

    print(f"CUDA 13 profile check failed: detected torch CUDA runtime '{cuda_runtime}'.")
    print("This project requires CUDA 13 wheels. Re-run Install.bat --repair.")
    return False


def ensure_accelerate_available() -> str:
    """Ensure accelerate is importable; raise RuntimeError otherwise."""
    try:
        import accelerate
        return accelerate.__version__
    except Exception as exc:
        raise RuntimeError(
            "Accelerate is required to load SDNQ quantized models with low_cpu_mem_usage=True. "
            "Re-run Launch.bat to reinstall dependencies or install with `pip install accelerate`."
        ) from exc


def log_provider_telemetry(
    profile: str,
    cuda_speed_status: dict,
    compile_probe: bool,
    optional_accelerators: bool,
) -> None:
    """Log startup provider telemetry and fallback context."""
    runtime_stack = {
        "profile": profile,
        "cuda_runtime": getattr(torch.version, "cuda", None) or "none",
        "tf32": cuda_speed_status.get("tf32", False),
        "matmul_precision": cuda_speed_status.get("matmul_precision") or "default",
        "sdp": cuda_speed_status.get("sdp", False),
        "allocator_conf": cuda_speed_status.get("allocator_conf"),
        "compile_probe": compile_probe,
        "optional_accelerators": optional_accelerators,
    }
    try:
        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[Runtime] GPU: {torch.cuda.get_device_name(0)} ({total_gb:.2f} GB)")
        else:
            print("[Runtime] GPU: CUDA unavailable")
    except Exception as e:
        print(f"[Runtime] GPU telemetry unavailable: {e}")

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"[Runtime] ONNX providers: {providers}")
    except Exception as e:
        print(f"[Runtime] ONNX telemetry unavailable: {e}")

    print(f"[Runtime] Acceleration stack: {describe_acceleration_stack(runtime_stack)}")


def ensure_cache_dirs() -> None:
    """Create cache directories from environment variables."""
    for path in [
        os.environ.get("UFIG_CACHE_DIR", os.path.join(BASE_DIR, "cache")),
        os.environ.get("HF_HOME", ""),
        os.environ.get("HF_HUB_CACHE", ""),
        os.environ.get("HF_XET_CACHE", ""),
        os.environ.get("HF_ASSETS_CACHE", ""),
        os.environ.get("TORCH_HOME", ""),
        os.environ.get("GRADIO_TEMP_DIR", ""),
    ]:
        if path:
            try:
                os.makedirs(path, exist_ok=True)
            except Exception:
                pass
