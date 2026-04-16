
import os
import threading

import torch
from insightface.app import FaceAnalysis

_face_analysis_cache: dict = {}
_face_analysis_lock = threading.Lock()


def _get_available_onnx_providers():
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except Exception:
        return []


def resolve_onnx_provider_candidates(device: str = "cuda"):
    if device != "cuda":
        return [["CPUExecutionProvider"]]

    available = _get_available_onnx_providers()
    candidates = []

    if "CUDAExecutionProvider" in available:
        candidates.append(["CUDAExecutionProvider", "CPUExecutionProvider"])

    candidates.append(["CPUExecutionProvider"])
    return candidates


def resolve_onnx_providers(device: str = "cuda"):
    """Backward-compatible helper returning the first provider candidate."""
    return resolve_onnx_provider_candidates(device)[0]


def get_face_analysis(device="cuda", name="antelopev2"):
    """
    Centralized provider for FaceAnalysis to save VRAM.

    Thread-safe: concurrent callers for the same (device, name) key share a
    single initialization rather than racing the dict.
    """
    key = f"{device}_{name}"
    with _face_analysis_lock:
        cached = _face_analysis_cache.get(key)
        if cached is not None:
            print(f"[FaceAnalysis] Reusing existing instance: {name} on {device}")
            return cached

        print(f"[FaceAnalysis] Initializing new instance: {name} on {device}")

        candidates = resolve_onnx_provider_candidates(device)
        errors = []
        for providers in candidates:
            try:
                app = FaceAnalysis(name=name, providers=providers)
                ctx_id = 0 if device == "cuda" else -1
                app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                _face_analysis_cache[key] = app
                print(f"[FaceAnalysis] Provider selected: {providers}")
                return app
            except Exception as exc:
                errors.append(f"{providers}: {exc}")
                print(f"[FaceAnalysis] Provider probe failed: {providers} ({exc})")

        joined = " | ".join(errors[-3:]) if errors else "No providers available"
        raise RuntimeError(f"Failed to initialize FaceAnalysis providers: {joined}")


def unload_face_analysis():
    with _face_analysis_lock:
        _face_analysis_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
