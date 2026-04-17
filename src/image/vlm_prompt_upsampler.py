import gc
import threading
from typing import Optional, Tuple, Dict, Any

import torch

from src.security import sanitize_prompt, MAX_PROMPT_LENGTH

_MODEL_ID = "microsoft/Florence-2-base"
# Florence-2 weights are ~460 MB; concurrent VLM-upsample requests without
# a lock can race into the ``from_pretrained`` branch and double-allocate
# the model + processor. Lock guards the get-or-create path only.
_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


def _get_device_key(device: str) -> str:
    return device if device else "cpu"


def _resolve_device(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(device: str):
    from transformers import AutoModelForCausalLM, AutoProcessor

    device_key = _get_device_key(device)
    with _CACHE_LOCK:
        cached = _CACHE.get(device_key)
        if cached:
            return cached["model"], cached["processor"]

        resolved_device = _resolve_device(device)
        dtype = torch.float16 if resolved_device == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            _MODEL_ID,
            trust_remote_code=True,
        )

        if resolved_device == "cuda":
            model = model.to("cuda")
        model.eval()

        _CACHE[device_key] = {"model": model, "processor": processor, "device": resolved_device}
        return model, processor


def _run_task(
    image,
    task: str,
    model,
    processor,
    device: str,
) -> str:
    inputs = processor(text=task, images=image, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    decoded = processor.batch_decode(generated, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        decoded,
        task=task,
        image_size=image.size,
    )

    try:
        result = parsed.get(task, "")
    except AttributeError:
        result = ""

    return str(result).strip()


def _merge_prompt(prompt: str, caption: str, ocr: str) -> str:
    prompt = prompt or ""
    parts = []
    if prompt.strip():
        parts.append(prompt.strip())
    if caption:
        parts.append(caption)
    if ocr:
        parts.append(f"Text in image: {ocr}")
    merged = ", ".join(parts)
    return sanitize_prompt(merged)[:MAX_PROMPT_LENGTH]


def upsample_prompt_from_image(
    prompt: str,
    image,
    device: str = "cuda",
) -> Tuple[str, Optional[str]]:
    """
    Upsample prompt using Florence-2 (detailed caption + OCR).
    Returns (merged_prompt, error_message).
    """
    if image is None:
        return prompt, None

    last_err: Optional[str] = None
    for attempt_device in (device, "cpu"):
        try:
            model, processor = _load_model(attempt_device)
            resolved_device = _resolve_device(attempt_device)
            caption = _run_task(image, "<MORE_DETAILED_CAPTION>", model, processor, resolved_device)
            ocr = _run_task(image, "<OCR>", model, processor, resolved_device)
            merged = _merge_prompt(prompt, caption, ocr)
            return merged, None
        except RuntimeError as exc:
            last_err = str(exc)
            if "out of memory" in last_err.lower():
                print("VLM upsampling OOM; retrying on CPU.")
            else:
                print(f"VLM upsampling failed: {last_err}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as exc:
            last_err = str(exc)
            print(f"VLM upsampling failed: {last_err}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        if attempt_device == "cpu":
            break

    return prompt, last_err
