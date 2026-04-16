"""
Security utilities for manga-to-realistic image generation pipeline.

Provides:
- Path traversal prevention (canonicalize + jail-root checks)
- File extension allowlisting for images and model files
- Bounded numeric coercion
- Secure JSON I/O (size-limited reads)
- Sanitized user-facing string trimming
"""

import os
import json
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Allowed file-extension sets
# ---------------------------------------------------------------------------
ALLOWED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
ALLOWED_MODEL_EXTENSIONS = frozenset({".safetensors", ".onnx", ".bin", ".pt", ".pth"})

# Maximum size for JSON state files (4 MB).  Prevents decompression-bomb style
# attacks where an adversary plants a huge file in the state directory.
MAX_STATE_JSON_BYTES = 4 * 1024 * 1024  # 4 MB

# Maximum length for prompt strings accepted from the UI.
MAX_PROMPT_LENGTH = 4096

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_safe_path(user_path: str, allowed_root: Optional[str] = None) -> Optional[str]:
    """
    Canonicalize *user_path* and, when *allowed_root* is given, verify that
    the result is contained within that root directory tree.

    Returns the absolute, normalised path on success, or ``None`` when the
    path would escape the allowed root.  An empty or ``None`` input also
    returns ``None``.
    """
    if not user_path or not isinstance(user_path, str):
        return None
    try:
        resolved = os.path.realpath(os.path.abspath(user_path))
    except Exception:
        return None

    if allowed_root is not None:
        try:
            root = os.path.realpath(os.path.abspath(allowed_root))
        except Exception:
            return None
        # commonpath raises ValueError if the paths are on different drives
        # (Windows) – treat that as "not contained".
        try:
            common = os.path.commonpath([resolved, root])
        except ValueError:
            return None
        if common != root:
            return None

    return resolved


def is_safe_filename(name: str) -> bool:
    """
    Return True when *name* is a plain filename (no directory separators,
    no leading dots, no null bytes, no shell-special characters).
    """
    if not name or not isinstance(name, str):
        return False
    if "\x00" in name:
        return False
    if os.sep in name or (os.altsep and os.altsep in name):
        return False
    if name.startswith("."):
        return False
    # Reject characters that have special meaning in shells or file systems.
    if re.search(r'[<>:"|?*\x00-\x1f]', name):
        return False
    return True


def validate_image_path(path: str, allowed_root: Optional[str] = None) -> Optional[str]:
    """
    Validate that *path* refers to a file with an allowed image extension
    that resides within *allowed_root* (when provided).

    Returns the safe absolute path or ``None``.
    """
    safe = resolve_safe_path(path, allowed_root)
    if safe is None:
        return None
    ext = os.path.splitext(safe)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return None
    return safe


def validate_model_path(path: str, allowed_root: Optional[str] = None) -> Optional[str]:
    """
    Validate that *path* refers to a file with an allowed model extension
    that resides within *allowed_root* (when provided).

    Returns the safe absolute path or ``None``.
    """
    safe = resolve_safe_path(path, allowed_root)
    if safe is None:
        return None
    ext = os.path.splitext(safe)[1].lower()
    if ext not in ALLOWED_MODEL_EXTENSIONS:
        return None
    return safe


def validate_directory(path: str, allowed_root: Optional[str] = None) -> Optional[str]:
    """
    Validate that *path* is an existing directory that resides within
    *allowed_root* (when provided).

    Returns the safe absolute path or ``None``.
    """
    safe = resolve_safe_path(path, allowed_root)
    if safe is None:
        return None
    if not os.path.isdir(safe):
        return None
    return safe


# ---------------------------------------------------------------------------
# Secure JSON I/O
# ---------------------------------------------------------------------------

def load_json_safe(file_path: str, max_bytes: int = MAX_STATE_JSON_BYTES) -> object:
    """
    Read a JSON file with a hard size cap to prevent resource exhaustion.

    Returns the parsed object on success, or an empty dict on any error
    (missing file, size exceeded, parse error).
    """
    if not os.path.isfile(file_path):
        return {}
    try:
        size = os.path.getsize(file_path)
    except OSError:
        return {}
    if size > max_bytes:
        print(f"[Security] Rejected oversized state file ({size} bytes): {file_path}")
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_json_safe(file_path: str, data: dict) -> bool:
    """
    Write *data* as JSON to *file_path*, using an atomic write pattern
    (write to a sibling temp file then rename) to avoid partial writes.

    Returns True on success, False on any error.
    """
    if not isinstance(data, dict):
        return False
    tmp_path = file_path + ".tmp"
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp_path, file_path)
        return True
    except Exception as exc:
        print(f"[Security] Failed to save state: {exc}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False


# ---------------------------------------------------------------------------
# Input sanitization helpers
# ---------------------------------------------------------------------------

def sanitize_prompt(text: str) -> str:
    """
    Trim and length-cap a prompt string received from user input.
    Null bytes are removed; the string is truncated to MAX_PROMPT_LENGTH.
    """
    if not isinstance(text, str):
        return ""
    # Remove null bytes
    text = text.replace("\x00", "")
    return text[:MAX_PROMPT_LENGTH]


def sanitize_folder_path(path: str) -> str:
    """
    Strip leading/trailing whitespace and remove null bytes from a folder
    path entered by the user.  Does NOT validate containment — call
    ``validate_directory`` for that.
    """
    if not isinstance(path, str):
        return ""
    return path.strip().replace("\x00", "")


def clamp_int(value, lo: int, hi: int, fallback: int) -> int:
    """Coerce *value* to int and clamp to [lo, hi].  Return *fallback* on error."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(lo, min(hi, n))


def clamp_float(value, lo: float, hi: float, fallback: float) -> float:
    """Coerce *value* to float and clamp to [lo, hi].  Return *fallback* on error."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(lo, min(hi, n))
