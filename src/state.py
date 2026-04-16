"""User state persistence: load/save UI state, image persistence, and directory helpers.

Extracted from app.py to keep the main entry point lean.
"""

import os
from typing import List, Optional

from PIL import Image

from src.config import BASE_DIR, STATE_DIR
from src.security import load_json_safe, save_json_safe

STATE_PATH = os.path.join(STATE_DIR, "ui_state.json")
STATE_IMAGES_DIR = os.path.join(STATE_DIR, "input_images")
STATE_CHAR_REFS_DIR = os.path.join(STATE_DIR, "character_references")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LORAS_DIR = os.path.join(BASE_DIR, "loras")


def ensure_state_dirs() -> None:
    """Create all state persistence directories."""
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(STATE_IMAGES_DIR, exist_ok=True)
    os.makedirs(STATE_CHAR_REFS_DIR, exist_ok=True)
    os.makedirs(LORAS_DIR, exist_ok=True)


def load_user_state() -> dict:
    """Load the persisted UI state from disk."""
    return load_json_safe(STATE_PATH)


def save_user_state(state: dict) -> None:
    """Save the current UI state to disk."""
    save_json_safe(STATE_PATH, state)


def save_images_to_dir(
    images, prefix: str, directory: str, max_count: int = 10
) -> List[str]:
    """Save a list of PIL images to *directory* and return the filenames."""
    if not images:
        return []
    os.makedirs(directory, exist_ok=True)
    saved_names: List[str] = []
    for idx, img in enumerate(images[:max_count]):
        if img is None:
            continue
        if isinstance(img, tuple):
            img = img[0]
        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(img)
            except Exception:
                continue
        path = os.path.join(directory, f"{prefix}_{idx + 1}.png")
        try:
            img.save(path)
            saved_names.append(os.path.basename(path))
        except Exception:
            continue
    return saved_names


def load_images_from_dir(image_names, directory: str) -> List[Image.Image]:
    """Load PIL images from *directory* given a list of filenames."""
    images: List[Image.Image] = []
    if not image_names:
        return images
    for name in image_names:
        path = os.path.join(directory, str(name))
        if not os.path.isfile(path):
            continue
        try:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        except Exception:
            continue
    return images


def clear_dir(directory: str) -> None:
    """Remove all files from a directory (non-recursive)."""
    try:
        if os.path.isdir(directory):
            for name in os.listdir(directory):
                path = os.path.join(directory, name)
                if os.path.isfile(path):
                    os.remove(path)
    except Exception:
        pass


def save_input_images(images) -> List[str]:
    """Clear old input images and save new ones."""
    clear_dir(STATE_IMAGES_DIR)
    return save_images_to_dir(images, "input", STATE_IMAGES_DIR, max_count=6)


def load_input_images(image_names) -> List[Image.Image]:
    """Load persisted input images."""
    return load_images_from_dir(image_names, STATE_IMAGES_DIR)
