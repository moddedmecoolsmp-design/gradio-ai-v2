"""
Persistent named-face library used by the post-generation Face Swap UI.

Each character is one directory under ``outputs/character_faces/<name>/``
containing:

* ``reference.png`` — the original source image the user provided or
  harvested from a generated image. This is what actually gets fed to
  inswapper; the embedding is only used for similarity / recall.
* ``thumbnail.png`` — a cropped face preview for the Gradio gallery.
* ``embedding.npy`` — 512-d normalized InsightFace embedding.
* ``meta.json`` — ``{"name": str, "created_at": float, "source": str}``.

The library is filesystem-only (no DB) so it survives process
restarts and the user can eyeball it / delete entries by hand.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

_LIBRARY_LOCK = threading.Lock()
_DEFAULT_DIR = os.path.join("outputs", "character_faces")


@dataclass
class CharacterEntry:
    """One saved named character in the library."""

    name: str
    directory: str
    reference_path: str
    thumbnail_path: str
    embedding_path: str
    created_at: float
    source: str

    @property
    def reference_image(self) -> Image.Image:
        """Lazy-load the reference image (the one fed to inswapper)."""
        return Image.open(self.reference_path).convert("RGB")

    @property
    def thumbnail_image(self) -> Image.Image:
        """Lazy-load the preview thumbnail."""
        return Image.open(self.thumbnail_path).convert("RGB")

    def load_embedding(self) -> np.ndarray:
        return np.load(self.embedding_path)


def _safe_name(name: str) -> str:
    """Return a filesystem-safe slug for ``name``.

    Strips anything that isn't alphanumeric / dash / underscore / space,
    collapses whitespace, and falls back to ``"character"`` if the input
    is empty after sanitization.
    """
    name = (name or "").strip()
    name = re.sub(r"[^\w\-\s]", "", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("_-")
    return name or "character"


def get_library_dir(root: Optional[str] = None) -> str:
    """Return (and create) the root directory for the character library."""
    path = root or os.environ.get("UFIG_CHARACTER_LIBRARY", _DEFAULT_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def list_characters(root: Optional[str] = None) -> List[CharacterEntry]:
    """Return every saved character, newest first."""
    base = get_library_dir(root)
    entries: List[CharacterEntry] = []
    if not os.path.isdir(base):
        return entries

    for child in sorted(os.listdir(base)):
        child_dir = os.path.join(base, child)
        meta_path = os.path.join(child_dir, "meta.json")
        ref_path = os.path.join(child_dir, "reference.png")
        thumb_path = os.path.join(child_dir, "thumbnail.png")
        emb_path = os.path.join(child_dir, "embedding.npy")
        if not (
            os.path.isdir(child_dir)
            and os.path.exists(meta_path)
            and os.path.exists(ref_path)
            and os.path.exists(emb_path)
        ):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        entries.append(
            CharacterEntry(
                name=str(meta.get("name", child)),
                directory=child_dir,
                reference_path=ref_path,
                thumbnail_path=thumb_path if os.path.exists(thumb_path) else ref_path,
                embedding_path=emb_path,
                created_at=float(meta.get("created_at", 0.0)),
                source=str(meta.get("source", "manual")),
            )
        )
    entries.sort(key=lambda e: e.created_at, reverse=True)
    return entries


def list_character_names(root: Optional[str] = None) -> List[str]:
    """Short helper for Gradio Dropdown ``choices``."""
    return [entry.name for entry in list_characters(root)]


def get_character(
    name: str, root: Optional[str] = None
) -> Optional[CharacterEntry]:
    """Look up a character by its stored (display) name, case-insensitive."""
    if not name:
        return None
    needle = name.strip().lower()
    for entry in list_characters(root):
        if entry.name.lower() == needle:
            return entry
    return None


def save_character(
    name: str,
    reference_image: Image.Image,
    *,
    thumbnail: Optional[Image.Image] = None,
    embedding: Optional[np.ndarray] = None,
    source: str = "manual",
    root: Optional[str] = None,
    overwrite: bool = True,
) -> CharacterEntry:
    """
    Persist a named character.

    The ``reference_image`` is what inswapper reads when the character
    is later selected in the UI, so pass the original source photo /
    generated image and not a tiny thumbnail. ``thumbnail`` is used
    purely for display; if omitted the reference is downscaled.
    ``embedding`` must already be L2-normalized (InsightFace's
    ``normed_embedding``) — if omitted, a zero vector is stored so
    similarity search will just never match this entry.

    ``overwrite=False`` raises ``FileExistsError`` if a directory with
    the sanitized name already exists.
    """
    safe = _safe_name(name)
    base = get_library_dir(root)
    target_dir = os.path.join(base, safe)

    with _LIBRARY_LOCK:
        if os.path.exists(target_dir) and not overwrite:
            raise FileExistsError(
                f"Character '{name}' already exists at {target_dir}"
            )
        os.makedirs(target_dir, exist_ok=True)

        reference_image = reference_image.convert("RGB")
        ref_path = os.path.join(target_dir, "reference.png")
        reference_image.save(ref_path, format="PNG")

        if thumbnail is None:
            thumbnail = reference_image.copy()
            thumbnail.thumbnail((160, 160), Image.Resampling.LANCZOS)
        else:
            thumbnail = thumbnail.convert("RGB")
        thumb_path = os.path.join(target_dir, "thumbnail.png")
        thumbnail.save(thumb_path, format="PNG")

        if embedding is None:
            embedding = np.zeros(512, dtype=np.float32)
        emb_path = os.path.join(target_dir, "embedding.npy")
        np.save(emb_path, np.asarray(embedding, dtype=np.float32))

        meta = {
            "name": name.strip() or safe,
            "created_at": time.time(),
            "source": source,
        }
        with open(
            os.path.join(target_dir, "meta.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

    return CharacterEntry(
        name=meta["name"],
        directory=target_dir,
        reference_path=ref_path,
        thumbnail_path=thumb_path,
        embedding_path=emb_path,
        created_at=meta["created_at"],
        source=meta["source"],
    )


def delete_character(name: str, root: Optional[str] = None) -> bool:
    """Remove a character directory. Returns ``True`` if one was deleted."""
    entry = get_character(name, root)
    if entry is None:
        return False
    import shutil

    with _LIBRARY_LOCK:
        shutil.rmtree(entry.directory, ignore_errors=True)
    return True
