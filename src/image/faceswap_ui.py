"""
Gradio handlers for the post-generation Face Swap feature.

The UI shows an accordion under the Generate tab with:

* a master enable checkbox
* a "Scan Faces" button that detects every face in the currently
  displayed output image (``last_output_image``)
* ``MAX_FACE_SLOTS`` pre-allocated per-face rows, each with:
    - a thumbnail of the detected face
    - a source image uploader (what to paste onto this face)
    - a character-library dropdown (saved reusable face)
    - a "save this detected face as a character" name input + button
* an "Apply Face Swap" button that runs inswapper on all assigned slots
* a "Saved Characters" gallery with refresh / delete controls

All actual model work happens inside ``src/image/faceswap_helper.py`` —
this module only contains Gradio-facing glue (handlers that return
component updates). Every inswapper call is routed through
``get_faceswap_helper(device=...)`` so it shares the cached CUDA
``InsightFace`` provider and the cached ``inswapper_128.onnx`` session.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from src.image import character_library
from src.image.faceswap_helper import DetectedFace, get_faceswap_helper

# Pre-allocated slot count. Multi-face generated images rarely exceed a
# handful of faces — 8 is plenty and keeps the UI manageable.
MAX_FACE_SLOTS = 8

# ``gr.update`` is imported lazily so this module can be imported from
# test suites / worker processes without pulling in Gradio.
def _gr_update(**kwargs):
    import gradio as gr  # local import to avoid top-level dependency

    return gr.update(**kwargs)


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------
def _empty_slot_updates() -> List[Any]:
    """Return a flat list of updates that hide every slot + clears thumbs."""
    updates: List[Any] = []
    for _ in range(MAX_FACE_SLOTS):
        updates.append(_gr_update(visible=False))  # row
        updates.append(_gr_update(value=None))     # thumbnail image
    return updates


def scan_output_for_faces(
    last_output_image: Optional[Image.Image],
    device: str,
) -> Tuple[Any, ...]:
    """
    Detect faces in ``last_output_image`` and prepare UI updates for the
    pre-allocated face-swap slots.

    Returns a flat tuple of ``(*slot_updates, detected_state, status_text)``
    suitable for binding directly to a Gradio ``outputs=`` list. The
    slot updates alternate ``(row_visible, thumbnail_value)`` in slot
    index order for a total of ``2 * MAX_FACE_SLOTS`` entries, followed
    by the detected-face state (stashed in ``gr.State`` so the Apply
    handler knows how many faces the target has) and the status string.
    """
    if last_output_image is None:
        return (
            *_empty_slot_updates(),
            [],
            "No generated image yet. Generate an image first, then scan for faces.",
        )

    try:
        swapper = get_faceswap_helper(device=device)
        detected = swapper.detect_faces(last_output_image)
    except Exception as exc:
        return (
            *_empty_slot_updates(),
            [],
            f"Face detection failed: {exc}",
        )

    if not detected:
        return (
            *_empty_slot_updates(),
            [],
            "No faces detected in the generated image.",
        )

    if len(detected) > MAX_FACE_SLOTS:
        detected = detected[:MAX_FACE_SLOTS]

    updates: List[Any] = []
    for idx in range(MAX_FACE_SLOTS):
        if idx < len(detected):
            updates.append(_gr_update(visible=True))
            updates.append(_gr_update(value=detected[idx].thumbnail))
        else:
            updates.append(_gr_update(visible=False))
            updates.append(_gr_update(value=None))

    status = (
        f"Detected {len(detected)} face(s). Assign a source image or saved "
        f"character to each slot, then click Apply."
    )
    return (*updates, detected, status)


# ----------------------------------------------------------------------
# Applying the swap
# ----------------------------------------------------------------------
def _resolve_slot_source(
    slot_image: Optional[Image.Image],
    slot_character: Optional[str],
) -> Optional[Image.Image]:
    """
    Pick the source image for one slot.

    Priority: uploaded file > selected saved character > nothing.
    Returns ``None`` if the slot should be left untouched.
    """
    if slot_image is not None:
        return slot_image
    if slot_character and slot_character != "(none)":
        entry = character_library.get_character(slot_character)
        if entry is not None:
            return entry.reference_image
    return None


def apply_face_swap(
    enabled: bool,
    last_output_image: Optional[Image.Image],
    detected_state: List[DetectedFace],
    device: str,
    *slot_values: Any,
) -> Tuple[Optional[Image.Image], str]:
    """
    Run inswapper for every assigned slot and return the final image.

    ``slot_values`` is a flat sequence of ``(slot_image, slot_character)``
    pairs, in slot order. ``enabled=False`` short-circuits to the input
    image with a status message so the user can toggle the feature off
    without having to clear inputs.
    """
    if not enabled:
        return last_output_image, "Face swap disabled. Toggle the switch to enable."
    if last_output_image is None:
        return None, "No generated image to swap faces in."
    if not detected_state:
        return (
            last_output_image,
            "No detected faces yet. Click 'Scan Faces' first.",
        )

    # slot_values is grouped (image, character) per slot.
    if len(slot_values) != 2 * MAX_FACE_SLOTS:
        return (
            last_output_image,
            f"Internal error: expected {2 * MAX_FACE_SLOTS} slot values, "
            f"got {len(slot_values)}.",
        )

    assignments: Dict[int, Image.Image] = {}
    for slot_idx in range(min(len(detected_state), MAX_FACE_SLOTS)):
        slot_image = slot_values[slot_idx * 2]
        slot_character = slot_values[slot_idx * 2 + 1]
        resolved = _resolve_slot_source(slot_image, slot_character)
        if resolved is not None:
            # Key by DetectedFace.index — NOT slot_idx — so swap_many targets
            # the correct face even if detect_faces skipped a degenerate bbox
            # (gaps in the index sequence would otherwise cause off-by-one).
            assignments[detected_state[slot_idx].index] = resolved

    if not assignments:
        return (
            last_output_image,
            "No source images assigned. Upload a face or pick a saved character.",
        )

    try:
        swapper = get_faceswap_helper(device=device)
        result, swapped = swapper.swap_many(last_output_image, assignments)
    except Exception as exc:
        return last_output_image, f"Face swap failed: {exc}"

    missing = sorted(set(assignments) - set(swapped))
    status = f"Swapped {len(swapped)} face(s) via inswapper on {device}."
    if missing:
        status += (
            f" Skipped slots {missing} (no face detected in their source image)."
        )
    return result, status


# ----------------------------------------------------------------------
# Character library handlers
# ----------------------------------------------------------------------
def _character_gallery_value() -> List[Tuple[Image.Image, str]]:
    """List-of-tuples format accepted by ``gr.Gallery``."""
    return [
        (entry.thumbnail_image, entry.name)
        for entry in character_library.list_characters()
    ]


def refresh_character_library() -> Tuple[Any, ...]:
    """
    Return updates for the character gallery and *every* slot dropdown.

    Gradio expects one return value per output component, and the UI
    wires MAX_FACE_SLOTS dropdowns to this handler, so we emit the
    gallery update first then MAX_FACE_SLOTS identical dropdown updates.
    """
    names = character_library.list_character_names()
    choices = ["(none)", *names]
    dropdown_updates = tuple(
        _gr_update(choices=choices) for _ in range(MAX_FACE_SLOTS)
    )
    return (_gr_update(value=_character_gallery_value()), *dropdown_updates)


def _crop_reference_around_face(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    expand: float = 1.75,
) -> Image.Image:
    """
    Crop ``image`` to a region centered on ``bbox`` expanded by ``expand``
    (e.g. 1.75 = 75% larger on each side).

    The crop keeps surrounding context (hairline, chin, ears) for inswapper
    to work well, while guaranteeing the stored reference contains only
    the intended face — not any other face from a multi-face generation.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = w * expand / 2.0
    half_h = h * expand / 2.0
    nx1 = int(max(0, cx - half_w))
    ny1 = int(max(0, cy - half_h))
    nx2 = int(min(image.width, cx + half_w))
    ny2 = int(min(image.height, cy + half_h))
    if nx2 <= nx1 or ny2 <= ny1:
        return image
    return image.crop((nx1, ny1, nx2, ny2))


def save_detected_face_as_character(
    slot_idx: int,
    name: str,
    last_output_image: Optional[Image.Image],
    detected_state: List[DetectedFace],
    device: str,
) -> str:
    """
    Persist the detected face at ``slot_idx`` as a named character.

    The *reference image* stored is cropped to a generous region around
    the selected face — NOT the entire generated image. Storing the full
    multi-face image would cause ``swap_many`` to pick the largest face
    in the reference when reusing the character, which may not be the
    face the user picked. The crop includes enough padding for inswapper
    to have hairline/chin/ear context.
    """
    if last_output_image is None:
        return "No generated image to save from."
    if not detected_state:
        return "No detected faces yet. Click 'Scan Faces' first."
    if slot_idx < 0 or slot_idx >= len(detected_state):
        return f"Slot {slot_idx} is out of range (detected {len(detected_state)} faces)."
    name = (name or "").strip()
    if not name:
        return "Enter a character name before saving."

    face = detected_state[slot_idx]
    reference_crop = _crop_reference_around_face(last_output_image, face.bbox)
    try:
        character_library.save_character(
            name,
            reference_image=reference_crop,
            thumbnail=face.thumbnail,
            embedding=face.embedding,
            source="generated",
            overwrite=True,
        )
    except Exception as exc:
        return f"Saving character failed: {exc}"
    return f"Saved character '{name}' from detected face #{slot_idx + 1}."


def save_upload_as_character(
    name: str,
    source_image: Optional[Image.Image],
    device: str,
) -> str:
    """Persist an externally-uploaded face image as a named character."""
    name = (name or "").strip()
    if not name:
        return "Enter a character name before saving."
    if source_image is None:
        return "Upload a face image first."

    try:
        swapper = get_faceswap_helper(device=device)
        face = swapper.extract_source_face(source_image)
    except Exception as exc:
        return f"Face detection failed while saving: {exc}"

    if face is None:
        embedding = None
        thumbnail = None
    else:
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            import numpy as np

            raw = np.asarray(face.embedding, dtype=np.float32)
            norm = float(np.linalg.norm(raw))
            embedding = raw / norm if norm > 0 else raw
        # Thumbnail: crop around detected bbox for a clean preview.
        try:
            x1, y1, x2, y2 = (int(max(0, v)) for v in face.bbox)
            thumbnail = source_image.convert("RGB").crop((x1, y1, x2, y2))
            thumbnail.thumbnail((160, 160), Image.Resampling.LANCZOS)
        except Exception:
            thumbnail = None

    try:
        character_library.save_character(
            name,
            reference_image=source_image,
            thumbnail=thumbnail,
            embedding=embedding,
            source="upload",
            overwrite=True,
        )
    except Exception as exc:
        return f"Saving character failed: {exc}"
    return f"Saved character '{name}'."


def delete_character_handler(name: str) -> str:
    """Delete a character by name. No-op if it doesn't exist."""
    if not name:
        return "Select a character to delete."
    ok = character_library.delete_character(name)
    return (
        f"Deleted character '{name}'." if ok else f"No character named '{name}'."
    )
