"""
Text preservation for manga / comic / caption-heavy source images.

When the diffusion model converts a manga panel to a realistic photo, any
text on the input (speech bubbles, captions, signage) is invariably
garbled or lost entirely — diffusion models are notoriously bad at
rendering legible text.  This module extracts the text from the source,
remembers its bounding boxes, and repaints the original text on the
generated output so readers can actually read it.

Pipeline
~~~~~~~~
    source image --[EasyOCR]--> [(text, bbox, color)]
    generated image + regions --[PIL paint]--> restored image

Design notes
~~~~~~~~~~~~
* EasyOCR is used because it runs on CUDA out of the box (no separate
  build step) and supports CUDA 13 via onnxruntime-gpu / torch CUDA 13
  wheels.  The loader is lazy, thread-safe, cached per device.
* Auto-installs ``easyocr`` via the first-run installer so users don't
  hit ImportError on first click.  Silent degrade if install fails.
* Source and output images may have different dimensions (upscale,
  downscale, aspect change).  Bounding boxes are stored in normalised
  ``[0, 1]`` coordinates and re-projected onto the output's actual size.
* Paints a white filled rectangle under each text region, then writes
  the text on top with a default font.  No attempt at fancy typography
  (outline, color-match to the bubble) — preservation, not restoration.
* Fails safe: if OCR can't load or extracts nothing, we return the
  generated image untouched.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """One OCR hit, normalised against the source image dimensions.

    ``bbox`` is ``(x0, y0, x1, y1)`` in ``[0, 1]`` coordinates (relative
    to the source image).  ``confidence`` is EasyOCR's score in
    ``[0, 1]``; callers can use it to drop low-confidence noise.
    """

    text: str
    bbox: Tuple[float, float, float, float]
    confidence: float


# Cache EasyOCR readers per (device, tuple(langs)) so repeated calls
# across a Gradio session don't re-download the ~64 MB detection +
# recognition models every time.  Guarded by a lock because Gradio
# handlers run concurrently and EasyOCR's constructor is neither
# reentrant nor cheap.
_READER_CACHE: dict = {}
_READER_LOCK = threading.Lock()


def _get_reader(
    device: str = "cuda",
    languages: Tuple[str, ...] = ("en",),
):
    """Return a cached ``easyocr.Reader`` or ``None`` if EasyOCR is
    unavailable.

    Never raises — the caller gates text preservation on a non-``None``
    return value and skips the feature otherwise.
    """

    try:
        import easyocr  # noqa: WPS433 — lazy import; EasyOCR pulls torch etc.
    except Exception as exc:
        logger.info("easyocr import failed; text preservation disabled: %s", exc)
        return None

    key = (str(device), tuple(languages))
    with _READER_LOCK:
        cached = _READER_CACHE.get(key)
        if cached is not None:
            return cached

        gpu = str(device).lower() == "cuda"
        try:
            reader = easyocr.Reader(list(languages), gpu=gpu, verbose=False)
        except Exception as exc:
            logger.warning("easyocr Reader init failed (device=%s): %s", device, exc)
            return None

        _READER_CACHE[key] = reader
        return reader


def extract_text_regions(
    source: Image.Image,
    device: str = "cuda",
    languages: Tuple[str, ...] = ("en",),
    min_confidence: float = 0.3,
) -> List[TextRegion]:
    """Run OCR on ``source`` and return detected text regions.

    Returns an empty list on any failure — callers should treat that as
    "no text to preserve" and skip the repaint step.
    """

    reader = _get_reader(device=device, languages=languages)
    if reader is None or source is None:
        return []

    # EasyOCR expects a numpy array in BGR order (OpenCV convention).
    try:
        import numpy as np
    except Exception:
        return []

    try:
        rgb = source.convert("RGB")
        arr = np.array(rgb)[:, :, ::-1]  # RGB -> BGR
        raw = reader.readtext(arr, detail=1, paragraph=False)
    except Exception as exc:
        logger.warning("easyocr readtext failed: %s", exc)
        return []

    w, h = source.size
    out: List[TextRegion] = []
    for hit in raw:
        # EasyOCR returns [poly, text, conf] per hit where poly is a
        # list of 4 (x, y) corner points (top-left, top-right,
        # bottom-right, bottom-left).
        try:
            poly, text, conf = hit
        except Exception:
            continue
        if not text or not text.strip():
            continue
        if float(conf) < min_confidence:
            continue

        xs = [float(pt[0]) for pt in poly]
        ys = [float(pt[1]) for pt in poly]
        x0, x1 = max(0.0, min(xs)), min(float(w), max(xs))
        y0, y1 = max(0.0, min(ys)), min(float(h), max(ys))
        if x1 <= x0 or y1 <= y0:
            continue

        out.append(
            TextRegion(
                text=str(text).strip(),
                bbox=(x0 / w, y0 / h, x1 / w, y1 / h),
                confidence=float(conf),
            )
        )

    logger.info("text_preservation: extracted %d region(s) from source", len(out))
    return out


def _load_default_font(pixel_height: int) -> ImageFont.ImageFont:
    """Pick a decent sans-serif font at the requested pixel height.

    Falls back to PIL's bitmap default if no TrueType is available.
    """

    # Try a handful of common system fonts in order.  The first that
    # loads wins; PIL silently raises OSError on missing fonts.
    candidates = [
        # Windows
        "arial.ttf",
        "segoeui.ttf",
        "calibri.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    size = max(8, int(pixel_height))
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except (OSError, IOError):
            continue
    # Last-resort: bitmap font (ignores ``size``).
    return ImageFont.load_default()


def _fit_text_in_bbox(
    draw: ImageDraw.ImageDraw,
    text: str,
    bbox_px: Tuple[int, int, int, int],
    min_size: int = 8,
    max_size: int = 200,
) -> Tuple[ImageFont.ImageFont, int, int]:
    """Binary-search a font size so ``text`` fits inside ``bbox_px``.

    Returns ``(font, width_px, height_px)``.  If even the minimum size
    overflows, we use ``min_size`` and accept the clip.
    """

    x0, y0, x1, y1 = bbox_px
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    lo, hi = min_size, min(max_size, bh)
    best_font = _load_default_font(min_size)
    best_w, best_h = 0, 0

    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_default_font(mid)
        try:
            # ``textbbox`` returns (left, top, right, bottom) of the
            # rendered text.  Pillow < 10 uses ``textsize``; we require
            # a recent Pillow elsewhere in the project so textbbox is
            # fine.
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            tw, th = right - left, bottom - top
        except AttributeError:
            tw, th = font.getsize(text)  # type: ignore[attr-defined]
        if tw <= bw and th <= bh:
            best_font, best_w, best_h = font, tw, th
            lo = mid + 1
        else:
            hi = mid - 1

    return best_font, best_w, best_h


def repaint_text_on_output(
    output: Image.Image,
    regions: List[TextRegion],
    background: str = "white",
    text_color: str = "black",
    padding: int = 2,
) -> Image.Image:
    """Paint each region's original text back onto ``output``.

    ``regions`` come from :func:`extract_text_regions` so bounding boxes
    are in normalised source coordinates — we re-project them onto
    ``output.size`` here so the call is safe regardless of any
    intermediate upscale / downscale.

    The function returns a new image (never mutates the input).
    """

    if not regions:
        return output

    # Work on a copy so callers can still reference the untouched
    # generated image for display / logging purposes.
    restored = output.convert("RGB").copy()
    draw = ImageDraw.Draw(restored)
    out_w, out_h = restored.size

    for region in regions:
        nx0, ny0, nx1, ny1 = region.bbox
        x0 = max(0, int(nx0 * out_w) - padding)
        y0 = max(0, int(ny0 * out_h) - padding)
        x1 = min(out_w, int(nx1 * out_w) + padding)
        y1 = min(out_h, int(ny1 * out_h) + padding)
        if x1 <= x0 or y1 <= y0:
            continue

        draw.rectangle([(x0, y0), (x1, y1)], fill=background)
        font, tw, th = _fit_text_in_bbox(draw, region.text, (x0, y0, x1, y1))
        # Center the text inside the padded bbox.
        tx = x0 + max(0, ((x1 - x0) - tw) // 2)
        ty = y0 + max(0, ((y1 - y0) - th) // 2)
        draw.text((tx, ty), region.text, font=font, fill=text_color)

    return restored


def preserve_text(
    source: Optional[Image.Image],
    output: Image.Image,
    device: str = "cuda",
    languages: Tuple[str, ...] = ("en",),
    min_confidence: float = 0.3,
) -> Tuple[Image.Image, str]:
    """High-level entry point used by the generation pipeline.

    Returns ``(image, status)`` where ``image`` is the restored output
    (unchanged on no-op paths) and ``status`` is a short human-readable
    line suitable for the generation status blurb.
    """

    if source is None:
        return output, "text preservation skipped (no source)"

    # Gallery items arrive as ``(image, caption)`` tuples — unwrap.
    if isinstance(source, (tuple, list)):
        source = source[0] if source else None
        if source is None:
            return output, "text preservation skipped (no source)"

    try:
        regions = extract_text_regions(
            source, device=device, languages=languages, min_confidence=min_confidence
        )
    except Exception as exc:
        logger.exception("text preservation extraction failed")
        return output, f"text preservation failed: {exc}"

    if not regions:
        return output, "text preservation: no text detected"

    try:
        restored = repaint_text_on_output(output, regions)
    except Exception as exc:
        logger.exception("text preservation repaint failed")
        return output, f"text preservation failed: {exc}"

    return restored, f"text preserved ({len(regions)} region{'s' if len(regions) != 1 else ''})"
