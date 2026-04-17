"""
InsightFace-based Face Swapping for Character Consistency
Memory-optimized for 8GB VRAM (RTX 3070)

Two levels of API here:

* ``FaceSwapHelper.swap_face(target, source, face_index=...)`` — single-face
  swap used by the generate pipeline for the legacy single-source flow.
* ``FaceSwapHelper.detect_faces(image) -> List[DetectedFace]`` +
  ``FaceSwapHelper.swap_many(target, assignments)`` — multi-face post-
  generation workflow. ``assignments`` maps detected-face index →
  source PIL image, so the Gradio UI can expose one source slot per
  detected target face.
"""

import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from src.image.face_analysis_provider import resolve_onnx_providers

# Global cache
_faceswap_cache = {}
_faceswap_cache_lock = threading.Lock()


@dataclass
class DetectedFace:
    """
    Lightweight structured result from ``FaceSwapHelper.detect_faces``.

    ``index`` is the slot order (left-to-right by bbox center-x) so the
    UI can keep a stable mapping between "face slot N" and the visible
    face even if the user scans the same image twice.
    ``thumbnail`` is a small PIL preview of the cropped face, safe to
    render as a Gradio Image component.
    ``embedding`` is the 512-d L2-normalized embedding from InsightFace;
    use ``cosine_similarity`` to compare across generations.
    """

    index: int
    bbox: Tuple[int, int, int, int]
    thumbnail: Image.Image
    embedding: np.ndarray
    confidence: float

class FaceSwapHelper:
    """Memory-optimized face swapper using InsightFace inswapper model."""

    def __init__(self, device="cuda"):
        self.device = device
        self.face_app = None
        self.swapper_model = None
        self.is_loaded = False

    def load_models(self):
        """Load face detection and swap models."""
        # Reuse FaceAnalysis provider to save VRAM
        try:
            from src.image.face_analysis_provider import get_face_analysis
            self.face_app = get_face_analysis(device=self.device, name='antelopev2')
            print("[FaceSwap] Using shared FaceAnalysis instance")
        except Exception as e:
            print(f"[FaceSwap] Error during FaceAnalysis setup: {e}")
            # Fallback
            from src.image.face_analysis_provider import get_face_analysis
            self.face_app = get_face_analysis(device=self.device, name='buffalo_l')

        # Load inswapper model
        # The model is usually inswapper_128.onnx
        try:
            providers = resolve_onnx_providers(self.device)
            self.swapper_model = get_model('inswapper_128.onnx', download=True, providers=providers)
            print("[FaceSwap] Inswapper model loaded successfully")
        except Exception as e:
            print(f"[FaceSwap] Error loading swapper model: {e}")
            print("[FaceSwap] Note: inswapper model download may fail. FaceSwap will not be available.")
            self.swapper_model = None

        self.is_loaded = True

    def find_similar_faces(self, faces, reference_face, similarity_threshold=0.6):
        """Find faces similar to the reference face based on embeddings."""
        if not faces or reference_face is None:
            return []

        # Get reference embedding
        ref_embedding = reference_face.embedding

        similar_faces = []
        for face in faces:
            if face.embedding is not None:
                # Calculate cosine similarity
                similarity = self.cosine_similarity(ref_embedding, face.embedding)
                if similarity >= similarity_threshold:
                    similar_faces.append((face, similarity))

        # Sort by similarity (highest first)
        similar_faces.sort(key=lambda x: x[1], reverse=True)
        return [face for face, sim in similar_faces]

    def cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two face embeddings."""
        import numpy as np
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)

        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def swap_face(
        self,
        target_image: Image.Image,
        source_image: Image.Image,
        face_index: int = 0,
        use_similarity: bool = True,
        similarity_threshold: float = 0.6
    ) -> Image.Image:
        """Swap face in target with face from source."""
        if not self.is_loaded:
            self.load_models()

        if self.swapper_model is None:
            raise ValueError("FaceSwap model not available. The inswapper model failed to download.")

        # Convert PIL to numpy (BGR for InsightFace/OpenCV)
        target_np = cv2.cvtColor(np.array(target_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        source_np = cv2.cvtColor(np.array(source_image.convert('RGB')), cv2.COLOR_RGB2BGR)

        # Detect faces
        source_faces = self.face_app.get(source_np)
        target_faces = self.face_app.get(target_np)

        if len(source_faces) == 0:
            raise ValueError("No face detected in source image")
        if len(target_faces) == 0:
            raise ValueError("No face detected in target image")

        # Select source face (largest by default)
        source_face = sorted(source_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))[-1]

        # Select target face
        if use_similarity and len(target_faces) > 1:
            # Find faces similar to the source face
            similar_faces = self.find_similar_faces(target_faces, source_face, similarity_threshold)
            if similar_faces:
                target_face = similar_faces[0]  # Use the most similar face
                print(f"Found {len(similar_faces)} similar faces, using the most similar one")
            else:
                # Fall back to size-based selection if no similar faces found
                sorted_target_faces = sorted(target_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                if face_index < len(sorted_target_faces):
                    target_face = sorted_target_faces[face_index]
                else:
                    target_face = sorted_target_faces[0]
                print("No similar faces found, using size-based selection")
        else:
            # Use traditional size-based selection
            sorted_target_faces = sorted(target_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            if face_index < len(sorted_target_faces):
                target_face = sorted_target_faces[face_index]
            else:
                target_face = sorted_target_faces[0]

        # Perform swap
        result = self.swapper_model.get(target_np, target_face, source_face, paste_back=True)

        # Convert back to PIL
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)

    # ------------------------------------------------------------------
    # Multi-face post-generation workflow
    # ------------------------------------------------------------------
    def detect_faces(
        self,
        image: Image.Image,
        *,
        thumbnail_size: int = 160,
    ) -> List[DetectedFace]:
        """
        Detect every face in ``image`` and return one ``DetectedFace`` per
        hit, ordered left-to-right by bbox center-x so the UI slot order
        stays stable across repeated scans.

        ``thumbnail_size`` is the longest side of the cropped face preview
        returned in ``DetectedFace.thumbnail``. 160 px is a good default
        for a Gradio Image tile without blowing up state size.

        Returns an empty list when no faces are detected; callers are
        expected to surface that to the user.
        """
        if not self.is_loaded:
            self.load_models()

        rgb = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        raw_faces = self.face_app.get(bgr)
        if not raw_faces:
            return []

        # Stable left-to-right ordering by bbox center-x.
        ordered = sorted(
            raw_faces,
            key=lambda f: float((f.bbox[0] + f.bbox[2]) / 2.0),
        )

        results: List[DetectedFace] = []
        h_img, w_img = rgb.shape[:2]
        for idx, face in enumerate(ordered):
            x1, y1, x2, y2 = (int(max(0, v)) for v in face.bbox)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Expand bbox a little so the thumbnail includes hairline/chin.
            pad_x = int((x2 - x1) * 0.15)
            pad_y = int((y2 - y1) * 0.20)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w_img, x2 + pad_x)
            cy2 = min(h_img, y2 + pad_y)
            crop = rgb[cy1:cy2, cx1:cx2]
            thumb = Image.fromarray(crop)
            thumb.thumbnail(
                (thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS
            )

            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                emb = np.asarray(face.embedding, dtype=np.float32)
                norm = float(np.linalg.norm(emb))
                if norm > 0:
                    emb = emb / norm
            else:
                emb = np.asarray(emb, dtype=np.float32)

            results.append(
                DetectedFace(
                    index=idx,
                    bbox=(x1, y1, x2, y2),
                    thumbnail=thumb,
                    embedding=emb,
                    confidence=float(getattr(face, "det_score", 0.0)),
                )
            )
        return results

    def swap_many(
        self,
        target_image: Image.Image,
        assignments: Dict[int, Image.Image],
    ) -> Tuple[Image.Image, List[int]]:
        """
        Swap each face in ``target_image`` whose index is in
        ``assignments`` with the largest face detected in the matching
        source PIL image.

        Returns ``(result_image, swapped_indices)`` where
        ``swapped_indices`` is the subset of ``assignments`` keys that
        were actually swapped (ones whose source had no detectable face
        are silently skipped and reported in the return value).

        Indices refer to the left-to-right detection order returned by
        :meth:`detect_faces`. Unassigned faces in the target image are
        left untouched.
        """
        if not self.is_loaded:
            self.load_models()
        if self.swapper_model is None:
            raise ValueError(
                "FaceSwap model not available. The inswapper model failed to download."
            )
        if not assignments:
            return target_image, []

        target_rgb = np.array(target_image.convert("RGB"))
        target_bgr = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)
        raw_faces = self.face_app.get(target_bgr)
        if not raw_faces:
            raise ValueError("No face detected in target image")

        ordered_target_faces = sorted(
            raw_faces,
            key=lambda f: float((f.bbox[0] + f.bbox[2]) / 2.0),
        )

        # Cache detected source-face per source image identity so the
        # same reference isn't re-detected for every slot.
        source_cache: Dict[int, object] = {}
        swapped: List[int] = []

        working = target_bgr
        for slot_idx, src_image in assignments.items():
            if src_image is None:
                continue
            if slot_idx < 0 or slot_idx >= len(ordered_target_faces):
                print(
                    f"[FaceSwap] Slot {slot_idx} out of range "
                    f"(detected {len(ordered_target_faces)} faces); skipping."
                )
                continue

            key = id(src_image)
            src_face = source_cache.get(key)
            if src_face is None:
                src_rgb = np.array(src_image.convert("RGB"))
                src_bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)
                src_faces = self.face_app.get(src_bgr)
                if not src_faces:
                    print(
                        f"[FaceSwap] No face detected in source for slot {slot_idx}; "
                        f"skipping."
                    )
                    continue
                src_face = max(
                    src_faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0])
                    * (f.bbox[3] - f.bbox[1]),
                )
                source_cache[key] = src_face

            target_face = ordered_target_faces[slot_idx]
            working = self.swapper_model.get(
                working, target_face, src_face, paste_back=True
            )
            swapped.append(slot_idx)

        result_rgb = cv2.cvtColor(working, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb), swapped

    def auto_swap_with_library(
        self,
        target_image: Image.Image,
        *,
        similarity_threshold: float = 0.35,
        max_faces: int = 8,
    ) -> Tuple[Image.Image, List[str]]:
        """
        Post-process ``target_image`` by swapping every detected face whose
        embedding matches a saved character in the library.

        This is the "consistent character across generations" path: save
        a character once (e.g. "Alice"), and every subsequent Single /
        Batch / Video output containing a face that resembles Alice's
        embedding gets her saved reference pasted back in via inswapper.

        ``similarity_threshold`` is cosine similarity on the normalized
        InsightFace embedding. 0.35 is a reasonable default — too high
        and repeated-character matches miss, too low and unrelated faces
        get accidentally swapped.

        Returns ``(result_image, matched_names)`` where ``matched_names``
        is the ordered list of library character names that were swapped
        in. An empty list means nothing matched and the image is
        unchanged.
        """
        from src.image import character_library

        entries = character_library.list_characters()
        if not entries:
            return target_image, []

        detected = self.detect_faces(target_image)
        if not detected:
            return target_image, []

        # For each detected face, find the best library match above the
        # threshold. We use each library character at most once per
        # image so two detected faces don't both map to "Alice".
        used_entries: set = set()
        assignments: Dict[int, Image.Image] = {}
        matched_names: List[str] = []
        entry_embeddings = [
            (entry, entry.load_embedding()) for entry in entries
        ]

        for face in detected[:max_faces]:
            best: Tuple[float, Any] = (similarity_threshold, None)
            for entry, emb in entry_embeddings:
                if entry.name in used_entries:
                    continue
                if emb is None or emb.shape != face.embedding.shape:
                    continue
                denom = (
                    float(np.linalg.norm(emb))
                    * float(np.linalg.norm(face.embedding))
                )
                if denom <= 0:
                    continue
                sim = float(np.dot(emb, face.embedding) / denom)
                if sim > best[0]:
                    best = (sim, entry)
            if best[1] is not None:
                entry = best[1]
                used_entries.add(entry.name)
                assignments[face.index] = entry.reference_image
                matched_names.append(entry.name)

        if not assignments:
            return target_image, []

        result, _ = self.swap_many(target_image, assignments)
        return result, matched_names

    def extract_source_face(self, image: Image.Image):
        """
        Return the InsightFace ``Face`` object for the largest face in
        ``image``, or ``None`` if no face is detected. Useful when the
        caller wants to reuse the same reference for several swaps
        without re-running detection per call.
        """
        if not self.is_loaded:
            self.load_models()
        rgb = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = self.face_app.get(bgr)
        if not faces:
            return None
        return max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

    def unload(self):
        """Unload models to free VRAM."""
        if self.swapper_model is not None:
            del self.swapper_model
            self.swapper_model = None
        # We only null the reference to face_app, as it might be shared
        self.face_app = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def get_faceswap_helper(device="cuda"):
    """Get or create FaceSwap helper singleton (thread-safe)."""
    with _faceswap_cache_lock:
        if device not in _faceswap_cache:
            _faceswap_cache[device] = FaceSwapHelper(device)
        return _faceswap_cache[device]
