"""
InsightFace-based Face Swapping for Character Consistency
Memory-optimized for 8GB VRAM (RTX 3070)
"""

import os
import threading
from typing import Optional, List, Union

import torch
import numpy as np
from PIL import Image
import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from src.image.face_analysis_provider import resolve_onnx_providers

# Global cache keyed by device, guarded by a lock so concurrent Gradio
# handlers share a single FaceSwapHelper per device.
_faceswap_cache: dict = {}
_faceswap_lock = threading.Lock()

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

    def unload(self):
        """Unload models to free VRAM."""
        if self.swapper_model is not None:
            del self.swapper_model
            self.swapper_model = None
        # We only null the reference to face_app, as it might be shared
        self.face_app = None
        self.is_loaded = False
        with _faceswap_lock:
            # Evict from cache so the next get_faceswap_helper() creates a fresh instance.
            _faceswap_cache.pop(self.device, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def get_faceswap_helper(device="cuda"):
    """Get or create FaceSwap helper singleton (thread-safe)."""
    with _faceswap_lock:
        helper = _faceswap_cache.get(device)
        if helper is None:
            helper = FaceSwapHelper(device)
            _faceswap_cache[device] = helper
        return helper
