"""
PuLID-FLUX Helper for Character Consistency
Optimized for 8GB VRAM (RTX 3070)
"""

import os
import json
import threading
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from src.constants import LEGACY_CHARACTER_MANAGER_STATE_FILENAME

# Global cache for PuLID models, keyed by ``{device}_{enable_fp8}``.
# A lock guards the get-or-create path so two concurrent Gradio handlers
# (e.g. a generation request and a Character Manager refresh) don't both
# instantiate PuLID and double-allocate ~1.5 GB of VRAM.
_pulid_cache = {}
_pulid_cache_lock = threading.Lock()

def get_memory_usage():
    """Get current CUDA memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x

class PuLIDEncoder(nn.Module):
    def __init__(self, clip_dim=1024, transformer_dim=3072):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
            MLP(transformer_dim, transformer_dim, transformer_dim * 4),
            MLP(transformer_dim, transformer_dim, transformer_dim * 4),
        )

    def forward(self, face_features):
        return self.projection(face_features)

class PuLIDHelper:
    """
    Memory-optimized PuLID helper for FLUX models.
    Uses CPU offloading and FP8/FP16 to work on 8GB VRAM.
    """
    def __init__(self, device="cuda", enable_fp8=True):
        self.device = device
        self.enable_fp8 = enable_fp8
        self.face_app = None
        self.eva_clip = None
        self.pulid_encoder = None
        self.is_loaded = False

    def load_models(self):
        """Load PuLID models with memory optimization."""
        if self.is_loaded:
            return

        print("Loading PuLID models...")
        try:
            # 1. Load face detection model (InsightFace)
            from src.image.face_analysis_provider import get_face_analysis
            self.face_app = get_face_analysis(device=self.device, name='antelopev2')
            print("  Face detection model loaded (shared)")

            # 2. Load EVA-CLIP (Visual Encoder)
            self.eva_clip = timm.create_model(
                "eva02_large_patch14_336.mim_in22k_ft_in1k",
                pretrained=True,
                num_classes=0,
            ).to(self.device)
            self.eva_clip.eval()
            if self.enable_fp8:
                self.eva_clip = self.eva_clip.to(torch.float16)
            print("  EVA-CLIP model loaded")

            # 3. Load PuLID weights
            model_path = hf_hub_download(
                repo_id="guozinan/PuLID",
                filename="pulid_flux_v0.9.1.safetensors",
                local_dir="models/pulid"
            )

            self.pulid_encoder = PuLIDEncoder().to(self.device)
            state_dict = load_file(model_path)
            # Filter prefix if present (some versions have it)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("pulid_encoder."):
                    new_state_dict[k.replace("pulid_encoder.", "")] = v
                else:
                    new_state_dict[k] = v

            self.pulid_encoder.load_state_dict(new_state_dict, strict=False)
            self.pulid_encoder.eval()
            if self.enable_fp8:
                self.pulid_encoder = self.pulid_encoder.to(torch.float16)
            print(f"  PuLID weights loaded")

            self.is_loaded = True
        except Exception as e:
            print(f"  Warning: PuLID loading failed: {e}")
            self.is_loaded = False

    def get_pulid_embedding(self, image: Image.Image):
        """Extract face features and project to PuLID embedding."""
        if not self.is_loaded:
            self.load_models()

        if self.face_app is None or self.eva_clip is None or self.pulid_encoder is None:
            return None

        try:
            img_np = np.array(image.convert('RGB'))
            faces = self.face_app.get(img_np)
            if len(faces) == 0:
                return None

            # Use largest face
            face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))[-1]

            # Align face (proxy for now, use norm_face if available)
            if hasattr(face, 'norm_face') and face.norm_face is not None:
                aligned_face = face.norm_face
            else:
                bbox = face.bbox.astype(int)
                # Pad bbox slightly
                h, w = img_np.shape[:2]
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = min(w, bbox[2])
                bbox[3] = min(h, bbox[3])
                aligned_face = img_np[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            # Preprocess
            face_tensor = torch.from_numpy(aligned_face).permute(2, 0, 1).unsqueeze(0).float()
            face_tensor = F.interpolate(face_tensor, size=(336, 336), mode='bicubic', align_corners=False)
            face_tensor = face_tensor.to(self.device)
            if self.enable_fp8:
                face_tensor = face_tensor.to(torch.float16)

            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            face_tensor = (face_tensor / 255.0 - mean) / std

            with torch.no_grad():
                features = self.eva_clip.forward_features(face_tensor)
                embedding = self.pulid_encoder(features)
                return embedding

        except Exception as e:
            print(f"  Warning: PuLID embedding extraction failed: {e}")
            return None

    def unload(self):
        if self.face_app: del self.face_app; self.face_app = None
        if self.eva_clip: del self.eva_clip; self.eva_clip = None
        if self.pulid_encoder: del self.pulid_encoder; self.pulid_encoder = None
        self.is_loaded = False
        if torch.cuda.is_available(): torch.cuda.empty_cache()

def get_pulid_helper(device="cuda", enable_fp8=True):
    """Get or create PuLIDHelper singleton (thread-safe).

    Holds ``_pulid_cache_lock`` across the check-then-act so two
    concurrent Gradio handlers can't both observe the cache as empty
    and double-instantiate PuLID (+~1.5 GB VRAM).
    """
    key = f"{device}_{enable_fp8}"
    with _pulid_cache_lock:
        helper = _pulid_cache.get(key)
        if helper is None:
            helper = PuLIDHelper(device, enable_fp8)
            _pulid_cache[key] = helper
        return helper

class MultiCharacterManager:
    """Manages character detection, clustering and embedding storage."""
    def __init__(self, device="cuda"):
        self.device = device
        self.characters = [] # List of dicts
        self.face_app = None
        self.pulid_helper = None

    def load_face_detector(self):
        if self.face_app is None:
            from src.image.face_analysis_provider import get_face_analysis
            self.face_app = get_face_analysis(device=self.device, name='antelopev2')

    def detect_characters_from_folder(self, folder_path: str):
        self.load_face_detector()
        all_faces = scan_folder_for_faces(folder_path, self.face_app)
        if not all_faces: return []

        clusters = cluster_faces_by_identity(all_faces)
        self.characters = []
        for idx, cluster in enumerate(clusters):
            rep = select_representative_face(cluster)
            self.characters.append({
                'character_id': f'char_{idx}',
                'representative_face': rep['face_crop'],
                'embedding': rep['embedding'],
                'count': len(cluster),
                'reference_image': None,
                'pulid_embedding': None
            })
        return self.characters

    def assign_reference_image(self, character_id, reference_image):
        if self.pulid_helper is None:
            self.pulid_helper = get_pulid_helper(self.device)

        for char in self.characters:
            if char['character_id'] == character_id:
                emb = self.pulid_helper.get_pulid_embedding(reference_image)
                if emb is not None:
                    char['reference_image'] = reference_image
                    char['pulid_embedding'] = emb
                    return True
        return False

    def get_embeddings_for_generation(self, target_dim: Optional[int] = None):
        """
        Get embeddings for generation, optionally padded/trimmed to target_dim.
        
        Args:
            target_dim: Optional target dimension to pad/trim embeddings to
            
        Returns:
            List of embeddings for characters with PuLID embeddings
        """
        embeddings = [char['pulid_embedding'] for char in self.characters if char['pulid_embedding'] is not None]
        
        if target_dim is not None and embeddings:
            processed_embeddings = []
            for emb in embeddings:
                # emb shape: [1, 3072] typically
                if emb.shape[-1] != target_dim:
                    if emb.shape[-1] > target_dim:
                        # Trim to target_dim
                        processed_emb = emb[..., :target_dim]
                    else:
                        # Pad with zeros to target_dim
                        pad_size = target_dim - emb.shape[-1]
                        processed_emb = F.pad(emb, (0, pad_size), mode='constant', value=0)
                    processed_embeddings.append(processed_emb)
                else:
                    processed_embeddings.append(emb)
            return processed_embeddings
        
        return embeddings

    def save_state(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serializable_chars = []
        for char in self.characters:
            c = char.copy()
            c['representative_face'] = None
            c['reference_image'] = None
            if c['pulid_embedding'] is not None:
                c['pulid_embedding_cpu'] = c['pulid_embedding'].detach().cpu().numpy().tolist()
            c['pulid_embedding'] = None
            serializable_chars.append(c)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable_chars, f)

    def load_state(self, path):
        state_path = path
        if not os.path.exists(state_path) and path.endswith(".json"):
            legacy_path = os.path.join(os.path.dirname(path), LEGACY_CHARACTER_MANAGER_STATE_FILENAME)
            if os.path.exists(legacy_path):
                state_path = legacy_path
            else:
                return
        elif not os.path.exists(state_path):
            return

        data = self._load_serialized_state(state_path)
        normalized_characters = []
        for char in data:
            normalized = dict(char)
            embedding_payload = normalized.pop('pulid_embedding_cpu', None)
            if embedding_payload is not None:
                embedding_array = np.asarray(embedding_payload, dtype=np.float32)
                normalized['pulid_embedding'] = torch.from_numpy(embedding_array).to(self.device)
            else:
                normalized['pulid_embedding'] = None
            normalized.setdefault('representative_face', None)
            normalized.setdefault('reference_image', None)
            normalized_characters.append(normalized)
        self.characters = normalized_characters

    def _load_serialized_state(self, path):
        if path.endswith(".pkl"):
            try:
                import pickle
                with open(path, "rb") as handle:
                    payload = pickle.load(handle)
                if isinstance(payload, list):
                    return payload
            except Exception:
                pass

        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            if isinstance(payload, list):
                return payload
        except Exception as exc:
            raise RuntimeError(f"Failed to load character state from {path}: {exc}") from exc

        raise RuntimeError(f"Unsupported character state payload in {path}")

def scan_folder_for_faces(folder_path, face_app):
    all_faces = []
    exts = {'.jpg', '.jpeg', '.png', '.webp'}
    for p in Path(folder_path).rglob("*"):
        if p.suffix.lower() in exts:
            try:
                img = Image.open(p)
                img_np = np.array(img.convert('RGB'))
                faces = face_app.get(img_np)
                for face in faces:
                    bbox = face.bbox.astype(int)
                    all_faces.append({
                        'image_path': str(p),
                        'face_crop': img.crop((bbox[0], bbox[1], bbox[2], bbox[3])),
                        'embedding': face.embedding,
                        'bbox': bbox
                    })
            except Exception:
                continue
    return all_faces

def cluster_faces_by_identity(faces, threshold=0.4):
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.cluster import DBSCAN
    embs = np.array([f['embedding'] for f in faces])
    dist = cosine_distances(embs)
    db = DBSCAN(eps=threshold, min_samples=2, metric='precomputed').fit(dist)
    clusters = {}
    for i, label in enumerate(db.labels_):
        l = f"single_{i}" if label == -1 else label
        if l not in clusters: clusters[l] = []
        clusters[l].append(faces[i])
    return list(clusters.values())

def select_representative_face(cluster):
    return max(cluster, key=lambda f: (f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1]))

def enhance_prompt_with_character_description(prompt: str, description: str) -> str:
    """
    Enhance a prompt by appending character description.
    
    Args:
        prompt: Original prompt
        description: Character description to append
        
    Returns:
        Enhanced prompt with character description
    """
    if not description:
        return prompt
    
    # Clean up the description and prompt
    prompt = prompt.strip()
    description = description.strip()
    
    # If prompt is empty, just return description
    if not prompt:
        return description
    
    # Append description with proper spacing
    if prompt.endswith(','):
        # If prompt ends with comma, just add space and description
        return f"{prompt} {description}"
    else:
        # Otherwise add comma and description
        return f"{prompt}, {description}"

class PuLIDFluxPatch:
    """
    Patches FLUX transformer blocks to inject PuLID face embeddings.
    """
    def __init__(self, transformer, embeddings: List[torch.Tensor], weight=1.0):
        self.transformer = transformer
        self.embeddings = embeddings
        self.weight = weight
        self.original_forward_methods = {}

    def patch(self):
        """Hijack the forward method of transformer blocks.

        Refuses to patch if the transformer is already patched by another
        PuLIDFluxPatch instance, to avoid silently dropping this instance's
        embeddings or corrupting the restore chain. Callers must unpatch the
        previous instance first.
        """
        import types

        # We need to inject into both FluxTransformerBlock and FluxSingleTransformerBlock
        # But for simplicity and memory, let's focus on the first few blocks or all double blocks

        # PuLID paper suggests injecting into the context (cross-attention)
        # In FLUX, context is shared across blocks.

        print(f"  Patching FLUX transformer with {len(self.embeddings)} face embeddings...")

        # For now, we concatenate all embeddings
        if not self.embeddings:
            return

        # Refuse to layer on top of another active patch — it would silently
        # stack embeddings or leave stale `_original_forward` refs after unpatch.
        if getattr(self.transformer, "_pulid_patch_owner", None) is not None:
            print("  Warning: transformer already has an active PuLID patch; skipping.")
            return

        # [N, 1, 3072] -> [1, N, 3072]
        all_embeddings = torch.cat(self.embeddings, dim=1)

        # Apply weighting
        all_embeddings = all_embeddings * self.weight

        def patched_forward(block_self, hidden_states, encoder_hidden_states, *args, **kwargs):
            # Inject face embeddings into encoder_hidden_states (context)
            # encoder_hidden_states shape: [B, L, C]

            # Expand face embeddings to match batch dimension
            all_emb = all_embeddings.to(encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
            if all_emb.shape[0] != encoder_hidden_states.shape[0]:
                all_emb = all_emb.expand(encoder_hidden_states.shape[0], -1, -1)

            # Concatenate face embeddings to context
            new_encoder_hidden_states = torch.cat([encoder_hidden_states, all_emb], dim=1)

            # Call original forward
            return block_self._original_forward(hidden_states, new_encoder_hidden_states, *args, **kwargs)

        # Patch double blocks
        if hasattr(self.transformer, 'transformer_blocks'):
            for i, block in enumerate(self.transformer.transformer_blocks):
                if not hasattr(block, '_original_forward'):
                    block._original_forward = block.forward
                    block.forward = types.MethodType(patched_forward, block)
                    self.original_forward_methods[f"double_{i}"] = block

        # Patch single blocks
        if hasattr(self.transformer, 'single_transformer_blocks'):
            for i, block in enumerate(self.transformer.single_transformer_blocks):
                if not hasattr(block, '_original_forward'):
                    block._original_forward = block.forward
                    block.forward = types.MethodType(patched_forward, block)
                    self.original_forward_methods[f"single_{i}"] = block

        # Mark the transformer so subsequent patches are rejected until unpatch.
        self.transformer._pulid_patch_owner = id(self)

    def unpatch(self):
        """Restore original forward methods."""
        for key, block in self.original_forward_methods.items():
            if hasattr(block, '_original_forward'):
                block.forward = block._original_forward
                del block._original_forward
        self.original_forward_methods = {}
        # Only clear the owner flag if we own it.
        if getattr(self.transformer, "_pulid_patch_owner", None) == id(self):
            self.transformer._pulid_patch_owner = None
        print("  FLUX transformer unpatched")

