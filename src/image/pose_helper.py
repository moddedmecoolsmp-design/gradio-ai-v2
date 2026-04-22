"""
Pose and Facial Expression Extraction Helper
Uses DWPose/OpenPose for preserving character poses and facial expressions
"""

import logging
import threading

import torch
import numpy as np
from PIL import Image
from typing import Optional, Literal

logger = logging.getLogger(__name__)

# Two independent caches that used to share a single dict with reversed key
# orders (``{detector}_{device}`` for preprocessors, ``{device}_{detector}``
# for extractors).  They didn't collide by accident, but ``PoseExtractor.
# unload()`` only evicted one side, leaving the other reference alive and
# preventing VRAM reclaim.  Splitting them makes the lifecycle explicit and
# lets the shared lock guard concurrent load paths cheaply.
_pose_preprocessor_cache = {}   # key: f"{detector_type}_{device}"
_pose_extractor_cache = {}      # key: f"{device}_{detector_type}"
_pose_cache_lock = threading.Lock()


class PoseExtractor:
    """
    Extract pose skeletons with facial landmarks from images.
    Supports DWPose and OpenPose preprocessors.
    """
    
    def __init__(self, 
                 device: str = "cuda",
                 detector_type: Literal["dwpose", "openpose"] = "dwpose"):
        """
        Args:
            device: Device to run pose detection on ("cuda" or "cpu")
            detector_type: Type of pose detector ("dwpose" or "openpose")
        """
        self.device = device
        self.detector_type = detector_type
        self.preprocessor = None
        
    def load_preprocessor(self):
        """Load pose detection preprocessor (thread-safe).

        Holds ``_pose_cache_lock`` across the entire check-then-load so
        two concurrent requests can't both observe the cache as empty
        and both fall through to ``from_pretrained``.  Loading is the
        slow step, not the lookup — making it atomic with the check is
        the whole point.
        """
        if self.preprocessor is not None:
            return

        cache_key = f"{self.detector_type}_{self.device}"
        with _pose_cache_lock:
            cached = _pose_preprocessor_cache.get(cache_key)
            if cached is not None:
                self.preprocessor = cached
                return

            try:
                from controlnet_aux import DWposeDetector, OpenposeDetector

                print(f"Loading {self.detector_type} pose detector...")

                if self.detector_type == "dwpose":
                    # DWPose: More accurate, includes facial landmarks
                    self.preprocessor = DWposeDetector.from_pretrained("lllyasviel/Annotators")
                else:
                    # OpenPose: Classic pose detector
                    self.preprocessor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
                _pose_preprocessor_cache[cache_key] = self.preprocessor
                print(f"  {self.detector_type} loaded successfully")

            except Exception as e:
                print(f"  Warning: Failed to load {self.detector_type}: {e}")
                print("  Pose preservation will not be available")
                self.preprocessor = None
    
    def extract_pose(self,
                    image: Image.Image,
                    mode: Literal["body", "body_face", "body_face_hands"] = "body_face",
                    detect_resolution: int = 512,
                    image_resolution: int = 512) -> Optional[Image.Image]:
        """
        Extract pose skeleton from image.

        Args:
            image: Input PIL Image
            mode: Detection mode
                - "body": Body skeleton only (18 keypoints)
                - "body_face": Body + facial landmarks (18 + 70 keypoints)
                - "body_face_hands": Full body including hands (18 + 70 + 42 keypoints)
            detect_resolution: Resolution for pose detection
            image_resolution: Output resolution

        Returns:
            PIL Image with pose skeleton visualization, or None if detection fails
        """
        logger.debug("Pose extraction: mode=%s detector=%s size=%s",
                     mode, self.detector_type, image.size if image else None)

        if self.preprocessor is None:
            self.load_preprocessor()

        if self.preprocessor is None:
            logger.warning("Pose extraction skipped: no preprocessor (%s)", self.detector_type)
            return None

        try:
            # Convert mode to detector parameters
            include_hands = mode == "body_face_hands"
            include_face = mode in ["body_face", "body_face_hands"]
            include_body = True  # Always include body

            # Extract pose
            if self.detector_type == "dwpose":
                pose_image = self.preprocessor(
                    image,
                    detect_resolution=detect_resolution,
                    image_resolution=image_resolution,
                    include_hands=include_hands,
                    include_face=include_face,
                    include_body=include_body,
                    output_type="pil"
                )
            else:
                # OpenPose parameters
                pose_image = self.preprocessor(
                    image,
                    detect_resolution=detect_resolution,
                    image_resolution=image_resolution,
                    hand_and_face=include_face and include_hands
                )

            logger.debug("Pose extracted: mode=%s size=%s",
                         mode, pose_image.size if pose_image else None)
            return pose_image

        except Exception as e:
            logger.warning("Pose extraction failed: %s (mode=%s, detector=%s)",
                           e, mode, self.detector_type)
            return None
    
    def unload(self):
        """Unload preprocessor and evict both caches to free memory."""
        if self.preprocessor is not None:
            del self.preprocessor
            self.preprocessor = None
            preprocessor_key = f"{self.detector_type}_{self.device}"
            extractor_key = f"{self.device}_{self.detector_type}"
            with _pose_cache_lock:
                _pose_preprocessor_cache.pop(preprocessor_key, None)
                _pose_extractor_cache.pop(extractor_key, None)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  {self.detector_type} unloaded")


def get_pose_extractor(device: str = "cuda",
                       detector_type: Literal["dwpose", "openpose"] = "dwpose") -> PoseExtractor:
    """
    Get or create PoseExtractor singleton (thread-safe).

    Previously stored into the same dict as the raw preprocessor with a
    reversed key order, which happened not to collide but also meant
    ``PoseExtractor.unload()`` couldn't reach this side to evict it.
    Dedicated cache + shared lock fix both problems.
    """
    cache_key = f"{device}_{detector_type}"
    with _pose_cache_lock:
        extractor = _pose_extractor_cache.get(cache_key)
        if extractor is None:
            extractor = PoseExtractor(device=device, detector_type=detector_type)
            _pose_extractor_cache[cache_key] = extractor
        return extractor


def batch_extract_poses(images: list,
                        mode: Literal["body", "body_face", "body_face_hands"] = "body_face",
                        device: str = "cuda",
                        detector_type: Literal["dwpose", "openpose"] = "dwpose") -> list:
    """
    Extract poses from multiple images.
    
    Args:
        images: List of PIL Images
        mode: Detection mode
        device: Device to run on
        detector_type: Type of detector
        
    Returns:
        List of pose skeleton images (PIL Images or None for failed extractions)
    """
    extractor = get_pose_extractor(device=device, detector_type=detector_type)
    extractor.load_preprocessor()
    
    pose_images = []
    for i, img in enumerate(images):
        print(f"Extracting pose {i+1}/{len(images)}...")
        pose_img = extractor.extract_pose(img, mode=mode)
        pose_images.append(pose_img)
    
    return pose_images


