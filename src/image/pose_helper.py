"""
Pose and Facial Expression Extraction Helper
Uses DWPose/OpenPose for preserving character poses and facial expressions
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import Optional, Literal

# Global cache for pose preprocessors
_pose_preprocessor_cache = {}
DEBUG_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "debug")
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
DEBUG_LOG_PATH = os.path.join(DEBUG_LOG_DIR, "ultra_fast_image_gen_debug.log")


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
        """Load pose detection preprocessor."""
        if self.preprocessor is not None:
            return
            
        cache_key = f"{self.detector_type}_{self.device}"
        if cache_key in _pose_preprocessor_cache:
            self.preprocessor = _pose_preprocessor_cache[cache_key]
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
        # #region agent log
        try:
            import time
            log_entry = {
                "timestamp": int(time.time() * 1000),
                "location": "pose_helper.py:63",
                "message": "Pose extraction attempt",
                "data": {"mode": mode, "detector_type": self.detector_type, "image_size": f"{image.size if image else 'None'}"},
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "B"
            }
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion

        if self.preprocessor is None:
            self.load_preprocessor()

        if self.preprocessor is None:
            # #region agent log
            try:
                log_entry = {
                    "timestamp": int(time.time() * 1000),
                    "location": "pose_helper.py:84",
                    "message": "Pose extraction failed - no preprocessor",
                    "data": {"detector_type": self.detector_type},
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B"
                }
                with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception:
                pass
            # #endregion
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

            print(f"  Pose extracted: {mode} mode")

            # #region agent log
            try:
                log_entry = {
                    "timestamp": int(time.time() * 1000),
                    "location": "pose_helper.py:114",
                    "message": "Pose extraction success",
                    "data": {"mode": mode, "pose_image_size": f"{pose_image.size if pose_image else 'None'}"},
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B"
                }
                with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception:
                pass
            # #endregion

            return pose_image

        except Exception as e:
            print(f"  Warning: Pose extraction failed: {e}")

            # #region agent log
            try:
                log_entry = {
                    "timestamp": int(time.time() * 1000),
                    "location": "pose_helper.py:118",
                    "message": "Pose extraction failed with error",
                    "data": {"mode": mode, "detector_type": self.detector_type, "error": str(e)},
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B"
                }
                with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception:
                pass
            # #endregion

            return None
    
    def unload(self):
        """Unload preprocessor to free memory."""
        if self.preprocessor is not None:
            del self.preprocessor
            self.preprocessor = None
            cache_key = f"{self.detector_type}_{self.device}"
            if cache_key in _pose_preprocessor_cache:
                del _pose_preprocessor_cache[cache_key]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  {self.detector_type} unloaded")


def get_pose_extractor(device: str = "cuda", 
                       detector_type: Literal["dwpose", "openpose"] = "dwpose") -> PoseExtractor:
    """
    Get or create PoseExtractor singleton.
    
    Args:
        device: Device to run on
        detector_type: Type of pose detector
        
    Returns:
        PoseExtractor instance
    """
    cache_key = f"{device}_{detector_type}"
    if cache_key in _pose_preprocessor_cache:
        return _pose_preprocessor_cache[cache_key]
    
    extractor = PoseExtractor(device=device, detector_type=detector_type)
    _pose_preprocessor_cache[cache_key] = extractor
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


