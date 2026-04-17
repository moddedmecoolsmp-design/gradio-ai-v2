"""
Pose + facial-expression preservation helper.

The preservation feature lets a user supply a *reference* image and have the
generated image inherit its pose and facial expression. It does **not**
overwrite the identity of the character the user prompts for — that's what
face swap is for. It's a pre-generation conditioning step:

    user reference ---[DWPose / OpenPose]---> pose skeleton
                                              (body + face landmarks)
    pose skeleton + prompt ---> FLUX.2-klein ---> generated image

FLUX.2-klein accepts multiple reference images via its native image-editing
path, so we prepend the pose skeleton to ``input_images`` and let the model
follow it. We also augment the prompt with a short directive so the model
treats the extra reference as pose guidance rather than content to copy.

Design notes
~~~~~~~~~~~~
* Auto-installs ``controlnet_aux`` on first use (the package supplies
  DWPose + OpenPose preprocessors).
* Falls back gracefully: if the preprocessor fails to load, preservation
  is silently skipped and the pipeline runs without pose conditioning.
* Works on CPU but much faster on CUDA (DWPose is ONNX-based and runs
  comfortably on RTX 3070 in ~50 ms per frame).
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

from PIL import Image

from src.image.pose_helper import get_pose_extractor

logger = logging.getLogger(__name__)

PoseMode = Literal["body", "body_face", "body_face_hands"]
Detector = Literal["dwpose", "openpose"]

_POSE_PROMPT_SUFFIX = (
    " Follow the pose and facial expression shown in the reference pose skeleton."
)


def extract_preservation_reference(
    reference_image: Image.Image,
    device: str = "cuda",
    detector: Detector = "dwpose",
    mode: PoseMode = "body_face",
    detect_resolution: int = 512,
    output_resolution: int = 1024,
) -> Optional[Image.Image]:
    """
    Run DWPose / OpenPose on ``reference_image`` and return a skeleton image.

    Returns ``None`` if the preprocessor can't load or finds no pose — callers
    should fall back to a plain generation in that case.
    """
    if reference_image is None:
        return None

    try:
        extractor = get_pose_extractor(device=device, detector_type=detector)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Preservation: failed to init pose extractor: %s", exc)
        return None

    try:
        skeleton = extractor.extract_pose(
            reference_image,
            mode=mode,
            detect_resolution=detect_resolution,
            image_resolution=output_resolution,
        )
    except Exception as exc:
        logger.warning("Preservation: pose extraction raised: %s", exc)
        return None

    if skeleton is None:
        logger.info("Preservation: no pose detected in reference image.")
        return None

    return skeleton


def build_preservation_inputs(
    preservation_input: Optional[Image.Image],
    existing_input_images: Optional[List] = None,
    device: str = "cuda",
    detector: Detector = "dwpose",
    mode: PoseMode = "body_face",
) -> tuple[Optional[List], Optional[Image.Image], str]:
    """
    Prepare reference inputs for the FLUX.2-klein pipeline.

    Returns a tuple ``(input_images, pose_skeleton, status)`` where:

    * ``input_images`` is the augmented reference list suitable for passing
      to the pipeline's ``image=`` kwarg (preserves existing references and
      prepends the pose skeleton).
    * ``pose_skeleton`` is the raw skeleton PIL image, returned separately
      so the UI can preview what the model will condition on.
    * ``status`` is a short human-readable message.

    If ``preservation_input`` is ``None`` or pose extraction fails, returns
    ``(existing_input_images, None, "<reason>")`` so the caller can carry on
    unmodified.
    """
    if preservation_input is None:
        return existing_input_images, None, "Preservation: no reference image."

    skeleton = extract_preservation_reference(
        preservation_input,
        device=device,
        detector=detector,
        mode=mode,
    )
    if skeleton is None:
        return (
            existing_input_images,
            None,
            "Preservation: pose extraction failed; generating without pose conditioning.",
        )

    augmented: List = []
    if existing_input_images:
        augmented.extend(existing_input_images)
    # Skeleton goes first so the pipeline treats it as the primary pose
    # reference when multiple references are supplied.
    augmented.insert(0, skeleton)
    # FLUX.2-klein accepts up to 6 reference images (see image_gen.py img2img
    # path); trim if the user already supplied the full 6.
    if len(augmented) > 6:
        augmented = augmented[:6]

    return augmented, skeleton, "Preservation: pose + expression reference applied."


def augment_prompt_for_preservation(prompt: str) -> str:
    """Append a directive so the model knows the extra reference is pose only."""
    if not prompt:
        return _POSE_PROMPT_SUFFIX.strip()
    if prompt.rstrip().endswith(_POSE_PROMPT_SUFFIX.strip()):
        return prompt
    return prompt.rstrip() + _POSE_PROMPT_SUFFIX


__all__ = [
    "PoseMode",
    "Detector",
    "build_preservation_inputs",
    "extract_preservation_reference",
    "augment_prompt_for_preservation",
]
