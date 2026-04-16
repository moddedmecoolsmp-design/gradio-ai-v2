from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from src.runtime_policies import choose_image_generation_mode, is_flux_model


@dataclass(frozen=True)
class VideoTemporalConfig:
    enabled: bool = True
    flow_method: str = "farneback"
    keyframe_interval: int = 12
    scene_cut_hist_corr_threshold: float = 0.65
    temporal_strength: float = 0.30
    keyframe_strength: float = 0.55
    occlusion_blend: float = 0.75
    deflicker_size: int = 5


def build_video_temporal_config(user_img2img_strength: float) -> VideoTemporalConfig:
    strength = float(user_img2img_strength)
    strength = max(0.0, min(1.0, strength))
    return VideoTemporalConfig(keyframe_strength=min(strength, 0.55))


def resolve_video_shot_seed(seed: int, randint_fn: Optional[Callable[[], int]] = None) -> int:
    if int(seed) != -1:
        return int(seed)
    if randint_fn is None:
        raise ValueError("randint_fn is required when resolving a random video seed")
    return int(randint_fn())


def resolve_video_frame_seed(shot_seed: int, frame_index: int) -> int:
    _ = frame_index
    return int(shot_seed)


def resolve_video_generation_mode(current_model: Optional[str], enable_pose_preservation: bool = False) -> str:
    if is_flux_model(current_model) and enable_pose_preservation:
        return "flux-pose"

    mode = choose_image_generation_mode(
        current_model=current_model,
        has_input_images=True,
        enable_pose_preservation=False,
    )
    if mode == "txt2img":
        raise ValueError(
            f"Video processing requires image-conditioned generation; model '{current_model}' does not provide it."
        )
    return mode


def _to_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3:
        raise ValueError("Expected an RGB image array")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    return array.astype(np.uint8)


def compute_hsv_histogram_correlation(
    previous_frame: Image.Image | np.ndarray,
    current_frame: Image.Image | np.ndarray,
) -> float:
    prev_hsv = cv2.cvtColor(_to_rgb_array(previous_frame), cv2.COLOR_RGB2HSV)
    curr_hsv = cv2.cvtColor(_to_rgb_array(current_frame), cv2.COLOR_RGB2HSV)

    prev_hist = cv2.calcHist([prev_hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256])
    curr_hist = cv2.calcHist([curr_hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256])
    prev_hist = cv2.normalize(prev_hist, None).astype(np.float32)
    curr_hist = cv2.normalize(curr_hist, None).astype(np.float32)

    return float(cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL))


def should_reset_temporal_history(
    frame_index: int,
    previous_frame: Image.Image | np.ndarray,
    current_frame: Image.Image | np.ndarray,
    config: VideoTemporalConfig,
) -> tuple[bool, str, float]:
    correlation = compute_hsv_histogram_correlation(previous_frame, current_frame)
    if frame_index > 0 and config.keyframe_interval > 0 and frame_index % config.keyframe_interval == 0:
        return True, f"temporal reset on keyframe refresh ({frame_index})", correlation
    if correlation < float(config.scene_cut_hist_corr_threshold):
        return True, f"temporal reset on scene cut ({correlation:.3f})", correlation
    return False, "", correlation


def compute_dense_optical_flows(
    previous_frame: Image.Image | np.ndarray,
    current_frame: Image.Image | np.ndarray,
    flow_method: str = "farneback",
) -> tuple[np.ndarray, np.ndarray]:
    method = str(flow_method).lower()
    if method != "farneback":
        raise ValueError(f"Unsupported flow method: {flow_method}")

    prev_gray = cv2.cvtColor(_to_rgb_array(previous_frame), cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(_to_rgb_array(current_frame), cv2.COLOR_RGB2GRAY)

    params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    forward_flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **params)
    backward_flow = cv2.calcOpticalFlowFarneback(curr_gray, prev_gray, None, **params)
    return forward_flow, backward_flow


def warp_image_with_flow(previous_image: Image.Image | np.ndarray, backward_flow: np.ndarray) -> np.ndarray:
    previous_rgb = _to_rgb_array(previous_image)
    height, width = backward_flow.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x + backward_flow[:, :, 0].astype(np.float32)
    map_y = grid_y + backward_flow[:, :, 1].astype(np.float32)
    return cv2.remap(
        previous_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def build_occlusion_mask(
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    fb_threshold: float = 1.5,
) -> np.ndarray:
    height, width = backward_flow.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    sample_x = grid_x + backward_flow[:, :, 0].astype(np.float32)
    sample_y = grid_y + backward_flow[:, :, 1].astype(np.float32)

    in_bounds = (
        (sample_x >= 0.0)
        & (sample_x <= float(width - 1))
        & (sample_y >= 0.0)
        & (sample_y <= float(height - 1))
    )

    sampled_forward_x = cv2.remap(
        forward_flow[:, :, 0].astype(np.float32),
        sample_x,
        sample_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    sampled_forward_y = cv2.remap(
        forward_flow[:, :, 1].astype(np.float32),
        sample_x,
        sample_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    fb_error = np.sqrt(
        np.square(sampled_forward_x + backward_flow[:, :, 0].astype(np.float32))
        + np.square(sampled_forward_y + backward_flow[:, :, 1].astype(np.float32))
    )
    return np.where(in_bounds & (fb_error <= float(fb_threshold)), 1.0, 0.0).astype(np.float32)


def prepare_temporal_condition_frame(
    previous_raw_frame: Image.Image | np.ndarray,
    current_raw_frame: Image.Image | np.ndarray,
    previous_stylized_frame: Image.Image | np.ndarray,
    config: VideoTemporalConfig,
) -> dict:
    forward_flow, backward_flow = compute_dense_optical_flows(
        previous_raw_frame,
        current_raw_frame,
        flow_method=config.flow_method,
    )
    warped_previous = warp_image_with_flow(previous_stylized_frame, backward_flow)
    confidence_mask = build_occlusion_mask(forward_flow, backward_flow)

    current_rgb = _to_rgb_array(current_raw_frame).astype(np.float32)
    warped_rgb = warped_previous.astype(np.float32)
    stylized_weight = (
        confidence_mask * float(config.temporal_strength)
        + (1.0 - confidence_mask) * (1.0 - float(config.occlusion_blend))
    )
    blended = current_rgb * (1.0 - stylized_weight[:, :, None]) + warped_rgb * stylized_weight[:, :, None]
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    return {
        "condition_image": Image.fromarray(blended, mode="RGB"),
        "warped_previous": Image.fromarray(warped_previous, mode="RGB"),
        "confidence_mask": confidence_mask,
        "confidence_ratio": float(confidence_mask.mean()),
        "forward_flow": forward_flow,
        "backward_flow": backward_flow,
    }


def build_deflicker_filter(deflicker_size: int) -> str:
    return f"deflicker=size={int(deflicker_size)}:mode=am"


def build_ffmpeg_frame_encode_command(
    ffmpeg_exe: Optional[str],
    fps: float,
    pattern: str,
    duration: float,
    output_path: str,
) -> list[str]:
    return [
        ffmpeg_exe or "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-t",
        str(duration),
        output_path,
    ]


def build_ffmpeg_deflicker_command(
    ffmpeg_exe: Optional[str],
    input_path: str,
    output_path: str,
    deflicker_size: int,
) -> list[str]:
    return [
        ffmpeg_exe or "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        build_deflicker_filter(deflicker_size),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        output_path,
    ]


def build_ffmpeg_audio_trim_command(
    ffmpeg_exe: Optional[str],
    audio_source: str,
    duration: float,
    output_path: str,
) -> list[str]:
    return [
        ffmpeg_exe or "ffmpeg",
        "-y",
        "-i",
        audio_source,
        "-t",
        str(duration),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        output_path,
    ]


def build_ffmpeg_merge_command(
    ffmpeg_exe: Optional[str],
    video_path: str,
    audio_path: str,
    output_path: str,
) -> list[str]:
    return [
        ffmpeg_exe or "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        output_path,
    ]
