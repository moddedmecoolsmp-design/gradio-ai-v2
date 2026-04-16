"""Video processing utilities for frame-by-frame video generation."""

import os
import shutil
import subprocess
from typing import Optional, Tuple, Any
from PIL import Image


def safe_remove_path(path):
    """Safely remove a file or directory."""
    if not path or not os.path.exists(path):
        return
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except Exception:
        pass


def count_processed_video_frames(processed_dir: str) -> int:
    """Count the number of processed video frames in the output directory."""
    if not processed_dir or not os.path.isdir(processed_dir):
        return 0
    return len([name for name in os.listdir(processed_dir) if name.endswith("_out.png")])


def prepare_video_workdirs(video_output_path: str) -> dict:
    """Prepare working directories for video processing."""
    output_dir = os.path.dirname(os.path.abspath(video_output_path)) or "."
    video_stem = os.path.splitext(os.path.basename(video_output_path))[0]
    work_root = os.path.join(output_dir, f".{video_stem}_video_work")
    raw_frames_dir = os.path.join(work_root, "raw_frames")
    preview_output_path = os.path.join(work_root, "preview.mp4")
    processed_dir = os.path.join(output_dir, f"{video_stem}_processed_frames")

    safe_remove_path(work_root)
    safe_remove_path(processed_dir)

    os.makedirs(raw_frames_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    if os.name == "nt":
        for path in (work_root, raw_frames_dir):
            try:
                subprocess.run(["attrib", "+h", path], check=False, capture_output=True)
            except Exception:
                pass

    return {
        "output_dir": output_dir,
        "video_stem": video_stem,
        "work_root": work_root,
        "raw_frames_dir": raw_frames_dir,
        "processed_dir": processed_dir,
        "preview_output_path": preview_output_path,
    }


def build_video_status(
    stage: str,
    processed_count: int,
    total_frames: int,
    detail: str,
    processed_dir: Optional[str] = None,
    final_video_path: Optional[str] = None,
    stop_requested: bool = False,
) -> str:
    """Build a status message for video processing progress."""
    lines = [f"Stage: {stage}"]
    if total_frames > 0:
        lines.append(f"Processed frames: {processed_count}/{total_frames}")
    else:
        lines.append(f"Processed frames: {processed_count}")

    if stop_requested:
        lines.append("Stop requested: current batch will finish, then a partial video will be assembled.")

    if detail:
        lines.append(detail)
    if processed_dir:
        lines.append(f"Processed frames folder: {processed_dir}")
    if final_video_path:
        lines.append(f"Final video path: {final_video_path}")
    return "\n".join(lines)


def save_processed_video_frame(image: Image.Image, source_frame_path: str, processed_dir: str) -> str:
    """Save a processed video frame to the output directory."""
    base_name = os.path.splitext(os.path.basename(source_frame_path))[0]
    output_path = os.path.join(processed_dir, f"{base_name}_out.png")
    image.convert("RGB").save(output_path)
    return output_path


def enhance_video_prompt_for_gender(
    prompt: str,
    negative_prompt: str,
    source_image: Image.Image,
    device: str,
    enable_gender_preservation: bool,
    gender_strength: float,
) -> Tuple[str, str]:
    """Enhance video prompt with gender preservation."""
    if not enable_gender_preservation:
        return prompt, negative_prompt

    try:
        from src.core.gender_helper import (
            enhance_prompt_with_gender,
            get_cached_face_app,
            get_gender_details,
            get_gender_negative_prompt,
            merge_negative_prompts,
        )

        face_app = get_cached_face_app(device=device)
        gender_info = get_gender_details(source_image, face_app)
        if gender_info["total_faces"] <= 0:
            return prompt, negative_prompt

        prompt = enhance_prompt_with_gender(prompt, gender_info, strength=gender_strength)
        gender_neg = get_gender_negative_prompt(gender_info, strength=gender_strength * 1.3)
        negative_prompt = merge_negative_prompts(negative_prompt, gender_neg)

        return prompt, negative_prompt
    except Exception as e:
        print(f"  Warning: Gender preservation failed for video frame: {e}")
        return prompt, negative_prompt


def resolve_video_tool_path(tool_name: str) -> Optional[str]:
    """Resolve the path to a video tool (ffmpeg or ffprobe)."""
    env_name = f"{tool_name.upper()}_BINARY"
    explicit_path = os.environ.get(env_name)
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    discovered_path = shutil.which(tool_name)
    if discovered_path:
        return discovered_path

    ffmpeg_env_path = os.environ.get("FFMPEG_BINARY") or os.environ.get("IMAGEIO_FFMPEG_EXE")
    if tool_name == "ffmpeg" and ffmpeg_env_path and os.path.exists(ffmpeg_env_path):
        return ffmpeg_env_path

    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_path = None

    if not ffmpeg_path or not os.path.exists(ffmpeg_path):
        return None

    if tool_name == "ffmpeg":
        return ffmpeg_path

    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    tool_filename = f"{tool_name}.exe" if os.name == "nt" else tool_name
    sibling_path = os.path.join(ffmpeg_dir, tool_filename)
    if os.path.exists(sibling_path):
        return sibling_path

    return None
