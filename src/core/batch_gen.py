import os
import time
import shutil
import subprocess
import tempfile
import cv2
import torch
from typing import List, Optional, Callable, Any, Generator

from src.constants import CHARACTER_MANAGER_STATE_FILENAME
from src.core.async_batch_integration import run_async_batch_processing
from src.runtime_policies import should_enable_autocast
from src.utils.device_utils import get_device_vram_gb

class BatchGenerator:
    """Core batch and video processing engine."""

    def __init__(self, pipeline_manager):
        self.pm = pipeline_manager
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def batch_process_folder(
        self,
        prompt,
        negative_prompt,
        input_folder,
        output_folder,
        batch_resolution_preset,
        downscale_factor,
        height,
        width,
        steps,
        seed,
        guidance,
        device,
        model_choice,
        lora_file,
        lora_strength,
        enable_multi_character,
        character_description,
        enable_faceswap,
        faceswap_source_image,
        faceswap_target_index,
        optimization_profile,
        enable_windows_compile_probe,
        enable_cuda_graphs,
        enable_optional_accelerators,
        enable_pose_preservation,
        pose_detector_type,
        pose_mode,
        controlnet_strength,
        enable_gender_preservation,
        gender_strength,
        enable_prompt_upsampling,
        enable_klein_anatomy_fix,
        character_references: Optional[List[Any]] = None,
        progress_callback: Optional[Callable] = None,
    ):
        self.stop_requested = False
        self.pm.configure_optimization_policy(
            device=device,
            profile=optimization_profile,
            enable_windows_compile_probe=enable_windows_compile_probe,
            enable_optional_accelerators=enable_optional_accelerators,
        )

        # 1. Validation
        if not input_folder or not os.path.isdir(input_folder):
            return "Input folder not found."
        if not output_folder:
            return "Output folder is required."
        os.makedirs(output_folder, exist_ok=True)

        # 2. Gather Images
        abs_input = os.path.abspath(input_folder)
        abs_output = os.path.abspath(output_folder)
        image_paths = []
        for root, dirs, files in os.walk(input_folder):
            # Prune output folder
            i = len(dirs) - 1
            while i >= 0:
                abs_dir = os.path.abspath(os.path.join(root, dirs[i]))
                if abs_dir == abs_output or abs_dir.startswith(abs_output + os.sep):
                    dirs.pop(i)
                i -= 1

            if os.path.abspath(root).startswith(abs_output):
                continue

            for name in sorted(files):
                if name.lower().endswith("_out.png"): continue
                if os.path.splitext(name)[1].lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                    image_paths.append(os.path.join(root, name))

        if not image_paths:
            return "No images found in the input folder."

        # 3. Model & Pipeline Preparation
        try:
            self.pm.ensure_models_downloaded(
                model_choice,
                enable_multi_character=enable_multi_character,
                enable_faceswap=enable_faceswap,
                enable_pose_preservation=enable_pose_preservation,
                enable_klein_anatomy_fix=enable_klein_anatomy_fix,
                progress=progress_callback
            )
        except Exception as e:
            return f"Model download failed: {e}"

        pipe = self.pm.load_pipeline(model_choice, device)
        current_model = self.pm.current_model

        # 4. Multi-Character PuLID Setup
        character_embeddings = []
        if enable_multi_character and character_references:
            try:
                from src.image.pulid_helper import MultiCharacterManager
                target_dim = 3072
                if "klein-4B" in current_model:
                    target_dim = 7680

                manager = MultiCharacterManager(device=device)
                char_state_path = os.path.join(self.pm.state_dir, CHARACTER_MANAGER_STATE_FILENAME)
                if os.path.exists(char_state_path):
                    manager.load_state(char_state_path)
                    for i, ref_img in enumerate(character_references):
                        if ref_img is not None and i < len(manager.characters):
                            manager.assign_reference_image(manager.characters[i]['character_id'], ref_img)
                    character_embeddings = manager.get_embeddings_for_generation(target_dim=target_dim)
            except Exception as e:
                print(f"  Warning: Multi-character PuLID setup failed: {e}")

        # 5. Pose & Helpers
        cn_pipe_instance = None
        extractor_instance = None
        current_extraction_mode = "body_face"
        if enable_pose_preservation:
            try:
                from src.image.pose_helper import get_pose_extractor
                from diffusers import FluxControlNetPipeline
                cn = self.pm.load_controlnet_union(device)
                if cn:
                    cn_pipe_instance = FluxControlNetPipeline(
                        scheduler=pipe.scheduler, vae=pipe.vae,
                        text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer,
                        text_encoder_2=getattr(pipe, 'text_encoder_2', None),
                        tokenizer_2=getattr(pipe, 'tokenizer_2', None),
                        transformer=pipe.transformer, controlnet=cn
                    )
                    extractor_instance = get_pose_extractor(device=device, detector_type=pose_detector_type)
                    mode_map = {"Body Only": "body", "Body + Face": "body_face", "Body + Face + Hands": "body_face_hands"}
                    current_extraction_mode = mode_map.get(pose_mode, "body_face")
            except Exception as e:
                print(f"  Warning: Batch pose setup failed: {e}")

        # 6. Run Batch Processing
        def progress_wrapper(p, desc):
            if progress_callback: progress_callback(p, desc=desc)
            if self.stop_requested: raise RuntimeError("Cancelled by user")

        try:
            stats = run_async_batch_processing(
                image_paths=image_paths,
                pipe=pipe,
                device=device,
                output_folder=output_folder,
                progress_callback=progress_wrapper,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance=guidance,
                seed=seed,
                img2img_strength=1.0,
                input_folder=input_folder,
                preset=batch_resolution_preset or "~1024px",
                downscale_factor=downscale_factor,
                enable_pose_preservation=(enable_pose_preservation and cn_pipe_instance and extractor_instance),
                cn_pipe=cn_pipe_instance,
                extractor=extractor_instance,
                controlnet_strength=controlnet_strength,
                extraction_mode=current_extraction_mode,
                pose_detector_type=pose_detector_type,
                enable_faceswap=enable_faceswap,
                faceswap_source_image=faceswap_source_image,
                faceswap_target_index=faceswap_target_index,
                enable_gender_preservation=enable_gender_preservation,
                gender_strength=gender_strength,
                enable_prompt_upsampling=enable_prompt_upsampling,
                character_embeddings=character_embeddings,
                enable_multi_character=enable_multi_character,
                optimization_profile=getattr(self.pm, "optimization_profile", "balanced"),
                autocast_ctx=(
                    torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if should_enable_autocast(device, current_model, pipe)
                    else None
                )
            )
            return f"Processed {stats.processed_images}/{stats.total_images} images in {stats.total_time:.1f}s."
        except Exception as e:
            return f"Batch processing failed: {e}"

    def process_video(
        self,
        prompt,
        negative_prompt,
        video_input,
        preserve_audio,
        video_output_path,
        video_resolution_preset,
        img2img_strength,
        height,
        width,
        steps,
        seed,
        guidance,
        device,
        model_choice,
        lora_file,
        lora_strength,
        enable_multi_character,
        character_description,
        enable_faceswap,
        faceswap_source_image,
        faceswap_target_index,
        optimization_profile,
        enable_windows_compile_probe,
        enable_cuda_graphs,
        enable_optional_accelerators,
        enable_pose_preservation,
        pose_detector_type,
        pose_mode,
        controlnet_strength,
        enable_gender_preservation,
        gender_strength,
        enable_prompt_upsampling,
        enable_klein_anatomy_fix,
        character_references: Optional[List[Any]] = None,
        progress_callback: Optional[Callable] = None,
    ):
        self.stop_requested = False
        self.pm.configure_optimization_policy(
            device=device,
            profile=optimization_profile,
            enable_windows_compile_probe=enable_windows_compile_probe,
            enable_optional_accelerators=enable_optional_accelerators,
        )
        if not video_input:
            yield "No video input provided.", None
            return

        temp_output_dir = tempfile.mkdtemp()
        if not video_output_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_output_path = os.path.join(self.pm.base_dir, "output", "videos", f"output_{timestamp}.mp4")

        output_dir = os.path.dirname(os.path.abspath(video_output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp()
        frames_dir = os.path.join(temp_dir, "frames")
        processed_dir = os.path.join(temp_dir, "processed_frames")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        try:
            # 1. Extract Frames
            if progress_callback: progress_callback(0, desc="Extracting frames...")
            cap = cv2.VideoCapture(video_input)
            if not cap.isOpened():
                yield "Failed to open video file.", None
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

            result = subprocess.run(["ffmpeg", "-y", "-i", video_input, "-vsync", "0", os.path.join(frames_dir, "frame_%05d.png")], capture_output=True)
            if result.returncode != 0:
                error_msg = result.stderr.decode() if result.stderr else "Unknown ffmpeg error"
                yield f"FFmpeg frame extraction failed: {error_msg}", None
                return
            extracted_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")])
            if not extracted_paths:
                yield "Failed to extract frames.", None
                return

            total_frames = len(extracted_paths)

            # 2. Model Prep
            self.pm.ensure_models_downloaded(model_choice, enable_multi_character=enable_multi_character, enable_faceswap=enable_faceswap, enable_pose_preservation=enable_pose_preservation)
            pipe = self.pm.load_pipeline(model_choice, device)

            # 3. Batch Processing
            def video_progress_wrapper(p, desc):
                if progress_callback: progress_callback(0.1 + p * 0.8, desc=f"Video: {desc}")
                if self.stop_requested: raise RuntimeError("Cancelled")

            stats = run_async_batch_processing(
                image_paths=extracted_paths, pipe=pipe, device=device, output_folder=processed_dir,
                progress_callback=video_progress_wrapper,
                prompt=prompt, negative_prompt=negative_prompt, steps=steps, guidance=guidance, seed=seed,
                img2img_strength=img2img_strength,
                input_folder=frames_dir, preset=video_resolution_preset, downscale_factor="1x",
                enable_pose_preservation=enable_pose_preservation, enable_faceswap=enable_faceswap,
                enable_gender_preservation=enable_gender_preservation,
                enable_prompt_upsampling=enable_prompt_upsampling,
                enable_multi_character=enable_multi_character
            )

            # 4. Final Assemble
            if progress_callback: progress_callback(0.9, desc="Assembling video...")
            success, msg, _ = self.create_video_from_frames(processed_dir, video_output_path, fps, audio_source=video_input if preserve_audio else None, temp_dir=temp_dir)

            yield f"Processed {stats.processed_images} frames. {msg}", video_output_path if success else None

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(temp_output_dir, ignore_errors=True)

    def create_video_from_frames(self, frames_dir, output_path, fps, audio_source=None, temp_dir=None):
        processed_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith("_out.png")])
        if not processed_frames: return False, "No frames", 0

        pattern = os.path.join(frames_dir, "frame_%05d_out.png")
        duration = len(processed_frames) / fps

        try:
            if audio_source and os.path.exists(audio_source):
                temp_video = os.path.join(temp_dir, "temp_no_audio.mp4")
                result = subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-t", str(duration), temp_video], capture_output=True)
                if result.returncode != 0:
                    return False, f"FFmpeg video creation failed: {result.stderr.decode() if result.stderr else 'Unknown error'}", len(processed_frames)

                temp_audio = os.path.join(temp_dir, "temp_audio.aac")
                result = subprocess.run(["ffmpeg", "-y", "-i", audio_source, "-t", str(duration), "-c:a", "aac", "-b:a", "192k", temp_audio], capture_output=True)
                if result.returncode != 0:
                    return False, f"FFmpeg audio extraction failed: {result.stderr.decode() if result.stderr else 'Unknown error'}", len(processed_frames)

                result = subprocess.run(["ffmpeg", "-y", "-i", temp_video, "-i", temp_audio, "-c:v", "copy", "-c:a", "aac", "-shortest", output_path], capture_output=True)
                if result.returncode != 0:
                    return False, f"FFmpeg video/audio merge failed: {result.stderr.decode() if result.stderr else 'Unknown error'}", len(processed_frames)

                return True, "Video with audio created", len(processed_frames)
            else:
                result = subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-t", str(duration), output_path], capture_output=True)
                if result.returncode != 0:
                    return False, f"FFmpeg video creation failed: {result.stderr.decode() if result.stderr else 'Unknown error'}", len(processed_frames)
                return True, "Video created", len(processed_frames)
        except Exception as e:
            return False, f"FFmpeg error: {e}", len(processed_frames)
