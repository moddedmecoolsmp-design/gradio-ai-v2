"""Extracted Gradio UI builder."""

from __future__ import annotations

from typing import Any, Mapping
from src.constants import BUILTIN_LORA_CHOICES
from src.image import faceswap_config, faceswap_ui

def on_builtin_lora_change(builtin_lora):
    """Toggle file upload visibility based on built-in LoRA selection."""
    if builtin_lora == "None (Custom File Upload)":
        return gr.update(visible=True), None  # Show file upload, clear lora_file
    else:
        return gr.update(visible=False), None  # Hide file upload, clear lora_file


def create_ui(context: Mapping[str, Any]):
    module_globals = globals()
    module_globals.update(context)
    with gr.Blocks(
        title="Ultra Fast Image Gen",
        delete_cache=(GRADIO_CACHE_CLEANUP_FREQUENCY_SECONDS, GRADIO_CACHE_TTL_SECONDS),
    ) as demo:
        gr.Markdown(
            """
            # Ultra Fast Image Gen

            Fast AI image generation and editing. Advanced controls are in the
            **Advanced** tab; model selection and LoRA in **Models & LoRA**.
            """
        )

        persisted_state = gr.State(initial_state)

        with gr.Tabs():
            # ==============================================================
            # Tab 1 — Generate
            # ==============================================================
            with gr.TabItem("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the image you want to generate...",
                            lines=3,
                            value=initial_state["prompt"],
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="What to avoid (e.g., blurry, low quality, extra limbs)",
                            lines=2,
                            value=initial_state["negative_prompt"],
                        )
                        preset_choice = gr.Dropdown(
                            choices=["None"] + list(ANIME_PHOTO_PRESETS.keys()),
                            value=initial_state["preset_choice"],
                            label="Preset",
                            info="Sets prompt, negative prompt, strength, and steps.",
                        )

                        main_tabs = gr.Tabs()
                        with main_tabs:
                            with gr.TabItem("Single / Image-to-Image"):
                                img2img_label = gr.Markdown(
                                    "Input Images (optional, up to 6 — enables image-to-image)",
                                    visible=True,
                                )
                                input_images = gr.Gallery(
                                    label="Input Images",
                                    type="pil",
                                    value=persisted_input_images,
                                    visible=True,
                                    columns=3,
                                    height="auto",
                                    interactive=True,
                                )
                                resolution_preset = gr.Radio(
                                    choices=SINGLE_RESOLUTION_PRESETS,
                                    value=initial_state["resolution_preset"],
                                    label="Output Resolution (longest side)",
                                    info="Maintains your image's aspect ratio.",
                                    visible=False,
                                )
                                single_downscale_preview = gr.Markdown(
                                    "Downscale applies to image-to-image and batch."
                                )

                            with gr.TabItem("Batch Folder") as batch_tab:
                                gr.Markdown(
                                    "Process every image in a folder (FLUX models only)."
                                )
                                with gr.Row():
                                    batch_input_folder = gr.Textbox(
                                        label="Input Folder",
                                        placeholder=r"C:\path\to\input_images",
                                        value=initial_state["batch_input_folder"],
                                    )
                                    batch_input_browse = gr.Button(
                                        "Browse", scale=0, min_width=100
                                    )
                                with gr.Row():
                                    batch_output_folder = gr.Textbox(
                                        label="Output Folder",
                                        placeholder=r"C:\path\to\output_images",
                                        value=initial_state["batch_output_folder"],
                                    )
                                    batch_output_browse = gr.Button(
                                        "Browse", scale=0, min_width=100
                                    )
                                batch_resolution_preset = gr.Radio(
                                    choices=BATCH_RESOLUTION_PRESETS,
                                    value=initial_state["batch_resolution_preset"],
                                    label="Batch Output Resolution (longest side)",
                                    info="Maintains each image's aspect ratio.",
                                )
                                batch_downscale_preview = gr.Markdown(
                                    "Select an input folder to preview downscale."
                                )
                                batch_run_btn = gr.Button("Run Batch", variant="secondary")
                                batch_summary = gr.Textbox(
                                    label="Batch Summary", interactive=False
                                )

                            with gr.TabItem("Video"):
                                gr.Markdown(
                                    "Process a video frame by frame, then reassemble to MP4."
                                )
                                video_input = gr.Video(
                                    label="Input Video", sources=["upload"]
                                )
                                with gr.Row():
                                    video_output_path = gr.Textbox(
                                        label="Output Video Path (optional)",
                                        placeholder="Leave empty for temp file, or specify a path like C:/output.mp4",
                                        value=initial_state.get("video_output_path", ""),
                                        info="If empty, video will be saved to a temporary location.",
                                    )
                                    video_output_browse = gr.Button(
                                        "Browse", scale=0, min_width=100
                                    )
                                video_resolution_preset = gr.Radio(
                                    choices=SINGLE_RESOLUTION_PRESETS,
                                    value=initial_state.get(
                                        "video_resolution_preset", "~1024px"
                                    ),
                                    label="Video Output Resolution (longest side)",
                                    info="Maintains aspect ratio. Higher resolutions require more VRAM.",
                                )
                                preserve_audio = gr.Checkbox(
                                    label="Preserve Audio",
                                    value=initial_state.get("preserve_audio", True),
                                    info="Merge audio from the source video into the output.",
                                )
                                with gr.Row():
                                    video_run_btn = gr.Button(
                                        "Run Video Processing", variant="primary"
                                    )
                                    video_stop_btn = gr.Button(
                                        "Stop Video Processing", variant="stop"
                                    )
                                video_output = gr.Video(
                                    label="Processed Video (Progressive Output)",
                                    interactive=False,
                                )
                                video_summary = gr.Textbox(
                                    label="Video Processing Status",
                                    interactive=False,
                                    lines=3,
                                )

                        with gr.Row():
                            with gr.Column(scale=4, min_width=100) as generate_btn_col:
                                generate_btn = gr.Button("Generate", variant="primary")
                            with gr.Column(scale=1, min_width=100):
                                stop_btn = gr.Button("Stop", variant="stop")
                        seed_info = gr.Textbox(
                            label="Generation Info", interactive=False
                        )

                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            label="Generated Image", type="pil", format="png"
                        )

                        with gr.Accordion(
                            "Face Swap (post-processing)", open=False
                        ) as face_swap_accordion:
                            gr.Markdown(
                                "Scan the generated image for faces, then assign a "
                                "source face to each detected face. Runs the "
                                "InsightFace **inswapper_128** model on CUDA when "
                                "available."
                            )
                            face_swap_enable = gr.Checkbox(
                                label="Enable Face Swap",
                                value=False,
                                info="When off, Apply is a no-op.",
                            )
                            with gr.Row():
                                face_scan_btn = gr.Button(
                                    "Scan Faces in Generated Image",
                                    variant="secondary",
                                )
                                face_apply_btn = gr.Button(
                                    "Apply Face Swap", variant="primary"
                                )
                            face_swap_status = gr.Textbox(
                                label="Status",
                                interactive=False,
                                lines=2,
                            )

                            # Pre-allocate per-face slots. Only the first N are
                            # made visible after a scan (where N = number of
                            # detected faces). Each slot has a thumbnail, a
                            # source-image uploader, a saved-character picker,
                            # and a "save detected face" shortcut.
                            face_slot_rows = []
                            face_slot_thumbnails = []
                            face_slot_source_images = []
                            face_slot_character_pickers = []
                            face_slot_save_names = []
                            face_slot_save_buttons = []
                            for slot_idx in range(faceswap_ui.MAX_FACE_SLOTS):
                                with gr.Row(visible=False) as slot_row:
                                    thumb = gr.Image(
                                        label=f"Detected Face #{slot_idx + 1}",
                                        type="pil",
                                        interactive=False,
                                        height=160,
                                    )
                                    source_img = gr.Image(
                                        label=f"Swap To (Source #{slot_idx + 1})",
                                        type="pil",
                                        sources=["upload", "clipboard"],
                                        height=160,
                                    )
                                    with gr.Column():
                                        char_pick = gr.Dropdown(
                                            choices=["(none)"],
                                            value="(none)",
                                            label="Saved Character",
                                            info=(
                                                "Use a saved character as the "
                                                "source instead of uploading."
                                            ),
                                        )
                                        save_name = gr.Textbox(
                                            label="Save Detected Face As",
                                            placeholder="Name (e.g. Alice)",
                                            lines=1,
                                        )
                                        save_btn = gr.Button(
                                            f"Save Face #{slot_idx + 1} to Library",
                                            size="sm",
                                        )
                                face_slot_rows.append(slot_row)
                                face_slot_thumbnails.append(thumb)
                                face_slot_source_images.append(source_img)
                                face_slot_character_pickers.append(char_pick)
                                face_slot_save_names.append(save_name)
                                face_slot_save_buttons.append(save_btn)

                            gr.Markdown(
                                "Manage saved characters in the **Face Swap** tab."
                            )

                        # State objects for the face-swap workflow. Detected
                        # faces are stashed here so the Apply handler knows the
                        # number and order of faces without re-running the
                        # detector.
                        face_detected_state = gr.State([])

                gr.Examples(
                    examples=[
                        ["A majestic mountain landscape at sunset, dramatic lighting, cinematic"],
                        ["Portrait of a young woman, soft studio lighting, professional photography"],
                        ["Cyberpunk city street at night, neon lights, rain reflections"],
                        ["A cute cat wearing a tiny hat, studio photo, soft lighting"],
                        ["Abstract art, vibrant colors, fluid shapes, modern design"],
                    ],
                    inputs=[prompt],
                    label="Prompt Examples",
                )
                gr.Examples(
                    examples=[
                        [
                            ANIME_PHOTO_PRESETS["Anime -> Photoreal (Balanced)"]["prompt"],
                            ANIME_PHOTO_PRESETS["Anime -> Photoreal (Balanced)"]["negative_prompt"],
                        ],
                    ],
                    inputs=[prompt, negative_prompt],
                    label="Anime to Photoreal Examples",
                )

            # ==============================================================
            # Tab 2 — Models & LoRA
            # ==============================================================
            with gr.TabItem("Models & LoRA"):
                gr.Markdown(
                    """
                    ### Model

                    - **FLUX.2-klein-4B (Int8)** — fastest FLUX on Windows + RTX 3070.
                      Supports image editing and LoRA.
                    - **FLUX.2-klein-4B (4bit SDNQ)** — low-VRAM FLUX fallback.
                      Supports image editing and LoRA.
                    - **Z-Image Turbo (Int8)** — separate Z-Image pipeline tuned for
                      RTX 3070-class VRAM and fast image editing.
                    """
                )
                logger.debug(
                    f"Creating device dropdown with choices={available_devices}, default={default_device}"
                )
                with gr.Row():
                    model_choice = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value=initial_state["model_choice"],
                        label="Model",
                        info="FLUX.2-klein and Z-Image support image editing.",
                    )
                    device = gr.Dropdown(
                        choices=available_devices,
                        value=initial_state["device"],
                        label="Device",
                        info="MPS for Mac, CUDA for NVIDIA, CPU is slow.",
                    )

                gr.Markdown("### LoRA")
                lora_label = gr.Markdown(
                    "Select a built-in LoRA, or upload a custom .safetensors file.",
                    visible=False,
                )
                builtin_lora = gr.Dropdown(
                    choices=[choice[0] for choice in BUILTIN_LORA_CHOICES],
                    value=initial_state.get("builtin_lora", "None (Custom File Upload)"),
                    label="Built-in LoRA",
                    info="Choose 'None' to upload your own.",
                    visible=False,
                )
                with gr.Row():
                    lora_file = gr.File(
                        label="LoRA File (Custom Upload)",
                        file_types=[".safetensors"],
                        file_count="single",
                        type="filepath",
                        value=initial_state["lora_file"],
                        visible=False,
                    )
                    clear_lora_btn = gr.Button(
                        "Clear LoRA", scale=0, min_width=100, visible=False
                    )
                lora_strength = gr.Slider(
                    0.0,
                    2.0,
                    value=initial_state["lora_strength"],
                    step=0.05,
                    label="LoRA Strength",
                    info="1.0 = full effect, 0.5 = half effect.",
                    visible=False,
                )

                gr.Markdown("### Model Quality")
                enable_klein_anatomy_fix = gr.Checkbox(
                    label="Klein Anatomy Quality Fixer",
                    value=initial_state.get("enable_klein_anatomy_fix", False),
                    info="Improves character anatomy for FLUX models.",
                )

            # ==============================================================
            # Tab 3 — Face Swap
            # ==============================================================
            with gr.TabItem("Face Swap"):
                gr.Markdown(
                    """
                    ### Face Swap & Character Library

                    Save a reference face once, and have it automatically swapped
                    onto every face it matches across Single, Batch, and Video
                    generations. Uses the InsightFace **inswapper_128** ONNX
                    model on CUDA when available.

                    The *per-image* scan (under the Generate tab's output)
                    lets you override any detected face manually, even without
                    a library entry.
                    """
                )
                with gr.Row():
                    faceswap_auto_enable = gr.Checkbox(
                        label="Auto-swap saved characters on every generation",
                        value=False,
                        info=(
                            "When on, every Single / Batch / Video output runs "
                            "through inswapper and any detected face whose "
                            "embedding matches a saved character is replaced."
                        ),
                    )
                    faceswap_similarity_threshold = gr.Slider(
                        minimum=0.15,
                        maximum=0.85,
                        value=0.35,
                        step=0.05,
                        label="Match Threshold",
                        info=(
                            "Cosine similarity cutoff. Lower = more aggressive "
                            "matches (may swap unrelated faces). Higher = "
                            "stricter (may miss the same character at "
                            "different angles)."
                        ),
                    )
                gr.Markdown("#### Saved Characters")
                character_gallery = gr.Gallery(
                    label="Saved Characters",
                    value=[],
                    columns=4,
                    height="auto",
                    interactive=False,
                )
                with gr.Row():
                    char_upload_name = gr.Textbox(
                        label="New Character Name",
                        placeholder="e.g. Alice",
                        lines=1,
                    )
                    char_upload_image = gr.Image(
                        label="Reference Photo",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=200,
                    )
                with gr.Row():
                    char_save_btn = gr.Button(
                        "Save New Character", variant="primary"
                    )
                    char_refresh_btn = gr.Button("Refresh Library")
                with gr.Row():
                    char_delete_name = gr.Textbox(
                        label="Delete Character By Name",
                        placeholder="Name to delete",
                        lines=1,
                    )
                    char_delete_btn = gr.Button("Delete", variant="stop")
                faceswap_tab_status = gr.Textbox(
                    label="Library Status", interactive=False
                )

            # ==============================================================
            # Tab 4 — Advanced
            # ==============================================================
            with gr.TabItem("Advanced"):
                gr.Markdown(
                    """
                    Expert controls. Defaults are tuned for the RTX 3070 fast path
                    and usually do not need to be changed.
                    """
                )

                with gr.Accordion("Image Settings", open=True):
                    downscale_factor = gr.Textbox(
                        label="Downscale Factor (e.g., 2x, 4x)",
                        value=initial_state["downscale_factor"],
                        info="Applies to image-to-image and batch. 1x = no downscale.",
                    )
                    img2img_strength = gr.Slider(
                        0.0,
                        1.0,
                        value=initial_state.get("img2img_strength", 0.6),
                        step=0.05,
                        label="Image Edit Strength (Z-Image)",
                        info="How much the source image changes. 0.3 = subtle, 0.8 = heavy rework.",
                    )
                    with gr.Row():
                        height = gr.Slider(
                            256, 2048, value=initial_state["height"], step=64, label="Height"
                        )
                        width = gr.Slider(
                            256, 2048, value=initial_state["width"], step=64, label="Width"
                        )
                    with gr.Row():
                        steps = gr.Slider(
                            1, 50, value=initial_state["steps"], step=1, label="Steps"
                        )
                        seed = gr.Number(value=initial_state["seed"], label="Seed (-1 = random)")
                    guidance_scale = gr.Slider(
                        0.0,
                        10.0,
                        value=initial_state["guidance_scale"],
                        step=0.5,
                        label="Guidance Scale (CFG)",
                        info="1.0 recommended. Higher values follow the prompt more strictly.",
                    )

                with gr.Accordion("Performance", open=False):
                    optimization_profile = gr.Dropdown(
                        choices=["max_speed", "balanced", "stability"],
                        value=initial_state.get(
                            "optimization_profile", DEFAULT_OPTIMIZATION_PROFILE
                        ),
                        label="Optimization Profile",
                        info="max_speed favors throughput; stability minimizes risky accelerators.",
                    )
                    enable_windows_compile_probe = gr.Checkbox(
                        label="Windows torch.compile probe",
                        value=initial_state.get(
                            "enable_windows_compile_probe", DEFAULT_WINDOWS_COMPILE_PROBE
                        ),
                        info="Probe compile once per model on CUDA 13; auto-disable on failure.",
                    )
                    enable_cuda_graphs = gr.Checkbox(
                        label="CUDA Graphs",
                        value=initial_state.get("enable_cuda_graphs", False),
                        info="Cache GPU execution graphs for 15–30% speedup on fixed resolutions (RTX 3070+). Uses extra VRAM.",
                    )
                    enable_optional_accelerators = gr.Checkbox(
                        label="Optional Accelerators",
                        value=initial_state.get(
                            "enable_optional_accelerators", DEFAULT_OPTIONAL_ACCELERATORS
                        ),
                        info="Allow optional acceleration paths (Pruna FORA, etc.) when compatible.",
                    )

            # ==============================================================
            # Tab 4 — Audio Tools
            # ==============================================================
            with gr.TabItem("Audio Tools"):
                gr.Markdown("Qwen3 TTS and speaker separation.")

                with gr.Tabs():
                    with gr.TabItem("Qwen3 TTS"):
                        with gr.Row():
                            with gr.Column():
                                tts_text = gr.Textbox(
                                    label="Text to Speak",
                                    lines=3,
                                    placeholder="Enter text here...",
                                )
                                tts_model = gr.State("Qwen TTS")
                                tts_mode = gr.Radio(
                                    choices=["Preset Voice", "Voice Design"],
                                    value="Preset Voice",
                                    label="Generation Mode",
                                )
                                tts_language = gr.Dropdown(
                                    choices=[
                                        "Auto",
                                        "Chinese",
                                        "English",
                                        "Japanese",
                                        "Korean",
                                        "German",
                                        "French",
                                        "Russian",
                                        "Portuguese",
                                        "Spanish",
                                        "Italian",
                                    ],
                                    value="Auto",
                                    label="Language",
                                )
                                tts_speaker = gr.Dropdown(
                                    choices=["Vivian", "Tech", "News"],
                                    value="Vivian",
                                    label="Speaker (Preset Mode)",
                                    visible=True,
                                )
                                tts_instruct = gr.Textbox(
                                    label="Voice Instructions (Style / Emotion)",
                                    placeholder="e.g. A young woman speaking excitedly",
                                    visible=False,
                                )
                                tts_gen_btn = gr.Button(
                                    "Generate Speech", variant="primary"
                                )
                            with gr.Column():
                                tts_output = gr.Audio(
                                    label="Generated Audio", type="filepath"
                                )
                                tts_status = gr.Textbox(
                                    label="Status", interactive=False
                                )

                    with gr.TabItem("Speaker Separation"):
                        gr.Markdown(
                            "Upload an audio or video file to split distinct "
                            "speakers into separate audio files."
                        )
                        with gr.Row():
                            with gr.Column():
                                sep_input = gr.Audio(
                                    label="Input Audio / Video", type="filepath"
                                )
                                sep_num_speakers = gr.Number(
                                    label="Number of Speakers (optional)",
                                    value=0,
                                    precision=0,
                                )
                                sep_hf_token = gr.Textbox(
                                    label="HuggingFace Token (required for PyAnnote)",
                                    type="password",
                                    placeholder="hf_...",
                                )
                                sep_btn = gr.Button(
                                    "Separate Speakers", variant="primary"
                                )
                            with gr.Column():
                                sep_output = gr.File(label="Separated Audio Files")
                                sep_status = gr.Textbox(
                                    label="Status", interactive=False
                                )

        # ==================================================================
        # Event handlers (unchanged wiring — only container layout moved)
        # ==================================================================
        main_tabs.select(
            fn=on_tab_select,
            inputs=[],
            outputs=[generate_btn_col],
        )

        model_choice.change(
            fn=update_ui_for_model,
            inputs=[model_choice, input_images, width, height, downscale_factor, device],
            outputs=[
                img2img_label,
                input_images,
                resolution_preset,
                img2img_strength,
                lora_label,
                builtin_lora,
                lora_file,
                lora_strength,
                clear_lora_btn,
                guidance_scale,
                width,
                height,
                single_downscale_preview,
                batch_resolution_preset,
                video_resolution_preset,
                batch_tab,
            ],
        )

        device.change(
            fn=update_ui_for_model,
            inputs=[model_choice, input_images, width, height, downscale_factor, device],
            outputs=[
                img2img_label,
                input_images,
                resolution_preset,
                img2img_strength,
                lora_label,
                builtin_lora,
                lora_file,
                lora_strength,
                clear_lora_btn,
                guidance_scale,
                width,
                height,
                single_downscale_preview,
                batch_resolution_preset,
                video_resolution_preset,
                batch_tab,
            ],
        )

        input_images.change(
            fn=on_image_upload,
            inputs=[input_images, resolution_preset, downscale_factor],
            outputs=[width, height, resolution_preset, single_downscale_preview],
        )

        resolution_preset.change(
            fn=on_resolution_preset_change,
            inputs=[resolution_preset, input_images, downscale_factor],
            outputs=[width, height, single_downscale_preview],
        )

        downscale_factor.change(
            fn=update_single_downscale_preview,
            inputs=[input_images, width, height, downscale_factor],
            outputs=[single_downscale_preview],
        )
        downscale_factor.change(
            fn=update_batch_downscale_preview,
            inputs=[batch_input_folder, batch_resolution_preset, downscale_factor],
            outputs=[batch_downscale_preview],
        )

        preset_choice.change(
            fn=apply_prompt_preset,
            inputs=[
                preset_choice,
                prompt,
                negative_prompt,
                img2img_strength,
                steps,
                lora_file,
                builtin_lora,
                lora_strength,
                guidance_scale,
            ],
            outputs=[
                prompt,
                negative_prompt,
                img2img_strength,
                steps,
                lora_file,
                builtin_lora,
                lora_strength,
                guidance_scale,
            ],
        )

        persistence_inputs = [
            model_choice,
            prompt,
            negative_prompt,
            preset_choice,
            input_images,
            resolution_preset,
            batch_input_folder,
            batch_output_folder,
            batch_resolution_preset,
            downscale_factor,
            img2img_strength,
            height,
            width,
            steps,
            seed,
            guidance_scale,
            enable_klein_anatomy_fix,
            device,
            lora_file,
            builtin_lora,
            lora_strength,
            optimization_profile,
            enable_windows_compile_probe,
            enable_cuda_graphs,
            enable_optional_accelerators,
            video_output_path,
            preserve_audio,
            video_resolution_preset,
        ]
        persistence_outputs = [persisted_state]

        for component in [
            model_choice,
            prompt,
            negative_prompt,
            preset_choice,
            input_images,
            resolution_preset,
            batch_input_folder,
            batch_output_folder,
            batch_resolution_preset,
            downscale_factor,
            img2img_strength,
            height,
            width,
            steps,
            seed,
            guidance_scale,
            device,
            lora_file,
            builtin_lora,
            lora_strength,
            optimization_profile,
            enable_windows_compile_probe,
            enable_optional_accelerators,
            enable_klein_anatomy_fix,
            video_output_path,
            preserve_audio,
        ]:
            component.change(
                fn=persist_ui_state,
                inputs=persistence_inputs,
                outputs=persistence_outputs,
            )

        batch_input_browse.click(
            fn=select_folder,
            inputs=[batch_input_folder],
            outputs=[batch_input_folder],
        )
        batch_input_folder.change(
            fn=update_batch_downscale_preview,
            inputs=[batch_input_folder, batch_resolution_preset, downscale_factor],
            outputs=[batch_downscale_preview],
        )
        batch_output_browse.click(
            fn=select_folder,
            inputs=[batch_output_folder],
            outputs=[batch_output_folder],
        )
        video_output_browse.click(
            fn=select_folder,
            inputs=[video_output_path],
            outputs=[video_output_path],
        )
        batch_resolution_preset.change(
            fn=update_batch_downscale_preview,
            inputs=[batch_input_folder, batch_resolution_preset, downscale_factor],
            outputs=[batch_downscale_preview],
        )

        stop_btn.click(
            fn=request_stop,
            outputs=[seed_info, batch_summary, video_summary],
        )

        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                negative_prompt,
                height,
                width,
                steps,
                seed,
                guidance_scale,
                device,
                model_choice,
                input_images,
                downscale_factor,
                img2img_strength,
                lora_file,
                builtin_lora,
                lora_strength,
                optimization_profile,
                enable_windows_compile_probe,
                enable_cuda_graphs,
                enable_optional_accelerators,
                preset_choice,
                enable_klein_anatomy_fix,
            ],
            outputs=[output_image, seed_info],
            show_progress=True,
        )
        generate_btn.click(
            fn=persist_ui_state,
            inputs=persistence_inputs,
            outputs=persistence_outputs,
        )

        batch_run_btn.click(
            fn=batch_process_folder,
            inputs=[
                prompt,
                negative_prompt,
                batch_input_folder,
                batch_output_folder,
                batch_resolution_preset,
                downscale_factor,
                height,
                width,
                steps,
                seed,
                guidance_scale,
                device,
                model_choice,
                lora_file,
                builtin_lora,
                lora_strength,
                optimization_profile,
                enable_windows_compile_probe,
                enable_cuda_graphs,
                enable_optional_accelerators,
                enable_klein_anatomy_fix,
                preset_choice,
            ],
            outputs=[batch_summary],
            show_progress=True,
        )
        batch_run_btn.click(
            fn=persist_ui_state,
            inputs=persistence_inputs,
            outputs=persistence_outputs,
        )

        clear_lora_btn.click(
            fn=clear_lora,
            outputs=[lora_file, seed_info],
        )
        lora_strength.change(
            fn=update_lora_strength,
            inputs=[lora_strength],
            outputs=[seed_info],
        )
        builtin_lora.change(
            fn=on_builtin_lora_change,
            inputs=[builtin_lora],
            outputs=[lora_file, lora_file],
        )

        video_run_btn.click(
            fn=process_video,
            inputs=[
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
                guidance_scale,
                device,
                model_choice,
                lora_file,
                builtin_lora,
                lora_strength,
                optimization_profile,
                enable_windows_compile_probe,
                enable_cuda_graphs,
                enable_optional_accelerators,
                enable_klein_anatomy_fix,
                preset_choice,
            ],
            outputs=[video_summary, video_output],
            show_progress=True,
        )
        video_run_btn.click(
            fn=persist_ui_state,
            inputs=persistence_inputs,
            outputs=persistence_outputs,
        )
        video_stop_btn.click(
            fn=request_stop,
            outputs=[seed_info, batch_summary, video_summary],
        )

        # Audio
        tts_mode.change(
            fn=audio_ui_helpers.update_qwen_tts_ui,
            inputs=[tts_mode],
            outputs=[tts_speaker, tts_instruct],
        )
        tts_gen_btn.click(
            fn=audio_ui_helpers.generate_tts,
            inputs=[
                tts_text,
                tts_model,
                tts_mode,
                tts_language,
                tts_speaker,
                tts_instruct,
            ],
            outputs=[tts_output, tts_status],
            show_progress=True,
        )
        sep_btn.click(
            fn=audio_ui_helpers.separate_audio,
            inputs=[sep_input, sep_num_speakers, sep_hf_token],
            outputs=[sep_output, sep_status],
            show_progress=True,
        )

        # ==============================================================
        # Face Swap (post-generation) wiring
        # ==============================================================
        # Flat list of (row_visible, thumbnail_value) pairs so the scan
        # handler can update every slot in one Gradio response.
        face_slot_flat_outputs: list = []
        for slot_idx in range(faceswap_ui.MAX_FACE_SLOTS):
            face_slot_flat_outputs.append(face_slot_rows[slot_idx])
            face_slot_flat_outputs.append(face_slot_thumbnails[slot_idx])

        face_scan_btn.click(
            fn=faceswap_ui.scan_output_for_faces,
            inputs=[output_image, device],
            outputs=face_slot_flat_outputs + [face_detected_state, face_swap_status],
        )

        # Flat list of (source_image, character_picker) pairs fed to
        # the apply handler in slot order.
        apply_slot_inputs: list = []
        for slot_idx in range(faceswap_ui.MAX_FACE_SLOTS):
            apply_slot_inputs.append(face_slot_source_images[slot_idx])
            apply_slot_inputs.append(face_slot_character_pickers[slot_idx])

        face_apply_btn.click(
            fn=faceswap_ui.apply_face_swap,
            inputs=(
                [
                    face_swap_enable,
                    output_image,
                    face_detected_state,
                    device,
                ]
                + apply_slot_inputs
            ),
            outputs=[output_image, face_swap_status],
            show_progress=True,
        )

        # "Save this detected face to the library" per-slot wiring.
        for slot_idx in range(faceswap_ui.MAX_FACE_SLOTS):
            face_slot_save_buttons[slot_idx].click(
                fn=lambda name, img, state, dev, idx=slot_idx: (
                    faceswap_ui.save_detected_face_as_character(
                        idx, name, img, state, dev
                    )
                ),
                inputs=[
                    face_slot_save_names[slot_idx],
                    output_image,
                    face_detected_state,
                    device,
                ],
                outputs=[face_swap_status],
            ).then(
                fn=faceswap_ui.refresh_character_library,
                inputs=[],
                outputs=[character_gallery]
                + face_slot_character_pickers,
            )

        # Auto-swap master toggle + threshold write into the singleton
        # config read by Single / Batch / Video post-processing.
        def _on_auto_swap_toggle(enabled):
            faceswap_config.set_config(auto_swap_from_library=bool(enabled))
            return (
                "Auto-swap ON — every generation will run through inswapper."
                if enabled
                else "Auto-swap OFF."
            )

        def _on_threshold_change(threshold):
            faceswap_config.set_config(similarity_threshold=float(threshold))
            return f"Threshold set to {float(threshold):.2f}."

        faceswap_auto_enable.change(
            fn=_on_auto_swap_toggle,
            inputs=[faceswap_auto_enable],
            outputs=[faceswap_tab_status],
        )
        faceswap_similarity_threshold.change(
            fn=_on_threshold_change,
            inputs=[faceswap_similarity_threshold],
            outputs=[faceswap_tab_status],
        )

        # Saved-character library controls (live in the Face Swap tab).
        char_save_btn.click(
            fn=faceswap_ui.save_upload_as_character,
            inputs=[char_upload_name, char_upload_image, device],
            outputs=[faceswap_tab_status],
        ).then(
            fn=faceswap_ui.refresh_character_library,
            inputs=[],
            outputs=[character_gallery] + face_slot_character_pickers,
        )

        char_refresh_btn.click(
            fn=faceswap_ui.refresh_character_library,
            inputs=[],
            outputs=[character_gallery] + face_slot_character_pickers,
        )

        char_delete_btn.click(
            fn=faceswap_ui.delete_character_handler,
            inputs=[char_delete_name],
            outputs=[faceswap_tab_status],
        ).then(
            fn=faceswap_ui.refresh_character_library,
            inputs=[],
            outputs=[character_gallery] + face_slot_character_pickers,
        )

        # Populate the saved-character gallery / dropdowns on app load so
        # the library survives process restarts.
        demo.load(
            fn=faceswap_ui.refresh_character_library,
            inputs=[],
            outputs=[character_gallery] + face_slot_character_pickers,
        )

    return demo
