"""Application constants and configuration."""

import os

# State migration key
FAST_FLUX_STATE_MIGRATION_KEY = "fast_flux_default_migrated_v1"
CHARACTER_MANAGER_STATE_FILENAME = "character_manager.json"
LEGACY_CHARACTER_MANAGER_STATE_FILENAME = "character_manager.pkl"

# Gradio settings
GRADIO_DELETE_CACHE = 7200  # 2 hours in seconds

# Klein Anatomy Quality Fixer LoRA (FLUX.2-klein 4B build).
# IMPORTANT: the Civitai download URL takes a *model version* ID, not the
# model ID.  2324991 was the bare model ID and resolved to the 9B build,
# which silently failed to match on 4B users' pipelines.  2617474 is the
# 4B version ID (see https://civitai.com/models/2324991/klein-anatomy-
# quality-fixer?modelVersionId=2617474).
KLEIN_ANATOMY_LORA_URL = "https://civitai.com/api/download/models/2617474"

# Klein High-Resolution LoRA (v2739957, 88 MB) — used as the quality-first
# upscaler path.  Trigger word "High resolution", strength 0.9.  ESRGAN
# remains the default upscaler for speed; this LoRA runs a full FLUX.2
# img2img pass so it's ~5x slower but preserves content pixel-for-pixel.
KLEIN_HIRES_LORA_URL = "https://civitai.com/api/download/models/2739957"
KLEIN_HIRES_LORA_TRIGGER = "High resolution"
KLEIN_HIRES_LORA_STRENGTH = 0.9

# Face Expression Transfer LoRA (v2658175, 88 MB) — quality complement to
# DWPose preservation.  Trained for illustration expression transfer via
# a dual-image trigger word, so it skips the DWPose extraction step
# entirely.  Strength ~1.0.
KLEIN_EXPRESSION_LORA_URL = "https://civitai.com/api/download/models/2658175"
KLEIN_EXPRESSION_LORA_TRIGGER = (
    "transfer character face expression in image1 "
    "with character face expression in image2"
)
KLEIN_EXPRESSION_LORA_STRENGTH = 1.0

# Built-in LoRAs for dropdown selection
BUILTIN_LORA_CHOICES = [
    ("None (Custom File Upload)", None),
    ("Klein Anatomy Fix", "klein_anatomy"),
    ("Klein High-Resolution (upscale LoRA)", "klein_hires"),
    ("Klein Face Expression Transfer", "klein_expression"),
    ("Realistic Snapshot v5 (Z-Image)", "zimage_realistic"),
    ("Ultra Real Amateur Selfies (FLUX 4B)", "flux_anime2real"),
]

# Pose modes
POSE_MODES = ["Body Only", "Body + Face", "Body + Face + Hands"]
POSE_DETECTOR_TYPES = ["dwpose", "openpose"]

# Note: MODEL_CHOICES, SINGLE_RESOLUTION_PRESETS, BATCH_RESOLUTION_PRESETS, and ANIME_PHOTO_PRESETS
# are defined in app.py due to dependencies on runtime_policies module. They cannot be moved
# to this file without creating circular import issues.
