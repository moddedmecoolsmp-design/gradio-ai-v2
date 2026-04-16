"""Application constants and configuration."""

import os

# State migration key
FAST_FLUX_STATE_MIGRATION_KEY = "fast_flux_default_migrated_v1"
CHARACTER_MANAGER_STATE_FILENAME = "character_manager.json"
LEGACY_CHARACTER_MANAGER_STATE_FILENAME = "character_manager.pkl"

# Gradio settings
GRADIO_DELETE_CACHE = 7200  # 2 hours in seconds

# Klein Anatomy LoRA
KLEIN_ANATOMY_LORA_URL = "https://civitai.com/api/download/models/2324991"

# Pose modes
POSE_MODES = ["Body Only", "Body + Face", "Body + Face + Hands"]
POSE_DETECTOR_TYPES = ["dwpose", "openpose"]

# Note: MODEL_CHOICES, SINGLE_RESOLUTION_PRESETS, BATCH_RESOLUTION_PRESETS, and ANIME_PHOTO_PRESETS
# are defined in app.py due to dependencies on runtime_policies module. They cannot be moved
# to this file without creating circular import issues.
