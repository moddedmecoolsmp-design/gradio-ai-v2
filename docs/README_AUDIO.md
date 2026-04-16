# Audio Tools: Qwen3 TTS, Speaker Separation, and Transcription

This project now includes integrated Audio Tools in the Web UI.

## Features

### 1. Qwen3 TTS (Text-to-Speech)
Generate high-quality speech using Alibaba's Qwen3-TTS models.
- **Languages**: Auto-detect (default), English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.
- **Preset Voice**: Choose from pre-defined voices.
- **Voice Design**: Describe the voice you want (e.g., "A deep, raspy voice").
- **Voice Cloning**: Upload a 3-10s reference audio clip.
    - *Hybrid Mode*: Add instructions (e.g., "Speak excitedly") to modify the cloned voice.
    - *Zero-Shot*: If you don't provide reference text, it uses pure X-Vector cloning.

**Auto Model Download & Offline Paths**:
- Models are automatically downloaded from Hugging Face on first use.
- Optional overrides for offline/local installs:
    - `QWEN_TTS_CACHE_DIR`
    - `QWEN_TTS_CUSTOM_MODEL_PATH` / `QWEN_TTS_CUSTOM_MODEL_ID`
    - `QWEN_TTS_DESIGN_MODEL_PATH` / `QWEN_TTS_DESIGN_MODEL_ID`
    - `QWEN_TTS_BASE_MODEL_PATH` / `QWEN_TTS_BASE_MODEL_ID`

**Optimization**:
- Runs in `float16` precision on NVIDIA GPUs (RTX 3070+) for maximum speed.

### 2. Speaker Separation (Diarization)
Upload an audio or video file (`.mp3`, `.mp4`, `.wav`) to automatically detect and separate different speakers.
- **Engine**: PyAnnote Audio 3.1 (State-of-the-art).
- **Output**: Generates separate `.mp3` files for each speaker in `output/separated/`.

### 3. Audio Transcription (Whisper)
Transcribe audio/video with multilingual support and automatic language detection (best for Swedish + English code-switching).

- **Engine**: `faster-whisper` (CTranslate2 optimized).
- **Model sizes**:
  - `large-v3`: ~5GB VRAM, best multilingual accuracy.
  - `medium`: ~2GB VRAM, strong balance of speed/quality.
  - `small` / `base`: <1GB VRAM, fastest but less accurate.
- **Output formats**: Text, SRT subtitles, or JSON with timestamps.
- **Auto-detect**: Recommended for mixed-language audio.

**First-time download**: `large-v3` is ~3GB and will download on first use.

## Prerequisites

1.  **FFmpeg**: Required for audio processing.
    *   Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
    *   Extract and add the `bin` folder to your Windows System PATH.

2.  **Hugging Face Token** (For Speaker Separation only):
    *   The separation model is gated. You must accept terms at:
        *   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
        *   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
    *   Enter your token in the Web UI when prompted.

## Usage

1.  Run `Launch.bat`.
2.  Go to the **Audio Tools** tab.
3.  Select **Qwen3 TTS**, **Speaker Separation**, or **Audio Transcription**.

## Environment Variables

- `WHISPER_CACHE_DIR`: Custom model cache location.
- `WHISPER_DEFAULT_MODEL`: Default model size (e.g., `large-v3`).
- `WHISPER_UNLOAD_AFTER_TRANSCRIBE`: Set to `1` to unload the model after each transcription.
- `WHISPER_DEVICE`: Force device (`cuda` or `cpu`).
