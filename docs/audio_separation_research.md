# Audio Separation and Diarization Research (2026)

## Summary
For local GPU-accelerated voice separation (splitting an audio file into separate files per speaker), the following libraries are the top choices.

### 1. WhisperX (Recommended for Accuracy)
- **Tech Stack**: Faster-Whisper + Pyannote 3.1 + VAD.
- **Pros**:
  - Extremely accurate diarization (who spoke when).
  - Handles alignment for perfect timestamps.
  - Native GPU support (CUDA).
- **Cons**: Requires a Hugging Face token for the diarization model (Pyannote).
- **Workflow**: MP4/MP3 -> WhisperX -> Timestamps -> Pydub -> Separate MP3s.

### 2. Pyannote.Audio 3.1 (Direct Diarization)
- **Tech Stack**: PyTorch-based diarization.
- **Pros**: The industry standard for pure diarization. Very fast on GPU.
- **Cons**: Requires HF token and manual acceptance of model terms. No built-in audio splitting.

### 3. SpeechBrain (Speaker Separation)
- **Tech Stack**: Deep learning for actual voice *separation* (removing background/overlapping speech).
- **Pros**: Can actually isolate overlapping voices (Source Separation).
- **Cons**: Higher GPU memory requirements; more complex to setup for general-purpose files.

### 4. NVIDIA NeMo (Enterprise Grade)
- **Pros**: Most accurate diarization on NVIDIA hardware.
- **Cons**: Very heavy dependencies (requires Linux/WSL usually, difficult Windows native setup).

## Recommendation
For a robust Python implementation, **WhisperX** is the best balance of accuracy and ease of use for "splitting people's voices". If the user wants to *remove* overlapping voices (not just split by time), **SpeechBrain**'s separation models are required.

## Technical Implementation Details
1. **Audio Extraction**: Use `moviepy` or `ffmpeg` to ensure a consistent 16kHz mono WAV format for processing.
2. **GPU Acceleration**: Both `faster-whisper` (used by WhisperX) and `pyannote` use `torch` with CUDA.
3. **Splitting**: Use `pydub` for lossless slicing of the original audio into speaker-specific MP3s.

---
**Sources**:
- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [Pyannote.Audio Documentation](https://github.com/pyannote/pyannote-audio)
- [SpeechBrain Separation Models](https://huggingface.co/speechbrain/sepformer-libri2mix)
- [Faster-Whisper Project](https://github.com/SYSTRAN/faster-whisper)
