import os
import logging
from typing import List, Tuple
import gradio as gr

# Library modules must not call ``logging.basicConfig``; see the note in
# ``src/audio/audio_separator.py``. Entry point owns root logger config.
logger = logging.getLogger(__name__)

# Try imports with graceful fallback
try:
    from src.audio.qwen_tts_helper import qwen_handler
except ImportError:
    qwen_handler = None

try:
    from src.audio.audio_separator import separator
except ImportError:
    separator = None

try:
    from src.audio.whisper_transcribe_helper import whisper_transcriber
except ImportError:
    whisper_transcriber = None

def generate_tts(text, tts_model="Qwen TTS", mode=None, language=None, speaker=None, instruct=None,
                  progress=gr.Progress()):
    """
    Generate TTS using canonical UI inputs.

    Args:
        text: Text to synthesize
        tts_model: "Qwen TTS" only (voice cloning removed)
        mode: Generation mode (Preset/Design for Qwen)
        language: Language selection (Qwen only)
        speaker: Speaker selection (Qwen preset mode only)
        instruct: Voice instructions (Qwen design mode)
    """
    try:
        mode = mode or "Preset Voice"
        language = language or "Auto"
        speaker = speaker or "Vivian"
        instruct = instruct or ""

        if not text:
            return None, "Error: Please enter text to generate."

        import time
        timestamp = int(time.time())
        output_path = os.path.join("output", f"tts_{timestamp}.wav")
        os.makedirs("output", exist_ok=True)

        # Only Qwen TTS is supported (voice cloning removed)
        if tts_model == "Qwen TTS":
            if not qwen_handler:
                return None, "Error: Qwen TTS not installed. Run: pip install qwen-tts"

            progress(0.1, desc="Initializing Qwen TTS...")

            # Map UI mode to internal mode
            internal_mode = "custom"
            if mode == "Voice Design":
                internal_mode = "design"
            elif mode == "Voice Cloning":
                return None, "Error: Voice Cloning mode has been removed."

            progress(0.2, desc="Loading Qwen3 Model (Download may take time on first run)...")

            final_path = qwen_handler.generate_speech(
                text=text,
                mode=internal_mode,
                language=language,
                speaker=speaker,
                instruct=instruct,
                ref_audio_path=None,
                ref_text=None,
                output_path=output_path
            )

            progress(0.9, desc="Finalizing audio...")
            return final_path, "Generation successful!"

        else:
            return None, f"Error: Unknown TTS model: {tts_model}"

    except Exception as e:
        logger.error(f"TTS Error: {e}")
        return None, f"Error: {str(e)}"

def separate_audio(audio_file, num_speakers, hf_token, progress=gr.Progress()):
    if not separator:
        return None, "Error: Audio separator module not loaded."

    try:
        if not audio_file:
            return "", "Error: Please upload an audio file."

        progress(0.1, desc="Initializing...")

        output_dir = os.path.join("output", "separated", os.path.splitext(os.path.basename(audio_file))[0])
        os.makedirs(output_dir, exist_ok=True)

        # Convert num_speakers to int or None
        n_speakers = int(num_speakers) if num_speakers and num_speakers > 0 else None

        # Use token from input or env
        token = hf_token.strip() if hf_token else os.environ.get("HF_TOKEN")

        progress(0.2, desc="Loading PyAnnote Pipeline (Download may take time on first run)...")

        output_files = separator.separate_voices(
            audio_path=audio_file,
            output_dir=output_dir,
            num_speakers=n_speakers,
            auth_token=token
        )

        progress(1.0, desc="Done!")
        return output_files, f"Separation complete! Saved {len(output_files)} tracks."
    except Exception as e:
        logger.error(f"Separation Error: {e}")
        msg = str(e)
        if "HuggingFace" in msg or "token" in msg.lower():
            msg += "\n\nTip: You need a Hugging Face token with access to 'pyannote/speaker-diarization-3.1'."
        return None, f"Error: {msg}"


def transcribe_audio_ui(audio_file, model_size, language_hint, output_format, beam_size, prompt_text, progress=gr.Progress()):
    if not whisper_transcriber:
        return "", "Error: Whisper transcription module not loaded. Install whisperx in a dedicated env if needed."

    try:
        if not audio_file:
            return "", "Error: Please upload an audio file."

        progress(0.1, desc="Initializing...")

        output_dir = os.path.join("output", "transcriptions")
        os.makedirs(output_dir, exist_ok=True)

        import time
        timestamp = int(time.time())
        normalized_format = (output_format or "Text").strip()
        if normalized_format.lower().startswith("plain"):
            normalized_format = "Text"
        if normalized_format.lower().startswith("srt"):
            ext = "srt"
        elif normalized_format.lower() == "json":
            ext = "json"
        else:
            ext = "txt"
        output_path = os.path.join(output_dir, f"transcription_{timestamp}.{ext}")

        progress(0.2, desc=f"Loading Whisper {model_size} model (Download may take time on first run)...")

        language = None if (language_hint or "").lower().startswith("auto") else language_hint
        initial_prompt = (prompt_text or "").strip() or None
        if initial_prompt is None and language is None:
            initial_prompt = "You are a professional transcriber. The speaker may switch between languages. Transcribe exactly as spoken."

        text, detected_language, language_prob = whisper_transcriber.transcribe_audio(
            audio_path=audio_file,
            model_size=model_size,
            language=language,
            output_format=normalized_format,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
        )

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(text)

        progress(0.9, desc="Finalizing transcription...")
        detected_str = detected_language or "Unknown"
        if language_prob is not None:
            detected_str = f"{detected_str} ({language_prob:.2f})"
        return text, f"Transcription complete! Detected language: {detected_str}"
    except Exception as e:
        logger.error(f"Transcription Error: {e}")
        return "", f"Error: {str(e)}"

def update_qwen_tts_ui(mode):
    """
    Compact Qwen-only UI updater used by app.py.
    Returns visibility updates for:
    [tts_speaker, tts_instruct]
    """
    if mode == "Preset Voice":
        return [
            gr.update(visible=True),
            gr.update(visible=False),
        ]
    if mode == "Voice Design":
        return [
            gr.update(visible=False),
            gr.update(visible=True),
        ]
    if mode == "Voice Cloning":
        return [
            gr.update(visible=False),
            gr.update(visible=True),
        ]
    return [gr.update(visible=False), gr.update(visible=False)]
