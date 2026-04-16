import os
import gc
import json
import logging
import shutil
from dataclasses import replace
from typing import Optional, List, Dict, Any, Tuple, Callable

import torch

if os.name == "nt":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        ffmpeg_alias = os.path.join(ffmpeg_dir, "ffmpeg.exe")
        if not os.path.exists(ffmpeg_alias):
            shutil.copyfile(ffmpeg_exe, ffmpeg_alias)
        os.environ["PATH"] = f"{ffmpeg_dir};{os.environ.get('PATH', '')}"
        os.environ.setdefault("FFMPEG_BINARY", ffmpeg_alias)
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", ffmpeg_alias)
    except Exception as exc:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to configure ffmpeg via imageio-ffmpeg: {exc}")

try:
    import whisperx
except Exception:
    whisperx = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperTranscriber:
    def __init__(self, device: Optional[str] = None):
        if os.name == "nt":
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        env_device = (os.getenv("WHISPER_DEVICE") or "").strip()
        if device:
            self.device = device
        elif env_device:
            self.device = env_device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.cache_dir = os.getenv("WHISPER_CACHE_DIR")
        self.supported_models = ["large-v3", "medium", "small", "base"]
        self.default_model = os.getenv("WHISPER_DEFAULT_MODEL", "large-v3")
        self.current_model = None
        self.current_model_size = None

    def _env_flag(self, name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    def _resolve_model_size(self, model_size: Optional[str]) -> str:
        size = (model_size or "").strip() or self.default_model
        if size not in self.supported_models:
            logger.warning(f"Unsupported Whisper model '{size}'. Falling back to {self.default_model}.")
            size = self.default_model
        return size

    def _resolve_compute_type(self) -> str:
        setting = (os.getenv("WHISPER_COMPUTE_TYPE") or "").strip().lower()
        if setting:
            return setting
        if self.device.startswith("cuda"):
            return "int8_float16"
        return "int8"

    def _resolve_batch_size(self) -> int:
        setting = (os.getenv("WHISPER_BATCH_SIZE") or "").strip()
        if setting:
            try:
                return max(1, int(setting))
            except ValueError:
                logger.warning(f"Invalid WHISPER_BATCH_SIZE '{setting}'. Using default.")
        return 4 if self.device.startswith("cuda") else 1

    def _unload_model(self) -> None:
        if self.current_model is None:
            return
        logger.info(f"Unloading Whisper model '{self.current_model_size}' to free memory...")
        del self.current_model
        self.current_model = None
        self.current_model_size = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, model_size: Optional[str]) -> None:
        if whisperx is None:
            raise ImportError("whisperx is not installed. Install with 'pip install whisperx'.")

        size = self._resolve_model_size(model_size)
        if self.current_model is not None and self.current_model_size == size:
            return

        if self.current_model is not None:
            self._unload_model()

        compute_type = self._resolve_compute_type()
        logger.info(f"Loading WhisperX model '{size}' on {self.device} ({compute_type})...")
        self.current_model = whisperx.load_model(
            size,
            self.device,
            compute_type=compute_type,
            download_root=self.cache_dir,
            vad_method="silero",
            asr_options={
                "beam_size": 5,
                "best_of": 5,
                "patience": 1.0,
                "length_penalty": 1.0,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0,
                "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": False,
                "prompt_reset_on_temperature": 0.5,
                "initial_prompt": None,
                "prefix": None,
                "suppress_blank": True,
                "suppress_tokens": [-1],
                "without_timestamps": False,
                "max_initial_timestamp": 1.0,
                "word_timestamps": False,
            },
        )
        self.current_model_size = size

    def _update_options(self, beam_size: int, initial_prompt: Optional[str]) -> None:
        if not self.current_model or not hasattr(self.current_model, "options"):
            return
        try:
            options = self.current_model.options
            updated = replace(
                options,
                beam_size=int(beam_size) if beam_size else options.beam_size,
                initial_prompt=initial_prompt,
                condition_on_previous_text=False,
                without_timestamps=False,
            )
            self.current_model.options = updated
        except Exception as exc:
            logger.warning(f"Failed to update WhisperX options: {exc}")

    def _format_text_timestamp(self, seconds: float) -> str:
        if seconds is None:
            seconds = 0.0
        milliseconds = int(round(seconds * 1000))
        hours = milliseconds // 3600000
        milliseconds %= 3600000
        minutes = milliseconds // 60000
        milliseconds %= 60000
        secs = milliseconds // 1000
        milliseconds %= 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

    def _format_srt_timestamp(self, seconds: float) -> str:
        if seconds is None:
            seconds = 0.0
        milliseconds = int(round(seconds * 1000))
        hours = milliseconds // 3600000
        milliseconds %= 3600000
        minutes = milliseconds // 60000
        milliseconds %= 60000
        secs = milliseconds // 1000
        milliseconds %= 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def _normalize_language(self, language: Optional[str]) -> Optional[str]:
        if not language:
            return None
        lowered = language.strip().lower()
        if lowered in {"auto", "auto-detect", "auto detect"}:
            return None
        return language

    def _transcribe_with_loaded_model(
        self,
        audio_path: str,
        language: Optional[str],
        output_format: str,
        beam_size: int,
        initial_prompt: Optional[str],
    ) -> Tuple[str, Optional[str], Optional[float]]:
        if self.current_model is None:
            raise RuntimeError("Whisper model is not loaded.")

        beam_size = int(beam_size) if beam_size else 5

        self._update_options(beam_size=beam_size, initial_prompt=initial_prompt)
        batch_size = self._resolve_batch_size()
        # Load audio and get VAD segments from the model
        import whisperx.audio
        audio = whisperx.audio.load_audio(audio_path)
        
        # Get VAD segments using the model's VAD implementation
        vad_model = self.current_model.vad_model
        if hasattr(vad_model, "preprocess_audio") and hasattr(vad_model, "merge_chunks"):
            waveform = vad_model.preprocess_audio(audio)
            merge_chunks = vad_model.merge_chunks
        else:
            from whisperx.vads import Pyannote
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks = Pyannote.merge_chunks
        vad_segments = vad_model({"waveform": waveform, "sample_rate": whisperx.audio.SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size=30,
            onset=self.current_model._vad_params["vad_onset"],
            offset=self.current_model._vad_params["vad_offset"],
        )
        
        # Transcribe each VAD segment separately with fresh language detection
        all_segments = []
        for seg in vad_segments:
            start_time = seg['start']
            end_time = seg['end']
            
            # Extract audio chunk
            f1 = int(start_time * whisperx.audio.SAMPLE_RATE)
            f2 = int(end_time * whisperx.audio.SAMPLE_RATE)
            chunk_audio = audio[f1:f2]
            
            # Save chunk to temp file
            import tempfile
            import soundfile as sf
            import os
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, chunk_audio, whisperx.audio.SAMPLE_RATE)
                temp_file = tmp.name
            
            try:
                # Transcribe chunk with language detection
                chunk_result = self.current_model.transcribe(
                    temp_file,
                    batch_size=1,
                    language=language if language else None,
                    task="transcribe",
                )
                
                # Adjust segment timings and add to results
                for segment in chunk_result.get("segments", []):
                    segment["start"] += start_time
                    segment["end"] += start_time
                    # Add detected language for this chunk
                    segment["language"] = chunk_result.get("language")
                    all_segments.append(segment)
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
        
        segments = all_segments

        format_key = (output_format or "text").strip().lower()
        output_text = ""

        if format_key.startswith("srt"):
            lines = []
            for idx, segment in enumerate(segments, start=1):
                start = self._format_srt_timestamp(segment.get("start", 0.0))
                end = self._format_srt_timestamp(segment.get("end", 0.0))
                text = (segment.get("text") or "").strip()
                lines.extend([str(idx), f"{start} --> {end}", text, ""])
            output_text = "\n".join(lines).strip() + "\n"
        elif format_key == "json":
            payload = {
                "language": "multilingual",  # Indicate multiple languages detected
                "language_probability": None,
                "segments": [
                    {
                        "start": segment.get("start"),
                        "end": segment.get("end"),
                        "text": (segment.get("text") or "").strip(),
                        "language": segment.get("language"),  # Include language per segment
                    }
                    for segment in segments
                ],
            }
            output_text = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            lines = [
                (segment.get("text") or "").strip()
                for segment in segments
                if (segment.get("text") or "").strip()
            ]
            output_text = "\n".join(lines).strip() + "\n"

        # For non-JSON formats, return the most common detected language
        languages = [seg.get("language") for seg in segments if seg.get("language")]
        detected_language = max(set(languages), key=languages.count) if languages else "multilingual"
        language_probability = None
        return output_text, detected_language, language_probability

    def transcribe_audio(
        self,
        audio_path: str,
        model_size: Optional[str] = None,
        language: Optional[str] = None,
        output_format: str = "Text",
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> Tuple[str, Optional[str], Optional[float]]:
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError("Audio file not found.")

        size = self._resolve_model_size(model_size)
        self.load_model(size)

        output_text, detected_language, language_probability = self._transcribe_with_loaded_model(
            audio_path=audio_path,
            language=language,
            output_format=output_format,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
        )

        if self._env_flag("WHISPER_UNLOAD_AFTER_TRANSCRIBE"):
            self._unload_model()

        return output_text, detected_language, language_probability

    def transcribe_batch(
        self,
        audio_files: List[str],
        model_size: Optional[str] = None,
        language: Optional[str] = None,
        output_format: str = "Text",
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
        progress_cb: Optional[Callable[[float, Optional[str]], None]] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if not audio_files:
            return results

        size = self._resolve_model_size(model_size)
        self.load_model(size)

        total = len(audio_files)
        for idx, audio_path in enumerate(audio_files, start=1):
            if progress_cb:
                progress_cb((idx - 1) / total, f"Transcribing {idx}/{total}...")
            try:
                output_text, detected_language, language_probability = self._transcribe_with_loaded_model(
                    audio_path=audio_path,
                    language=language,
                    output_format=output_format,
                    beam_size=beam_size,
                    initial_prompt=initial_prompt,
                )
                results.append({
                    "path": audio_path,
                    "output_text": output_text,
                    "language": detected_language,
                    "language_probability": language_probability,
                })
            except Exception as exc:
                results.append({
                    "path": audio_path,
                    "error": str(exc),
                })

        if progress_cb:
            progress_cb(1.0, "Batch transcription complete.")

        if self._env_flag("WHISPER_UNLOAD_AFTER_TRANSCRIBE"):
            self._unload_model()

        return results


whisper_transcriber = WhisperTranscriber()
