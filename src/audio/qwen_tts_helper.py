import os
import io
import gc
import inspect
import torch
import logging
import numpy as np
import importlib
import contextlib
from typing import Optional, Callable, Any

try:
    import soundfile as sf
except Exception:
    sf = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenTTSHandler:
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.current_model = None
        self.current_model_type = None  # "custom", "design", or "base"

        # Model IDs / paths (override with env vars for offline/local use)
        self.cache_dir = os.getenv("QWEN_TTS_CACHE_DIR")
        self.model_sources = {
            "custom": {
                "id": os.getenv("QWEN_TTS_CUSTOM_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
                "path": os.getenv("QWEN_TTS_CUSTOM_MODEL_PATH"),
            },
            "design": {
                "id": os.getenv("QWEN_TTS_DESIGN_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
                "path": os.getenv("QWEN_TTS_DESIGN_MODEL_PATH"),
            },
            "base": {
                "id": os.getenv("QWEN_TTS_BASE_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
                "path": os.getenv("QWEN_TTS_BASE_MODEL_PATH"),
            },
        }
        self.supported_speakers = None
        self.supported_languages = None

    def _env_flag(self, name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    def _resolve_torch_dtype(self) -> torch.dtype:
        setting = (os.getenv("QWEN_TTS_DTYPE") or "auto").strip().lower()
        default_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        mapping = {
            "auto": default_dtype,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        dtype = mapping.get(setting)
        if dtype is None:
            logger.warning(f"Unknown QWEN_TTS_DTYPE '{setting}'. Falling back to {default_dtype}.")
            dtype = default_dtype

        if dtype == torch.bfloat16 and self.device.startswith("cuda"):
            is_supported = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(is_supported) and not is_supported():
                logger.warning("bfloat16 not supported on this GPU. Falling back to float16.")
                dtype = torch.float16
        return dtype

    def _unload_current_model(self) -> None:
        if self.current_model is None:
            return
        logger.info(f"Unloading {self.current_model_type} model to free VRAM...")
        del self.current_model
        self.current_model = None
        self.current_model_type = None
        self.supported_speakers = None
        self.supported_languages = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _write_audio(self, output_path: str, audio: np.ndarray, sr: int) -> None:
        if audio.ndim == 2 and audio.shape[0] <= 4 and audio.shape[0] < audio.shape[1]:
            audio = audio.T

        if sf is not None:
            sf.write(output_path, audio, sr)
            return

        from scipy.io import wavfile

        if np.issubdtype(audio.dtype, np.floating):
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * np.iinfo(np.int16).max).astype(np.int16)

        wavfile.write(output_path, sr, audio)

    def _run_suppress_flash_attn(self, func: Callable[[], Any]) -> Any:
        buffer = io.StringIO()
        if os.name == "nt":
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                result = func()
        else:
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            try:
                with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                    result = func()
            finally:
                os.dup2(old_stdout_fd, 1)
                os.dup2(old_stderr_fd, 2)
                os.close(old_stdout_fd)
                os.close(old_stderr_fd)
                os.close(devnull)
        output = buffer.getvalue()
        if output:
            filtered = "\n".join(
                line for line in output.splitlines()
                if "flash-attn" not in line.lower() and "sox" not in line.lower()
            )
            if filtered.strip():
                print(filtered)
        return result

    def _resolve_model_source(self, model_type: str) -> str:
        model_info = self.model_sources.get(model_type)
        if not model_info:
            raise ValueError(f"Unknown model type: {model_type}")

        model_path = model_info.get("path")
        if model_path:
            if os.path.exists(model_path):
                return model_path
            logger.warning(f"Model path not found: {model_path}. Falling back to model ID.")

        return model_info["id"]

    def load_model(self, model_type: str):
        """
        Load the specific model type.
        model_type: "custom" (for presets), "design" (for instructions), or "base" (for cloning)
        """
        # If we already have the right model loaded, do nothing
        if self.current_model is not None and self.current_model_type == model_type:
            return

        # Unload previous model to save VRAM
        if self.current_model is not None:
            self._unload_current_model()

        try:
            qwen_module = self._run_suppress_flash_attn(
                lambda: importlib.import_module("qwen_tts")
            )
            Qwen3TTSModel = qwen_module.Qwen3TTSModel

            target_id = self._resolve_model_source(model_type)
            logger.info(f"Loading Qwen3 TTS model: {target_id} on {self.device}")

            torch_dtype = self._resolve_torch_dtype()
            load_kwargs = {
                "dtype": torch_dtype,
                "cache_dir": self.cache_dir,
            }
            try:
                sig = inspect.signature(Qwen3TTSModel.from_pretrained)
                if "device" in sig.parameters:
                    load_kwargs["device"] = self.device
                elif "device_map" in sig.parameters:
                    load_kwargs["device_map"] = "auto" if self.device.startswith("cuda") else "cpu"

                attn_impl = (os.getenv("QWEN_TTS_ATTN_IMPLEMENTATION") or "").strip()
                if attn_impl and "attn_implementation" in sig.parameters:
                    load_kwargs["attn_implementation"] = attn_impl
            except (TypeError, ValueError):
                pass
            self.current_model = self._run_suppress_flash_attn(
                lambda: Qwen3TTSModel.from_pretrained(
                    target_id,
                    **load_kwargs,
                )
            )
            if hasattr(self.current_model, "eval"):
                self.current_model.eval()
            self.current_model_type = model_type
            self.supported_speakers = None
            self.supported_languages = None

            if hasattr(self.current_model, "get_supported_speakers"):
                self.supported_speakers = self.current_model.get_supported_speakers()
            if hasattr(self.current_model, "get_supported_languages"):
                self.supported_languages = self.current_model.get_supported_languages()

            logger.info("Model loaded successfully.")
        except ImportError:
            logger.error("qwen-tts package not installed. Please install it via pip install qwen-tts")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_speech(
        self,
        text: str,
        mode: str = "custom",  # custom, design
        language: str = "English",
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        output_path: str = "output_tts.wav"
    ) -> str:
        """
        Generate speech using Qwen3 TTS.
        """
        try:
            logger.info(f"Generating speech in mode: {mode}")
            audio_tensor = None

            # Determine which model we need
            if mode == "design":
                needed_model_type = "design"
            else:
                needed_model_type = "custom"
            self.load_model(needed_model_type)

            # Handle Auto language: pass None to let model auto-detect
            if language and language.lower() == "auto":
                language = None
            elif language and self.supported_languages and language not in self.supported_languages:
                logger.warning(f"Unsupported language '{language}'. Falling back to {self.supported_languages[0]}.")
                language = self.supported_languages[0]

            if mode == "custom":
                # Pre-defined speakers
                if not speaker:
                    speaker = "Vivian" # Default
                if self.supported_speakers and speaker not in self.supported_speakers:
                    logger.warning(f"Unsupported speaker '{speaker}'. Falling back to {self.supported_speakers[0]}.")
                    speaker = self.supported_speakers[0]
                with torch.inference_mode():
                    audio_tensor = self._run_suppress_flash_attn(
                        lambda: self.current_model.generate_custom_voice(
                            text=text,
                            language=language,
                            speaker=speaker,
                            instruct=instruct
                        )
                    )

            elif mode == "design":
                # Voice design with instructions
                if not instruct:
                    instruct = "Speak naturally."
                with torch.inference_mode():
                    audio_tensor = self._run_suppress_flash_attn(
                        lambda: self.current_model.generate_voice_design(
                            text=text,
                            language=language,
                            instruct=instruct
                        )
                    )
            else:
                raise ValueError(f"Unknown mode: {mode}. Voice cloning has been removed.")

            # Save output
            sr = None
            # Check if it returned a file path (some APIs do)
            if isinstance(audio_tensor, str) and os.path.exists(audio_tensor):
                import shutil
                shutil.copy(audio_tensor, output_path)
                return output_path

            # If it's a list/tuple (some models return (audio, sr))
            if isinstance(audio_tensor, (list, tuple)) and len(audio_tensor) == 2 and isinstance(audio_tensor[1], (int, float)):
                audio_tensor, sr = audio_tensor
            if isinstance(audio_tensor, (list, tuple)):
                audio_tensor = audio_tensor[0]

            if isinstance(audio_tensor, np.ndarray):
                audio_tensor = torch.from_numpy(audio_tensor)

            # If tensor, handle moving to CPU and saving
            if hasattr(audio_tensor, "cpu"):
                audio_tensor = audio_tensor.cpu()

            # Normalize tensor dimensions for torchaudio [channels, time]
            if isinstance(audio_tensor, torch.Tensor):
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)

            # Get sample rate - defaults to 24000 if not found
            # Try to find it on the model config
            if sr is None:
                sr = 24000
                if hasattr(self.current_model, "config") and hasattr(self.current_model.config, "sampling_rate"):
                    sr = self.current_model.config.sampling_rate
                elif hasattr(self.current_model, "sampling_rate"):
                    sr = self.current_model.sampling_rate

            if isinstance(audio_tensor, torch.Tensor):
                audio_tensor = audio_tensor.numpy()
            audio_tensor = np.asarray(audio_tensor)

            self._write_audio(output_path, audio_tensor, sr)
            logger.info(f"Audio saved to {output_path}")
            if self._env_flag("QWEN_TTS_UNLOAD_AFTER_GEN"):
                self._unload_current_model()
            return output_path

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

# Global instance for easy import
qwen_handler = QwenTTSHandler()
