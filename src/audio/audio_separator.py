import os
import torch
import logging
from typing import List, Tuple, Optional
from pydub import AudioSegment

# Library modules must not call ``logging.basicConfig`` — the first importer
# wins and silently fixes the root handler/formatter for the whole process.
# Logging configuration is the entry-point's responsibility; we just
# acquire our named logger here.
logger = logging.getLogger(__name__)

class AudioSeparator:
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pipeline = None

    def load_pipeline(self, use_auth_token: Optional[str] = None):
        if self.pipeline is None:
            try:
                from pyannote.audio import Pipeline
                # Try loading the standard pipeline.
                # Note: This usually requires a HuggingFace token and acceptance of terms for pyannote/speaker-diarization-3.1
                # If user hasn't set it up, this might fail.
                # We can fallback or ask for token.

                model_id = "pyannote/speaker-diarization-3.1"
                logger.info(f"Loading Diarization pipeline: {model_id} on {self.device}")

                self.pipeline = Pipeline.from_pretrained(
                    model_id,
                    use_auth_token=use_auth_token
                )

                if self.pipeline is None:
                    # Fallback to older model or error
                     model_id = "pyannote/speaker-diarization"
                     logger.warning(f"Failed to load 3.1, trying {model_id}")
                     self.pipeline = Pipeline.from_pretrained(model_id, use_auth_token=use_auth_token)

                if self.pipeline:
                    self.pipeline.to(torch.device(self.device))
                    logger.info("Diarization pipeline loaded.")
                else:
                    raise ValueError("Could not load pyannote pipeline. Please ensure you have accepted terms on HuggingFace and provided a valid token.")

            except Exception as e:
                logger.error(f"Failed to load pipeline: {e}")
                # Suggest token solution
                logger.error("Note: pyannote.audio models often require a HuggingFace token. Set HF_TOKEN env var or pass it in.")
                raise

    def separate_voices(
        self,
        audio_path: str,
        output_dir: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        auth_token: Optional[str] = None
    ) -> List[str]:
        """
        Diarize audio and split into separate files per speaker.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        os.makedirs(output_dir, exist_ok=True)

        self.load_pipeline(use_auth_token=auth_token)

        # Run inference
        logger.info(f"Processing {audio_path}...")
        try:
            diarization = self.pipeline(
                audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

        # Load audio for slicing
        full_audio = AudioSegment.from_file(audio_path)

        # Group segments by speaker
        speaker_segments = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []

            # turn.start and turn.end are in seconds
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)

            segment_audio = full_audio[start_ms:end_ms]
            speaker_segments[speaker].append(segment_audio)

        # Merge and save
        output_files = []
        for speaker, segments in speaker_segments.items():
            if not segments:
                continue

            # Concatenate all segments for this speaker
            combined = segments[0]
            for seg in segments[1:]:
                # Add a small silence between segments if desired? For now just append.
                combined += seg

            out_filename = f"{speaker}.mp3"
            out_path = os.path.join(output_dir, out_filename)

            combined.export(out_path, format="mp3")
            output_files.append(out_path)
            logger.info(f"Saved {speaker} to {out_path}")

        return output_files

# Global instance
separator = AudioSeparator()
