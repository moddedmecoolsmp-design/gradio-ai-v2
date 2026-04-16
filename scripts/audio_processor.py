import os
import torch
import whisperx
from pydub import AudioSegment
import imageio_ffmpeg
import argparse

# Configure ffmpeg path for pydub and moviepy
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
AudioSegment.converter = FFMPEG_PATH

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

class AudioProcessor:
    def __init__(self, hf_token=None, device="cuda"):
        self.device = device
        self.hf_token = hf_token
        self.compute_type = "float16" if device == "cuda" else "int8"

    def extract_audio(self, input_path):
        """Extracts audio from video or ensures audio is in correct format."""
        filename = os.path.basename(input_path).split('.')[0]
        temp_audio = f"temp_{filename}.wav"

        if input_path.lower().endswith(('.mp4', '.mkv', '.avi')):
            print(f"Extracting audio from {input_path}...")
            video = VideoFileClip(input_path)
            video.audio.write_audiofile(temp_audio, fps=16000, nbytes=2, codec='pcm_s16le')
            video.close()
        else:
            # For mp3/wav, we still convert to 16kHz mono for optimal processing
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(temp_audio, format="wav")

        return temp_audio

    def process(self, input_path, output_dir="output_voices"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        audio_file = self.extract_audio(input_path)

        # 1. Load WhisperX model
        print("Loading transcription model...")
        model = whisperx.load_model("base", self.device, compute_type=self.compute_type)

        # 2. Transcribe
        print("Transcribing...")
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=16)

        # 3. Align (optional but recommended for better timestamps)
        print("Aligning...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        # 4. Diarization
        print("Running speaker diarization...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)
        diarize_segments = diarize_model(audio)

        # 5. Combine transcript with diarization
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # 6. Extract segments per speaker
        print("Splitting audio by speaker...")
        full_audio = AudioSegment.from_wav(audio_file)

        speaker_segments = {} # speaker_id -> list of (start, end)

        for segment in result["segments"]:
            speaker = segment.get("speaker", "UNKNOWN")
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append((segment["start"] * 1000, segment["end"] * 1000)) # ms

        # Export files
        for speaker, segments in speaker_segments.items():
            combined = AudioSegment.empty()
            # Merge segments with small gaps (optional enhancement)
            for start, end in segments:
                combined += full_audio[start:end]

            output_file = os.path.join(output_dir, f"{speaker}.mp3")
            combined.export(output_file, format="mp3")
            print(f"Saved {output_file}")

        # Cleanup
        if os.path.exists(audio_file):
            os.remove(audio_file)

        print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split audio/video into separate speaker files.")
    parser.add_argument("input", help="Path to mp4 or mp3 file")
    parser.add_argument("--token", help="Hugging Face Token (required for diarization)", required=True)
    parser.add_argument("--out", default="output_voices", help="Output directory")

    args = parser.parse_args()

    processor = AudioProcessor(hf_token=args.token)
    processor.process(args.input, args.out)
