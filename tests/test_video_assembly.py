from pathlib import Path
import unittest

from src.core.video_temporal import (
    build_deflicker_filter,
    build_ffmpeg_audio_trim_command,
    build_ffmpeg_deflicker_command,
    build_ffmpeg_frame_encode_command,
    build_ffmpeg_merge_command,
)


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class VideoAssemblyTests(unittest.TestCase):
    def test_create_video_from_frames_applies_deflicker_before_audio_merge(self):
        deflicker_cmd = build_ffmpeg_deflicker_command("ffmpeg", "temp_no_audio.mp4", "temp_deflickered.mp4", 5)
        merge_cmd = build_ffmpeg_merge_command("ffmpeg", "temp_deflickered.mp4", "temp_audio.aac", "final.mp4")
        self.assertIn("-vf", deflicker_cmd)
        self.assertIn(build_deflicker_filter(5), deflicker_cmd)
        self.assertEqual(merge_cmd[merge_cmd.index("-i") + 1], "temp_deflickered.mp4")

        app_source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("build_ffmpeg_deflicker_command", app_source)
        self.assertIn("video_path=temp_filtered_video", app_source)

    def test_create_video_from_frames_applies_deflicker_without_audio(self):
        encode_cmd = build_ffmpeg_frame_encode_command("ffmpeg", 24.0, "frame_%05d_out.png", 2.0, "temp_no_audio.mp4")
        deflicker_cmd = build_ffmpeg_deflicker_command("ffmpeg", "temp_no_audio.mp4", "temp_deflickered.mp4", 5)
        self.assertIn("frame_%05d_out.png", encode_cmd)
        self.assertIn("temp_no_audio.mp4", deflicker_cmd)
        self.assertIn("temp_deflickered.mp4", deflicker_cmd)

        app_source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("shutil.copy(temp_filtered_video, output_path)", app_source)

    def test_create_video_from_frames_reports_failure_when_filtered_ffmpeg_step_fails(self):
        app_source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("FFmpeg deflicker failed:", app_source)

    def test_audio_trim_command_uses_expected_duration(self):
        audio_cmd = build_ffmpeg_audio_trim_command("ffmpeg", "input.mp4", 3.25, "audio.aac")
        self.assertIn("-t", audio_cmd)
        self.assertIn("3.25", audio_cmd)


if __name__ == "__main__":
    unittest.main()
