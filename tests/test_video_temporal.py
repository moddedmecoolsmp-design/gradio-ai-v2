from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from src.core.video_temporal import (
    build_video_temporal_config,
    build_occlusion_mask,
    prepare_temporal_condition_frame,
    resolve_video_frame_seed,
    resolve_video_generation_mode,
    resolve_video_shot_seed,
    should_reset_temporal_history,
)


class VideoTemporalTests(unittest.TestCase):
    def _make_frame(self, size=(96, 96), square_xy=(16, 28), square_size=24, bg=0, color=(255, 255, 255)):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        if bg:
            frame[:, :] = bg
        x, y = square_xy
        frame[y:y + square_size, x:x + square_size] = np.array(color, dtype=np.uint8)
        return Image.fromarray(frame, mode="RGB")

    def test_temporal_config_defaults_match_v1_plan(self):
        config = build_video_temporal_config(0.8)
        self.assertTrue(config.enabled)
        self.assertEqual(config.flow_method, "farneback")
        self.assertEqual(config.keyframe_interval, 12)
        self.assertAlmostEqual(config.scene_cut_hist_corr_threshold, 0.65)
        self.assertAlmostEqual(config.temporal_strength, 0.30)
        self.assertAlmostEqual(config.keyframe_strength, 0.55)
        self.assertAlmostEqual(config.occlusion_blend, 0.75)
        self.assertEqual(config.deflicker_size, 5)

    def test_resolve_shot_seed_reuses_explicit_seed_for_all_frames(self):
        shot_seed = resolve_video_shot_seed(1234)
        frame_seeds = [resolve_video_frame_seed(shot_seed, idx) for idx in range(6)]
        self.assertEqual(frame_seeds, [1234] * 6)

    def test_resolve_shot_seed_generates_once_for_seed_minus_one(self):
        calls = {"count": 0}

        def fake_randint():
            calls["count"] += 1
            return 424242

        shot_seed = resolve_video_shot_seed(-1, fake_randint)
        frame_seeds = [resolve_video_frame_seed(shot_seed, idx) for idx in range(4)]
        self.assertEqual(shot_seed, 424242)
        self.assertEqual(frame_seeds, [424242] * 4)
        self.assertEqual(calls["count"], 1)

    def test_histogram_scene_cut_detection_resets_temporal_chain(self):
        config = build_video_temporal_config(0.4)
        previous_frame = Image.new("RGB", (96, 96), (255, 0, 0))
        current_frame = Image.new("RGB", (96, 96), (0, 255, 0))
        should_reset, reason, correlation = should_reset_temporal_history(1, previous_frame, current_frame, config)
        self.assertTrue(should_reset)
        self.assertIn("scene cut", reason)
        self.assertLess(correlation, config.scene_cut_hist_corr_threshold)

    def test_keyframe_refresh_resets_temporal_chain(self):
        config = build_video_temporal_config(0.4)
        frame = self._make_frame(color=(255, 255, 255))
        should_reset, reason, _ = should_reset_temporal_history(config.keyframe_interval, frame, frame, config)
        self.assertTrue(should_reset)
        self.assertIn("keyframe refresh", reason)

    def test_optical_flow_warp_tracks_translated_subject(self):
        previous_raw = self._make_frame(square_xy=(16, 32), color=(255, 255, 255))
        current_raw = self._make_frame(square_xy=(28, 32), color=(255, 255, 255))
        previous_stylized = self._make_frame(square_xy=(16, 32), color=(255, 0, 0))
        result = prepare_temporal_condition_frame(
            previous_raw_frame=previous_raw,
            current_raw_frame=current_raw,
            previous_stylized_frame=previous_stylized,
            config=build_video_temporal_config(0.5),
        )

        warped = np.asarray(result["warped_previous"], dtype=np.uint8)
        red_channel = warped[:, :, 0].sum(axis=0)
        peak_column = int(np.argmax(red_channel))
        self.assertGreaterEqual(peak_column, 24)
        self.assertLessEqual(peak_column, 40)

    def test_occlusion_mask_zeroes_out_invalid_warp_regions(self):
        forward_flow = np.zeros((12, 12, 2), dtype=np.float32)
        backward_flow = np.zeros((12, 12, 2), dtype=np.float32)
        backward_flow[:, :, 0] = 100.0
        mask = build_occlusion_mask(forward_flow, backward_flow)
        self.assertEqual(float(mask.max()), 0.0)
        self.assertEqual(float(mask.min()), 0.0)

    def test_video_dispatch_uses_zimage_img2img_path(self):
        self.assertEqual(resolve_video_generation_mode("zimage-int8"), "zimage-img2img")
        self.assertEqual(resolve_video_generation_mode("flux2-klein-int8"), "flux-img2img")
        self.assertEqual(resolve_video_generation_mode("flux2-klein-int8", enable_pose_preservation=True), "flux-pose")

    def test_video_dispatch_rejects_prompt_only_fallback(self):
        with self.assertRaises(ValueError):
            resolve_video_generation_mode("unknown-model")


if __name__ == "__main__":
    unittest.main()
