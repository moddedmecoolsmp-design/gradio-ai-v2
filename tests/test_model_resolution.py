import unittest

from src.runtime_policies import (
    FAST_FLUX_MODEL_CHOICE,
    FAST_RESOLUTION_PRESET,
    LOW_VRAM_FLUX_MODEL_CHOICE,
    resolve_default_flux_model_choice,
    resolve_default_resolution_preset,
    resolve_model_choice_for_device,
    should_use_attention_slicing,
    should_use_vae_slicing,
)


class ModelResolutionTests(unittest.TestCase):
    def test_windows_3070_fast_default_picks_int8(self):
        self.assertEqual(
            resolve_default_flux_model_choice(
                device="cuda",
                vram_gb=8.0,
                gpu_name="NVIDIA GeForce RTX 3070",
                platform_system="Windows",
            ),
            FAST_FLUX_MODEL_CHOICE,
        )

    def test_supported_manual_sdnq_choice_passes_through(self):
        model, reason = resolve_model_choice_for_device(
            model_choice=LOW_VRAM_FLUX_MODEL_CHOICE,
            device="cuda",
            vram_gb=8.0,
            gpu_name="NVIDIA GeForce RTX 3070",
            platform_system="Windows",
        )
        self.assertEqual(model, LOW_VRAM_FLUX_MODEL_CHOICE)
        self.assertIsNone(reason)

    def test_zimage_int8_choice_is_not_rewritten_by_runtime_policy(self):
        model, reason = resolve_model_choice_for_device(
            model_choice="Z-Image Turbo (Int8 - 8GB Safe)",
            device="cuda",
            vram_gb=8.0,
        )
        self.assertEqual(model, "Z-Image Turbo (Int8 - 8GB Safe)")
        self.assertIsNone(reason)

    def test_fast_profile_uses_small_default_resolution_for_int8(self):
        self.assertEqual(
            resolve_default_resolution_preset(
                FAST_FLUX_MODEL_CHOICE,
                mode="single",
                device="cuda",
                vram_gb=8.0,
                gpu_name="NVIDIA GeForce RTX 3070",
                platform_system="Windows",
            ),
            FAST_RESOLUTION_PRESET,
        )
        self.assertEqual(
            resolve_default_resolution_preset(
                LOW_VRAM_FLUX_MODEL_CHOICE,
                mode="single",
                device="cuda",
                vram_gb=8.0,
                gpu_name="NVIDIA GeForce RTX 3070",
                platform_system="Windows",
            ),
            "~1024px",
        )

    def test_attention_slicing_stays_off_for_fast_path_until_retry(self):
        self.assertFalse(
            should_use_attention_slicing(
                device="cuda",
                model_key="flux2-klein-int8",
                vram_gb=8.0,
                optimization_profile="max_speed",
            )
        )
        self.assertTrue(
            should_use_attention_slicing(
                device="cuda",
                model_key="flux2-klein-int8",
                vram_gb=8.0,
                optimization_profile="max_speed",
                oom_retry=True,
            )
        )

    def test_vae_slicing_stays_off_for_fast_path_until_retry(self):
        self.assertFalse(
            should_use_vae_slicing(
                device="cuda",
                vram_gb=8.0,
                optimization_profile="max_speed",
            )
        )
        self.assertTrue(
            should_use_vae_slicing(
                device="cuda",
                vram_gb=8.0,
                optimization_profile="max_speed",
                oom_retry=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
