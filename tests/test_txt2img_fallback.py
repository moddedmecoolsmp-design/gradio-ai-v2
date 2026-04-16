import unittest

from src.runtime_policies import choose_image_generation_mode


class Txt2ImgFallbackTests(unittest.TestCase):
    def test_zimage_without_input_images_uses_txt2img(self):
        mode = choose_image_generation_mode(
            current_model="zimage-int8",
            has_input_images=False,
            enable_pose_preservation=False,
        )
        self.assertEqual(mode, "txt2img")

    def test_flux_pose_mode_keeps_txt2img_path(self):
        mode = choose_image_generation_mode(
            current_model="flux2-klein-int8",
            has_input_images=True,
            enable_pose_preservation=True,
        )
        self.assertEqual(mode, "txt2img")


if __name__ == "__main__":
    unittest.main()
