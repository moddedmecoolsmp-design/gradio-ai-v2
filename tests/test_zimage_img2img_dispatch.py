import unittest

from src.runtime_policies import choose_image_generation_mode, resolve_zimage_img2img_steps


class ZImageDispatchTests(unittest.TestCase):
    def test_zimage_with_input_images_uses_img2img_mode(self):
        mode = choose_image_generation_mode(
            current_model="zimage-int8",
            has_input_images=True,
            enable_pose_preservation=False,
        )
        self.assertEqual(mode, "zimage-img2img")

    def test_zimage_img2img_steps_are_clamped(self):
        steps, clamped = resolve_zimage_img2img_steps(steps=4, minimum_steps=8)
        self.assertEqual(steps, 8)
        self.assertTrue(clamped)


if __name__ == "__main__":
    unittest.main()
