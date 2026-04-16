import unittest

from src.runtime_policies import should_show_edit_controls


class UiVisibilityTests(unittest.TestCase):
    def test_zimage_model_shows_edit_controls(self):
        self.assertTrue(should_show_edit_controls("Z-Image Turbo (Int8 - 8GB Safe)"))

    def test_flux_model_shows_edit_controls(self):
        self.assertTrue(should_show_edit_controls("FLUX.2-klein-4B (Int8)"))


if __name__ == "__main__":
    unittest.main()
