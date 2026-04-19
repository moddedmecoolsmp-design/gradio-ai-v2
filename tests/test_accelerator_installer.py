import unittest
from unittest import mock

import src.utils.accelerator_installer as accelerator_installer
from src.utils.accelerator_installer import _module_importable


class TestAcceleratorInstaller(unittest.TestCase):
    def test_broken_import_is_treated_as_unavailable(self):
        with mock.patch("importlib.import_module", side_effect=RuntimeError("binary mismatch")):
            self.assertFalse(_module_importable("xformers"))

    def test_python313_skips_xformers_probe(self):
        with mock.patch.object(accelerator_installer.sys, "version_info", (3, 13, 5)):
            self.assertEqual(accelerator_installer._install_xformers(), "skipped")


if __name__ == "__main__":
    unittest.main()
