from pathlib import Path
import unittest

from src.runtime_policies import (
    build_dependency_profile_metadata,
    compute_file_sha256,
    is_cuda13_runtime,
    is_dependency_metadata_current,
    select_requirements_file,
)


class StartupPreflightTests(unittest.TestCase):
    def test_select_requirements_prefers_cuda_lockfile_on_windows(self):
        tmp_path = Path("tests/.tmp_preflight")
        tmp_path.mkdir(parents=True, exist_ok=True)
        try:
            (tmp_path / "requirements-lock-cu130.txt").write_text("x", encoding="utf-8")
            selected = select_requirements_file(
                base_dir=str(tmp_path),
                is_windows=True,
                cuda_available=True,
            )
            self.assertEqual(selected, "requirements-lock-cu130.txt")
        finally:
            for child in tmp_path.glob("*"):
                child.unlink(missing_ok=True)
            tmp_path.rmdir()

    def test_select_requirements_prefers_lockfile_even_when_cuda_unavailable(self):
        tmp_path = Path("tests/.tmp_preflight")
        tmp_path.mkdir(parents=True, exist_ok=True)
        try:
            (tmp_path / "requirements-lock-cu130.txt").write_text("x", encoding="utf-8")
            selected = select_requirements_file(
                base_dir=str(tmp_path),
                is_windows=True,
                cuda_available=False,
            )
            self.assertEqual(selected, "requirements-lock-cu130.txt")
        finally:
            for child in tmp_path.glob("*"):
                child.unlink(missing_ok=True)
            tmp_path.rmdir()

    def test_dependency_metadata_detects_stale_and_current_profiles(self):
        tmp_path = Path("tests/.tmp_preflight")
        tmp_path.mkdir(parents=True, exist_ok=True)
        try:
            req = tmp_path / "requirements.txt"
            req.write_text("gradio==6.0.0", encoding="utf-8")
            metadata = build_dependency_profile_metadata("requirements.txt", compute_file_sha256(str(req)))

            self.assertFalse(is_dependency_metadata_current({}, metadata))
            self.assertTrue(is_dependency_metadata_current(metadata, metadata))
        finally:
            for child in tmp_path.glob("*"):
                child.unlink(missing_ok=True)
            tmp_path.rmdir()

    def test_app_contains_startup_preflight_path(self):
        app_source = Path("app.py").read_text(encoding="utf-8")
        self.assertIn("def run_dependency_preflight()", app_source)
        self.assertIn("def enforce_cuda13_runtime_profile()", app_source)
        self.assertIn("if not enforce_cuda13_runtime_profile()", app_source)
        self.assertNotIn('"-m", "pip", "install", "-r", requirements_path', app_source)
        self.assertIn("Run Install.bat --repair, then relaunch.", app_source)

    def test_legacy_dependency_metadata_handling_exists(self):
        app_source = Path("app.py").read_text(encoding="utf-8")
        self.assertIn("_legacy_metadata_text", app_source)

    def test_startup_uses_hf_home_without_transformers_cache_override(self):
        app_source = Path("app.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.setdefault("HF_HOME"', app_source)
        self.assertNotIn('os.environ.setdefault("TRANSFORMERS_CACHE"', app_source)

    def test_app_uses_pipeline_manager_for_offload_and_zimage_img2img(self):
        app_source = Path("app.py").read_text(encoding="utf-8")
        self.assertIn("pipeline_manager.should_enable_cpu_offload", app_source)
        self.assertIn("pipeline_manager.get_zimage_img2img_pipeline", app_source)
        self.assertIn("def _probe_optional_dependency_statuses()", app_source)

    def test_tts_event_wiring_uses_canonical_model_argument(self):
        app_source = Path("app.py").read_text(encoding="utf-8")
        self.assertIn('tts_model = gr.State("Qwen TTS")', app_source)
        self.assertIn("fn=audio_ui_helpers.update_qwen_tts_ui", app_source)

    def test_app_only_exposes_3070_safe_models(self):
        app_source = Path("app.py").read_text(encoding="utf-8")
        self.assertIn("FAST_FLUX_MODEL_CHOICE", app_source)
        self.assertIn("LOW_VRAM_FLUX_MODEL_CHOICE", app_source)
        self.assertIn('"Z-Image Turbo (Int8 - 8GB Safe)"', app_source)
        self.assertNotIn('"Z-Image Turbo (Official BF16)"', app_source)
        self.assertNotIn("FLUX.2-klein-9B (4bit SDNQ - Higher Quality)", app_source)
        self.assertNotIn("FLUX.2-klein-4B (NVFP4 - Experimental)", app_source)

    def test_app_prefers_int8_first_for_3070_fast_path(self):
        app_source = Path("app.py").read_text(encoding="utf-8")
        self.assertLess(
            app_source.index("FAST_FLUX_MODEL_CHOICE"),
            app_source.index("LOW_VRAM_FLUX_MODEL_CHOICE"),
        )
        self.assertIn("fastest FLUX default for Windows 11 + RTX 3070", app_source)
        self.assertIn("manual low-VRAM FLUX fallback for RTX 3070", app_source)

    def test_tensorrt_references_are_removed_from_install_and_verify_paths(self):
        app_source = Path("app.py").read_text(encoding="utf-8")
        install_source = Path("Install.bat").read_text(encoding="utf-8")
        verify_bat_source = Path("scripts/Verify.bat").read_text(encoding="utf-8")
        verify_py_source = Path("scripts/verify_install.py").read_text(encoding="utf-8")
        self.assertNotIn("Enable TensorRT", app_source)
        self.assertNotIn("enable_tensorrt", app_source)
        self.assertNotIn("ENABLE_TRT", install_source)
        self.assertNotIn("torch-tensorrt", install_source)
        self.assertNotIn("torch_tensorrt", verify_bat_source)
        self.assertNotIn("ENABLE_TRT", verify_py_source)
        self.assertNotIn("torch_tensorrt", verify_py_source)

    def test_cuda13_runtime_detector(self):
        self.assertTrue(is_cuda13_runtime("13.0"))
        self.assertTrue(is_cuda13_runtime("13.1"))
        self.assertFalse(is_cuda13_runtime("12.8"))
        self.assertFalse(is_cuda13_runtime(None))


if __name__ == "__main__":
    unittest.main()
