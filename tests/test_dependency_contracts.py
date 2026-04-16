from pathlib import Path
import json
import re
import subprocess
import sys
import unittest
import tomli

from src.runtime_policies import build_dependency_profile_metadata


def _parse_requirements(path: Path) -> dict:
    parsed = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if " @ " in line:
            name, value = line.split(" @ ", 1)
            parsed[name.strip()] = value.strip()
            continue
        if "==" in line:
            name, value = line.split("==", 1)
            parsed[name.strip()] = value.strip()
            continue
    return parsed


class DependencyContractTests(unittest.TestCase):
    def test_lockfile_core_versions_are_compatible(self):
        lock = _parse_requirements(Path("requirements-lock-cu130.txt"))
        self.assertEqual(lock.get("transformers"), "4.57.3")
        self.assertEqual(lock.get("qwen-tts"), "0.1.1")
        self.assertEqual(lock.get("huggingface_hub[hf_xet]"), "0.36.2")

    def test_git_dependencies_are_commit_pinned(self):
        lock_lines = Path("requirements-lock-cu130.txt").read_text(encoding="utf-8")
        self.assertRegex(
            lock_lines,
            r"diffusers\s+@\s+git\+https://github\.com/huggingface/diffusers@[0-9a-f]{40}",
        )
        self.assertRegex(
            lock_lines,
            r"sdnq\s+@\s+git\+https://github\.com/Disty0/sdnq\.git@[0-9a-f]{40}",
        )

    def test_requirements_and_lockfile_align_for_core_packages(self):
        lock = _parse_requirements(Path("requirements-lock-cu130.txt"))
        req = _parse_requirements(Path("requirements.txt"))
        self.assertEqual(req.get("transformers"), lock.get("transformers"))
        self.assertEqual(req.get("qwen-tts"), lock.get("qwen-tts"))
        self.assertEqual(req.get("huggingface_hub[hf_xet]"), lock.get("huggingface_hub[hf_xet]"))

    def test_dependency_metadata_is_json_roundtrippable(self):
        metadata = build_dependency_profile_metadata("requirements-lock-cu130.txt", "abc123")
        encoded = json.dumps(metadata)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["requirements_file"], "requirements-lock-cu130.txt")
        self.assertEqual(decoded["requirements_hash"], "abc123")
        self.assertIn("python_version", decoded)
        self.assertIn("platform", decoded)

    def test_metadata_writer_script_outputs_expected_json(self):
        tmp_dir = Path("tests/.tmp_metadata")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        req_path = tmp_dir / "requirements.txt"
        out_path = tmp_dir / ".dependencies_verified"
        req_path.write_text("transformers==4.57.3\n", encoding="utf-8")
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/write_dependency_metadata.py",
                    "--requirements",
                    str(req_path),
                    "--output",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["requirements_file"], "requirements.txt")
            self.assertIn("requirements_hash", payload)
            self.assertIn("python_version", payload)
            self.assertIn("platform", payload)
        finally:
            for child in tmp_dir.glob("*"):
                child.unlink(missing_ok=True)
            tmp_dir.rmdir()

    def test_pyproject_toml_aligns_with_lockfile_for_git_deps(self):
        """Ensure pyproject.toml uses the same commit-pinned diffusers and sdnq as lockfile."""
        lock = _parse_requirements(Path("requirements-lock-cu130.txt"))
        pyproject_content = Path("pyproject.toml").read_text(encoding="utf-8")
        pyproject = tomli.loads(pyproject_content)

        # Extract commit hashes from lockfile
        diffusers_lock = lock.get("diffusers")
        sdnq_lock = lock.get("sdnq")
        self.assertIsNotNone(diffusers_lock, "diffusers not found in lockfile")
        self.assertIsNotNone(sdnq_lock, "sdnq not found in lockfile")

        diffusers_commit = diffusers_lock.split("@")[-1]
        sdnq_commit = sdnq_lock.split("@")[-1]

        # Extract commit hashes from pyproject.toml uv.sources
        uv_sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
        diffusers_source = uv_sources.get("diffusers", {})
        sdnq_source = uv_sources.get("sdnq", {})

        self.assertIn("rev", diffusers_source, "diffusers in pyproject.toml missing commit pin")
        self.assertIn("rev", sdnq_source, "sdnq in pyproject.toml missing commit pin")

        self.assertEqual(
            diffusers_source["rev"],
            diffusers_commit,
            f"pyproject.toml diffusers commit ({diffusers_source['rev']}) does not match lockfile ({diffusers_commit})"
        )
        self.assertEqual(
            sdnq_source["rev"],
            sdnq_commit,
            f"pyproject.toml sdnq commit ({sdnq_source['rev']}) does not match lockfile ({sdnq_commit})"
        )


if __name__ == "__main__":
    unittest.main()
