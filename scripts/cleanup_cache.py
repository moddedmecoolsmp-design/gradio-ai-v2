import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path


def resolve_env_path(name, default_path):
    value = os.environ.get(name)
    return Path(value) if value else Path(default_path)


def remove_path(path: Path, label: str, dry_run: bool) -> int:
    if not path.exists():
        return 0
    if dry_run:
        print(f"[dry-run] Would remove {label}: {path}")
        return 0
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
        else:
            shutil.rmtree(path)
        print(f"Removed {label}: {path}")
        return 1
    except Exception as exc:
        print(f"Failed to remove {label}: {path} ({exc})")
        return 0


def remove_pycache(root: Path, dry_run: bool, include_venv: bool) -> int:
    removed = 0
    for path in root.rglob("__pycache__"):
        if not include_venv and "venv" in path.parts:
            continue
        removed += remove_path(path, "__pycache__", dry_run)
    return removed


def remove_incomplete_files(cache_root: Path, dry_run: bool) -> int:
    removed = 0
    if not cache_root.exists():
        return removed
    for path in cache_root.rglob("*.incomplete"):
        removed += remove_path(path, "incomplete download", dry_run)
    return removed


def remove_stale_lfs_tmp(repo_root: Path, dry_run: bool, max_age_hours: int = 24) -> int:
    removed = 0
    lfs_tmp = repo_root / ".git" / "lfs" / "tmp"
    if not lfs_tmp.exists():
        return removed

    cutoff = time.time() - (max_age_hours * 3600)
    for path in lfs_tmp.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime <= cutoff:
                removed += remove_path(path, "stale git-lfs tmp", dry_run)
        except OSError:
            continue
    return removed


def run_command(cmd, cwd: Path, dry_run: bool, label: str) -> int:
    text = " ".join(cmd)
    if dry_run:
        print(f"[dry-run] Would run {label}: {text}")
        return 0
    try:
        result = subprocess.run(cmd, cwd=str(cwd), check=False)
        if result.returncode == 0:
            print(f"Completed {label}: {text}")
            return 1
        print(f"Failed {label} ({result.returncode}): {text}")
        return 0
    except Exception as exc:
        print(f"Failed {label}: {text} ({exc})")
        return 0


def prune_hf_cache_unreferenced(cache_dir: Path, dry_run: bool) -> int:
    try:
        from huggingface_hub import scan_cache_dir
    except Exception as exc:
        print(f"Failed huggingface cache prune import: {exc}")
        return 0

    if not cache_dir.exists():
        return 0

    info = scan_cache_dir(cache_dir)
    revision_hashes = []
    for repo in info.repos:
        for rev in repo.revisions:
            if not rev.refs:
                revision_hashes.append(rev.commit_hash)

    if not revision_hashes:
        print("No unreferenced Hugging Face revisions to prune.")
        return 0

    strategy = info.delete_revisions(*revision_hashes)
    freed_gb = strategy.expected_freed_size / (1024 ** 3)
    if dry_run:
        print(f"[dry-run] Would prune {len(revision_hashes)} Hugging Face revision(s) (~{freed_gb:.2f} GB).")
        return 0

    strategy.execute()
    print(f"Pruned {len(revision_hashes)} Hugging Face revision(s), freed ~{freed_gb:.2f} GB.")
    return 1


def main():
    parser = argparse.ArgumentParser(description="Clean caches and large temporary artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    parser.add_argument(
        "--mode",
        choices=["safe", "aggressive"],
        default="safe",
        help="Cleanup intensity level",
    )
    parser.add_argument(
        "--include-state",
        action="store_true",
        help="Also remove user_state caches (input images, refs, ui_state)",
    )
    parser.add_argument(
        "--include-output",
        action="store_true",
        help="Also remove the default output directory under the project",
    )
    parser.add_argument(
        "--include-venv-pycache",
        action="store_true",
        help="Also remove __pycache__ inside venv directories",
    )
    parser.add_argument(
        "--hf-prune",
        action="store_true",
        help="Prune unreferenced Hugging Face cache revisions",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cache_root = resolve_env_path("UFIG_CACHE_DIR", repo_root / "cache")
    hf_home = resolve_env_path("HF_HOME", cache_root / "huggingface")
    hf_hub_cache = resolve_env_path("HF_HUB_CACHE", hf_home / "hub")
    hf_xet_cache = resolve_env_path("HF_XET_CACHE", hf_home / "xet")
    hf_assets_cache = resolve_env_path("HF_ASSETS_CACHE", hf_home / "assets")
    transformers_cache = resolve_env_path("TRANSFORMERS_CACHE", hf_home / "transformers")
    torch_home = resolve_env_path("TORCH_HOME", cache_root / "torch")
    gradio_temp_dir = resolve_env_path("GRADIO_TEMP_DIR", cache_root / "gradio")

    removed_count = 0

    if args.mode == "safe":
        targets = [
            ("Gradio temp cache", gradio_temp_dir),
            ("Torch cache", torch_home),
        ]
    else:
        targets = [
            ("Gradio temp cache", gradio_temp_dir),
            ("Hugging Face hub cache", hf_hub_cache),
            ("Hugging Face xet cache", hf_xet_cache),
            ("Hugging Face assets cache", hf_assets_cache),
            ("Transformers cache", transformers_cache),
            ("Torch cache", torch_home),
        ]

    for label, path in targets:
        removed_count += remove_path(path, label, args.dry_run)

    removed_count += remove_pycache(repo_root, args.dry_run, args.include_venv_pycache)
    removed_count += remove_incomplete_files(cache_root, args.dry_run)
    removed_count += remove_stale_lfs_tmp(repo_root, args.dry_run, max_age_hours=24)

    if args.include_state:
        state_dir = repo_root / "user_state"
        removed_count += remove_path(state_dir / "input_images", "user_state input images", args.dry_run)
        removed_count += remove_path(state_dir / "character_references", "user_state character refs", args.dry_run)
        removed_count += remove_path(state_dir / "ui_state.json", "user_state ui_state.json", args.dry_run)

    if args.include_output:
        removed_count += remove_path(repo_root / "output", "output directory", args.dry_run)

    if args.mode == "aggressive":
        removed_count += run_command(["git", "lfs", "prune"], repo_root, args.dry_run, "git lfs prune")
        removed_count += run_command(
            ["git", "reflog", "expire", "--expire=now", "--all"],
            repo_root,
            args.dry_run,
            "git reflog expire",
        )
        removed_count += run_command(
            ["git", "gc", "--prune=now", "--aggressive"],
            repo_root,
            args.dry_run,
            "git gc",
        )

    if args.hf_prune:
        removed_count += prune_hf_cache_unreferenced(hf_hub_cache, args.dry_run)

    if not args.dry_run:
        try:
            cache_root.rmdir()
        except Exception:
            pass

    print(f"Cleanup complete. Removed/ran {removed_count} target(s).")


if __name__ == "__main__":
    main()
