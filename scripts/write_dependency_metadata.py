import argparse
import hashlib
import json
import os
import platform
import sys


def compute_sha256(path: str) -> str:
    if not os.path.exists(path):
        return "missing"
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Write dependency profile metadata.")
    parser.add_argument("--requirements", required=True, help="Requirements file path.")
    parser.add_argument("--output", required=True, help="Metadata output path.")
    args = parser.parse_args()

    requirements_path = os.path.abspath(args.requirements)
    output_path = os.path.abspath(args.output)
    metadata = {
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "requirements_file": os.path.basename(requirements_path),
        "requirements_hash": compute_sha256(requirements_path),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Wrote dependency metadata to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
