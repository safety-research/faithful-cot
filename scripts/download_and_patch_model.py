#!/usr/bin/env python3
"""
Download a HuggingFace model and optionally patch config fields.

Usage:
    python scripts/download_and_patch_model.py \
        --repo-id Qwen/Qwen2.5-Math-7B \
        --cache-dir /workspace-vast/pretrained_ckpts \
        --patch max_position_embeddings=32768
"""

import argparse
import json
import os
from pathlib import Path


def download_model(repo_id: str, cache_dir: str, token: str | None) -> Path:
    from huggingface_hub import snapshot_download
    print(f"Downloading {repo_id} to {cache_dir} ...")
    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        token=token,
    )
    print(f"Downloaded to: {path}")
    return Path(path)


def patch_config(model_path: Path, patches: dict):
    config_path = model_path / "config.json"
    if not config_path.exists():
        print(f"WARNING: config.json not found at {config_path}")
        return

    with open(config_path) as f:
        config = json.load(f)

    print(f"\nPatching {config_path}:")
    for key, value in patches.items():
        old = config.get(key, "<not set>")
        config[key] = value
        print(f"  {key}: {old} -> {value}")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("Config patched.")


def parse_patch(patch_str: str):
    """Parse 'key=value' into (key, typed_value)."""
    key, _, raw = patch_str.partition("=")
    # Try int, then float, then bool, then string
    for cast in (int, float):
        try:
            return key.strip(), cast(raw.strip())
        except ValueError:
            pass
    if raw.strip().lower() in ("true", "false"):
        return key.strip(), raw.strip().lower() == "true"
    return key.strip(), raw.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID")
    parser.add_argument("--cache-dir", default="/workspace-vast/pretrained_ckpts")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--patch",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Patch config.json field (can be repeated)",
    )
    args = parser.parse_args()

    model_path = download_model(args.repo_id, args.cache_dir, args.token)

    if args.patch:
        patches = dict(parse_patch(p) for p in args.patch)
        patch_config(model_path, patches)
    else:
        print("\nNo patches specified, skipping config patch.")

    print("\nDone.")


if __name__ == "__main__":
    main()
