#!/usr/bin/env python3
"""
Create shared udocker container pool before training starts.
This runs ONCE before training, not inside the reward function.
"""

import os
import subprocess
import sys

POOL_SIZE = 32
CONTAINER_PREFIX = "code-eval-shared"
UDOCKER_IMAGE = "python:3.11-slim"
UDOCKER_EXECMODE = "P1"

def main():
    # Set udocker directory
    udocker_dir = os.environ.get("UDOCKER_DIR", f"/tmp/udocker_{os.getenv('USER', 'user')}")
    os.environ["UDOCKER_DIR"] = udocker_dir

    print("=" * 60)
    print("Creating Shared Container Pool")
    print("=" * 60)
    print(f"Storage: {udocker_dir}")
    print(f"Size: {POOL_SIZE} containers")
    print(f"Prefix: {CONTAINER_PREFIX}")
    print()

    # Check if udocker is installed
    if subprocess.run(["which", "udocker"], capture_output=True).returncode != 0:
        print("ERROR: udocker not found!")
        print("Install: pip install udocker")
        sys.exit(1)

    # Initialize udocker
    print("Initializing udocker...")
    subprocess.run(["udocker", "install"], capture_output=True)

    # Pull image if not present
    result = subprocess.run(["udocker", "images"], capture_output=True, text=True)
    if UDOCKER_IMAGE not in result.stdout:
        print(f"Pulling {UDOCKER_IMAGE} (this may take a few minutes)...")
        subprocess.run(["udocker", "pull", UDOCKER_IMAGE], check=True)
        print("✓ Image pulled")
    else:
        print(f"✓ Image {UDOCKER_IMAGE} already available")

    # Check existing containers
    result = subprocess.run(["udocker", "ps", "-a"], capture_output=True, text=True)
    existing = result.stdout

    # Create containers
    print(f"\nCreating {POOL_SIZE} containers...")
    created = 0
    reused = 0

    for i in range(POOL_SIZE):
        container_name = f"{CONTAINER_PREFIX}-{i}"

        if container_name in existing:
            print(f"  [{i:2d}] Reusing: {container_name}")
            reused += 1
            continue

        # Create container
        result = subprocess.run(
            ["udocker", "create", f"--name={container_name}", UDOCKER_IMAGE],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"  [{i:2d}] FAILED: {container_name} - {result.stderr}")
            continue

        # Set execution mode for performance
        subprocess.run(
            ["udocker", "setup", f"--execmode={UDOCKER_EXECMODE}", container_name],
            capture_output=True
        )

        print(f"  [{i:2d}] Created: {container_name} ({UDOCKER_EXECMODE} mode)")
        created += 1

    print()
    print("=" * 60)
    print(f"✓ Pool Ready: {created} created, {reused} reused")
    print("=" * 60)

    # List all containers
    print("\nContainer list:")
    subprocess.run(["udocker", "ps", "-a"])

if __name__ == "__main__":
    main()
