#!/usr/bin/env python3
"""Run COLMAP bundle_adjuster with database position and rotation priors."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the COLMAP bundle_adjuster with covariance-weighted position "
            "and quaternion rotation priors from pose_priors."
        )
    )
    parser.add_argument("--database_path", type=Path, required=True)
    parser.add_argument(
        "--colmap_path",
        default="colmap",
        help="COLMAP executable or command name (default: colmap)",
    )
    args, bundle_adjuster_args = parser.parse_known_args()

    if not args.database_path.is_file():
        parser.error(f"database does not exist: {args.database_path}")

    if "--input_path" not in bundle_adjuster_args:
        parser.error("--input_path must be passed to bundle_adjuster")
    if "--output_path" not in bundle_adjuster_args:
        parser.error("--output_path must be passed to bundle_adjuster")

    env = os.environ.copy()
    env["COLMAP_BUNDLE_ADJUSTER_DATABASE_PATH"] = str(
        args.database_path.resolve()
    )
    command = [args.colmap_path, "bundle_adjuster", *bundle_adjuster_args]
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
