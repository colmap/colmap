#!/bin/bash
# Build COLMAP Docker image from the repository root.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

docker build "$REPO_ROOT" \
    -f "$REPO_ROOT/docker/Dockerfile" \
    -t colmap:latest
# In some cases, you may have to explicitly specify the compute architecture:
#   docker build . -f docker/Dockerfile -t colmap:latest --build-arg CUDA_ARCHITECTURES=75
