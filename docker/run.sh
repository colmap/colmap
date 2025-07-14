#!/bin/bash

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <host_directory>"
    echo "Example: $0 ../dataset/"
    exit 1
fi

# Check if the provided directory exists
if [ ! -d "$1" ]; then
    echo "Error: Directory '$1' does not exist"
    exit 1
fi

# Get absolute path
HOST_DIR=$(realpath "$1")

# Check if local colmap:latest image exists (in case you ran build.sh), otherwise use official image
if docker image inspect colmap:latest >/dev/null 2>&1; then
    echo "Using local COLMAP Docker image..."
    COLMAP_IMAGE="colmap:latest"
else
    echo "Local COLMAP image not found, pulling official image..."
    docker pull colmap/colmap:latest
    COLMAP_IMAGE="colmap/colmap:latest"
fi

echo "Running COLMAP container with directory: $HOST_DIR"

# Determine GPU arguments
GPU_ARGS=""
echo "Testing GPU access..."
if docker run --rm --runtime=nvidia $COLMAP_IMAGE find /usr/local/cuda-*/targets/*/lib -name "libcudart.so*" 2>/dev/null | head -1 >/dev/null 2>&1; then
    echo "✅ Using GPU acceleration with --gpus all"
    GPU_ARGS="--runtime=nvidia"
else
    echo "⚠️  Falling back to CPU mode. Fix NVIDIA Container Toolkit for GPU support."
fi

docker run -it --rm ${GPU_ARGS} -w /working -v "$HOST_DIR":/working $COLMAP_IMAGE bash
