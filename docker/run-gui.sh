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

# Check if local colmap:latest image exists
if docker image inspect colmap:latest >/dev/null 2>&1; then
    echo "Using local COLMAP Docker image..."
    COLMAP_IMAGE="colmap:latest"
else
    echo "Local COLMAP image not found, pulling official image..."
    docker pull colmap/colmap:latest
    COLMAP_IMAGE="colmap/colmap:latest"
fi

echo "Running COLMAP container with directory: $HOST_DIR"

# Allow local connections to the X server (for X11 compatibility)
xhost +local:root > /dev/null

# Determine GPU arguments
GPU_ARGS=""
echo "Testing GPU access..."
if docker run --rm --runtime=nvidia $COLMAP_IMAGE find /usr/local/cuda-*/targets/*/lib -name "libcudart.so*" 2>/dev/null | head -1 >/dev/null 2>&1; then
    echo "✅ Using GPU acceleration with --gpus all"
    GPU_ARGS="--runtime=nvidia"
else
    echo "⚠️  Falling back to CPU mode. Fix NVIDIA Container Toolkit for GPU support."
fi

# --- FIX: Use host networking for robust display connection ---
echo "Launching GUI..."
docker run \
    -it --rm \
    ${GPU_ARGS} \
    --net=host \
    -e DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e __NV_PRIME_RENDER_OFFLOAD=1 \
    -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    -w /working \
    -v "$HOST_DIR":/working \
    $COLMAP_IMAGE \
    colmap gui

# Revoke permission for security
xhost -local:root > /dev/null