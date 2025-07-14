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

# Try different GPU configurations in order of preference
echo "Testing GPU access..."

# Test modern --runtime=nvidia flag first
if docker run --rm --runtime=nvidia $COLMAP_IMAGE find /usr/local/cuda-*/targets/*/lib -name "libcudart.so*" 2>/dev/null | head -1 >/dev/null 2>&1; then
    echo "✅ Using GPU acceleration with --runtime=nvidia"
    docker run --runtime=nvidia \
    --privileged \
    -e QT_XCB_GL_INTEGRATION=xcb_egl \
    -e DISPLAY=:0 \
    -w /working \
    -v "$HOST_DIR":/working \
    -it $COLMAP_IMAGE \
    colmap gui
else echo "⚠️  GPU not available, using CPU mode"
    docker run -e QT_XCB_GL_INTEGRATION=xcb_egl \
    -e DISPLAY=:0 \
    -w /working \
    -v "$HOST_DIR":/working \
    -it $COLMAP_IMAGE \
    colmap gui
    docker run -e QT_XCB_GL_INTEGRATION=xcb_egl \
    -e DISPLAY=:0 \
    -w /working \
    -v "$HOST_DIR":/working \
    -it $COLMAP_IMAGE \
    colmap gui
fi