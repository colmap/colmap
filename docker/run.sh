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

echo "Pulling latest COLMAP Docker image..."
docker pull colmap/colmap:latest

echo "Running COLMAP container with directory: $HOST_DIR"

# Try different GPU configurations in order of preference
echo "Testing GPU access..."

# Test modern --gpus all flag first
if docker run --rm --gpus all colmap/colmap:latest nvidia-smi >/dev/null 2>&1; then
    echo "✅ Using GPU acceleration with --gpus all"
    docker run --gpus all -w /working -v "$HOST_DIR":/working -it colmap/colmap:latest bash -c "
        echo '[INFO] GPU-enabled COLMAP container starting...'
        echo '[GPU Info]:' && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
        echo '[INFO] Container ready. GPU acceleration enabled.'
        exec bash
    "
elif docker run --rm --runtime=nvidia colmap/colmap:latest nvidia-smi >/dev/null 2>&1; then
    echo "✅ Using GPU acceleration with --runtime=nvidia"
    docker run --runtime=nvidia -w /working -v "$HOST_DIR":/working -it colmap/colmap:latest bash -c "
        echo '[INFO] GPU-enabled COLMAP container starting...'
        echo '[GPU Info]:' && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
        echo '[INFO] Container ready. GPU acceleration enabled.'
        exec bash
    "
elif docker run --rm colmap/colmap:latest colmap --help >/dev/null 2>&1; then
    echo "⚠️  GPU not available, using CPU mode"
    docker run -w /working -v "$HOST_DIR":/working -it colmap/colmap:latest bash -c "
        echo '[INFO] CPU-only COLMAP container starting...'
        echo '[WARNING] GPU acceleration disabled. Dense reconstruction will be slower.'
        exec bash
    "
else
    echo "⚠️  Container test failed, trying anyway in CPU mode"
    docker run -w /working -v "$HOST_DIR":/working -it colmap/colmap:latest
fi
