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
if docker run --rm --runtime=nvidia colmap/colmap:latest colmap --help >/dev/null 2>&1; then
    echo "✅ Using GPU acceleration with --runtime=nvidia"
    docker run --runtime=nvidia -w /working -v "$HOST_DIR":/working -it colmap/colmap:latest
elif docker run --rm colmap/colmap:latest colmap --help >/dev/null 2>&1; then
    echo "✅ Using default runtime (may include GPU if configured)"
    docker run -w /working -v "$HOST_DIR":/working -it colmap/colmap:latest
else
    echo "⚠️  Container test failed, trying anyway in CPU mode"
    docker run -w /working -v "$HOST_DIR":/working -it colmap/colmap:latest
fi
