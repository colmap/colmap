#!/bin/bash
#
# A robust script to run the COLMAP GUI inside a Docker container,
# with automatic GPU detection and flexible argument passing.

# --- Initial Checks ---
# Exit immediately if a command exits with a non-zero status.
set -e

# --- Cleanup Function and Trap ---
# This function will be called automatically when the script exits
# to ensure that X server permissions are always restored.
function cleanup {
    echo "Revoking X-server access..."
    # The '|| true' prevents the script from failing if xhost has issues.
    xhost -local:root > /dev/null || true
}
trap cleanup EXIT

# Check if any argument is provided.
if [ $# -eq 0 ]; then
    echo "Usage: $0 <host_directory>"
    echo "Example: $0 ../dataset/"
    exit 1
fi

# --- Docker Image Selection ---
# Check if local colmap:latest image exists (in case you ran build.sh), otherwise use official image
if docker image inspect colmap:latest >/dev/null 2>&1; then
    echo "Using local COLMAP Docker image..."
    COLMAP_IMAGE="colmap:latest"
else
    echo "Local COLMAP image not found, pulling official image..."
    docker pull colmap/colmap:latest
    COLMAP_IMAGE="colmap/colmap:latest"
fi

# Get absolute path
HOST_DIR=$(realpath "$1")
if [ ! -d "$HOST_DIR" ]; then
    echo "Error: Directory '$HOST_DIR' does not exist."
    exit 1
fi
echo "Running COLMAP container with directory: $HOST_DIR"


# --- Build Docker Arguments ---
# Start with the base arguments for GUI forwarding.
DOCKER_ARGS=(
    -it --rm
    --net=host
    -e DISPLAY
    -v "${HOST_DIR}:/working"
    -w /working
)

# --- GPU Detection and Configuration ---
echo "Testing for GPU acceleration..."
# A successful `nvidia-smi` call is the most reliable test.
if docker run --rm --runtime=nvidia "${COLMAP_IMAGE}" nvidia-smi >/dev/null 2>&1; then
    echo "✅ GPU detected. Using --runtime=nvidia and mapping all graphics devices."
    # Use the nvidia runtime AND also pass through the host's render devices.
    # This can solve driver conflicts on hybrid graphics systems.
    DOCKER_ARGS+=(
        --runtime=nvidia
        -e NVIDIA_DRIVER_CAPABILITIES=all
    )
    if [ -d /dev/dri ]; then
        DOCKER_ARGS+=( --device=/dev/dri )
    fi
else
    echo "⚠️  GPU not detected. Falling back to CPU rendering."
    # For CPU mode, we give the container access to the host's render devices.
    if [ -d /dev/dri ]; then
        DOCKER_ARGS+=( --device=/dev/dri )
    fi
fi

# --- X11 Forwarding Security ---
# Grant permissions just before running the container.
xhost +local:root > /dev/null

# --- Execute the Container ---
# Pass all arguments after the directory path ("${@:2}") to the colmap gui command.
echo "Launching GUI..."
# "${@:2}" accepts extra arguments for the COLMAP GUI.
docker run "${DOCKER_ARGS[@]}" "${COLMAP_IMAGE}" colmap gui "${@:2}"

# The `trap` will automatically call the `cleanup` function here,
