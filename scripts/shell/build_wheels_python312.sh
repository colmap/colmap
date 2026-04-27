#!/bin/bash
# Build pycolmap wheels for Python 3.12 against CUDA 13.0 and CUDA 12.8.
#
# Each CUDA variant gets its own COLMAP C++ build + install tree, then a
# separate Python wheel.  Output wheels land in polybee_colmap/dist/ and are
# renamed to include the CUDA version, e.g.:
#   pycolmap-4.1.0.dev0-cp312-cp312-linux_x86_64-cuda13.0.whl
#
# Usage:
#   ./scripts/shell/build_wheels_python312.sh [CUDA_ARCH]
#
# CUDA_ARCH defaults to 89 (Ada Lovelace / RTX 40xx).
# Override for other GPUs, e.g. 80 (Ampere), 86, 75 (Turing).

set -euo pipefail

CUDA_ARCH="${1:-89}"
PYTHON="python3.12"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST_DIR="$REPO_ROOT/dist"

mkdir -p "$DIST_DIR"

# ---------------------------------------------------------------------------
# Helper: build COLMAP C++ and then the Python wheel for one CUDA version.
# ---------------------------------------------------------------------------
build_for_cuda() {
    local CUDA_VERSION="$1"           # e.g. "13.0" or "12.8"
    local NVCC_PATH="$2"              # e.g. /usr/local/cuda-13.0/bin/nvcc

    if [ ! -x "$NVCC_PATH" ]; then
        echo "[SKIP] nvcc not found at $NVCC_PATH — skipping CUDA $CUDA_VERSION"
        return
    fi

    local CUDA_LABEL="cuda${CUDA_VERSION}"
    local BUILD_DIR="$REPO_ROOT/build/$CUDA_LABEL"
    local INSTALL_DIR="$REPO_ROOT/install/$CUDA_LABEL"

    echo ""
    echo "========================================================"
    echo " Building COLMAP C++ for CUDA $CUDA_VERSION"
    echo "========================================================"
    mkdir -p "$BUILD_DIR"
    cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_CUDA_COMPILER="$NVCC_PATH" \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DBLA_VENDOR=Intel10_64lp
    ninja -j1 -C "$BUILD_DIR" install

    echo ""
    echo "========================================================"
    echo " Building pycolmap wheel for CUDA $CUDA_VERSION / Python 3.12"
    echo "========================================================"
    local WHEEL_BUILD_DIR="$REPO_ROOT/build/wheel-$CUDA_LABEL"
    mkdir -p "$WHEEL_BUILD_DIR"

    CMAKE_ARGS="\
-Dcolmap_DIR=$INSTALL_DIR/share/colmap \
-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH \
-DCMAKE_CUDA_COMPILER=$NVCC_PATH" \
    LD_LIBRARY_PATH="$INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}" \
    "$PYTHON" -m pip wheel \
        --no-deps \
        --wheel-dir "$WHEEL_BUILD_DIR" \
        "$REPO_ROOT"

    # Rename the wheel to embed the CUDA version.
    local WHEEL
    WHEEL=$(ls "$WHEEL_BUILD_DIR"/*.whl | head -1)
    local WHEEL_BASENAME
    WHEEL_BASENAME="$(basename "$WHEEL" .whl)-$CUDA_LABEL.whl"
    cp "$WHEEL" "$DIST_DIR/$WHEEL_BASENAME"
    echo ""
    echo "Wheel written to: $DIST_DIR/$WHEEL_BASENAME"
}

# ---------------------------------------------------------------------------
# Build for each CUDA version.
# ---------------------------------------------------------------------------
build_for_cuda "13.0" "/usr/local/cuda-13.0/bin/nvcc"
# build_for_cuda "12.8" "/usr/local/cuda-12.8/bin/nvcc"

echo ""
echo "========================================================"
echo " Done. Wheels in $DIST_DIR:"
ls "$DIST_DIR"/*.whl 2>/dev/null || echo "  (none found)"
echo "========================================================"
