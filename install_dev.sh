#!/bin/bash
#
# Build and install this COLMAP fork into the system (/usr/local).
# Extracted from medida-3d's old setup.sh (pre private-wheels era).
#
# Builds Eigen + Ceres into a local prefix under TMP_BUILD_PATH, then
# configures/builds COLMAP against those and runs `sudo cmake --install`.
#
# Tested historically on Mac M3 Pro with Homebrew; Linux apt path included.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TMP_BUILD_PATH="${TMP_BUILD_PATH:-${ROOT_DIR}/tmp}"

# Step 1: Install dependencies
if [ "$(uname -s)" == "Linux" ]; then
    sudo apt-get install -y \
        cmake \
        ninja-build \
        libflann-dev \
        python3-pybind11 \
        libboost-all-dev \
        libglew-dev \
        libcgal-dev \
        libmetis-dev \
        libfreeimage-dev \
        libopencv-dev
    # Note: libceres-dev removed - we build from source to match Eigen version
elif [ "$(uname -s)" == "Darwin" ]; then
    brew install \
        cmake \
        ninja \
        pybind11 \
        flann \
        boost \
        glew \
        cgal \
        metis \
        freeimage \
        opencv
else
    echo "Unsupported platform"
    exit 1
fi

mkdir -p "$TMP_BUILD_PATH"
cd "$TMP_BUILD_PATH"

if [ "$(uname -s)" == "Linux" ]; then
    NPROC=$(nproc)
else
    NPROC=$(sysctl -n hw.ncpu)
fi

# Local install prefixes - nothing gets installed to the system except final COLMAP
LOCAL_PREFIX="$(pwd)/install"
EIGEN_INSTALL_PREFIX="${LOCAL_PREFIX}/eigen"
EIGEN3_CMAKE_DIR="${EIGEN_INSTALL_PREFIX}/share/eigen3/cmake"
CERES_INSTALL_PREFIX="${LOCAL_PREFIX}/ceres"
CERES_CMAKE_DIR="${CERES_INSTALL_PREFIX}/lib/cmake/Ceres"

# Step 1a: Build Eigen from source into local prefix
if [ ! -d "eigen" ]; then
    git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git
fi
cd eigen
cmake -B build \
    -DCMAKE_INSTALL_PREFIX="$EIGEN_INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build
cd ..

# Step 2: Build Ceres from source into local prefix (static, to avoid runtime deps)
if [ ! -d "ceres-solver" ]; then
    git clone --recurse-submodules https://github.com/ceres-solver/ceres-solver.git
fi
cd ceres-solver
git checkout 2.2.0
git submodule update --init --recursive
cmake -B _build . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$CERES_INSTALL_PREFIX" \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DEigen3_DIR="$EIGEN3_CMAKE_DIR"
cmake --build _build -j"$NPROC"
cmake --install _build
cd ..

# Step 3: Build and install this COLMAP tree into the system
cd "$ROOT_DIR"

if [ "$(uname -s)" == "Darwin" ]; then
    export QT_DIR="$(brew --prefix qt@5)/lib/cmake/Qt5"
    export Qt5_DIR="$(brew --prefix qt@5)/lib/cmake/Qt5"
fi
export Python_EXECUTABLE="$(which python3)"

cmake -B build/ . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ENABLED=OFF \
    -DGUI_ENABLED=OFF \
    -DEigen3_DIR="$EIGEN3_CMAKE_DIR" \
    -DEIGEN3_ROOT_DIR="$EIGEN_INSTALL_PREFIX" \
    -DEIGEN3_INCLUDE_DIR="${EIGEN_INSTALL_PREFIX}/include/eigen3" \
    -DCeres_DIR="$CERES_CMAKE_DIR" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1
cmake --build build/ -j"$NPROC"
sudo cmake --install build/

COLMAP_CMAKE_DIR="/usr/local/share/colmap"
if [ ! -f "$COLMAP_CMAKE_DIR/colmap-config.cmake" ]; then
    echo "ERROR: expected fork's colmap-config.cmake at $COLMAP_CMAKE_DIR — did 'sudo cmake --install build/' succeed?"
    exit 1
fi

echo "Installed COLMAP to /usr/local (colmap-config.cmake at $COLMAP_CMAKE_DIR)"
