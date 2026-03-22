# syntax=docker/dockerfile:1

ARG UBUNTU_VERSION=24.04
ARG NVIDIA_CUDA_VERSION=12.9.1
ARG CUDA_MAJOR=12
ARG CUDA_MINOR=9

#
# Docker builder stage.
#
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG CUDA_MAJOR
ARG CUDA_MINOR
ARG CUDA_ARCHITECTURES=all-major
ENV QT_XCB_GL_INTEGRATION=xcb_egl
ARG FETCHCONTENT_FULLY_DISCONNECTED=OFF
ENV CCACHE_DIR=/colmap/build/.ccache
ENV CCACHE_BASEDIR=/colmap

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building.
RUN apt-get update && \
    apt-get install -y \
        ccache \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libopenimageio-dev \
        openimageio-tools \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qt6-base-dev \
        libqt6opengl6-dev \
        libqt6openglwidgets6 \
        libcgal-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        libopenblas-dev \
        # Ceres dependencies
        libsuitesparse-dev \
        liblapack-dev \
        curl \
        git && \
        # Pycolmap
        python3 \
        python3-pip \
        python3-dev && \
        # cuDNN (required by ONNX Runtime CUDA provider for ALIKED/LightGlue)
        libcudnn9-dev-cuda-${CUDA_MAJOR} \
        # cuBLAS (needed at link time by cuDSS; not included in the cuda:devel base image)
        libcublas-dev-${CUDA_MAJOR}-${CUDA_MINOR} \
        libcublas-${CUDA_MAJOR}-${CUDA_MINOR} && \
    rm -rf /var/lib/apt/lists/*

# Install cuDSS (CUDA sparse solver) from NVIDIA's local repo installer.
ARG CUDSS_VERSION=0.7.1
RUN cd /tmp && \
    curl -LO "https://developer.download.nvidia.com/compute/cudss/${CUDSS_VERSION}/local_installers/cudss-local-repo-ubuntu2404-${CUDSS_VERSION}_${CUDSS_VERSION}-1_amd64.deb" && \
    dpkg -i cudss-local-repo-ubuntu2404-${CUDSS_VERSION}_${CUDSS_VERSION}-1_amd64.deb && \
    cp /var/cudss-local-repo-ubuntu2404-${CUDSS_VERSION}/cudss-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y cudss-cuda-${CUDA_MAJOR} libcudss0-cuda-${CUDA_MAJOR} libcudss0-dev-cuda-${CUDA_MAJOR} libcudss0-static-cuda-${CUDA_MAJOR} --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/cudss-local-repo-ubuntu2404-${CUDSS_VERSION}_${CUDSS_VERSION}-1_amd64.deb

# Build Ceres from source with CUDA + cuDSS support.
# 2.3.0 (main branch, unreleased) is required for cuDSS sparse solver support;
# the latest release tag (2.2.0) only satisfies the CUDA dense solver check.
ARG CERES_GIT_REF=master
RUN CUDSS_CMAKE_DIR=$(find /usr/lib /usr/local/lib -name "cudss-config.cmake" -o -name "cudssConfig.cmake" 2>/dev/null | tail -1 | xargs -r dirname) && \
    git clone --recurse-submodules --branch ${CERES_GIT_REF} https://github.com/ceres-solver/ceres-solver.git /tmp/ceres-solver && \
    cmake -S /tmp/ceres-solver -B /tmp/ceres-build \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DUSE_CUDA=ON \
    ${CUDSS_CMAKE_DIR:+-Dcudss_DIR="${CUDSS_CMAKE_DIR}"} \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_BENCHMARKS=OFF && \
    ninja -C /tmp/ceres-build install && \
    grep -q "find_dependency(CUDAToolkit" /usr/local/lib/cmake/Ceres/CeresConfig.cmake || \                                                                                                                                                           
            { echo "FATAL: Ceres was built without CUDA support"; exit 1; } && \                                                                                                                                                                          
    grep -q "find_dependency(cudss" /usr/local/lib/cmake/Ceres/CeresConfig.cmake || \                                                                                                                                                                 
            { echo "FATAL: Ceres was built without cuDSS support"; exit 1; } && \
    rm -rf /tmp/ceres-solver /tmp/ceres-build

# Fix issue in Ubuntu's openimageio CMake config.
# We don't depend on any of openimageio's OpenCV functionality,
# but it still requires the OpenCV include directory to exist.
RUN mkdir -p /usr/include/opencv4

# Copy source into the image.
COPY colmap/ /colmap

# Build and install COLMAP.
RUN cd /colmap && \
    mkdir -p build/.ccache && \
    cd build && \
    CUDSS_CMAKE_DIR=$(find /usr/lib /usr/local/lib -name "cudss-config.cmake" -o -name "cudssConfig.cmake" 2>/dev/null | tail -1 | xargs -r dirname) && \
    cmake .. \
        -GNinja \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        ${CUDSS_CMAKE_DIR:+-Dcudss_DIR="${CUDSS_CMAKE_DIR}"} \
        -DCMAKE_INSTALL_PREFIX=/colmap-install \
        -DFETCHCONTENT_FULLY_DISCONNECTED=${FETCHCONTENT_FULLY_DISCONNECTED} && \
    ninja install

RUN pip3 install --no-cache-dir --break-system-packages \
    scikit-build-core pybind11 pybind11-stubgen ruff numpy

RUN cd /colmap && \
    colmap_DIR=/colmap-install \
    CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
    pip3 wheel --no-deps --no-build-isolation \
    --wheel-dir /pycolmap-dist \
    .

#
# Stage to export build caches for CI round-tripping via actions/cache.
#
FROM scratch AS cache-export
COPY --from=builder /colmap/build/.ccache/ /.ccache/
COPY --from=builder /colmap/build/_deps/ /_deps/

#
# Docker runtime stage.
#
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG CUDA_MAJOR
ARG CUDA_MINOR

RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        libboost-program-options1.83.0 \
        libc6 \
        libomp5 \
        libopengl0 \
        libmetis5 \
        libopenimageio2.4t64 \
        libgcc-s1 \
        libgl1 \
        libglew2.2 \
        libgoogle-glog0v6t64 \
        libqt6core6 \
        libqt6gui6 \
        libqt6widgets6 \
        libqt6openglwidgets6 \
        libcurl4 \
        libssl3t64 \
        libcudnn9-cuda-${CUDA_MAJOR} \
        libopenblas0 \
        libcublas-${CUDA_MAJOR}-${CUDA_MINOR} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy all files from /colmap-install/ in the builder stage to /usr/local/ in
# the runtime stage. This simulates installing COLMAP in the default location
# (/usr/local/), which simplifies environment variables. It also allows the user
# of this Docker image to use it as a base image for compiling against COLMAP as
# a library. For instance, CMake will be able to find COLMAP easily with the
# command: find_package(COLMAP REQUIRED).
COPY --from=builder /usr/local/lib/libceres* /usr/local/lib/
COPY --from=builder /usr/local/lib/cmake/Ceres/ /usr/local/lib/cmake/Ceres/
COPY --from=builder /usr/local/include/ceres/ /usr/local/include/ceres/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcholmod* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/liblapack* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libblas* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcudss* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /colmap-install/ /usr/local/

# Bring the pre-built pycolmap wheel from the builder stage.
COPY --from=builder /pycolmap-dist/ /pycolmap-dist/
