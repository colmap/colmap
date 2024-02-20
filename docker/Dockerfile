ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=12.3.1

# Docker builder stage.
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder

ARG COLMAP_GIT_COMMIT=unset
ARG CUDA_ARCHITECTURES=native
ENV QT_XCB_GL_INTEGRATION=xcb_egl

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev

# Dynamically determine COLMAP_GIT_COMMIT if not set by user, log the commit being built, and build COLMAP.
RUN git clone https://github.com/colmap/colmap.git && \
    cd colmap && \
    git fetch --tags && \
    COLMAP_GIT_COMMIT=$(if [ "$COLMAP_GIT_COMMIT" = "unset" ]; then git describe --tags `git rev-list --tags --max-count=1`; else echo $COLMAP_GIT_COMMIT; fi) && \
    echo "Building COLMAP at commit/tag: $COLMAP_GIT_COMMIT" && \
    git checkout $COLMAP_GIT_COMMIT && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_INSTALL_PREFIX=/colmap_installed && \
    ninja install

