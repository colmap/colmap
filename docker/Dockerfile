FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

# Prevent stop building ubuntu at time zone selection.  
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building.
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev

# Build and install ceres solver.
RUN apt-get -y install \
    libatlas-base-dev \
    libsuitesparse-dev
ARG CERES_SOLVER_VERSION=2.1.0
RUN git clone https://github.com/ceres-solver/ceres-solver.git --branch ${CERES_SOLVER_VERSION}
RUN cd ceres-solver && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -GNinja && \
    ninja && \
    ninja install && \
    cd .. && rm -rf ceres-solver

# Build and install COLMAP.
ARG COLMAP_VERSION=dev
RUN git clone https://github.com/colmap/colmap.git && git checkout ${COLMAP_VERSION}
RUN cd colmap && \
    git checkout dev && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja && \
    ninja && \
    ninja install && \
    cd .. && rm -rf colmap
