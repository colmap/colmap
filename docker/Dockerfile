FROM ubuntu:18.04

# Prepare and empty machine for building:
RUN apt-get update -yq
RUN apt-get install -yq
RUN apt-get -y install git cmake vim
RUN apt-get -y install \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev

# CGAL
RUN apt-get -y install libcgal-qt5-dev

# CERES SOLVER
RUN apt-get -y install libatlas-base-dev libsuitesparse-dev
RUN git clone https://github.com/ceres-solver/ceres-solver.git --branch 1.14.0
RUN cd ceres-solver &&\
	mkdir build &&\
	cd build &&\
	cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF &&\
	make -j4 &&\
	make install

# COLMAP INSTALL

RUN git clone https://github.com/colmap/colmap.git --branch 3.6-dev.3

# Uncomment below (and comment the above) to use the latest default branch commit instead
# RUN git clone https://github.com/colmap/colmap.git

RUN cd colmap &&\
	git checkout dev &&\
	mkdir build &&\
	cd build &&\
	cmake .. &&\
	make -j4 &&\
	make install
