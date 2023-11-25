# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


set(FOLDER_NAME "geometry")

COLMAP_ADD_LIBRARY(
    NAME colmap_geometry
    SRCS
        essential_matrix.h essential_matrix.cc
        gps.h gps.cc
        homography_matrix.h homography_matrix.cc
        pose.h pose.cc
        rigid3.h
        sim3.h sim3.cc
        triangulation.h triangulation.cc
    PUBLIC_LINK_LIBS
        colmap_util
        colmap_math
        Eigen3::Eigen
)

COLMAP_ADD_TEST(
    NAME essential_matrix_utils_test
    SRCS essential_matrix_test.cc
    LINK_LIBS colmap_geometry
)
COLMAP_ADD_TEST(
    NAME gps_test
    SRCS gps_test.cc
    LINK_LIBS colmap_geometry
)
COLMAP_ADD_TEST(
    NAME homography_matrix_utils_test
    SRCS homography_matrix_test.cc
    LINK_LIBS colmap_geometry
)
COLMAP_ADD_TEST(
    NAME pose_test
    SRCS pose_test.cc
    LINK_LIBS colmap_geometry
)
COLMAP_ADD_TEST(
    NAME rigid3_test
    SRCS rigid3_test.cc
    LINK_LIBS colmap_geometry
)
COLMAP_ADD_TEST(
    NAME sim3_test
    SRCS sim3_test.cc
    LINK_LIBS colmap_geometry
)
COLMAP_ADD_TEST(
    NAME triangulation_test
    SRCS triangulation_test.cc
    LINK_LIBS colmap_geometry
)
