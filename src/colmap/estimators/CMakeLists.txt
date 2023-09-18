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
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

set(FOLDER_NAME "estimators")

COLMAP_ADD_LIBRARY(
    NAME colmap_estimators
    SRCS
        absolute_pose.h absolute_pose.cc
        affine_transform.h affine_transform.cc
        alignment.h alignment.cc
        bundle_adjustment.h bundle_adjustment.cc
        coordinate_frame.h coordinate_frame.cc
        cost_functions.h
        essential_matrix.h essential_matrix.cc
        euclidean_transform.h
        fundamental_matrix.h fundamental_matrix.cc
        generalized_absolute_pose.h generalized_absolute_pose.cc
        generalized_absolute_pose_coeffs.h generalized_absolute_pose_coeffs.cc
        generalized_relative_pose.h generalized_relative_pose.cc
        homography_matrix.h homography_matrix.cc
        pose.h pose.cc
        generalized_pose.h generalized_pose.cc
        similarity_transform.h
        translation_transform.h
        triangulation.h triangulation.cc
        two_view_geometry.h two_view_geometry.cc
        utils.h utils.cc
    PUBLIC_LINK_LIBS
        colmap_util
        colmap_math
        colmap_feature
        colmap_geometry
        colmap_sensor
        colmap_image
        colmap_scene
        colmap_optim
        Eigen3::Eigen
        Ceres::ceres
)

COLMAP_ADD_TEST(
    NAME absolute_pose_test
    SRCS absolute_pose_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME affine_transform_test
    SRCS affine_transform_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME alignment_test
    SRCS alignment_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME bundle_adjustment_test
    SRCS bundle_adjustment_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME coordinate_frame_test
    SRCS coordinate_frame_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME cost_functions_test
    SRCS cost_functions_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME essential_matrix_test
    SRCS essential_matrix_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME estimators_utils_test
    SRCS utils_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME fundamental_matrix_test
    SRCS fundamental_matrix_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME generalized_absolute_pose_test
    SRCS generalized_absolute_pose_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME generalized_pose_test
    SRCS generalized_pose_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME generalized_relative_pose_test
    SRCS generalized_relative_pose_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME homography_matrix_test
    SRCS homography_matrix_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME pose_test
    SRCS pose_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME similarity_transform_test
    SRCS similarity_transform_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME translation_transform_test
    SRCS translation_transform_test.cc
    LINK_LIBS colmap_estimators
)
