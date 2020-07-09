# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

COLMAP_ADD_SOURCES(
    absolute_pose.h absolute_pose.cc
    affine_transform.h affine_transform.cc
    coordinate_frame.h coordinate_frame.cc
    essential_matrix.h essential_matrix.cc
    euclidean_transform.h
    fundamental_matrix.h fundamental_matrix.cc
    generalized_absolute_pose.h generalized_absolute_pose.cc
    generalized_absolute_pose_coeffs.h generalized_absolute_pose_coeffs.cc
    generalized_relative_pose.h generalized_relative_pose.cc
    homography_matrix.h homography_matrix.cc
    pose.h pose.cc
    similarity_transform.h
    translation_transform.h
    triangulation.h triangulation.cc
    two_view_geometry.h two_view_geometry.cc
    utils.h utils.cc
)

COLMAP_ADD_TEST(absolute_pose_test absolute_pose_test.cc)
COLMAP_ADD_TEST(affine_transform_test affine_transform_test.cc)
COLMAP_ADD_TEST(coordinate_frame_test coordinate_frame_test.cc)
COLMAP_ADD_TEST(essential_matrix_test essential_matrix_test.cc)
COLMAP_ADD_TEST(estimators_utils_test utils_test.cc)
COLMAP_ADD_TEST(fundamental_matrix_test fundamental_matrix_test.cc)
COLMAP_ADD_TEST(generalized_absolute_pose_test generalized_absolute_pose_test.cc)
COLMAP_ADD_TEST(generalized_relative_pose_test generalized_relative_pose_test.cc)
COLMAP_ADD_TEST(homography_matrix_test homography_matrix_test.cc)
COLMAP_ADD_TEST(translation_transform_test translation_transform_test.cc)
COLMAP_ADD_TEST(two_view_geometry_test two_view_geometry_test.cc)
