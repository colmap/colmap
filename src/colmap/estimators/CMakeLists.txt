# Copyright (c), ETH Zurich and UNC Chapel Hill.
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

set(FOLDER_NAME "estimators")

add_subdirectory(cost_functions)
add_subdirectory(solvers)

COLMAP_ADD_LIBRARY(
    NAME colmap_estimators
    SRCS
        alignment.h alignment.cc
        bundle_adjustment.h bundle_adjustment.cc
        bundle_adjustment_ceres.h bundle_adjustment_ceres.cc
        coordinate_frame.h coordinate_frame.cc
        covariance.h covariance.cc
        generalized_pose.h generalized_pose.cc
        global_positioning.h global_positioning.cc
        gravity_refinement.h gravity_refinement.cc
        pose.h pose.cc
        rotation_averaging.h rotation_averaging.cc
        rotation_averaging_impl.h rotation_averaging_impl.cc
        triangulation.h triangulation.cc
        two_view_geometry.h two_view_geometry.cc
        view_graph_calibration.h view_graph_calibration.cc
    PUBLIC_LINK_LIBS
        colmap_util
        colmap_math
        colmap_feature_types
        colmap_geometry
        colmap_sensor
        colmap_image
        colmap_scene
        colmap_optim
        colmap_estimators_cost_functions
        colmap_estimators_solvers
        Eigen3::Eigen
        Ceres::ceres
        PoseLib::PoseLib
)

if(CUDA_ENABLED)
    target_link_libraries(colmap_estimators PUBLIC colmap_util_cuda)
endif()

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
    NAME bundle_adjustment_ceres_test
    SRCS bundle_adjustment_ceres_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME coordinate_frame_test
    SRCS coordinate_frame_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME covariance_test
    SRCS covariance_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME generalized_pose_test
    SRCS generalized_pose_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME pose_test
    SRCS pose_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME two_view_geometry_test
    SRCS two_view_geometry_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME view_graph_calibration_test
    SRCS view_graph_calibration_test.cc
    LINK_LIBS colmap_estimators
)

COLMAP_ADD_TEST(
    NAME glomap_rotation_averaging_test
    SRCS rotation_averaging_test.cc
    LINK_LIBS colmap_estimators
)

COLMAP_ADD_TEST(
    NAME glomap_gravity_refinement_test
    SRCS gravity_refinement_test.cc
    LINK_LIBS colmap_estimators
)

COLMAP_ADD_TEST(
    NAME glomap_global_positioning_test
    SRCS global_positioning_test.cc
    LINK_LIBS colmap_estimators
)
