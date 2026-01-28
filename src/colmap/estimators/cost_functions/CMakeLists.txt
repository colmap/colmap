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

set(FOLDER_NAME "estimators_cost_functions")

COLMAP_ADD_LIBRARY(
    NAME colmap_estimators_cost_functions
    TYPE INTERFACE
    SRCS
        alignment.h
        calibration.h
        manifold.h
        motion_averaging.h
        pose_prior.h
        reprojection_error.h
        sampson_error.h
        utils.h
    INTERFACE_LINK_LIBS
        colmap_geometry
        colmap_sensor
        Eigen3::Eigen
        Ceres::ceres
)

COLMAP_ADD_TEST(
    NAME alignment_test
    SRCS alignment_test.cc
    LINK_LIBS colmap_estimators_cost_functions
)
COLMAP_ADD_TEST(
    NAME calibration_test
    SRCS calibration_test.cc
    LINK_LIBS colmap_estimators_cost_functions
)
COLMAP_ADD_TEST(
    NAME utils_test
    SRCS utils_test.cc
    LINK_LIBS colmap_estimators_cost_functions
)
COLMAP_ADD_TEST(
    NAME reprojection_error_test
    SRCS reprojection_error_test.cc
    LINK_LIBS colmap_estimators_cost_functions
)
COLMAP_ADD_TEST(
    NAME sampson_error_test
    SRCS sampson_error_test.cc
    LINK_LIBS colmap_estimators_cost_functions
)
COLMAP_ADD_TEST(
    NAME pose_prior_test
    SRCS pose_prior_test.cc
    LINK_LIBS colmap_estimators_cost_functions
)
COLMAP_ADD_TEST(
    NAME motion_averaging_test
    SRCS motion_averaging_test.cc
    LINK_LIBS colmap_estimators_cost_functions
)
