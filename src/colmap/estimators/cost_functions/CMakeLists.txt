# SPDX-License-Identifier: BSD-3-Clause

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
        quaternion_utils.h
        reprojection_error.h
        sampson_error.h
        tiny_manifold.h
        tiny_sampson_error.h
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
    NAME tiny_manifold_test
    SRCS tiny_manifold_test.cc
    LINK_LIBS colmap_estimators_cost_functions
)
COLMAP_ADD_TEST(
    NAME tiny_sampson_error_test
    SRCS tiny_sampson_error_test.cc
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
COLMAP_ADD_TEST(
    NAME quaternion_utils_test
    SRCS quaternion_utils_test.cc
    LINK_LIBS colmap_estimators_cost_functions colmap_math
)
