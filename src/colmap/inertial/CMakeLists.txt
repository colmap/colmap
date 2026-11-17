# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "inertial")

COLMAP_ADD_LIBRARY(
    NAME colmap_inertial
    SRCS
        preintegration.h preintegration.cc
        preintegration_cost.h
    PUBLIC_LINK_LIBS
        colmap_util
        colmap_geometry
        colmap_sensor
        colmap_estimators_cost_functions
        Eigen3::Eigen
        Ceres::ceres
)

COLMAP_ADD_TEST(
    NAME preintegration_test
    SRCS preintegration_test.cc
    LINK_LIBS colmap_inertial
)
COLMAP_ADD_TEST(
    NAME preintegration_cost_test
    SRCS preintegration_cost_test.cc
    LINK_LIBS colmap_inertial
)
