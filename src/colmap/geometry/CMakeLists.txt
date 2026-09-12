# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "geometry")

COLMAP_ADD_LIBRARY(
    NAME colmap_geometry
    SRCS
        bbox.h bbox.cc
        essential_matrix.h essential_matrix.cc
        gps.h gps.cc
        homography_matrix.h homography_matrix.cc
        normalization.h normalization.cc
        pose.h pose.cc
        pose_prior.h pose_prior.cc
        rigid3.h rigid3.cc
        sim3.h sim3.cc
        triangulation.h triangulation.cc
    PUBLIC_LINK_LIBS
        colmap_util
        colmap_math
        Eigen3::Eigen
)

if(TESTS_ENABLED)
    target_sources(
        colmap_geometry
        PRIVATE
            rigid3_matchers.h
            sim3_matchers.h
    )
endif()

COLMAP_ADD_TEST(
    NAME bbox_test
    SRCS bbox_test.cc
    LINK_LIBS colmap_geometry
)
COLMAP_ADD_TEST(
    NAME essential_matrix_test
    SRCS essential_matrix_test.cc
    LINK_LIBS colmap_geometry
)
COLMAP_ADD_TEST(
    NAME gps_test
    SRCS gps_test.cc
    LINK_LIBS colmap_geometry
)
COLMAP_ADD_TEST(
    NAME homography_matrix_test
    SRCS homography_matrix_test.cc
    LINK_LIBS colmap_geometry
)
COLMAP_ADD_TEST(
    NAME normalization_test
    SRCS normalization_test.cc
    LINK_LIBS colmap_geometry
)
COLMAP_ADD_TEST(
    NAME pose_test
    SRCS pose_test.cc
    LINK_LIBS colmap_geometry
)
COLMAP_ADD_TEST(
    NAME pose_prior_test
    SRCS pose_prior_test.cc
    LINK_LIBS colmap_geometry
)
COLMAP_ADD_TEST(
    NAME rigid3_test
    SRCS rigid3_test.cc
    LINK_LIBS colmap_geometry
)
COLMAP_ADD_TEST(
    NAME rigid3_matchers_test
    SRCS rigid3_matchers_test.cc
    LINK_LIBS colmap_geometry
)
COLMAP_ADD_TEST(
    NAME sim3_test
    SRCS sim3_test.cc
    LINK_LIBS colmap_geometry
)
COLMAP_ADD_TEST(
    NAME sim3_matchers_test
    SRCS sim3_matchers_test.cc
    LINK_LIBS colmap_geometry
)
COLMAP_ADD_TEST(
    NAME triangulation_test
    SRCS triangulation_test.cc
    LINK_LIBS colmap_geometry
)
