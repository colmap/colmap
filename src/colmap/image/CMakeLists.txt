# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "image")

COLMAP_ADD_LIBRARY(
    NAME colmap_image
    SRCS
        apriltag.h apriltag.cc
        line.h line.cc
        undistortion.h undistortion.cc
        warp.h warp.cc
    PUBLIC_LINK_LIBS
        colmap_scene
        colmap_util
        Eigen3::Eigen
    PRIVATE_LINK_LIBS
        colmap_sensor
        colmap_vlfeat
)
if(LSD_ENABLED)
    target_link_libraries(colmap_image PRIVATE colmap_lsd)
endif()
if(APRILTAG_ENABLED)
    target_link_libraries(colmap_image PRIVATE apriltag::apriltag)
endif()

COLMAP_ADD_TEST(
    NAME apriltag_test
    SRCS apriltag_test.cc
    LINK_LIBS colmap_image
)

COLMAP_ADD_TEST(
    NAME line_test
    SRCS line_test.cc
    LINK_LIBS colmap_image
)
COLMAP_ADD_TEST(
    NAME undistortion_test
    SRCS undistortion_test.cc
    LINK_LIBS colmap_image
)
COLMAP_ADD_TEST(
    NAME warp_test
    SRCS warp_test.cc
    LINK_LIBS colmap_image
)
