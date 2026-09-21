# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "sensor")

COLMAP_ADD_LIBRARY(
    NAME colmap_sensor
    SRCS
        bitmap.h bitmap.cc
        database.h database.cc
        models.h models.cc
        models/base.h
        models/division.h
        models/eucm.h
        models/fisheye.h
        models/fisheye_radial.h
        models/fov.h
        models/jacobian.h
        models/opencv.h
        models/pinhole.h
        models/radial.h
        models/runtime.h
        models/spherical.h
        models/thin_prism.h
        rig.h rig.cc
        specs.h specs.cc
    PUBLIC_LINK_LIBS
        colmap_geometry
        colmap_util
        Ceres::ceres
        Eigen3::Eigen
    PRIVATE_LINK_LIBS
        OpenImageIO::OpenImageIO
)

COLMAP_ADD_TEST(
    NAME bitmap_test
    SRCS bitmap_test.cc
    LINK_LIBS
        colmap_sensor
)
if(TESTS_ENABLED AND MSVC)
    target_compile_options(colmap_sensor_bitmap_test PRIVATE /utf-8)
endif()
COLMAP_ADD_TEST(
    NAME database_test
    SRCS database_test.cc
    LINK_LIBS colmap_sensor
)
COLMAP_ADD_TEST(
    NAME models_test
    SRCS models_test.cc
    LINK_LIBS colmap_sensor
)
COLMAP_ADD_TEST(
    NAME models_jacobian_test
    SRCS models/jacobian_test.cc
    LINK_LIBS colmap_sensor
)
COLMAP_ADD_TEST(
    NAME rig_test
    SRCS rig_test.cc
    LINK_LIBS colmap_sensor
)
