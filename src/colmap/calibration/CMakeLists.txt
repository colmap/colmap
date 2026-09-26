# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "calibration")

COLMAP_ADD_LIBRARY(
    NAME colmap_calibration
    SRCS
        anycalib.h anycalib.cc
        calibrator.h calibrator.cc
        exif.h exif.cc
        ray_fitting.h ray_fitting.cc
        resources.h
    PUBLIC_LINK_LIBS
        colmap_geometry
        colmap_scene_types
        colmap_sensor
        colmap_util
        Eigen3::Eigen
    PRIVATE_LINK_LIBS
        Ceres::ceres
)

if(ONNX_ENABLED)
    target_link_libraries(colmap_calibration PRIVATE colmap_onnx_util)
endif()

COLMAP_ADD_TEST(
    NAME calibrator_test
    SRCS calibrator_test.cc
    LINK_LIBS colmap_calibration
)

COLMAP_ADD_TEST(
    NAME exif_test
    SRCS exif_test.cc
    LINK_LIBS colmap_calibration
)

COLMAP_ADD_TEST(
    NAME ray_fitting_test
    SRCS ray_fitting_test.cc
    LINK_LIBS colmap_calibration
)

if(ONNX_ENABLED)
    COLMAP_ADD_TEST(
        NAME anycalib_test
        SRCS anycalib_test.cc
        LINK_LIBS
            colmap_calibration
            onnxruntime::onnxruntime
    )
endif()
