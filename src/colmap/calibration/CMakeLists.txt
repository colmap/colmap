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


set(FOLDER_NAME "calibration")

COLMAP_ADD_LIBRARY(
    NAME colmap_calibration
    SRCS
        anycalib.h anycalib.cc
        calibrator.h calibrator.cc
        ray_fitting.h ray_fitting.cc
        resources.h
    PUBLIC_LINK_LIBS
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
