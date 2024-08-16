# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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


set(FOLDER_NAME "feature")

COLMAP_ADD_LIBRARY(
    NAME colmap_feature_types
    SRCS
        types.h types.cc
    PUBLIC_LINK_LIBS
        colmap_util
        Eigen3::Eigen
)

COLMAP_ADD_TEST(
    NAME types_test
    SRCS types_test.cc
    LINK_LIBS colmap_feature_types
)

COLMAP_ADD_LIBRARY(
    NAME colmap_feature
    SRCS
        extractor.h
        matcher.h matcher.cc
        pairing.h pairing.cc
        sift.h sift.cc
        types.h types.cc
        utils.h utils.cc
    PUBLIC_LINK_LIBS
        colmap_feature_types
        colmap_geometry
        colmap_retrieval
        colmap_scene
        colmap_util
    PRIVATE_LINK_LIBS
        colmap_math
        colmap_sensor
        colmap_vlfeat
        Eigen3::Eigen
        flann
        lz4
)
if(GPU_ENABLED)
    target_link_libraries(colmap_feature PRIVATE colmap_sift_gpu)
    if(NOT GUI_ENABLED)
        target_link_libraries(colmap_feature PRIVATE GLEW::GLEW)
    endif()
endif()

COLMAP_ADD_TEST(
    NAME feature_utils_test
    SRCS utils_test.cc
    LINK_LIBS colmap_feature
)
COLMAP_ADD_TEST(
    NAME pairing_test
    SRCS pairing_test.cc
    LINK_LIBS
        colmap_feature
        colmap_scene
)
COLMAP_ADD_TEST(
    NAME sift_test
    SRCS sift_test.cc
    LINK_LIBS
        colmap_feature
)
if(TESTS_ENABLED AND GUI_ENABLED)
    target_link_libraries(colmap_feature_sift_test Qt5::Widgets)
endif()
