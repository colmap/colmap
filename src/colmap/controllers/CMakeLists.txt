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


set(FOLDER_NAME "controllers")

COLMAP_ADD_LIBRARY(
    NAME colmap_controllers
    SRCS
        automatic_reconstruction.h automatic_reconstruction.cc
        bundle_adjustment.h bundle_adjustment.cc
        hierarchical_pipeline.h hierarchical_pipeline.cc
        feature_extraction.h feature_extraction.cc
        feature_matching.h feature_matching.cc
        feature_matching_utils.h feature_matching_utils.cc
        image_reader.h image_reader.cc
        incremental_pipeline.h incremental_pipeline.cc
        option_manager.h option_manager.cc
    PUBLIC_LINK_LIBS
        colmap_estimators
        colmap_feature
        colmap_geometry
        colmap_scene
        colmap_util
        Eigen3::Eigen
        Boost::program_options
    PRIVATE_LINK_LIBS
        colmap_image
        colmap_math
        colmap_mvs
        colmap_sfm
        Ceres::ceres
        Boost::boost
)

COLMAP_ADD_TEST(
    NAME hierarchical_pipeline_test
    SRCS hierarchical_pipeline_test.cc
    LINK_LIBS colmap_controllers
)
COLMAP_ADD_TEST(
    NAME incremental_pipeline_test
    SRCS incremental_pipeline_test.cc
    LINK_LIBS colmap_controllers
)
COLMAP_ADD_TEST(
    NAME image_reader_test
    SRCS image_reader_test.cc
    LINK_LIBS colmap_controllers
)
