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

set(FOLDER_NAME "glomap_estimators")

COLMAP_ADD_LIBRARY(
    NAME colmap_glomap_estimators
    SRCS
        bundle_adjustment.h bundle_adjustment.cc
        cost_functions.h
        global_positioning.h global_positioning.cc
        global_rotation_averaging.h global_rotation_averaging.cc
        gravity_refinement.h gravity_refinement.cc
        optimization_base.h
        relpose_estimation.h relpose_estimation.cc
        rotation_initializer.h rotation_initializer.cc
        view_graph_calibration.h view_graph_calibration.cc
    PUBLIC_LINK_LIBS
        colmap_glomap_math
        colmap_glomap_scene
        colmap_estimators
        colmap_geometry
        colmap_math
        colmap_optim
        colmap_scene
        colmap_util
        Eigen3::Eigen
        Ceres::ceres
        PoseLib::PoseLib
)

COLMAP_ADD_TEST(
    NAME glomap_cost_functions_test
    SRCS cost_functions_test.cc
    LINK_LIBS colmap_glomap_estimators
)
