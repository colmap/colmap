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

set(FOLDER_NAME "glomap_controllers")

COLMAP_ADD_LIBRARY(
    NAME colmap_glomap_controllers
    SRCS
        global_mapper.h global_mapper.cc
        option_manager.h option_manager.cc
        rotation_averager.h rotation_averager.cc
        track_establishment.h track_establishment.cc
        track_retriangulation.h track_retriangulation.cc
    PUBLIC_LINK_LIBS
        colmap_glomap_estimators
        colmap_glomap_io
        colmap_glomap_math
        colmap_glomap_processors
        colmap_glomap_scene
        colmap_controllers
        colmap_estimators
        colmap_sfm
        colmap_util
)

if(MSVC)
    target_compile_options(colmap_glomap_controllers PRIVATE /bigobj)
endif()

COLMAP_ADD_TEST(
    NAME glomap_global_mapper_test
    SRCS global_mapper_test.cc
    LINK_LIBS colmap_glomap_controllers
)
COLMAP_ADD_TEST(
    NAME glomap_rotation_averager_test
    SRCS rotation_averager_test.cc
    LINK_LIBS colmap_glomap_controllers
)
