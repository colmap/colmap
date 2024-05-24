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


set(FOLDER_NAME "exe")

if(IS_MSVC)
    add_compile_options("/bigobj")
endif()

set(OPTIONAL_LIBS)
if(CUDA_ENABLED)
    list(APPEND OPTIONAL_LIBS
        colmap_util_cuda
        colmap_mvs_cuda
    )
endif()
if(GUI_ENABLED)
    list(APPEND OPTIONAL_LIBS
        colmap_ui
    )
endif()

COLMAP_ADD_LIBRARY(
    NAME colmap_exe
    SRCS
        feature.h feature.cc
        sfm.h sfm.cc
        model.h model.cc
    PUBLIC_LINK_LIBS
        colmap_controllers
        colmap_estimators
        colmap_geometry
        colmap_optim
        colmap_scene
        colmap_sfm
        colmap_util
    PRIVATE_LINK_LIBS
        Boost::boost
        colmap_sfm
)

COLMAP_ADD_EXECUTABLE(
    NAME colmap_main
    SRCS
        feature.cc
        sfm.cc
        colmap.cc
        database.cc
        feature.cc
        gui.cc
        image.cc
        model.cc
        mvs.cc
        sfm.cc
        vocab_tree.cc
    LINK_LIBS
        colmap_controllers
        colmap_retrieval
        colmap_scene
        colmap_sfm
        colmap_util
        ${OPTIONAL_LIBS}
        Boost::boost
)
set_target_properties(colmap_main PROPERTIES OUTPUT_NAME colmap)
