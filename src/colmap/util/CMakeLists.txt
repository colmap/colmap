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


set(FOLDER_NAME "util")

COLMAP_ADD_LIBRARY(
    NAME colmap_util
    SRCS
        cache.h
        eigen_alignment.h
        logging.h logging.cc
        misc.h misc.cc
        opengl_utils.h opengl_utils.cc
        ply.h ply.cc
        sqlite3_utils.h
        string.h string.cc
        threading.h threading.cc
        timer.h timer.cc
        types.h
        version.h version.cc
    PUBLIC_LINK_LIBS
        Boost::filesystem
        Eigen3::Eigen
        glog::glog
        SQLite::SQLite3
)
if(TESTS_ENABLED)
    target_sources(
        colmap_util
        PRIVATE
            testing.h testing.cc
    )
endif()

if(GUI_ENABLED)
    target_link_libraries(colmap_util PUBLIC Qt5::Core Qt5::OpenGL OpenGL::GL)
endif()

if(CUDA_ENABLED)
    COLMAP_ADD_LIBRARY(
        NAME colmap_util_cuda
        SRCS
            cuda.h cuda.cc
            cudacc.h cudacc.cc
        PUBLIC_LINK_LIBS
            colmap_util
            CUDA::cudart
    )
endif()

COLMAP_ADD_TEST(
    NAME cache_test
    SRCS cache_test.cc
    LINK_LIBS colmap_util
)
COLMAP_ADD_TEST(
    NAME endian_test
    SRCS endian_test.cc
    LINK_LIBS colmap_util
)
COLMAP_ADD_TEST(
    NAME misc_test
    SRCS misc_test.cc
    LINK_LIBS colmap_util
)
COLMAP_ADD_TEST(
    NAME string_test
    SRCS string_test.cc
    LINK_LIBS colmap_util
)
COLMAP_ADD_TEST(
    NAME threading_test
    SRCS threading_test.cc
    LINK_LIBS colmap_util
)
COLMAP_ADD_TEST(
    NAME timer_test
    SRCS timer_test.cc
    LINK_LIBS colmap_util
)
COLMAP_ADD_TEST(
    NAME types_test
    SRCS types_test.cc
    LINK_LIBS colmap_util
)
if(GUI_ENABLED)
    COLMAP_ADD_TEST(
        NAME opengl_utils_test
        SRCS opengl_utils_test.cc
        LINK_LIBS colmap_util
    )
endif()
