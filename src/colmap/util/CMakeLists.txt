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
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

set(FOLDER_NAME "util")

COLMAP_ADD_SOURCES(
    bitmap.h bitmap.cc
    cache.h
    camera_specs.h camera_specs.cc
    logging.h logging.cc
    math.h math.cc
    matrix.h
    misc.h misc.cc
    opengl_utils.h opengl_utils.cc
    option_manager.h option_manager.cc
    ply.h ply.cc
    random.h random.cc
    sqlite3_utils.h
    string.h string.cc
    threading.h threading.cc
    timer.h timer.cc
    testing.h
    types.h
    version.h version.cc
)

if(CUDA_ENABLED)
    COLMAP_ADD_CUDA_SOURCES(
        cuda.h cuda.cc
        cudacc.h cudacc.cc
    )
endif()

COLMAP_ADD_TEST(bitmap_test bitmap_test.cc)
COLMAP_ADD_TEST(cache_test cache_test.cc)
COLMAP_ADD_TEST(endian_test endian_test.cc)
COLMAP_ADD_TEST(math_test math_test.cc)
COLMAP_ADD_TEST(matrix_test matrix_test.cc)
COLMAP_ADD_TEST(misc_test misc_test.cc)
COLMAP_ADD_TEST(random_test random_test.cc)
COLMAP_ADD_TEST(string_test string_test.cc)
COLMAP_ADD_TEST(threading_test threading_test.cc)
COLMAP_ADD_TEST(timer_test timer_test.cc)
if(GUI_ENABLED)
    COLMAP_ADD_TEST(opengl_utils_test opengl_utils_test.cc)
endif()
