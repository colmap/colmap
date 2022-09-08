# Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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

if(IS_MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
elseif(IS_GNU OR IS_CLANG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    # We want c++14 mainly for supporting latest dependencies, but older GCC
    # versions in combination with CUDA do not yet support it. In the worst
    # case, this will lead to compile errors and there is not much we can
    # do on the COLMAP side.
    if(CUDA_ENABLED AND
       IS_GNU AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    endif()
endif()

add_subdirectory(base)
add_subdirectory(controllers)
add_subdirectory(estimators)
add_subdirectory(exe)
add_subdirectory(feature)
add_subdirectory(mvs)
add_subdirectory(optim)
add_subdirectory(retrieval)
add_subdirectory(sfm)
add_subdirectory(tools)
add_subdirectory(util)
if (GUI_ENABLED)
    add_subdirectory(ui)
endif()

COLMAP_ADD_STATIC_LIBRARY(colmap ${COLMAP_SOURCES})

if(CUDA_ENABLED)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math")

    # Use a separate stream per thread to allow for concurrent kernel execution
    # between multiple threads on the same device.
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --default-stream per-thread")

    # Fix for Ubuntu 16.04.
    add_definitions("-D_MWAITXINTRIN_H_INCLUDED")

    if(MSVC)
        # Avoid pulling in too many header files through <windows.h>
        add_definitions("-DWIN32_LEAN_AND_MEAN")
    endif()

    COLMAP_ADD_STATIC_CUDA_LIBRARY(colmap_cuda ${COLMAP_CUDA_SOURCES})
    target_link_libraries(colmap_cuda colmap)

    target_link_libraries(colmap
        colmap_cuda ${COLMAP_INTERNAL_LIBRARIES} ${COLMAP_EXTERNAL_LIBRARIES})
else()
    target_link_libraries(colmap
        ${COLMAP_INTERNAL_LIBRARIES} ${COLMAP_EXTERNAL_LIBRARIES})
endif()
