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


if(IS_MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
    if(WERROR_ENABLED)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Werror")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Werror")
    endif()
    # Avoid pulling in too many header files through <windows.h>
    add_compile_definitions(WIN32_LEAN_AND_MEAN)
elseif(IS_GNU OR IS_CLANG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    if(WERROR_ENABLED)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    endif()
endif()

if(CUDA_ENABLED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

    # Use a separate stream per thread to allow for concurrent kernel execution
    # between multiple threads on the same device.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")

    # Suppress warnings:
    # ptxas warning : Stack size for entry function X cannot be statically determined.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-suppress-stack-size-warning")
endif()

if(CASPAR_ENABLED)
    cmake_policy(SET CMP0094 NEW)
    message(STATUS "Enabling CASPAR support")
    add_compile_definitions(CASPAR_ENABLED)

    find_package(CUDAToolkit REQUIRED)
    include(ExternalProject)

    set(CASPAR_USE_DOUBLE OFF CACHE BOOL "Use double precision in Caspar solver")
    if(CASPAR_USE_DOUBLE)
        set(_caspar_precision "f64")
        add_compile_definitions(CASPAR_USE_DOUBLE)
    else()
        set(_caspar_precision "f32")
    endif()

    set(CASPAR_GEN_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/estimators/caspar/generated/${_caspar_precision}")

    if(NOT EXISTS "${CASPAR_GEN_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "Caspar: pre-generated code not found for precision '${_caspar_precision}'.\n"
            "Expected: ${CASPAR_GEN_DIR}/CMakeLists.txt\n"
            "Re-run generate_caspar.py to regenerate.")
    endif()

    set(CMAKE_POLICY_DEFAULT_CMP0146 OLD)

    set(_caspar_build_dir "${CMAKE_CURRENT_BINARY_DIR}/caspar_generated")
    set(_caspar_lib       "${_caspar_build_dir}/libcaspar_lib_core.a")

    ExternalProject_Add(caspar_external
        SOURCE_DIR  ${CASPAR_GEN_DIR}
        BINARY_DIR  ${_caspar_build_dir}
        CMAKE_ARGS
            -G ${CMAKE_GENERATOR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
        BUILD_BYPRODUCTS ${_caspar_lib}
        # No install step needed — we import the .a directly
        INSTALL_COMMAND ""
    )

    add_library(caspar_lib_core STATIC IMPORTED GLOBAL)
    set_target_properties(caspar_lib_core PROPERTIES
        IMPORTED_LOCATION             ${_caspar_lib}
        INTERFACE_INCLUDE_DIRECTORIES ${CASPAR_GEN_DIR}
    )
    target_link_libraries(caspar_lib_core INTERFACE CUDA::cudart)

    add_dependencies(caspar_lib_core caspar_external)

    message(STATUS "Caspar precision:  ${_caspar_precision}")
    message(STATUS "Caspar source dir: ${CASPAR_GEN_DIR}")
endif()

if(COVERAGE_ENABLED)
    add_compile_options(-coverage -fprofile-update=atomic)
    add_link_options(-coverage)
endif()

add_subdirectory(controllers)
add_subdirectory(estimators)
add_subdirectory(exe)
add_subdirectory(feature)
add_subdirectory(geometry)
add_subdirectory(image)
add_subdirectory(math)
if(MVS_ENABLED)
    add_subdirectory(mvs)
endif()
add_subdirectory(optim)
add_subdirectory(retrieval)
add_subdirectory(scene)
add_subdirectory(sensor)
add_subdirectory(sfm)
add_subdirectory(tools)
add_subdirectory(util)
if (GUI_ENABLED)
    add_subdirectory(ui)
endif()
