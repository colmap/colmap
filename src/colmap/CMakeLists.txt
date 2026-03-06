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
    # Get first Python interpreter found 
    # (best for finding active venv)
    cmake_policy(SET CMP0094 NEW)

    message(STATUS "Enabling CASPAR support")
    add_compile_definitions(CASPAR_ENABLED)

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    find_package(CUDAToolkit REQUIRED)

    set(CASPAR_CAMERA_MODEL "SIMPLE_RADIAL" CACHE STRING
        "Camera model to build the Caspar solver for")
    set_property(CACHE CASPAR_CAMERA_MODEL PROPERTY STRINGS
        SIMPLE_RADIAL PINHOLE)

    string(TOLOWER ${CASPAR_CAMERA_MODEL} _caspar_model_lower)

    set(_caspar_solver_script
        "${CMAKE_CURRENT_SOURCE_DIR}/estimators/caspar/${_caspar_model_lower}_solver.py")
    set(_caspar_adapter_src
        "${CMAKE_CURRENT_SOURCE_DIR}/estimators/caspar/adapters/caspar_model_adapter_${_caspar_model_lower}.h")

    set(CASPAR_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/caspar_generated")
    set(_caspar_solver_header "${CASPAR_GEN_DIR}/solver.h")
    set(_caspar_lib           "${CASPAR_GEN_DIR}/build/libcaspar_lib_core.a")
    set(_caspar_adapter_dst   "${CASPAR_GEN_DIR}/caspar_model_adapter.h")

    if(NOT EXISTS ${_caspar_solver_script})
        message(FATAL_ERROR
            "Caspar: no solver spec for model '${CASPAR_CAMERA_MODEL}'.\n"
            "Expected: ${_caspar_solver_script}")
    endif()
    if(NOT EXISTS ${_caspar_adapter_src})
        message(FATAL_ERROR
            "Caspar: no adapter for model '${CASPAR_CAMERA_MODEL}'.\n"
            "Expected: ${_caspar_adapter_src}")
    endif()

    set(CASPAR_USE_DOUBLE OFF CACHE BOOL "Use double precision in Caspar solver")

    set(_caspar_precision_arg "")
    if(CASPAR_USE_DOUBLE)
        set(_caspar_precision_arg "--double")
        add_compile_definitions(CASPAR_USE_DOUBLE)
    endif()

    # The script receives CASPAR_GEN_DIR as argv[1] and writes both
    # solver.h and libcaspar_lib_core.a there.
    # Both are declared as OUTPUT so the build system tracks them correctly
    # and only reruns this command when the spec file changes.
    add_custom_command(
        OUTPUT  ${_caspar_solver_header} ${_caspar_lib}
        COMMAND ${Python3_EXECUTABLE} ${_caspar_solver_script}
                ${CASPAR_GEN_DIR} ${_caspar_precision_arg}
        DEPENDS ${_caspar_solver_script}
        COMMENT "Generating Caspar solver (${CASPAR_CAMERA_MODEL}, ${_caspar_precision_arg})..."
        VERBATIM
    )

    add_custom_target(caspar_generate
        DEPENDS ${_caspar_solver_header} ${_caspar_lib}
    )

    add_library(caspar_lib_core STATIC IMPORTED GLOBAL)
    set_target_properties(caspar_lib_core PROPERTIES
        IMPORTED_LOCATION             ${_caspar_lib}
        INTERFACE_INCLUDE_DIRECTORIES ${CASPAR_GEN_DIR}
    )
    target_link_libraries(caspar_lib_core INTERFACE CUDA::cudart)

    # configure_file destination is also in the build tree.
    # CASPAR_GEN_DIR is already on the include path via caspar_lib_core,
    # so #include "caspar_model_adapter.h" resolves correctly.
    configure_file(${_caspar_adapter_src} ${_caspar_adapter_dst} COPYONLY)

    message(STATUS "Caspar camera model:  ${CASPAR_CAMERA_MODEL}")
    message(STATUS "Caspar output dir:    ${CASPAR_GEN_DIR}")
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
add_subdirectory(mvs)
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
