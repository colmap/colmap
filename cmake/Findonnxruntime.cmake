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


# Find package module for onnxruntime library.
#
# The following variables are set by this module:
#
#   onnxruntime_FOUND: TRUE if onnxruntime is found.
#   onnxruntime::onnxruntime: Imported target to link against.
#
# The following variables control the behavior of this module:
#
# onnxruntime_INCLUDE_DIR_HINTS: List of additional directories in which to
#                         search for onnxruntime includes.
# onnxruntime_LIBRARY_DIR_HINTS: List of additional directories in which to
#                         search for onnxruntime libraries.

set(onnxruntime_INCLUDE_DIR_HINTS "" CACHE PATH "onnxruntime include directory")
set(onnxruntime_LIBRARY_DIR_HINTS "" CACHE PATH "onnxruntime library directory")

unset(onnxruntime_FOUND)
unset(onnxruntime_INCLUDE_DIRS)
unset(onnxruntime_LIBRARIES)

find_package(onnxruntime CONFIG QUIET)
if(TARGET onnxruntime::onnxruntime)
    set(onnxruntime_FOUND TRUE)
    message(STATUS "Found onnxruntime")
    message(STATUS "  Target : onnxruntime::onnxruntime")
else()
    find_path(onnxruntime_INCLUDE_DIRS
        NAMES
        onnxruntime/onnxruntime_cxx_api.h
        PATHS
        ${onnxruntime_INCLUDE_DIR_HINTS}
        /usr/include
        /usr/local/include
        /sw/include
        /opt/include
        /opt/local/include)
    list(APPEND onnxruntime_LIBRARY_DIR_HINTS
        /usr/lib64
        /usr/lib
        /usr/local/lib64
        /usr/local/lib
        /sw/lib
        /opt/lib
        /opt/local/lib)
    find_library(onnxruntime_LIBRARIES
        NAMES onnxruntime libonnxruntime
        PATHS ${onnxruntime_LIBRARY_DIR_HINTS})
    find_library(onnxruntime_PROVIDERS_SHARED_LIBRARY
        NAMES onnxruntime_providers_shared libonnxruntime_providers_shared
        PATHS ${onnxruntime_LIBRARY_DIR_HINTS})
    if(CUDA_ENABLED)
        find_library(onnxruntime_PROVIDERS_CUDA_LIBRARY
            NAMES onnxruntime_providers_cuda libonnxruntime_providers_cuda
            PATHS ${onnxruntime_LIBRARY_DIR_HINTS})
    endif()

    if(onnxruntime_INCLUDE_DIRS AND onnxruntime_LIBRARIES)
        if(onnxruntime_PROVIDERS_SHARED_LIBRARY)
            list(APPEND onnxruntime_LIBRARIES ${onnxruntime_PROVIDERS_SHARED_LIBRARY})
        endif()
        if(onnxruntime_PROVIDERS_CUDA_LIBRARY)
            list(APPEND onnxruntime_LIBRARIES ${onnxruntime_PROVIDERS_CUDA_LIBRARY})
        endif()
        set(onnxruntime_FOUND TRUE)
        message(STATUS "Found onnxruntime")
        message(STATUS "  Includes : ${onnxruntime_INCLUDE_DIRS}")
        message(STATUS "  Libraries : ${onnxruntime_LIBRARIES}")
    else()
        set(onnxruntime_FOUND FALSE)
    endif()

    add_library(onnxruntime::onnxruntime INTERFACE IMPORTED)
    target_include_directories(
        onnxruntime::onnxruntime INTERFACE ${onnxruntime_INCLUDE_DIRS}/onnxruntime)
    target_link_libraries(
        onnxruntime::onnxruntime INTERFACE ${onnxruntime_LIBRARIES})
    # This is a hack to make sure that the onnxruntime dll is copied to the output directory,
    # since vcpkg's custom add_library/add_executable macros copy any dependencies from vcpkg's
    # installed directory to the output directory.
    # See: https://github.com/microsoft/vcpkg/blob/fb7ba3b89b0d8e3e56b0508a144fe85015edfab6/scripts/buildsystems/vcpkg.cmake#L607
    if(IS_WINDOWS AND VCPKG_INSTALLED_DIR)
        foreach(_lib ${onnxruntime_LIBRARIES})
            get_filename_component(_lib_dir "${_lib}" DIRECTORY)
            get_filename_component(_lib_name "${_lib}" NAME_WE)
            set(_dll "${_lib_dir}/${_lib_name}.dll")
            file(COPY "${_dll}" DESTINATION "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
        endforeach()
    endif()
endif()

if(NOT onnxruntime_FOUND AND onnxruntime_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find onnxruntime")
endif()
