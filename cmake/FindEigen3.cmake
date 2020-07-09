# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

# Find package module for Eigen 3 library.
#
# The following variables are set by this module:
#
#   EIGEN3_FOUND: TRUE if Eigen is found.
#   EIGEN3_VERSION: Eigen library version.
#   EIGEN3_INCLUDE_DIRS: Include directories for Eigen.
#
# The following variables control the behavior of this module:
#
# EIGEN3_INCLUDE_DIR_HINTS: List of additional directories in which to
#                           search for Eigen includes.

set(EIGEN3_INCLUDE_DIR_HINTS "" CACHE PATH "Eigen include directory")

unset(EIGEN3_FOUND)
unset(EIGEN3_VERSION)
unset(EIGEN3_INCLUDE_DIRS)

if(NOT Eigen3_FIND_VERSION)
    if(NOT Eigen3_FIND_VERSION_MAJOR)
        set(Eigen3_FIND_VERSION_MAJOR 2)
    endif()

    if(NOT Eigen3_FIND_VERSION_MINOR)
        set(Eigen3_FIND_VERSION_MINOR 91)
    endif()

    if(NOT Eigen3_FIND_VERSION_PATCH)
        set(Eigen3_FIND_VERSION_PATCH 0)
    endif()

    set(Eigen3_FIND_VERSION "${Eigen3_FIND_VERSION_MAJOR}.${Eigen3_FIND_VERSION_MINOR}.${Eigen3_FIND_VERSION_PATCH}")
endif()

macro(_EIGEN3_CHECK_VERSION)
    file(READ "${EIGEN3_INCLUDE_DIRS}/Eigen/src/Core/util/Macros.h" _EIGEN3_VERSION_HEADER)

    string(REGEX MATCH "define[ \t]+EIGEN_WORLD_VERSION[ \t]+([0-9]+)" _EIGEN3_WORLD_VERSION_MATCH "${_EIGEN3_VERSION_HEADER}")
    set(EIGEN3_WORLD_VERSION "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define[ \t]+EIGEN_MAJOR_VERSION[ \t]+([0-9]+)" _EIGEN3_MAJOR_VERSION_MATCH "${_EIGEN3_VERSION_HEADER}")
    set(EIGEN3_MAJOR_VERSION "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define[ \t]+EIGEN_MINOR_VERSION[ \t]+([0-9]+)" _EIGEN3_MINOR_VERSION_MATCH "${_EIGEN3_VERSION_HEADER}")
    set(EIGEN3_MINOR_VERSION "${CMAKE_MATCH_1}")

    set(EIGEN3_VERSION ${EIGEN3_WORLD_VERSION}.${EIGEN3_MAJOR_VERSION}.${EIGEN3_MINOR_VERSION})

    if(${EIGEN3_VERSION} VERSION_LESS ${Eigen3_FIND_VERSION})
        set(EIGEN3_VERSION_OK FALSE)
    else()
        set(EIGEN3_VERSION_OK TRUE)
    endif()

    if(NOT EIGEN3_VERSION_OK)
        message(STATUS "Eigen version ${EIGEN3_VERSION} found in ${EIGEN3_INCLUDE_DIRS}, "
                       "but at least version ${Eigen3_FIND_VERSION} is required.")
    endif()
endmacro()

if(EIGEN3_INCLUDE_DIRS)
    _EIGEN3_CHECK_VERSION()
    set(EIGEN3_FOUND ${EIGEN3_VERSION_OK})
else()
    find_path(EIGEN3_INCLUDE_DIRS
        NAMES
        signature_of_eigen3_matrix_library
        PATHS
        ${EIGEN3_INCLUDE_DIR_HINTS}
        /usr/include
        /usr/local/include
        /opt/include
        /opt/local/include
        PATH_SUFFIXES
        eigen3
        eigen
    )

    if(EIGEN3_INCLUDE_DIRS)
        _EIGEN3_CHECK_VERSION()
    endif()

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(Eigen3 DEFAULT_MSG EIGEN3_INCLUDE_DIRS EIGEN3_VERSION_OK)

    mark_as_advanced(EIGEN3_INCLUDE_DIRS)
endif()

if(EIGEN3_FOUND)
    message(STATUS "Found Eigen")
    message(STATUS "  Includes : ${EIGEN3_INCLUDE_DIRS}")
else()
    if(Eigen3_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find Eigen")
    endif()
endif()
