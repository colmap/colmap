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

# Find package module for FLANN library.
#
# The following variables are set by this module:
#
#   FLANN_FOUND: TRUE if FLANN is found.
#   FLANN_INCLUDE_DIRS: Include directories for FLANN.
#   FLANN_LIBRARIES: Libraries required to link FLANN.
#
# The following variables control the behavior of this module:
#
# FLANN_INCLUDE_DIR_HINTS: List of additional directories in which to
#                              search for FLANN includes.
# FLANN_LIBRARY_DIR_HINTS: List of additional directories in which to
#                              search for FLANN libraries.

set(FLANN_INCLUDE_DIR_HINTS "" CACHE PATH "FLANN include directory")
set(FLANN_LIBRARY_DIR_HINTS "" CACHE PATH "FLANN library directory")

unset(FLANN_FOUND)
unset(FLANN_INCLUDE_DIRS)
unset(FLANN_LIBRARIES)

list(APPEND FLANN_CHECK_INCLUDE_DIRS
    ${FLANN_INCLUDE_DIR_HINTS}
    /usr/include
    /usr/local/include
    /opt/include
    /opt/local/include
)

list(APPEND FLANN_CHECK_LIBRARY_DIRS
    ${FLANN_LIBRARY_DIR_HINTS}
    /usr/lib
    /usr/local/lib
    /opt/lib
    /opt/local/lib
)

find_path(FLANN_INCLUDE_DIRS
    NAMES
    flann/flann.hpp
    PATHS
    ${FLANN_CHECK_INCLUDE_DIRS})
find_library(FLANN_LIBRARIES
    NAMES
    flann
    PATHS
    ${FLANN_CHECK_LIBRARY_DIRS})

if(FLANN_INCLUDE_DIRS AND FLANN_LIBRARIES)
    set(FLANN_FOUND TRUE)
endif()

if(FLANN_FOUND)
    message(STATUS "Found FLANN")
    message(STATUS "  Includes : ${FLANN_INCLUDE_DIRS}")
    message(STATUS "  Libraries : ${FLANN_LIBRARIES}")
else()
    if(FLANN_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find FLANN")
    endif()
endif()
