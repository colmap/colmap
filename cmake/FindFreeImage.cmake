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

# Find package module for FreeImage library.
#
# The following variables are set by this module:
#
#   FREEIMAGE_FOUND: TRUE if FreeImage is found.
#   FREEIMAGE_INCLUDE_DIRS: Include directories for FreeImage.
#   FREEIMAGE_LIBRARIES: Libraries required to link FreeImage.
#
# The following variables control the behavior of this module:
#
# FREEIMAGE_INCLUDE_DIR_HINTS: List of additional directories in which to
#                              search for FreeImage includes.
# FREEIMAGE_LIBRARY_DIR_HINTS: List of additional directories in which to
#                              search for FreeImage libraries.

set(FREEIMAGE_INCLUDE_DIR_HINTS "" CACHE PATH "FreeImage include directory")
set(FREEIMAGE_LIBRARY_DIR_HINTS "" CACHE PATH "FreeImage library directory")

unset(FREEIMAGE_FOUND)
unset(FREEIMAGE_INCLUDE_DIRS)
unset(FREEIMAGE_LIBRARIES)

list(APPEND FREEIMAGE_CHECK_INCLUDE_DIRS
    ${FREEIMAGE_INCLUDE_DIR_HINTS}
    /usr/include
    /usr/local/include
    /opt/include
    /opt/local/include
)

list(APPEND FREEIMAGE_CHECK_LIBRARY_DIRS
    ${FREEIMAGE_LIBRARY_DIR_HINTS}
    /usr/lib
    /usr/local/lib
    /opt/lib
    /opt/local/lib
)

find_path(FREEIMAGE_INCLUDE_DIRS
    NAMES
    FreeImage.h
    PATHS
    ${FREEIMAGE_CHECK_INCLUDE_DIRS})
find_library(FREEIMAGE_LIBRARIES
    NAMES
    freeimage
    PATHS
    ${FREEIMAGE_CHECK_LIBRARY_DIRS})

if(FREEIMAGE_INCLUDE_DIRS AND FREEIMAGE_LIBRARIES)
    set(FREEIMAGE_FOUND TRUE)
endif()

if(FREEIMAGE_FOUND)
    message(STATUS "Found FreeImage")
    message(STATUS "  Includes : ${FREEIMAGE_INCLUDE_DIRS}")
    message(STATUS "  Libraries : ${FREEIMAGE_LIBRARIES}")
else()
    if(FreeImage_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find FreeImage")
    endif()
endif()

add_library(freeimage INTERFACE IMPORTED)
target_include_directories(
    freeimage INTERFACE ${FREEIMAGE_INCLUDE_DIRS})
target_link_libraries(
    freeimage INTERFACE ${FREEIMAGE_LIBRARIES})
