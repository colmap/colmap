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


# Find package module for LZ4 library.
#
# The following variables are set by this module:
#
#   LZ4_FOUND: TRUE if LZ4 is found.
#   LZ4_INCLUDE_DIRS: Include directories for LZ4.
#   LZ4_LIBRARIES: Libraries required to link LZ4.
#
# The following variables control the behavior of this module:
#
# LZ4_INCLUDE_DIR_HINTS: List of additional directories in which to
#                              search for LZ4 includes.
# LZ4_LIBRARY_DIR_HINTS: List of additional directories in which to
#                              search for LZ4 libraries.

set(LZ4_INCLUDE_DIR_HINTS "" CACHE PATH "LZ4 include directory")
set(LZ4_LIBRARY_DIR_HINTS "" CACHE PATH "LZ4 library directory")

unset(LZ4_FOUND)
unset(LZ4_INCLUDE_DIRS)
unset(LZ4_LIBRARIES)

list(APPEND LZ4_CHECK_INCLUDE_DIRS
    ${LZ4_INCLUDE_DIR_HINTS}
    /usr/include
    /usr/local/include
    /opt/include
    /opt/local/include
)

list(APPEND LZ4_CHECK_LIBRARY_DIRS
    ${LZ4_LIBRARY_DIR_HINTS}
    /usr/lib
    /usr/local/lib
    /usr/lib/x86_64-linux-gnu
    /opt/lib
    /opt/local/lib
)

find_path(LZ4_INCLUDE_DIRS
    NAMES
    lz4.h
    PATHS
    ${LZ4_CHECK_INCLUDE_DIRS})
find_library(LZ4_LIBRARIES
    NAMES
    lz4
    PATHS
    ${LZ4_CHECK_LIBRARY_DIRS})

if(LZ4_INCLUDE_DIRS AND LZ4_LIBRARIES)
    set(LZ4_FOUND TRUE)
endif()

if(LZ4_FOUND)
    message(STATUS "Found LZ4")
    message(STATUS "  Includes : ${LZ4_INCLUDE_DIRS}")
    message(STATUS "  Libraries : ${LZ4_LIBRARIES}")
else()
    if(LZ4_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find LZ4")
    endif()
endif()

add_library(lz4 INTERFACE IMPORTED)
target_include_directories(
    lz4 INTERFACE ${LZ4_INCLUDE_DIRS})
target_link_libraries(
    lz4 INTERFACE ${LZ4_LIBRARIES})
