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


# Find package module for CHOLMOD library.
#
# The following variables are set by this module:
#
#   CHOLMOD_FOUND: TRUE if CHOLMOD is found.
#   CHOLMOD::CHOLMOD: Imported target to link against.
#
# The following variables control the behavior of this module:
#
# CHOLMOD_INCLUDE_DIR_HINTS: List of additional directories in which to
#                         search for CHOLMOD includes.
# CHOLMOD_LIBRARY_DIR_HINTS: List of additional directories in which to
#                         search for CHOLMOD libraries.

set(CHOLMOD_INCLUDE_DIR_HINTS "" CACHE PATH "CHOLMOD include directory")
set(CHOLMOD_LIBRARY_DIR_HINTS "" CACHE PATH "CHOLMOD library directory")

unset(CHOLMOD_FOUND)
unset(CHOLMOD_INCLUDE_DIRS)
unset(CHOLMOD_LIBRARIES)

find_package(CHOLMOD CONFIG QUIET)
if(TARGET CHOLMOD::CHOLMOD)
    set(CHOLMOD_FOUND TRUE)
    message(STATUS "Found CHOLMOD")
    message(STATUS "  Target : CHOLMOD::CHOLMOD")
else()
    find_path(CHOLMOD_INCLUDE_DIRS
        NAMES
        suitesparse/cholmod.h
        PATHS
        ${CHOLMOD_INCLUDE_DIR_HINTS}
        /usr/include
        /usr/local/include
        /sw/include
        /opt/include
        /opt/local/include)
    find_library(CHOLMOD_LIBRARIES
        NAMES
        cholmod
        PATHS
        ${CHOLMOD_LIBRARY_DIR_HINTS}
        /usr/lib64
        /usr/lib
        /usr/local/lib64
        /usr/local/lib
        /sw/lib
        /opt/lib
        /opt/local/lib)

    if(CHOLMOD_INCLUDE_DIRS AND CHOLMOD_LIBRARIES)
        set(CHOLMOD_FOUND TRUE)
        message(STATUS "Found CHOLMOD")
        message(STATUS "  Includes : ${CHOLMOD_INCLUDE_DIRS}")
        message(STATUS "  Libraries : ${CHOLMOD_LIBRARIES}")
    else()
        set(CHOLMOD_FOUND FALSE)
    endif()

    add_library(CHOLMOD::CHOLMOD INTERFACE IMPORTED)
    target_include_directories(
        CHOLMOD::CHOLMOD INTERFACE ${CHOLMOD_INCLUDE_DIRS}/suitesparse)
    target_link_libraries(
        CHOLMOD::CHOLMOD INTERFACE ${CHOLMOD_LIBRARIES})
endif()

if(NOT CHOLMOD_FOUND AND CHOLMOD_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find CHOLMOD")
endif()
