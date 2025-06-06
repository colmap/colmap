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


# Find package module for Glew library.
#
# The following variables are set by this module:
#
#   GLEW_FOUND: TRUE if Glew is found.
#   GLEW::GLEW: Imported target to link against.
#
# The following variables control the behavior of this module:
#
# GLEW_INCLUDE_DIR_HINTS: List of additional directories in which to
#                         search for Glew includes.
# GLEW_LIBRARY_DIR_HINTS: List of additional directories in which to
#                         search for Glew libraries.

set(GLEW_INCLUDE_DIR_HINTS "" CACHE PATH "Glew include directory")
set(GLEW_LIBRARY_DIR_HINTS "" CACHE PATH "Glew library directory")

unset(GLEW_FOUND)
unset(GLEW_INCLUDE_DIRS)
unset(GLEW_LIBRARIES)

find_package(Glew CONFIG QUIET)
if(TARGET GLEW::GLEW)
    set(GLEW_FOUND TRUE)
    message(STATUS "Found Glew")
    message(STATUS "  Target : GLEW::GLEW")
else()
    find_path(GLEW_INCLUDE_DIRS
        NAMES
        GL/glew.h
        PATHS
        ${GLEW_INCLUDE_DIR_HINTS}
        /usr/include
        /usr/local/include
        /sw/include
        /opt/include
        /opt/local/include)
    find_library(GLEW_LIBRARIES
        NAMES
        GLEW
        Glew
        glew
        glew32
        PATHS
        ${GLEW_LIBRARY_DIR_HINTS}
        /usr/lib64
        /usr/lib
        /usr/local/lib64
        /usr/local/lib
        /sw/lib
        /opt/lib
        /opt/local/lib)

    if(GLEW_INCLUDE_DIRS AND GLEW_LIBRARIES)
        set(GLEW_FOUND TRUE)
        message(STATUS "Found Glew")
        message(STATUS "  Includes : ${GLEW_INCLUDE_DIRS}")
        message(STATUS "  Libraries : ${GLEW_LIBRARIES}")
    else()
        set(GLEW_FOUND FALSE)
    endif()

    add_library(GLEW::GLEW INTERFACE IMPORTED)
    target_include_directories(
        GLEW::GLEW INTERFACE ${GLEW_INCLUDE_DIRS})
    target_link_libraries(
        GLEW::GLEW INTERFACE ${GLEW_LIBRARIES})
endif()

if(NOT GLEW_FOUND AND GLEW_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find Glew")
endif()
