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
# Author: Johannes L. Schoenberger (jsch at inf.ethz.ch)

# Find package module for Qt5 library.
#
# The following variables are set by this module:
#
#   Qt5_FOUND: TRUE if Qt5 is found.
#   Qt5_VERSION: Qt5 library version.
#   Qt5_INCLUDE_DIRS: Include directories for Qt5.
#   Qt5_LIBRARIES: Libraries required to link Qt5.
#
# The following variables control the behavior of this module:
#
# QT5_CMAKE_CONFIG_DIR_HINTS: List of additional directories in which to
#                             search for the Qt5 CMake configuration.

set(QT5_CMAKE_CONFIG_DIR_HINTS "" CACHE PATH "Qt5 CMake config directory")

unset(Qt5_FOUND)
unset(Qt5_INCLUDE_DIRS)
unset(Qt5_LIBRARIES)

if(QT5_CMAKE_CONFIG_DIR_HINTS)
    set(Qt5Core_DIR "${QT5_CMAKE_CONFIG_DIR_HINTS}/Qt5Core")
    set(Qt5OpenGL_DIR "${QT5_CMAKE_CONFIG_DIR_HINTS}/Qt5OpenGL")
endif()

find_package(Qt5Core QUIET)
find_package(Qt5OpenGL QUIET)

if(Qt5Core_FOUND AND Qt5OpenGL_FOUND)
    set(Qt5_FOUND TRUE)
    set(Qt5_INCLUDE_DIRS "${Qt5Core_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS}")
    set(Qt5_LIBRARIES "${Qt5Core_LIBRARIES} ${Qt5OpenGL_LIBRARIES}")
endif()

if(Qt5_FOUND)
    if(Qt5Core_VERSION VERSION_LESS Qt5_FIND_VERSION)
        message(FATAL_ERROR "Qt5 version ${Qt5_FIND_VERSION} or newer needed, "
                            "but only found ${Qt5Core_VERSION}")
    else()
        set(Qt5_VERSION ${Qt5Core_VERSION_STRING})
        message(STATUS "Found Qt5")
    endif()
else()
    if(Qt5_FIND_REQUIRED)
        set(Qt5_ERROR_MESSAGE "Could not find Qt5 modules:")
        if(NOT Qt5Core_FOUND)
            set(Qt5_ERROR_MESSAGE "${Qt5_ERROR_MESSAGE} Qt5Core")
        endif()
        if(NOT Qt5OpenGL_FOUND)
            set(Qt5_ERROR_MESSAGE "${Qt5_ERROR_MESSAGE} Qt5OpenGL")
        endif()
        set(Qt5_ERROR_MESSAGE "${Qt5_ERROR_MESSAGE} not found, try "
            "setting -DQt5Core_DIR and -DQt5OpenGL_DIR to the directories "
            "containing Qt5CoreConfig.cmake and Qt5OpenGLConfig.cmake")
        message(FATAL_ERROR ${Qt5_ERROR_MESSAGE})
    endif()
endif()
