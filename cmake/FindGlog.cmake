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


# Find package module for Glog library.
#
# The following variables are set by this module:
#
#   GLOG_FOUND: TRUE if Glog is found.
#   glog::glog: Imported target to link against.
#
# The following variables control the behavior of this module:
#
# GLOG_INCLUDE_DIR_HINTS: List of additional directories in which to
#                         search for Glog includes.
# GLOG_LIBRARY_DIR_HINTS: List of additional directories in which to
#                         search for Glog libraries.

set(GLOG_INCLUDE_DIR_HINTS "" CACHE PATH "Glog include directory")
set(GLOG_LIBRARY_DIR_HINTS "" CACHE PATH "Glog library directory")

unset(GLOG_FOUND)

find_package(glog CONFIG QUIET)
if(TARGET glog::glog)
    set(GLOG_FOUND TRUE)
    message(STATUS "Found Glog")
    message(STATUS "  Target : glog::glog")
else()
    # Older versions of glog don't come with a find_package config.
    # Fall back to custom logic to find the library and remap to imported target.

    include(FindPackageHandleStandardArgs)

    list(APPEND GLOG_CHECK_INCLUDE_DIRS
        /usr/local/include
        /usr/local/homebrew/include
        /opt/local/var/macports/software
        /opt/local/include
        /usr/include)
    list(APPEND GLOG_CHECK_PATH_SUFFIXES
        glog/include
        glog/Include
        Glog/include
        Glog/Include
        src/windows)

    list(APPEND GLOG_CHECK_LIBRARY_DIRS
        /usr/local/lib
        /usr/local/homebrew/lib
        /opt/local/lib
        /usr/lib)
    list(APPEND GLOG_CHECK_LIBRARY_SUFFIXES
        glog/lib
        glog/Lib
        Glog/lib
        Glog/Lib
        x64/Release)

    find_path(GLOG_INCLUDE_DIRS
        NAMES
        glog/logging.h
        PATHS
        ${GLOG_INCLUDE_DIR_HINTS}
        ${GLOG_CHECK_INCLUDE_DIRS}
        PATH_SUFFIXES
        ${GLOG_CHECK_PATH_SUFFIXES})
    find_library(GLOG_LIBRARIES
        NAMES
        glog
        libglog
        PATHS
        ${GLOG_LIBRARY_DIR_HINTS}
        ${GLOG_CHECK_LIBRARY_DIRS}
        PATH_SUFFIXES
        ${GLOG_CHECK_LIBRARY_SUFFIXES})

    if(GLOG_INCLUDE_DIRS AND GLOG_LIBRARIES)
        set(GLOG_FOUND TRUE)
        message(STATUS "Found Glog")
        message(STATUS "  Includes : ${GLOG_INCLUDE_DIRS}")
        message(STATUS "  Libraries : ${GLOG_LIBRARIES}")
    endif()

    add_library(glog::glog INTERFACE IMPORTED)
    target_include_directories(glog::glog INTERFACE ${GLOG_INCLUDE_DIRS})
    target_link_libraries(glog::glog INTERFACE ${GLOG_LIBRARIES})
endif()

if(NOT GLOG_FOUND AND GLOG_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find Glog")
endif()
