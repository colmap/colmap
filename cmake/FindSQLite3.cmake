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

# Find package module for SQLite3 library.
#
# The following variables are set by this module:
#
#   SQLite3_FOUND: TRUE if SQLite3 is found.
#   SQLite3_INCLUDE_DIRS: Include directories for SQLite3.
#   SQLite3_LIBRARIES: Libraries required to link SQLite3.
#
# The following variables control the behavior of this module:
#
# SQLite3_INCLUDE_DIR_HINTS: List of additional directories in which to
#                              search for SQLite3 includes.
# SQLite3_LIBRARY_DIR_HINTS: List of additional directories in which to
#                              search for SQLite3 libraries.

set(SQLite3_INCLUDE_DIR_HINTS "" CACHE PATH "SQLite3 include directory")
set(SQLite3_LIBRARY_DIR_HINTS "" CACHE PATH "SQLite3 library directory")

unset(SQLite3_FOUND)
unset(SQLite3_INCLUDE_DIRS)
unset(SQLite3_LIBRARIES)

list(APPEND SQLite3_CHECK_INCLUDE_DIRS
    ${SQLite3_INCLUDE_DIR_HINTS}
    /usr/include
    /usr/local/include
    /opt/include
    /opt/local/include
)

list(APPEND SQLite3_CHECK_LIBRARY_DIRS
    ${SQLite3_LIBRARY_DIR_HINTS}
    /usr/lib
    /usr/local/lib
    /opt/lib
    /opt/local/lib
)

find_path(SQLite3_INCLUDE_DIRS
    NAMES
    sqlite3.h
    PATHS
    ${SQLite3_CHECK_INCLUDE_DIRS})
find_library(SQLite3_LIBRARIES
    NAMES
    sqlite3
    PATHS
    ${SQLite3_CHECK_LIBRARY_DIRS})

if(SQLite3_INCLUDE_DIRS AND SQLite3_LIBRARIES)
    set(SQLite3_FOUND TRUE)
endif()

if(SQLite3_FOUND)
    message(STATUS "Found SQLite3")
    message(STATUS "  Includes : ${SQLite3_INCLUDE_DIRS}")
    message(STATUS "  Libraries : ${SQLite3_LIBRARIES}")
else()
    if(SQLite3_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find SQLite3")
    endif()
endif()
