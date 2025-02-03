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


# Find package module for CryptoPP library.
#
# The following variables are set by this module:
#
#   CryptoPP_FOUND: TRUE if CryptoPP is found.
#   cryptopp: Imported target to link against.
#
# The following variables control the behavior of this module:
#
# CryptoPP_INCLUDE_DIR_HINTS: List of additional directories in which to
#                              search for CryptoPP includes.
# CryptoPP_LIBRARY_DIR_HINTS: List of additional directories in which to
#                              search for CryptoPP libraries.

set(CryptoPP_INCLUDE_DIR_HINTS "" CACHE PATH "CryptoPP include directory")
set(CryptoPP_LIBRARY_DIR_HINTS "" CACHE PATH "CryptoPP library directory")

unset(CryptoPP_FOUND)
unset(CryptoPP_INCLUDE_DIRS)
unset(CryptoPP_LIBRARIES)

list(APPEND CryptoPP_CHECK_INCLUDE_DIRS
    ${CryptoPP_INCLUDE_DIR_HINTS}
    /usr/include
    /usr/local/include
    /opt/include
    /opt/local/include
)

list(APPEND CryptoPP_CHECK_LIBRARY_DIRS
    ${CryptoPP_LIBRARY_DIR_HINTS}
    /usr/lib
    /usr/local/lib
    /opt/lib
    /opt/local/lib
)

find_path(CryptoPP_INCLUDE_DIRS
    NAMES
    cryptopp/cryptlib.h
    PATHS
    ${CryptoPP_CHECK_INCLUDE_DIRS})
find_library(CryptoPP_LIBRARIES
    NAMES
    cryptopp
    PATHS
    ${CryptoPP_CHECK_LIBRARY_DIRS})

if(CryptoPP_INCLUDE_DIRS AND CryptoPP_LIBRARIES)
    set(CryptoPP_FOUND TRUE)
endif()

if(CryptoPP_FOUND)
    message(STATUS "Found CryptoPP")
    message(STATUS "  Includes : ${CryptoPP_INCLUDE_DIRS}")
    message(STATUS "  Libraries : ${CryptoPP_LIBRARIES}")
else()
    if(CryptoPP_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find CryptoPP")
    endif()
endif()

add_library(cryptopp INTERFACE IMPORTED)
target_include_directories(
    cryptopp INTERFACE ${CryptoPP_INCLUDE_DIRS})
target_link_libraries(
    cryptopp INTERFACE ${CryptoPP_LIBRARIES})
