# SPDX-License-Identifier: BSD-3-Clause

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
