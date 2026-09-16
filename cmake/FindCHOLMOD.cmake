# SPDX-License-Identifier: BSD-3-Clause

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
    list(APPEND CHOLMOD_INCLUDE_SEARCH_PATHS
        ${CHOLMOD_INCLUDE_DIR_HINTS}
        /usr/include
        /usr/local/include
        /sw/include
        /opt/include
        /opt/local/include)

    # Some distros don't package suitesparse under a /suitesparse subdirectory (e.g. NixOS).
    # Search for both layouts separately so that the suitesparse/ subdirectory
    # layout is reliably preferred across all search paths.  A single find_path
    # call with both names would iterate search paths first and names second,
    # which could pick up a bare cholmod.h from an earlier path over
    # suitesparse/cholmod.h from a later one.
    find_path(CHOLMOD_INCLUDE_DIRS
        NAMES suitesparse/cholmod.h
        PATHS ${CHOLMOD_INCLUDE_SEARCH_PATHS})
    if(NOT CHOLMOD_INCLUDE_DIRS)
        unset(CHOLMOD_INCLUDE_DIRS CACHE)
        find_path(CHOLMOD_INCLUDE_DIRS
            NAMES cholmod.h
            PATHS ${CHOLMOD_INCLUDE_SEARCH_PATHS})
    endif()

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

    if(EXISTS "${CHOLMOD_INCLUDE_DIRS}/suitesparse/cholmod.h")
        set(CHOLMOD_INTERFACE_INCLUDE_DIRS "${CHOLMOD_INCLUDE_DIRS}/suitesparse")
    else()
        set(CHOLMOD_INTERFACE_INCLUDE_DIRS "${CHOLMOD_INCLUDE_DIRS}")
    endif()

    add_library(CHOLMOD::CHOLMOD INTERFACE IMPORTED)
    target_include_directories(
        CHOLMOD::CHOLMOD INTERFACE ${CHOLMOD_INTERFACE_INCLUDE_DIRS})
    target_link_libraries(
        CHOLMOD::CHOLMOD INTERFACE ${CHOLMOD_LIBRARIES})
endif()

if(NOT CHOLMOD_FOUND AND CHOLMOD_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find CHOLMOD")
endif()
