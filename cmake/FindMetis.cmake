# SPDX-License-Identifier: BSD-3-Clause

# Find package module for Metis library.
#
# The following variables are set by this module:
#
#   METIS_FOUND: TRUE if Metis is found.
#   metis: Imported target to link against.
#
# The following variables control the behavior of this module:
#
# METIS_INCLUDE_DIR_HINTS: List of additional directories in which to
#                              search for Metis includes.
# METIS_LIBRARY_DIR_HINTS: List of additional directories in which to
#                              search for Metis libraries.

set(METIS_INCLUDE_DIR_HINTS "" CACHE PATH "Metis include directory")
set(METIS_LIBRARY_DIR_HINTS "" CACHE PATH "Metis library directory")

unset(METIS_FOUND)

find_package(metis CONFIG QUIET)
if(TARGET metis)
    set(METIS_FOUND TRUE)
    message(STATUS "Found Metis")
    message(STATUS "  Target : metis")
else()
    list(APPEND METIS_CHECK_INCLUDE_DIRS
        ${METIS_INCLUDE_DIR_HINTS}
        /usr/include
        /usr/local/include
        /opt/include
        /opt/local/include
    )

    list(APPEND METIS_CHECK_LIBRARY_DIRS
        ${METIS_LIBRARY_DIR_HINTS}
        /usr/lib
        /usr/local/lib
        /opt/lib
        /opt/local/lib
    )

    find_path(METIS_INCLUDE_DIRS
        NAMES
        metis.h
        PATHS
        ${METIS_CHECK_INCLUDE_DIRS})
    find_library(METIS_LIBRARIES
        NAMES
        metis
        PATHS
        ${METIS_CHECK_LIBRARY_DIRS})
    find_library(GK_LIBRARIES
        NAMES
        GKlib
        PATHS
        ${METIS_CHECK_LIBRARY_DIRS})

    if(GK_LIBRARIES)
        set(METIS_LIBRARIES ${METIS_LIBRARIES} ${GK_LIBRARIES})
        message(STATUS "Found GKlib")
    endif()

    if(METIS_INCLUDE_DIRS AND METIS_LIBRARIES)
        set(METIS_FOUND TRUE)
        message(STATUS "Found Metis")
        message(STATUS "  Includes : ${METIS_INCLUDE_DIRS}")
        message(STATUS "  Libraries : ${METIS_LIBRARIES}")
    endif()

    add_library(metis INTERFACE IMPORTED)
    target_include_directories(
        metis INTERFACE ${METIS_INCLUDE_DIRS})
    target_link_libraries(
        metis INTERFACE ${METIS_LIBRARIES})
endif()

if(NOT METIS_FOUND AND METIS_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find Metis")
endif()
