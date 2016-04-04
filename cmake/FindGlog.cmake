# Sets:
#   GLOG_FOUND: TRUE if Glog is found.
#   GLOG_INCLUDE_DIRS: Include directories for Glog.
#   GLOG_LIBRARIES: Libraries required to link Glog.
#
# The following variables control the behavior of this module:
#
# GLOG_INCLUDE_DIR_HINTS: List of additional directories in which to
#                         search for Glog includes.
# GLOG_LIBRARY_DIR_HINTS: List of additional directories in which to
#                         search for Glog libraries.

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

if (GLOG_INCLUDE_DIRS AND GLOG_LIBRARIES)
    set(GLOG_FOUND TRUE)
    message(STATUS "Found Glog")
    message(STATUS "  Includes : ${GLOG_INCLUDE_DIRS}")
    message(STATUS "  Libraries : ${GLOG_LIBRARIES}")
else()
    if(Glog_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find Glog")
    endif()
endif()
