# Sets:
#   GLEW_FOUND: TRUE if Glew is found.
#   GLEW_INCLUDE_DIRS: Include directories for Glew.
#   GLEW_LIBRARIES: Libraries required to link Glew.
#
# The following variables control the behavior of this module:
#
# GLEW_INCLUDE_DIR_HINTS: List of additional directories in which to
#                         search for Glew includes.
# GLEW_LIBRARY_DIR_HINTS: List of additional directories in which to
#                         search for Glew libraries.

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
    if(Glew_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find Glew")
    endif()
endif()
