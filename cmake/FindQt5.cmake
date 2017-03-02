# Sets:
#   Qt5_FOUND: TRUE if FreeImage is found.
#   Qt5_INCLUDE_DIRS: Include directories for FreeImage.
#   Qt5_LIBRARIES: Libraries required to link FreeImage.

find_package(Qt5Core QUIET)
find_package(Qt5OpenGL QUIET)

if(Qt5Core_FOUND AND Qt5OpenGL_FOUND)
    set(Qt5_FOUND TRUE)
    set(Qt5_INCLUDE_DIRS "${Qt5Core_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS}")
    set(Qt5_LIBRARIES "${Qt5Core_LIBRARIES} ${Qt5OpenGL_LIBRARIES}")
endif()

if(Qt5_FOUND)
    message(STATUS "Found Qt ${Qt5Core_VERSION_STRING}")
else()
    if(Qt5_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find Qt5")
    endif()
endif()
