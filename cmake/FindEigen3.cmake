# Sets:
#   EIGEN3_FOUND - TRUE if System has Eigen library with correct version.
#   EIGEN3_INCLUDE_DIRS - The Eigen include directory.
#   EIGEN3_VERSION - Eigen version.

if(NOT Eigen3_FIND_VERSION)
    if(NOT Eigen3_FIND_VERSION_MAJOR)
        set(Eigen3_FIND_VERSION_MAJOR 2)
    endif()

    if(NOT Eigen3_FIND_VERSION_MINOR)
        set(Eigen3_FIND_VERSION_MINOR 91)
    endif()

    if(NOT Eigen3_FIND_VERSION_PATCH)
        set(Eigen3_FIND_VERSION_PATCH 0)
    endif()

    set(Eigen3_FIND_VERSION "${Eigen3_FIND_VERSION_MAJOR}.${Eigen3_FIND_VERSION_MINOR}.${Eigen3_FIND_VERSION_PATCH}")
endif()

macro(_eigen3_check_version)
    file(READ "${EIGEN3_INCLUDE_DIRS}/Eigen/src/Core/util/Macros.h" _eigen3_version_header)

    string(REGEX MATCH "define[ \t]+EIGEN_WORLD_VERSION[ \t]+([0-9]+)" _eigen3_world_version_match "${_eigen3_version_header}")
    set(EIGEN3_WORLD_VERSION "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define[ \t]+EIGEN_MAJOR_VERSION[ \t]+([0-9]+)" _eigen3_major_version_match "${_eigen3_version_header}")
    set(EIGEN3_MAJOR_VERSION "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define[ \t]+EIGEN_MINOR_VERSION[ \t]+([0-9]+)" _eigen3_minor_version_match "${_eigen3_version_header}")
    set(EIGEN3_MINOR_VERSION "${CMAKE_MATCH_1}")

    set(EIGEN3_VERSION ${EIGEN3_WORLD_VERSION}.${EIGEN3_MAJOR_VERSION}.${EIGEN3_MINOR_VERSION})
    if(${EIGEN3_VERSION} VERSION_LESS ${Eigen3_FIND_VERSION})
        set(EIGEN3_VERSION_OK FALSE)
    else()
        set(EIGEN3_VERSION_OK TRUE)
    endif()

    if(NOT EIGEN3_VERSION_OK)
        message(STATUS "Eigen3 version ${EIGEN3_VERSION} found in ${EIGEN3_INCLUDE_DIRS}, "
                       "but at least version ${Eigen3_FIND_VERSION} is required.")
    endif()
endmacro()

if(EIGEN3_INCLUDE_DIRS)
    _eigen3_check_version()
    set(EIGEN3_FOUND ${EIGEN3_VERSION_OK})
else()
    find_path(EIGEN3_INCLUDE_DIRS
        NAMES
        signature_of_eigen3_matrix_library
        PATHS
        /usr/include
        /usr/local/include
        /opt/include
        /opt/local/include
        PATH_SUFFIXES
        eigen3
        eigen
    )

    if(EIGEN3_INCLUDE_DIRS)
        _eigen3_check_version()
    endif()

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(Eigen3 DEFAULT_MSG EIGEN3_INCLUDE_DIRS EIGEN3_VERSION_OK)

    mark_as_advanced(EIGEN3_INCLUDE_DIRS)
endif()

if(EIGEN3_FOUND)
    message(STATUS "Found Eigen3")
    message(STATUS "  Includes : ${EIGEN3_INCLUDE_DIRS}")
else()
    if(Eigen3_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find Eigen3")
    endif()
endif()
