cmake_minimum_required(VERSION 2.8)

project(COLMAP)
add_definitions("-DCOLMAP_VERSION=\"1.1\"")


################################################################################
# Include CMake dependencies
################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)

# Include helper macros and commands, and allow the included file to override
# the CMake policies in this file
include(${CMAKE_SOURCE_DIR}/cmake/cmake_helper.cmake NO_POLICY_SCOPE)

if(EXISTS ${CMAKE_SOURCE_DIR}/LocalConfig.cmake)
    include(${CMAKE_SOURCE_DIR}/LocalConfig.cmake)
endif()


################################################################################
# Options
################################################################################

enable_testing()

option(OPENMP_ENABLED "Whether to enable OpenMP" ON)
option(LTO_ENABLED "Whether to enable link-time optimization" ON)
option(CUDA_ENABLED "Whether to enable CUDA, if available" ON)

set(Boost_USE_STATIC_LIBS ON)


################################################################################
# Find packages
################################################################################

if(OPENMP_ENABLED)
    find_package(OpenMP QUIET)
endif()

find_package(Boost COMPONENTS
             program_options
             filesystem
             system
             chrono
             regex
             unit_test_framework
             REQUIRED)

find_package(Eigen3 REQUIRED)

find_package(FreeImage REQUIRED)

find_package(Glog REQUIRED)

find_package(Ceres REQUIRED)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(Glew REQUIRED)

if(CUDA_ENABLED)
    find_package(CUDA QUIET)
endif()

find_package(Qt5Core REQUIRED)
find_package(Qt5OpenGL REQUIRED)


################################################################################
# Compiler specific configuration
################################################################################

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not specified, using RelWithDebInfo")
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
    set(IS_DEBUG TRUE)
endif()

if(IS_MSVC)
    add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
    add_definitions("-DGL_GLEXT_PROTOTYPES")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
endif()


if(OPENMP_ENABLED AND OPENMP_FOUND)
    message(STATUS "Enabling OpenMP support")
    add_definitions("-DOPENMP_ENABLED")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "Disabling OpenMP support")
endif()

if(LTO_ENABLED AND NOT IS_DEBUG AND NOT IS_GNU)
    CHECK_CXX_COMPILER_FLAG("-flto" HAS_LTO)
    if(HAS_LTO)
        message(STATUS "Enabling link-time optimization (-flto)")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
    endif()
else()
    message(STATUS "Disabling link-time optimization (-flto)")
endif()

if(CUDA_FOUND)
    if(CUDA_ENABLED)
        add_definitions("-DCUDA_ENABLED")
        message(STATUS "Enabling CUDA support")
    else()
        set(CUDA_FOUND FALSE)
        message(STATUS "Disabling CUDA support")
    endif()
else()
    set(CUDA_ENABLED FALSE)
    message(STATUS "Disabling CUDA support")
endif()

# Qt5 was built with -reduce-relocations.
if(Qt5_POSITION_INDEPENDENT_CODE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()


################################################################################
# Add sources
################################################################################

add_subdirectory(src)


################################################################################
# Uninstall script
################################################################################

configure_file("${CMAKE_SOURCE_DIR}/cmake/cmake_uninstall.cmake"
               "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
               IMMEDIATE @ONLY)
add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
set_target_properties(uninstall PROPERTIES FOLDER ${CMAKE_TARGETS_ROOT_FOLDER})
