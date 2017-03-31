cmake_minimum_required(VERSION 2.8)

project(COLMAP)

set(COLMAP_VERSION "2.1")


################################################################################
# Include CMake dependencies
################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)

# Include helper macros and commands, and allow the included file to override
# the CMake policies in this file
include(${CMAKE_SOURCE_DIR}/cmake/CMakeHelper.cmake NO_POLICY_SCOPE)

include(${CMAKE_SOURCE_DIR}/cmake/CheckSSEExtensions.cmake)

if(EXISTS ${CMAKE_SOURCE_DIR}/LocalConfig.cmake)
    include(${CMAKE_SOURCE_DIR}/LocalConfig.cmake)
endif()


################################################################################
# Options
################################################################################

option(OPENMP_ENABLED "Whether to enable OpenMP" ON)
option(LTO_ENABLED "Whether to enable link-time optimization" ON)
option(CUDA_ENABLED "Whether to enable CUDA, if available" ON)
option(TESTS_ENABLED "Whether to build test binaries" ON)
option(PROFILING_ENABLED "Whether to enable google-perftools linker flags" OFF)
option(BOOST_STATIC "Whether to enable static boost library linker flags" ON)

if(TESTS_ENABLED)
    enable_testing()
endif()

if(BOOST_STATIC)
    set(Boost_USE_STATIC_LIBS ON)
else()
    add_definitions(-DBOOST_TEST_DYN_LINK)
endif()

################################################################################
# Find packages
################################################################################

if(OPENMP_ENABLED)
    find_package(OpenMP QUIET)
endif()

find_package(Boost COMPONENTS
             program_options
             filesystem
             regex
             system
             unit_test_framework
             REQUIRED)

find_package(Eigen3 REQUIRED)

find_package(FreeImage REQUIRED)

find_package(Glog REQUIRED)

find_package(Ceres REQUIRED)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(Glew REQUIRED)
find_package(Git REQUIRED)

if(CUDA_ENABLED)
    find_package(CUDA QUIET)
endif()

find_package(Qt5 REQUIRED)


################################################################################
# Compiler specific configuration
################################################################################

if(CMAKE_BUILD_TYPE)
    message(STATUS "Build type specified as ${CMAKE_BUILD_TYPE}")
else()
    message(STATUS "Build type not specified, using RelWithDebInfo")
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
    set(IS_DEBUG TRUE)
endif()

if(IS_MSVC)
    add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
    add_definitions("-DGL_GLEXT_PROTOTYPES")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

if(IS_GNU)
    # Hide incorrect warnings for uninitialized Eigen variables under GCC.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-maybe-uninitialized")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
endif()

if(IS_DEBUG)
    add_definitions("-DEIGEN_INITIALIZE_MATRICES_BY_NAN")
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
        message(STATUS "Enabling CUDA support (Version ${CUDA_VERSION_STRING})")
    else()
        set(CUDA_FOUND FALSE)
        message(STATUS "Disabling CUDA support")
    endif()
else()
    set(CUDA_ENABLED FALSE)
    message(STATUS "Disabling CUDA support")
endif()

if (CUDA_ENABLED)
    # Configure the NVCC flags:
    #   CUDA >= 5.0 <6.5:
    #     SASS:   sm_20, sm_30, sm_35
    #     PTX:    compute_35
    #   CUDA == 6.5:
    #     SASS:   sm_20, sm_30, sm_35, sm_37 sm_50
    #     PTX:    compute_50
    #   CUDA >= 7.0
    #     SASS:   sm_20, sm_30, sm_35, sm_37, sm_50, sm_52
    #     PTX:    compute_52
    #   CUDA >= 8.0
    #     SASS:   sm_20, sm_30, sm_35, sm_37, sm_50, sm_52, sm_60, sm_61
    #     PTX:    compute_60, compute_61

    list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")

    # Add flags to generate SASS code.

    list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_20,code=sm_20")
    list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_30,code=sm_30")
    list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_35,code=sm_35")

    if(NOT CUDA_VERSION VERSION_LESS "6.5") # >= 6.5
        list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_37,code=sm_37")
        list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_50,code=sm_50")
    endif()

    if(NOT CUDA_VERSION VERSION_LESS "7.0") # >= 7.0
        list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_52,code=sm_52")
    endif()

    if(NOT CUDA_VERSION VERSION_LESS "8.0") # >= 8.0
        list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_60,code=sm_60")
        list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_61,code=sm_61")
    endif()

    # Add flags to generate PTX code.

    if(CUDA_VERSION VERSION_LESS "6.5")
        list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_35,code=compute_35")
    elseif(CUDA_VERSION VERSION_LESS "7.0")
        list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_50,code=compute_50")
    elseif(CUDA_VERSION VERSION_LESS "8.0")
        list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_52,code=compute_52")
    else() # >= 8.0
        list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_60,code=compute_60")
        list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_61,code=compute_61")
    endif()
endif()

if(PROFILING_ENABLED)
    message(STATUS "Enabling profiling support")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lprofiler -ltcmalloc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lprofiler -ltcmalloc")
else()
    message(STATUS "Disabling profiling support")
endif()

# Qt5 was built with -reduce-relocations.
if(Qt5_POSITION_INDEPENDENT_CODE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()


################################################################################
# Add sources
################################################################################

include(GenerateVersionDefinitions)

add_subdirectory(src)


################################################################################
# Uninstall script
################################################################################

configure_file("${CMAKE_SOURCE_DIR}/cmake/CMakeUninstall.cmake"
               "${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake"
               IMMEDIATE @ONLY)
add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake)
set_target_properties(uninstall PROPERTIES FOLDER ${CMAKE_TARGETS_ROOT_FOLDER})
