# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

cmake_minimum_required(VERSION 3.10)

project(COLMAP LANGUAGES C CXX)

set(COLMAP_VERSION "3.9")
set(COLMAP_VERSION_NUMBER "3900")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set_property(GLOBAL PROPERTY GLOBAL_DEPENDS_NO_CYCLES ON)

################################################################################
# Include CMake dependencies
################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)

# Include helper macros and commands, and allow the included file to override
# the CMake policies in this file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeHelper.cmake NO_POLICY_SCOPE)

# Build position-independent code, so that shared libraries can link against
# COLMAP's static libraries.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# Options
################################################################################

option(SIMD_ENABLED "Whether to enable SIMD optimizations" ON)
option(OPENMP_ENABLED "Whether to enable OpenMP parallelization" ON)
option(IPO_ENABLED "Whether to enable interprocedural optimization" ON)
option(CUDA_ENABLED "Whether to enable CUDA, if available" ON)
option(GUI_ENABLED "Whether to enable the graphical UI" ON)
option(OPENGL_ENABLED "Whether to enable OpenGL, if available" ON)
option(TESTS_ENABLED "Whether to build test binaries" OFF)
option(ASAN_ENABLED "Whether to enable AddressSanitizer flags" OFF)
option(PROFILING_ENABLED "Whether to enable google-perftools linker flags" OFF)
option(CCACHE_ENABLED "Whether to enable compiler caching, if available" ON)
option(CGAL_ENABLED "Whether to enable the CGAL library" ON)

if(TESTS_ENABLED)
    enable_testing()
endif()

################################################################################
# Dependency configuration
################################################################################

set(COLMAP_FIND_QUIETLY FALSE)
include(cmake/FindCOLMAPDependencies.cmake)

################################################################################
# Compiler specific configuration
################################################################################

if(CMAKE_BUILD_TYPE)
    message(STATUS "Build type specified as ${CMAKE_BUILD_TYPE}")
else()
    message(STATUS "Build type not specified, using Release")
    set(CMAKE_BUILD_TYPE Release)
    set(IS_DEBUG OFF)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "ClangTidy")
    find_program(CLANG_TIDY_EXE NAMES clang-tidy)
    if(NOT CLANG_TIDY_EXE)
        message(FATAL_ERROR "Could not find the clang-tidy executable, please set CLANG_TIDY_EXE")
    endif()
else()
    unset(CLANG_TIDY_EXE)
endif()

if(IS_MSVC)
    # Some fixes for the Glog library.
    add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
    add_definitions("-DGL_GLEXT_PROTOTYPES")
    add_definitions("-DNOMINMAX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    # Enable object level parallel builds in Visual Studio.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

if(IS_GNU)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        message(FATAL_ERROR "GCC version 4.8 or older not supported")
    endif()

    # Hide incorrect warnings for uninitialized Eigen variables under GCC.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-maybe-uninitialized")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
endif()

if(IS_DEBUG)
    add_definitions("-DEIGEN_INITIALIZE_MATRICES_BY_NAN")
endif()

if(SIMD_ENABLED)
    message(STATUS "Enabling SIMD support")
else()
    message(STATUS "Disabling SIMD support")
endif()

if(IPO_ENABLED AND NOT IS_DEBUG AND NOT IS_GNU)
    message(STATUS "Enabling interprocedural optimization")
    set_property(DIRECTORY PROPERTY INTERPROCEDURAL_OPTIMIZATION 1)
else()
    message(STATUS "Disabling interprocedural optimization")
endif()

if(ASAN_ENABLED)
    message(STATUS "Enabling ASan support")
    if(IS_CLANG OR IS_GNU)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer -fsanitize-address-use-after-scope)
        add_link_options(-fsanitize=address)
    else()
        message(FATAL_ERROR "Unsupported compiler for ASan mode")
    endif()
endif()

if(CCACHE_ENABLED)
    find_program(CCACHE ccache)
    if(CCACHE)
        message(STATUS "Enabling ccache support")
        set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    else()
        message(STATUS "Disabling ccache support")
    endif()
else()
    message(STATUS "Disabling ccache support")
endif()

if(PROFILING_ENABLED)
    message(STATUS "Enabling profiling support")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lprofiler -ltcmalloc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lprofiler -ltcmalloc")
else()
    message(STATUS "Disabling profiling support")
endif()

################################################################################
# Add sources
################################################################################

# Generate source file with version definitions.
include(GenerateVersionDefinitions)

include_directories(src)
link_directories(${COLMAP_LINK_DIRS})

add_subdirectory(src/colmap)
add_subdirectory(src/thirdparty)

################################################################################
# Generate source groups for Visual Studio, XCode, etc.
################################################################################

COLMAP_ADD_SOURCE_DIR(src/colmap/controllers CONTROLLERS_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/estimators ESTIMATORS_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/exe EXE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/feature FEATURE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/geometry GEOMETRY_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/image IMAGE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/math MATH_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/mvs MVS_SRCS *.h *.cc *.cu)
COLMAP_ADD_SOURCE_DIR(src/colmap/optim OPTIM_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/retrieval RETRIEVAL_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/scene SCENE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/sensor SENSOR_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/sfm SFM_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/tools TOOLS_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/ui UI_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/util UTIL_SRCS *.h *.cc)

COLMAP_ADD_SOURCE_DIR(src/thirdparty/LSD THIRDPARTY_LSD_SRCS *.h *.c)
COLMAP_ADD_SOURCE_DIR(src/thirdparty/PoissonRecon THIRDPARTY_POISSON_RECON_SRCS *.h *.cpp *.inl)
COLMAP_ADD_SOURCE_DIR(src/thirdparty/SiftGPU THIRDPARTY_SIFT_GPU_SRCS *.h *.cpp *.cu)
COLMAP_ADD_SOURCE_DIR(src/thirdparty/VLFeat THIRDPARTY_VLFEAT_SRCS *.h *.c *.tc)

# Add all of the source files to a regular library target, as using a custom
# target does not allow us to set its C++ include directories (and thus
# intellisense can't find any of the included files).
add_library(
    ${COLMAP_SRC_ROOT_FOLDER}
    ${CONTROLLERS_SRCS}
    ${ESTIMATORS_SRCS}
    ${EXE_SRCS}
    ${FEATURE_SRCS}
    ${GEOMETRY_SRCS}
    ${IMAGE_SRCS}
    ${MATH_SRCS}
    ${MVS_SRCS}
    ${OPTIM_SRCS}
    ${RETRIEVAL_SRCS}
    ${SCENE_SRCS}
    ${SENSOR_SRCS}
    ${SFM_SRCS}
    ${TOOLS_SRCS}
    ${UI_SRCS}
    ${UTIL_SRCS}
    ${THIRDPARTY_LSD_SRCS}
    ${THIRDPARTY_POISSON_RECON_SRCS}
    ${THIRDPARTY_SIFT_GPU_SRCS}
    ${THIRDPARTY_VLFEAT_SRCS}
)

# Prevent the library from being compiled automatically.
set_target_properties(
    ${COLMAP_SRC_ROOT_FOLDER} PROPERTIES
    EXCLUDE_FROM_ALL 1
    EXCLUDE_FROM_DEFAULT_BUILD 1)


################################################################################
# Install and uninstall scripts
################################################################################

# Install batch scripts under Windows.
if(IS_MSVC)
    install(FILES "scripts/shell/COLMAP.bat" "scripts/shell/RUN_TESTS.bat"
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
            DESTINATION "/")
endif()

# Install application meny entry under Linux/Unix.
if(UNIX AND NOT APPLE)
    install(FILES "doc/COLMAP.desktop" DESTINATION "share/applications")
endif()

# Configure the uninstallation script.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeUninstall.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake"
               IMMEDIATE @ONLY)
add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake)
set_target_properties(uninstall PROPERTIES FOLDER ${CMAKE_TARGETS_ROOT_FOLDER})

set(COLMAP_EXPORT_LIBS
    # Internal.
    colmap_controllers
    colmap_estimators
    colmap_feature
    colmap_geometry
    colmap_image
    colmap_math
    colmap_mvs
    colmap_optim
    colmap_retrieval
    colmap_scene
    colmap_sensor
    colmap_sfm
    colmap_ui
    colmap_util
    # Third-party.
    colmap_lsd
    colmap_poisson_recon
    colmap_vlfeat
    colmap_sift_gpu
)
if(CUDA_ENABLED)
    list(APPEND COLMAP_EXPORT_LIBS
         colmap_util_cuda
         colmap_mvs_cuda
    )
endif()

# Add unified interface library target to export.
add_library(colmap INTERFACE)
target_link_libraries(colmap INTERFACE ${COLMAP_EXPORT_LIBS})
set(INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")
target_include_directories(
    colmap INTERFACE $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>)

install(
    TARGETS colmap ${COLMAP_EXPORT_LIBS}
    EXPORT colmap-targets
    LIBRARY DESTINATION thirdparty/)

# Generate config and version.
include(CMakePackageConfigHelpers)
set(PACKAGE_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/colmap-config.cmake")
set(INSTALL_CONFIG_DIR "share/colmap")
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/colmap-config.cmake.in ${PACKAGE_CONFIG_FILE}
    INSTALL_DESTINATION ${INSTALL_CONFIG_DIR})
install(FILES ${PACKAGE_CONFIG_FILE} DESTINATION ${INSTALL_CONFIG_DIR})

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/colmap-config-version.cmake.in"
                "${CMAKE_CURRENT_BINARY_DIR}/colmap-config-version.cmake" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/colmap-config-version.cmake"
        DESTINATION "share/colmap")

# Install targets.
install(
    EXPORT colmap-targets
    FILE colmap-targets.cmake
    NAMESPACE colmap::
    DESTINATION ${INSTALL_CONFIG_DIR})

# Install header files.
install(
    DIRECTORY src/colmap
    DESTINATION include
    FILES_MATCHING PATTERN "*.h")
install(
    DIRECTORY src/thirdparty
    DESTINATION include/colmap
    FILES_MATCHING REGEX ".*[.]h|.*[.]hpp|.*[.]inl")

# Install find_package scripts for dependencies.
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    DESTINATION share/colmap
    FILES_MATCHING PATTERN "Find*.cmake")
