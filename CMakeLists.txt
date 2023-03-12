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

cmake_minimum_required(VERSION 3.1)

project(COLMAP LANGUAGES C CXX)

set(COLMAP_VERSION "3.9")
set(COLMAP_VERSION_NUMBER "3900")

################################################################################
# Include CMake dependencies
################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)

# Include helper macros and commands, and allow the included file to override
# the CMake policies in this file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeHelper.cmake NO_POLICY_SCOPE)


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
option(BOOST_STATIC "Whether to enable static boost library linker flags" ON)

if(TESTS_ENABLED)
    enable_testing()
endif()

if(BOOST_STATIC)
    set(Boost_USE_STATIC_LIBS ON)
else()
    add_definitions("-DBOOST_TEST_DYN_LINK")
endif()

# Build position-independent code, so that shared libraries can link against
# COLMAP's static libraries.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# Find packages
################################################################################

if(OPENMP_ENABLED)
    find_package(OpenMP QUIET)
endif()

find_package(Ceres REQUIRED)

find_package(Boost REQUIRED COMPONENTS
             program_options
             filesystem
             graph
             system
             unit_test_framework)

find_package(Eigen3 REQUIRED)

find_package(FreeImage REQUIRED)

find_package(FLANN REQUIRED)
find_package(LZ4 REQUIRED)

find_package(Metis REQUIRED)

find_package(Glog REQUIRED)

find_package(SQLite3 REQUIRED)

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(Glew REQUIRED)
find_package(Git)

if(CGAL_ENABLED)
    set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)
    # We do not use CGAL data. This prevents an unnecessary warning by CMake.
    set(CGAL_DATA_DIR "unused")
    find_package(CGAL QUIET)
endif()

set(CUDA_MIN_VERSION "7.0")
if(CUDA_ENABLED)
    if(CMAKE_VERSION VERSION_LESS 3.17)
        find_package(CUDA QUIET)
        if(CUDA_FOUND)
            message(STATUS "Found CUDA version ${CUDA_VERSION} installed in "
                    "${CUDA_TOOLKIT_ROOT_DIR} via legacy CMake (<3.17) module. "
                    "Using the legacy CMake module means that any installation of "
                    "COLMAP will require that the CUDA libraries are "
                    "available under LD_LIBRARY_PATH.")
            enable_language(CUDA)

            macro(declare_imported_cuda_target module)
                add_library(CUDA::${module} INTERFACE IMPORTED)
                target_include_directories(
                    CUDA::${module} INTERFACE ${CUDA_INCLUDE_DIRS})
                target_link_libraries(
                    CUDA::${module} INTERFACE ${CUDA_${module}_LIBRARY} ${ARGN})
            endmacro()

            declare_imported_cuda_target(cudart ${CUDA_LIBRARIES})
            declare_imported_cuda_target(curand ${CUDA_LIBRARIES})
            
            set(CUDAToolkit_VERSION "${CUDA_VERSION_STRING}")
            set(CUDAToolkit_BIN_DIR "${CUDA_TOOLKIT_ROOT_DIR}/bin")
        endif()
    else()
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(CUDA_FOUND ON)
            enable_language(CUDA)
        endif()
    endif()
endif()

if(GUI_ENABLED)
    find_package(Qt5 5.4 COMPONENTS Core OpenGL Widgets)
    if(Qt5_FOUND)
        message(STATUS "Found Qt")
        message(STATUS "  Module : ${Qt5Core_DIR}")
        message(STATUS "  Module : ${Qt5OpenGL_DIR}")
        message(STATUS "  Module : ${Qt5Widgets_DIR}")
    else()
        set(GUI_ENABLED OFF)
    endif()
endif()

if(OPENGL_ENABLED AND NOT GUI_ENABLED)
    message(STATUS "Disabling GUI will also disable OpenGL")
    set(OPENGL_ENABLED OFF)
endif()

if(CGAL_FOUND)
    list(APPEND CGAL_LIBRARY ${CGAL_LIBRARIES})
    message(STATUS "Found CGAL")
    message(STATUS "  Includes : ${CGAL_INCLUDE_DIRS}")
    message(STATUS "  Libraries : ${CGAL_LIBRARY}")
endif()


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

if(IS_MSVC)
    # Some fixes for the Glog library.
    add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
    add_definitions("-DGL_GLEXT_PROTOTYPES")
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

if(OPENMP_ENABLED AND OPENMP_FOUND)
    message(STATUS "Enabling OpenMP support")
    add_definitions("-DOPENMP_ENABLED")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "Disabling OpenMP support")
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

if(CUDA_ENABLED AND CUDA_FOUND)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        message(
            FATAL_ERROR "You must set CMAKE_CUDA_ARCHITECTURES to e.g. 'native', 'all-major', '70', etc. "
            "More information at https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html")
    endif()

    add_definitions("-DCUDA_ENABLED")

    # Fix for some combinations of CUDA and GCC (e.g. under Ubuntu 16.04).
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FORCE_INLINES")
    # Do not show warnings if the architectures are deprecated.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
    # Explicitly set PIC flags for CUDA targets.
    if(NOT IS_MSVC)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fPIC")
    endif()

    message(STATUS "Enabling CUDA support (version: ${CUDAToolkit_VERSION}, "
                    "archs: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    set(CUDA_ENABLED OFF)
    message(STATUS "Disabling CUDA support")
endif()

if(GUI_ENABLED AND Qt5_FOUND)
    add_definitions("-DGUI_ENABLED")
    message(STATUS "Enabling GUI support")
else()
    message(STATUS "Disabling GUI support")
endif()

if(OPENGL_ENABLED)
    add_definitions("-DOPENGL_ENABLED")
    message(STATUS "Enabling OpenGL support")
else()
    message(STATUS "Disabling OpenGL support")
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

if(CGAL_FOUND AND CGAL_ENABLED)
    message(STATUS "Enabling CGAL support")
    add_definitions("-DCGAL_ENABLED")
else()
    message(STATUS "Disabling CGAL support")
    set(CGAL_ENABLED OFF)
endif()

if(Qt5_FOUND)
    # Qt5 was built with -reduce-relocations.
    if(Qt5_POSITION_INDEPENDENT_CODE)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        # Workaround for Qt5 CMake config bug under Ubuntu 20.04: https://gitlab.kitware.com/cmake/cmake/-/issues/16915
        if(${Qt5_VERSION} VERSION_EQUAL "5.12.8" AND TARGET Qt5::Core)
            get_property(core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
            string(REPLACE "-fPIC" "" new_qt5_core_options ${core_options})
            set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_qt5_core_options})
            set_property(TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON")
            if(NOT IS_MSVC)
                set(CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC")
            endif()
        endif()
    endif()

    # Enable automatic compilation of Qt resource files.
    set(CMAKE_AUTORCC ON)
endif()

################################################################################
# Add sources
################################################################################

# Generate source file with version definitions.
include(GenerateVersionDefinitions)

set(COLMAP_INCLUDE_DIRS
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${FLANN_INCLUDE_DIRS}
    ${LZ4_INCLUDE_DIRS}
    ${FREEIMAGE_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${METIS_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${SQLite3_INCLUDE_DIRS}
)

set(COLMAP_LINK_DIRS
    ${Boost_LIBRARY_DIRS}
)

set(COLMAP_EXTERNAL_LIBRARIES
    ${CMAKE_DL_LIBS}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${GLOG_LIBRARIES}
    ${FLANN_LIBRARIES}
    ${LZ4_LIBRARIES}
    ${FREEIMAGE_LIBRARIES}
    ${METIS_LIBRARIES}
    ${CERES_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${SQLite3_LIBRARIES}
)

if(OPENMP_FOUND)
    list(APPEND COLMAP_EXTERNAL_LIBRARIES ${OpenMP_libomp_LIBRARY})
endif()

if(Qt5_FOUND)
    list(APPEND COLMAP_INCLUDE_DIRS ${Qt5Core_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS})
    list(APPEND COLMAP_EXTERNAL_LIBRARIES ${Qt5Core_LIBRARIES} ${Qt5OpenGL_LIBRARIES} ${Qt5Widgets_LIBRARIES})
endif()

if(CGAL_FOUND)
    list(APPEND COLMAP_INCLUDE_DIRS ${CGAL_INCLUDE_DIRS} ${GMP_INCLUDE_DIR})
    list(APPEND COLMAP_EXTERNAL_LIBRARIES ${CGAL_LIBRARY} ${GMP_LIBRARIES})
    list(APPEND COLMAP_LINK_DIRS ${CGAL_LIBRARIES_DIR})
endif()

if(UNIX)
    list(APPEND COLMAP_EXTERNAL_LIBRARIES pthread)
endif()

set(COLMAP_INTERNAL_LIBRARIES
    lsd
    poisson_recon
    sift_gpu
    vlfeat
)

include_directories(
    lib
    src
    ${COLMAP_INCLUDE_DIRS}
)

link_directories(${COLMAP_LINK_DIRS})

add_subdirectory(lib)
add_subdirectory(src)


################################################################################
# Generate source groups for Visual Studio, XCode, etc.
################################################################################

COLMAP_ADD_SOURCE_DIR(lib/LSD LIB_LSD_SRCS *.h *.c)
COLMAP_ADD_SOURCE_DIR(lib/PoissonRecon LIB_POISSON_RECON_SRCS *.h *.cpp *.inl)
COLMAP_ADD_SOURCE_DIR(lib/SiftGPU LIB_SIFT_GPU_SRCS *.h *.cpp *.cu)
COLMAP_ADD_SOURCE_DIR(lib/VLFeat LIB_VLFEAT_SRCS *.h *.c *.tc)

COLMAP_ADD_SOURCE_DIR(src/base BASE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/controllers CONTROLLERS_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/estimators ESTIMATORS_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/exe EXE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/feature FEATURE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/mvs MVS_SRCS *.h *.cc *.cu)
COLMAP_ADD_SOURCE_DIR(src/optim OPTIM_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/retrieval RETRIEVAL_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/sfm SFM_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/tools TOOLS_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/ui UI_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/util UTIL_SRCS *.h *.cc)

# Add all of the source files to a regular library target, as using a custom
# target does not allow us to set its C++ include directories (and thus
# intellisense can't find any of the included files).
add_library(
    ${COLMAP_SRC_ROOT_FOLDER}
    ${LIB_LSD_SRCS}
    ${LIB_POISSON_RECON_SRCS}
    ${LIB_SIFT_GPU_SRCS}
    ${LIB_VLFEAT_SRCS}
    ${BASE_SRCS}
    ${CONTROLLERS_SRCS}
    ${ESTIMATORS_SRCS}
    ${EXE_SRCS}
    ${FEATURE_SRCS}
    ${MVS_SRCS}
    ${OPTIM_SRCS}
    ${RETRIEVAL_SRCS}
    ${SFM_SRCS}
    ${TOOLS_SRCS}
    ${UI_SRCS}
    ${UTIL_SRCS}
)

# Prevent the library from being compiled automatically.
set_target_properties(
    ${COLMAP_SRC_ROOT_FOLDER} PROPERTIES
    EXCLUDE_FROM_ALL 1
    EXCLUDE_FROM_DEFAULT_BUILD 1)


################################################################################
# Install and uninstall scripts
################################################################################

# Install header files.
install(DIRECTORY src/
        DESTINATION include/colmap
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY lib/
        DESTINATION include/colmap/lib
        FILES_MATCHING REGEX ".*[.]h|.*[.]hpp|.*[.]inl")

# Generate and install CMake configuration.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeConfig.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/COLMAPConfig.cmake" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/COLMAPConfig.cmake"
        DESTINATION "share/colmap")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeConfigVersion.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/COLMAPConfigVersion.cmake" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/COLMAPConfigVersion.cmake"
        DESTINATION "share/colmap")

# Install find_package scripts for dependencies.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake
        DESTINATION share/colmap
        FILES_MATCHING PATTERN "Find*.cmake")

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
