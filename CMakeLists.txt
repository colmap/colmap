cmake_minimum_required(VERSION 2.8.11)

project(COLMAP)

set(COLMAP_VERSION "3.4")
set(COLMAP_VERSION_NUMBER "3400")


################################################################################
# Include CMake dependencies
################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)

# Include helper macros and commands, and allow the included file to override
# the CMake policies in this file
include(${CMAKE_SOURCE_DIR}/cmake/CMakeHelper.cmake NO_POLICY_SCOPE)

if(EXISTS ${CMAKE_SOURCE_DIR}/LocalConfig.cmake)
    include(${CMAKE_SOURCE_DIR}/LocalConfig.cmake)
endif()


################################################################################
# Options
################################################################################

option(SIMD_ENABLED "Whether to enable SIMD optimizations" ON)
option(OPENMP_ENABLED "Whether to enable OpenMP parallelization" ON)
option(IPO_ENABLED "Whether to enable interprocedural optimization" ON)
option(CUDA_ENABLED "Whether to enable CUDA, if available" ON)
option(OPENGL_ENABLED "Whether to enable OpenGL, if available" ON)
option(TESTS_ENABLED "Whether to build test binaries" OFF)
option(PROFILING_ENABLED "Whether to enable google-perftools linker flags" OFF)
option(BOOST_STATIC "Whether to enable static boost library linker flags" ON)
option(CUDA_MULTI_ARCH "Whether to generate CUDA code for multiple architectures" OFF)

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
find_package(Glew REQUIRED)
find_package(Git)

set(CUDA_MIN_VERSION "7.0")
if(CUDA_ENABLED)
    find_package(CUDA ${CUDA_MIN_VERSION} QUIET)
endif()

find_package(Qt5 5.4 REQUIRED)


################################################################################
# Compiler specific configuration
################################################################################

if(CMAKE_BUILD_TYPE)
    message(STATUS "Build type specified as ${CMAKE_BUILD_TYPE}")
else()
    message(STATUS "Build type not specified, using Release")
    set(CMAKE_BUILD_TYPE Release)
    set(IS_DEBUG FALSE)
endif()

if(IS_MSVC)
    add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
    add_definitions("-DGL_GLEXT_PROTOTYPES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
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

if(CUDA_FOUND)
    if(CUDA_ENABLED)
        add_definitions("-DCUDA_ENABLED")

        include(${CMAKE_SOURCE_DIR}/cmake/SelectCudaComputeArch.cmake)

        if(CUDA_MULTI_ARCH)
            CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS All)
        else()
            CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS Auto)
        endif()

        list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")
        list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})

        message(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
                       " archs: ${CUDA_ARCH_FLAGS_readable})")
    else()
        set(CUDA_FOUND FALSE)
        message(STATUS "Disabling CUDA support")
    endif()
else()
    set(CUDA_ENABLED FALSE)
    if(CUDA_VERSION_STRING)
        message(STATUS "Disabling CUDA support (found version "
                "${CUDA_VERSION_STRING} but >= ${CUDA_MIN_VERSION} required)")
    else()
        message(STATUS "Disabling CUDA support")
    endif()
endif()

if(OPENGL_ENABLED)
    add_definitions("-DOPENGL_ENABLED")
    message(STATUS "Enabling OpenGL support")
else()
    message(STATUS "Disabling OpenGL support")
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
    if(CUDA_ENABLED AND NOT IS_MSVC)
        list(APPEND CUDA_NVCC_FLAGS "--compiler-options -fPIC")
    endif()
endif()

# Set the Qt modules for COLMAP.
set(COLMAP_QT_MODULES Core Widgets OpenGL)


################################################################################
# Add sources
################################################################################

include(GenerateVersionDefinitions)

include_directories(
    lib
    src
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${FREEIMAGE_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${Qt5_INCLUDE_DIRS}
)

link_directories(
    ${Boost_LIBRARY_DIRS}
)

add_subdirectory(lib)
add_subdirectory(src)


# This macro will search for source files in a given directory, will add them
# to a source group (folder within a project), and will then return paths to
# each of the found files. The usage of the macro is as follows:
# ADD_SOURCE_DIR(
#     <source directory to search>
#     <output variable with found source files>
#     <search expressions such as *.h *.cc>)
macro(ADD_SOURCE_DIR SRC_DIR SRC_VAR)
    # Create the list of expressions to be used in the search.
    set(GLOB_EXPRESSIONS "")
    foreach(ARG ${ARGN})
        list(APPEND GLOB_EXPRESSIONS ${SRC_DIR}/${ARG})
    endforeach()
    # Perform the search for the source files.
    file(GLOB ${SRC_VAR} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ${GLOB_EXPRESSIONS})
    # Create the source group.
    string(REPLACE "/" "\\" GROUP_NAME ${SRC_DIR})
    source_group(${GROUP_NAME} FILES ${${SRC_VAR}})
    # Clean-up.
    unset(GLOB_EXPRESSIONS)
    unset(ARG)
    unset(GROUP_NAME)
endmacro(ADD_SOURCE_DIR)

ADD_SOURCE_DIR(src/base BASE_SRC *.h *.cc)
ADD_SOURCE_DIR(src/controllers CONTROLLERS_SRC *.h *.cc)
ADD_SOURCE_DIR(src/estimators ESTIMATORS_SRC *.h *.cc)
ADD_SOURCE_DIR(src/exe EXE_SRC *.h *.cc)
ADD_SOURCE_DIR(lib/FLANN EXT_FLANN_SRC *.h *.cpp *.hpp *.cu)
ADD_SOURCE_DIR(lib/Graclus EXT_GRACLUS_SRC *.h *.c)
ADD_SOURCE_DIR(lib/LSD EXT_LSD_SRC *.h *.c)
ADD_SOURCE_DIR(lib/PBA EXT_PBA_SRC *.h *.cpp *.cu)
ADD_SOURCE_DIR(lib/PoissonRecon EXT_POISSON_RECON_SRC *.h *.cpp *.inl)
ADD_SOURCE_DIR(lib/SiftGPU EXT_SIFT_GPU_SRC *.h *.cpp *.cu)
ADD_SOURCE_DIR(lib/SQLite EXT_SQLITE_SRC *.h *.c)
ADD_SOURCE_DIR(lib/VLFeat EXT_VLFEAT_SRC *.h *.c *.tc)
ADD_SOURCE_DIR(src/feature FEATURE_SRC *.h *.cc)
ADD_SOURCE_DIR(src/mvs MVS_SRC *.h *.cc *.cu)
ADD_SOURCE_DIR(src/optim OPTIM_SRC *.h *.cc)
ADD_SOURCE_DIR(src/retrieval RETRIEVAL_SRC *.h *.cc)
ADD_SOURCE_DIR(src/sfm SFM_SRC *.h *.cc)
ADD_SOURCE_DIR(src/tools TOOLS_SRC *.h *.cc)
ADD_SOURCE_DIR(src/ui UI_SRC *.h *.cc)
ADD_SOURCE_DIR(src/util UTIL_SRC *.h *.cc)

# Add all of the source files to a regular library target, as using a custom
# target does not allow us to set its C++ include directories (and thus
# intellisense can't find any of the included files).
add_library(
    ${COLMAP_SRC_ROOT_FOLDER}
    ${BASE_SRC}
    ${CONTROLLERS_SRC}
    ${ESTIMATORS_SRC}
    ${EXE_SRC}
    ${EXT_FLANN_SRC}
    ${EXT_GRACLUS_SRC}
    ${EXT_LSD_SRC}
    ${EXT_PBA_SRC}
    ${EXT_POISSON_RECON_SRC}
    ${EXT_SIFT_GPU_SRC}
    ${EXT_SQLITE_SRC}
    ${EXT_VLFEAT_SRC}
    ${FEATURE_SRC}
    ${MVS_SRC}
    ${OPTIM_SRC}
    ${RETRIEVAL_SRC}
    ${SFM_SRC}
    ${TOOLS_SRC}
    ${UI_SRC}
    ${UTIL_SRC})

# Prevent the library from being compiled automatically.
set_target_properties(
    ${COLMAP_SRC_ROOT_FOLDER} PROPERTIES
    EXCLUDE_FROM_ALL 1
    EXCLUDE_FROM_DEFAULT_BUILD 1)


################################################################################
# Install/uninstall script
################################################################################

if(IS_MSVC)
    install(FILES "scripts/shell/COLMAP.bat" "scripts/shell/RUN_TESTS.bat"
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
            DESTINATION "/")
endif()

if(UNIX AND NOT APPLE)
    install(FILES "doc/COLMAP.desktop" DESTINATION "share/applications")
endif()

configure_file("${CMAKE_SOURCE_DIR}/cmake/CMakeUninstall.cmake"
               "${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake"
               IMMEDIATE @ONLY)
add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake)
set_target_properties(uninstall PROPERTIES FOLDER ${CMAKE_TARGETS_ROOT_FOLDER})
