cmake_minimum_required(VERSION 3.10)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})
option(GENERATE_STUBS "Whether to generate stubs" ON)

# Qt5 configuration
if(NOT DEFINED Qt5_DIR)
    execute_process(
        COMMAND brew --prefix qt5
        OUTPUT_VARIABLE BREW_QT5_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE BREW_RESULT
    )
    
    if(BREW_RESULT EQUAL 0 AND EXISTS "${BREW_QT5_PREFIX}")
        set(Qt5_DIR "${BREW_QT5_PREFIX}/lib/cmake/Qt5" CACHE PATH "Path to Qt5 CMake config")
        message(STATUS "Setting Qt5_DIR to: ${Qt5_DIR}")
    else()
        message(WARNING "Could not determine Qt5 path from brew. Please set Qt5_DIR manually if needed.")
    endif()
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(APPLE)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Some fixes for the Glog library.
    add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
    add_definitions("-DGL_GLEXT_PROTOTYPES")
    add_definitions("-DNOMINMAX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    # Enable object level parallel builds in Visual Studio.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

find_package(colmap REQUIRED)
find_package(PoseLib REQUIRED)
find_package(OpenMP REQUIRED)

if (CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()
find_package(Python REQUIRED COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

find_package(pybind11 2.13.0 REQUIRED)

file(GLOB_RECURSE SOURCE_FILES "${PROJECT_SOURCE_DIR}/../src/pycolmap/*.cc")
pybind11_add_module(_core ${SOURCE_FILES})
target_include_directories(_core PRIVATE ${PROJECT_SOURCE_DIR}/../src/)
target_link_libraries(_core PRIVATE colmap::colmap glog::glog Ceres::ceres PoseLib::PoseLib OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
target_compile_definitions(_core PRIVATE VERSION_INFO="${PROJECT_VERSION}")
install(TARGETS _core LIBRARY DESTINATION pycolmap)

if(GENERATE_STUBS AND UNIX)
    message(STATUS "Enabling stubs generation")
    set(STUBGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/_core")
    add_custom_command(
        TARGET _core POST_BUILD
        COMMAND
          "${CMAKE_COMMAND}" -E env
          "PYTHONPATH=$<TARGET_FILE_DIR:_core>:$ENV{PYTHONPATH}"
          bash ${PROJECT_SOURCE_DIR}/generate_stubs.sh "${Python_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating pybind11 stubs"
        VERBATIM
    )
    install(DIRECTORY ${STUBGEN_OUTPUT_DIR} DESTINATION pycolmap)
endif()

find_package(Eigen3 REQUIRED)
target_link_libraries(_core PRIVATE Eigen3::Eigen)
