cmake_minimum_required(VERSION 3.10)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})
option(GENERATE_STUBS "Whether to generate stubs" ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Some fixes for the Glog library.
    add_compile_definitions(GLOG_NO_ABBREVIATED_SEVERITIES)
    add_compile_definitions(GL_GLEXT_PROTOTYPES)
    add_compile_definitions(NOMINMAX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    # Enable object level parallel builds in Visual Studio.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

find_package(colmap REQUIRED)

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
target_link_libraries(_core PRIVATE colmap::colmap glog::glog Ceres::ceres)
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
