cmake_minimum_required(VERSION 3.10)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})

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

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

find_package(pybind11 2.11.1 REQUIRED)

pybind11_add_module(pycolmap  ${PROJECT_SOURCE_DIR}/../src/pycolmap/main.cc)
target_include_directories(pycolmap PRIVATE ${PROJECT_SOURCE_DIR}/../src/)
target_link_libraries(pycolmap PRIVATE colmap::colmap freeimage::FreeImage glog::glog Ceres::ceres)
target_compile_definitions(pycolmap PRIVATE VERSION_INFO="${PROJECT_VERSION}")
install(TARGETS pycolmap LIBRARY DESTINATION .)
