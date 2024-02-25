cmake_minimum_required(VERSION 3.10)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
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

set(PYCOLMAP_DIR ${PROJECT_SOURCE_DIR}/../src/pycolmap)

set(PYCOLMAP_LIBRARIES "")
macro(PYCOLMAP_ADD_LIBRARY LIB_NAME)
    set(LIB_PATH "${PYCOLMAP_DIR}/${LIB_NAME}")
    file(GLOB LIB_FILES "${LIB_PATH}/*.h" "${LIB_PATH}/*.cc")
    set(TARGET_NAME "pycolmap_${LIB_NAME}")
    add_library(${TARGET_NAME} STATIC ${LIB_FILES})
    target_include_directories(${TARGET_NAME}
                             PRIVATE ${PROJECT_SOURCE_DIR}/../src/)
    target_link_libraries(${TARGET_NAME} PRIVATE colmap::colmap
                                                 pybind11::headers
                                                 pybind11::module)
    list(APPEND PYCOLMAP_LIBRARIES ${TARGET_NAME})
endmacro(PYCOLMAP_ADD_LIBRARY)

PYCOLMAP_ADD_LIBRARY(estimators)
PYCOLMAP_ADD_LIBRARY(feature)
PYCOLMAP_ADD_LIBRARY(geometry)
PYCOLMAP_ADD_LIBRARY(optim)
PYCOLMAP_ADD_LIBRARY(pipeline)
PYCOLMAP_ADD_LIBRARY(scene)
PYCOLMAP_ADD_LIBRARY(sfm)

pybind11_add_module(pycolmap ${PYCOLMAP_DIR}/main.cc)
target_include_directories(pycolmap PRIVATE ${PROJECT_SOURCE_DIR}/../src/)
target_link_libraries(pycolmap PRIVATE ${PYCOLMAP_LIBRARIES}
                                       colmap::colmap
                                       glog::glog
                                       Ceres::ceres)
target_compile_definitions(pycolmap PRIVATE VERSION_INFO="${PROJECT_VERSION}")
install(TARGETS pycolmap LIBRARY DESTINATION .)
