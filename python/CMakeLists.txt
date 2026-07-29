cmake_minimum_required(VERSION 3.10)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})
option(GENERATE_STUBS "Whether to generate stubs" ON)
option(CCACHE_ENABLED "Whether to enable compiler caching, if available" ON)
option(WERROR_ENABLED "Whether to treat compiler warnings as errors" OFF)

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
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/W3>)
    if(WERROR_ENABLED)
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/WX>)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wall>)
    if(WERROR_ENABLED)
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Werror>)
    endif()
endif()

find_package(colmap REQUIRED)

# On Windows with MSVC, the linker receives bare "gflags.lib" (no path) because
# CMake's multi-config generators fail to propagate IMPORTED_IMPLIB_<CONFIG>
# from the gflags imported target to the linker. Both glog::glog and Ceres::ceres
# reference gflags in their INTERFACE_LINK_LIBRARIES, so replace those references
# with the absolute path to gflags.lib to bypass CMake target resolution.
if(MSVC)
    if(TARGET gflags::gflags_shared)
        set(_gflags_target gflags::gflags_shared)
    elseif(TARGET gflags_shared)
        set(_gflags_target gflags_shared)
    endif()
    if(_gflags_target)
        get_target_property(_gflags_implib ${_gflags_target} IMPORTED_IMPLIB_RELEASE)
        if(_gflags_implib)
            if(TARGET glog::glog)
                get_target_property(_glog_iface glog::glog INTERFACE_LINK_LIBRARIES)
                string(REPLACE "gflags::gflags" "${_gflags_implib}" _glog_iface_fixed "${_glog_iface}")
                set_target_properties(glog::glog PROPERTIES
                    INTERFACE_LINK_LIBRARIES "${_glog_iface_fixed}")
            endif()
            if(TARGET Ceres::ceres)
                get_target_property(_ceres_iface Ceres::ceres INTERFACE_LINK_LIBRARIES)
                string(REPLACE ";gflags;" ";${_gflags_implib};" _ceres_iface_fixed "${_ceres_iface}")
                set_target_properties(Ceres::ceres PROPERTIES
                    INTERFACE_LINK_LIBRARIES "${_ceres_iface_fixed}")
            endif()
        endif()
    endif()
    unset(_gflags_target)
    unset(_gflags_implib)
endif()

if(CCACHE_ENABLED)
    find_program(CCACHE ccache)
    if(CCACHE)
        message(STATUS "Enabling ccache support")
        set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
        if(CUDA_ENABLED)
            set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE})
        endif()
    else()
        message(STATUS "Disabling ccache support")
    endif()
else()
    message(STATUS "Disabling ccache support")
endif()

# If colmap was built with ONNX support, we need to set the rpath so the
# Python module can find the ONNX Runtime library at runtime.
set(ONNX_RPATH "")
if(DEFINED onnxruntime_LIBRARY_DIR_HINTS AND EXISTS "${onnxruntime_LIBRARY_DIR_HINTS}")
    message(STATUS "ONNX Runtime library directory: ${onnxruntime_LIBRARY_DIR_HINTS}")
    set(ONNX_RPATH "${onnxruntime_LIBRARY_DIR_HINTS}")
elseif(TARGET onnxruntime::onnxruntime)
    # Derive the library directory from the imported target.
    get_target_property(_ort_type onnxruntime::onnxruntime TYPE)
    if(_ort_type STREQUAL "INTERFACE_LIBRARY")
        # Find module creates an INTERFACE target with link libraries.
        get_target_property(_ort_libs onnxruntime::onnxruntime INTERFACE_LINK_LIBRARIES)
        list(GET _ort_libs 0 _ort_location)
    else()
        get_target_property(_ort_location onnxruntime::onnxruntime IMPORTED_LOCATION)
        if(NOT _ort_location)
            get_target_property(_ort_location onnxruntime::onnxruntime IMPORTED_LOCATION_RELEASE)
        endif()
    endif()
    if(_ort_location)
        get_filename_component(ONNX_RPATH "${_ort_location}" DIRECTORY)
        message(STATUS "ONNX Runtime library directory (from target): ${ONNX_RPATH}")
    endif()
endif()

if (CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()
find_package(Python REQUIRED COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

find_package(pybind11 3.0.2 REQUIRED)

file(GLOB_RECURSE SOURCE_FILES "${PROJECT_SOURCE_DIR}/../src/pycolmap/*.cc")
if(NOT MVS_ENABLED)
    list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/mvs/.*")
    list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/pipeline/mvs\\.cc$")
    list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/pipeline/meshing\\.cc$")
endif()
pybind11_add_module(_core ${SOURCE_FILES})
target_include_directories(_core PRIVATE ${PROJECT_SOURCE_DIR}/../src/)
target_link_libraries(_core PRIVATE colmap::colmap glog::glog Ceres::ceres)
target_compile_definitions(_core PRIVATE VERSION_INFO="${PROJECT_VERSION}")

# Set rpath to find ONNX Runtime library at runtime.
if(ONNX_RPATH)
    if(APPLE)
        set_target_properties(_core PROPERTIES
            BUILD_RPATH "${ONNX_RPATH}"
            INSTALL_RPATH "${ONNX_RPATH}"
        )
    elseif(UNIX)
        set_target_properties(_core PROPERTIES
            BUILD_RPATH "${ONNX_RPATH}"
            INSTALL_RPATH "$ORIGIN/../lib:${ONNX_RPATH}"
        )
    endif()
endif()

install(TARGETS _core LIBRARY DESTINATION pycolmap)

if(GENERATE_STUBS AND UNIX)
    message(STATUS "Enabling stubs generation")
    set(STUBGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/_core")
    # Set library path for stub generation to find ONNX Runtime.
    if(APPLE AND ONNX_RPATH)
        set(STUBGEN_LIB_PATH_VAR "DYLD_LIBRARY_PATH=${ONNX_RPATH}:$ENV{DYLD_LIBRARY_PATH}")
    elseif(ONNX_RPATH)
        set(STUBGEN_LIB_PATH_VAR "LD_LIBRARY_PATH=${ONNX_RPATH}:$ENV{LD_LIBRARY_PATH}")
    else()
        set(STUBGEN_LIB_PATH_VAR "")
    endif()
    add_custom_command(
        TARGET _core POST_BUILD
        COMMAND
          "${CMAKE_COMMAND}" -E env
          "PYTHONPATH=$<TARGET_FILE_DIR:_core>:$ENV{PYTHONPATH}"
          ${STUBGEN_LIB_PATH_VAR}
          bash ${PROJECT_SOURCE_DIR}/generate_stubs.sh "${Python_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating pybind11 stubs"
        VERBATIM
    )
    install(DIRECTORY ${STUBGEN_OUTPUT_DIR} DESTINATION pycolmap)
endif()
