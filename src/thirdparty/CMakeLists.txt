set(FOLDER_NAME "thirdparty")

# Only show moderate warnings for external library code.
if(IS_MSVC)
    string(REGEX REPLACE "/W[0-4]" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REGEX REPLACE "/Wall" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REGEX REPLACE "/W[0-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REGEX REPLACE "/Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
elseif(IS_GNU OR IS_CLANG)
    string(REGEX REPLACE "-Wall" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REGEX REPLACE "-Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(IS_MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
elseif(IS_GNU OR IS_CLANG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
endif()

set(CLANG_TIDY_EXE_TEMP ${CLANG_TIDY_EXE})
set(CLANG_TIDY_EXE "")
if(LSD_ENABLED)
    add_subdirectory(LSD)
endif()
add_subdirectory(PoissonRecon)
if(GPU_ENABLED)
    add_subdirectory(SiftGPU)
endif()
add_subdirectory(VLFeat)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.28")
    set(_fetch_content_declare_args "SYSTEM" "EXCLUDE_FROM_ALL")
endif()

if(FETCH_POSELIB)
    include(FetchContent)
    FetchContent_Declare(poselib
        URL https://github.com/PoseLib/PoseLib/archive/v2.0.5.zip
        URL_HASH SHA256=3d8d71d94813da8e0c3a6df11ec3591614ad2e731e73f3de4982b9297321a38d
        ${_fetch_content_declare_args}
    )
    message(STATUS "Configuring PoseLib...")
    set(MARCH_NATIVE OFF CACHE BOOL "")
    FetchContent_MakeAvailable(poselib)
    # Rename PoseLib's benchmark folder to avoid conflict with Google Benchmark's
    # benchmark/benchmark.h header (PoseLib exposes its entire source dir as include path)
    if(EXISTS "${poselib_SOURCE_DIR}/benchmark")
        file(RENAME "${poselib_SOURCE_DIR}/benchmark" "${poselib_SOURCE_DIR}/poselib_benchmark")
    endif()
    message(STATUS "Configuring PoseLib... done")
endif()

if(FETCH_FAISS)
    include(FetchContent)
    FetchContent_Declare(faiss
        URL https://github.com/facebookresearch/faiss/archive/refs/tags/v1.14.1.zip
        URL_HASH SHA256=4b1ae7e7a0a46385b4084f0e3945623a15fcf99d793bf44d82aae8e24f11e5f5
        ${_fetch_content_declare_args}
    )
    message(STATUS "Configuring faiss...")
    # Use dynamic dispatch (dd) to automatically select the highest SIMD
    # instruction set supported by the target machine (e.g., AVX2, AVX512).
    # Dynamic dispatch is not supported on MSVC (faiss v1.14.1).
    if(NOT IS_MSVC)
        set(FAISS_OPT_LEVEL dd)
    endif()
    set(FAISS_ENABLE_GPU OFF)
    set(FAISS_ENABLE_PYTHON OFF)
    set(FAISS_ENABLE_MKL OFF)
    set(BUILD_TESTING OFF)
    FetchContent_MakeAvailable(faiss)
    # Fix MSVC linker warning LNK4286: faiss headers use FAISS_API which
    # defaults to dllimport on Windows, but we build faiss as a static library.
    # Adding FAISS_MAIN_LIB as PUBLIC ensures consumers also get this defined.
    if(WIN32 AND TARGET faiss)
        target_compile_definitions(faiss PUBLIC FAISS_MAIN_LIB)
    endif()
    message(STATUS "Configuring faiss... done")
endif()

if(CASPAR_ENABLED AND NOT CUDA_ENABLED)
    message(FATAL_ERROR "Caspar requires CUDA")
endif()

if(CASPAR_ENABLED AND CUDA_ENABLED)

    # TODO(tordnat): Make Caspar set CASPAR_MIN_ARCH when generating build files
    set(CASPAR_MIN_ARCH 75)
    foreach(_arch ${CMAKE_CUDA_ARCHITECTURES})
        if(_arch LESS ${CASPAR_MIN_ARCH})
            message(FATAL_ERROR "Caspar requires CUDA architecture >= ${CASPAR_MIN_ARCH}, but ${_arch} was requested. "
                "Set CMAKE_CUDA_ARCHITECTURES to ${CASPAR_MIN_ARCH} or higher, or disable Caspar with -DCASPAR_ENABLED=OFF.")
        endif()
    endforeach()

    message(STATUS "Enabling Caspar support")

    if(CASPAR_USE_DOUBLE)
        set(_caspar_precision "f64")
    else()
        set(_caspar_precision "f32")
    endif()

    set(CASPAR_GEN_DIR
        "${CMAKE_SOURCE_DIR}/src/thirdparty/Symforce-Caspar/generated/${_caspar_precision}")

    if(NOT EXISTS "${CASPAR_GEN_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "Caspar: pre-generated code not found for precision '${_caspar_precision}'.\n"
            "Expected: ${CASPAR_GEN_DIR}/CMakeLists.txt\n"
            "Re-run generate_caspar.py to regenerate.")
    endif()

    include(FetchContent)
    FetchContent_Declare(caspar
        SOURCE_DIR ${CASPAR_GEN_DIR}
        ${_fetch_content_declare_args}
    )
    message(STATUS "Configuring Caspar...")
    FetchContent_MakeAvailable(caspar)
    # Wrap the source-tree include path in BUILD_INTERFACE so caspar_lib_core
    # can be part of install(EXPORT). Caspar's generated CMakeLists.txt uses a
    # raw ${CMAKE_CURRENT_SOURCE_DIR} which CMake rejects on exported targets.
    set_target_properties(caspar_lib_core PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${CASPAR_GEN_DIR}>")
    # Silence warnings from auto-generated Caspar CUDA sources, matching the
    # blanket `-w`/`/W0` policy applied to other thirdparty C++ code above.
    target_compile_options(caspar_lib_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-w>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=550>
        $<$<COMPILE_LANGUAGE:CXX>:$<IF:$<CXX_COMPILER_ID:MSVC>,/W0,-w>>
    )
    if(IS_MSVC)
        set(_MSVC_COMPACT_HEADER "${CASPAR_GEN_DIR}/../../msvc_compact.h")
        target_compile_options(caspar_lib_core PRIVATE
            $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:/FI"${_MSVC_COMPACT_HEADER}">
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /FI"${_MSVC_COMPACT_HEADER}">
        )
    endif()
    message(STATUS "Configuring Caspar... done")

    message(STATUS "Caspar precision:  ${_caspar_precision}")
    message(STATUS "Caspar source dir: ${CASPAR_GEN_DIR}")
endif()

set(CLANG_TIDY_EXE ${CLANG_TIDY_EXE_TEMP})
