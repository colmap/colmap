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
        URL https://github.com/PoseLib/PoseLib/archive/1bb8881f08e68f0b3d9b2ecc411fc15e7766ef1b.zip
        URL_HASH SHA256=ee46322581c7901fe24e91434eeec27163f820d481cba5faa43d37651fabc8e8
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
        URL https://github.com/facebookresearch/faiss/archive/refs/tags/v1.15.1.zip
        URL_HASH SHA256=f46290cc461f5ae88bc52f400e2066c537746328e494f79faa1fa719815e7b66
        ${_fetch_content_declare_args}
    )
    message(STATUS "Configuring faiss...")
    # Use dynamic dispatch (dd) to automatically select the highest SIMD
    # instruction set supported by the target machine (e.g., AVX2, AVX512).
    # Supported on all platforms including MSVC since faiss v1.15
    # (per-file /arch flags in the dd block of faiss/CMakeLists.txt).
    set(FAISS_OPT_LEVEL dd)
    set(FAISS_ENABLE_GPU OFF)
    set(FAISS_ENABLE_PYTHON OFF)
    set(FAISS_ENABLE_MKL OFF)
    set(BUILD_TESTING OFF)
    FetchContent_MakeAvailable(faiss)
    # faiss v1.15 dynamic dispatch unconditionally compiles
    # impl/scalar_quantizer/sq-avx512-spr.cpp with the Sapphire Rapids flag
    # set, including -mavx512fp16, which requires GCC 12+ (Ubuntu 22.04
    # ships GCC 11). On x86 compilers without that flag, drop the SPR
    # dispatch level: the source is fully guarded by
    # COMPILE_SIMD_AVX512_SPR, so dispatch falls back to AVX512_VPOPCNT.
    # MSVC needs no handling (/arch:AVX512 has no per-feature flags).
    if(IS_X86 AND NOT WIN32 AND TARGET faiss AND FAISS_OPT_LEVEL STREQUAL "dd")
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(
            "-mavx512fp16" COLMAP_COMPILER_SUPPORTS_AVX512FP16)
        if(NOT COLMAP_COMPILER_SUPPORTS_AVX512FP16)
            message(
                STATUS
                "Compiler lacks -mavx512fp16, disabling faiss AVX512_SPR dispatch"
            )
            get_target_property(_faiss_defs faiss COMPILE_DEFINITIONS)
            if(_faiss_defs)
                list(REMOVE_ITEM _faiss_defs COMPILE_SIMD_AVX512_SPR)
                set_target_properties(
                    faiss PROPERTIES COMPILE_DEFINITIONS "${_faiss_defs}")
            endif()
            set_source_files_properties(
                "${faiss_SOURCE_DIR}/faiss/impl/scalar_quantizer/sq-avx512-spr.cpp"
                TARGET_DIRECTORY faiss
                PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-mpopcnt")
        endif()
    endif()
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
        set(_MSVC_CUDA_COMPACT_HEADER "${CASPAR_GEN_DIR}/../../msvc_cuda_compact.h")
        target_compile_options(caspar_lib_core PRIVATE
            $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:/FI"${_MSVC_COMPACT_HEADER}">
            # Keep CUDA pre-includes minimal and use NVCC's native pre-include
            # option to avoid forcing C++ standard library headers into NVCC's
            # MSVC host pass.
            $<$<COMPILE_LANGUAGE:CUDA>:--pre-include="${_MSVC_CUDA_COMPACT_HEADER}">
        )
    endif()
    message(STATUS "Configuring Caspar... done")

    message(STATUS "Caspar precision:  ${_caspar_precision}")
    message(STATUS "Caspar source dir: ${CASPAR_GEN_DIR}")
endif()

set(CLANG_TIDY_EXE ${CLANG_TIDY_EXE_TEMP})
