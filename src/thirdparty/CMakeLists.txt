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

# boost::unordered_node_map requires Boost >= 1.84, which is newer than the apt
# Boost on Ubuntu 24.04 and earlier. The containers are part of COLMAP's ABI
# (see cmake/FindDependencies.cmake), so rather than falling back to a second
# container implementation, vendor a pinned copy of the header-only modules that
# boost-unordered transitively includes. The list is the include closure of
# boost/unordered/unordered_{flat,node}_{map,set}.hpp, ~5 MB in total.
if(NOT COLMAP_BOOST_UNORDERED_FROM_SYSTEM AND FETCH_BOOST_UNORDERED)
    include(FetchContent)
    set(_boost_unordered_tag "boost-1.90.0")
    set(_boost_unordered_modules
        unordered       7dd15a1196e36de483fc287a5487210b596979761c4708b4c3067a717c765a05
        assert          1f48f929e69146db222fdf9538ec81c7cbd3ca88b94343644942b477031a3bba
        config          1390bd79fbf270e40cfe5ac6b899f4a9c4e47139b6e17b4c1f7f78822adaa9fc
        container_hash  415d6a01b2d8a07ab36dbf9e31a8fec6fa0e51185d1015a087c153de28cfaba5
        core            0444b7741bcf20f80ff6a4d191a920f6aba92d26734c81fa481f8fd0a274b487
        describe        4ca9c54d50c5bd8ec68811475ef96ec1ac0abbfbf19f55789823aabcbb704812
        mp11            ca52bde753586dcba29739cfd55abb473bf80c934c6d49a2b58023ed4e48adf6
        predef          d9144eb3c7d965991e8798fc7e96d5ea862951bffcbdba0b8a23892fd5c99f7c
        static_assert   3b7980c5968d585b04f980bbb4e81bbbf0b6d1a0b7c965c2a3426b0acb4a5f49
        throw_exception 30e2834ba64587a363750e35c627ee045f89be05d0cc9eab7251414eba88e110
    )
    message(STATUS "Configuring boost-unordered (${_boost_unordered_tag})...")
    add_library(colmap_boost_unordered INTERFACE)
    while(_boost_unordered_modules)
        list(POP_FRONT _boost_unordered_modules _module _hash)
        # SOURCE_SUBDIR names a directory without a CMakeLists.txt so that the
        # module is only downloaded, never add_subdirectory()'d. The per-module
        # Boost CMake files expect the Boost superproject and would collide with
        # the Boost:: targets from find_package(Boost).
        FetchContent_Declare(boost_${_module}
            URL https://github.com/boostorg/${_module}/archive/refs/tags/${_boost_unordered_tag}.tar.gz
            URL_HASH SHA256=${_hash}
            SOURCE_SUBDIR do-not-configure
            ${_fetch_content_declare_args}
        )
        FetchContent_MakeAvailable(boost_${_module})
        target_include_directories(colmap_boost_unordered SYSTEM INTERFACE
            $<BUILD_INTERFACE:${boost_${_module}_SOURCE_DIR}/include>)
        install(DIRECTORY "${boost_${_module}_SOURCE_DIR}/include/boost"
                DESTINATION "${COLMAP_BOOST_UNORDERED_INSTALL_DIR}")
    endwhile()
    target_include_directories(colmap_boost_unordered SYSTEM INTERFACE
        $<INSTALL_INTERFACE:${COLMAP_BOOST_UNORDERED_INSTALL_DIR}>)
    message(STATUS "Configuring boost-unordered (${_boost_unordered_tag})... done")
endif()

if(FETCH_POSELIB)
    include(FetchContent)
    FetchContent_Declare(poselib
        URL https://github.com/PoseLib/PoseLib/archive/fa7280fee27f97aff31ae7f98bab7f583fac7d08.zip
        URL_HASH SHA256=5408d4ae8ce367cb2f076bc6c5f0f6f78abd3573d2c015304b04e46f23455f5b
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
