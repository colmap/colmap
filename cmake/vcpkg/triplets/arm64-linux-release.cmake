# SPDX-License-Identifier: BSD-3-Clause

set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_BUILD_TYPE release)

# FAISS calls single-threaded BLAS from multiple threads.
if(PORT STREQUAL "openblas")
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS -DUSE_LOCKING=ON)
endif()
