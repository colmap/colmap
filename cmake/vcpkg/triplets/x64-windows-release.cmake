set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_BUILD_TYPE release)

# The current METIS DLL crashes in METIS_PartGraphKway on Windows. Keep the
# remaining dependencies dynamic, but link METIS statically until the upstream
# DLL issue is resolved.
if(PORT STREQUAL "metis")
    set(VCPKG_LIBRARY_LINKAGE static)
else()
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()
