# vcpkg overlay ports

These ports override the versions from the registry baseline for vcpkg
manifest builds. They are registered in the repository's
`vcpkg-configuration.json`, so they apply to both CI and local builds.

## METIS and GKlib

The newer METIS/GKlib pair in the pinned vcpkg baseline crashes in
`METIS_PartGraphKway` on Windows. The older versions used here are known to
work on Windows, but selecting them with `overrides` in `vcpkg.json` is not
sufficient: the old METIS port enables `-march=native`, which can produce
AVX-512 instructions when a package is built on one machine and then cause an
illegal-instruction failure when the binary cache restores it on another.

The overlay keeps the known-good METIS/GKlib source revisions and patches
their build scripts to use the compiler flags supplied by the vcpkg triplet.
This makes the packages portable while preserving the Windows behavior.

vcpkg overlay ports replace a complete registry port; they cannot add a patch
to an existing port. Consequently, the port manifests, portfiles, and upstream
vcpkg patches are copied here as a unit. The `overrides` entries would be
redundant because overlay ports take precedence during version resolution.

Remove these overlays after a newer METIS/GKlib pair passes the Windows graph
cut and scene clustering tests and produces portable binaries on Linux.

## OpenBLAS

COLMAP and FAISS call BLAS from multiple threads, including when OpenBLAS's
own thread pool is disabled. The registry port does not enable `USE_LOCKING`
for this configuration, so concurrent calls can share scratch buffers and
return corrupted results or crash.

The OpenBLAS overlay is copied from the pinned vcpkg baseline
`c748cb44f2a435fcf015c35225c9d5545fe0021c` (OpenBLAS 0.3.33), with
`-DUSE_LOCKING=ON` added to the configure options. This keeps single-threaded
kernels while protecting their shared allocator; it does not enable a nested
BLAS thread pool. The existing `threads` and `dynamic-arch` features retain
their behavior.

Remove this overlay once the registry baseline provides a thread-safe
single-threaded OpenBLAS build.
