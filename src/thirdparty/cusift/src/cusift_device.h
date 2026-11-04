// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yonatan Simson (see ../LICENSE)

// COLMAP-added shim (not upstream cusift) -- see ../README.md.
//
// Exists so that COLMAP-side code can bind a thread to its GPU and initialize
// cusift's memory pool WITHOUT including <cuda_runtime.h> or cudautils.h. That
// keeps every CUDA type and every CUDA API reference inside this library, so
// colmap_feature does not need to link the CUDA runtime on its own account.

#ifndef CUSIFT_DEVICE_H
#define CUSIFT_DEVICE_H

#include <cstddef>

namespace cusift {

// Binds the calling thread to `gpu_index` (cudaSetDevice is per-thread) and
// creates cusift's own stream-ordered memory pool on that device, whose
// unbounded release threshold turns cusift's ~64 allocations per image into
// pool hits. The device's default pool is left untouched. Idempotent per
// device.
//
// A negative gpu_index selects the best available device using the same
// preference order as colmap::SetBestCudaDevice (highest compute capability,
// then most multiprocessors), so behavior matches the rest of COLMAP. Returns
// false if no CUDA device is available or the index is out of range.
bool InitDevice(int gpu_index);

// Returns the memory cusift's pool on the current device holds but no longer
// uses to the driver, after the work already queued on the device has
// finished. Memory still in use, by this or another thread, is not touched.
void ReleaseUnusedMemory();

// Bytes of cusift's pool on the current device that are allocated (in use)
// and reserved from the driver (in use or held for reuse). Diagnostics.
struct MemoryPoolUsage {
  std::size_t used = 0;
  std::size_t reserved = 0;
};
MemoryPoolUsage GetMemoryPoolUsage();

}  // namespace cusift

#endif  // CUSIFT_DEVICE_H
