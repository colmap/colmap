// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yonatan Simson (see ../LICENSE)

// COLMAP-added shim (not upstream cusift) -- see ../README.md.
//
// Lets COLMAP-side code initialise and release cusift's memory pool without
// including <cuda_runtime.h> or cudautils.h. Device selection is COLMAP's
// (colmap::SetBestCudaDevice); these functions act on the calling thread's
// current device.

#ifndef CUSIFT_DEVICE_H
#define CUSIFT_DEVICE_H

#include <cstddef>

namespace cusift {

// Creates cusift's own stream-ordered memory pool on the calling thread's
// current device, whose unbounded release threshold turns cusift's per-image
// allocations into pool hits. The device's default pool is left untouched.
// Idempotent per device.
void InitMemoryPool();

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
