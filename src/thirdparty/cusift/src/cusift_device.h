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
#include <stdexcept>

namespace cusift {

// The calling thread's current device, for a caller that must come back to it
// (colmap::SetBestCudaDevice) from another thread.
int CurrentDevice();

// Creates cusift's own stream-ordered memory pool on the calling thread's
// current device, whose unbounded release threshold turns cusift's per-image
// allocations into pool hits. The device's default pool is left untouched.
// Idempotent per device.
void InitMemoryPool();

// Returns the memory cusift's pool on the current device holds but no longer
// uses to the driver, after the work already queued on the device has
// finished. Memory still in use, by this or another thread, is not touched.
// Does nothing on a device that fell back to its default pool, which other
// code in the process shares.
void ReleaseUnusedMemory();

// As ReleaseUnusedMemory, on `device` instead of the current device, which it
// restores. Never throws, so destructors may call it.
void ReleaseUnusedMemoryOnDevice(int device);

// Bytes of cusift's pool on the current device that are allocated (in use)
// and reserved from the driver (in use or held for reuse). Diagnostics.
// Zeros on a device that fell back to its default pool.
struct MemoryPoolUsage {
  std::size_t used = 0;
  std::size_t reserved = 0;
};
MemoryPoolUsage GetMemoryPoolUsage();

// Thrown when a CUDA runtime call reports cudaErrorMemoryAllocation (device or
// pinned host memory). Unlike other CUDA errors, it depends on the image, so a
// caller may skip the image and go on.
class OutOfMemoryError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Test only: the `n`-th next allocation from cusift's pool, on any thread,
// fails as if the device were out of memory. 0 disarms it.
void FailAllocationForTesting(int n);

// Test only: devices whose pool is created after this call fall back to their
// default pool, as on a device that cannot create one. Call it before any
// cusift work in the process.
void FailPoolCreationForTesting();

}  // namespace cusift

#endif  // CUSIFT_DEVICE_H
