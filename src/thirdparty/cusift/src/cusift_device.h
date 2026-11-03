// COLMAP-added shim (not upstream cusift) -- see ../README.md.
//
// Exists so that COLMAP-side code can bind a thread to its GPU and initialize
// cusift's memory pool WITHOUT including <cuda_runtime.h> or cudautils.h. That
// keeps every CUDA type and every CUDA API reference inside this library, so
// colmap_feature does not need to link the CUDA runtime on its own account.

#ifndef CUSIFT_DEVICE_H
#define CUSIFT_DEVICE_H

namespace cusift {

// Binds the calling thread to `gpu_index` (cudaSetDevice is per-thread) and
// raises the stream-ordered memory pool release threshold, which is what turns
// cusift's ~64 allocations per image into pool hits. Idempotent per device.
//
// A negative gpu_index selects the best available device using the same
// preference order as colmap::SetBestCudaDevice (highest compute capability,
// then most multiprocessors), so behavior matches the rest of COLMAP. Returns
// false if no CUDA device is available or the index is out of range.
bool InitDevice(int gpu_index);

}  // namespace cusift

#endif  // CUSIFT_DEVICE_H
