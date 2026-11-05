// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>

namespace colmap {

int GetNumCudaDevices();

int FindBestCudaDevice();

void SetBestCudaDevice(int gpu_index);

// Returns whether the CUDA device has compute capability `major`.0 or newer,
// e.g., 8 for native bf16 arithmetic (Ampere) or 7 for fast fp16 arithmetic
// (Volta, Turing). Returns false if the device cannot be queried, e.g., for an
// invalid index or a missing driver. Always returns false for HIP builds.
bool CudaDeviceHasComputeCapability(int gpu_index, int major);

// Queries free and total memory in bytes on `gpu_index`, preserving the
// caller's active CUDA device. Returns false (clearing any non-sticky CUDA
// error) if the device cannot be queried.
bool GetAvailableCudaMemoryBytes(int gpu_index,
                                 size_t* free_bytes,
                                 size_t* total_bytes);

}  // namespace colmap
