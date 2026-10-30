// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace colmap {

int GetNumCudaDevices();

int FindBestCudaDevice();

void SetBestCudaDevice(int gpu_index);

// Returns whether the CUDA device has compute capability `major`.0 or newer,
// e.g., 8 for native bf16 arithmetic (Ampere) or 7 for fast fp16 arithmetic
// (Volta, Turing). Returns false if the device cannot be queried, e.g., for an
// invalid index or a missing driver. Always returns false for HIP builds.
bool CudaDeviceHasComputeCapability(int gpu_index, int major);

}  // namespace colmap
