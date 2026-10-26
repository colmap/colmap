// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace colmap {

int GetNumCudaDevices();

int FindBestCudaDevice();

void SetBestCudaDevice(int gpu_index);

// Returns whether the CUDA device natively supports bf16 arithmetic, i.e., has
// compute capability 8.0 (Ampere) or newer. Returns false if the device cannot
// be queried, e.g., for an invalid index or a missing driver. Always returns
// false for HIP builds.
bool CudaDeviceSupportsBf16(int gpu_index);

// Returns whether the CUDA device has tensor cores for fp16 arithmetic, i.e.,
// has compute capability 7.0 (Volta) or newer. Returns false if the device
// cannot be queried, e.g., for an invalid index or a missing driver. Always
// returns false for HIP builds.
bool CudaDeviceHasTensorCores(int gpu_index);

}  // namespace colmap
