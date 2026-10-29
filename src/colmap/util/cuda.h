// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace colmap {

int GetNumCudaDevices();

int FindBestCudaDevice();

void SetBestCudaDevice(int gpu_index);

// Returns whether the CUDA device natively supports bf16 arithmetic, i.e., has
// compute capability 8.0 (Ampere) or newer. Returns false if the device cannot
// be queried, e.g., for an invalid index or a missing driver.
bool CudaDeviceSupportsBf16(int gpu_index);

}  // namespace colmap
