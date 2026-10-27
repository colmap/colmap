// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>

namespace colmap {

int GetNumCudaDevices();

int FindBestCudaDevice();

void SetBestCudaDevice(int gpu_index);

// Returns whether the CUDA device selected by `gpu_index` supports BF16
// matrix kernels, which require Ampere (compute capability 8.0) or newer.
// `gpu_index` uses the single-device form parsed by CSVToVector (e.g. "0");
// a negative index selects device 0, matching the ONNX provider behavior.
// Returns false when `gpu_index` is invalid or the device cannot be queried,
// so callers can safely fall back to FP32 whenever BF16 support is
// unconfirmed.
bool CudaDeviceSupportsBf16(const std::string& gpu_index);

}  // namespace colmap
