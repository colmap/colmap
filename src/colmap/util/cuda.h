// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace colmap {

int GetNumCudaDevices();

int FindBestCudaDevice();

void SetBestCudaDevice(int gpu_index);

}  // namespace colmap
