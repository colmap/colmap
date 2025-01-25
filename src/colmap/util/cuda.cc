// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/util/cuda.h"

#include "colmap/util/cudacc.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

namespace colmap {
namespace {

// Check whether the first Cuda device is better than the second.
bool CompareCudaDevice(const cudaDeviceProp& d1, const cudaDeviceProp& d2) {
  bool result = (d1.major > d2.major) ||
                ((d1.major == d2.major) && (d1.minor > d2.minor)) ||
                ((d1.major == d2.major) && (d1.minor == d2.minor) &&
                 (d1.multiProcessorCount > d2.multiProcessorCount));
  return result;
}

}  // namespace

int GetNumCudaDevices() {
  int num_cuda_devices;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&num_cuda_devices));
  return num_cuda_devices;
}

void SetBestCudaDevice(const int gpu_index) {
  const int num_cuda_devices = GetNumCudaDevices();
  THROW_CHECK_GT(num_cuda_devices, 0) << "No CUDA devices available";

  int selected_gpu_index = -1;
  if (gpu_index >= 0) {
    selected_gpu_index = gpu_index;
  } else {
    std::vector<cudaDeviceProp> all_devices(num_cuda_devices);
    for (int device_id = 0; device_id < num_cuda_devices; ++device_id) {
      cudaGetDeviceProperties(&all_devices[device_id], device_id);
    }
    std::sort(all_devices.begin(), all_devices.end(), CompareCudaDevice);
    CUDA_SAFE_CALL(cudaChooseDevice(&selected_gpu_index, all_devices.data()));
    VLOG(2) << "Found " << num_cuda_devices << " CUDA device(s), "
            << "selected device " << selected_gpu_index << " with name "
            << all_devices[selected_gpu_index].name;
  }

  THROW_CHECK_GE(selected_gpu_index, 0);
  THROW_CHECK_LT(selected_gpu_index, num_cuda_devices)
      << "Invalid CUDA GPU selected";

  cudaDeviceProp device;
  cudaGetDeviceProperties(&device, selected_gpu_index);
  CUDA_SAFE_CALL(cudaSetDevice(selected_gpu_index));
}

}  // namespace colmap
