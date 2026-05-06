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

int FindBestCudaDevice() {
  const int num_devices = GetNumCudaDevices();
  THROW_CHECK_GT(num_devices, 0) << "No CUDA devices available";
  std::vector<cudaDeviceProp> all_devices(num_devices);
  std::vector<int> indices(num_devices);
  for (int id = 0; id < num_devices; ++id) {
    indices[id] = id;
    cudaGetDeviceProperties(&all_devices[id], id);
  }
  std::sort(indices.begin(), indices.end(), [&](int a, int b) {
    return CompareCudaDevice(all_devices[a], all_devices[b]);
  });
  const int selected = indices[0];
  VLOG(2) << "Found " << num_devices << " CUDA device(s), "
          << "selected device " << selected << " with name "
          << all_devices[selected].name;
  return selected;
}

void SetBestCudaDevice(const int gpu_index) {
  const int num_cuda_devices = GetNumCudaDevices();
  THROW_CHECK_GT(num_cuda_devices, 0) << "No CUDA devices available";
  const int selected = (gpu_index >= 0) ? gpu_index : FindBestCudaDevice();
  THROW_CHECK_LT(selected, num_cuda_devices) << "Invalid CUDA GPU selected";
  CUDA_SAFE_CALL(cudaSetDevice(selected));
}

}  // namespace colmap
