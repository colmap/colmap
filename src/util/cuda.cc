// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "util/cuda.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "util/cudacc.h"
#include "util/logging.h"

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
  cudaGetDeviceCount(&num_cuda_devices);
  return num_cuda_devices;
}

void SetBestCudaDevice(const int gpu_index) {
  const int num_cuda_devices = GetNumCudaDevices();
  CHECK_GT(num_cuda_devices, 0) << "No CUDA devices available";

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
  }

  CHECK_GE(selected_gpu_index, 0);
  CHECK_LT(selected_gpu_index, num_cuda_devices) << "Invalid CUDA GPU selected";

  cudaDeviceProp device;
  cudaGetDeviceProperties(&device, selected_gpu_index);
  CUDA_SAFE_CALL(cudaSetDevice(selected_gpu_index));
}

}  // namespace colmap
