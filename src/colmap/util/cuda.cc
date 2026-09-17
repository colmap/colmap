// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cuda.h"

#include "colmap/util/cudacc.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <iostream>

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
  int num_cuda_devices = 0;
  const cudaError_t error = cudaGetDeviceCount(&num_cuda_devices);
#ifdef COLMAP_CUDA_ENABLED
  if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver) {
    return 0;
  }
#endif
  CUDA_SAFE_CALL(error);
  return num_cuda_devices;
}

int FindBestCudaDevice() {
  const int num_devices = GetNumCudaDevices();
  THROW_CHECK_GT(num_devices, 0) << "No CUDA devices available";
  std::vector<cudaDeviceProp> all_devices(num_devices);
  std::vector<int> indices(num_devices);
  for (int id = 0; id < num_devices; ++id) {
    indices[id] = id;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&all_devices[id], id));
  }
  std::sort(indices.begin(), indices.end(), [&](int a, int b) {
    return CompareCudaDevice(all_devices[a], all_devices[b]);
  });
  const int selected = indices.front();
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
