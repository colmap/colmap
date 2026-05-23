// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//

#include "colmap/util/cuda.h"

#include "colmap/util/cudacc.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <iostream>

#if defined(__HIPCC__) || defined(COLMAP_HIP_ENABLED)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace colmap {
namespace {

#if defined(__HIPCC__) || defined(COLMAP_HIP_ENABLED)
using DeviceProp = hipDeviceProp_t;
#else
using DeviceProp = cudaDeviceProp;
#endif

bool CompareCudaDevice(const DeviceProp& d1, const DeviceProp& d2) {
  bool result = (d1.major > d2.major) ||
                ((d1.major == d2.major) && (d1.minor > d2.minor)) ||
                ((d1.major == d2.major) && (d1.minor == d2.minor) &&
                 (d1.multiProcessorCount > d2.multiProcessorCount));
  return result;
}

}  // namespace

int GetNumCudaDevices() {
  int num_cuda_devices;
  CUDA_SAFE_CALL(hipGetDeviceCount(&num_cuda_devices));
  return num_cuda_devices;
}

int FindBestCudaDevice() {
  const int num_devices = GetNumCudaDevices();
  THROW_CHECK_GT(num_devices, 0) << "No CUDA devices available";
  std::vector<DeviceProp> all_devices(num_devices);
  std::vector<int> indices(num_devices);
  for (int id = 0; id < num_devices; ++id) {
    indices[id] = id;
    hipGetDeviceProperties(&all_devices[id], id);
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
  CUDA_SAFE_CALL(hipSetDevice(selected));
}

}  // namespace colmap
