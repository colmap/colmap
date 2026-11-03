// COLMAP-added shim (not upstream cusift) -- see ../README.md.

#include "cusift_device.h"

#include <algorithm>
#include <vector>

#include <cuda_runtime.h>

#include "cudautils.h"

namespace cusift {
namespace {

// Mirrors CompareCudaDevice in colmap/util/cuda.cc: prefer higher compute
// capability, then more multiprocessors. Kept in sync deliberately so that
// gpu_index = -1 selects the same device the rest of COLMAP would select.
bool IsBetterDevice(const cudaDeviceProp& d1, const cudaDeviceProp& d2) {
  return (d1.major > d2.major) ||
         ((d1.major == d2.major) && (d1.minor > d2.minor)) ||
         ((d1.major == d2.major) && (d1.minor == d2.minor) &&
          (d1.multiProcessorCount > d2.multiProcessorCount));
}

int FindBestDevice(int num_devices) {
  std::vector<cudaDeviceProp> props(num_devices);
  std::vector<int> indices(num_devices);
  for (int id = 0; id < num_devices; ++id) {
    indices[id] = id;
    if (cudaGetDeviceProperties(&props[id], id) != cudaSuccess) {
      return 0;
    }
  }
  std::sort(indices.begin(), indices.end(), [&props](int a, int b) {
    return IsBetterDevice(props[a], props[b]);
  });
  return indices[0];
}

}  // namespace

bool InitDevice(int gpu_index) {
  int num_devices = 0;
  if (cudaGetDeviceCount(&num_devices) != cudaSuccess || num_devices == 0) {
    return false;
  }
  const int selected =
      (gpu_index >= 0) ? gpu_index : FindBestDevice(num_devices);
  if (selected >= num_devices) {
    return false;
  }
  if (cudaSetDevice(selected) != cudaSuccess) {
    return false;
  }
  cusiftInitMemPool();
  return true;
}

}  // namespace cusift
