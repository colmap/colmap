// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "mvs/cuda_utils.h"

#include <algorithm>
#include <iostream>
#include <vector>

namespace colmap {
namespace mvs {
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

CudaTimer::CudaTimer() {
  CUDA_SAFE_CALL(cudaEventCreate(&start_));
  CUDA_SAFE_CALL(cudaEventCreate(&stop_));
  CUDA_SAFE_CALL(cudaEventRecord(start_, 0));
}

CudaTimer::~CudaTimer() {
  CUDA_SAFE_CALL(cudaEventDestroy(start_));
  CUDA_SAFE_CALL(cudaEventDestroy(stop_));
}

void CudaTimer::Print(const std::string& message) {
  CUDA_SAFE_CALL(cudaEventRecord(stop_, 0));
  CUDA_SAFE_CALL(cudaEventSynchronize(stop_));
  CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time_, start_, stop_));
  printf("%s: %.4fs\n", message.c_str(), elapsed_time_ / 1000.0f);
}

void CudaSafeCall(const cudaError_t error, const std::string file,
                  const int line) {
  if (error != cudaSuccess) {
    printf("%s in %s at line %i\n", cudaGetErrorString(error), file.c_str(),
           line);
    exit(EXIT_FAILURE);
  }
}

void CudaCheckError(const char* file, const int line) {
  cudaError error = cudaGetLastError();
  if (cudaSuccess != error) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
            cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  error = cudaDeviceSynchronize();
  if (cudaSuccess != error) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file,
            line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

void SetBestCudaDevice(const int gpu_id) {
  int selected_gpu_index = -1;
  if (gpu_id >= 0) {
    selected_gpu_index = gpu_id;
  } else {
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1) {
      std::vector<cudaDeviceProp> all_devices(num_devices);
      for (int device_id = 0; device_id < num_devices; ++device_id) {
        cudaGetDeviceProperties(&all_devices[device_id], device_id);
      }
      std::sort(all_devices.begin(), all_devices.end(), CompareCudaDevice);
      CUDA_SAFE_CALL(cudaChooseDevice(&selected_gpu_index, all_devices.data()));
    } else if (num_devices == 0) {
      std::cerr << "Error: No CUDA device is detected in the machine"
                << std::endl;
      exit(EXIT_FAILURE);
    } else {
      selected_gpu_index = 0;
    }
  }

  cudaDeviceProp device;
  cudaGetDeviceProperties(&device, selected_gpu_index);
  CUDA_SAFE_CALL(cudaSetDevice(selected_gpu_index));
}

void CheckGlobalMemSize() {
  size_t free_memory, total_memory, used_memory;
  cudaMemGetInfo(&free_memory, &total_memory);
  used_memory = total_memory - free_memory;
  std::cout << "The current free global memory is: "
            << free_memory / 1024.0 / 1024.0
            << "The current used global memory is: "
            << used_memory / 1024.0 / 1024.0 << std::endl;
}

void CheckSharedMem(const size_t shared_memory_used) {
  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  std::cout << "The graphics cards has shared memory: "
            << device_prop.sharedMemPerBlock << " bytes" << std::endl;
  if (device_prop.sharedMemPerBlock < shared_memory_used) {
    std::cerr << "Error: There is not enough shared memory. User fewer images."
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

}  // namespace mvs
}  // namespace colmap
