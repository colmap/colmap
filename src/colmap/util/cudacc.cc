// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cudacc.h"

#include "colmap/util/logging.h"
#include "colmap/util/string.h"

namespace colmap {

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
  LOG(INFO) << StringPrintf(
      "%s: %.4fs", message.c_str(), elapsed_time_ / 1000.0f);
}

void CudaSafeCall(const cudaError_t error,
                  const std::string& file,
                  const int line) {
  if (error != cudaSuccess) {
    LOG(FATAL_THROW) << StringPrintf("CUDA error at %s:%i - %s",
                                     file.c_str(),
                                     line,
                                     cudaGetErrorString(error));
  }
}

void CudaCheck(const char* file, const int line) {
  const cudaError_t error = cudaGetLastError();
  while (error != cudaSuccess) {
    LOG(FATAL_THROW) << StringPrintf(
        "CUDA error at %s:%i - %s", file, line, cudaGetErrorString(error));
  }
}

void CudaSyncAndCheck(const char* file, const int line) {
  // Synchronizes the default stream which is a nullptr.
  const cudaError_t error = cudaStreamSynchronize(nullptr);
  if (cudaSuccess != error) {
    LOG(FATAL_THROW)
        << StringPrintf("CUDA error at %s:%i - %s",
                        file,
                        line,
                        cudaGetErrorString(error))
        << "\nThis error is likely caused by the graphics card timeout "
           "detection mechanism of your operating system. Please refer "
           "to the FAQ in the documentation on how to solve this "
           "problem.";
  }
}

}  // namespace colmap
