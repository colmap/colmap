// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cudacc.h"

#include "colmap/util/logging.h"
#include "colmap/util/string.h"

namespace colmap {

std::string GetCudaErrorMessage(const cudaError_t error) {
  const char* description = cudaGetErrorString(error);
  std::string message =
      StringPrintf("%s (error %d)",
                   description != nullptr && description[0] != '\0'
                       ? description
                       : "Unknown GPU runtime error",
                   static_cast<int>(error));
  if (error == cudaErrorInsufficientDriver) {
#if defined(COLMAP_HIP_ENABLED)
    message +=
        ". The GPU driver is missing or insufficient for this HIP "
        "runtime. Install a compatible AMD driver or use a compatible "
        "COLMAP build.";
#else
    message +=
        ". The NVIDIA driver is missing or insufficient for this CUDA "
        "runtime. Update the NVIDIA driver or use a COLMAP build "
        "compatible with the installed driver.";
#endif
  }
  return message;
}

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
                                     GetCudaErrorMessage(error).c_str());
  }
}

void CudaCheck(const char* file, const int line) {
  const cudaError_t error = cudaGetLastError();
  while (error != cudaSuccess) {
    LOG(FATAL_THROW) << StringPrintf("CUDA error at %s:%i - %s",
                                     file,
                                     line,
                                     GetCudaErrorMessage(error).c_str());
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
                        GetCudaErrorMessage(error).c_str())
        << "\nThis error is likely caused by the graphics card timeout "
           "detection mechanism of your operating system. Please refer "
           "to the FAQ in the documentation on how to solve this "
           "problem.";
  }
}

}  // namespace colmap
