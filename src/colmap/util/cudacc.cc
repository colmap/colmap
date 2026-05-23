// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "colmap/util/cudacc.h"

#include "colmap/util/logging.h"

namespace colmap {

CudaTimer::CudaTimer() {
  CUDA_SAFE_CALL(hipEventCreate(&start_));
  CUDA_SAFE_CALL(hipEventCreate(&stop_));
  CUDA_SAFE_CALL(hipEventRecord(start_, 0));
}

CudaTimer::~CudaTimer() {
  CUDA_SAFE_CALL(hipEventDestroy(start_));
  CUDA_SAFE_CALL(hipEventDestroy(stop_));
}

void CudaTimer::Print(const std::string& message) {
  CUDA_SAFE_CALL(hipEventRecord(stop_, 0));
  CUDA_SAFE_CALL(hipEventSynchronize(stop_));
  CUDA_SAFE_CALL(hipEventElapsedTime(&elapsed_time_, start_, stop_));
  LOG(INFO) << StringPrintf(
      "%s: %.4fs", message.c_str(), elapsed_time_ / 1000.0f);
}

#if defined(__HIPCC__) || defined(COLMAP_HIP_ENABLED)
void CudaSafeCall(const hipError_t error,
                  const std::string& file,
                  const int line) {
  if (error != hipSuccess) {
    LOG(FATAL_THROW) << StringPrintf("HIP error at %s:%i - %s",
                                     file.c_str(),
                                     line,
                                     hipGetErrorString(error));
  }
}
#else
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
#endif

void CudaCheck(const char* file, const int line) {
#if defined(__HIPCC__) || defined(COLMAP_HIP_ENABLED)
  const hipError_t error = hipGetLastError();
  while (error != hipSuccess) {
    LOG(FATAL_THROW) << StringPrintf(
        "HIP error at %s:%i - %s", file, line, hipGetErrorString(error));
  }
#else
  const cudaError error = cudaGetLastError();
  while (error != cudaSuccess) {
    LOG(FATAL_THROW) << StringPrintf(
        "CUDA error at %s:%i - %s", file, line, cudaGetErrorString(error));
  }
#endif
}

void CudaSyncAndCheck(const char* file, const int line) {
#if defined(__HIPCC__) || defined(COLMAP_HIP_ENABLED)
  const hipError_t error = hipStreamSynchronize(nullptr);
  if (hipSuccess != error) {
    LOG(FATAL_THROW)
        << StringPrintf("HIP error at %s:%i - %s",
                        file,
                        line,
                        hipGetErrorString(error))
        << "\nThis error is likely caused by the graphics card timeout "
           "detection mechanism of your operating system.";
  }
#else
  const cudaError error = cudaStreamSynchronize(nullptr);
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
#endif
}

}  // namespace colmap
