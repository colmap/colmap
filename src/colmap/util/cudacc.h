// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//

#pragma once

#include <string>

#if defined(__HIPCC__) || defined(COLMAP_HIP_ENABLED)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define HIP_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK() CudaCheck(__FILE__, __LINE__)
#define CUDA_SYNC_AND_CHECK() CudaSyncAndCheck(__FILE__, __LINE__)
#define HIP_SYNC_AND_CHECK() CudaSyncAndCheck(__FILE__, __LINE__)

namespace colmap {

class CudaTimer {
 public:
  CudaTimer();
  ~CudaTimer();

  void Print(const std::string& message);

 private:
#if defined(__HIPCC__) || defined(COLMAP_HIP_ENABLED)
  hipEvent_t start_;
  hipEvent_t stop_;
#else
  cudaEvent_t start_;
  cudaEvent_t stop_;
#endif
  float elapsed_time_;
};

#if defined(__HIPCC__) || defined(COLMAP_HIP_ENABLED)
void CudaSafeCall(const hipError_t error,
                  const std::string& file,
                  const int line);
#else
void CudaSafeCall(const cudaError_t error,
                  const std::string& file,
                  const int line);
#endif

void CudaCheck(const char* file, const int line);
void CudaSyncAndCheck(const char* file, const int line);

}  // namespace colmap
