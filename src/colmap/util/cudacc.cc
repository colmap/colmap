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

#include "colmap/util/cudacc.h"

#include "colmap/util/logging.h"
#include "colmap/util/string.h"

namespace colmap {

CudaTimer::CudaTimer() {
#if defined(__HIPCC__) || defined(COLMAP_HIP_ENABLED)
  CUDA_SAFE_CALL(hipEventCreate(&start_));
  CUDA_SAFE_CALL(hipEventCreate(&stop_));
  CUDA_SAFE_CALL(hipEventRecord(start_, 0));
#else
  CUDA_SAFE_CALL(cudaEventCreate(&start_));
  CUDA_SAFE_CALL(cudaEventCreate(&stop_));
  CUDA_SAFE_CALL(cudaEventRecord(start_, 0));
#endif
}

CudaTimer::~CudaTimer() {
#if defined(__HIPCC__) || defined(COLMAP_HIP_ENABLED)
  CUDA_SAFE_CALL(hipEventDestroy(start_));
  CUDA_SAFE_CALL(hipEventDestroy(stop_));
#else
  CUDA_SAFE_CALL(cudaEventDestroy(start_));
  CUDA_SAFE_CALL(cudaEventDestroy(stop_));
#endif
}

void CudaTimer::Print(const std::string& message) {
#if defined(__HIPCC__) || defined(COLMAP_HIP_ENABLED)
  CUDA_SAFE_CALL(hipEventRecord(stop_, 0));
  CUDA_SAFE_CALL(hipEventSynchronize(stop_));
  CUDA_SAFE_CALL(hipEventElapsedTime(&elapsed_time_, start_, stop_));
#else
  CUDA_SAFE_CALL(cudaEventRecord(stop_, 0));
  CUDA_SAFE_CALL(cudaEventSynchronize(stop_));
  CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time_, start_, stop_));
#endif
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
  // Synchronizes the default stream which is a nullptr.
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
