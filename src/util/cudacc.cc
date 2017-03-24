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

#include "util/cudacc.h"

#include "util/logging.h"

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

}  // namespace colmap
