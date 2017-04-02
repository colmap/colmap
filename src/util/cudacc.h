// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#ifndef COLMAP_SRC_UTIL_CUDACC_H_
#define COLMAP_SRC_UTIL_CUDACC_H_

#include <string>

#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() CudaCheckError(__FILE__, __LINE__)

namespace colmap {

class CudaTimer {
 public:
  CudaTimer();
  ~CudaTimer();

  void Print(const std::string& message);

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
  float elapsed_time_;
};

void CudaSafeCall(const cudaError_t error, const std::string& file,
                  const int line);

void CudaCheckError(const char* file, const int line);

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_CUDACC_H_
