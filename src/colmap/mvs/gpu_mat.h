// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#include "colmap/mvs/cuda_flip.h"
#include "colmap/mvs/cuda_rotate.h"
#include "colmap/mvs/cuda_transpose.h"
#include "colmap/mvs/mat.h"
#include "colmap/util/cuda.h"
#include "colmap/util/cudacc.h"
#include "colmap/util/endian.h"

#include <fstream>
#include <iterator>
#include <memory>
#include <string>

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace colmap {
namespace mvs {

template <typename T>
class GpuMat {
 public:
  GpuMat(const size_t width, const size_t height, const size_t depth = 1);
  ~GpuMat();

  __host__ __device__ const T* GetPtr() const;
  __host__ __device__ T* GetPtr();

  __host__ __device__ size_t GetPitch() const;
  __host__ __device__ size_t GetWidth() const;
  __host__ __device__ size_t GetHeight() const;
  __host__ __device__ size_t GetDepth() const;

  __device__ T Get(const size_t row,
                   const size_t col,
                   const size_t slice = 0) const;
  __device__ void GetSlice(const size_t row, const size_t col, T* values) const;

  __device__ T& GetRef(const size_t row, const size_t col);
  __device__ T& GetRef(const size_t row, const size_t col, const size_t slice);

  __device__ void Set(const size_t row, const size_t col, const T value);
  __device__ void Set(const size_t row,
                      const size_t col,
                      const size_t slice,
                      const T value);
  __device__ void SetSlice(const size_t row, const size_t col, const T* values);

  void FillWithScalar(const T value);
  void FillWithVector(const T* values);
  void FillWithRandomNumbers(const T min_value,
                             const T max_value,
                             GpuMat<curandState> random_state);

  void CopyToDevice(const T* data, const size_t pitch);
  void CopyToHost(T* data, const size_t pitch) const;
  Mat<T> CopyToMat() const;

  // Transpose array by swapping x and y coordinates.
  void Transpose(GpuMat<T>* output);

  // Flip array along vertical axis.
  void FlipHorizontal(GpuMat<T>* output);

  // Rotate array in counter-clockwise direction.
  void Rotate(GpuMat<T>* output);

  void Read(const std::string& path);
  void Write(const std::string& path);
  void Write(const std::string& path, const size_t slice);

 protected:
  void ComputeCudaConfig();

  const static size_t kBlockDimX = 32;
  const static size_t kBlockDimY = 16;

  std::shared_ptr<T> array_;
  T* array_ptr_;

  size_t pitch_;
  size_t width_;
  size_t height_;
  size_t depth_;

  dim3 blockSize_;
  dim3 gridSize_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__

namespace internal {

template <typename T>
__global__ void FillWithScalarKernel(GpuMat<T> output, const T value) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < output.GetHeight() && col < output.GetWidth()) {
    for (size_t slice = 0; slice < output.GetDepth(); ++slice) {
      output.Set(row, col, slice, value);
    }
  }
}

template <typename T>
__global__ void FillWithVectorKernel(const T* values, GpuMat<T> output) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < output.GetHeight() && col < output.GetWidth()) {
    for (size_t slice = 0; slice < output.GetDepth(); ++slice) {
      output.Set(row, col, slice, values[slice]);
    }
  }
}

template <typename T>
__global__ void FillWithRandomNumbersKernel(GpuMat<T> output,
                                            GpuMat<curandState> random_state,
                                            const T min_value,
                                            const T max_value) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < output.GetHeight() && col < output.GetWidth()) {
    curandState local_state = random_state.Get(row, col);
    for (size_t slice = 0; slice < output.GetDepth(); ++slice) {
      const T random_value =
          curand_uniform(&local_state) * (max_value - min_value) + min_value;
      output.Set(row, col, slice, random_value);
    }
    random_state.Set(row, col, local_state);
  }
}

}  // namespace internal

template <typename T>
GpuMat<T>::GpuMat(const size_t width, const size_t height, const size_t depth)
    : array_(nullptr),
      array_ptr_(nullptr),
      width_(width),
      height_(height),
      depth_(depth) {
  CUDA_SAFE_CALL(cudaMallocPitch(
      (void**)&array_ptr_, &pitch_, width_ * sizeof(T), height_ * depth_));

  array_ = std::shared_ptr<T>(array_ptr_, cudaFree);

  ComputeCudaConfig();
}

template <typename T>
GpuMat<T>::~GpuMat() {
  array_.reset();
  array_ptr_ = nullptr;
  pitch_ = 0;
  width_ = 0;
  height_ = 0;
  depth_ = 0;
}

template <typename T>
__host__ __device__ const T* GpuMat<T>::GetPtr() const {
  return array_ptr_;
}

template <typename T>
__host__ __device__ T* GpuMat<T>::GetPtr() {
  return array_ptr_;
}

template <typename T>
__host__ __device__ size_t GpuMat<T>::GetPitch() const {
  return pitch_;
}

template <typename T>
__host__ __device__ size_t GpuMat<T>::GetWidth() const {
  return width_;
}

template <typename T>
__host__ __device__ size_t GpuMat<T>::GetHeight() const {
  return height_;
}

template <typename T>
__host__ __device__ size_t GpuMat<T>::GetDepth() const {
  return depth_;
}

template <typename T>
__device__ T GpuMat<T>::Get(const size_t row,
                            const size_t col,
                            const size_t slice) const {
  return *((T*)((char*)array_ptr_ + pitch_ * (slice * height_ + row)) + col);
}

template <typename T>
__device__ void GpuMat<T>::GetSlice(const size_t row,
                                    const size_t col,
                                    T* values) const {
  for (size_t slice = 0; slice < depth_; ++slice) {
    values[slice] = Get(row, col, slice);
  }
}

template <typename T>
__device__ T& GpuMat<T>::GetRef(const size_t row, const size_t col) {
  return GetRef(row, col, 0);
}

template <typename T>
__device__ T& GpuMat<T>::GetRef(const size_t row,
                                const size_t col,
                                const size_t slice) {
  return *((T*)((char*)array_ptr_ + pitch_ * (slice * height_ + row)) + col);
}

template <typename T>
__device__ void GpuMat<T>::Set(const size_t row,
                               const size_t col,
                               const T value) {
  Set(row, col, 0, value);
}

template <typename T>
__device__ void GpuMat<T>::Set(const size_t row,
                               const size_t col,
                               const size_t slice,
                               const T value) {
  *((T*)((char*)array_ptr_ + pitch_ * (slice * height_ + row)) + col) = value;
}

template <typename T>
__device__ void GpuMat<T>::SetSlice(const size_t row,
                                    const size_t col,
                                    const T* values) {
  for (size_t slice = 0; slice < depth_; ++slice) {
    Set(row, col, slice, values[slice]);
  }
}

template <typename T>
void GpuMat<T>::FillWithScalar(const T value) {
  internal::FillWithScalarKernel<T><<<gridSize_, blockSize_>>>(*this, value);
  CUDA_SYNC_AND_CHECK();
}

template <typename T>
void GpuMat<T>::FillWithVector(const T* values) {
  T* values_device;
  CUDA_SAFE_CALL(cudaMalloc((void**)&values_device, depth_ * sizeof(T)));
  CUDA_SAFE_CALL(cudaMemcpy(
      values_device, values, depth_ * sizeof(T), cudaMemcpyHostToDevice));
  internal::FillWithVectorKernel<T>
      <<<gridSize_, blockSize_>>>(values_device, *this);
  CUDA_SYNC_AND_CHECK();
  CUDA_SAFE_CALL(cudaFree(values_device));
}

template <typename T>
void GpuMat<T>::FillWithRandomNumbers(const T min_value,
                                      const T max_value,
                                      const GpuMat<curandState> random_state) {
  internal::FillWithRandomNumbersKernel<T>
      <<<gridSize_, blockSize_>>>(*this, random_state, min_value, max_value);
  CUDA_SYNC_AND_CHECK();
}

template <typename T>
void GpuMat<T>::CopyToDevice(const T* data, const size_t pitch) {
  CUDA_SAFE_CALL(cudaMemcpy2D((void*)array_ptr_,
                              (size_t)pitch_,
                              (void*)data,
                              pitch,
                              width_ * sizeof(T),
                              height_ * depth_,
                              cudaMemcpyHostToDevice));
}

template <typename T>
void GpuMat<T>::CopyToHost(T* data, const size_t pitch) const {
  CUDA_SAFE_CALL(cudaMemcpy2D((void*)data,
                              pitch,
                              (void*)array_ptr_,
                              (size_t)pitch_,
                              width_ * sizeof(T),
                              height_ * depth_,
                              cudaMemcpyDeviceToHost));
}

template <typename T>
Mat<T> GpuMat<T>::CopyToMat() const {
  Mat<T> mat(width_, height_, depth_);
  CopyToHost(mat.GetPtr(), mat.GetWidth() * sizeof(T));
  return mat;
}

template <typename T>
void GpuMat<T>::Transpose(GpuMat<T>* output) {
  for (size_t slice = 0; slice < depth_; ++slice) {
    CudaTranspose(array_ptr_ + slice * pitch_ / sizeof(T) * GetHeight(),
                  output->GetPtr() +
                      slice * output->pitch_ / sizeof(T) * output->GetHeight(),
                  width_,
                  height_,
                  pitch_,
                  output->pitch_);
  }
  CUDA_SYNC_AND_CHECK();
}

template <typename T>
void GpuMat<T>::FlipHorizontal(GpuMat<T>* output) {
  for (size_t slice = 0; slice < depth_; ++slice) {
    CudaFlipHorizontal(array_ptr_ + slice * pitch_ / sizeof(T) * GetHeight(),
                       output->GetPtr() + slice * output->pitch_ / sizeof(T) *
                                              output->GetHeight(),
                       width_,
                       height_,
                       pitch_,
                       output->pitch_);
  }
  CUDA_SYNC_AND_CHECK();
}

template <typename T>
void GpuMat<T>::Rotate(GpuMat<T>* output) {
  for (size_t slice = 0; slice < depth_; ++slice) {
    CudaRotate((T*)((char*)array_ptr_ + slice * pitch_ * GetHeight()),
               (T*)((char*)output->GetPtr() +
                    slice * output->pitch_ * output->GetHeight()),
               width_,
               height_,
               pitch_,
               output->pitch_);
  }
  CUDA_SYNC_AND_CHECK();
  // This is equivalent to the following code:
  //   GpuMat<T> flipped_array(width_, height_, GetDepth());
  //   FlipHorizontal(&flipped_array);
  //   flipped_array.Transpose(output);
}

template <typename T>
void GpuMat<T>::Read(const std::string& path) {
  std::fstream text_file(path, std::ios::in | std::ios::binary);
  CHECK(text_file.is_open()) << path;

  size_t width;
  size_t height;
  size_t depth;
  char unused_char;
  text_file >> width >> unused_char >> height >> unused_char >> depth >>
      unused_char;
  std::streampos pos = text_file.tellg();
  text_file.close();

  std::fstream binary_file(path, std::ios::in | std::ios::binary);
  binary_file.seekg(pos);

  std::vector<T> source(width_ * height_ * depth_);
  ReadBinaryLittleEndian<T>(&binary_file, &source);
  binary_file.close();

  CopyToDevice(source.data(), width_ * sizeof(T));
}

template <typename T>
void GpuMat<T>::Write(const std::string& path) {
  std::vector<T> dest(width_ * height_ * depth_);
  CopyToHost(dest.data(), width_ * sizeof(T));

  std::fstream text_file(path, std::ios::out);
  text_file << width_ << "&" << height_ << "&" << depth_ << "&";
  text_file.close();

  std::fstream binary_file(path,
                           std::ios::out | std::ios::binary | std::ios::app);
  WriteBinaryLittleEndian<T>(&binary_file, dest);
  binary_file.close();
}

template <typename T>
void GpuMat<T>::Write(const std::string& path, const size_t slice) {
  std::vector<T> dest(width_ * height_);
  CUDA_SAFE_CALL(
      cudaMemcpy2D((void*)dest.data(),
                   width_ * sizeof(T),
                   (void*)(array_ptr_ + slice * height_ * pitch_ / sizeof(T)),
                   pitch_,
                   width_ * sizeof(T),
                   height_,
                   cudaMemcpyDeviceToHost));

  std::fstream text_file(path, std::ios::out);
  text_file << width_ << "&" << height_ << "&" << 1 << "&";
  text_file.close();

  std::fstream binary_file(path,
                           std::ios::out | std::ios::binary | std::ios::app);
  WriteBinaryLittleEndian<T>(&binary_file, dest);
  binary_file.close();
}

template <typename T>
void GpuMat<T>::ComputeCudaConfig() {
  blockSize_.x = kBlockDimX;
  blockSize_.y = kBlockDimY;
  blockSize_.z = 1;

  gridSize_.x = (width_ - 1) / kBlockDimX + 1;
  gridSize_.y = (height_ - 1) / kBlockDimY + 1;
  gridSize_.z = 1;
}

#endif  // __CUDACC__

}  // namespace mvs
}  // namespace colmap
