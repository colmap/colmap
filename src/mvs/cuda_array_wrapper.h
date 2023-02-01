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

#ifndef COLMAP_SRC_MVS_CUDA_ARRAY_WRAPPER_H_
#define COLMAP_SRC_MVS_CUDA_ARRAY_WRAPPER_H_

#include <memory>

#include <cuda_runtime.h>

#include "mvs/gpu_mat.h"
#include "util/cudacc.h"

namespace colmap {
namespace mvs {

template <typename T>
class CudaArrayWrapper {
 public:
  CudaArrayWrapper(const size_t width, const size_t height, const size_t depth);
  ~CudaArrayWrapper();

  const cudaArray* GetPtr() const;
  cudaArray* GetPtr();

  size_t GetWidth() const;
  size_t GetHeight() const;
  size_t GetDepth() const;

  void CopyToDevice(const T* data);
  void CopyToHost(const T* data);
  void CopyFromGpuMat(const GpuMat<T>& array);

 private:
  // Define class as non-copyable and non-movable.
  CudaArrayWrapper(CudaArrayWrapper const&) = delete;
  void operator=(CudaArrayWrapper const& obj) = delete;
  CudaArrayWrapper(CudaArrayWrapper&&) = delete;

  void Allocate();
  void Deallocate();

  cudaArray* array_;

  size_t width_;
  size_t height_;
  size_t depth_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
CudaArrayWrapper<T>::CudaArrayWrapper(const size_t width, const size_t height,
                                      const size_t depth)
    : width_(width), height_(height), depth_(depth), array_(nullptr) {}

template <typename T>
CudaArrayWrapper<T>::~CudaArrayWrapper() {
  Deallocate();
}

template <typename T>
const cudaArray* CudaArrayWrapper<T>::GetPtr() const {
  return array_;
}

template <typename T>
cudaArray* CudaArrayWrapper<T>::GetPtr() {
  return array_;
}

template <typename T>
size_t CudaArrayWrapper<T>::GetWidth() const {
  return width_;
}

template <typename T>
size_t CudaArrayWrapper<T>::GetHeight() const {
  return height_;
}

template <typename T>
size_t CudaArrayWrapper<T>::GetDepth() const {
  return depth_;
}

template <typename T>
void CudaArrayWrapper<T>::CopyToDevice(const T* data) {
  cudaMemcpy3DParms params = {0};
  Allocate();
  params.extent = make_cudaExtent(width_, height_, depth_);
  params.kind = cudaMemcpyHostToDevice;
  params.dstArray = array_;
  params.srcPtr =
      make_cudaPitchedPtr((void*)data, width_ * sizeof(T), width_, height_);
  CUDA_SAFE_CALL(cudaMemcpy3D(&params));
}

template <typename T>
void CudaArrayWrapper<T>::CopyToHost(const T* data) {
  cudaMemcpy3DParms params = {0};
  params.extent = make_cudaExtent(width_, height_, depth_);
  params.kind = cudaMemcpyDeviceToHost;
  params.dstPtr =
      make_cudaPitchedPtr((void*)data, width_ * sizeof(T), width_, height_);
  params.srcArray = array_;
  CUDA_SAFE_CALL(cudaMemcpy3D(&params));
}

template <typename T>
void CudaArrayWrapper<T>::CopyFromGpuMat(const GpuMat<T>& array) {
  Allocate();
  cudaMemcpy3DParms parameters = {0};
  parameters.extent = make_cudaExtent(width_, height_, depth_);
  parameters.kind = cudaMemcpyDeviceToDevice;
  parameters.dstArray = array_;
  parameters.srcPtr = make_cudaPitchedPtr((void*)array.GetPtr(),
                                          array.GetPitch(), width_, height_);
  CUDA_SAFE_CALL(cudaMemcpy3D(&parameters));
}

template <typename T>
void CudaArrayWrapper<T>::Allocate() {
  Deallocate();
  struct cudaExtent extent = make_cudaExtent(width_, height_, depth_);
  cudaChannelFormatDesc fmt = cudaCreateChannelDesc<T>();
  CUDA_SAFE_CALL(cudaMalloc3DArray(&array_, &fmt, extent, cudaArrayLayered));
}

template <typename T>
void CudaArrayWrapper<T>::Deallocate() {
  if (array_ != nullptr) {
    CUDA_SAFE_CALL(cudaFreeArray(array_));
    array_ = nullptr;
  }
}

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_CUDA_ARRAY_WRAPPER_H_
