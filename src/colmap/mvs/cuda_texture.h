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

#ifndef COLMAP_SRC_MVS_CUDA_TEXTURE_H_
#define COLMAP_SRC_MVS_CUDA_TEXTURE_H_

#include "colmap/mvs/gpu_mat.h"
#include "colmap/util/cudacc.h"
#include "colmap/util/logging.h"

#include <memory>

#include <cuda_runtime.h>

namespace colmap {
namespace mvs {

template <typename T>
class CudaArrayLayeredTexture {
 public:
  static std::unique_ptr<CudaArrayLayeredTexture<T>> FromGpuMat(
      const cudaTextureDesc& texture_desc, const GpuMat<T>& mat);
  static std::unique_ptr<CudaArrayLayeredTexture<T>> FromHostArray(
      const cudaTextureDesc& texture_desc,
      const size_t width,
      const size_t height,
      const size_t depth,
      const T* data);

  cudaTextureObject_t GetObj() const;

  size_t GetWidth() const;
  size_t GetHeight() const;
  size_t GetDepth() const;

  CudaArrayLayeredTexture(const cudaTextureDesc& texture_desc,
                          const size_t width,
                          const size_t height,
                          const size_t depth);
  ~CudaArrayLayeredTexture();

 private:
  // Define class as non-copyable and non-movable.
  CudaArrayLayeredTexture(CudaArrayLayeredTexture const&) = delete;
  void operator=(CudaArrayLayeredTexture const& obj) = delete;
  CudaArrayLayeredTexture(CudaArrayLayeredTexture&&) = delete;

  const size_t width_;
  const size_t height_;
  const size_t depth_;

  cudaArray_t array_;
  const cudaTextureDesc texture_desc_;
  cudaResourceDesc resource_desc_;
  cudaTextureObject_t texture_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
std::unique_ptr<CudaArrayLayeredTexture<T>>
CudaArrayLayeredTexture<T>::FromGpuMat(const cudaTextureDesc& texture_desc,
                                       const GpuMat<T>& mat) {
  auto array = std::make_unique<CudaArrayLayeredTexture<T>>(
      texture_desc, mat.GetWidth(), mat.GetHeight(), mat.GetDepth());

  cudaMemcpy3DParms params;
  memset(&params, 0, sizeof(params));
  params.extent =
      make_cudaExtent(mat.GetWidth(), mat.GetHeight(), mat.GetDepth());
  params.kind = cudaMemcpyDeviceToDevice;
  params.srcPtr = make_cudaPitchedPtr(
      (void*)mat.GetPtr(), mat.GetPitch(), mat.GetWidth(), mat.GetHeight());
  params.dstArray = array->array_;
  CUDA_SAFE_CALL(cudaMemcpy3D(&params));

  return array;
}

template <typename T>
std::unique_ptr<CudaArrayLayeredTexture<T>>
CudaArrayLayeredTexture<T>::FromHostArray(const cudaTextureDesc& texture_desc,
                                          const size_t width,
                                          const size_t height,
                                          const size_t depth,
                                          const T* data) {
  auto array = std::make_unique<CudaArrayLayeredTexture<T>>(
      texture_desc, width, height, depth);

  cudaMemcpy3DParms params;
  memset(&params, 0, sizeof(params));
  params.extent = make_cudaExtent(width, height, depth);
  params.kind = cudaMemcpyHostToDevice;
  params.srcPtr =
      make_cudaPitchedPtr((void*)data, width * sizeof(T), width, height);
  params.dstArray = array->array_;
  CUDA_SAFE_CALL(cudaMemcpy3D(&params));

  return array;
}

template <typename T>
CudaArrayLayeredTexture<T>::CudaArrayLayeredTexture(
    const cudaTextureDesc& texture_desc,
    const size_t width,
    const size_t height,
    const size_t depth)
    : texture_desc_(texture_desc),
      width_(width),
      height_(height),
      depth_(depth) {
  CHECK_GT(width_, 0);
  CHECK_GT(height_, 0);
  CHECK_GT(depth_, 0);

  cudaExtent extent = make_cudaExtent(width_, height_, depth_);
  cudaChannelFormatDesc fmt = cudaCreateChannelDesc<T>();
  CUDA_SAFE_CALL(cudaMalloc3DArray(&array_, &fmt, extent, cudaArrayLayered));

  memset(&resource_desc_, 0, sizeof(resource_desc_));
  resource_desc_.resType = cudaResourceTypeArray;
  resource_desc_.res.array.array = array_;

  CUDA_SAFE_CALL(cudaCreateTextureObject(
      &texture_, &resource_desc_, &texture_desc_, nullptr));
}

template <typename T>
CudaArrayLayeredTexture<T>::~CudaArrayLayeredTexture() {
  CUDA_SAFE_CALL(cudaFreeArray(array_));
  CUDA_SAFE_CALL(cudaDestroyTextureObject(texture_));
}

template <typename T>
cudaTextureObject_t CudaArrayLayeredTexture<T>::GetObj() const {
  return texture_;
}

template <typename T>
size_t CudaArrayLayeredTexture<T>::GetWidth() const {
  return width_;
}

template <typename T>
size_t CudaArrayLayeredTexture<T>::GetHeight() const {
  return height_;
}

template <typename T>
size_t CudaArrayLayeredTexture<T>::GetDepth() const {
  return depth_;
}

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_CUDA_TEXTURE_H_
