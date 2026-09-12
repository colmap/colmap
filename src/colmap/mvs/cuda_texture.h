// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/gpu_mat.h"
#include "colmap/util/cudacc.h"
#include "colmap/util/logging.h"

#include <memory>

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
  THROW_CHECK_GT(width_, 0);
  THROW_CHECK_GT(height_, 0);
  THROW_CHECK_GT(depth_, 0);

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
  CUDA_SAFE_CALL(cudaDestroyTextureObject(texture_));
  CUDA_SAFE_CALL(cudaFreeArray(array_));
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
