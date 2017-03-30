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
    cudaFreeArray(array_);
    array_ = nullptr;
  }
}

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_CUDA_ARRAY_WRAPPER_H_
