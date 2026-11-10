// SPDX-License-Identifier: MIT
// Copyright (c) 2017 Mårten Björkman (CudaSift; see ../LICENSE.CudaSift)
// Copyright (c) 2026 Yonatan Simson (see ../LICENSE)

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#ifndef CUDAIMAGE_H
#define CUDAIMAGE_H

// A pitched float image on the device. Owns its device buffer when Allocate
// allocated it, otherwise it is a view of `devMem`; `h_data` is never owned.
// Allocate first frees the buffer the image owns, so `devMem` must not point
// into it. Move-only, so an owning image cannot be freed twice.
class CudaImage {
public:
  int width, height;
  int pitch;
  float *h_data;
  float *d_data;
  bool d_internalAlloc;
public:
  CudaImage();
  ~CudaImage();
  CudaImage(const CudaImage &) = delete;
  CudaImage &operator=(const CudaImage &) = delete;
  CudaImage(CudaImage &&other) noexcept;
  CudaImage &operator=(CudaImage &&other) noexcept;
  void Allocate(int width, int height, int pitch, float *devMem = NULL, float *hostMem = NULL);
  // Copies h_data to the device.
  void Upload();
};

int iDivUp(int a, int b);
int iAlignUp(int a, int b);

#endif // CUDAIMAGE_H
