// SPDX-License-Identifier: MIT
// Copyright (c) 2017 Mårten Björkman (CudaSift; see ../LICENSE.CudaSift)
// Copyright (c) 2026 Yonatan Simson (see ../LICENSE)

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#include <utility>

#include "cudautils.h"
#include "cudaImage.h"

int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }
int iAlignUp(int a, int b) { return (a%b != 0) ?  (a - a%b + b) : a; }

void CudaImage::Allocate(int w, int h, int p, float *devmem, float *hostmem)
{
  width = w;
  height = h;
  pitch = p;
  d_data = devmem;
  h_data = hostmem;
  if (devmem==NULL) {
    // Pool-backed (see cusiftMallocPitch); pitch is the same value
    // cudaMallocPitch returned before, computed rather than queried.
    safeCall(cusiftMallocPitch(&d_data, &pitch, width, height));
    d_internalAlloc = true;
  }
}

CudaImage::CudaImage() :
  width(0), height(0), pitch(0), h_data(NULL), d_data(NULL), d_internalAlloc(false)
{
}

CudaImage::CudaImage(CudaImage &&other) noexcept : CudaImage()
{
  *this = std::move(other);
}

CudaImage &CudaImage::operator=(CudaImage &&other) noexcept
{
  std::swap(width, other.width);
  std::swap(height, other.height);
  std::swap(pitch, other.pitch);
  std::swap(h_data, other.h_data);
  std::swap(d_data, other.d_data);
  std::swap(d_internalAlloc, other.d_internalAlloc);
  return *this;
}

CudaImage::~CudaImage()
{
  // Destructors must not throw; a failed free cannot be acted on here.
  if (d_internalAlloc && d_data!=NULL)
    (void)cusiftFree(d_data);
  d_data = NULL;
}

void CudaImage::Download()
{
  int p = sizeof(float)*pitch;
  if (d_data!=NULL && h_data!=NULL)
    safeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
}
