// SPDX-License-Identifier: MIT
// Copyright (c) 2017 Mårten Björkman (CudaSift; see ../LICENSE.CudaSift)
// Copyright (c) 2026 Yonatan Simson (see ../LICENSE)

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#include <cstdio>

#include "cudautils.h"
#include "cudaImage.h"

int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }
int iAlignUp(int a, int b) { return (a%b != 0) ?  (a - a%b + b) : a; }

void CudaImage::Allocate(int w, int h, int p, bool host, float *devmem, float *hostmem) 
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
    if (d_data==NULL)
      printf("Failed to allocate device data\n");
    d_internalAlloc = true;
  }
  if (host && hostmem==NULL) {
    h_data = (float *)malloc(sizeof(float)*pitch*height);
    h_internalAlloc = true;
  }
}

CudaImage::CudaImage() : 
  width(0), height(0), d_data(NULL), h_data(NULL), d_internalAlloc(false), h_internalAlloc(false)
{

}

CudaImage::~CudaImage()
{
  if (d_internalAlloc && d_data!=NULL)
    safeCall(cusiftFree(d_data));
  d_data = NULL;
  if (h_internalAlloc && h_data!=NULL) 
    free(h_data);
  h_data = NULL;
}
  
double CudaImage::Download()  
{
  TimerGPU timer(0);
  int p = sizeof(float)*pitch;
  if (d_data!=NULL && h_data!=NULL) 
    safeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Download time =               %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}


 
