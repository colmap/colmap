//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#include <cstdio>

#include "cudautils.h"
#include "cudaImage.h"

int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }
int iDivDown(int a, int b) { return a/b; }
int iAlignUp(int a, int b) { return (a%b != 0) ?  (a - a%b + b) : a; }
int iAlignDown(int a, int b) { return a - a%b; }

void CudaImage::Allocate(int w, int h, int p, bool host, float *devmem, float *hostmem) 
{
  width = w;
  height = h; 
  pitch = p; 
  d_data = devmem;
  h_data = hostmem; 
  t_data = NULL; 
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
  width(0), height(0), d_data(NULL), h_data(NULL), t_data(NULL), d_internalAlloc(false), h_internalAlloc(false)
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
  if (t_data!=NULL) 
    safeCall(cudaFreeArray((cudaArray *)t_data));
  t_data = NULL;
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

double CudaImage::Readback()
{
  TimerGPU timer(0);
  int p = sizeof(float)*pitch;
  safeCall(cudaMemcpy2D(h_data, sizeof(float)*width, d_data, p, sizeof(float)*width, height, cudaMemcpyDeviceToHost));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Readback time =               %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

double CudaImage::InitTexture()
{
  TimerGPU timer(0);
  cudaChannelFormatDesc t_desc = cudaCreateChannelDesc<float>(); 
  safeCall(cudaMallocArray((cudaArray **)&t_data, &t_desc, pitch, height)); 
  if (t_data==NULL)
    printf("Failed to allocated texture data\n");
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("InitTexture time =            %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}
 
double CudaImage::CopyToTexture(CudaImage &dst, bool host)
{
  if (dst.t_data==NULL) {
    printf("Error CopyToTexture: No texture data\n");
    return 0.0;
  }
  if ((!host || h_data==NULL) && (host || d_data==NULL)) {
    printf("Error CopyToTexture: No source data\n");
    return 0.0;
  }
  TimerGPU timer(0);
  if (host)
    safeCall(cudaMemcpyToArray((cudaArray *)dst.t_data, 0, 0, h_data, sizeof(float)*pitch*dst.height, cudaMemcpyHostToDevice));
  else
    safeCall(cudaMemcpyToArray((cudaArray *)dst.t_data, 0, 0, d_data, sizeof(float)*pitch*dst.height, cudaMemcpyDeviceToDevice));
  safeCall(cudaDeviceSynchronize());
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("CopyToTexture time =          %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}
