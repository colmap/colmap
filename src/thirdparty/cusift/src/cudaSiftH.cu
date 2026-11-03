//********************************************************//
// CUDA SIFT extractor by Mårten Björkman aka Celebrandil //
//********************************************************//  

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <array>
#include <vector>
#include <chrono>
#include <map>
#include <string>
#include "cudautils.h"

#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftD.h"
#include "cudaSiftH.h"

#include "cudaSiftD.cu"

// Stage-attribution timers for the performance work (docs/PERFORMANCE.md). Enabled
// by setting CUSIFT_TIMING=1; off, they compile to a getenv-cached bool test.
// ncu is the better tool but needs GPU-counter permissions, and nsys' CUDA
// tracing does not capture under WSL2 -- this covers the "which stage owns the
// milliseconds" question without either. Each scope synchronizes on entry and
// exit, so the numbers attribute correctly but do not sum to the pipelined
// wall time.
namespace {

class StageTimer {
 public:
  static bool Enabled() {
    static const bool e = std::getenv("CUSIFT_TIMING") != nullptr;
    return e;
  }
  explicit StageTimer(const char *name) : name_(name) {
    if (!Enabled()) return;
    safeCall(cudaDeviceSynchronize());
    t0_ = std::chrono::steady_clock::now();
  }
  ~StageTimer() {
    if (!Enabled()) return;
    safeCall(cudaDeviceSynchronize());
    Acc()[name_] += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0_).count();
  }
  static void Report(const char *tag) {
    if (!Enabled()) return;
    std::vector<std::pair<std::string, double>> v(Acc().begin(), Acc().end());
    std::sort(v.begin(), v.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });
    double tot = 0;
    for (const auto &kv : v) tot += kv.second;
    std::fprintf(stderr, "[cusift timing] %s: %.2f ms attributed\n", tag, tot);
    for (const auto &kv : v)
      std::fprintf(stderr, "  %-22s %8.2f ms  %5.1f%%\n", kv.first.c_str(),
                   kv.second, 100.0 * kv.second / tot);
    Acc().clear();
  }

 private:
  static std::map<std::string, double> &Acc() {
    static std::map<std::string, double> acc;
    return acc;
  }
  const char *name_;
  std::chrono::steady_clock::time_point t0_;
};

}  // namespace

void InitCuda(int devNum)
{
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  if (!nDevices) {
    std::cerr << "No CUDA devices available" << std::endl;
    return;
  }
  devNum = std::min(nDevices-1, devNum);
  deviceInit(devNum);
  cusiftInitMemPool();
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devNum);
  printf("Device Number: %d\n", devNum);
  printf("  Device name: %s\n", prop.name);
  printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("  Total Global Memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
  printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
  printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
  printf("  Multiprocessor Count: %d\n\n", prop.multiProcessorCount);
}

float *AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp)
{
  TimerGPU timer(0);
  const int nd = NUM_SCALES + 3;
  int w = width*(scaleUp ? 2 : 1); 
  int h = height*(scaleUp ? 2 : 1);
  int p = iAlignUp(w, 128);
  int size = h*p;                 // image sizes
  int sizeTmp = nd*h*p;           // laplace buffer sizes
  for (int i=0;i<numOctaves;i++) {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h*p;
    sizeTmp += nd*h*p; 
  }
  float *memoryTmp = NULL; 
  size_t pitch;
  size += sizeTmp;
  safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float)));
#ifdef VERBOSE
  printf("Allocated memory size: %d bytes\n", size);
  printf("Memory allocation time =      %.2f ms\n\n", timer.read());
#endif
  return memoryTmp;
}

void FreeSiftTempMemory(float *memoryTmp)
{
  if (memoryTmp)
    safeCall(cudaFree(memoryTmp));
}

void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, bool scaleUp, float *tempMemory) 
{
  TimerGPU timer(0);
  unsigned int *d_PointCounterAddr;
  safeCall(cudaGetSymbolAddress((void**)&d_PointCounterAddr, d_PointCounter));
  safeCall(cudaMemset(d_PointCounterAddr, 0, (8*2+1)*sizeof(int)));
  safeCall(cudaMemcpyToSymbol(d_MaxNumPoints, &siftData.maxPts, sizeof(int)));

  const int nd = NUM_SCALES + 3;
  int w = img.width*(scaleUp ? 2 : 1);
  int h = img.height*(scaleUp ? 2 : 1);
  int p = iAlignUp(w, 128);
  int width = w, height = h;
  int size = h*p;                 // image sizes
  int sizeTmp = nd*h*p;           // laplace buffer sizes
  for (int i=0;i<numOctaves;i++) {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h*p;
    sizeTmp += nd*h*p; 
  }
  float *memoryTmp = tempMemory; 
  size += sizeTmp;
  if (!tempMemory) {
    size_t pitch;
    safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float)));
#ifdef VERBOSE
    printf("Allocated memory size: %d bytes\n", size);
    printf("Memory allocation time =      %.2f ms\n\n", timer.read());
#endif
  }
  float *memorySub = memoryTmp + sizeTmp;

  CudaImage lowImg;
  lowImg.Allocate(width, height, iAlignUp(width, 128), false, memorySub);
  if (!scaleUp) {
    float kernel[8*12*16];
    PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
    safeCall(cudaMemcpyToSymbolAsync(d_LaplaceKernel, kernel, 8*12*16*sizeof(float)));
    LowPass(lowImg, img, max(initBlur, 0.001f));
    TimerGPU timer1(0);
    ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale, 1.0f, memoryTmp, memorySub + height*iAlignUp(width, 128));
    safeCall(cudaMemcpy(&siftData.numPts, &d_PointCounterAddr[2*numOctaves], sizeof(int), cudaMemcpyDeviceToHost)); 
    siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
    printf("SIFT extraction time =        %.2f ms %d\n", timer1.read(), siftData.numPts);
  } else {
    CudaImage upImg;
    upImg.Allocate(width, height, iAlignUp(width, 128), false, memoryTmp);
    TimerGPU timer1(0); 
    ScaleUp(upImg, img);
    LowPass(lowImg, upImg, max(initBlur, 0.001f));
    float kernel[8*12*16];
    PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
    safeCall(cudaMemcpyToSymbolAsync(d_LaplaceKernel, kernel, 8*12*16*sizeof(float)));
    ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale*2.0f, 1.0f, memoryTmp, memorySub + height*iAlignUp(width, 128));
    safeCall(cudaMemcpy(&siftData.numPts, &d_PointCounterAddr[2*numOctaves], sizeof(int), cudaMemcpyDeviceToHost)); 
    siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
    RescalePositions(siftData, 0.5f);
    printf("SIFT extraction time =        %.2f ms\n", timer1.read());
  } 
  
  if (!tempMemory)
    safeCall(cudaFree(memoryTmp));
#ifdef MANAGEDMEM
  safeCall(cudaDeviceSynchronize());
#else
  if (siftData.h_data)
    safeCall(cudaMemcpy(siftData.h_data, siftData.d_data, sizeof(SiftPoint)*siftData.numPts, cudaMemcpyDeviceToHost));
#endif
  double totTime = timer.read();
  printf("Incl prefiltering & memcpy =  %.2f ms %d\n\n", totTime, siftData.numPts);
}

int ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub) 
{
#ifdef VERBOSE
  TimerGPU timer(0);
#endif
  int w = img.width;
  int h = img.height;
  if (numOctaves>1) {
    CudaImage subImg;
    int p = iAlignUp(w/2, 128);
    subImg.Allocate(w/2, h/2, p, false, memorySub); 
    ScaleDown(subImg, img, 0.5f);
    float totInitBlur = (float)sqrt(initBlur*initBlur + 0.5f*0.5f) / 2.0f;
    ExtractSiftLoop(siftData, subImg, numOctaves-1, totInitBlur, thresh, lowestScale, subsampling*2.0f, memoryTmp, memorySub + (h/2)*p);
  }
  ExtractSiftOctave(siftData, img, numOctaves, thresh, lowestScale, subsampling, memoryTmp);
#ifdef VERBOSE
  double totTime = timer.read();
  printf("ExtractSift time total =      %.2f ms %d\n\n", totTime, numOctaves);
#endif
  return 0;
}

void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh, float lowestScale, float subsampling, float *memoryTmp)
{
  const int nd = NUM_SCALES + 3;
#ifdef VERBOSE
  unsigned int *d_PointCounterAddr;
  safeCall(cudaGetSymbolAddress((void**)&d_PointCounterAddr, d_PointCounter));
  unsigned int fstPts, totPts;
  safeCall(cudaMemcpy(&fstPts, &d_PointCounterAddr[2*octave-1], sizeof(int), cudaMemcpyDeviceToHost)); 
  TimerGPU timer0;
#endif
  CudaImage diffImg[nd];
  int w = img.width; 
  int h = img.height;
  int p = iAlignUp(w, 128);
  for (int i=0;i<nd-1;i++) 
    diffImg[i].Allocate(w, h, p, false, memoryTmp + i*p*h); 

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = img.d_data;
  resDesc.res.pitch2D.width = img.width;
  resDesc.res.pitch2D.height = img.height;
  resDesc.res.pitch2D.pitchInBytes = img.pitch*sizeof(float);  
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeClamp;
  texDesc.addressMode[1]   = cudaAddressModeClamp;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

#ifdef VERBOSE
  TimerGPU timer1;
#endif
  float baseBlur = pow(2.0f, -1.0f/NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
  LaplaceMulti(texObj, img, diffImg, octave); 
  FindPointsMulti(diffImg, siftData, thresh, 10.0f, 1.0f/NUM_SCALES, lowestScale/subsampling, subsampling, octave);
#ifdef VERBOSE
  double gpuTimeDoG = timer1.read();
  TimerGPU timer4;
#endif
  ComputeOrientations(texObj, img, siftData, octave); 
  ExtractSiftDescriptors(texObj, siftData, subsampling, octave); 
  //OrientAndExtract(texObj, siftData, subsampling, octave); 
  
  safeCall(cudaDestroyTextureObject(texObj));
#ifdef VERBOSE
  double gpuTimeSift = timer4.read();
  double totTime = timer0.read();
  printf("GPU time : %.2f ms + %.2f ms + %.2f ms = %.2f ms\n", totTime-gpuTimeDoG-gpuTimeSift, gpuTimeDoG, gpuTimeSift, totTime);
  safeCall(cudaMemcpy(&totPts, &d_PointCounterAddr[2*octave+1], sizeof(int), cudaMemcpyDeviceToHost));
  totPts = (totPts<siftData.maxPts ? totPts : siftData.maxPts);
  if (totPts>0) 
    printf("           %.2f ms / DoG,  %.4f ms / Sift,  #Sift = %d\n", gpuTimeDoG/NUM_SCALES, gpuTimeSift/(totPts-fstPts), totPts-fstPts); 
#endif
}

void InitSiftData(SiftData &data, int num, bool host, bool dev)
{
  data.numPts = 0;
  data.maxPts = num;
  int sz = sizeof(SiftPoint)*num;
#ifdef MANAGEDMEM
  safeCall(cudaMallocManaged((void **)&data.m_data, sz));
#else
  data.h_data = NULL;
  if (host)
    data.h_data = (SiftPoint *)malloc(sz);
  data.d_data = NULL;
  if (dev)
    safeCall(cudaMallocAsync((void **)&data.d_data, sz, 0));
#endif
}

void FreeSiftData(SiftData &data)
{
#ifdef MANAGEDMEM
  safeCall(cudaFree(data.m_data));
#else
  if (data.d_data!=NULL)
    safeCall(cusiftFree(data.d_data));
  data.d_data = NULL;
  if (data.h_data!=NULL)
    free(data.h_data);
#endif
  data.numPts = 0;
  data.maxPts = 0;
}

void PrintSiftData(SiftData &data)
{
#ifdef MANAGEDMEM
  SiftPoint *h_data = data.m_data;
#else
  SiftPoint *h_data = data.h_data;
  if (data.h_data==NULL) {
    h_data = (SiftPoint *)malloc(sizeof(SiftPoint)*data.maxPts);
    safeCall(cudaMemcpy(h_data, data.d_data, sizeof(SiftPoint)*data.numPts, cudaMemcpyDeviceToHost));
    data.h_data = h_data;
  }
#endif
  for (int i=0;i<data.numPts;i++) {
    printf("xpos         = %.2f\n", h_data[i].xpos);
    printf("ypos         = %.2f\n", h_data[i].ypos);
    printf("scale        = %.2f\n", h_data[i].scale);
    printf("sharpness    = %.2f\n", h_data[i].sharpness);
    printf("edgeness     = %.2f\n", h_data[i].edgeness);
    printf("orientation  = %.2f\n", h_data[i].orientation);
    printf("score        = %.2f\n", h_data[i].score);
    float *siftData = (float*)&h_data[i].data;
    for (int j=0;j<8;j++) {
      if (j==0) 
	printf("data = ");
      else 
	printf("       ");
      for (int k=0;k<16;k++)
	if (siftData[j+8*k]<0.05)
	  printf(" .   ");
	else
	  printf("%.2f ", siftData[j+8*k]);
      printf("\n");
    }
  }
  printf("Number of available points: %d\n", data.numPts);
  printf("Number of allocated points: %d\n", data.maxPts);
}

///////////////////////////////////////////////////////////////////////////////
// Host side master functions
///////////////////////////////////////////////////////////////////////////////

double ScaleDown(CudaImage &res, CudaImage &src, float variance)
{
  static float oldVariance = -1.0f;
  if (res.d_data==NULL || src.d_data==NULL) {
    printf("ScaleDown: missing data\n");
    return 0.0;
  }
  if (oldVariance!=variance) {
    float h_Kernel[5];
    float kernelSum = 0.0f;
    for (int j=0;j<5;j++) {
      h_Kernel[j] = (float)expf(-(double)(j-2)*(j-2)/2.0/variance);      
      kernelSum += h_Kernel[j];
    }
    for (int j=0;j<5;j++)
      h_Kernel[j] /= kernelSum;  
    safeCall(cudaMemcpyToSymbol(d_ScaleDownKernel, h_Kernel, 5*sizeof(float)));
    oldVariance = variance;
  }
#if 0
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4, SCALEDOWN_H + 4);
  ScaleDownDenseShift<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
#else
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4);
  ScaleDown<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
#endif
  checkMsg("ScaleDown() execution failed\n");
  return 0.0;
}

double ScaleUp(CudaImage &res, CudaImage &src)
{
  if (res.d_data==NULL || src.d_data==NULL) {
    printf("ScaleUp: missing data\n");
    return 0.0;
  }
  dim3 blocks(iDivUp(res.width, SCALEUP_W), iDivUp(res.height, SCALEUP_H));
  dim3 threads(SCALEUP_W/2, SCALEUP_H/2);
  ScaleUp<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch); 
  checkMsg("ScaleUp() execution failed\n");
  return 0.0;
}   

double ComputeOrientations(cudaTextureObject_t texObj, CudaImage &src, SiftData &siftData, int octave)
{
  dim3 blocks(512); 
#ifdef MANAGEDMEM
  ComputeOrientationsCONST<<<blocks, threads>>>(texObj, siftData.m_data, octave);
#else
#if 1
  dim3 threads(11*11);
  ComputeOrientationsCONST<<<blocks, threads>>>(texObj, siftData.d_data, octave);
#else
  dim3 threads(256); 
  ComputeOrientationsCONSTNew<<<blocks, threads>>>(src.d_data, src.width, src.pitch, src.height, siftData.d_data, octave);
#endif
#endif
  checkMsg("ComputeOrientations() execution failed\n");
  return 0.0;
}

double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave)
{
  dim3 blocks(512); 
  dim3 threads(16, 8);
#ifdef MANAGEDMEM
  ExtractSiftDescriptorsCONST<<<blocks, threads>>>(texObj, siftData.m_data, subsampling, octave);
#else
  ExtractSiftDescriptorsCONSTNew<<<blocks, threads>>>(texObj, siftData.d_data, subsampling, octave);
#endif
  checkMsg("ExtractSiftDescriptors() execution failed\n");
  return 0.0; 
}

double OrientAndExtract(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave)
{
  dim3 blocks(256); 
  dim3 threads(128);
#ifdef MANAGEDMEM
  OrientAndExtractCONST<<<blocks, threads>>>(texObj, siftData.m_data, subsampling, octave);
#else
  OrientAndExtractCONST<<<blocks, threads>>>(texObj, siftData.d_data, subsampling, octave);
#endif
  checkMsg("OrientAndExtract() execution failed\n");
  return 0.0;
}

double RescalePositions(SiftData &siftData, float scale)
{
  dim3 blocks(iDivUp(siftData.numPts, 64));
  dim3 threads(64);
  RescalePositions<<<blocks, threads>>>(siftData.d_data, siftData.numPts, scale);
  checkMsg("RescapePositions() execution failed\n");
  return 0.0; 
}

double LowPass(CudaImage &res, CudaImage &src, float scale)
{
  float kernel[2*LOWPASS_R+1];
  static float oldScale = -1.0f;
  if (scale!=oldScale) {
    float kernelSum = 0.0f;
    float ivar2 = 1.0f/(2.0f*scale*scale);
    for (int j=-LOWPASS_R;j<=LOWPASS_R;j++) {
      kernel[j+LOWPASS_R] = (float)expf(-(double)j*j*ivar2);
      kernelSum += kernel[j+LOWPASS_R]; 
    }
    for (int j=-LOWPASS_R;j<=LOWPASS_R;j++) 
      kernel[j+LOWPASS_R] /= kernelSum;  
    safeCall(cudaMemcpyToSymbol(d_LowPassKernel, kernel, (2*LOWPASS_R+1)*sizeof(float)));
    oldScale = scale;
  }  
  int width = res.width;
  int pitch = res.pitch;
  int height = res.height;
  dim3 blocks(iDivUp(width, LOWPASS_W), iDivUp(height, LOWPASS_H));
#if 1
  dim3 threads(LOWPASS_W+2*LOWPASS_R, 4); 
  LowPassBlock<<<blocks, threads>>>(src.d_data, res.d_data, width, pitch, height);
#else
  dim3 threads(LOWPASS_W+2*LOWPASS_R, LOWPASS_H);
  LowPass<<<blocks, threads>>>(src.d_data, res.d_data, width, pitch, height);
#endif
  checkMsg("LowPass() execution failed\n");
  return 0.0; 
}

//==================== Multi-scale functions ===================//

void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel)
{
  if (numOctaves>1) {
    float totInitBlur = (float)sqrt(initBlur*initBlur + 0.5f*0.5f) / 2.0f;
    PrepareLaplaceKernels(numOctaves-1, totInitBlur, kernel);
  }
  float scale = pow(2.0f, -1.0f/NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
  for (int i=0;i<NUM_SCALES+3;i++) {
    float kernelSum = 0.0f;
    float var = scale*scale - initBlur*initBlur;
    for (int j=0;j<=LAPLACE_R;j++) {
      kernel[numOctaves*12*16 + 16*i + j] = (float)expf(-(double)j*j/2.0/var);
      kernelSum += (j==0 ? 1 : 2)*kernel[numOctaves*12*16 + 16*i + j]; 
    }
    for (int j=0;j<=LAPLACE_R;j++)
      kernel[numOctaves*12*16 + 16*i + j] /= kernelSum;
    scale *= diffScale;
  }
}
 
double LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage, CudaImage *results, int octave) 
{
  int width = results[0].width;
  int pitch = results[0].pitch;
  int height = results[0].height;
#if 1
  dim3 threads(LAPLACE_W+2*LAPLACE_R);
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  LaplaceMultiMem<<<blocks, threads>>>(baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);
  dim3 blocks(iDivUp(width, LAPLACE_W), iDivUp(height, LAPLACE_H));
  LaplaceMultiMemTest<<<blocks, threads>>>(baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  LaplaceMultiMemOld<<<blocks, threads>>>(baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  LaplaceMultiTex<<<blocks, threads>>>(texObj, results[0].d_data, width, pitch, height, octave);
#endif
  checkMsg("LaplaceMulti() execution failed\n");
  return 0.0; 
}

double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave)
{
  if (sources->d_data==NULL) {
    printf("FindPointsMulti: missing data\n");
    return 0.0;
  }
  int w = sources->width;
  int p = sources->pitch;
  int h = sources->height;
#if 0
  dim3 blocks(iDivUp(w, MINMAX_W)*NUM_SCALES, iDivUp(h, MINMAX_H));
  dim3 threads(MINMAX_W + 2, MINMAX_H);
  FindPointsMultiTest<<<blocks, threads>>>(sources->d_data, siftData.d_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave); 
#endif
#if 1
  dim3 blocks(iDivUp(w, MINMAX_W)*NUM_SCALES, iDivUp(h, MINMAX_H));
  dim3 threads(MINMAX_W + 2); 
#ifdef MANAGEDMEM
  FindPointsMulti<<<blocks, threads>>>(sources->d_data, siftData.m_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave); 
#else
  FindPointsMultiNew<<<blocks, threads>>>(sources->d_data, siftData.d_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
#endif
#endif
  checkMsg("FindPointsMulti() execution failed\n");
  return 0.0;
}

//======= Part 2b': VLFeat-schedule pyramid + descriptor (clean-room from ====//
//======= colmap/src/thirdparty/VLFeat/sift.c, BSD; not SiftGPU)          ====//
// Replaces the original Part 2a per-octave-texture harness. Builds an
// INDEPENDENT scale-space pyramid (not grafted onto cusift's existing octave
// images, whose accumulated blur is in CudaSift's own internal bookkeeping
// convention) referenced to VLFeat's own absolute-sigma schedule, so that
// octave/level assignment for externally supplied keypoints is exact by
// construction rather than cross-referenced against a different convention.
// This merges the original Part 2b (descriptor rewrite) and Part 2e (scale
// convention) -- the scale convention comes out native here, not fitted.
//
// Formulas verified directly against colmap/src/thirdparty/VLFeat/sift.c:
//   vl_sift_new (874-935):                sigman=0.5, sigma0=1.6*sigmak,
//                                          sigmak=2^(1/S), s_min=-1, s_max=S+1
//   vl_sift_process_first_octave (977-1062): base-level blur-up, per-level
//                                          incremental blur dsigma0*sigmak^s
//   vl_sift_process_next_octave (1097-1136): pure 2x decimation of the
//                                          previous octave's level s_min+S,
//                                          NO extra blur needed (sa==sb by
//                                          construction -- verified exactly)
//   copy_and_downsample (836-851):        pure decimation, no anti-alias blur
//                                          (NOT the same as cusift's own
//                                          ScaleDown, which bakes one in)
//   update_gradient (1447-1484):          central differences, one-sided at
//                                          borders, angle wrapped to [0,2pi)
//   vl_sift_keypoint_init (2161-2190):    sigma -> (octave, level) inversion;
//                                          replicated exactly here, since an
//                                          off-by-one here silently misreads
//                                          the wrong pyramid level

// Replaces reused-LowPass calls in BuildVLFeatOctave -- see cudaSiftD.cu's
// GaussianBlurHorizontalKernel/GaussianBlurVerticalKernel comment for why
// cusift's own LowPass (fixed radius=4) was silently under-blurring the
// VLFeat sigma schedule.
// Tier 1.3: memoized slot in the constant kernel bank for `sigma`. Keyed on the
// exact float bit pattern, so a hit reuses weights that are bit-identical to
// what the old per-blur upload would have produced; a miss computes them with
// the same expressions and appends. ~8 distinct sigmas per process, so all
// misses happen on the first image and every later blur is upload-free.
//
// thread_local, not a plain static (docs/COLMAP_INTEGRATION.md §6 open item 2):
// COLMAP creates one extraction thread per GPU index, so a process-wide bank
// map is racy AND wrong across devices -- d_AdaptiveGaussKernel is a per-device
// __constant__ symbol, so a "hit" recorded by a thread on device 0 would hand a
// thread on device 1 an offset it never uploaded to. Per-thread bookkeeping is
// correct for both: each thread uploads to the device it is bound to. Two
// threads on the same device duplicate ~27 floats per distinct sigma in a
// 4096-float bank, which is free.
static int GaussKernelSlot(float sigma, int radius)
{
  static thread_local std::vector<std::pair<uint32_t, int>> slots;   // (sigma bits, offset)
  static thread_local int used = 0;
  uint32_t key;
  std::memcpy(&key, &sigma, sizeof(key));
  for (const auto &s : slots)
    if (s.first == key) return s.second;

  int n = 2 * radius + 1;
  if (used + n > GAUSS_BANK_FLOATS) {
    std::fprintf(stderr, "GaussKernelSlot: constant bank exhausted (%d slots, %d floats)\n",
                 (int)slots.size(), used);
    std::abort();
  }
  std::vector<float> kernel(n);
  float sum = 0.0f;
  for (int j = -radius; j <= radius; j++) {
    kernel[j + radius] = std::exp(-(float)(j * j) / (2.0f * sigma * sigma));
    sum += kernel[j + radius];
  }
  for (auto &v : kernel) v /= sum;
  safeCall(cudaMemcpyToSymbol(d_AdaptiveGaussKernel, kernel.data(),
                              kernel.size() * sizeof(float), used * sizeof(float)));
  int offset = used;
  used += n;
  slots.emplace_back(key, offset);
  return offset;
}

static void AdaptiveGaussianBlur(CudaImage &dst, CudaImage &src, float sigma)
{
  int radius = std::min((int)std::ceil(4.0f * sigma), MAX_GAUSS_RADIUS);
  radius = std::max(radius, 1);
  int kOffset = GaussKernelSlot(sigma, radius);

  CudaImage tmp;
  tmp.Allocate(src.width, src.height, iAlignUp(src.width, 128), false, NULL);
  dim3 threads(16, 16);
  dim3 blocks(iDivUp(src.width, 16), iDivUp(src.height, 16));
  GaussianBlurHorizontalKernel<<<blocks, threads>>>(tmp.d_data, src.d_data,
      src.width, tmp.pitch, src.pitch, src.height, radius, kOffset);
  checkMsg("GaussianBlurHorizontalKernel() execution failed\n");
  GaussianBlurVerticalKernel<<<blocks, threads>>>(dst.d_data, tmp.d_data,
      dst.width, dst.pitch, tmp.pitch, dst.height, radius, kOffset);
  checkMsg("GaussianBlurVerticalKernel() execution failed\n");
}

static void Decimate2x(CudaImage &dst, CudaImage &src)
{
  dim3 threads(16, 16);
  dim3 blocks(iDivUp(dst.width, 16), iDivUp(dst.height, 16));
  PureDecimate2xKernel<<<blocks, threads>>>(dst.d_data, src.d_data,
      dst.width, dst.pitch, dst.height, src.pitch);
  checkMsg("PureDecimate2xKernel() execution failed\n");
}

// Test-only: 2x upsample matching VLFeat's copy_and_upsample_rows (see
// Upsample2xHorizontalKernel/Upsample2xVerticalKernel comment). Used solely
// to test the o_min=-1 hypothesis (see plan) -- NOT wired into any
// production path.
static void Upsample2x(CudaImage &dst, CudaImage &src)
{
  CudaImage tmp;
  tmp.Allocate(2*src.width, src.height, iAlignUp(2*src.width, 128), false, NULL);
  dim3 threadsH(16, 16);
  dim3 blocksH(iDivUp(2*src.width, 16), iDivUp(src.height, 16));
  Upsample2xHorizontalKernel<<<blocksH, threadsH>>>(tmp.d_data, src.d_data,
      src.width, tmp.pitch, src.pitch, src.height);
  checkMsg("Upsample2xHorizontalKernel() execution failed\n");
  dim3 threadsV(16, 16);
  dim3 blocksV(iDivUp(dst.width, 16), iDivUp(2*src.height, 16));
  Upsample2xVerticalKernel<<<blocksV, threadsV>>>(dst.d_data, tmp.d_data,
      dst.width, dst.pitch, tmp.pitch, src.height);
  checkMsg("Upsample2xVerticalKernel() execution failed\n");
}

// Debug-only: poison a gradient level slice with signaling NaN -- see
// FillPlaneKernel's comment. Also serves as a live test of the Tier 2.3 index
// invariant: if a Debug build still passes the accuracy gate with every skipped
// plane poisoned, nothing reads them.
static void PoisonGradientLevel(CudaImage &combined, int levelIdx, int w, int h, int p)
{
  uint32_t snanBits = 0x7FA00000u;   // signaling NaN
  float snan;
  std::memcpy(&snan, &snanBits, sizeof(snan));
  dim3 threads(16, 16);
  dim3 blocks(iDivUp(w, 16), iDivUp(h, 16));
  FillPlaneKernel<<<blocks, threads>>>(combined.d_data + (size_t)levelIdx * h * p,
                                       w, p, h, snan);
  checkMsg("FillPlaneKernel() execution failed\n");
}

static void ComputeGradientPlaneFor(CudaImage &level, CudaImage &gradMag, CudaImage &gradAng)
{
  dim3 threads(16, 16);
  dim3 blocks(iDivUp(level.width, 16), iDivUp(level.height, 16));
  ComputeGradientPlaneKernel<<<blocks, threads>>>(level.d_data, level.width, level.pitch, level.height,
      gradMag.d_data, gradAng.d_data);
  checkMsg("ComputeGradientPlaneKernel() execution failed\n");
}

// One VLFeat octave: nLevels = numScales+3 Gaussian levels (s = s_min..s_max,
// s_min=-1) stored as contiguous slices of one buffer each for the Gaussian
// image, gradient magnitude, and gradient angle (slice stride = height*pitch).
struct VLFeatOctaveBuffers {
  int width, height, pitch;
  int nLevels;
  CudaImage gauss;    // nLevels slices, self-allocated as one block below
  CudaImage gradMag;  // see the Tier 3.1 note on ComputeGradientPlaneKernel for
  CudaImage gradAng;  // why these are NOT interleaved into one float2 plane
};

// Non-owning per-level view into a combined (nLevels-stacked) buffer -- slice
// i occupies rows [i*h, (i+1)*h) of the combined image.
static CudaImage LevelView(CudaImage &combined, int levelIdx, int w, int h, int p)
{
  CudaImage v;
  v.Allocate(w, h, p, false, combined.d_data + (size_t)levelIdx * h * p);
  return v;
}

// docs/PERFORMANCE.md Tier 2.3 (the unconditional half): gradient planes are computed
// only for the level indices a consumer can actually address. Keypoint levels are
// `is` in [0, numScales-1] and every consumer indexes gradients as `is - s_min`
// (see AssignOctaveLevel, which clamps to exactly that, and the integrated
// pipeline's `empty[0] = is + 1`), so indices 1..numScales are the whole reachable
// set: the s_min and the two top levels exist in the Gaussian pyramid for DoG
// purposes but can never host a keypoint. That is 3 of 6 planes for
// numScales=3 -- half the gradient bandwidth of the pyramid stage, and at octave
// -1 those are 33 MB planes. Detection-only callers do not read gradients at all.
// `allGradients` restores the full set for DumpPyramidDebug, which compares every
// plane against vlfeat_dump's ground truth.
//
// The skipped planes are left UNINITIALIZED rather than zeroed (zeroing would
// spend the bandwidth this saves); the index invariant above is what keeps them
// unreachable.
static void BuildVLFeatOctave(VLFeatOctaveBuffers &oct, CudaImage *prevTopLevel /*NULL for octave 0*/,
                              CudaImage *img01 /*only used for octave 0*/, int numScales,
                              bool allGradients = false)
{
  const int s_min = -1, s_max = numScales + 1;
  const int nLevels = s_max - s_min + 1;  // = numScales+3
  const float sigmak = std::pow(2.0f, 1.0f / numScales);
  const float sigma0 = 1.6f * sigmak;
  const float dsigma0 = sigma0 * std::sqrt(1.0f - 1.0f / (sigmak * sigmak));

  int w, h;
  if (prevTopLevel) { w = prevTopLevel->width / 2; h = prevTopLevel->height / 2; }
  else              { w = img01->width;            h = img01->height; }

  // NOTE: CudaImage::Allocate with devMem=NULL self-allocates via
  // cudaMallocPitch, which OVERWRITES `pitch` with whatever CUDA actually
  // returns -- the requested value passed in is discarded. Must read back
  // the true pitch after allocation and use THAT everywhere below, not the
  // iAlignUp guess, or level-slice addressing silently corrupts.
  oct.gauss.Allocate(w, h * nLevels, iAlignUp(w, 128), false, NULL);
  oct.gradMag.Allocate(w, h * nLevels, iAlignUp(w, 128), false, NULL);
  oct.gradAng.Allocate(w, h * nLevels, iAlignUp(w, 128), false, NULL);
  int p = oct.gauss.pitch;
  // gradMag/gradAng were allocated with the identical (w, h*nLevels) request,
  // so cudaMallocPitch is expected to return the same pitch deterministically.
  if (oct.gradMag.pitch != p || oct.gradAng.pitch != p) {
    std::fprintf(stderr, "BuildVLFeatOctave: unexpected pitch mismatch (%d vs %d/%d)\n",
                p, oct.gradMag.pitch, oct.gradAng.pitch);
    std::abort();
  }
  oct.width = w; oct.height = h; oct.pitch = p; oct.nLevels = nLevels;

  CudaImage lvl0 = LevelView(oct.gauss, 0, w, h, p);
  if (prevTopLevel) {
    Decimate2x(lvl0, *prevTopLevel);
    // sa == sb by construction at octave transitions (verified against
    // vl_sift_process_next_octave: sigma0*sigmak^s_min == sigma0*sigmak^(s_best-S)
    // when s_best=s_min+numScales) -- no extra blur needed here.
  } else {
    // AdaptiveGaussianBlur takes explicit separate src/dst pitches, so no
    // same-pitch assumption between img01 and lvl0 is needed here.
    float sa = sigma0 * std::pow(sigmak, (float)s_min);
    float sb = 0.5f;  // sigman, o_min=0 (non-upsampled)
    if (sa > sb) AdaptiveGaussianBlur(lvl0, *img01, std::sqrt(sa * sa - sb * sb));
    else         AdaptiveGaussianBlur(lvl0, *img01, 0.001f);
  }
  for (int s = s_min + 1; s <= s_max; s++) {
    CudaImage prev = LevelView(oct.gauss, s - s_min - 1, w, h, p);
    CudaImage cur = LevelView(oct.gauss, s - s_min, w, h, p);
    float sd = dsigma0 * std::pow(sigmak, (float)s);
    AdaptiveGaussianBlur(cur, prev, std::max(sd, 0.001f));
  }
  const int gradLo = allGradients ? 0 : 1;                    // see Tier 2.3 note
  const int gradHi = allGradients ? nLevels - 1 : numScales;
  for (int i = gradLo; i <= gradHi; i++) {
    CudaImage lvl = LevelView(oct.gauss, i, w, h, p);
    CudaImage gm = LevelView(oct.gradMag, i, w, h, p);
    CudaImage ga = LevelView(oct.gradAng, i, w, h, p);
    ComputeGradientPlaneFor(lvl, gm, ga);
  }
#if !defined(NDEBUG) || defined(CUSIFT_POISON_SKIPPED_GRADIENTS)
  // Debug builds, or an opt-in Release build used to TEST the invariant:
  //   cmake -B build-poison -DCMAKE_BUILD_TYPE=Release \
  //         -DCMAKE_CUDA_FLAGS=-DCUSIFT_POISON_SKIPPED_GRADIENTS
  // keeps --use_fast_math (so descriptors stay comparable to the gate baseline)
  // while poisoning every skipped plane. If the accuracy gate still passes,
  // nothing reads them.
  for (int i = 0; i < nLevels; i++) {
    if (i >= gradLo && i <= gradHi) continue;
    PoisonGradientLevel(oct.gradMag, i, w, h, p);
    PoisonGradientLevel(oct.gradAng, i, w, h, p);
  }
#endif
}

// Test-only: build octave -1 (VLFeat's real o_min=-1 default, sift.c:1017-1029)
// by 2x-upsampling img01 then blurring with sb=sigman*2^(-o_min)=1.0 (NOT the
// o_min=0 base blur's sb=0.5 -- verified sift.c:1046, sb = sigman*pow(2,-o_min)).
static void BuildUpsampledOctaveMinus1(VLFeatOctaveBuffers &oct, CudaImage &img01, int numScales,
                                       bool allGradients = false)
{
  const int s_min = -1, s_max = numScales + 1;
  const int nLevels = s_max - s_min + 1;
  const float sigmak = std::pow(2.0f, 1.0f / numScales);
  const float sigma0 = 1.6f * sigmak;
  const float dsigma0 = sigma0 * std::sqrt(1.0f - 1.0f / (sigmak * sigmak));

  int w = img01.width * 2, h = img01.height * 2;
  oct.gauss.Allocate(w, h * nLevels, iAlignUp(w, 128), false, NULL);
  oct.gradMag.Allocate(w, h * nLevels, iAlignUp(w, 128), false, NULL);
  oct.gradAng.Allocate(w, h * nLevels, iAlignUp(w, 128), false, NULL);
  int p = oct.gauss.pitch;
  if (oct.gradMag.pitch != p || oct.gradAng.pitch != p) { std::abort(); }
  oct.width = w; oct.height = h; oct.pitch = p; oct.nLevels = nLevels;

  CudaImage img2x;
  img2x.Allocate(w, h, iAlignUp(w, 128), false, NULL);
  Upsample2x(img2x, img01);

  CudaImage lvl0 = LevelView(oct.gauss, 0, w, h, p);
  float sa = sigma0 * std::pow(sigmak, (float)s_min);
  float sb = 1.0f;  // sigman(0.5) * 2^(-o_min) = 0.5*2 = 1.0 for o_min=-1
  if (sa > sb) AdaptiveGaussianBlur(lvl0, img2x, std::sqrt(sa * sa - sb * sb));
  else         AdaptiveGaussianBlur(lvl0, img2x, 0.001f);
  for (int s = s_min + 1; s <= s_max; s++) {
    CudaImage prev = LevelView(oct.gauss, s - s_min - 1, w, h, p);
    CudaImage cur = LevelView(oct.gauss, s - s_min, w, h, p);
    float sd = dsigma0 * std::pow(sigmak, (float)s);
    AdaptiveGaussianBlur(cur, prev, std::max(sd, 0.001f));
  }
  const int gradLo = allGradients ? 0 : 1;                    // see Tier 2.3 note
  const int gradHi = allGradients ? nLevels - 1 : numScales;
  for (int i = gradLo; i <= gradHi; i++) {
    CudaImage lvl = LevelView(oct.gauss, i, w, h, p);
    CudaImage gm = LevelView(oct.gradMag, i, w, h, p);
    CudaImage ga = LevelView(oct.gradAng, i, w, h, p);
    ComputeGradientPlaneFor(lvl, gm, ga);
  }
#if !defined(NDEBUG) || defined(CUSIFT_POISON_SKIPPED_GRADIENTS)
  // Debug builds, or an opt-in Release build used to TEST the invariant:
  //   cmake -B build-poison -DCMAKE_BUILD_TYPE=Release \
  //         -DCMAKE_CUDA_FLAGS=-DCUSIFT_POISON_SKIPPED_GRADIENTS
  // keeps --use_fast_math (so descriptors stay comparable to the gate baseline)
  // while poisoning every skipped plane. If the accuracy gate still passes,
  // nothing reads them.
  for (int i = 0; i < nLevels; i++) {
    if (i >= gradLo && i <= gradHi) continue;
    PoisonGradientLevel(oct.gradMag, i, w, h, p);
    PoisonGradientLevel(oct.gradAng, i, w, h, p);
  }
#endif
}

// Part 2d: detection. DetectedKeypoint (declared in cudaSift.h) mirrors
// VlSiftKeypoint's own fields exactly (o, is = coarse detection level, ix/iy
// = coarse pixel, x/y/sigma = refined absolute-pixel-unit position/scale,
// s = continuous refined level). `is` is carried as-is from the coarse scan
// through refinement (VLFeat never changes it during Newton iteration, only
// x/y can relocate -- verified directly against sift.c:1267-1427) so it's
// available as a first-class field for orientation/descriptor, never
// re-derived from sigma (the harness limitation found earlier this session).

// Tier 1.2: persistent pinned host staging for the per-octave DoG download.
// Grows monotonically to the largest octave's cube (octave -1 of a 1920x1080
// frame: 3840*2160*5 floats = 166 MB) and is then reused for every later octave
// and every later image. thread_local so a multi-threaded COLMAP extractor gets
// one buffer per worker rather than a silently shared one.
static float *PinnedStaging(size_t floats)
{
  static thread_local float *buf = nullptr;
  static thread_local size_t cap = 0;
  if (floats > cap) {
    if (buf) safeCall(cudaFreeHost(buf));
    safeCall(cudaMallocHost((void **)&buf, floats * sizeof(float)));
    cap = floats;
  }
  return buf;
}

// docs/PERFORMANCE.md Tier 1.1, the "two SiftData slabs at max capacity" item: the
// integrated pipeline used to InitSiftData/FreeSiftData once per octave per pass
// (8 allocate/free cycles per image) with a pageable `malloc` host side, so every
// SiftPoint transfer -- 576 B per keypoint, two transfers per pass per octave --
// ran at roughly half PCIe rate on top of the allocation churn. Two grow-only
// slabs (slot 0 = orientation pass, slot 1 = descriptor pass) with PINNED host
// memory, reused across octaves and images. thread_local for the same reason as
// PinnedStaging.
static SiftData &SiftScratch(int slot, int n)
{
  static thread_local SiftData scratch[2] = {};
  SiftData &d = scratch[slot];
  if (n > d.maxPts) {
    if (d.h_data) safeCall(cudaFreeHost(d.h_data));
    if (d.d_data) safeCall(cusiftFree(d.d_data));
    safeCall(cudaMallocHost((void **)&d.h_data, (size_t)n * sizeof(SiftPoint)));
    safeCall(cudaMallocAsync((void **)&d.d_data, (size_t)n * sizeof(SiftPoint), 0));
    d.maxPts = n;
  }
  d.numPts = n;
  return d;
}

// Packed scratch for the VLFeat pipeline's kernel I/O (see VLFeatKp in
// cudaSift.h). Same grow-only, pinned-host, thread_local pattern as SiftScratch,
// which it replaces on the hot path: 20 B/row for keypoints and 512 B/row for
// descriptors, instead of 576 B/row of SiftPoint moved twice per octave.
struct PackedScratch {
  VLFeatKp *h_kps = nullptr, *d_kps = nullptr;
  VLFeatOri *h_ori = nullptr, *d_ori = nullptr;
  float *h_desc = nullptr, *d_desc = nullptr;
  int cap = 0;
};

static PackedScratch &Packed(int n, bool wantDesc)
{
  static thread_local PackedScratch p;
  if (n > p.cap) {
    if (p.h_kps) { safeCall(cudaFreeHost(p.h_kps)); safeCall(cusiftFree(p.d_kps)); }
    if (p.h_ori) { safeCall(cudaFreeHost(p.h_ori)); safeCall(cusiftFree(p.d_ori)); }
    if (p.h_desc) { safeCall(cudaFreeHost(p.h_desc)); safeCall(cusiftFree(p.d_desc)); }
    safeCall(cudaMallocHost((void **)&p.h_kps, (size_t)n * sizeof(VLFeatKp)));
    safeCall(cudaMallocAsync((void **)&p.d_kps, (size_t)n * sizeof(VLFeatKp), 0));
    safeCall(cudaMallocHost((void **)&p.h_ori, (size_t)n * sizeof(VLFeatOri)));
    safeCall(cudaMallocAsync((void **)&p.d_ori, (size_t)n * sizeof(VLFeatOri), 0));
    safeCall(cudaMallocHost((void **)&p.h_desc, (size_t)n * 128 * sizeof(float)));
    safeCall(cudaMallocAsync((void **)&p.d_desc, (size_t)n * 128 * sizeof(float), 0));
    p.cap = n;
  }
  (void)wantDesc;
  return p;
}

// Harness boundary (see VLFeatKp in cudaSift.h): the public SiftData-based
// entry points keep their SiftPoint ABI and convert here. Cost is a 20 B/row
// pack, negligible next to the 576 B/row transfers it replaces.
static void PackKps(const SiftPoint *src, int n, VLFeatKp *dst, bool oriInDegrees)
{
  for (int i = 0; i < n; i++) {
    dst[i].x = src[i].xpos;
    dst[i].y = src[i].ypos;
    dst[i].sigma = src[i].scale;
    dst[i].ori = oriInDegrees ? src[i].orientation * 0.017453292519943295f : 0.0f;
    dst[i].level = (int)(src[i].empty[0] + 0.5f);
  }
}

// Candidates and refined keypoints are appended on the GPU through atomic
// counters, so their order depends on thread scheduling and changes from run
// to run. Sort each octave's keypoints into a fixed order: by integer location
// (level, row, column), the order VLFeat's own raster scan produces, with the
// refined values as a tie-break for the rare candidates that converge to the
// same location. Every later stage processes keypoints in this order, which
// makes the extractor's output order reproducible.
static void SortOctaveKeypoints(std::vector<DetectedKeypoint> &kps, size_t begin)
{
  std::sort(kps.begin() + begin, kps.end(),
            [](const DetectedKeypoint &a, const DetectedKeypoint &b) {
              if (a.is != b.is) return a.is < b.is;
              if (a.iy != b.iy) return a.iy < b.iy;
              if (a.ix != b.ix) return a.ix < b.ix;
              if (a.s != b.s) return a.s < b.s;
              if (a.y != b.y) return a.y < b.y;
              return a.x < b.x;
            });
}

// Full detection pipeline for one octave: DoG cube (GPU) -> coarse extrema
// scan (GPU) -> serial Newton refinement + edge test (CPU, per-candidate,
// matching VLFeat's own CPU-serial design -- see FindDogExtremaKernel's
// comment for why this split, not the refinement, runs on the GPU).
static void DetectOctaveVLFeat(VLFeatOctaveBuffers &oct, int vo,
                               float peakThresh, float edgeThresh, int numScales,
                               std::vector<DetectedKeypoint> &out,
                               const char *debugDumpPrefix = nullptr)
{
  const size_t outBegin = out.size();
  const int s_min = -1, s_max = numScales + 1;
  int w = oct.width, h = oct.height, p = oct.pitch;
  int nDogLevels = oct.nLevels - 1;

  CudaImage dogCombined;
  dogCombined.Allocate(w, h*nDogLevels, iAlignUp(w, 128), false, NULL);
  int dp = dogCombined.pitch;
  {
  StageTimer __t("detect.dog");
  for (int i = 0; i < nDogLevels; i++) {
    CudaImage a = LevelView(oct.gauss, i, w, h, p);
    CudaImage b = LevelView(oct.gauss, i+1, w, h, p);
    CudaImage d = LevelView(dogCombined, i, w, h, dp);
    dim3 threads(16, 16);
    dim3 blocks(iDivUp(w, 16), iDivUp(h, 16));
    ComputeDogPlaneKernel<<<blocks, threads>>>(a.d_data, b.d_data, w, p, h, d.d_data);
    checkMsg("ComputeDogPlaneKernel() execution failed\n");
  }
  }
  // Only the debug path still refines on the CPU (Tier 2.1-B), and only it
  // needs the DoG cube on the host: a TIGHTLY PACKED (pitch==width) copy, since
  // the host instantiation of RefineOneKeypoint indexes a flat
  // width*height*nDogLevels array. Staged through a persistent PINNED buffer
  // (Tier 1.2) rather than a fresh std::vector. In the production path this
  // ~221 MB/image transfer does not happen at all.
  const bool cpuRefine = (debugDumpPrefix != nullptr);
  float *dogHost = nullptr;
  if (cpuRefine) {
    dogHost = PinnedStaging((size_t)w*h*nDogLevels);
    StageTimer __t("detect.dog_d2h");
    for (int i = 0; i < nDogLevels; i++) {
      safeCall(cudaMemcpy2D(dogHost + (size_t)i*w*h, w*sizeof(float),
          dogCombined.d_data + (size_t)i*h*dp, dp*sizeof(float),
          w*sizeof(float), h, cudaMemcpyDeviceToHost));
    }
  }

  const int maxCandidates = 200000;
  int *d_CandX, *d_CandY, *d_CandS; unsigned int *d_CandCount;
  safeCall(cudaMallocAsync(&d_CandX, maxCandidates*sizeof(int), 0));
  safeCall(cudaMallocAsync(&d_CandY, maxCandidates*sizeof(int), 0));
  safeCall(cudaMallocAsync(&d_CandS, maxCandidates*sizeof(int), 0));
  safeCall(cudaMallocAsync(&d_CandCount, sizeof(unsigned int), 0));
  safeCall(cudaMemset(d_CandCount, 0, sizeof(unsigned int)));

  unsigned int candCount = 0;
  std::vector<int> hCandX, hCandY, hCandS;
  {
  StageTimer __t("detect.extrema");
  dim3 threads(16, 16, 1);
  dim3 blocks(iDivUp(w-2, 16), iDivUp(h-2, 16), std::max(nDogLevels-2, 1));
  FindDogExtremaKernel<<<blocks, threads>>>(dogCombined.d_data, w, dp, h, nDogLevels,
      peakThresh, d_CandX, d_CandY, d_CandS, maxCandidates, d_CandCount);
  checkMsg("FindDogExtremaKernel() execution failed\n");

  safeCall(cudaMemcpy(&candCount, d_CandCount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  candCount = std::min(candCount, (unsigned int)maxCandidates);
  if (cpuRefine && candCount > 0) {
    hCandX.resize(candCount); hCandY.resize(candCount); hCandS.resize(candCount);
    safeCall(cudaMemcpy(hCandX.data(), d_CandX, candCount*sizeof(int), cudaMemcpyDeviceToHost));
    safeCall(cudaMemcpy(hCandY.data(), d_CandY, candCount*sizeof(int), cudaMemcpyDeviceToHost));
    safeCall(cudaMemcpy(hCandS.data(), d_CandS, candCount*sizeof(int), cudaMemcpyDeviceToHost));
  }
  }

  FILE *dbgCoarse = nullptr, *dbgRefine = nullptr, *dbgDog = nullptr;
  if (debugDumpPrefix) {
    char path[1024];
    snprintf(path, sizeof(path), "%s_o%d_coarse.txt", debugDumpPrefix, vo);
    dbgCoarse = std::fopen(path, "w");
    std::fprintf(dbgCoarse, "ix iy is\n");
    snprintf(path, sizeof(path), "%s_o%d_refine.txt", debugDumpPrefix, vo);
    dbgRefine = std::fopen(path, "w");
    std::fprintf(dbgRefine, "ix iy is accepted x y sigma s\n");
    snprintf(path, sizeof(path), "%s_o%d_dog.f32", debugDumpPrefix, vo);
    dbgDog = std::fopen(path, "wb");
    std::fwrite(dogHost, sizeof(float), (size_t)w*h*nDogLevels, dbgDog);
    std::fclose(dbgDog);
  }

  const float sigmak = std::pow(2.0f, 1.0f/numScales);
  const float sigma0 = 1.6f*sigmak;

  if (!cpuRefine) {
    // Tier 2.1-B production path: refine every candidate on the GPU, straight
    // out of the pitched DoG buffer, and bring back only the survivors.
    std::vector<DetectedKeypoint> accepted;
    if (candCount > 0) {
      StageTimer __t("detect.refine_gpu");
      DetectedKeypoint *d_Kp = nullptr; unsigned int *d_KpCount = nullptr;
      safeCall(cudaMallocAsync(&d_Kp, (size_t)candCount*sizeof(DetectedKeypoint), 0));
      safeCall(cudaMallocAsync(&d_KpCount, sizeof(unsigned int), 0));
      safeCall(cudaMemset(d_KpCount, 0, sizeof(unsigned int)));
      int threadsR = 128, blocksR = iDivUp((int)candCount, threadsR);
      RefineKeypointsKernel<<<blocksR, threadsR>>>(dogCombined.d_data, w, h, dp, h*dp,
          nDogLevels, s_min, s_max, d_CandX, d_CandY, d_CandS, candCount,
          peakThresh, edgeThresh, d_Kp, d_KpCount);
      checkMsg("RefineKeypointsKernel() execution failed\n");
      unsigned int nAcc = 0;
      safeCall(cudaMemcpy(&nAcc, d_KpCount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
      accepted.resize(nAcc);
      if (nAcc)
        safeCall(cudaMemcpy(accepted.data(), d_Kp, (size_t)nAcc*sizeof(DetectedKeypoint),
                            cudaMemcpyDeviceToHost));
      safeCall(cusiftFree(d_Kp)); safeCall(cusiftFree(d_KpCount));
    }
    safeCall(cusiftFree(d_CandX)); safeCall(cusiftFree(d_CandY));
    safeCall(cusiftFree(d_CandS)); safeCall(cusiftFree(d_CandCount));
    out.reserve(out.size() + accepted.size());
    for (DetectedKeypoint kp : accepted) {
      kp.o = vo;
      kp.sigma = sigma0 * std::pow(sigmak, kp.s);
      out.push_back(kp);
    }
    SortOctaveKeypoints(out, outBegin);
    return;
  }

  safeCall(cusiftFree(d_CandX)); safeCall(cusiftFree(d_CandY));
  safeCall(cusiftFree(d_CandS)); safeCall(cusiftFree(d_CandCount));
#ifdef CUSIFT_TIME_REFINE
  auto __t0 = std::chrono::steady_clock::now();
#endif
  StageTimer __tr("detect.refine_cpu");
  for (unsigned int c = 0; c < candCount; c++) {
    if (dbgCoarse) std::fprintf(dbgCoarse, "%d %d %d\n", hCandX[c], hCandY[c], hCandS[c]);
    DetectedKeypoint kp;
    bool accepted = RefineOneKeypoint(dogHost, w, h, w, w*h, nDogLevels, s_min, s_max,
                                      hCandX[c], hCandY[c], hCandS[c],
                                      peakThresh, edgeThresh, kp);
    if (dbgRefine) {
      if (accepted) {
        std::fprintf(dbgRefine, "%d %d %d 1 %.6f %.6f %.6f %.6f\n",
                     hCandX[c], hCandY[c], hCandS[c], kp.x, kp.y,
                     sigma0*std::pow(sigmak, kp.s), kp.s);
      } else {
        std::fprintf(dbgRefine, "%d %d %d 0 0 0 0 0\n", hCandX[c], hCandY[c], hCandS[c]);
      }
    }
    if (!accepted) continue;
    kp.o = vo;
    // x, y, sigma stay in this octave's own LOCAL (native) pixel units here
    // -- xper scaling to absolute image coordinates is the CALLER's job
    // (DetectKeypointsVLFeat for standalone use; the integrated
    // detect+orient+descriptor pipeline needs local units directly, since
    // that's what the gradient buffers/kernels are indexed in).
    kp.sigma = sigma0 * std::pow(sigmak, kp.s);
    out.push_back(kp);
  }
#ifdef CUSIFT_TIME_REFINE
  double __ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - __t0).count();
  std::fprintf(stderr, "octave %d: %u coarse candidates, refine = %.3f ms\n", vo, candCount, __ms);
#endif
  SortOctaveKeypoints(out, outBegin);
  if (dbgCoarse) std::fclose(dbgCoarse);
  if (dbgRefine) std::fclose(dbgRefine);
}

// Full multi-octave VLFeat-faithful detection. o_min=0 (matches this
// session's harness convention throughout -- the o_min=-1 hypothesis was
// tested directly and found not to explain the residual Gate 1/2 gap, see
// plan). Deduplicates nothing -- exact-duplicate rows (the ~0.06%
// Newton-convergence artifact found earlier this session) are expected and
// left as-is, matching real ref_cpu's own behavior.
void DetectKeypointsVLFeat(CudaImage &img, int numOctaves, float peakThresh, float edgeThresh,
                           std::vector<DetectedKeypoint> &out, const char *debugDumpPrefix)
{
  const int numScales = NUM_SCALES;
  CudaImage img01;
  img01.Allocate(img.width, img.height, iAlignUp(img.width, 128), false, NULL);
  {
    StageTimer __t("img_scale");
    dim3 threads(16, 16);
    dim3 blocks(iDivUp(img.width, 16), iDivUp(img.height, 16));
    ScaleImageKernel<<<blocks, threads>>>(img01.d_data, img.d_data,
        img.width, img01.pitch, img.pitch, img.height, 1.0f/255.0f);
    checkMsg("ScaleImageKernel() execution failed\n");
  }

  out.clear();
  std::vector<VLFeatOctaveBuffers> octaves(numOctaves);
  CudaImage prevTopLevel;
  for (int vo = 0; vo < numOctaves; vo++) {
    if (vo == 0) {
      BuildVLFeatOctave(octaves[vo], nullptr, &img01, numScales);
    } else {
      VLFeatOctaveBuffers &prev = octaves[vo-1];
      prevTopLevel = LevelView(prev.gauss, numScales, prev.width, prev.height, prev.pitch);
      BuildVLFeatOctave(octaves[vo], &prevTopLevel, nullptr, numScales);
    }
    float xper = std::pow(2.0f, (float)vo);
    size_t before = out.size();
    DetectOctaveVLFeat(octaves[vo], vo, peakThresh, edgeThresh, numScales, out,
                       debugDumpPrefix);
    // DetectOctaveVLFeat returns LOCAL (per-octave) coordinates; scale this
    // octave's newly-added entries to absolute image coordinates here, at
    // the standalone-detection API boundary (the integrated pipeline below
    // scales at ITS OWN output boundary instead, using local coords for the
    // orientation/descriptor kernels in between).
    for (size_t i = before; i < out.size(); i++) {
      out[i].x *= xper; out[i].y *= xper; out[i].sigma *= xper;
    }
  }
}

// Part 2d integration: full detect -> orient pipeline. `is`/x/y/sigma are
// carried natively from detection straight into orientation -- each
// octave's keypoints go directly from RefineOneKeypoint's own LOCAL
// coordinates into ComputeOrientationsVLFeat using that SAME octave's own
// gradient buffers, still resident from the pyramid build. There is no
// AssignOctaveLevel sigma->level inversion anywhere in this path -- that
// inversion was a property of the external-candidate-injection harness
// (Parts 2a-2c), not of cusift's own kernels, and is structurally absent
// here since detection and orientation share the same in-memory state.
// outAngles in RADIANS (VLFeat convention), outKps.x/y/sigma in ABSOLUTE
// image coordinates (VLFeat's own frame, no +0.5 -- that convention is
// applied only at the file/DB output boundary, same rule as detection).
void DetectAndOrientVLFeat(CudaImage &img, int numOctaves, float peakThresh, float edgeThresh,
                          std::vector<DetectedKeypoint> &outKps,
                          std::vector<std::array<float, 4>> &outAngles,
                          std::vector<int> &outCounts)
{
  const int numScales = NUM_SCALES;
  CudaImage img01;
  img01.Allocate(img.width, img.height, iAlignUp(img.width, 128), false, NULL);
  {
    dim3 threads(16, 16);
    dim3 blocks(iDivUp(img.width, 16), iDivUp(img.height, 16));
    ScaleImageKernel<<<blocks, threads>>>(img01.d_data, img.d_data,
        img.width, img01.pitch, img.pitch, img.height, 1.0f/255.0f);
    checkMsg("ScaleImageKernel() execution failed\n");
  }

  outKps.clear(); outAngles.clear(); outCounts.clear();
  std::vector<VLFeatOctaveBuffers> octaves(numOctaves);
  CudaImage prevTopLevel;
  for (int vo = 0; vo < numOctaves; vo++) {
    if (vo == 0) {
      BuildVLFeatOctave(octaves[vo], nullptr, &img01, numScales);
    } else {
      VLFeatOctaveBuffers &prev = octaves[vo-1];
      prevTopLevel = LevelView(prev.gauss, numScales, prev.width, prev.height, prev.pitch);
      BuildVLFeatOctave(octaves[vo], &prevTopLevel, nullptr, numScales);
    }
    float xper = std::pow(2.0f, (float)vo);
    VLFeatOctaveBuffers &oct = octaves[vo];

    std::vector<DetectedKeypoint> localKps;  // still in LOCAL (per-octave) coords
    DetectOctaveVLFeat(oct, vo, peakThresh, edgeThresh, numScales, localKps, nullptr);
    int n = (int)localKps.size();
    if (n == 0) continue;

    SiftData siftData;
    InitSiftData(siftData, n, true, true);
    siftData.numPts = n;
    for (int i = 0; i < n; i++) {
      SiftPoint &p = siftData.h_data[i];
      memset(&p, 0, sizeof(SiftPoint));
      p.xpos = localKps[i].x;
      p.ypos = localKps[i].y;
      p.scale = localKps[i].sigma;
      // ComputeOrientationsVLFeat indexes its gradient buffer with `level`
      // as an ARRAY INDEX (offset by -s_min), not the raw VLFeat level
      // value -- matches AssignOctaveLevel's own `levelIdx = is - s_min`
      // convention used by the external-injection path. localKps[i].is is
      // the true level value (range [0, numScales-1]); s_min is always -1.
      p.empty[0] = (float)(localKps[i].is - (-1));
    }
    PackedScratch &pk = Packed(n, false);
    PackKps(siftData.h_data, n, pk.h_kps, false);
    safeCall(cudaMemcpy(pk.d_kps, pk.h_kps, sizeof(VLFeatKp)*n, cudaMemcpyHostToDevice));

    int levelStride = oct.height * oct.pitch;
    dim3 blocks(512), threads(256);
    ComputeOrientationsVLFeat<<<blocks, threads>>>(
        oct.gradMag.d_data, oct.gradAng.d_data,
        oct.width, oct.pitch, oct.height, levelStride,
        pk.d_kps, pk.d_ori, n);
    checkMsg("ComputeOrientationsVLFeat() execution failed\n");

    safeCall(cudaMemcpy(pk.h_ori, pk.d_ori, sizeof(VLFeatOri)*n, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
      DetectedKeypoint kp = localKps[i];
      kp.o = vo;
      kp.x *= xper; kp.y *= xper; kp.sigma *= xper;
      outKps.push_back(kp);
      const VLFeatOri &p = pk.h_ori[i];
      outAngles.push_back({p.angles[0], p.angles[1], p.angles[2], p.angles[3]});
      outCounts.push_back(p.count);
    }
    FreeSiftData(siftData);
  }
}

// Full Part 2d/2b'/2c integration: detect -> orient -> descriptor, one pass
// per octave, is/x/y/sigma threaded natively throughout (same rationale as
// DetectAndOrientVLFeat). Each keypoint is expanded into up to
// `maxNumOrientations` rows (COLMAP default 2, matching this session's
// established `max_num_orientations` finding on real multi-orientation
// rate), one per surviving angle -- mirrors COLMAP's own
// `SiftCPUFeatureExtractor::Extract` orientation-expansion loop
// (sift.cc:210-286), not something invented here. Output x/y/sigma are
// ABSOLUTE image coordinates, orientation in RADIANS (VLFeat convention),
// descriptor already UBC-ordered + L2/clamp/L2-normalized (same kernel as
// the already-gate-verified Part 2b' descriptor path). No +0.5 applied --
// that's the output/DB-boundary's job, same rule as detection.
void ExtractSiftFullVLFeat(CudaImage &img, int numOctaves, float peakThresh, float edgeThresh,
                           int maxNumOrientations, int firstOctave, int maxNumFeatures,
                           std::vector<float> &outX, std::vector<float> &outY,
                           std::vector<float> &outSigma, std::vector<float> &outOrientation,
                           std::vector<std::array<float, 128>> &outDescriptors)
{
  const int numScales = NUM_SCALES;
  // firstOctave (VLFeat's o_min) must be 0 or -1. COLMAP's shipping default
  // is -1 (SiftExtractionOptions::first_octave, sift.h:46), i.e. the image is
  // 2x-upsampled and an extra finer octave -1 is scanned -- which roughly
  // DOUBLES the keypoint count vs o_min=0 (measured: 10517 vs 5764 rows on
  // one 1920x1080 GoPro frame). Supporting it is therefore required for
  // count/coverage parity with a default-configured COLMAP, not optional.
  if (firstOctave != 0 && firstOctave != -1) {
    std::fprintf(stderr, "ExtractSiftFullVLFeat: firstOctave must be 0 or -1 (got %d)\n", firstOctave);
    std::abort();
  }
  CudaImage img01;
  img01.Allocate(img.width, img.height, iAlignUp(img.width, 128), false, NULL);
  {
    dim3 threads(16, 16);
    dim3 blocks(iDivUp(img.width, 16), iDivUp(img.height, 16));
    ScaleImageKernel<<<blocks, threads>>>(img01.d_data, img.d_data,
        img.width, img01.pitch, img.pitch, img.height, 1.0f/255.0f);
    checkMsg("ScaleImageKernel() execution failed\n");
  }

  outX.clear(); outY.clear(); outSigma.clear(); outOrientation.clear(); outDescriptors.clear();
  std::vector<VLFeatOctaveBuffers> octaves(numOctaves);
  // Per-octave staging: detection+orientation runs for every octave first,
  // then max_num_features decides which whole DOG levels survive, and only
  // then are descriptors computed (the octave gradient buffers stay alive in
  // `octaves` for the whole function, so the second pass can still use them).
  std::vector<std::vector<int>> levelKpCount(numOctaves, std::vector<int>(numScales, 0));
  std::vector<std::vector<char>> keepLevel(numOctaves, std::vector<char>(numScales, 1));
  std::vector<std::vector<VLFeatKp>> pendingRows(numOctaves);
  std::vector<std::vector<int>> pendingLevel(numOctaves);
  std::vector<float> pendingXper(numOctaves, 1.0f);
  CudaImage prevTopLevel;
  for (int oi = 0; oi < numOctaves; oi++) {
    int vo = firstOctave + oi;   // TRUE VLFeat octave index (may be -1)
    {
      StageTimer __t("pyramid");
      if (oi == 0) {
        if (vo == -1) BuildUpsampledOctaveMinus1(octaves[oi], img01, numScales);
        else          BuildVLFeatOctave(octaves[oi], nullptr, &img01, numScales);
      } else {
        VLFeatOctaveBuffers &prev = octaves[oi-1];
        prevTopLevel = LevelView(prev.gauss, numScales, prev.width, prev.height, prev.pitch);
        BuildVLFeatOctave(octaves[oi], &prevTopLevel, nullptr, numScales);
      }
    }
    // xper = 2^o, so octave -1 (upsampled) maps back with 0.5 -- positions
    // and sigmas in the upsampled frame are HALVED to original image units.
    float xper = std::pow(2.0f, (float)vo);
    VLFeatOctaveBuffers &oct = octaves[oi];

    std::vector<DetectedKeypoint> localKps;
    DetectOctaveVLFeat(oct, vo, peakThresh, edgeThresh, numScales, localKps, nullptr);
    int n = (int)localKps.size();
    if (n == 0) continue;

    // Orientation pass. Packed I/O: 20 B/row in, 20 B/row out (was 576 B both
    // ways as SiftPoint), and no d_PointCounter/d_MaxNumPoints symbol writes --
    // the kernel takes an explicit count now.
    PackedScratch &pk = Packed(n, false);
    StageTimer *__ts = new StageTimer("ori_setup");
    for (int i = 0; i < n; i++) {
      pk.h_kps[i].x = localKps[i].x;
      pk.h_kps[i].y = localKps[i].y;
      pk.h_kps[i].sigma = localKps[i].sigma;
      pk.h_kps[i].ori = 0.0f;
      pk.h_kps[i].level = localKps[i].is - (-1);
    }
    safeCall(cudaMemcpy(pk.d_kps, pk.h_kps, sizeof(VLFeatKp)*n, cudaMemcpyHostToDevice));
    delete __ts;

    int levelStride = oct.height * oct.pitch;
    {
      StageTimer __t("orient");
      dim3 blocks(512), threads(256);
      ComputeOrientationsVLFeat<<<blocks, threads>>>(
          oct.gradMag.d_data, oct.gradAng.d_data,
          oct.width, oct.pitch, oct.height, levelStride,
          pk.d_kps, pk.d_ori, n);
      checkMsg("ComputeOrientationsVLFeat() execution failed\n");
    }
    {
      StageTimer __t("ori_d2h");
      safeCall(cudaMemcpy(pk.h_ori, pk.d_ori, sizeof(VLFeatOri)*n, cudaMemcpyDeviceToHost));
    }

    StageTimer __tx("expand_cpu");
    // Expand: one descriptor-kernel input row per surviving angle, capped at
    // maxNumOrientations, matching COLMAP's own per-DOG-level expansion.
    // Rows are VLFeatKp (20 B) reserved up front -- this loop used to build and
    // copy 576 B SiftPoints and was ~22% of extraction time on its own.
    // Angles stay in RADIANS end to end now; the deg round-trip existed only
    // because SiftPoint.orientation was the kernel's degree-convention field.
    std::vector<VLFeatKp> &expanded = pendingRows[oi];
    std::vector<int> &expandedLevel = pendingLevel[oi];
    expanded.reserve((size_t)n * maxNumOrientations);
    expandedLevel.reserve((size_t)n * maxNumOrientations);
    for (int i = 0; i < n; i++) {
      const VLFeatOri &op = pk.h_ori[i];
      int count = std::min(op.count, maxNumOrientations);
      for (int a = 0; a < count; a++) {
        VLFeatKp p;
        p.x = localKps[i].x;
        p.y = localKps[i].y;
        p.sigma = localKps[i].sigma;
        p.ori = op.angles[a];
        p.level = localKps[i].is - (-1);
        expanded.push_back(p);
        expandedLevel.push_back(localKps[i].is);
      }
    }
    // Per-(octave,level) pre-orientation keypoint counts, for COLMAP's
    // max_num_features enforcement applied after all octaves (below).
    for (int i = 0; i < n; i++) levelKpCount[oi][localKps[i].is]++;
    pendingXper[oi] = xper;
  }

  // COLMAP's max_num_features enforcement is LEVEL-GRANULAR, not per-keypoint:
  // it walks DOG levels from the coarsest backwards, accumulating keypoint
  // counts, and as soon as the running total exceeds max_num_features it drops
  // every FINER level entirely (sift.cc:292-315, `first_level_to_keep`). It is
  // NOT a "top-N strongest/largest" selection -- reproducing it as one would
  // give a different keypoint set. Levels are ordered ascending-sigma by
  // (octave, level), which is exactly the flat order below since
  // sigma = sigma0*2^(s/S)*2^o is monotonic in that ordering.
  int firstKeepFlat = 0;
  {
    std::vector<std::pair<int,int>> flat;   // (octaveIdx, level), ascending sigma
    for (int oi = 0; oi < numOctaves; oi++)
      for (int lev = 0; lev < numScales; lev++) flat.push_back({oi, lev});
    long long acc = 0;
    for (int k = (int)flat.size() - 1; k >= 0; k--) {
      acc += levelKpCount[flat[k].first][flat[k].second];
      if (acc > maxNumFeatures) { firstKeepFlat = k; break; }
    }
    for (int k = 0; k < (int)flat.size(); k++)
      keepLevel[flat[k].first][flat[k].second] = (k >= firstKeepFlat) ? 1 : 0;
  }

  for (int oi = 0; oi < numOctaves; oi++) {
    if (pendingRows[oi].empty()) continue;
    VLFeatOctaveBuffers &oct = octaves[oi];
    float xper = pendingXper[oi];
    int levelStride = oct.height * oct.pitch;
    // Count first, then copy the surviving rows straight into the descriptor
    // input buffer -- an intermediate `expanded` vector here was a second full
    // copy of the same data.
    int m = 0;
    for (size_t i = 0; i < pendingRows[oi].size(); i++)
      if (keepLevel[oi][pendingLevel[oi][i]]) m++;
    if (m == 0) continue;

    StageTimer *__ts = new StageTimer("desc_setup");
    PackedScratch &pk = Packed(m, true);
    for (size_t i = 0, k = 0; i < pendingRows[oi].size(); i++)
      if (keepLevel[oi][pendingLevel[oi][i]]) pk.h_kps[k++] = pendingRows[oi][i];
    safeCall(cudaMemcpy(pk.d_kps, pk.h_kps, sizeof(VLFeatKp)*m, cudaMemcpyHostToDevice));
    delete __ts;

    {
      StageTimer __t("descriptor");
      // ExtractSiftDescriptorsVLFeat indexes threads via a FLAT 1D
      // threadIdx.x (tid/nthreads=blockDim.x) -- a 2D block here would give
      // 8 redundant physical threads per distinct tid, each atomicAdd-ing
      // the same contribution into the same descriptor bin (silently
      // inflating every histogram entry ~8x). Must match the verified-
      // working call elsewhere in this file (dim3 threads(256), 1D).
      dim3 blocks(512), threads(256);
      ExtractSiftDescriptorsVLFeat<<<blocks, threads>>>(
          oct.gradMag.d_data, oct.gradAng.d_data,
          oct.width, oct.pitch, oct.height, levelStride,
          pk.d_kps, pk.d_desc, m, xper, 3.0f /*magnif*/);
      checkMsg("ExtractSiftDescriptorsVLFeat() execution failed\n");
    }
    {
      StageTimer __t("desc_d2h");
      safeCall(cudaMemcpy(pk.h_kps, pk.d_kps, sizeof(VLFeatKp)*m, cudaMemcpyDeviceToHost));
      safeCall(cudaMemcpy(pk.h_desc, pk.d_desc, sizeof(float)*128*m, cudaMemcpyDeviceToHost));
    }
    StageTimer __to("output_copy");
    for (int i = 0; i < m; i++) {
      outX.push_back(pk.h_kps[i].x);
      outY.push_back(pk.h_kps[i].y);
      outSigma.push_back(pk.h_kps[i].sigma);
      outOrientation.push_back(pk.h_kps[i].ori);   // radians throughout now
      std::array<float, 128> desc;
      std::copy(pk.h_desc + (size_t)i*128, pk.h_desc + (size_t)i*128 + 128, desc.begin());
      outDescriptors.push_back(desc);
    }
  }
  StageTimer::Report("ExtractSiftFullVLFeat");
}

// Debug-only: build the VLFeat pyramid up through octave `dumpOctave` and
// write its Gaussian levels + gradient buffers to raw float32 files,
// matching vlfeat_dump.c's layout, for direct ground-truth comparison
// (diagnosing whether cusift's own CUDA-computed pyramid matches VLFeat's
// real internal state as closely as the from-scratch Python reconstruction
// does -- these are NOT guaranteed identical just because both were checked
// against ground truth separately).
void DumpPyramidDebug(CudaImage &img, int numOctaves, int dumpOctave, const char *prefix)
{
  const int numScales = NUM_SCALES;
  CudaImage img01;
  img01.Allocate(img.width, img.height, iAlignUp(img.width, 128), false, NULL);
  {
    dim3 threads(16, 16);
    dim3 blocks(iDivUp(img.width, 16), iDivUp(img.height, 16));
    ScaleImageKernel<<<blocks, threads>>>(img01.d_data, img.d_data,
        img.width, img01.pitch, img.pitch, img.height, 1.0f / 255.0f);
    checkMsg("ScaleImageKernel() execution failed\n");
  }

  std::vector<VLFeatOctaveBuffers> octaves(dumpOctave + 1);
  CudaImage prevTopLevel;
  for (int vo = 0; vo <= dumpOctave; vo++) {
    // allGradients=true: this dump is compared plane-for-plane against
    // vlfeat_dump's ground truth, including the levels no keypoint can use.
    if (vo == 0) {
      BuildVLFeatOctave(octaves[vo], nullptr, &img01, numScales, true);
    } else {
      VLFeatOctaveBuffers &prev = octaves[vo - 1];
      prevTopLevel = LevelView(prev.gauss, numScales, prev.width, prev.height, prev.pitch);
      BuildVLFeatOctave(octaves[vo], &prevTopLevel, nullptr, numScales, true);
    }
  }

  VLFeatOctaveBuffers &oct = octaves[dumpOctave];
  const int s_min = -1;
  int W = oct.width, H = oct.height;
  std::vector<float> hostBuf((size_t)W * H);
  std::vector<float> hostBufGrad((size_t)W * H * 2);
  for (int i = 0; i < oct.nLevels; i++) {
    int s = s_min + i;
    CudaImage lvl = LevelView(oct.gauss, i, W, H, oct.pitch);
    // pitch may exceed W; copy row-by-row to a tightly-packed W*H buffer.
    safeCall(cudaMemcpy2D(hostBuf.data(), W * sizeof(float), lvl.d_data,
        oct.pitch * sizeof(float), W * sizeof(float), H, cudaMemcpyDeviceToHost));
    char path[1024];
    snprintf(path, sizeof(path), "%s_o%d_s%d_gauss.f32", prefix, dumpOctave, s);
    FILE *f = fopen(path, "wb");
    fwrite(hostBuf.data(), sizeof(float), (size_t)W * H, f);
    fclose(f);
  }
  for (int i = 0; i < oct.nLevels; i++) {
    int s = s_min + i;
    CudaImage gm = LevelView(oct.gradMag, i, W, H, oct.pitch);
    CudaImage ga = LevelView(oct.gradAng, i, W, H, oct.pitch);
    std::vector<float> mag((size_t)W * H), ang((size_t)W * H);
    safeCall(cudaMemcpy2D(mag.data(), W * sizeof(float), gm.d_data,
        oct.pitch * sizeof(float), W * sizeof(float), H, cudaMemcpyDeviceToHost));
    safeCall(cudaMemcpy2D(ang.data(), W * sizeof(float), ga.d_data,
        oct.pitch * sizeof(float), W * sizeof(float), H, cudaMemcpyDeviceToHost));
    for (size_t k = 0; k < (size_t)W * H; k++) {
      hostBufGrad[2*k] = mag[k];
      hostBufGrad[2*k+1] = ang[k];
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s_o%d_s%d_grad.f32", prefix, dumpOctave, s);
    FILE *f = fopen(path, "wb");
    fwrite(hostBufGrad.data(), sizeof(float), (size_t)W * H * 2, f);
    fclose(f);
  }
  fprintf(stderr, "DumpPyramidDebug: octave %d (W=%d H=%d) written to %s_o%d_s*.f32\n",
          dumpOctave, W, H, prefix, dumpOctave);
}

// Exact replica of vl_sift_keypoint_init's sigma -> (octave, level) inversion
// (sift.c:2161-2190). absSigma is in absolute (full original image) pixel
// units, matching what's stored in a VLFeat-derived .sift file's scale field.
// Returns vlfeatOctave (0 = full resolution, increasing = coarser) and
// levelIdx (slice index into a VLFeatOctaveBuffers' combined buffers, i.e.
// already offset by -s_min so it's directly usable as an array index).
static void AssignOctaveLevel(float absSigma, int numOctaves, int numScales,
                              int &vlfeatOctave, int &levelIdx)
{
  const int s_min = -1, s_max = numScales + 1;
  const float sigmak = std::pow(2.0f, 1.0f / numScales);
  const float sigma0 = 1.6f * sigmak;
  double phi = std::log2(((double)absSigma + 1e-10) / sigma0);
  int o = (int)std::floor(phi - ((double)s_min + 0.5) / numScales);
  o = std::min(o, numOctaves - 1);
  o = std::max(o, 0);
  double s = numScales * (phi - o);
  int is = (int)(s + 0.5);
  is = std::min(is, s_max - 2);
  is = std::max(is, s_min + 1);
  vlfeatOctave = o;
  levelIdx = is - s_min;
}

// Runs ONLY descriptor extraction on externally supplied keypoints, skipping
// detection (DoG/extrema-finding) and orientation (both still TODO in Parts
// 2c/2d) entirely -- isolates descriptor-kernel correctness so it can be
// validated independently. Non-upsampled operating point only.
//
// Input:  siftData.h_data[0..numPts) must have xpos/ypos/scale = ABSOLUTE
//         image-pixel values (scale = VLFeat sigma convention), orientation
//         = DEGREES (SiftPoint convention).
// Output: same array, same order; data[128] filled with the descriptor.
//         xpos/ypos/scale round-trip to their input values.
void ExtractSiftDescriptorsOnly(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur)
{
  (void)initBlur;  // superseded by VLFeat's own sigman=0.5 base-blur handling
  int n = siftData.numPts;
  const int numScales = NUM_SCALES;

  // 1. Normalize [0,255] input to [0,1] -- COLMAP does the same before
  //    calling VLFeat (sift.cc:186-188); VLFeat's sigma0/peak-threshold
  //    constants assume this range.
  CudaImage img01;
  img01.Allocate(img.width, img.height, iAlignUp(img.width, 128), false, NULL);
  {
    dim3 threads(16, 16);
    dim3 blocks(iDivUp(img.width, 16), iDivUp(img.height, 16));
    ScaleImageKernel<<<blocks, threads>>>(img01.d_data, img.d_data,
        img.width, img01.pitch, img.pitch, img.height, 1.0f / 255.0f);
    checkMsg("ScaleImageKernel() execution failed\n");
  }

  // 2. Assign each point to a VLFeat octave + level via the exact inversion
  //    formula (not a guessed/derived one -- an off-by-one here silently
  //    reads gradients from the wrong pyramid level).
  std::vector<int> vlOctaveOf(n), levelIdxOf(n);
  std::vector<int> cusiftOctaveOf(n);
  std::vector<float> ssOf(n);
  for (int i = 0; i < n; i++) {
    float absScale = std::max(siftData.h_data[i].scale, 1e-6f);
    // COLMAP delta: AssignOctaveLevel CLAMPS the octave to >= 0. That was safe
    // under this function's original contract (its caller generated the
    // keypoints), but externally supplied keypoints can legitimately belong to
    // VLFeat octave -1, and clamping them describes them from the wrong pyramid
    // level -- a silently wrong descriptor. This function is documented
    // "non-upsampled operating point only", so make that contract loud at the
    // boundary rather than trusting every caller to pre-filter.
    {
      const float sigmak = std::pow(2.0f, 1.0f / numScales);
      const float sigma0 = 1.6f * sigmak;
      const double phi = std::log2(((double)absScale + 1e-10) / sigma0);
      const int trueOctave = (int)std::floor(phi - (-1.0 + 0.5) / numScales);
      if (trueOctave < 0) {
        std::fprintf(stderr,
                     "ExtractSiftDescriptorsOnly: keypoint %d (sigma %.6f) maps "
                     "to VLFeat octave %d; this path builds octaves 0..%d only "
                     "and must not clamp. Filter octave<0 keypoints first.\n",
                     i, absScale, trueOctave, numOctaves - 1);
        std::abort();
      }
    }
    int vo, li;
    AssignOctaveLevel(absScale, numOctaves, numScales, vo, li);
    vlOctaveOf[i] = vo;
    levelIdxOf[i] = li;
    cusiftOctaveOf[i] = numOctaves - vo;   // cusift's d_PointCounter convention
    ssOf[i] = std::pow(2.0f, (float)vo);   // VLFeat's xper = 2^o
  }

  // 3. Stable sort by ascending cusift-octave (coarsest first, matching the
  //    contiguous per-octave layout FindPointsMulti would have produced).
  std::vector<int> order(n);
  for (int i = 0; i < n; i++) order[i] = i;
  std::stable_sort(order.begin(), order.end(),
                    [&](int a, int b) { return cusiftOctaveOf[a] < cusiftOctaveOf[b]; });

  std::vector<SiftPoint> reordered(n);
  std::vector<int> srcOfPos(n);
  std::vector<int> perOctave(numOctaves + 1, 0);
  for (int i = 0; i < n; i++) {
    int src = order[i];
    SiftPoint p = siftData.h_data[src];
    float ss = ssOf[src];
    p.xpos /= ss;
    p.ypos /= ss;
    p.scale /= ss;
    p.subsampling = ss;
    p.empty[0] = (float)levelIdxOf[src];  // repurposed to carry the assigned level
    reordered[i] = p;
    srcOfPos[i] = src;
    perOctave[cusiftOctaveOf[src]]++;
  }
  std::copy(reordered.begin(), reordered.end(), siftData.h_data);

  // Boundary pack: the harness ABI stays SiftPoint, the kernel takes VLFeatKp.
  // Orientation arrives here in DEGREES (SiftPoint convention) and is converted.
  // The per-octave slices that used to be communicated through d_PointCounter
  // are now explicit (pointer + count) per launch.
  PackedScratch &pk = Packed(n, true);
  PackKps(siftData.h_data, n, pk.h_kps, true);
  safeCall(cudaMemcpy(pk.d_kps, pk.h_kps, sizeof(VLFeatKp) * n, cudaMemcpyHostToDevice));
  std::vector<int> sliceStart(numOctaves + 1, 0), sliceCount(numOctaves + 1, 0);
  {
    int cum = 0;
    for (int o = 1; o <= numOctaves; o++) {
      sliceStart[o] = cum;
      sliceCount[o] = perOctave[o];
      cum += perOctave[o];
    }
  }

  // 4. Build the VLFeat pyramid octave by octave (each depends on the
  //    previous), then run the VLFeat-parity descriptor kernel on each
  //    octave's point range against that octave's gradient buffers.
  std::vector<VLFeatOctaveBuffers> octaves(numOctaves);
  CudaImage prevTopLevel;  // must outlive the BuildVLFeatOctave call using it
  for (int vo = 0; vo < numOctaves; vo++) {
    if (vo == 0) {
      BuildVLFeatOctave(octaves[vo], nullptr, &img01, numScales);
    } else {
      VLFeatOctaveBuffers &prev = octaves[vo - 1];
      prevTopLevel = LevelView(prev.gauss, numScales, prev.width, prev.height, prev.pitch);
      BuildVLFeatOctave(octaves[vo], &prevTopLevel, nullptr, numScales);
    }
  }

  dim3 blocks(512), threads(256);
  for (int vo = 0; vo < numOctaves; vo++) {
    int cusiftOctave = numOctaves - vo;
    float ss = std::pow(2.0f, (float)vo);
    int levelStride = octaves[vo].height * octaves[vo].pitch;
    int first = sliceStart[cusiftOctave], cnt = sliceCount[cusiftOctave];
    if (cnt <= 0) continue;
    ExtractSiftDescriptorsVLFeat<<<blocks, threads>>>(
        octaves[vo].gradMag.d_data, octaves[vo].gradAng.d_data,
        octaves[vo].width, octaves[vo].pitch, octaves[vo].height, levelStride,
        pk.d_kps + first, pk.d_desc + (size_t)first * 128, cnt, ss, 3.0f /*magnif*/);
    checkMsg("ExtractSiftDescriptorsVLFeat() execution failed\n");
  }

  // 5. Read back (unpacking into the SiftPoint ABI) and undo the sort so output
  //    order matches input order.
  safeCall(cudaMemcpy(pk.h_kps, pk.d_kps, sizeof(VLFeatKp) * n, cudaMemcpyDeviceToHost));
  safeCall(cudaMemcpy(pk.h_desc, pk.d_desc, sizeof(float) * 128 * n, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; i++) {
    siftData.h_data[i].xpos = pk.h_kps[i].x;
    siftData.h_data[i].ypos = pk.h_kps[i].y;
    siftData.h_data[i].scale = pk.h_kps[i].sigma;
    std::copy(pk.h_desc + (size_t)i * 128, pk.h_desc + (size_t)i * 128 + 128,
              siftData.h_data[i].data);
  }
  std::vector<SiftPoint> restored(n);
  for (int i = 0; i < n; i++)
    restored[srcOfPos[i]] = siftData.h_data[i];
  std::copy(restored.begin(), restored.end(), siftData.h_data);
}

// Part 2c: computes VLFeat-parity orientation(s) for externally supplied
// (x,y,scale)-only candidates -- mirrors ExtractSiftDescriptorsOnly's pyramid
// setup exactly (kept as a separate function rather than refactored to share
// code, so as not to risk regressing the already-gate-verified descriptor
// path). Up to 4 angles per candidate (VLFeat's own cap); outAngles/outCount
// indexed by ORIGINAL candidate order (0..n-1), angles in RADIANS (VLFeat's
// own convention, not yet converted to degrees).
void ComputeOrientationsOnly(SiftData &siftData, CudaImage &img, int numOctaves,
                             std::vector<std::array<float, 4>> &outAngles,
                             std::vector<int> &outCount)
{
  int n = siftData.numPts;
  const int numScales = NUM_SCALES;

  CudaImage img01;
  img01.Allocate(img.width, img.height, iAlignUp(img.width, 128), false, NULL);
  {
    dim3 threads(16, 16);
    dim3 blocks(iDivUp(img.width, 16), iDivUp(img.height, 16));
    ScaleImageKernel<<<blocks, threads>>>(img01.d_data, img.d_data,
        img.width, img01.pitch, img.pitch, img.height, 1.0f / 255.0f);
    checkMsg("ScaleImageKernel() execution failed\n");
  }

  std::vector<int> vlOctaveOf(n), levelIdxOf(n);
  std::vector<int> cusiftOctaveOf(n);
  std::vector<float> ssOf(n);
  for (int i = 0; i < n; i++) {
    float absScale = std::max(siftData.h_data[i].scale, 1e-6f);
    int vo, li;
    AssignOctaveLevel(absScale, numOctaves, numScales, vo, li);
    vlOctaveOf[i] = vo;
    levelIdxOf[i] = li;
    cusiftOctaveOf[i] = numOctaves - vo;
    ssOf[i] = std::pow(2.0f, (float)vo);
  }

  std::vector<int> order(n);
  for (int i = 0; i < n; i++) order[i] = i;
  std::stable_sort(order.begin(), order.end(),
                    [&](int a, int b) { return cusiftOctaveOf[a] < cusiftOctaveOf[b]; });

  std::vector<SiftPoint> reordered(n);
  std::vector<int> srcOfPos(n);
  std::vector<int> perOctave(numOctaves + 1, 0);
  for (int i = 0; i < n; i++) {
    int src = order[i];
    SiftPoint p = siftData.h_data[src];
    float ss = ssOf[src];
    p.xpos /= ss;
    p.ypos /= ss;
    p.scale /= ss;
    p.subsampling = ss;
    p.empty[0] = (float)levelIdxOf[src];
    reordered[i] = p;
    srcOfPos[i] = src;
    perOctave[cusiftOctaveOf[src]]++;
  }
  std::copy(reordered.begin(), reordered.end(), siftData.h_data);

  safeCall(cudaMemcpy(siftData.d_data, siftData.h_data, sizeof(SiftPoint) * n, cudaMemcpyHostToDevice));
  unsigned int counters[8 * 2 + 1];
  memset(counters, 0, sizeof(counters));
  int cum = 0;
  for (int o = 1; o <= numOctaves; o++) {
    counters[2 * o - 1] = cum;
    cum += perOctave[o];
    counters[2 * o + 1] = cum;
  }
  // Boundary pack (harness ABI stays SiftPoint); per-octave slices explicit.
  PackedScratch &pk = Packed(n, false);
  PackKps(siftData.h_data, n, pk.h_kps, false);
  safeCall(cudaMemcpy(pk.d_kps, pk.h_kps, sizeof(VLFeatKp) * n, cudaMemcpyHostToDevice));
  std::vector<int> sliceStart(numOctaves + 2, 0), sliceCount(numOctaves + 2, 0);
  for (int o = 1; o <= numOctaves; o++) {
    sliceStart[o] = (int)counters[2 * o - 1];
    sliceCount[o] = (int)counters[2 * o + 1] - (int)counters[2 * o - 1];
  }

  std::vector<VLFeatOctaveBuffers> octaves(numOctaves);
  CudaImage prevTopLevel;
  for (int vo = 0; vo < numOctaves; vo++) {
    if (vo == 0) {
      BuildVLFeatOctave(octaves[vo], nullptr, &img01, numScales);
    } else {
      VLFeatOctaveBuffers &prev = octaves[vo - 1];
      prevTopLevel = LevelView(prev.gauss, numScales, prev.width, prev.height, prev.pitch);
      BuildVLFeatOctave(octaves[vo], &prevTopLevel, nullptr, numScales);
    }
  }

  dim3 blocks(512), threads(256);
  for (int vo = 0; vo < numOctaves; vo++) {
    int cusiftOctave = numOctaves - vo;
    int levelStride = octaves[vo].height * octaves[vo].pitch;
    int first = sliceStart[cusiftOctave], cnt = sliceCount[cusiftOctave];
    if (cnt <= 0) continue;
    ComputeOrientationsVLFeat<<<blocks, threads>>>(
        octaves[vo].gradMag.d_data, octaves[vo].gradAng.d_data,
        octaves[vo].width, octaves[vo].pitch, octaves[vo].height, levelStride,
        pk.d_kps + first, pk.d_ori + first, cnt);
    checkMsg("ComputeOrientationsVLFeat() execution failed\n");
  }

  safeCall(cudaMemcpy(pk.h_ori, pk.d_ori, sizeof(VLFeatOri) * n, cudaMemcpyDeviceToHost));
  outAngles.assign(n, std::array<float, 4>{0, 0, 0, 0});
  outCount.assign(n, 0);
  for (int i = 0; i < n; i++) {
    int src = srcOfPos[i];
    const VLFeatOri &p = pk.h_ori[i];
    outAngles[src] = {p.angles[0], p.angles[1], p.angles[2], p.angles[3]};
    outCount[src] = p.count;
  }
}

// Test-only variant of ComputeOrientationsOnly: builds an extra upsampled
// "octave -1" first (matching COLMAP's real production default,
// first_octave=-1 -- checked directly, colmap_ref.py's own comment says
// "leave sift at COLMAP production defaults" and pycolmap's default is
// first_octave=-1, NOT the o_min=0 this harness has assumed throughout
// Parts 2b'/2c). Octave 0 (and all coarser octaves) are then built by
// DECIMATING octave -1's top level, not by blurring the original image
// directly -- these are NOT numerically identical (upsample->blur->decimate
// is not a lossless round-trip), even though real ref_cpu happens to have
// zero keypoints with sigma<1.6 (i.e. octave -1 itself never contributes a
// keypoint on this dataset) -- so this exists purely to test whether octave
// 0's resulting pixel-content difference (measured ~0.09% RMS vs the o_min=0
// pyramid) explains any of the residual Gate 1/2 gap. NOT wired into any
// production path; AssignOctaveLevel's o>=0 clamp is intentionally left
// unchanged since no real candidate in this dataset needs o=-1 itself.
void ComputeOrientationsOnlyOm1(SiftData &siftData, CudaImage &img, int numOctaves,
                                std::vector<std::array<float, 4>> &outAngles,
                                std::vector<int> &outCount)
{
  int n = siftData.numPts;
  const int numScales = NUM_SCALES;

  CudaImage img01;
  img01.Allocate(img.width, img.height, iAlignUp(img.width, 128), false, NULL);
  {
    dim3 threads(16, 16);
    dim3 blocks(iDivUp(img.width, 16), iDivUp(img.height, 16));
    ScaleImageKernel<<<blocks, threads>>>(img01.d_data, img.d_data,
        img.width, img01.pitch, img.pitch, img.height, 1.0f / 255.0f);
    checkMsg("ScaleImageKernel() execution failed\n");
  }

  std::vector<int> vlOctaveOf(n), levelIdxOf(n);
  std::vector<int> cusiftOctaveOf(n);
  std::vector<float> ssOf(n);
  for (int i = 0; i < n; i++) {
    float absScale = std::max(siftData.h_data[i].scale, 1e-6f);
    int vo, li;
    AssignOctaveLevel(absScale, numOctaves, numScales, vo, li);
    vlOctaveOf[i] = vo;
    levelIdxOf[i] = li;
    cusiftOctaveOf[i] = numOctaves - vo;
    ssOf[i] = std::pow(2.0f, (float)vo);
  }

  std::vector<int> order(n);
  for (int i = 0; i < n; i++) order[i] = i;
  std::stable_sort(order.begin(), order.end(),
                    [&](int a, int b) { return cusiftOctaveOf[a] < cusiftOctaveOf[b]; });

  std::vector<SiftPoint> reordered(n);
  std::vector<int> srcOfPos(n);
  std::vector<int> perOctave(numOctaves + 1, 0);
  for (int i = 0; i < n; i++) {
    int src = order[i];
    SiftPoint p = siftData.h_data[src];
    float ss = ssOf[src];
    p.xpos /= ss;
    p.ypos /= ss;
    p.scale /= ss;
    p.subsampling = ss;
    p.empty[0] = (float)levelIdxOf[src];
    reordered[i] = p;
    srcOfPos[i] = src;
    perOctave[cusiftOctaveOf[src]]++;
  }
  std::copy(reordered.begin(), reordered.end(), siftData.h_data);

  safeCall(cudaMemcpy(siftData.d_data, siftData.h_data, sizeof(SiftPoint) * n, cudaMemcpyHostToDevice));
  unsigned int counters[8 * 2 + 1];
  memset(counters, 0, sizeof(counters));
  int cum = 0;
  for (int o = 1; o <= numOctaves; o++) {
    counters[2 * o - 1] = cum;
    cum += perOctave[o];
    counters[2 * o + 1] = cum;
  }
  // Boundary pack (harness ABI stays SiftPoint); per-octave slices explicit.
  PackedScratch &pk = Packed(n, false);
  PackKps(siftData.h_data, n, pk.h_kps, false);
  safeCall(cudaMemcpy(pk.d_kps, pk.h_kps, sizeof(VLFeatKp) * n, cudaMemcpyHostToDevice));
  std::vector<int> sliceStart(numOctaves + 2, 0), sliceCount(numOctaves + 2, 0);
  for (int o = 1; o <= numOctaves; o++) {
    sliceStart[o] = (int)counters[2 * o - 1];
    sliceCount[o] = (int)counters[2 * o + 1] - (int)counters[2 * o - 1];
  }

  // Build octave -1 (upsampled) as buffer slot 0, then octaves 0..numOctaves-1
  // as slots 1..numOctaves, each decimated from the previous slot's top level.
  std::vector<VLFeatOctaveBuffers> octaves(numOctaves + 1);
  BuildUpsampledOctaveMinus1(octaves[0], img01, numScales);
  CudaImage prevTopLevel;
  for (int vo = 0; vo < numOctaves; vo++) {
    VLFeatOctaveBuffers &prev = octaves[vo];
    prevTopLevel = LevelView(prev.gauss, numScales, prev.width, prev.height, prev.pitch);
    BuildVLFeatOctave(octaves[vo + 1], &prevTopLevel, nullptr, numScales);
  }

  dim3 blocks(512), threads(256);
  for (int vo = 0; vo < numOctaves; vo++) {
    int cusiftOctave = numOctaves - vo;
    VLFeatOctaveBuffers &buf = octaves[vo + 1];  // shifted: vo=0 (real octave 0) lives in slot 1
    int levelStride = buf.height * buf.pitch;
    int first = sliceStart[cusiftOctave], cnt = sliceCount[cusiftOctave];
    if (cnt <= 0) continue;
    ComputeOrientationsVLFeat<<<blocks, threads>>>(
        buf.gradMag.d_data, buf.gradAng.d_data,
        buf.width, buf.pitch, buf.height, levelStride,
        pk.d_kps + first, pk.d_ori + first, cnt);
    checkMsg("ComputeOrientationsVLFeat() execution failed\n");
  }

  safeCall(cudaMemcpy(pk.h_ori, pk.d_ori, sizeof(VLFeatOri) * n, cudaMemcpyDeviceToHost));
  outAngles.assign(n, std::array<float, 4>{0, 0, 0, 0});
  outCount.assign(n, 0);
  for (int i = 0; i < n; i++) {
    int src = srcOfPos[i];
    const VLFeatOri &p = pk.h_ori[i];
    outAngles[src] = {p.angles[0], p.angles[1], p.angles[2], p.angles[3]};
    outCount[src] = p.count;
  }
}

