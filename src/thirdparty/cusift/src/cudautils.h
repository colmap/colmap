#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <cstdint>
#include <cstdio>
#include <iostream>

#define safeCall(err)       __safeCall(err, __FILE__, __LINE__)
#define safeThreadSync()    __safeThreadSync(__FILE__, __LINE__)
#define checkMsg(msg)       __checkMsg(msg, __FILE__, __LINE__)

inline void __safeCall(cudaError err, const char *file, const int line)
{
  if (cudaSuccess != err) {
    fprintf(stderr, "safeCall() Runtime API error in file <%s>, line %i : %s.\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

inline void __safeThreadSync(const char *file, const int line)
{
  cudaError err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "threadSynchronize() Driver API error in file '%s' in line %i : %s.\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

inline void __checkMsg(const char *errorMessage, const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "checkMsg() CUDA error: %s in file <%s>, line %i : %s.\n", errorMessage, file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

// Tier 1.1 (interim form): every buffer in the VLFeat path is
// allocated per image and freed at the end of it (~64 cudaMalloc/cudaFree and
// ~1 GB of churn per 1920x1080 frame), and cudaMalloc/cudaFree are both
// device-synchronizing. Routing those through the device's default stream-
// ordered memory pool with an unbounded release threshold makes every
// allocation after the first image a pool hit: no driver call, no sync, memory
// stays resident. No numerics change -- same sizes, same pitches (see
// cusiftMallocPitch), same kernels.
//
// Called once per process, idempotent; safe to call before any allocation.
inline void cusiftInitMemPool()
{
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return;
  cudaMemPool_t pool = nullptr;
  if (cudaDeviceGetDefaultMemPool(&pool, dev) != cudaSuccess) return;
  uint64_t threshold = UINT64_MAX;   // never release back to the driver
  cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
}

// Pool-backed replacement for cudaMallocPitch. `pitchFloats` is computed, not
// queried: cudaMallocPitch pads rows to a 512-byte multiple, i.e. exactly
// iAlignUp(width, 128) floats, so every pitch in the pipeline is unchanged and
// the pitch-equality assertions in BuildVLFeatOctave still hold by
// construction. Allocations are stream-ordered on the legacy default stream,
// which is where the whole pipeline already runs.
inline cudaError_t cusiftMallocPitch(float **ptr, int *pitchFloats, int width, int height)
{
  int p = (width % 128) ? (width - width % 128 + 128) : width;
  *pitchFloats = p;
  return cudaMallocAsync((void **)ptr, (size_t)p * height * sizeof(float), 0);
}

inline cudaError_t cusiftFree(void *ptr) { return cudaFreeAsync(ptr, 0); }

inline bool deviceInit(int dev)
{
  int deviceCount;
  safeCall(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
    return false;
  }
  if (dev < 0) dev = 0;						
  if (dev > deviceCount-1) dev = deviceCount - 1;
  cudaDeviceProp deviceProp;
  safeCall(cudaGetDeviceProperties(&deviceProp, dev));
  if (deviceProp.major < 1) {
    fprintf(stderr, "error: device does not support CUDA.\n");
    return false;					
  }
  safeCall(cudaSetDevice(dev));
  return true;
}

class TimerGPU {
public:
  cudaEvent_t start, stop; 
  cudaStream_t stream;
  TimerGPU(cudaStream_t stream_ = 0) : stream(stream_) {
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 
    cudaEventRecord(start, stream); 
  }
  ~TimerGPU() {
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);  
  }
  float read() {
    cudaEventRecord(stop, stream); 
    cudaEventSynchronize(stop); 
    float time;
    cudaEventElapsedTime(&time, start, stop);
    return time;
  }
};

template <class T>
__device__ __inline__ T ShiftDown(T var, unsigned int delta, int width = 32) {
#if (CUDART_VERSION >= 9000)
  return __shfl_down_sync(0xffffffff, var, delta, width);
#else
  return __shfl_down(var, delta, width);
#endif
}

template <class T>
__device__ __inline__ T ShiftUp(T var, unsigned int delta, int width = 32) {
#if (CUDART_VERSION >= 9000)
  return __shfl_up_sync(0xffffffff, var, delta, width);
#else
  return __shfl_up(var, delta, width);
#endif
}

template <class T>
__device__ __inline__ T Shuffle(T var, unsigned int lane, int width = 32) {
#if (CUDART_VERSION >= 9000)
  return __shfl_sync(0xffffffff, var, lane, width);
#else
  return __shfl(var, lane, width);
#endif
}


#endif

