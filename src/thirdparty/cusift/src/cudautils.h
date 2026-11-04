// SPDX-License-Identifier: MIT
// Copyright (c) 2017 Mårten Björkman (CudaSift; see ../LICENSE.CudaSift)
// Copyright (c) 2026 Yonatan Simson (see ../LICENSE)

#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

#define safeCall(err)       __safeCall(err, __FILE__, __LINE__)
#define safeThreadSync()    __safeThreadSync(__FILE__, __LINE__)
#define checkMsg(msg)       __checkMsg(msg, __FILE__, __LINE__)

// CUDA errors are reported as exceptions, never by ending the process: COLMAP
// propagates exceptions from worker threads to the caller (pycolmap, the GUI,
// the CLI), which decides what to do. Out-of-memory in particular is
// recoverable.
[[noreturn]] inline void __throwCudaError(const char *what, cudaError_t err,
                                          const char *file, const int line)
{
  throw std::runtime_error(std::string("cusift: ") + what + ": " +
                           cudaGetErrorString(err) + " (" + file + ":" +
                           std::to_string(line) + ")");
}

inline void __safeCall(cudaError err, const char *file, const int line)
{
  if (cudaSuccess != err) {
    // A failed call also sets the thread's last error; clear it, or the next
    // checkMsg would report this already-thrown failure again.
    (void)cudaGetLastError();
    __throwCudaError("CUDA runtime API error", err, file, line);
  }
}

inline void __safeThreadSync(const char *file, const int line)
{
  cudaError err = cudaDeviceSynchronize();
  if (cudaSuccess != err) __throwCudaError("device synchronization failed", err, file, line);
}

inline void __checkMsg(const char *errorMessage, const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) __throwCudaError(errorMessage, err, file, line);
}

// Every buffer in the VLFeat path is allocated per image and freed at the end
// of it, and cudaMalloc/cudaFree are both device-synchronizing. Routing those
// through a stream-ordered memory pool with an unbounded release threshold
// makes every allocation after the first image a pool hit: no driver call, no
// sync, memory stays resident. Sizes and pitches are unchanged (see
// cusiftMallocPitch).
//
// The pool is cusift's own, one per device (cusift_device.cu), not the
// device's default pool: the unbounded release threshold must not change how
// other cudaMallocAsync users in the process get their memory back.
// cusift::ReleaseUnusedMemory() hands what cusift no longer uses back to the
// driver. Falls back to the default pool, with its settings untouched, on a
// device that cannot create one.
cudaMemPool_t cusiftMemPool();

// Creates the current device's pool if needed. Idempotent; safe to call
// before any allocation.
inline void cusiftInitMemPool() { (void)cusiftMemPool(); }

// Stream-ordered allocation from cusift's pool on the legacy default stream,
// which is where the whole pipeline runs. Free with cusiftFree.
template <typename T>
inline cudaError_t cusiftMallocAsync(T **ptr, size_t bytes)
{
  return cudaMallocFromPoolAsync((void **)ptr, bytes, cusiftMemPool(), 0);
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
  return cusiftMallocAsync((void **)ptr, (size_t)p * height * sizeof(float));
}

inline cudaError_t cusiftFree(void *ptr) { return cudaFreeAsync(ptr, 0); }


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





#endif

