// SPDX-License-Identifier: MIT
// Copyright (c) 2017 Mårten Björkman (CudaSift; see ../LICENSE.CudaSift)
// Copyright (c) 2026 Yonatan Simson (see ../LICENSE)

#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#define safeCall(err)       __safeCall(err, __FILE__, __LINE__)
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

// The calling thread's current device.
inline int cusiftCurrentDevice() {
  int device = 0;
  safeCall(cudaGetDevice(&device));
  return device;
}

// Makes `device` current for the lifetime of the object, then restores the
// previous one. For freeing a per-thread holder's buffers on the device they
// were allocated on after the thread has moved to another.
class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(int device) : previous_(cusiftCurrentDevice()) {
    safeCall(cudaSetDevice(device));
  }
  ~ScopedCudaDevice() { cudaSetDevice(previous_); }
  ScopedCudaDevice(const ScopedCudaDevice &) = delete;
  ScopedCudaDevice &operator=(const ScopedCudaDevice &) = delete;

 private:
  const int previous_;
};

// As ScopedCudaDevice, for destructors: never throws, and does nothing for
// device -1 (nothing was allocated).
class ScopedCudaDeviceNoThrow {
 public:
  explicit ScopedCudaDeviceNoThrow(int device) {
    if (device < 0 || cudaGetDevice(&previous_) != cudaSuccess) return;
    if (cudaSetDevice(device) != cudaSuccess) previous_ = -1;
  }
  ~ScopedCudaDeviceNoThrow() {
    if (previous_ >= 0) cudaSetDevice(previous_);
  }
  ScopedCudaDeviceNoThrow(const ScopedCudaDeviceNoThrow &) = delete;
  ScopedCudaDeviceNoThrow &operator=(const ScopedCudaDeviceNoThrow &) = delete;

 private:
  int previous_ = -1;
};

// Owns one allocation from cusift's pool, so a throw between allocation and
// use cannot leak it.
template <typename T>
class PoolBuffer {
 public:
  PoolBuffer() = default;
  PoolBuffer(const PoolBuffer &) = delete;
  PoolBuffer &operator=(const PoolBuffer &) = delete;
  ~PoolBuffer() { if (ptr_) cusiftFree(ptr_); }
  // Replaces the allocation with one of `count` elements.
  void Reset(size_t count) {
    T *const old = ptr_;
    ptr_ = nullptr;
    if (old) safeCall(cusiftFree(old));
    safeCall(cusiftMallocAsync(&ptr_, count * sizeof(T)));
  }
  T *get() const { return ptr_; }

 private:
  T *ptr_ = nullptr;
};

#endif

