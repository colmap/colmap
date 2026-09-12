// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Single point of CUDA <-> HIP source compatibility. All MVS/util GPU code is
// written in the CUDA spelling (cudaXxx, curandXxx). On ROCm builds, this
// header aliases those names to their HIP equivalents, so original .cu files
// compile unmodified when CMake's set_source_files_properties(LANGUAGE HIP)
// hands them to the HIP toolchain. The only file that should reference
// hipXxx symbols directly is this header.

#if defined(COLMAP_HIP_ENABLED)

// Workaround: rocrand's mtgp32 header uses printf() without including
// <cstdio>, which the CUDA toolchain's transitive <cstdio> include used to
// hide. Pull it in explicitly so host translation units that only need
// runtime/error types don't trip over rocrand's missing include.
#include <cstdio>

#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Errors, streams
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize

// Events
using cudaEvent_t = hipEvent_t;
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime

// Device management
using cudaDeviceProp = hipDeviceProp_t;
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties

// Memory
#define cudaMalloc hipMalloc
#define cudaMallocPitch hipMallocPitch
#define cudaMalloc3DArray hipMalloc3DArray
#define cudaFree hipFree
#define cudaFreeArray hipFreeArray
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemcpy3D hipMemcpy3D
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemset hipMemset
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyHostToHost hipMemcpyHostToHost

// Extents and pitched pointers
using cudaExtent = hipExtent;
using cudaPos = hipPos;
using cudaPitchedPtr = hipPitchedPtr;
using cudaMemcpy3DParms = hipMemcpy3DParms;
#define make_cudaExtent make_hipExtent
#define make_cudaPos make_hipPos
#define make_cudaPitchedPtr make_hipPitchedPtr

// Textures and arrays
using cudaArray = hipArray;
using cudaArray_t = hipArray_t;
using cudaChannelFormatDesc = hipChannelFormatDesc;
using cudaChannelFormatKind = hipChannelFormatKind;
using cudaResourceDesc = hipResourceDesc;
using cudaResourceType = hipResourceType;
using cudaTextureDesc = hipTextureDesc;
using cudaTextureObject_t = hipTextureObject_t;
using cudaTextureAddressMode = hipTextureAddressMode;
using cudaTextureFilterMode = hipTextureFilterMode;
using cudaTextureReadMode = hipTextureReadMode;
#define cudaCreateChannelDesc hipCreateChannelDesc
#define cudaCreateTextureObject hipCreateTextureObject
#define cudaDestroyTextureObject hipDestroyTextureObject

#define cudaArrayDefault hipArrayDefault
#define cudaArrayLayered hipArrayLayered
#define cudaResourceTypeArray hipResourceTypeArray

#define cudaAddressModeWrap hipAddressModeWrap
#define cudaAddressModeClamp hipAddressModeClamp
#define cudaAddressModeMirror hipAddressModeMirror
#define cudaAddressModeBorder hipAddressModeBorder
#define cudaFilterModePoint hipFilterModePoint
#define cudaFilterModeLinear hipFilterModeLinear
#define cudaReadModeElementType hipReadModeElementType
#define cudaReadModeNormalizedFloat hipReadModeNormalizedFloat

// cuRAND device API
using curandState = hiprandState;
#define curand_init hiprand_init
#define curand_uniform hiprand_uniform
#define curand_normal hiprand_normal

#elif defined(COLMAP_CUDA_ENABLED) || defined(__CUDACC__)

#include <cuda_runtime.h>
#include <curand_kernel.h>

#endif  // COLMAP_HIP_ENABLED
