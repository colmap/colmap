// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

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

#else  // !COLMAP_HIP_ENABLED

#include <cuda_runtime.h>
#include <curand_kernel.h>

#endif  // COLMAP_HIP_ENABLED
