////////////////////////////////////////////////////////////////////////////
//  File:           ProgramCU.cu
//  Author:         Changchang Wu
//  Description :   implementation of ProgramCU and all CUDA kernels
//
//  Copyright (c) 2011  Changchang Wu (ccwu@cs.washington.edu)
//    and the University of Washington at Seattle
//
//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation; either
//  Version 3 of the License, or (at your option) any later version.
//
//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  General Public License for more details.
//
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <float.h>
#include "CuTexImage.h"
#include "ProgramCU.h"

#define IMUL(X, Y) __mul24(X, Y)
#define FDIV(X, Y) __fdividef(X, Y)
#define FDIV2(X, Y) ((X) / (Y))
#define MAX_BLOCKLEN 65535
#define MAX_BLOCKLEN_ALIGN 65504
#define MAX_TEXSIZE (1 << 29)
#define TEX_TOOBIG4(sz) (sz >> 31)
#define REDUCTION_NBLOCK 32

namespace pba {

inline void CuTexImage::BindTexture(textureReference& texRef) {
  size_t sz = GetDataSize();
  if (sz > MAX_TEXSIZE)
    fprintf(stderr, "cudaBindTexture: %lX > %d\n", sz, MAX_TEXSIZE);
  cudaError_t e =
      cudaBindTexture(NULL, &texRef, data(), &texRef.channelDesc, sz);
}

inline void CuTexImage::BindTexture(textureReference& texRef, int offset,
                                    size_t size) {
  cudaError_t e = cudaBindTexture(NULL, &texRef, (char*)_cuData + offset,
                                  &texRef.channelDesc, size);
  if (e) fprintf(stderr, "cudaBindTexture: none-zero offset\n");
}

inline void CuTexImage::BindTexture2(textureReference& texRef1,
                                     textureReference& texRef2) {
  size_t sz = GetDataSize();
  if (sz <= MAX_TEXSIZE) {
    BindTexture(texRef1);
  } else {
    BindTexture(texRef1, 0, MAX_TEXSIZE);
    BindTexture(texRef2, MAX_TEXSIZE, sz - MAX_TEXSIZE);
  }
}

inline void CuTexImage::BindTexture4(textureReference& texRef1,
                                     textureReference& texRef2,
                                     textureReference& texRef3,
                                     textureReference& texRef4) {
  size_t sz = GetDataSize();
  if (sz <= MAX_TEXSIZE) {
    BindTexture(texRef1);
  } else {
    BindTexture(texRef1, 0, MAX_TEXSIZE);
    if (sz <= 2 * MAX_TEXSIZE) {
      BindTexture(texRef2, MAX_TEXSIZE, sz - MAX_TEXSIZE);
    } else {
      BindTexture(texRef2, MAX_TEXSIZE, MAX_TEXSIZE);
      if (sz <= 3 * MAX_TEXSIZE) {
        BindTexture(texRef3, MAX_TEXSIZE * 2, sz - MAX_TEXSIZE * 2);
      } else {
        BindTexture(texRef3, MAX_TEXSIZE * 2, MAX_TEXSIZE);
        BindTexture(texRef4, MAX_TEXSIZE * 3, sz - MAX_TEXSIZE * 3);
      }
    }
  }
}

inline int CuTexImage::BindTextureX(textureReference& texRef1,
                                    textureReference& texRef2,
                                    textureReference& texRef3,
                                    textureReference& texRef4, bool force4) {
  size_t szjc = GetDataSize();
  if (TEX_TOOBIG4(szjc)) {
    return 0;
  } else if (force4) {
    BindTexture4(texRef1, texRef2, texRef3, texRef4);
    return 4;
  } else if (szjc > 2 * MAX_TEXSIZE) {
    return 0;
  } else if (szjc > MAX_TEXSIZE) {
    BindTexture2(texRef1, texRef2);
    return 2;
  } else {
    BindTexture(texRef1);
    return 1;
  }
}

void ProgramCU::FinishWorkCUDA() { cudaDeviceSynchronize(); }

int ProgramCU::CheckErrorCUDA(const char* location) {
  cudaError_t e = cudaGetLastError();
  if (e) {
    if (location) fprintf(stderr, "%s:\t", location);
    fprintf(stderr, "%s(%d)\n", cudaGetErrorString(e), e);
    throw location;
  } else {
    // fprintf(stderr, "%s:\n",  location);
    return 0;
  }
}

inline void ProgramCU::GetBlockConfiguration(unsigned int nblock,
                                             unsigned int& bw,
                                             unsigned int& bh) {
  if (nblock <= MAX_BLOCKLEN) {
    bw = nblock;
    bh = 1;
  } else {
    bh = (nblock + MAX_BLOCKLEN_ALIGN - 1) / MAX_BLOCKLEN_ALIGN;
    bw = (nblock + bh - 1) / bh;
    bw = ((bw + 31) / 32) * 32;
    bh = (nblock + bw - 1) / bw;
  }
}

void ProgramCU::ClearPreviousError() { cudaGetLastError(); }

void ProgramCU::ResetCurrentDevice() {
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceReset();
  if (device > 0) cudaSetDevice(device);
}

size_t ProgramCU::GetCudaMemoryCap() {
  int device;
  if (cudaGetDevice(&device) != cudaSuccess) return 0;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
    if (prop.major == 9999 && prop.minor == 9999) return 0;
    return prop.totalGlobalMem;
  } else
    return 0;
}
int ProgramCU::SetCudaDevice(int device) {
  int count = 0, device_used;
  if (cudaGetDeviceCount(&count) || count <= 0) {
    ProgramCU::CheckErrorCUDA("CheckCudaDevice");
    return 0;
  } else if (count == 1) {
    cudaDeviceProp deviceProp;
    if (cudaGetDeviceProperties(&deviceProp, 0) != cudaSuccess) {
      fprintf(stderr, "CheckCudaDevice: no device supporting CUDA.\n");
      return 0;
    }
    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
      fprintf(stderr, "CheckCudaDevice: no device supporting CUDA.\n");
      return 0;
    }
  }

  if (device > 0 && device < count) {
    cudaSetDevice(device);
    CheckErrorCUDA("cudaSetDevice\n");
  }
  cudaGetDevice(&device_used);
  if (device != device_used)
    fprintf(stderr,
            "ERROR:   Cannot set device to %d\n"
            "WARNING: Use  device-%d instead (out of %d)\n",
            device, device_used, count);
  return 1;
}

#define WARP_REDUCTION_32(value)                                       \
  __syncthreads();                                                     \
  if (threadIdx.x < 16) value[threadIdx.x] += value[threadIdx.x + 16]; \
  if (threadIdx.x < 8) value[threadIdx.x] += value[threadIdx.x + 8];   \
  if (threadIdx.x < 4) value[threadIdx.x] += value[threadIdx.x + 4];   \
  if (threadIdx.x < 2) value[threadIdx.x] += value[threadIdx.x + 2];

#define WARP_REDUCTION_64(value)                                       \
  __syncthreads();                                                     \
  if (threadIdx.x < 32) value[threadIdx.x] += value[threadIdx.x + 32]; \
  WARP_REDUCTION_32(value)

#define WARP_REDUCTION_128(value)                                      \
  __syncthreads();                                                     \
  if (threadIdx.x < 64) value[threadIdx.x] += value[threadIdx.x + 64]; \
  WARP_REDUCTION_64(value)

#define WARP_REDUCTION_256(value)                                        \
  __syncthreads();                                                       \
  if (threadIdx.x < 128) value[threadIdx.x] += value[threadIdx.x + 128]; \
  WARP_REDUCTION_128(value)

__global__ void vector_max_kernel(const float* x, int len, int blen,
                                  float* result) {
  __shared__ float value[256];
  int bstart = blen * blockIdx.x;
  int start = bstart + threadIdx.x;
  int end = min(len, bstart + blen);

  float v = 0;
  for (int i = start; i < end; i += blockDim.x) v = max(v, fabs(x[i]));
  value[threadIdx.x] = v;
  // reduce to the first two values
  __syncthreads();
  if (threadIdx.x < 128)
    value[threadIdx.x] = max(value[threadIdx.x], value[threadIdx.x + 128]);
  __syncthreads();
  if (threadIdx.x < 64)
    value[threadIdx.x] = max(value[threadIdx.x], value[threadIdx.x + 64]);
  __syncthreads();
  if (threadIdx.x < 32)
    value[threadIdx.x] = max(value[threadIdx.x], value[threadIdx.x + 32]);
  if (threadIdx.x < 16)
    value[threadIdx.x] = max(value[threadIdx.x], value[threadIdx.x + 16]);
  if (threadIdx.x < 8)
    value[threadIdx.x] = max(value[threadIdx.x], value[threadIdx.x + 8]);
  if (threadIdx.x < 4)
    value[threadIdx.x] = max(value[threadIdx.x], value[threadIdx.x + 4]);
  if (threadIdx.x < 2)
    value[threadIdx.x] = max(value[threadIdx.x], value[threadIdx.x + 2]);
  // write back
  if (threadIdx.x == 0) result[blockIdx.x] = max(value[0], value[1]);
}

float ProgramCU::ComputeVectorMax(CuTexImage& vector, CuTexImage& buf) {
  const unsigned int nblock = 32;
  const unsigned int bsize = 256;
  int len = vector.GetLength();
  int blen = ((len + nblock - 1) / nblock + bsize - 1) / bsize * bsize;

  ////////////////////////////////
  dim3 grid(nblock), block(bsize);

  /////////////////////////////////
  buf.InitTexture(nblock, 1);
  vector_max_kernel<<<grid, block>>>(vector.data(), len, blen, buf.data());
  ProgramCU::CheckErrorCUDA("ComputeVectorMax");

  float data[nblock], result = 0;
  buf.CopyToHost(data);
  for (unsigned int i = 0; i < nblock; ++i) result = max(result, data[i]);
  return result;
}

__global__ void vector_norm_kernel(const float* x, int len, int blen,
                                   float* result) {
  __shared__ float value[256];
  int bstart = blen * blockIdx.x;
  int start = bstart + threadIdx.x;
  int end = min(len, bstart + blen);

  float v = 0;
  for (int i = start; i < end; i += blockDim.x) {
    float temp = x[i];
    v += (temp * temp);
  }
  value[threadIdx.x] = v;
  // reduce to the first two values
  WARP_REDUCTION_256(value);

  // write back
  if (threadIdx.x == 0) result[blockIdx.x] = (value[0] + value[1]);
}

double ProgramCU::ComputeVectorNorm(CuTexImage& vector, CuTexImage& buf) {
  const unsigned int nblock = REDUCTION_NBLOCK;
  unsigned int bsize = 256;
  int len = vector.GetLength();
  int blen = ((len + nblock - 1) / nblock + bsize - 1) / bsize * bsize;

  ////////////////////////////////
  dim3 grid(nblock), block(bsize);

  /////////////////////////////////
  buf.InitTexture(nblock, 1);
  vector_norm_kernel<<<grid, block>>>(vector.data(), len, blen, buf.data());
  ProgramCU::CheckErrorCUDA("ComputeVectorNorm");

  float data[nblock];
  buf.CopyToHost(data);
  double result = 0;
  for (unsigned int i = 0; i < nblock; ++i) result += data[i];
  return result;
}

__global__ void vector_sum_kernel(const float* x, int len, int blen,
                                  float* result) {
  __shared__ float value[256];
  int bstart = blen * blockIdx.x;
  int start = bstart + threadIdx.x;
  int end = min(len, bstart + blen);
  float v = 0;
  for (int i = start; i < end; i += blockDim.x) v += x[i];

  value[threadIdx.x] = v;
  // reduce to the first two values
  WARP_REDUCTION_256(value);

  // write back
  if (threadIdx.x == 0) result[blockIdx.x] = (value[0] + value[1]);
}

float ProgramCU::ComputeVectorSum(CuTexImage& vector, CuTexImage& buf,
                                  int skip) {
  const unsigned int nblock = REDUCTION_NBLOCK;
  unsigned int bsize = 256;
  int len = vector.GetLength() - skip;
  int blen = ((len + nblock - 1) / nblock + bsize - 1) / bsize * bsize;

  ////////////////////////////////
  dim3 grid(nblock), block(bsize);

  /////////////////////////////////
  buf.InitTexture(nblock, 1);
  vector_sum_kernel<<<grid, block>>>((vector.data()) + skip, len, blen,
                                     buf.data());
  ProgramCU::CheckErrorCUDA("ComputeVectorSum");

  float data[nblock];
  buf.CopyToHost(data);
  double result = 0;
  for (unsigned int i = 0; i < nblock; ++i) result += data[i];
  return (float)result;
}

__global__ void vector_dotproduct_kernel(const float* a, const float* b,
                                         int len, int blen, float* result) {
  __shared__ float value[256];
  int bstart = blen * blockIdx.x;
  int start = bstart + threadIdx.x;
  int end = min(len, bstart + blen);

  float v = 0;
  for (int i = start; i < end; i += blockDim.x) v += (a[i] * b[i]);
  value[threadIdx.x] = v;

  // reduce to the first two values
  WARP_REDUCTION_256(value);

  // write back
  if (threadIdx.x == 0) result[blockIdx.x] = (value[0] + value[1]);
}

double ProgramCU::ComputeVectorDot(CuTexImage& vector1, CuTexImage& vector2,
                                   CuTexImage& buf) {
  const unsigned int nblock = REDUCTION_NBLOCK;
  unsigned int bsize = 256;
  int len = vector1.GetLength();
  int blen = ((len + nblock - 1) / nblock + bsize - 1) / bsize * bsize;

  ////////////////////////////////
  dim3 grid(nblock), block(bsize);

  /////////////////////////////////
  buf.InitTexture(nblock, 1);
  vector_dotproduct_kernel<<<grid, block>>>(vector1.data(), vector2.data(), len,
                                            blen, buf.data());
  ProgramCU::CheckErrorCUDA("ComputeVectorDot");

  float data[nblock];
  buf.CopyToHost(data);

  double result = 0;
  for (unsigned int i = 0; i < nblock; ++i) result += data[i];
  return result;
}

__global__ void vector_weighted_norm_kernel(const float* vec, const float* w,
                                            int len, int blen, float* result) {
  __shared__ float value[256];
  int bstart = blen * blockIdx.x;
  int start = bstart + threadIdx.x;
  int end = min(len, bstart + blen);

  float v = 0;
  for (int i = start; i < end; i += blockDim.x) v += (vec[i] * w[i] * vec[i]);
  value[threadIdx.x] = v;

  // reduce to the first two values
  WARP_REDUCTION_256(value);

  // write back
  if (threadIdx.x == 0) result[blockIdx.x] = (value[0] + value[1]);
}

double ProgramCU::ComputeVectorNormW(CuTexImage& vector, CuTexImage& weight,
                                     CuTexImage& buf) {
  if (weight.IsValid()) {
    const unsigned int nblock = REDUCTION_NBLOCK;
    unsigned int bsize = 256;
    int len = vector.GetLength();
    int blen = ((len + nblock - 1) / nblock + bsize - 1) / bsize * bsize;

    ////////////////////////////////
    dim3 grid(nblock), block(bsize);

    /////////////////////////////////
    buf.InitTexture(nblock, 1);

    vector_weighted_norm_kernel<<<grid, block>>>(vector.data(), weight.data(),
                                                 len, blen, buf.data());

    ProgramCU::CheckErrorCUDA("ComputeVectorNormW");

    float data[nblock];
    buf.CopyToHost(data);

    double result = 0;
    for (unsigned int i = 0; i < nblock; ++i) result += data[i];
    return result;
  } else {
    return ComputeVectorNorm(vector, buf);
  }
}
// given vector x, y, and a weight a
// return a * x + y
__global__ void saxpy_kernel(const float a, const float* x, const float* y,
                             float* result, unsigned int len) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) result[idx] = a * x[idx] + y[idx];
}

__global__ void saxpy_kernel_large(const float a, const float* x,
                                   const float* y, float* result,
                                   unsigned int len, unsigned int rowsz) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * rowsz;
  if (idx < len) result[idx] = a * x[idx] + y[idx];
}

void ProgramCU::ComputeSAXPY(float a, CuTexImage& texX, CuTexImage& texY,
                             CuTexImage& result) {
  unsigned int len = result.GetLength();
  unsigned int bsize = 128;
  unsigned int nblock = (len + bsize - 1) / bsize;
  if (nblock > MAX_BLOCKLEN) {
    unsigned int bw, bh;
    GetBlockConfiguration(nblock, bw, bh);
    dim3 grid(bw, bh), block(bsize);
    saxpy_kernel_large<<<grid, block>>>(a, texX.data(), texY.data(),
                                        result.data(), len, bw * bsize);
  } else {
    dim3 grid(nblock), block(bsize);
    saxpy_kernel<<<grid, block>>>(a, texX.data(), texY.data(), result.data(),
                                  len);
  }
  ProgramCU::CheckErrorCUDA("ComputeSAXPY");
}

__global__ void sxypz_kernel_large(float a, const float* x, const float* y,
                                   const float* z, float* result,
                                   unsigned int len, unsigned int rowsz) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * rowsz;
  if (idx < len) result[idx] = a * x[idx] * y[idx] + z[idx];
}

void ProgramCU::ComputeSXYPZ(float a, CuTexImage& texX, CuTexImage& texY,
                             CuTexImage& texZ, CuTexImage& result) {
  if (texX.IsValid()) {
    unsigned int len = texX.GetLength();
    unsigned int bsize = 128;
    unsigned int nblock = (len + bsize - 1) / bsize;
    unsigned int bw, bh;
    GetBlockConfiguration(nblock, bw, bh);
    dim3 grid(bw, bh), block(bsize);
    sxypz_kernel_large<<<grid, block>>>(a, texX.data(), texY.data(),
                                        texZ.data(), result.data(), len,
                                        bw * bsize);
  } else {
    ComputeSAXPY(a, texY, texZ, result);
  }
}

__global__ void vxy_kernel(const float* x, float* y, float* result,
                           unsigned int len) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) result[idx] = x[idx] * y[idx];
}

__global__ void vxy_kernel_large(const float* x, float* y, float* result,
                                 unsigned int len, unsigned int rowsz) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x + rowsz * blockIdx.y;
  if (idx < len) result[idx] = x[idx] * y[idx];
}

void ProgramCU::ComputeVXY(CuTexImage& texX, CuTexImage& texY,
                           CuTexImage& result, unsigned int part,
                           unsigned int skip) {
  unsigned int len = part ? part : texX.GetLength();
  unsigned int bsize = 128;
  unsigned int nblock = (len + bsize - 1) / bsize;
  if (nblock > MAX_BLOCKLEN) {
    unsigned int bw, bh;
    GetBlockConfiguration(nblock, bw, bh);
    dim3 grid(bw, bh), block(bsize);
    vxy_kernel_large<<<grid, block>>>(texX.data() + skip, texY.data() + skip,
                                      result.data() + skip, len, bsize * bw);
  } else {
    dim3 grid(nblock), block(bsize);
    vxy_kernel<<<grid, block>>>(texX.data() + skip, texY.data() + skip,
                                result.data() + skip, len);
  }
  ProgramCU::CheckErrorCUDA("ComputeVXY");
}

__global__ void sqrt_kernel_large(float* x, unsigned int len,
                                  unsigned int rowsz) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * rowsz;
  if (idx < len) x[idx] = sqrt(x[idx]);
}

void ProgramCU::ComputeSQRT(CuTexImage& tex) {
  unsigned int len = tex.GetLength();
  unsigned int bsize = 128;
  unsigned int nblock = (len + bsize - 1) / bsize;
  unsigned int bw, bh;
  GetBlockConfiguration(nblock, bw, bh);
  dim3 grid(bw, bh), block(bsize);
  sqrt_kernel_large<<<grid, block>>>(tex.data(), len, bw * bsize);
  ProgramCU::CheckErrorCUDA("ComputeSQRT");
}

__global__ void rsqrt_kernel_large(float* x, unsigned int len,
                                   unsigned int rowsz) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * rowsz;
  if (idx < len) x[idx] = x[idx] > 0 ? rsqrt(x[idx]) : 0;
}

void ProgramCU::ComputeRSQRT(CuTexImage& tex) {
  unsigned int len = tex.GetLength();
  unsigned int bsize = 128;
  unsigned int nblock = (len + bsize - 1) / bsize;
  unsigned int bw, bh;
  GetBlockConfiguration(nblock, bw, bh);
  dim3 grid(bw, bh), block(bsize);
  rsqrt_kernel_large<<<grid, block>>>(tex.data(), len, bw * bsize);

  ProgramCU::CheckErrorCUDA("ComputeRSQRT");
}

__global__ void sax_kernel(const float a, const float* x, float* result,
                           unsigned int len) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) result[idx] = a * x[idx];
}

__global__ void sax_kernel_large(const float a, const float* x, float* result,
                                 unsigned int len, unsigned int rowsz) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * rowsz;
  if (idx < len) result[idx] = a * x[idx];
}

void ProgramCU::ComputeSAX(float a, CuTexImage& texX, CuTexImage& result) {
  unsigned int len = texX.GetLength();
  unsigned int bsize = 128;
  unsigned int nblock = (len + bsize - 1) / bsize;

  if (nblock > MAX_BLOCKLEN) {
    unsigned int bw, bh;
    GetBlockConfiguration(nblock, bw, bh);
    dim3 grid(bw, bh), block(bsize);
    sax_kernel_large<<<grid, block>>>(a, texX.data(), result.data(), len,
                                      bw * bsize);
  } else {
    dim3 grid(nblock), block(bsize);
    sax_kernel<<<grid, block>>>(a, texX.data(), result.data(), len);
  }
  ProgramCU::CheckErrorCUDA("ComputeSAX");
}

#define JACOBIAN_FRT_KWIDTH 64

texture<float4, 1, cudaReadModeElementType> tex_jacobian_cam;
texture<float4, 1, cudaReadModeElementType> tex_jacobian_pts;
texture<int2, 1, cudaReadModeElementType> tex_jacobian_idx;
texture<float2, 1, cudaReadModeElementType> tex_jacobian_meas;
texture<float4, 1, cudaReadModeElementType> tex_jacobian_sj;
texture<int, 1, cudaReadModeElementType> tex_jacobian_shuffle;

#ifndef PBA_DISABLE_CONST_CAMERA
#define JACOBIAN_SET_JC_BEGIN if (r3.w == 0.0f) {
#define JFRT_SET_JC_END                       \
  }                                           \
  else {                                      \
    jc[jc_pos] = make_float4(0, 0, 0, 0);     \
    jc[jc_pos + 1] = make_float4(0, 0, 0, 0); \
    jc[jc_pos + 2] = make_float4(0, 0, 0, 0); \
    jc[jc_pos + 3] = make_float4(0, 0, 0, 0); \
  }
#define JACOBIAN_SET_JC_END \
  }                         \
  else {                    \
    jxc[0] = 0;             \
    jxc[1] = 0;             \
    jxc[2] = 0;             \
    jxc[3] = 0;             \
    jxc[4] = 0;             \
    jxc[5] = 0;             \
    jxc[6] = 0;             \
    jxc[7] = 0;             \
    jyc[0] = 0;             \
    jyc[1] = 0;             \
    jyc[2] = 0;             \
    jyc[3] = 0;             \
    jyc[4] = 0;             \
    jyc[5] = 0;             \
    jyc[6] = 0;             \
    jyc[7] = 0;             \
  }
#else
#define JACOBIAN_SET_JC_BEGIN
#define JFRT_SET_JC_END
#define JACOBIAN_SET_JC_END
#endif

// projection model ei = K(RX + T)  - (1 + r * m^2) * m
template <bool md, bool pd, bool scaling, bool shuffle>
__global__ void jacobian_frt_kernel(float4* jc, float4* jp, int nproj, int ptx,
                                    int rowsz, float jic) {
  ////////////////////////////////
  int tidx = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * rowsz;

  if (tidx >= nproj) return;
  int2 proj = tex1Dfetch(tex_jacobian_idx, tidx);
  int camera_pos = proj.x << 1;

  __shared__ float rr_data[JACOBIAN_FRT_KWIDTH * 9];
  float* r = rr_data + IMUL(9, threadIdx.x);
  float4 ft = tex1Dfetch(tex_jacobian_cam, camera_pos);
  float4 r1 = tex1Dfetch(tex_jacobian_cam, camera_pos + 1);
  r[0] = r1.x;
  r[1] = r1.y;
  r[2] = r1.z;
  r[3] = r1.w;
  float4 r2 = tex1Dfetch(tex_jacobian_cam, camera_pos + 2);
  r[4] = r2.x;
  r[5] = r2.y;
  r[6] = r2.z;
  r[7] = r2.w;
  float4 r3 = tex1Dfetch(tex_jacobian_cam, camera_pos + 3);
  r[8] = r3.x;

  float4 temp = tex1Dfetch(tex_jacobian_pts, proj.y);
  float m[3];
  m[0] = temp.x;
  m[1] = temp.y;
  m[2] = temp.z;

  float x0 = r[0] * m[0] + r[1] * m[1] + r[2] * m[2];
  float y0 = r[3] * m[0] + r[4] * m[1] + r[5] * m[2];
  float z0 = r[6] * m[0] + r[7] * m[1] + r[8] * m[2];
  float f_p2 = FDIV(ft.x, z0 + ft.w);
  float p0_p2 = FDIV(x0 + ft.y, z0 + ft.w);
  float p1_p2 = FDIV(y0 + ft.z, z0 + ft.w);

  // dp/dx = [f/p2  0      -f*p0/p2/p2]
  //        [0     f/p2   -f*p1/p2/p2]
  // dx/dw = [ 0  z -y]
  //        [-z  0  x]
  //        [ y -x  0]
  // R(dw) (x y z)' = (0 -z y)' dw0 + (z 0  -x)'dw1 + (-y x 0)'dw2
  int jc_pos;
  if (shuffle) {
    jc_pos = tex1Dfetch(tex_jacobian_shuffle, tidx) << 2;
  } else {
    jc_pos = tidx << 2;
  }

  if (pd) {
    float rr1 = r3.y * p0_p2 * p0_p2;
    float rr2 = r3.y * p1_p2 * p1_p2;
    float f_p2_x = f_p2 * (1.0 + 3.0 * rr1 + rr2);
    float f_p2_y = f_p2 * (1.0 + 3.0 * rr2 + rr1);
    if (scaling == false) {
      if (jc) {
        JACOBIAN_SET_JC_BEGIN
        // float jic = (r3.w != 1.0f && r3.w != 2.0f) ? 1.0f : 0.0f;
        // float jec = (r3.w != 1.0f && r3.w != 3.0f) ? 1.0f : 0.0f;
        float jfc = jic * (1 + rr1 + rr2);
        float ft_x_pn = jic * ft.x * (p0_p2 * p0_p2 + p1_p2 * p1_p2);
        jc[jc_pos] = make_float4(p0_p2 * jfc, f_p2_x, 0, -f_p2_x * p0_p2);
        jc[jc_pos + 1] =
            make_float4(-f_p2_x * p0_p2 * y0, f_p2_x * (z0 + x0 * p0_p2),
                        -f_p2_x * y0, ft_x_pn * p0_p2);
        jc[jc_pos + 2] = make_float4(p1_p2 * jfc, 0, f_p2_y, -f_p2 * p1_p2);
        jc[jc_pos + 3] =
            make_float4(-f_p2_y * (z0 + y0 * p1_p2), f_p2_y * x0 * p1_p2,
                        f_p2_y * x0, ft_x_pn * p1_p2);
        JFRT_SET_JC_END
      }
      ////////////////////
      jp[(tidx << 1)] = make_float4(f_p2_x * (r[0] - r[6] * p0_p2),
                                    f_p2_x * (r[1] - r[7] * p0_p2),
                                    f_p2_x * (r[2] - r[8] * p0_p2), 0);
      jp[(tidx << 1) + 1] = make_float4(f_p2_y * (r[3] - r[6] * p1_p2),
                                        f_p2_y * (r[4] - r[7] * p1_p2),
                                        f_p2_y * (r[5] - r[8] * p1_p2), 0);
    } else {
      ////////////////////
      if (jc) {
        JACOBIAN_SET_JC_BEGIN
        float jfc = jic * (1 + rr1 + rr2);
        float ft_x_pn = jic * ft.x * (p0_p2 * p0_p2 + p1_p2 * p1_p2);
        float4 sc1 = tex1Dfetch(tex_jacobian_sj, proj.x);
        jc[jc_pos] = make_float4(p0_p2 * jfc * sc1.x, f_p2_x * sc1.y, 0,
                                 -f_p2_x * p0_p2 * sc1.w);
        jc[jc_pos + 2] = make_float4(p1_p2 * jfc * sc1.x, 0, f_p2_y * sc1.z,
                                     -f_p2_y * p1_p2 * sc1.w);

        float4 sc2 = tex1Dfetch(tex_jacobian_sj, proj.x + 1);
        jc[jc_pos + 1] = make_float4(
            -sc2.x * f_p2_x * p0_p2 * y0, sc2.y * f_p2_x * (z0 + x0 * p0_p2),
            -sc2.z * f_p2_x * y0, ft_x_pn * p0_p2 * sc2.w);
        jc[jc_pos + 3] = make_float4(
            -sc2.x * f_p2_y * (z0 + y0 * p1_p2), sc2.y * f_p2_y * x0 * p1_p2,
            sc2.z * f_p2_y * x0, ft_x_pn * p1_p2 * sc2.w);
        JFRT_SET_JC_END
      }

      float4 sc3 = tex1Dfetch(tex_jacobian_sj, proj.y + ptx);
      jp[(tidx << 1)] = make_float4(sc3.x * f_p2_x * (r[0] - r[6] * p0_p2),
                                    sc3.y * f_p2_x * (r[1] - r[7] * p0_p2),
                                    sc3.z * f_p2_x * (r[2] - r[8] * p0_p2), 0);
      jp[(tidx << 1) + 1] =
          make_float4(sc3.x * f_p2_y * (r[3] - r[6] * p1_p2),
                      sc3.y * f_p2_y * (r[4] - r[7] * p1_p2),
                      sc3.z * f_p2_y * (r[5] - r[8] * p1_p2), 0);
    }
  } else if (md) {
    if (scaling == false) {
      if (jc) {
        JACOBIAN_SET_JC_BEGIN
        float2 ms = tex1Dfetch(tex_jacobian_meas, tidx);
        float msn = (ms.x * ms.x + ms.y * ms.y) * jic;
        jc[jc_pos] = make_float4(p0_p2 * jic, f_p2, 0, -f_p2 * p0_p2);
        jc[jc_pos + 1] =
            make_float4(-f_p2 * p0_p2 * y0, f_p2 * (z0 + x0 * p0_p2),
                        -f_p2 * y0, -ms.x * msn);
        jc[jc_pos + 2] = make_float4(p1_p2 * jic, 0, f_p2, -f_p2 * p1_p2);
        jc[jc_pos + 3] = make_float4(-f_p2 * (z0 + y0 * p1_p2),
                                     f_p2 * x0 * p1_p2, f_p2 * x0, -ms.y * msn);
        JFRT_SET_JC_END
      }
      ////////////////////
      jp[(tidx << 1)] = make_float4(f_p2 * (r[0] - r[6] * p0_p2),
                                    f_p2 * (r[1] - r[7] * p0_p2),
                                    f_p2 * (r[2] - r[8] * p0_p2), 0);
      jp[(tidx << 1) + 1] = make_float4(f_p2 * (r[3] - r[6] * p1_p2),
                                        f_p2 * (r[4] - r[7] * p1_p2),
                                        f_p2 * (r[5] - r[8] * p1_p2), 0);
    } else {
      if (jc) {
        JACOBIAN_SET_JC_BEGIN
        float4 sc1 = tex1Dfetch(tex_jacobian_sj, proj.x);
        jc[jc_pos] = make_float4(p0_p2 * jic * sc1.x, f_p2 * sc1.y, 0,
                                 -f_p2 * p0_p2 * sc1.w);
        jc[jc_pos + 2] = make_float4(p1_p2 * jic * sc1.x, 0, f_p2 * sc1.z,
                                     -f_p2 * p1_p2 * sc1.w);

        float4 sc2 = tex1Dfetch(tex_jacobian_sj, proj.x + 1);
        float2 ms = tex1Dfetch(tex_jacobian_meas, tidx);
        float msn = (ms.x * ms.x + ms.y * ms.y) * jic;
        jc[jc_pos + 1] = make_float4(-sc2.x * f_p2 * p0_p2 * y0,
                                     sc2.y * f_p2 * (z0 + x0 * p0_p2),
                                     -sc2.z * f_p2 * y0, -msn * ms.x * sc2.w);
        jc[jc_pos + 3] = make_float4(-sc2.x * f_p2 * (z0 + y0 * p1_p2),
                                     sc2.y * f_p2 * x0 * p1_p2,
                                     sc2.z * f_p2 * x0, -msn * ms.y * sc2.w);
        JFRT_SET_JC_END
      }
      float4 sc3 = tex1Dfetch(tex_jacobian_sj, proj.y + ptx);
      jp[(tidx << 1)] = make_float4(sc3.x * f_p2 * (r[0] - r[6] * p0_p2),
                                    sc3.y * f_p2 * (r[1] - r[7] * p0_p2),
                                    sc3.z * f_p2 * (r[2] - r[8] * p0_p2), 0);
      jp[(tidx << 1) + 1] =
          make_float4(sc3.x * f_p2 * (r[3] - r[6] * p1_p2),
                      sc3.y * f_p2 * (r[4] - r[7] * p1_p2),
                      sc3.z * f_p2 * (r[5] - r[8] * p1_p2), 0);
    }

  } else {
    if (scaling == false) {
      if (jc) {
        JACOBIAN_SET_JC_BEGIN
        jc[jc_pos] = make_float4(p0_p2 * jic, f_p2, 0, -f_p2 * p0_p2);
        jc[jc_pos + 1] = make_float4(-f_p2 * p0_p2 * y0,
                                     f_p2 * (z0 + x0 * p0_p2), -f_p2 * y0, 0);
        jc[jc_pos + 2] = make_float4(p1_p2 * jic, 0, f_p2, -f_p2 * p1_p2);
        jc[jc_pos + 3] = make_float4(-f_p2 * (z0 + y0 * p1_p2),
                                     f_p2 * x0 * p1_p2, f_p2 * x0, 0);
        JFRT_SET_JC_END
      }
      ////////////////////
      jp[(tidx << 1)] = make_float4(f_p2 * (r[0] - r[6] * p0_p2),
                                    f_p2 * (r[1] - r[7] * p0_p2),
                                    f_p2 * (r[2] - r[8] * p0_p2), 0);
      jp[(tidx << 1) + 1] = make_float4(f_p2 * (r[3] - r[6] * p1_p2),
                                        f_p2 * (r[4] - r[7] * p1_p2),
                                        f_p2 * (r[5] - r[8] * p1_p2), 0);
    } else {
      if (jc) {
        JACOBIAN_SET_JC_BEGIN
        float4 sc1 = tex1Dfetch(tex_jacobian_sj, proj.x);
        jc[jc_pos] = make_float4(p0_p2 * jic * sc1.x, f_p2 * sc1.y, 0,
                                 -f_p2 * p0_p2 * sc1.w);
        jc[jc_pos + 2] = make_float4(p1_p2 * jic * sc1.x, 0, f_p2 * sc1.z,
                                     -f_p2 * p1_p2 * sc1.w);
        float4 sc2 = tex1Dfetch(tex_jacobian_sj, proj.x + 1);
        jc[jc_pos + 1] = make_float4(-sc2.x * f_p2 * p0_p2 * y0,
                                     sc2.y * f_p2 * (z0 + x0 * p0_p2),
                                     -sc2.z * f_p2 * y0, 0);
        jc[jc_pos + 3] =
            make_float4(-sc2.x * f_p2 * (z0 + y0 * p1_p2),
                        sc2.y * f_p2 * x0 * p1_p2, sc2.z * f_p2 * x0, 0);
        JFRT_SET_JC_END
      }

      float4 sc3 = tex1Dfetch(tex_jacobian_sj, proj.y + ptx);
      jp[(tidx << 1)] = make_float4(sc3.x * f_p2 * (r[0] - r[6] * p0_p2),
                                    sc3.y * f_p2 * (r[1] - r[7] * p0_p2),
                                    sc3.z * f_p2 * (r[2] - r[8] * p0_p2), 0);
      jp[(tidx << 1) + 1] =
          make_float4(sc3.x * f_p2 * (r[3] - r[6] * p1_p2),
                      sc3.y * f_p2 * (r[4] - r[7] * p1_p2),
                      sc3.z * f_p2 * (r[5] - r[8] * p1_p2), 0);
    }
  }
}

/////////////////////////////////
void ProgramCU::ComputeJacobian(CuTexImage& camera, CuTexImage& point,
                                CuTexImage& jc, CuTexImage& jp,
                                CuTexImage& proj_map, CuTexImage& sj,
                                CuTexImage& meas, CuTexImage& cmlist,
                                bool intrinsic_fixed, int radial_distortion,
                                bool shuffle) {
  float jfc = intrinsic_fixed ? 0.0f : 1.0f;
  unsigned int len = proj_map.GetImgWidth();
  unsigned int bsize = JACOBIAN_FRT_KWIDTH;
  unsigned int nblock = (len + bsize - 1) / bsize;
  unsigned int bw, bh;
  GetBlockConfiguration(nblock, bw, bh);
  dim3 grid(bw, bh), block(bsize);

  camera.BindTexture(tex_jacobian_cam);
  point.BindTexture(tex_jacobian_pts);
  proj_map.BindTexture(tex_jacobian_idx);

  if (!jc.IsValid()) shuffle = false;
  if (shuffle) cmlist.BindTexture(tex_jacobian_shuffle);
  if (sj.IsValid()) sj.BindTexture(tex_jacobian_sj);

  if (radial_distortion == -1) {
    meas.BindTexture(tex_jacobian_meas);
    if (sj.IsValid()) {
      if (shuffle)
        jacobian_frt_kernel<true, false, true, true><<<grid, block>>>(
            (float4*)jc.data(), (float4*)jp.data(), len,
            camera.GetImgWidth() * 2, bw * bsize, jfc);
      else
        jacobian_frt_kernel<true, false, true, false><<<grid, block>>>(
            (float4*)jc.data(), (float4*)jp.data(), len,
            camera.GetImgWidth() * 2, bw * bsize, jfc);
    } else {
      if (shuffle)
        jacobian_frt_kernel<true, false, false, true><<<grid, block>>>(
            (float4*)jc.data(), (float4*)jp.data(), len,
            camera.GetImgWidth() * 2, bw * bsize, jfc);
      else
        jacobian_frt_kernel<true, false, false, false><<<grid, block>>>(
            (float4*)jc.data(), (float4*)jp.data(), len,
            camera.GetImgWidth() * 2, bw * bsize, jfc);
    }
  } else if (radial_distortion) {
    if (sj.IsValid()) {
      if (shuffle)
        jacobian_frt_kernel<false, true, true, true><<<grid, block>>>(
            (float4*)jc.data(), (float4*)jp.data(), len,
            camera.GetImgWidth() * 2, bw * bsize, jfc);
      else
        jacobian_frt_kernel<false, true, true, false><<<grid, block>>>(
            (float4*)jc.data(), (float4*)jp.data(), len,
            camera.GetImgWidth() * 2, bw * bsize, jfc);
    } else {
      if (shuffle)
        jacobian_frt_kernel<false, true, false, true><<<grid, block>>>(
            (float4*)jc.data(), (float4*)jp.data(), len,
            camera.GetImgWidth() * 2, bw * bsize, jfc);
      else
        jacobian_frt_kernel<false, true, false, false><<<grid, block>>>(
            (float4*)jc.data(), (float4*)jp.data(), len,
            camera.GetImgWidth() * 2, bw * bsize, jfc);
    }
  } else {
    if (sj.IsValid()) {
      if (shuffle)
        jacobian_frt_kernel<false, false, true, true><<<grid, block>>>(
            (float4*)jc.data(), (float4*)jp.data(), len,
            camera.GetImgWidth() * 2, bw * bsize, jfc);
      else
        jacobian_frt_kernel<false, false, true, false><<<grid, block>>>(
            (float4*)jc.data(), (float4*)jp.data(), len,
            camera.GetImgWidth() * 2, bw * bsize, jfc);
    } else {
      if (shuffle)
        jacobian_frt_kernel<false, false, false, true><<<grid, block>>>(
            (float4*)jc.data(), (float4*)jp.data(), len,
            camera.GetImgWidth() * 2, bw * bsize, jfc);
      else
        jacobian_frt_kernel<false, false, false, false><<<grid, block>>>(
            (float4*)jc.data(), (float4*)jp.data(), len,
            camera.GetImgWidth() * 2, bw * bsize, jfc);
    }
  }

  ProgramCU::CheckErrorCUDA("ComputeJacobian");
}

texture<float4, 1, cudaReadModeElementType> tex_compact_cam;
__global__ void uncompress_frt_kernel(int ncam, float4* ucam) {
  int tidx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
  if (tidx >= ncam) return;
  int fetch_index = tidx << 1;
  int write_index = IMUL(tidx, 4);
  float4 temp1 = tex1Dfetch(tex_compact_cam, fetch_index);
  ucam[write_index] = temp1;

  float4 temp2 = tex1Dfetch(tex_compact_cam, fetch_index + 1);
  float rx = temp2.x;
  float ry = temp2.y;
  float rz = temp2.z;
  float rx_rx = rx * rx;
  float ry_ry = ry * ry;
  float rz_rz = rz * rz;
  float aa = sqrt(rx_rx + ry_ry + rz_rz);
  float caa, saa;
  sincosf(aa, &saa, &caa);
  float ct = aa == 0.0 ? 0.5 : FDIV2(1.0 - caa, aa * aa);
  float st = aa == 0.0 ? 1 : FDIV2(saa, aa);
  float rz_st = rz * st;
  float rx_st = rx * st;
  float ry_st = ry * st;
  float ry_ry_ct = ry_ry * ct;
  float rx_rx_ct = rx_rx * ct;
  float rz_rz_ct = rz_rz * ct;
  float rx_ry_ct = rx * ry * ct;
  float rz_rx_ct = rz * rx * ct;
  float ry_rz_ct = ry * rz * ct;

  ////////////////////////////////////////////////////////////
  ucam[write_index + 1] =
      make_float4((1.0 - (ry_ry_ct + rz_rz_ct)), (rx_ry_ct - rz_st),
                  (rz_rx_ct + ry_st), (rx_ry_ct + rz_st));

  ucam[write_index + 2] =
      make_float4((1.0 - (rz_rz_ct + rx_rx_ct)), (ry_rz_ct - rx_st),
                  (rz_rx_ct - ry_st), (ry_rz_ct + rx_st));

  ucam[write_index + 3] =
      make_float4((1.0 - (rx_rx_ct + ry_ry_ct)), temp2.w, 0, 0);
}

void ProgramCU::UncompressCamera(int ncam, CuTexImage& camera,
                                 CuTexImage& result) {
  unsigned int len = ncam;
  unsigned int bsize = 64;
  unsigned int nblock = (len + bsize - 1) / bsize;
  dim3 grid(nblock);
  dim3 block(bsize);
  camera.BindTexture(tex_compact_cam);
  uncompress_frt_kernel<<<grid, block>>>(len, (float4*)result.data());
  CheckErrorCUDA("UncompressCamera");
}

texture<float4, 1, cudaReadModeElementType> tex_uncompressed_cam;

__global__ void compress_frt_kernel(int ncam, float4* zcam) {
  int tidx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
  if (tidx >= ncam) return;
  int fetch_index = tidx << 2;
  int write_index = tidx << 1;
  float4 temp1 = tex1Dfetch(tex_compact_cam, fetch_index);
  zcam[write_index] = temp1;

  float4 r1 = tex1Dfetch(tex_compact_cam, fetch_index + 1);
  float4 r2 = tex1Dfetch(tex_compact_cam, fetch_index + 2);
  float4 r3 = tex1Dfetch(tex_compact_cam, fetch_index + 3);

  float a = (r1.x + r2.x + r3.x - 1.0) / 2.0;
  if (a >= 1.0) {
    zcam[write_index + 1] = make_float4(0, 0, 0, 0);
  } else {
    float aa = acos(a), b = 0.5 * aa * rsqrt(1 - a * a);
    zcam[write_index + 1] = make_float4(b * (r2.w - r2.y), b * (r1.z - r2.z),
                                        b * (r1.w - r1.y), r3.y);
  }
}

void ProgramCU::CompressCamera(int ncam, CuTexImage& camera0,
                               CuTexImage& result) {
  unsigned int len = ncam;
  unsigned int bsize = 64;
  unsigned int nblock = (len + bsize - 1) / bsize;
  dim3 grid(nblock), block(bsize);
  camera0.BindTexture(tex_uncompressed_cam);
  compress_frt_kernel<<<grid, block>>>(ncam, (float4*)result.data());
  CheckErrorCUDA("CompressCamera");
}

__device__ inline void uncompress_rodrigues_rotation(float rx, float ry,
                                                     float rz, float* r) {
  float rx_rx = rx * rx;
  float ry_ry = ry * ry;
  float rz_rz = rz * rz;
  float aa = sqrt(rx_rx + ry_ry + rz_rz);
  float caa, saa;
  sincosf(aa, &saa, &caa);
  float ct = aa == 0.0 ? 0.5 : FDIV2(1.0 - caa, aa * aa);
  float st = aa == 0.0 ? 1 : FDIV2(saa, aa);
  float rz_st = rz * st;
  float rx_st = rx * st;
  float ry_st = ry * st;
  float ry_ry_ct = ry_ry * ct;
  float rx_rx_ct = rx_rx * ct;
  float rz_rz_ct = rz_rz * ct;
  float rx_ry_ct = rx * ry * ct;
  float rz_rx_ct = rz * rx * ct;
  float ry_rz_ct = ry * rz * ct;
  r[0] = (1.0 - (ry_ry_ct + rz_rz_ct));
  r[1] = (rx_ry_ct - rz_st);
  r[2] = (rz_rx_ct + ry_st);
  r[3] = (rx_ry_ct + rz_st);
  r[4] = (1.0 - (rz_rz_ct + rx_rx_ct));
  r[5] = (ry_rz_ct - rx_st);
  r[6] = (rz_rx_ct - ry_st);
  r[7] = (ry_rz_ct + rx_st);
  r[8] = (1.0 - (rx_rx_ct + ry_ry_ct));
}

texture<float4, 1, cudaReadModeElementType> tex_update_cam;
texture<float4, 1, cudaReadModeElementType> tex_update_cam_delta;

__global__ void update_camera_kernel(int ncam, float4* newcam) {
  int tidx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
  if (tidx >= ncam) return;
  int index0 = tidx << 2;
  int index1 = tidx << 1;
  {
    float4 c1 = tex1Dfetch(tex_update_cam, index0);
    float4 d1 = tex1Dfetch(tex_update_cam_delta, index1);
    float4 c2 = make_float4(max(c1.x + d1.x, 1e-10f), c1.y + d1.y, c1.z + d1.z,
                            c1.w + d1.w);
    newcam[index0] = c2;
  }
  {
    float r[9], dr[9];  //, nr[9];
    float4 r1 = tex1Dfetch(tex_update_cam, index0 + 1);
    r[0] = r1.x;
    r[1] = r1.y;
    r[2] = r1.z;
    r[3] = r1.w;
    float4 r2 = tex1Dfetch(tex_update_cam, index0 + 2);
    r[4] = r2.x;
    r[5] = r2.y;
    r[6] = r2.z;
    r[7] = r2.w;
    float4 r3 = tex1Dfetch(tex_update_cam, index0 + 3);
    r[8] = r3.x;

    float4 dd = tex1Dfetch(tex_update_cam_delta, index1 + 1);
    uncompress_rodrigues_rotation(dd.x, dd.y, dd.z, dr);

    ///////////////////////////////////////////////
    newcam[index0 + 1] =
        make_float4(dr[0] * r[0] + dr[1] * r[3] + dr[2] * r[6],
                    dr[0] * r[1] + dr[1] * r[4] + dr[2] * r[7],
                    dr[0] * r[2] + dr[1] * r[5] + dr[2] * r[8],
                    dr[3] * r[0] + dr[4] * r[3] + dr[5] * r[6]);
    newcam[index0 + 2] =
        make_float4(dr[3] * r[1] + dr[4] * r[4] + dr[5] * r[7],
                    dr[3] * r[2] + dr[4] * r[5] + dr[5] * r[8],
                    dr[6] * r[0] + dr[7] * r[3] + dr[8] * r[6],
                    dr[6] * r[1] + dr[7] * r[4] + dr[8] * r[7]);
    newcam[index0 + 3] = make_float4(dr[6] * r[2] + dr[7] * r[5] + dr[8] * r[8],
                                     r3.y + dd.w, r3.z, r3.w);
  }
}

void ProgramCU::UpdateCameraPoint(int ncam, CuTexImage& camera,
                                  CuTexImage& point, CuTexImage& delta,
                                  CuTexImage& new_camera, CuTexImage& new_point,
                                  int mode) {
  if (mode != 2) {
    unsigned int len = ncam;
    unsigned int bsize = 64;
    unsigned int nblock = (len + bsize - 1) / bsize;
    dim3 grid(nblock), block(bsize);
    camera.BindTexture(tex_update_cam);
    delta.BindTexture(tex_update_cam_delta);
    update_camera_kernel<<<grid, block>>>(len, (float4*)new_camera.data());
    CheckErrorCUDA("UpdateCamera");
  }

  // update the points
  if (mode != 1) {
    CuTexImage dp;
    dp.SetTexture(delta.data() + 8 * ncam, point.GetLength());
    ComputeSAXPY(1.0f, dp, point, new_point);
    CheckErrorCUDA("UpdatePoint");
  }
}

#define PROJECTION_FRT_KWIDTH 64

texture<float4, 1, cudaReadModeElementType> tex_projection_cam;
texture<int2, 1, cudaReadModeElementType> tex_projection_idx;
texture<float4, 1, cudaReadModeElementType> tex_projection_pts;
texture<float2, 1, cudaReadModeElementType> tex_projection_mea;

// run 32/64/128 projections in a block
template <bool md, bool pd>
__global__ void projection_frt_kernel(int nproj, int rowsz, float2* pj) {
  ////////////////////////////////
  int tidx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * rowsz;
  if (tidx >= nproj) return;
  float f, m[3], t[3];  // r[9],
  __shared__ float rr_data[PROJECTION_FRT_KWIDTH * 9];
  float* r = rr_data + IMUL(9, threadIdx.x);
  int2 proj = tex1Dfetch(tex_projection_idx, tidx);
  int cpos = proj.x << 1;
  float4 ft = tex1Dfetch(tex_projection_cam, cpos);
  f = ft.x;
  t[0] = ft.y;
  t[1] = ft.z;
  t[2] = ft.w;
  float4 r1 = tex1Dfetch(tex_projection_cam, cpos + 1);
  r[0] = r1.x;
  r[1] = r1.y;
  r[2] = r1.z;
  r[3] = r1.w;
  float4 r2 = tex1Dfetch(tex_projection_cam, cpos + 2);
  r[4] = r2.x;
  r[5] = r2.y;
  r[6] = r2.z;
  r[7] = r2.w;
  float4 r3 = tex1Dfetch(tex_projection_cam, cpos + 3);
  r[8] = r3.x;

  float4 temp = tex1Dfetch(tex_projection_pts, proj.y);
  m[0] = temp.x;
  m[1] = temp.y;
  m[2] = temp.z;

  float p0 = r[0] * m[0] + r[1] * m[1] + r[2] * m[2] + t[0];
  float p1 = r[3] * m[0] + r[4] * m[1] + r[5] * m[2] + t[1];
  float p2 = r[6] * m[0] + r[7] * m[1] + r[8] * m[2] + t[2];

  if (pd) {
    float rr = 1.0 + r3.y * (p0 * p0 + p1 * p1) / (p2 * p2);
    float f_p2 = FDIV2(f * rr, p2);
    float2 ms = tex1Dfetch(tex_projection_mea, tidx);
    pj[tidx] = make_float2(ms.x - p0 * f_p2, ms.y - p1 * f_p2);
  } else if (md) {
    float f_p2 = FDIV2(f, p2);
    float2 ms = tex1Dfetch(tex_projection_mea, tidx);
    float rd = 1.0 + r3.y * (ms.x * ms.x + ms.y * ms.y);
    pj[tidx] = make_float2(ms.x * rd - p0 * f_p2, ms.y * rd - p1 * f_p2);
  } else {
    float f_p2 = FDIV2(f, p2);
    float2 ms = tex1Dfetch(tex_projection_mea, tidx);
    pj[tidx] = make_float2(ms.x - p0 * f_p2, ms.y - p1 * f_p2);
  }
}

void ProgramCU::ComputeProjection(CuTexImage& camera, CuTexImage& point,
                                  CuTexImage& meas, CuTexImage& proj_map,
                                  CuTexImage& proj, int radial) {
  unsigned int len = proj_map.GetImgWidth();
  unsigned int bsize = PROJECTION_FRT_KWIDTH;
  unsigned int nblock = (len + bsize - 1) / bsize;
  camera.BindTexture(tex_projection_cam);
  point.BindTexture(tex_projection_pts);
  proj_map.BindTexture(tex_projection_idx);
  unsigned int bw, bh;
  GetBlockConfiguration(nblock, bw, bh);
  dim3 grid(bw, bh), block(bsize);
  meas.BindTexture(tex_projection_mea);
  if (radial == -1)
    projection_frt_kernel<true, false><<<grid, block>>>(len, bw * bsize,
                                                        (float2*)proj.data());
  else if (radial)
    projection_frt_kernel<false, true><<<grid, block>>>(len, bw * bsize,
                                                        (float2*)proj.data());
  else
    projection_frt_kernel<false, false><<<grid, block>>>(len, bw * bsize,
                                                         (float2*)proj.data());
  CheckErrorCUDA("ComputeProjection");
}

template <bool md, bool pd>
__global__ void projectionx_frt_kernel(int nproj, int rowsz, float2* pj) {
  ////////////////////////////////
  int tidx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * rowsz;
  if (tidx >= nproj) return;
  float f, m[3], t[3];  // r[9],
  __shared__ float rr_data[PROJECTION_FRT_KWIDTH * 9];
  float* r = rr_data + IMUL(9, threadIdx.x);
  int2 proj = tex1Dfetch(tex_projection_idx, tidx);
  int cpos = proj.x << 1;
  float4 ft = tex1Dfetch(tex_projection_cam, cpos);
  f = ft.x;
  t[0] = ft.y;
  t[1] = ft.z;
  t[2] = ft.w;
  float4 r1 = tex1Dfetch(tex_projection_cam, cpos + 1);
  r[0] = r1.x;
  r[1] = r1.y;
  r[2] = r1.z;
  r[3] = r1.w;
  float4 r2 = tex1Dfetch(tex_projection_cam, cpos + 2);
  r[4] = r2.x;
  r[5] = r2.y;
  r[6] = r2.z;
  r[7] = r2.w;
  float4 r3 = tex1Dfetch(tex_projection_cam, cpos + 3);
  r[8] = r3.x;

  float4 temp = tex1Dfetch(tex_projection_pts, proj.y);
  m[0] = temp.x;
  m[1] = temp.y;
  m[2] = temp.z;

  float p0 = r[0] * m[0] + r[1] * m[1] + r[2] * m[2] + t[0];
  float p1 = r[3] * m[0] + r[4] * m[1] + r[5] * m[2] + t[1];
  float p2 = r[6] * m[0] + r[7] * m[1] + r[8] * m[2] + t[2];
  if (pd) {
    float rr = 1.0 + r3.y * (p0 * p0 + p1 * p1) / (p2 * p2);
    float f_p2 = FDIV2(f, p2);
    float2 ms = tex1Dfetch(tex_projection_mea, tidx);
    pj[tidx] = make_float2(ms.x / rr - p0 * f_p2, ms.y / rr - p1 * f_p2);
  } else if (md) {
    float f_p2 = FDIV2(f, p2);
    float2 ms = tex1Dfetch(tex_projection_mea, tidx);
    float rd = 1.0 + r3.y * (ms.x * ms.x + ms.y * ms.y);
    pj[tidx] = make_float2(ms.x - p0 * f_p2 / rd, ms.y - p1 * f_p2 / rd);
  } else {
    float f_p2 = FDIV2(f, p2);
    float2 ms = tex1Dfetch(tex_projection_mea, tidx);
    pj[tidx] = make_float2(ms.x - p0 * f_p2, ms.y - p1 * f_p2);
  }
}

void ProgramCU::ComputeProjectionX(CuTexImage& camera, CuTexImage& point,
                                   CuTexImage& meas, CuTexImage& proj_map,
                                   CuTexImage& proj, int radial) {
  unsigned int len = proj_map.GetImgWidth();
  unsigned int bsize = PROJECTION_FRT_KWIDTH;
  unsigned int nblock = (len + bsize - 1) / bsize;
  camera.BindTexture(tex_projection_cam);
  point.BindTexture(tex_projection_pts);
  proj_map.BindTexture(tex_projection_idx);
  unsigned int bw, bh;
  GetBlockConfiguration(nblock, bw, bh);
  dim3 grid(bw, bh), block(bsize);
  meas.BindTexture(tex_projection_mea);
  if (radial == -1)
    projectionx_frt_kernel<true, false><<<grid, block>>>(len, bw * bsize,
                                                         (float2*)proj.data());
  else if (radial)
    projectionx_frt_kernel<false, true><<<grid, block>>>(len, bw * bsize,
                                                         (float2*)proj.data());
  else
    projectionx_frt_kernel<false, false><<<grid, block>>>(len, bw * bsize,
                                                          (float2*)proj.data());
  CheckErrorCUDA("ComputeProjection");
}

texture<float2, 1, cudaReadModeElementType> tex_jte_pe;
texture<float, 1, cudaReadModeElementType> tex_jte_pex;
texture<float4, 1, cudaReadModeElementType> tex_jte_jc;
texture<float4, 1, cudaReadModeElementType> tex_jte_jc2;
texture<int, 1, cudaReadModeElementType> tex_jte_cmp;
texture<int, 1, cudaReadModeElementType> tex_jte_cmt;
texture<float4, 1, cudaReadModeElementType> tex_jte_jc3;
texture<float4, 1, cudaReadModeElementType> tex_jte_jc4;

__global__ void jte_cam_kernel(int num, float* jc, float* jte) {
  __shared__ float value[128];

  // 8thread per camera
  int col = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
  if (col >= num) return;

  int cam = col >> 4;  // 8 thread per camera

  // read data range for this camera, 8 thread will do the same thing
  int idx1 = tex1Dfetch(tex_jte_cmp, cam) << 4;  // first camera
  int idx2 = tex1Dfetch(tex_jte_cmp, cam + 1) << 4;  // last camera + 1

  ///////////////////////////////
  int offset = threadIdx.x & 0xf;  // which parameter of this camera
  int part = offset >= 8 ? 1 : 0;
  /////////////////////////////

  float result = 0;
  // loop to read the index of the projection.
  // so to get the location to read the jacobian
  for (int i = idx1 + offset; i < idx2; i += 16) {
    float temp = jc[i];
    // every 8 thread will read the same position.
    int index = tex1Dfetch(tex_jte_cmt, i >> 4);
    float v = tex1Dfetch(tex_jte_pex, (index << 1) + part);
    //////////////////////
    result += temp * v;
  }
  value[threadIdx.x] = result;
  // write back
  if (offset < 8) jte[(cam << 3) + offset] = (result + value[threadIdx.x + 8]);
}

template <int KH, int TEXN>
__global__ void jte_cam_vec_kernel(int num, float* jte) {
  __shared__ float value[KH * 128];
  int cam = blockIdx.x * KH + threadIdx.y;
  if (cam >= num) return;

  // read data range for this camera
  // 8 thread will do the same thing
  int idx1 = tex1Dfetch(tex_jte_cmp, cam) << 2;  // first camera
  int idx2 = tex1Dfetch(tex_jte_cmp, cam + 1) << 2;  // last camera + 1
  int part = (threadIdx.x & 0x02) ? 1 : 0;

  float rx = 0, ry = 0, rz = 0, rw = 0;
  // loop to read the index of the projection.
  // so to get the location to read the jacobian
  for (int i = idx1 + threadIdx.x; i < idx2; i += 32) {
    float4 temp;
    if (TEXN == 1) {
      temp = tex1Dfetch(tex_jte_jc, i);
    }
    if (TEXN == 2) {
      int texid = i >> 25;
      if (texid == 0)
        temp = tex1Dfetch(tex_jte_jc, i);
      else
        temp = tex1Dfetch(tex_jte_jc2, (i & 0x1ffffff));
    }
    if (TEXN == 4) {
      int index = tex1Dfetch(tex_jte_cmt, i >> 2);
      int iii = (index << 2) + (i & 0x3);
      int texid = iii >> 25;
      /////////////////////////////////
      if (texid == 0)
        temp = tex1Dfetch(tex_jte_jc, iii);
      else if (texid == 1)
        temp = tex1Dfetch(tex_jte_jc2, (iii & 0x1ffffff));
      else if (texid == 2)
        temp = tex1Dfetch(tex_jte_jc3, (iii & 0x1ffffff));
      else
        temp = tex1Dfetch(tex_jte_jc4, (iii & 0x1ffffff));
    }
    int index = tex1Dfetch(tex_jte_cmt, i >> 2);
    float vv = tex1Dfetch(tex_jte_pex, (index << 1) + part);
    rx += temp.x * vv;
    ry += temp.y * vv;
    rz += temp.z * vv;
    rw += temp.w * vv;
  }
  ////////////////////////////////////
  int widx = (threadIdx.y << 7) + (threadIdx.x << 2);
  ///////////////////////////////////
  // write back
  value[widx] = rx;
  value[widx + 1] = ry;
  value[widx + 2] = rz;
  value[widx + 3] = rw;
  ////////////////////////////////////
  int ridx = (threadIdx.y << 7) + threadIdx.x;
  value[ridx] = ((value[ridx] + value[ridx + 32]) +
                 (value[ridx + 64] + value[ridx + 96]));
  if (threadIdx.x < 16) value[ridx] += value[ridx + 16];
  if (threadIdx.x < 8)
    jte[(cam << 3) + threadIdx.x] = value[ridx] + value[ridx + 8];
}

template <int KH, bool JT>
__global__ void jte_cam_vec32_kernel(int num, float* jc, float* jte) {
  __shared__ float value[KH * 32];
  int cam = blockIdx.x * KH + threadIdx.y;
  if (cam >= num) return;
  float sum = 0;
  int rowpos = (threadIdx.y << 5);
  int index = threadIdx.x + rowpos;
  int xypart = (threadIdx.x & 0x08) ? 1 : 0;
  int part2 = threadIdx.x & 0xf;
  // read data range for this camera
  // 8 thread will do the same thing
  int idx1 = tex1Dfetch(tex_jte_cmp, cam) << 4;  // first camera
  int idx2 = tex1Dfetch(tex_jte_cmp, cam + 1) << 4;  // last camera + 1

  // loop to read the index of the projection.
  // so to get the location to read the jacobian
  for (int i = idx1 + threadIdx.x; i < idx2; i += 32) {
    int index = tex1Dfetch(tex_jte_cmt, i >> 4);
    float temp;
    if (JT)
      temp = jc[i];
    else
      temp = jc[(index << 4) + part2];

    float v = tex1Dfetch(tex_jte_pex, (index << 1) + xypart);
    sum += temp * v;
  }
  value[index] = sum;

  if (threadIdx.x < 16) value[index] += value[index + 16];
  if (threadIdx.x < 8)
    jte[(cam << 3) + threadIdx.x] = value[index] + value[index + 8];
}

/////////////////////////////////////////////////////////////
texture<float4, 1, cudaReadModeElementType> tex_jte_jp;
texture<int, 1, cudaReadModeElementType> tex_jte_pmp;
texture<float4, 1, cudaReadModeElementType> tex_jte_jp2;

__global__ void jte_point_kernel(int num, float4* jte) {
  ////////////////////////////
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num) return;

  int idx1 = tex1Dfetch(tex_jte_pmp, index);  // first camera
  int idx2 = tex1Dfetch(tex_jte_pmp, index + 1);  // last camera + 1
  float4 result = make_float4(0, 0, 0, 0);
  for (int i = idx1; i < idx2; ++i) {
    // error vector
    float2 ev = tex1Dfetch(tex_jte_pe, i);

    float4 j1 = tex1Dfetch(tex_jte_jp, i << 1);
    result.x += j1.x * ev.x;
    result.y += j1.y * ev.x;
    result.z += j1.z * ev.x;

    float4 j2 = tex1Dfetch(tex_jte_jp, 1 + (i << 1));
    result.x += j2.x * ev.y;
    result.y += j2.y * ev.y;
    result.z += j2.z * ev.y;
  }
  jte[index] = result;
}

////////////////////
// faster but not always more accurate
//#define JTE_POINT_VEC2

template <int KH, int TEXN>
__global__ void jte_point_vec_kernel(int num, int rowsz, float* jte) {
  ////////////////////////////
  __shared__ float value[KH * 128];
  int index = blockIdx.x * KH + threadIdx.y + blockIdx.y * rowsz;
  if (index >= num) return;
#ifdef JTE_POINT_VEC2
  int idx1 = tex1Dfetch(tex_jte_pmp, index);  // first
  int idx2 = tex1Dfetch(tex_jte_pmp, index + 1);  // last  + 1
#else
  int idx1 = tex1Dfetch(tex_jte_pmp, index) << 1;  // first
  int idx2 = tex1Dfetch(tex_jte_pmp, index + 1) << 1;  // last  + 1
#endif
  float rx = 0, ry = 0, rz = 0;
  for (int i = idx1 + threadIdx.x; i < idx2; i += 32) {
    if (TEXN == 2 && i >> 25) {
#ifdef JTE_POINT_VEC2

      float2 vv = tex1Dfetch(tex_jte_pe, i);
      float4 jp1 = tex1Dfetch(tex_jte_jp, ((i & 0x1ffffff) << 1));
      float4 jp2 = tex1Dfetch(tex_jte_jp, ((i & 0x1ffffff) << 1) + 1);
      rx += (jp1.x * vv.x + jp2.x * vv.y);
      ry += (jp1.y * vv.x + jp2.y * vv.y);
      rz += (jp1.z * vv.x + jp2.z * vv.y);
#else
      float vv = tex1Dfetch(tex_jte_pex, i);
      float4 jpi = tex1Dfetch(tex_jte_jp2, i & 0x1ffffff);
      rx += jpi.x * vv;
      ry += jpi.y * vv;
      rz += jpi.z * vv;
#endif
    } else {
#ifdef JTE_POINT_VEC2
      float2 vv = tex1Dfetch(tex_jte_pe, i);
      float4 jp1 = tex1Dfetch(tex_jte_jp, (i << 1));
      float4 jp2 = tex1Dfetch(tex_jte_jp, (i << 1) + 1);
      rx += (jp1.x * vv.x + jp2.x * vv.y);
      ry += (jp1.y * vv.x + jp2.y * vv.y);
      rz += (jp1.z * vv.x + jp2.z * vv.y);
#else
      float vv = tex1Dfetch(tex_jte_pex, i);
      float4 jpi = tex1Dfetch(tex_jte_jp, i);
      rx += jpi.x * vv;
      ry += jpi.y * vv;
      rz += jpi.z * vv;
#endif
    }
  }

  int rowp = threadIdx.y << 7;
  int loc = (threadIdx.x << 2) + rowp;
  value[loc] = rx;
  value[loc + 1] = ry;
  value[loc + 2] = rz;
  value[loc + 3] = 0;

  int ridx = threadIdx.x + rowp;
  value[ridx] = ((value[ridx] + value[ridx + 32]) +
                 (value[ridx + 64] + value[ridx + 96]));
  if (threadIdx.x < 16) value[ridx] += value[ridx + 16];
  if (threadIdx.x < 8) value[ridx] += value[ridx + 8];
  if (threadIdx.x < 4)
    jte[(index << 2) + threadIdx.x] = value[ridx] + value[ridx + 4];
}

#define JTE_CAMERA_VEC
#define JTE_POINT_VEC

void ProgramCU::ComputeJtE(CuTexImage& pe, CuTexImage& jc, CuTexImage& cmap,
                           CuTexImage& cmlist, CuTexImage& jp, CuTexImage& pmap,
                           CuTexImage& jte, bool jc_transpose, int mode) {
  //////////////////////////////////////////////////////////
  int ncam = int(cmap.GetImgWidth() - 1);  // how many cameras
  size_t szjc = jc.GetDataSize();

  //////////////////////////////
  cmap.BindTexture(tex_jte_cmp);
  cmlist.BindTexture(tex_jte_cmt);
#ifdef JTE_CAMERA_VEC2
  pe.BindTexture(tex_jte_pex);
  const unsigned int bheight = 2;
  dim3 block1(32, bheight), grid1((ncam + bheight - 1) / bheight);
  if (mode == 2) {
  } else if (jc_transpose)
    jte_cam_vec32_kernel<bheight, true><<<grid1, block1>>>(ncam, jc.data(),
                                                           jte.data());
  else
    jte_cam_vec32_kernel<bheight, false><<<grid1, block1>>>(ncam, jc.data(),
                                                            jte.data());

#elif defined(JTE_CAMERA_VEC)
  pe.BindTexture(tex_jte_pex);
  const unsigned int bheight = 2;
  unsigned int len1 = ncam * 32;
  unsigned int bsize1 = 32 * bheight;
  unsigned int nblock1 = (len1 + bsize1 - 1) / bsize1;
  dim3 grid1(nblock1);
  dim3 block1(32, bheight);
  if (mode == 2) {
    // skip camera
  } else if (szjc > 2 * MAX_TEXSIZE || !jc_transpose) {
    if (jc_transpose)
      jte_cam_vec32_kernel<bheight, true><<<grid1, block1>>>(ncam, jc.data(),
                                                             jte.data());
    else
      jte_cam_vec32_kernel<bheight, false><<<grid1, block1>>>(ncam, jc.data(),
                                                              jte.data());
  } else if (szjc > MAX_TEXSIZE) {
    jc.BindTexture2(tex_jte_jc, tex_jte_jc2);
    jte_cam_vec_kernel<bheight, 2><<<grid1, block1>>>(ncam, jte.data());
  } else {
    jc.BindTexture(tex_jte_jc);
    jte_cam_vec_kernel<bheight, 1><<<grid1, block1>>>(ncam, jte.data());
  }
#else
  pe.BindTexture(tex_jte_pex);
  unsigned int len1 = ncam * 16;
  unsigned int bsize1 = len1 > 32 * 128 ? 128 : (len1 > 32 * 64 ? 64 : 32);
  unsigned int nblock1 = (len1 + bsize1 - 1) / bsize1;
  dim3 grid1(nblock1), block1(bsize1);
  jte_cam_kernel<<<grid1, block1>>>(len1, jc.data(), jte.data());
#endif
  CheckErrorCUDA("ComputeJtE<Camera>");

  ////////////////////////////////////////////
  pmap.BindTexture(tex_jte_pmp);
  unsigned int npoint = (pmap.GetImgWidth() - 1);
#ifndef JTE_POINT_VEC
  size_t len2 = npoint;
  unsigned int bsize2 = 64;
  unsigned int nblock2 = (len2 + bsize2 - 1) / bsize2;
  dim3 grid2(nblock2), block2(bsize2);
  pe.BindTexture(tex_jte_pe);
  jp.BindTexture(tex_jte_jp);
  jte_point_kernel<<<grid2, block2>>>(len2, ((float4*)jte.data()) + 2 * ncam);
#else

#ifdef JTE_POINT_VEC2
  pe.BindTexture(tex_jte_pe);
#else
  pe.BindTexture(tex_jte_pex);
#endif
  const unsigned int bheight2 = 2;
  unsigned int bsize2 = 32;
  unsigned int nblock2 = (unsigned int)((npoint + bheight2 - 1) / bheight2);
  unsigned int offsetv = 8 * ncam;
  unsigned int bw, bh;
  GetBlockConfiguration(nblock2, bw, bh);
  dim3 grid2(bw, bh), block2(bsize2, bheight2);
  if (mode == 1) {
    // skip point
  } else if (jp.GetDataSize() > MAX_TEXSIZE) {
    jp.BindTexture2(tex_jte_jp, tex_jte_jp2);
    jte_point_vec_kernel<bheight2, 2><<<grid2, block2>>>(
        npoint, bw * bheight2, ((float*)jte.data()) + offsetv);
  } else {
    jp.BindTexture(tex_jte_jp);
    jte_point_vec_kernel<bheight2, 1><<<grid2, block2>>>(
        npoint, bw * bheight2, ((float*)jte.data()) + offsetv);
  }
#endif
  CheckErrorCUDA("ComputeJtE<Point>");
}

texture<int, 1, cudaReadModeElementType> tex_jtjd_cmp;
texture<int, 1, cudaReadModeElementType> tex_jtjd_cmlist;

template <int VN, int KH, bool JT>
__global__ void jtjd_cam_vec32_kernel(int num, int add_existing_dq, float* jc,
                                      float* jtjd, float* jtjdi) {
  __shared__ float value[KH * 32];

  // 8thread per camera
  int cam = blockIdx.x * KH + threadIdx.y;
  int part = threadIdx.x & 0x7;  // which parameter of this camera
  int part2 = threadIdx.x & 0xf;
  int campos = threadIdx.y << 5;
  int index = threadIdx.x + campos;
  float sum = 0;
  if (cam < num && part < VN) {
    // read data range for this camera
    // 8 thread will do the same thing
    int idx1 = tex1Dfetch(tex_jtjd_cmp, cam) << 4;  // first camera
    int idx2 = tex1Dfetch(tex_jtjd_cmp, cam + 1) << 4;  // last camera + 1

    // loop to read the index of the projection.
    // so to get the location to read the jacobian
    for (int i = idx1 + threadIdx.x; i < idx2; i += 32) {
      if (JT) {
        float temp = jc[i];
        sum += temp * temp;
      } else {
        int ii = tex1Dfetch(tex_jtjd_cmlist, i >> 4) << 4;
        float temp = jc[ii + part2];
        sum += temp * temp;
      }
    }
  }
  __syncthreads();

  if (cam >= num) return;
  // save all the results?
  value[index] = sum;
  if (threadIdx.x < 16) value[index] += value[index + 16];
  if (threadIdx.x < 8)

    // write back
    if (threadIdx.x < 8) {
      float temp = value[index] + value[index + 8];
      int wpos = threadIdx.x + (cam << 3);
      if (add_existing_dq) temp += jtjd[wpos];
      jtjd[wpos] = temp;
      jtjdi[wpos] = temp == 0 ? 0 : 1 / (temp);
    }
}

texture<float4, 1, cudaReadModeElementType> tex_jtjd_jp;
texture<int, 1, cudaReadModeElementType> tex_jtjd_pmp;
texture<float4, 1, cudaReadModeElementType> tex_jtjd_jp2;

#define JTJD_POINT_KWIDTH 64

template <int TEXN>
__global__ void jtjd_point_kernel(int num, int rowsz, float4* jtjd,
                                  float4* jtjdi) {
  ////////////////////////////
  int index = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * rowsz;
  if (index >= num) return;

  int idx1 = tex1Dfetch(tex_jtjd_pmp, index);  // first camera
  int idx2 = tex1Dfetch(tex_jtjd_pmp, index + 1);  // last camera + 1
  float rx = 0, ry = 0, rz = 0;
  for (int i = idx1; i < idx2; ++i) {
    if (TEXN == 2 && i > 0xffffff) {
      float4 j1 = tex1Dfetch(tex_jtjd_jp2, (i & 0xffffff) << 1);
      rx += j1.x * j1.x;
      ry += j1.y * j1.y;
      rz += j1.z * j1.z;

      float4 j2 = tex1Dfetch(tex_jtjd_jp2, 1 + ((i & 0xffffff) << 1));
      rx += j2.x * j2.x;
      ry += j2.y * j2.y;
      rz += j2.z * j2.z;
    } else {
      float4 j1 = tex1Dfetch(tex_jtjd_jp, i << 1);
      rx += j1.x * j1.x;
      ry += j1.y * j1.y;
      rz += j1.z * j1.z;

      float4 j2 = tex1Dfetch(tex_jtjd_jp, 1 + (i << 1));
      rx += j2.x * j2.x;
      ry += j2.y * j2.y;
      rz += j2.z * j2.z;
    }
  }

  if (jtjd) jtjd[index] = make_float4(rx, ry, rz, 0.0f);
  jtjdi[index] = make_float4(1.0f / rx, 1.0f / ry, 1.0f / rz, 0.0f);
}

void ProgramCU::ComputeDiagonal(CuTexImage& jc, CuTexImage& cmap,
                                CuTexImage& jp, CuTexImage& pmap,
                                CuTexImage& cmlist, CuTexImage& jtjd,
                                CuTexImage& jtjdi, bool jc_transpose,
                                int radial, bool add_existing_diagc) {
  //////////////////////////////////////////////////////////
  size_t szjc = jc.GetDataSize();
  unsigned int ncam = (cmap.GetImgWidth() - 1);  // how many cameras

  const unsigned int bheight = 2;
  dim3 block1x(32, bheight), grid1x((ncam + bheight - 1) / bheight);
  cmap.BindTexture(tex_jtjd_cmp);
  if (jc_transpose) {
    if (radial)
      jtjd_cam_vec32_kernel<8, bheight, true><<<grid1x, block1x>>>(
          ncam, add_existing_diagc, jc.data(), jtjd.data(), jtjdi.data());
    else
      jtjd_cam_vec32_kernel<7, bheight, true><<<grid1x, block1x>>>(
          ncam, add_existing_diagc, jc.data(), jtjd.data(), jtjdi.data());
  } else {
    cmlist.BindTexture(tex_jtjd_cmlist);
    if (radial)
      jtjd_cam_vec32_kernel<8, bheight, false><<<grid1x, block1x>>>(
          ncam, add_existing_diagc, jc.data(), jtjd.data(), jtjdi.data());
    else
      jtjd_cam_vec32_kernel<7, bheight, false><<<grid1x, block1x>>>(
          ncam, add_existing_diagc, jc.data(), jtjd.data(), jtjdi.data());
  }
  CheckErrorCUDA("ComputeDiagonal<Camera>");

  ////////////////////////////////////////////
  unsigned int npoint = (pmap.GetImgWidth() - 1);
  unsigned int len2 = npoint;
  unsigned int bsize2 = JTJD_POINT_KWIDTH;
  unsigned int nblock2 = (len2 + bsize2 - 1) / bsize2;
  unsigned int bw, bh;
  GetBlockConfiguration(nblock2, bw, bh);
  dim3 grid2(bw, bh), block2(bsize2);
  pmap.BindTexture(tex_jtjd_pmp);

  if (jp.GetDataSize() > MAX_TEXSIZE) {
    jp.BindTexture2(tex_jtjd_jp, tex_jtjd_jp2);
    jtjd_point_kernel<2><<<grid2, block2>>>(len2, (bw * bsize2),
                                            ((float4*)jtjd.data()) + 2 * ncam,
                                            ((float4*)jtjdi.data()) + 2 * ncam);
  } else {
    jp.BindTexture(tex_jtjd_jp);
    jtjd_point_kernel<1><<<grid2, block2>>>(len2, (bw * bsize2),
                                            ((float4*)jtjd.data()) + 2 * ncam,
                                            ((float4*)jtjdi.data()) + 2 * ncam);
  }
  CheckErrorCUDA("ComputeDiagonal<Point>");
}

// for each
template <bool SJ>
__global__ void jtjd_cam_q_kernel(int num, int rowsz, float* qw, float4* diag) {
  int bindex = IMUL(blockIdx.x, blockDim.x) + rowsz * blockIdx.y;
  int index = bindex + threadIdx.x;
  if (index >= num) return;
  int tid = index & 0x1;
  float w = qw[index], ws = w * w * 2.0f;
  if (SJ) {
    float4 sj = tex1Dfetch(tex_jacobian_sj, index);
    float4 dj = tid == 0 ? make_float4(sj.x * sj.x * ws, 0, 0, 0)
                         : make_float4(0, 0, 0, sj.w * sj.w * ws);
    diag[index] = dj;
  } else {
    float4 dj = tid == 0 ? make_float4(ws, 0, 0, 0) : make_float4(0, 0, 0, ws);
    diag[index] = dj;
  }
}

void ProgramCU::ComputeDiagonalQ(CuTexImage& qlistw, CuTexImage& sj,
                                 CuTexImage& diag) {
  unsigned int bsize = 32;
  unsigned int len = qlistw.GetImgWidth() * 2;
  unsigned int nblock = (len + bsize - 1) / bsize;
  unsigned int bw, bh;
  GetBlockConfiguration(nblock, bw, bh);
  dim3 grid(bw, bh), block(bsize);
  if (sj.IsValid()) {
    sj.BindTexture(tex_jacobian_sj);
    jtjd_cam_q_kernel<true><<<grid, block>>>(len, (bw * bsize), qlistw.data(),
                                             (float4*)diag.data());
  } else {
    jtjd_cam_q_kernel<false><<<grid, block>>>(len, (bw * bsize), qlistw.data(),
                                              (float4*)diag.data());
  }
  CheckErrorCUDA("ComputeDiagonalQ");
}

template <int VN, int KH, bool JT>
__global__ void jtjd_cam_block_vec32_kernel(int num, float lambda1,
                                            float lambda2, float* jc,
                                            float* diag, float* blocks,
                                            bool add_existing_diagc) {
  __shared__ float value[KH * 32 * VN];

  // 8thread per camera
  int cam = blockIdx.x * KH + threadIdx.y;
  int part = threadIdx.x & 0x7;  // which parameter of this camera
  int part2 = threadIdx.x & 0xf;
  int index = threadIdx.x + (threadIdx.y << 5);
  float row[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (cam < num) {
    int rowpos = index - part;
    // read data range for this camera
    // 8 thread will do the same thing
    int idx1 = tex1Dfetch(tex_jtjd_cmp, cam) << 4;  // first camera
    int idx2 = tex1Dfetch(tex_jtjd_cmp, cam + 1) << 4;  // last camera + 1

    // loop to read the index of the projection.
    // so to get the location to read the jacobian
    for (int i = idx1 + threadIdx.x; i < idx2; i += 32) {
      if (JT) {
        float temp = jc[i];
        value[index] = temp;
        for (int j = 0; j < VN; ++j) row[j] += (temp * value[rowpos + j]);
      } else {
        int ii = tex1Dfetch(tex_jtjd_cmlist, i >> 4) << 4;
        float temp = jc[ii + part2];
        value[index] = temp;
        for (int j = 0; j < VN; ++j) row[j] += (temp * value[rowpos + j]);
      }
    }
  }
  __syncthreads();

  if (cam >= num) return;
  // save all the results?
  for (int i = 0; i < VN; ++i) value[index * VN + i] = row[i];
  int campos = threadIdx.y * (32 * VN);
  for (int i = threadIdx.x; i < (VN * 16); i += 32)
    value[campos + i] += value[campos + i + (16 * VN)];
  for (int i = threadIdx.x; i < (VN * 8); i += 32)
    value[campos + i] += value[campos + i + (8 * VN)];

  if (VN == 7) {
    bool zero = (part >= VN);

    // write back
    if (threadIdx.x < 8) {
      float* dp = value + campos + threadIdx.x * (VN + 1);
      float temp = zero ? 0 : dp[0];
      int didx = threadIdx.x + (cam << 3);
      if (add_existing_diagc) temp += diag[didx];
      diag[didx] = temp;
      dp[0] = lambda1 + lambda2 * temp;
    }
    int wpos = cam * (8 * VN) + threadIdx.x;
    int rpos = campos + threadIdx.x - (threadIdx.x >> 3);
    blocks[wpos] = zero ? 0 : value[rpos];
    if (threadIdx.x < (VN * 8 - 32))
      blocks[wpos + 32] = zero ? 0 : value[rpos + 28];
  } else {
    // write back
    if (threadIdx.x < 8) {
      float* dp = value + campos + threadIdx.x * (VN + 1);
      float temp = dp[0];
      int didx = threadIdx.x + (cam << 3);
      if (add_existing_diagc) temp += diag[didx];
      diag[didx] = temp;
      dp[0] = lambda1 + lambda2 * temp;  // max(, 1e-6) ;
    }
    int wpos = cam * (8 * VN) + threadIdx.x;
    int rpos = campos + threadIdx.x;
    blocks[wpos] = value[rpos];
    blocks[wpos + 32] = value[rpos + 32];
  }
}

#define JTJD_POINT_BLOCK_KWIDTH 64

template <int TEXN>
__global__ void jtjd_point_block_kernel(int num, int rowsz, float lambda1,
                                        float lambda2, float4* diag,
                                        float4* blocks) {
  ////////////////////////////
  int index = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * rowsz;
  if (index >= num) return;

  int idx1 = tex1Dfetch(tex_jtjd_pmp, index);  // first camera
  int idx2 = tex1Dfetch(tex_jtjd_pmp, index + 1);  // last camera + 1

  float M00 = 0, M01 = 0, M02 = 0, M11 = 0, M12 = 0, M22 = 0;
  for (int i = idx1; i < idx2; ++i) {
    if (TEXN == 2 && i > 0xffffff) {
      float4 j1 = tex1Dfetch(tex_jtjd_jp2, (i & 0xffffff) << 1);
      M00 += j1.x * j1.x;
      M01 += j1.x * j1.y;
      M02 += j1.x * j1.z;
      M11 += j1.y * j1.y;
      M12 += j1.y * j1.z;
      M22 += j1.z * j1.z;

      float4 j2 = tex1Dfetch(tex_jtjd_jp2, 1 + ((i & 0xffffff) << 1));
      M00 += j2.x * j2.x;
      M01 += j2.x * j2.y;
      M02 += j2.x * j2.z;
      M11 += j2.y * j2.y;
      M12 += j2.y * j2.z;
      M22 += j2.z * j2.z;
    } else {
      float4 j1 = tex1Dfetch(tex_jtjd_jp, i << 1);
      M00 += j1.x * j1.x;
      M01 += j1.x * j1.y;
      M02 += j1.x * j1.z;
      M11 += j1.y * j1.y;
      M12 += j1.y * j1.z;
      M22 += j1.z * j1.z;

      float4 j2 = tex1Dfetch(tex_jtjd_jp, 1 + (i << 1));
      M00 += j2.x * j2.x;
      M01 += j2.x * j2.y;
      M02 += j2.x * j2.z;
      M11 += j2.y * j2.y;
      M12 += j2.y * j2.z;
      M22 += j2.z * j2.z;
    }
  }

  diag[index] = make_float4(M00, M11, M22, 0);

  M00 = lambda2 * M00 + lambda1;
  M11 = lambda2 * M11 + lambda1;
  M22 = lambda2 * M22 + lambda1;

  // invert the 3x3 matrix.
  float det = (M00 * M11 - M01 * M01) * M22 + 2.0 * M01 * M12 * M02 -
              M02 * M02 * M11 - M12 * M12 * M00;
  if (det >= FLT_MAX || det <= FLT_MIN * 2.0f) {
    int write_pos = index * 3;
    blocks[write_pos] = make_float4(0, 0, 0, 0);
    blocks[write_pos + 1] = make_float4(0, 0, 0, 0);
    blocks[write_pos + 2] = make_float4(0, 0, 0, 0);
  } else {
    float m00 = (M11 * M22 - M12 * M12) / det;
    float m01 = -(M01 * M22 - M12 * M02) / det;
    float m02 = (M01 * M12 - M02 * M11) / det;
    int write_pos = index * 3;
    blocks[write_pos] = make_float4(m00, m01, m02, 0);

    float m11 = (M00 * M22 - M02 * M02) / det;
    float m12 = -(M00 * M12 - M01 * M02) / det;
    blocks[write_pos + 1] = make_float4(m01, m11, m12, 0);

    float m22 = (M00 * M11 - M01 * M01) / det;
    blocks[write_pos + 2] = make_float4(m02, m12, m22, 0);
  }
}

#define JTJD_BLOCK_CAM_INVERT_KWIDTH 64
template <int VN>
__global__ void jtjd_cam_block_invert_kernel(int num, float4* blocks) {
  // N /  8 cameras...each have 64 floats,,,, N * 8 float
  // each will read 8 float......
  __shared__ float value[JTJD_BLOCK_CAM_INVERT_KWIDTH * VN];
  __shared__ bool invalid[JTJD_BLOCK_CAM_INVERT_KWIDTH / 8];
  //////////////////////////////////////////////

  int bindex = IMUL(blockIdx.x, blockDim.x);
  int index = bindex + threadIdx.x;
  int block_read_pos = IMUL(bindex, VN);
  for (int i = 0; i < JTJD_BLOCK_CAM_INVERT_KWIDTH * VN;
       i += JTJD_BLOCK_CAM_INVERT_KWIDTH)
    value[threadIdx.x + i] = ((float*)blocks)[block_read_pos + threadIdx.x + i];
  __syncthreads();
  const int cam_id = threadIdx.x >> 3;
  const int cam_pos = IMUL(cam_id, VN * 8);
  const int col = threadIdx.x & 0x7, rowj_pos = col << 3;
  ;  //

  float* a = value + cam_pos;
  for (int i = 0; i < VN; ++i) {
    int rowi_pos = i << 3, dpos = i + rowi_pos;
    if (col == i && a[dpos] > 0) a[dpos] = rsqrt(a[dpos]);
    __syncthreads();
    float diag = a[dpos];
    if (diag == 0 || col >= VN) continue;
    if (col < i) {
      a[rowi_pos + col] = 0;
    } else if (col > i) {
      float aij = a[rowi_pos + col] * diag;
      a[rowi_pos + col] = aij;
      for (int k = col; k < VN; ++k) a[rowj_pos + k] -= a[rowi_pos + k] * aij;
    }
  }

  if (index >= num) return;

  if (col == 0) invalid[cam_id] = false;
  if (col < VN) {
    for (int i = 1; i < VN; ++i) {
      int rowi_pos = i << 3, dpos = i + rowi_pos;
      if (a[dpos] == 0) continue;
      if (col < i) {
        float sum = 0;
        for (int k = col; k < i; ++k)
          sum += (a[(k << 3) + i] * a[rowj_pos + k]);
        a[rowj_pos + i] = -sum * a[dpos];
      }
    }
    float ai[8], amax = 0;
    for (int i = 0; i < VN * 8; i += 8) {
      float sum = 0;
      for (int k = 0; k < VN; k++) sum += a[rowj_pos + k] * a[i + k];
      ai[i >> 3] = sum;
      amax = max(amax, sum);
    }

    if (isinf(amax)) invalid[cam_id] = true;
    int write_pos = IMUL((index >> 3), (VN * 2)) + (col << 1);
    if (invalid[cam_id])  // a better way would be using a threshold
    {
      blocks[write_pos] = make_float4(0, 0, 0, 0);
      blocks[write_pos + 1] = make_float4(0, 0, 0, 0);
    } else {
      blocks[write_pos] = make_float4(ai[0], ai[1], ai[2], ai[3]);
      blocks[write_pos + 1] =
          make_float4(ai[4], ai[5], ai[6], VN < 8 ? 0 : ai[7]);
    }
  }
}

void ProgramCU::ComputeDiagonalBlock(float lambda, bool dampd, CuTexImage& jc,
                                     CuTexImage& cmap, CuTexImage& jp,
                                     CuTexImage& pmap, CuTexImage& cmlist,
                                     CuTexImage& diag, CuTexImage& blocks,
                                     int radial_distortion, bool jc_transpose,
                                     bool add_existing_diagc, int mode) {
  size_t szjc = jc.GetDataSize();
  unsigned int ncam = (cmap.GetImgWidth() - 1);  // how many cameras
  float lambda1 = dampd ? 0.0f : lambda;
  float lambda2 = dampd ? (1.0f + lambda) : 1.0f;
  const unsigned int bheight = 2;
  dim3 block1x(32, bheight), grid1x((ncam + bheight - 1) / bheight);
  cmap.BindTexture(tex_jtjd_cmp);

  if (mode == 2) {
    // point only mode?
  } else if (radial_distortion) {
    if (jc_transpose) {
      jtjd_cam_block_vec32_kernel<8, bheight, true><<<grid1x, block1x>>>(
          ncam, lambda1, lambda2, jc.data(), diag.data(), blocks.data(),
          add_existing_diagc);
    } else {
      cmlist.BindTexture(tex_jtjd_cmlist);
      jtjd_cam_block_vec32_kernel<8, bheight, false><<<grid1x, block1x>>>(
          ncam, lambda1, lambda2, jc.data(), diag.data(), blocks.data(),
          add_existing_diagc);
    }
  } else {
    if (jc_transpose) {
      jtjd_cam_block_vec32_kernel<7, bheight, true><<<grid1x, block1x>>>(
          ncam, lambda1, lambda2, jc.data(), diag.data(), blocks.data(),
          add_existing_diagc);
    } else {
      cmlist.BindTexture(tex_jtjd_cmlist);
      jtjd_cam_block_vec32_kernel<7, bheight, false><<<grid1x, block1x>>>(
          ncam, lambda1, lambda2, jc.data(), diag.data(), blocks.data(),
          add_existing_diagc);
    }
  }
  CheckErrorCUDA("ComputeDiagonalBlock<Camera>");

  ////////////////////////////////////////////
  unsigned int npoint = (pmap.GetImgWidth() - 1);
  unsigned int len2 = npoint;
  unsigned int bsize2 = JTJD_POINT_BLOCK_KWIDTH;
  unsigned int nblock2 = (len2 + bsize2 - 1) / bsize2;
  unsigned int bw, bh;
  unsigned int offsetd = 2 * ncam;
  unsigned int offsetb = (radial_distortion ? 16 : 14) * ncam;
  GetBlockConfiguration(nblock2, bw, bh);
  dim3 grid2(bw, bh), block2(bsize2);
  pmap.BindTexture(tex_jtjd_pmp);
  if (mode == 1) {
    // camera only mode?
  } else if (jp.GetDataSize() > MAX_TEXSIZE) {
    jp.BindTexture2(tex_jtjd_jp, tex_jtjd_jp2);
    jtjd_point_block_kernel<2><<<grid2, block2>>>(
        len2, (bw * bsize2), lambda1, lambda2, ((float4*)diag.data()) + offsetd,
        ((float4*)blocks.data()) + offsetb);
  } else {
    jp.BindTexture(tex_jtjd_jp);
    jtjd_point_block_kernel<1><<<grid2, block2>>>(
        len2, (bw * bsize2), lambda1, lambda2, ((float4*)diag.data()) + offsetd,
        ((float4*)blocks.data()) + offsetb);
  }
  CheckErrorCUDA("ComputeDiagonalBlock<Point>");

  if (mode != 2) {
    unsigned int len3 = ncam * 8;
    unsigned int bsize3 = JTJD_BLOCK_CAM_INVERT_KWIDTH;
    unsigned int nblock3 = (len3 + bsize3 - 1) / bsize3;
    dim3 grid3(nblock3), block3(bsize3);
    if (radial_distortion)
      jtjd_cam_block_invert_kernel<8><<<grid3, block3>>>(
          len3, (float4*)blocks.data());
    else
      jtjd_cam_block_invert_kernel<7><<<grid3, block3>>>(
          len3, (float4*)blocks.data());
    CheckErrorCUDA("ComputeDiagonalBlockInverse<Camera>");
  }
}

template <int WIDTH, int BBIT, int VSZ>
__global__ void multiply_block_conditioner_kernel(int num, int rowsz,
                                                  float* blocks, float* x,
                                                  float* result) {
  __shared__ float mat[WIDTH * VSZ];
  __shared__ float val[WIDTH];
  const int BSZ = 1 << BBIT;
  const int BMASK = BSZ - 1;
  int bindex = IMUL(blockIdx.x, blockDim.x) + rowsz * blockIdx.y;
  int index = bindex + threadIdx.x;
  int block_read_pos = bindex * VSZ;
  val[threadIdx.x] = x[index];
  for (int i = 0; i < VSZ * WIDTH; i += WIDTH)
    mat[i + threadIdx.x] = blocks[i + block_read_pos + threadIdx.x];
  __syncthreads();
  if (index >= num) return;
  float* ac = mat + (threadIdx.x >> BBIT) * (BSZ * VSZ) + (threadIdx.x & BMASK);
  float* xc = val + (threadIdx.x & (~BMASK));
  float sum = 0;
  for (int i = 0; i < VSZ; ++i) sum += ac[i << BBIT] * xc[i];
  result[index] = sum;  // isinf(sum) ? 0 : sum ; //
}

void ProgramCU::MultiplyBlockConditioner(int ncam, int npoint,
                                         CuTexImage& blocks, CuTexImage& vector,
                                         CuTexImage& result, int radial,
                                         int mode) {
  const unsigned int bsize1 = 64;
  unsigned int bw, bh;

  if (mode != 2) {
    unsigned int len1 = ncam * 8;
    unsigned int nblock1 = (len1 + bsize1 - 1) / bsize1;
    GetBlockConfiguration(nblock1, bw, bh);
    dim3 grid1(bw, bh), block1(bsize1);
    if (radial)
      multiply_block_conditioner_kernel<bsize1, 3, 8><<<grid1, block1>>>(
          len1, (bw * bsize1), blocks.data(), vector.data(), result.data());
    else
      multiply_block_conditioner_kernel<bsize1, 3, 7><<<grid1, block1>>>(
          len1, (bw * bsize1), blocks.data(), vector.data(), result.data());
    CheckErrorCUDA("MultiplyBlockConditioner<Camera>");
  }

  if (mode != 1) {
    const unsigned int bsize2 = 128;
    unsigned int len2 = npoint * 4;
    unsigned int nblock2 = (len2 + bsize2 - 1) / bsize2;
    unsigned int cbsz = radial ? 64 : 56;
    unsigned int offsetb = ncam * cbsz;
    unsigned int offsetd = ncam * 8;
    GetBlockConfiguration(nblock2, bw, bh);
    dim3 grid2(bw, bh), block2(bsize2);
    multiply_block_conditioner_kernel<bsize2, 2, 3><<<grid2, block2>>>(
        len2, (bw * bsize2), blocks.data() + offsetb, vector.data() + offsetd,
        result.data() + offsetd);
    CheckErrorCUDA("MultiplyBlockConditioner<Point>");
  }
}

texture<float4, 1, cudaReadModeElementType> tex_shuffle_jc;
texture<int, 1, cudaReadModeElementType> tex_shuffle_map;
texture<float4, 1, cudaReadModeElementType> tex_shuffle_jc2;
template <int TEXN>
__global__ void shuffle_camera_jacobian_kernel(int num, int bwidth,
                                               float4* jc) {
  int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * bwidth;
  if (index >= num) return;
  int fetch_idx = tex1Dfetch(tex_shuffle_map, index >> 2);
  if (TEXN == 2) {
    int texidx = fetch_idx >> 23,
        fidx = ((fetch_idx & 0x7fffff) << 2) + (index & 0x3);
    if (texidx == 0)
      jc[index] = tex1Dfetch(tex_shuffle_jc, fidx);
    else if (texidx == 1)
      jc[index] = tex1Dfetch(tex_shuffle_jc2, fidx);
  }
  if (TEXN == 1) {
    jc[index] = tex1Dfetch(tex_shuffle_jc, (fetch_idx << 2) + (index & 0x3));
  }
}

bool ProgramCU::ShuffleCameraJacobian(CuTexImage& jc, CuTexImage& map,
                                      CuTexImage& result) {
  if (!result.IsValid()) return false;
  size_t szjc = jc.GetDataSize();
  unsigned int len = map.GetImgWidth() * 4;
  unsigned int bsize = 128;
  unsigned int nblock = (len + bsize - 1) / bsize;

  map.BindTexture(tex_shuffle_map);

  if (szjc > 2 * MAX_TEXSIZE) {
    fprintf(stderr, "datasize way too big %lX, %lX+...\n", szjc,
            (szjc) / MAX_TEXSIZE);
    return false;
  } else if (szjc > MAX_TEXSIZE) {
    unsigned int bw, bh;
    GetBlockConfiguration(nblock, bw, bh);
    dim3 grid(bw, bh), block(bsize);
    jc.BindTexture2(tex_shuffle_jc, tex_shuffle_jc2);
    shuffle_camera_jacobian_kernel<2><<<grid, block>>>(len, (bw * bsize),
                                                       (float4*)result.data());
  } else {
    jc.BindTexture(tex_shuffle_jc);
    unsigned int bw, bh;
    GetBlockConfiguration(nblock, bw, bh);
    dim3 grid(bw, bh), block(bsize);
    shuffle_camera_jacobian_kernel<1><<<grid, block>>>(len, (bw * bsize),
                                                       (float4*)result.data());
  }
  CheckErrorCUDA("ShuffleCameraJacobian");
  return true;
}

texture<float4, 1, cudaReadModeElementType> tex_mjx_jc;
texture<float4, 1, cudaReadModeElementType> tex_mjx_jc2;
texture<float4, 1, cudaReadModeElementType> tex_mjx_jc3;
texture<float4, 1, cudaReadModeElementType> tex_mjx_jc4;
texture<float4, 1, cudaReadModeElementType> tex_mjx_jp;
texture<float4, 1, cudaReadModeElementType> tex_mjx_jp2;
texture<int2, 1, cudaReadModeElementType> tex_mjx_idx;
texture<float4, 1, cudaReadModeElementType> tex_mjx_x;

template <int TEXN>
__global__ void multiply_jx_kernel(int num, int bwidth, int offset,
                                   float* result) {
  int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * bwidth;
  if (index >= num) return;

  if (TEXN == 4 && (index >> 24) == 3) {
    ////////////////////////////////////////////
    int2 proj = tex1Dfetch(tex_mjx_idx, index >> 1);
    float4 xc1 = tex1Dfetch(tex_mjx_x, proj.x);
    float4 xc2 = tex1Dfetch(tex_mjx_x, proj.x + 1);
    float4 xp = tex1Dfetch(tex_mjx_x, proj.y + offset);

    ////////////////////////////////////////////
    float4 jp, jc1, jc2;
    jp = tex1Dfetch(tex_mjx_jp2, index & 0x1ffffff);
    jc1 = tex1Dfetch(tex_mjx_jc4, (index & 0xffffff) << 1);
    jc2 = tex1Dfetch(tex_mjx_jc4, ((index & 0xffffff) << 1) + 1);

    /////////////////////////////////////
    result[index] = jc1.x * xc1.x + jc1.y * xc1.y + jc1.z * xc1.z +
                    jc1.w * xc1.w + jc2.x * xc2.x + jc2.y * xc2.y +
                    jc2.z * xc2.z + jc2.w * xc2.w + jp.x * xp.x + jp.y * xp.y +
                    jp.z * xp.z;
  } else if (TEXN > 2 && (index >> 24) == 2) {
    ////////////////////////////////////////////
    int2 proj = tex1Dfetch(tex_mjx_idx, index >> 1);
    float4 xc1 = tex1Dfetch(tex_mjx_x, proj.x);
    float4 xc2 = tex1Dfetch(tex_mjx_x, proj.x + 1);
    float4 xp = tex1Dfetch(tex_mjx_x, proj.y + offset);

    ////////////////////////////////////////////
    float4 jp, jc1, jc2;
    jp = tex1Dfetch(tex_mjx_jp2, index & 0x1ffffff);
    jc1 = tex1Dfetch(tex_mjx_jc3, (index & 0xffffff) << 1);
    jc2 = tex1Dfetch(tex_mjx_jc3, ((index & 0xffffff) << 1) + 1);

    /////////////////////////////////////
    result[index] = jc1.x * xc1.x + jc1.y * xc1.y + jc1.z * xc1.z +
                    jc1.w * xc1.w + jc2.x * xc2.x + jc2.y * xc2.y +
                    jc2.z * xc2.z + jc2.w * xc2.w + jp.x * xp.x + jp.y * xp.y +
                    jp.z * xp.z;
  } else if (TEXN > 1 && (index > 0xffffff)) {
    ////////////////////////////////////////////
    int2 proj = tex1Dfetch(tex_mjx_idx, index >> 1);
    float4 xc1 = tex1Dfetch(tex_mjx_x, proj.x);
    float4 xc2 = tex1Dfetch(tex_mjx_x, proj.x + 1);
    float4 xp = tex1Dfetch(tex_mjx_x, proj.y + offset);

    ////////////////////////////////////////////
    float4 jp, jc1, jc2;
    jp = tex1Dfetch(tex_mjx_jp, index & 0x1ffffff);
    jc1 = tex1Dfetch(tex_mjx_jc2, (index & 0xffffff) << 1);
    jc2 = tex1Dfetch(tex_mjx_jc2, ((index & 0xffffff) << 1) + 1);

    /////////////////////////////////////
    result[index] = jc1.x * xc1.x + jc1.y * xc1.y + jc1.z * xc1.z +
                    jc1.w * xc1.w + jc2.x * xc2.x + jc2.y * xc2.y +
                    jc2.z * xc2.z + jc2.w * xc2.w + jp.x * xp.x + jp.y * xp.y +
                    jp.z * xp.z;
  } else {
    ////////////////////////////////////////////
    int2 proj = tex1Dfetch(tex_mjx_idx, index >> 1);
    float4 xc1 = tex1Dfetch(tex_mjx_x, proj.x);
    float4 xc2 = tex1Dfetch(tex_mjx_x, proj.x + 1);
    float4 xp = tex1Dfetch(tex_mjx_x, proj.y + offset);

    ////////////////////////////////////////////
    float4 jp, jc1, jc2;
    jp = tex1Dfetch(tex_mjx_jp, index);
    jc1 = tex1Dfetch(tex_mjx_jc, index << 1);
    jc2 = tex1Dfetch(tex_mjx_jc, (index << 1) + 1);

    /////////////////////////////////////
    result[index] = jc1.x * xc1.x + jc1.y * xc1.y + jc1.z * xc1.z +
                    jc1.w * xc1.w + jc2.x * xc2.x + jc2.y * xc2.y +
                    jc2.z * xc2.z + jc2.w * xc2.w + jp.x * xp.x + jp.y * xp.y +
                    jp.z * xp.z;
  }
}

template <int TEXN>
__global__ void multiply_jcx_kernel(int num, int bwidth, float* result) {
  int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * bwidth;
  if (index >= num) return;

  if (TEXN == 4 && (index >> 24) == 3) {
    ////////////////////////////////////////////
    int2 proj = tex1Dfetch(tex_mjx_idx, index >> 1);
    float4 xc1 = tex1Dfetch(tex_mjx_x, proj.x);
    float4 xc2 = tex1Dfetch(tex_mjx_x, proj.x + 1);

    ////////////////////////////////////////////
    float4 jc1, jc2;
    jc1 = tex1Dfetch(tex_mjx_jc4, (index & 0xffffff) << 1);
    jc2 = tex1Dfetch(tex_mjx_jc4, ((index & 0xffffff) << 1) + 1);

    /////////////////////////////////////
    result[index] = jc1.x * xc1.x + jc1.y * xc1.y + jc1.z * xc1.z +
                    jc1.w * xc1.w + jc2.x * xc2.x + jc2.y * xc2.y +
                    jc2.z * xc2.z + jc2.w * xc2.w;
  } else if (TEXN > 2 && (index >> 24) == 2) {
    ////////////////////////////////////////////
    int2 proj = tex1Dfetch(tex_mjx_idx, index >> 1);
    float4 xc1 = tex1Dfetch(tex_mjx_x, proj.x);
    float4 xc2 = tex1Dfetch(tex_mjx_x, proj.x + 1);

    ////////////////////////////////////////////
    float4 jc1, jc2;
    jc1 = tex1Dfetch(tex_mjx_jc3, (index & 0xffffff) << 1);
    jc2 = tex1Dfetch(tex_mjx_jc3, ((index & 0xffffff) << 1) + 1);

    /////////////////////////////////////
    result[index] = jc1.x * xc1.x + jc1.y * xc1.y + jc1.z * xc1.z +
                    jc1.w * xc1.w + jc2.x * xc2.x + jc2.y * xc2.y +
                    jc2.z * xc2.z + jc2.w * xc2.w;
  } else if (TEXN > 1 && (index > 0xffffff)) {
    ////////////////////////////////////////////
    int2 proj = tex1Dfetch(tex_mjx_idx, index >> 1);
    float4 xc1 = tex1Dfetch(tex_mjx_x, proj.x);
    float4 xc2 = tex1Dfetch(tex_mjx_x, proj.x + 1);

    ////////////////////////////////////////////
    float4 jc1, jc2;
    jc1 = tex1Dfetch(tex_mjx_jc2, (index & 0xffffff) << 1);
    jc2 = tex1Dfetch(tex_mjx_jc2, ((index & 0xffffff) << 1) + 1);

    /////////////////////////////////////
    result[index] = jc1.x * xc1.x + jc1.y * xc1.y + jc1.z * xc1.z +
                    jc1.w * xc1.w + jc2.x * xc2.x + jc2.y * xc2.y +
                    jc2.z * xc2.z + jc2.w * xc2.w;
  } else {
    ////////////////////////////////////////////
    int2 proj = tex1Dfetch(tex_mjx_idx, index >> 1);
    float4 xc1 = tex1Dfetch(tex_mjx_x, proj.x);
    float4 xc2 = tex1Dfetch(tex_mjx_x, proj.x + 1);

    ////////////////////////////////////////////
    float4 jc1, jc2;
    jc1 = tex1Dfetch(tex_mjx_jc, index << 1);
    jc2 = tex1Dfetch(tex_mjx_jc, (index << 1) + 1);

    /////////////////////////////////////
    result[index] = jc1.x * xc1.x + jc1.y * xc1.y + jc1.z * xc1.z +
                    jc1.w * xc1.w + jc2.x * xc2.x + jc2.y * xc2.y +
                    jc2.z * xc2.z + jc2.w * xc2.w;
  }
}

template <int TEXN>
__global__ void multiply_jpx_kernel(int num, int bwidth, int offset,
                                    float* result) {
  int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * bwidth;
  if (index >= num) return;

  if (TEXN == 2 && index > 0x1ffffff) {
    ////////////////////////////////////////////
    int2 proj = tex1Dfetch(tex_mjx_idx, index >> 1);
    float4 xp = tex1Dfetch(tex_mjx_x, proj.y + offset);
    ////////////////////////////////////////////
    float4 jp = tex1Dfetch(tex_mjx_jp2, index & 0x1ffffff);
    /////////////////////////////////////
    result[index] = jp.x * xp.x + jp.y * xp.y + jp.z * xp.z;
  } else {
    ////////////////////////////////////////////
    int2 proj = tex1Dfetch(tex_mjx_idx, index >> 1);
    float4 xp = tex1Dfetch(tex_mjx_x, proj.y + offset);

    ////////////////////////////////////////////
    float4 jp = tex1Dfetch(tex_mjx_jp, index);
    /////////////////////////////////////
    result[index] = jp.x * xp.x + jp.y * xp.y + jp.z * xp.z;
  }
}

template <int KW>
__global__ void multiply_jx_notex2_kernel(int num, int bwidth, int offset,
                                          float* jcx, float* jpx,
                                          float* result) {
  int bindex = blockIdx.x * blockDim.x + blockIdx.y * bwidth;
  int index = threadIdx.x + bindex;

  ////////////////////////////////////////////
  int2 proj = tex1Dfetch(tex_mjx_idx, index >> 1);
  float4 xc1 = tex1Dfetch(tex_mjx_x, proj.x);
  float4 xc2 = tex1Dfetch(tex_mjx_x, proj.x + 1);
  float4 xp = tex1Dfetch(tex_mjx_x, proj.y + offset);
  ////////////////////////////////////////////
  __shared__ float jps[KW * 4];
  __shared__ float jcs[KW * 8];

  for (int i = threadIdx.x; i < 4 * KW; i += KW)
    jps[i] = jpx[(bindex << 2) + i];
  for (int i = threadIdx.x; i < 8 * KW; i += KW)
    jcs[i] = jcx[(bindex << 3) + i];

  __syncthreads();
  if (index >= num) return;

  /////////////////////////////////////
  float *jp = jps + threadIdx.x * 4, *jc = jcs + threadIdx.x * 8;
  result[index] = jc[0] * xc1.x + jc[1] * xc1.y + jc[2] * xc1.z +
                  jc[3] * xc1.w + jc[4] * xc2.x + jc[5] * xc2.y +
                  jc[6] * xc2.z + jc[7] * xc2.w + jp[0] * xp.x + jp[1] * xp.y +
                  jp[2] * xp.z;
}

template <int KW>
__global__ void multiply_jpx_notex2_kernel(int num, int bwidth, int offset,
                                           float* jpx, float* result) {
  int bindex = blockIdx.x * blockDim.x + blockIdx.y * bwidth;
  int index = threadIdx.x + bindex;

  ////////////////////////////////////////////
  int2 proj = tex1Dfetch(tex_mjx_idx, index >> 1);
  float4 xp = tex1Dfetch(tex_mjx_x, proj.y + offset);
  ////////////////////////////////////////////
  __shared__ float jps[KW * 4];

  for (int i = threadIdx.x; i < 4 * KW; i += KW)
    jps[i] = jpx[(bindex << 2) + i];

  __syncthreads();
  if (index >= num) return;

  /////////////////////////////////////
  float* jp = jps + threadIdx.x * 4;
  result[index] = jp[0] * xp.x + jp[1] * xp.y + jp[2] * xp.z;
}

template <int KW>
__global__ void multiply_jcx_notex2_kernel(int num, int bwidth, float* jcx,
                                           float* result) {
  int bindex = blockIdx.x * blockDim.x + blockIdx.y * bwidth;
  int index = threadIdx.x + bindex;

  ////////////////////////////////////////////
  int2 proj = tex1Dfetch(tex_mjx_idx, index >> 1);
  float4 xc1 = tex1Dfetch(tex_mjx_x, proj.x);
  float4 xc2 = tex1Dfetch(tex_mjx_x, proj.x + 1);
  ////////////////////////////////////////////

  __shared__ float jcs[KW * 8];
  for (int i = threadIdx.x; i < 8 * KW; i += KW)
    jcs[i] = jcx[(bindex << 3) + i];

  __syncthreads();
  if (index >= num) return;

  /////////////////////////////////////
  float* jc = jcs + threadIdx.x * 8;
  result[index] = jc[0] * xc1.x + jc[1] * xc1.y + jc[2] * xc1.z +
                  jc[3] * xc1.w + jc[4] * xc2.x + jc[5] * xc2.y +
                  jc[6] * xc2.z + jc[7] * xc2.w;
}

void ProgramCU::ComputeJX(int point_offset, CuTexImage& x, CuTexImage& jc,
                          CuTexImage& jp, CuTexImage& jmap, CuTexImage& result,
                          int mode) {
  // given a vector of parameters....
  // multiply the Jacobian Matrix with it [jc jp] * p
  // for each measurment, read back the jacobian
  // multiply and summ up th corresponding

  unsigned int nproj = jmap.GetImgWidth();
  unsigned int len = nproj * 2;
  unsigned int bsize = 64;
  unsigned int nblock = (len + bsize - 1) / bsize;
  unsigned int bw, bh;
  jmap.BindTexture(tex_mjx_idx);
  x.BindTexture(tex_mjx_x);

  if (mode == 0) {
    size_t szjc = jc.GetDataSize();
    if (TEX_TOOBIG4(szjc)) {
      GetBlockConfiguration(nblock, bw, bh);
      dim3 grid(bw, bh), block(bsize);
      multiply_jx_notex2_kernel<64><<<grid, block>>>(
          len, (bw * bsize), point_offset, jc.data(), jp.data(), result.data());
    } else if (szjc > 2 * MAX_TEXSIZE) {
      jp.BindTexture2(tex_mjx_jp, tex_mjx_jp2);
      jc.BindTexture4(tex_mjx_jc, tex_mjx_jc2, tex_mjx_jc3, tex_mjx_jc4);
      GetBlockConfiguration(nblock, bw, bh);
      dim3 grid(bw, bh), block(bsize);
      multiply_jx_kernel<4><<<grid, block>>>(len, (bw * bsize), point_offset,
                                             result.data());
    } else if (szjc > MAX_TEXSIZE) {
      jp.BindTexture(tex_mjx_jp);
      jc.BindTexture2(tex_mjx_jc, tex_mjx_jc2);
      GetBlockConfiguration(nblock, bw, bh);
      dim3 grid(bw, bh), block(bsize);
      multiply_jx_kernel<2><<<grid, block>>>(len, (bw * bsize), point_offset,
                                             result.data());
    } else {
      jp.BindTexture(tex_mjx_jp);
      jc.BindTexture(tex_mjx_jc);
      GetBlockConfiguration(nblock, bw, bh);
      dim3 grid(bh, bw), block(bsize);
      multiply_jx_kernel<1><<<grid, block>>>(len, (bh * bsize), point_offset,
                                             result.data());
    }
    CheckErrorCUDA("ComputeJX");
  } else if (mode == 1) {
    size_t szjc = jc.GetDataSize();
    if (TEX_TOOBIG4(szjc)) {
      GetBlockConfiguration(nblock, bw, bh);
      dim3 grid(bw, bh), block(bsize);
      multiply_jcx_notex2_kernel<64><<<grid, block>>>(len, (bw * bsize),
                                                      jc.data(), result.data());
    } else if (szjc > 2 * MAX_TEXSIZE) {
      jc.BindTexture4(tex_mjx_jc, tex_mjx_jc2, tex_mjx_jc3, tex_mjx_jc4);
      GetBlockConfiguration(nblock, bw, bh);
      dim3 grid(bw, bh), block(bsize);
      multiply_jcx_kernel<4><<<grid, block>>>(len, (bw * bsize), result.data());
    } else if (szjc > MAX_TEXSIZE) {
      jc.BindTexture2(tex_mjx_jc, tex_mjx_jc2);
      GetBlockConfiguration(nblock, bw, bh);
      dim3 grid(bw, bh), block(bsize);
      multiply_jcx_kernel<2><<<grid, block>>>(len, (bw * bsize), result.data());
    } else {
      jc.BindTexture(tex_mjx_jc);
      GetBlockConfiguration(nblock, bw, bh);
      dim3 grid(bh, bw), block(bsize);
      multiply_jcx_kernel<1><<<grid, block>>>(len, (bh * bsize), result.data());
    }
    CheckErrorCUDA("ComputeJCX");
  } else if (mode == 2) {
    size_t szjp = jp.GetDataSize();
    if (szjp > MAX_TEXSIZE) {
      jp.BindTexture(tex_mjx_jp);
      GetBlockConfiguration(nblock, bw, bh);
      dim3 grid(bw, bh), block(bsize);
      multiply_jpx_kernel<2><<<grid, block>>>(len, (bw * bsize), point_offset,
                                              result.data());
    } else {
      jp.BindTexture(tex_mjx_jp);
      GetBlockConfiguration(nblock, bw, bh);
      dim3 grid(bh, bw), block(bsize);
      multiply_jpx_kernel<1><<<grid, block>>>(len, (bh * bsize), point_offset,
                                              result.data());
    }
    CheckErrorCUDA("ComputeJPX");
  }
}

template <bool md, bool pd>
__device__ void jacobian_internal(int camera_pos, int pt_pos, int tidx,
                                  float* r, float jic, float* jxc, float* jyc,
                                  float* jxp, float* jyp) {
  float m[3];
  float4 ft = tex1Dfetch(tex_jacobian_cam, camera_pos);
  float4 r1 = tex1Dfetch(tex_jacobian_cam, camera_pos + 1);
  r[0] = r1.x;
  r[1] = r1.y;
  r[2] = r1.z;
  r[3] = r1.w;
  float4 r2 = tex1Dfetch(tex_jacobian_cam, camera_pos + 2);
  r[4] = r2.x;
  r[5] = r2.y;
  r[6] = r2.z;
  r[7] = r2.w;
  float4 r3 = tex1Dfetch(tex_jacobian_cam, camera_pos + 3);
  r[8] = r3.x;

  float4 temp = tex1Dfetch(tex_jacobian_pts, pt_pos);
  m[0] = temp.x;
  m[1] = temp.y;
  m[2] = temp.z;

  float x0 = r[0] * m[0] + r[1] * m[1] + r[2] * m[2];
  float y0 = r[3] * m[0] + r[4] * m[1] + r[5] * m[2];
  float z0 = r[6] * m[0] + r[7] * m[1] + r[8] * m[2];
  float f_p2 = FDIV(ft.x, z0 + ft.w);
  float p0_p2 = FDIV(x0 + ft.y, z0 + ft.w);
  float p1_p2 = FDIV(y0 + ft.z, z0 + ft.w);

  if (pd) {
    float rr1 = r3.y * p0_p2 * p0_p2;
    float rr2 = r3.y * p1_p2 * p1_p2;
    float f_p2_x = f_p2 * (1.0 + 3.0 * rr1 + rr2);
    float f_p2_y = f_p2 * (1.0 + 3.0 * rr2 + rr1);

    JACOBIAN_SET_JC_BEGIN
    float jfc = jic * (1 + rr1 + rr2);
    float ft_x_pn = jic * ft.x * (p0_p2 * p0_p2 + p1_p2 * p1_p2);
    /////////////////////////////////////////////////////
    jxc[0] = p0_p2 * jfc;
    jxc[1] = f_p2_x;
    jxc[2] = 0;
    jxc[3] = -f_p2_x * p0_p2;
    jxc[4] = -f_p2_x * p0_p2 * y0;
    jxc[5] = f_p2_x * (z0 + x0 * p0_p2);
    jxc[6] = -f_p2_x * y0;
    jxc[7] = ft_x_pn * p0_p2;

    jyc[0] = p1_p2 * jfc;
    jyc[1] = 0;
    jyc[2] = f_p2_y;
    jyc[3] = -f_p2_y * p1_p2;
    jyc[4] = -f_p2_y * (z0 + y0 * p1_p2);
    jyc[5] = f_p2_y * x0 * p1_p2;
    jyc[6] = f_p2_y * x0;
    jyc[7] = ft_x_pn * p1_p2;
    JACOBIAN_SET_JC_END
    ///////////////////////////////////
    jxp[0] = f_p2_x * (r[0] - r[6] * p0_p2);
    jxp[1] = f_p2_x * (r[1] - r[7] * p0_p2);
    jxp[2] = f_p2_x * (r[2] - r[8] * p0_p2);
    jyp[0] = f_p2_y * (r[3] - r[6] * p1_p2);
    jyp[1] = f_p2_y * (r[4] - r[7] * p1_p2);
    jyp[2] = f_p2_y * (r[5] - r[8] * p1_p2);
  } else {
    JACOBIAN_SET_JC_BEGIN
    jxc[0] = p0_p2 * jic;
    jxc[1] = f_p2;
    jxc[2] = 0;
    jxc[3] = -f_p2 * p0_p2;
    jxc[4] = -f_p2 * p0_p2 * y0;
    jxc[5] = f_p2 * (z0 + x0 * p0_p2);
    jxc[6] = -f_p2 * y0;

    jyc[0] = p1_p2 * jic;
    jyc[1] = 0;
    jyc[2] = f_p2;
    jyc[3] = -f_p2 * p1_p2;
    jyc[4] = -f_p2 * (z0 + y0 * p1_p2);
    jyc[5] = f_p2 * x0 * p1_p2;
    jyc[6] = f_p2 * x0;

    if (md) {
      float2 ms = tex1Dfetch(tex_jacobian_meas, tidx);
      float msn = (ms.x * ms.x + ms.y * ms.y) * jic;
      jxc[7] = -ms.x * msn;
      jyc[7] = -ms.y * msn;
    } else {
      jxc[7] = 0;
      jyc[7] = 0;
    }
    JACOBIAN_SET_JC_END
    ///////////////////////////////////
    jxp[0] = f_p2 * (r[0] - r[6] * p0_p2);
    jxp[1] = f_p2 * (r[1] - r[7] * p0_p2);
    jxp[2] = f_p2 * (r[2] - r[8] * p0_p2);
    jyp[0] = f_p2 * (r[3] - r[6] * p1_p2);
    jyp[1] = f_p2 * (r[4] - r[7] * p1_p2);
    jyp[2] = f_p2 * (r[5] - r[8] * p1_p2);
  }
}

template <bool md, bool pd>
__device__ void jacobian_camera_internal(int camera_pos, int pt_pos, int tidx,
                                         float* r, float jic, float* jxc,
                                         float* jyc) {
  float m[3];
  float4 ft = tex1Dfetch(tex_jacobian_cam, camera_pos);
  float4 r1 = tex1Dfetch(tex_jacobian_cam, camera_pos + 1);
  r[0] = r1.x;
  r[1] = r1.y;
  r[2] = r1.z;
  r[3] = r1.w;
  float4 r2 = tex1Dfetch(tex_jacobian_cam, camera_pos + 2);
  r[4] = r2.x;
  r[5] = r2.y;
  r[6] = r2.z;
  r[7] = r2.w;
  float4 r3 = tex1Dfetch(tex_jacobian_cam, camera_pos + 3);
  r[8] = r3.x;

  float4 temp = tex1Dfetch(tex_jacobian_pts, pt_pos);
  m[0] = temp.x;
  m[1] = temp.y;
  m[2] = temp.z;

  float x0 = r[0] * m[0] + r[1] * m[1] + r[2] * m[2];
  float y0 = r[3] * m[0] + r[4] * m[1] + r[5] * m[2];
  float z0 = r[6] * m[0] + r[7] * m[1] + r[8] * m[2];
  float f_p2 = FDIV(ft.x, z0 + ft.w);
  float p0_p2 = FDIV(x0 + ft.y, z0 + ft.w);
  float p1_p2 = FDIV(y0 + ft.z, z0 + ft.w);
#ifndef PBA_DISABLE_CONST_CAMERA
  if (r3.w != 0.0f) {
    jxc[0] = 0;
    jxc[1] = 0;
    jxc[2] = 0;
    jxc[3] = 0;
    jxc[4] = 0;
    jxc[5] = 0;
    jxc[6] = 0;
    jxc[7] = 0;
    jyc[0] = 0;
    jyc[1] = 0;
    jyc[2] = 0;
    jyc[3] = 0;
    jyc[4] = 0;
    jyc[5] = 0;
    jyc[6] = 0;
    jyc[7] = 0;
  } else
#endif
      if (pd) {
    float rr1 = r3.y * p0_p2 * p0_p2;
    float rr2 = r3.y * p1_p2 * p1_p2;
    float f_p2_x = f_p2 * (1.0 + 3.0 * rr1 + rr2);
    float f_p2_y = f_p2 * (1.0 + 3.0 * rr2 + rr1);
    float jfc = jic * (1 + rr1 + rr2);
    float ft_x_pn = jic * ft.x * (p0_p2 * p0_p2 + p1_p2 * p1_p2);
    /////////////////////////////////////////////////////
    jxc[0] = p0_p2 * jfc;
    jxc[1] = f_p2_x;
    jxc[2] = 0;
    jxc[3] = -f_p2_x * p0_p2;
    jxc[4] = -f_p2_x * p0_p2 * y0;
    jxc[5] = f_p2_x * (z0 + x0 * p0_p2);
    jxc[6] = -f_p2_x * y0;
    jxc[7] = ft_x_pn * p0_p2;

    jyc[0] = p1_p2 * jfc;
    jyc[1] = 0;
    jyc[2] = f_p2_y;
    jyc[3] = -f_p2_y * p1_p2;
    jyc[4] = -f_p2_y * (z0 + y0 * p1_p2);
    jyc[5] = f_p2_y * x0 * p1_p2;
    jyc[6] = f_p2_y * x0;
    jyc[7] = ft_x_pn * p1_p2;
  } else {
    jxc[0] = p0_p2 * jic;
    jxc[1] = f_p2;
    jxc[2] = 0;
    jxc[3] = -f_p2 * p0_p2;
    jxc[4] = -f_p2 * p0_p2 * y0;
    jxc[5] = f_p2 * (z0 + x0 * p0_p2);
    jxc[6] = -f_p2 * y0;

    jyc[0] = p1_p2 * jic;
    jyc[1] = 0;
    jyc[2] = f_p2;
    jyc[3] = -f_p2 * p1_p2;
    jyc[4] = -f_p2 * (z0 + y0 * p1_p2);
    jyc[5] = f_p2 * x0 * p1_p2;
    jyc[6] = f_p2 * x0;

    if (md) {
      float2 ms = tex1Dfetch(tex_jacobian_meas, tidx);
      float msn = (ms.x * ms.x + ms.y * ms.y) * jic;
      jxc[7] = -ms.x * msn;
      jyc[7] = -ms.y * msn;
    } else {
      jxc[7] = 0;
      jyc[7] = 0;
    }
  }
}

template <bool pd>
__device__ void jacobian_point_internal(int camera_pos, int pt_pos, int tidx,
                                        float* r, float* jxp, float* jyp) {
  float m[3];
  float4 ft = tex1Dfetch(tex_jacobian_cam, camera_pos);
  float4 r1 = tex1Dfetch(tex_jacobian_cam, camera_pos + 1);
  r[0] = r1.x;
  r[1] = r1.y;
  r[2] = r1.z;
  r[3] = r1.w;
  float4 r2 = tex1Dfetch(tex_jacobian_cam, camera_pos + 2);
  r[4] = r2.x;
  r[5] = r2.y;
  r[6] = r2.z;
  r[7] = r2.w;
  float4 r3 = tex1Dfetch(tex_jacobian_cam, camera_pos + 3);
  r[8] = r3.x;

  float4 temp = tex1Dfetch(tex_jacobian_pts, pt_pos);
  m[0] = temp.x;
  m[1] = temp.y;
  m[2] = temp.z;

  float x0 = r[0] * m[0] + r[1] * m[1] + r[2] * m[2];
  float y0 = r[3] * m[0] + r[4] * m[1] + r[5] * m[2];
  float z0 = r[6] * m[0] + r[7] * m[1] + r[8] * m[2];
  float f_p2 = FDIV(ft.x, z0 + ft.w);
  float p0_p2 = FDIV(x0 + ft.y, z0 + ft.w);
  float p1_p2 = FDIV(y0 + ft.z, z0 + ft.w);

  if (pd) {
    float rr1 = r3.y * p0_p2 * p0_p2;
    float rr2 = r3.y * p1_p2 * p1_p2;
    float f_p2_x = f_p2 * (1.0 + 3.0 * rr1 + rr2);
    float f_p2_y = f_p2 * (1.0 + 3.0 * rr2 + rr1);
    ///////////////////////////////////
    jxp[0] = f_p2_x * (r[0] - r[6] * p0_p2);
    jxp[1] = f_p2_x * (r[1] - r[7] * p0_p2);
    jxp[2] = f_p2_x * (r[2] - r[8] * p0_p2);
    jyp[0] = f_p2_y * (r[3] - r[6] * p1_p2);
    jyp[1] = f_p2_y * (r[4] - r[7] * p1_p2);
    jyp[2] = f_p2_y * (r[5] - r[8] * p1_p2);
  } else {
    ///////////////////////////////////
    jxp[0] = f_p2 * (r[0] - r[6] * p0_p2);
    jxp[1] = f_p2 * (r[1] - r[7] * p0_p2);
    jxp[2] = f_p2 * (r[2] - r[8] * p0_p2);
    jyp[0] = f_p2 * (r[3] - r[6] * p1_p2);
    jyp[1] = f_p2 * (r[4] - r[7] * p1_p2);
    jyp[2] = f_p2 * (r[5] - r[8] * p1_p2);
  }
}

template <bool md, bool pd>
__global__ void multiply_jx_noj_kernel(int num, int bwidth, int offset,
                                       float jic, float2* result) {
  int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * bwidth;
  if (index >= num) return;

  __shared__ float data[9 * 64];
  ////////////////////////////////////////////
  int2 proj = tex1Dfetch(tex_mjx_idx, index);
  float4 xc1 = tex1Dfetch(tex_mjx_x, proj.x);
  float4 xc2 = tex1Dfetch(tex_mjx_x, proj.x + 1);
  float4 xp = tex1Dfetch(tex_mjx_x, proj.y + offset);

  ////////////////////////////////////////////
  float jxc[8], jyc[8], jxp[3], jyp[3];
  jacobian_internal<md, pd>(proj.x << 1, proj.y, index, data + 9 * threadIdx.x,
                            jic, jxc, jyc, jxp, jyp);

  /////////////////////////////////////
  result[index] = make_float2(
      jxc[0] * xc1.x + jxc[1] * xc1.y + jxc[2] * xc1.z + jxc[3] * xc1.w +
          jxc[4] * xc2.x + jxc[5] * xc2.y + jxc[6] * xc2.z + jxc[7] * xc2.w +
          jxp[0] * xp.x + jxp[1] * xp.y + jxp[2] * xp.z,
      jyc[0] * xc1.x + jyc[1] * xc1.y + jyc[2] * xc1.z + jyc[3] * xc1.w +
          jyc[4] * xc2.x + jyc[5] * xc2.y + jyc[6] * xc2.z + jyc[7] * xc2.w +
          jyp[0] * xp.x + jyp[1] * xp.y + jyp[2] * xp.z);
}

template <bool md, bool pd>
__global__ void multiply_jcx_noj_kernel(int num, int bwidth, float jic,
                                        float2* result) {
  int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * bwidth;
  if (index >= num) return;

  __shared__ float data[9 * 64];
  ////////////////////////////////////////////
  int2 proj = tex1Dfetch(tex_mjx_idx, index);
  float4 xc1 = tex1Dfetch(tex_mjx_x, proj.x);
  float4 xc2 = tex1Dfetch(tex_mjx_x, proj.x + 1);

  ////////////////////////////////////////////
  float jxc[8], jyc[8];
  jacobian_camera_internal<md, pd>(proj.x << 1, proj.y, index,
                                   data + 9 * threadIdx.x, jic, jxc, jyc);

  /////////////////////////////////////
  result[index] = make_float2(
      jxc[0] * xc1.x + jxc[1] * xc1.y + jxc[2] * xc1.z + jxc[3] * xc1.w +
          jxc[4] * xc2.x + jxc[5] * xc2.y + jxc[6] * xc2.z + jxc[7] * xc2.w,
      jyc[0] * xc1.x + jyc[1] * xc1.y + jyc[2] * xc1.z + jyc[3] * xc1.w +
          jyc[4] * xc2.x + jyc[5] * xc2.y + jyc[6] * xc2.z + jyc[7] * xc2.w);
}

template <bool pd>
__global__ void multiply_jpx_noj_kernel(int num, int bwidth, int offset,
                                        float2* result) {
  int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * bwidth;
  if (index >= num) return;

  __shared__ float data[9 * 64];
  ////////////////////////////////////////////
  int2 proj = tex1Dfetch(tex_mjx_idx, index);
  float4 xp = tex1Dfetch(tex_mjx_x, proj.y + offset);

  ////////////////////////////////////////////
  float jxp[3], jyp[3];
  jacobian_point_internal<pd>(proj.x << 1, proj.y, index,
                              data + 9 * threadIdx.x, jxp, jyp);

  /////////////////////////////////////
  result[index] = make_float2(jxp[0] * xp.x + jxp[1] * xp.y + jxp[2] * xp.z,
                              jyp[0] * xp.x + jyp[1] * xp.y + jyp[2] * xp.z);
}

void ProgramCU::ComputeJX_(CuTexImage& x, CuTexImage& jx, CuTexImage& camera,
                           CuTexImage& point, CuTexImage& meas,
                           CuTexImage& pjmap, bool intrinsic_fixed,
                           int radial_distortion, int mode) {
  unsigned int nproj = pjmap.GetImgWidth();
  unsigned int len = nproj;
  unsigned int bsize = 64;
  unsigned int nblock = (len + bsize - 1) / bsize;
  unsigned int bw, bh;
  int point_offset = camera.GetImgWidth() * 2;
  float jfc = intrinsic_fixed ? 0 : 1.0f;

  /////////////////////////////
  pjmap.BindTexture(tex_mjx_idx);
  x.BindTexture(tex_mjx_x);
  camera.BindTexture(tex_jacobian_cam);
  point.BindTexture(tex_jacobian_pts);

  ///////////////////////////////////
  GetBlockConfiguration(nblock, bw, bh);
  dim3 grid(bw, bh), block(bsize);

  if (mode == 0) {
    if (radial_distortion == -1) {
      meas.BindTexture(tex_jacobian_meas);
      multiply_jx_noj_kernel<true, false><<<grid, block>>>(
          len, (bw * bsize), point_offset, jfc, (float2*)jx.data());
    } else if (radial_distortion) {
      multiply_jx_noj_kernel<false, true><<<grid, block>>>(
          len, (bw * bsize), point_offset, jfc, (float2*)jx.data());
    } else {
      multiply_jx_noj_kernel<false, false><<<grid, block>>>(
          len, (bw * bsize), point_offset, jfc, (float2*)jx.data());
    }

    CheckErrorCUDA("ComputeJX_");
  } else if (mode == 1) {
    if (radial_distortion == -1) {
      meas.BindTexture(tex_jacobian_meas);
      multiply_jcx_noj_kernel<true, false><<<grid, block>>>(
          len, (bw * bsize), jfc, (float2*)jx.data());
    } else if (radial_distortion) {
      multiply_jcx_noj_kernel<false, true><<<grid, block>>>(
          len, (bw * bsize), jfc, (float2*)jx.data());
    } else {
      multiply_jcx_noj_kernel<false, false><<<grid, block>>>(
          len, (bw * bsize), jfc, (float2*)jx.data());
    }

    CheckErrorCUDA("ComputeJCX_");
  } else if (mode == 2) {
    if (radial_distortion == 1) {
      multiply_jpx_noj_kernel<true><<<grid, block>>>(
          len, (bw * bsize), point_offset, (float2*)jx.data());
    } else {
      multiply_jpx_noj_kernel<false><<<grid, block>>>(
          len, (bw * bsize), point_offset, (float2*)jx.data());
    }

    CheckErrorCUDA("ComputeJX_");
  }
}

template <bool md, bool pd, int KH>
__global__ void jte_cam_vec_noj_kernel(int num, int rowsz, float jic,
                                       float* jte) {
  __shared__ float value[KH * 32 * 9];  // 8 * KH * 32
  int cam = blockIdx.x * KH + threadIdx.y + blockIdx.y * rowsz;
  if (cam >= num) return;

  // read data range for this camera
  // 8 thread will do the same thing
  int idx1 = tex1Dfetch(tex_jte_cmp, cam);  // first camera
  int idx2 = tex1Dfetch(tex_jte_cmp, cam + 1);  // last camera + 1

  float* valuec = value + 32 * 9 * threadIdx.y;
  float* rp = valuec + threadIdx.x * 9;
  float rr[8], jxc[8], jyc[8];
  for (int i = 0; i < 8; ++i) rr[i] = 0;

  // loop to read the index of the projection.
  // so to get the location to read the jacobian
  for (int i = idx1 + threadIdx.x; i < idx2; i += 32) {
    int index = tex1Dfetch(tex_jte_cmt, i);
    int2 proj = tex1Dfetch(tex_jacobian_idx, index);
    jacobian_camera_internal<md, pd>(cam << 2, proj.y, index, rp, jic, jxc,
                                     jyc);
    float2 vv = tex1Dfetch(tex_jte_pe, index);
    //
    for (int j = 0; j < 8; ++j) rr[j] += (jxc[j] * vv.x + jyc[j] * vv.y);
  }

  float* valuei = valuec + 8 * threadIdx.x;
  for (int i = 0; i < 8; ++i) valuei[i] = rr[i];
  valuec[threadIdx.x] = (valuec[threadIdx.x] + valuec[threadIdx.x + 32] +
                         valuec[threadIdx.x + 64] + valuec[threadIdx.x + 96] +
                         valuec[threadIdx.x + 128] + valuec[threadIdx.x + 160] +
                         valuec[threadIdx.x + 192] + valuec[threadIdx.x + 224]);
  if (threadIdx.x < 16) valuec[threadIdx.x] += valuec[threadIdx.x + 16];
  if (threadIdx.x < 8)
    valuec[threadIdx.x] = valuec[threadIdx.x] + valuec[threadIdx.x + 8];

  ////////////////////////////////////
  if (threadIdx.x < 8) jte[(cam << 3) + threadIdx.x] = valuec[threadIdx.x];
}

template <bool pd, int KH>
__global__ void jte_point_vec_noj_kernel(int num, int rowsz, float* jte) {
  ////////////////////////////
  __shared__ float value[KH * (9 * 32)];
  int index = blockIdx.x * KH + threadIdx.y + blockIdx.y * rowsz;
  if (index >= num) return;

  int idx1 = tex1Dfetch(tex_jte_pmp, index);  // first
  int idx2 = tex1Dfetch(tex_jte_pmp, index + 1);  // last + 1
  float rx = 0, ry = 0, rz = 0, jxp[3], jyp[3];
  int rowp = threadIdx.y * 9 * 32;
  float* rp = value + threadIdx.x * 9 + rowp;
  for (int i = idx1 + threadIdx.x; i < idx2; i += 32) {
    float2 ev = tex1Dfetch(tex_jte_pe, i);
    int2 proj = tex1Dfetch(tex_jacobian_idx, i);
    jacobian_point_internal<pd>(proj.x << 1, proj.y, i, rp, jxp, jyp);
    rx += (jxp[0] * ev.x + jyp[0] * ev.y);
    ry += (jxp[1] * ev.x + jyp[1] * ev.y);
    rz += (jxp[2] * ev.x + jyp[2] * ev.y);
  }

  int loc = (threadIdx.x << 2) + rowp;
  value[loc] = rx;
  value[loc + 1] = ry;
  value[loc + 2] = rz;
  value[loc + 3] = 0;

  int ridx = threadIdx.x + rowp;
  value[ridx] = ((value[ridx] + value[ridx + 32]) +
                 (value[ridx + 64] + value[ridx + 96]));
  if (threadIdx.x < 16) value[ridx] += value[ridx + 16];
  if (threadIdx.x < 8) value[ridx] += value[ridx + 8];
  if (threadIdx.x < 4)
    jte[(index << 2) + threadIdx.x] = value[ridx] + value[ridx + 4];
}

void ProgramCU::ComputeJtE_(CuTexImage& e, CuTexImage& jte, CuTexImage& camera,
                            CuTexImage& point, CuTexImage& meas,
                            CuTexImage& cmap, CuTexImage& cmlist,
                            CuTexImage& pmap, CuTexImage& pjmap, CuTexImage& jp,
                            bool intrinsic_fixed, int radial_distortion,
                            int mode) {
  pjmap.BindTexture(tex_jacobian_idx);
  camera.BindTexture(tex_jacobian_cam);
  point.BindTexture(tex_jacobian_pts);
  if (radial_distortion) meas.BindTexture(tex_jacobian_meas);

  cmap.BindTexture(tex_jte_cmp);
  cmlist.BindTexture(tex_jte_cmt);
  e.BindTexture(tex_jte_pe);

  //
  unsigned int bw, bh;
  float jfc = intrinsic_fixed ? 0 : 1.0f;
  int ncam = camera.GetImgWidth();
  const int bheight1 = 2, bsize = 32;
  int nblock1 = (ncam + bheight1 - 1) / bheight1;
  GetBlockConfiguration(nblock1, bw, bh);
  dim3 grid(bw, bh), block(bsize, bheight1);
  if (mode == 2) {
  } else if (radial_distortion == -1)
    jte_cam_vec_noj_kernel<true, false, bheight1><<<grid, block>>>(
        ncam, bw * bheight1, jfc, jte.data());
  else if (radial_distortion)
    jte_cam_vec_noj_kernel<false, true, bheight1><<<grid, block>>>(
        ncam, bw * bheight1, jfc, jte.data());
  else
    jte_cam_vec_noj_kernel<false, false, bheight1><<<grid, block>>>(
        ncam, bw * bheight1, jfc, jte.data());
  CheckErrorCUDA("ComputeJtE_<Camera>");

  int npt = point.GetImgWidth();
  unsigned int offsetv = 8 * ncam;
  const int bheight2 = 2, bsize2 = 32;
  int nblock2 = (npt + bheight2 - 1) / bheight2;
  GetBlockConfiguration(nblock2, bw, bh);
  dim3 grid2(bw, bh), block2(bsize2, bheight2);
  if (mode == 1) {
  } else if (jp.IsValid()) {
    pmap.BindTexture(tex_jte_pmp);
    e.BindTexture(tex_jte_pex);
    jp.BindTexture2(tex_jte_jp, tex_jte_jp2);
    if (jp.GetDataSize() > MAX_TEXSIZE)
      jte_point_vec_kernel<bheight2, 2><<<grid2, block2>>>(
          npt, bw * bheight2, jte.data() + offsetv);
    else
      jte_point_vec_kernel<bheight2, 1><<<grid2, block2>>>(
          npt, bw * bheight2, jte.data() + offsetv);
  } else {
    pmap.BindTexture(tex_jte_pmp);
    if (radial_distortion && radial_distortion != -1)
      jte_point_vec_noj_kernel<true, bheight2><<<grid2, block2>>>(
          npt, bw * bheight2, jte.data() + offsetv);
    else
      jte_point_vec_noj_kernel<false, bheight2><<<grid2, block2>>>(
          npt, bw * bheight2, jte.data() + offsetv);
  }
  CheckErrorCUDA("ComputeJtE_<Point>");
}

template <int KH, bool md, bool pd, bool scaling>
__global__ void jtjd_cam_block_noj_kernel(int num, int rowsz, float lambda1,
                                          float lambda2, float jic, float* diag,
                                          float* blocks,
                                          bool add_existing_diagc) {
  const int VN = (md || pd) ? 8 : 7;
  __shared__ float buffer_all[32 * 9 * KH];
  __shared__ float value_all[64 * KH];

  // 8thread per camera
  int bcam = blockIdx.x * KH + blockIdx.y * rowsz;

  int cam = bcam + threadIdx.y;
  if (cam >= num) return;

  float* buffer = buffer_all + threadIdx.y * (32 * 9);
  float* value = value_all + threadIdx.y * 64;

  float jxc[8], jyc[8];
  float* rp = buffer + threadIdx.x * 9;
  float row0[VN], row1[VN - 1], row2[VN - 2], row3[VN - 3];
  float row4[VN - 4], row5[VN - 5], row6[VN - 6], row7[1] = {0};
  // read data range for this camera
  // 8 thread will do the same thing
  int idx1 = tex1Dfetch(tex_jtjd_cmp, cam);  // first camera
  int idx2 = tex1Dfetch(tex_jtjd_cmp, cam + 1);  // last camera + 1

#define REPEAT7(FUNC) \
  FUNC(0);            \
  FUNC(1);            \
  FUNC(2);            \
  FUNC(3);            \
  FUNC(4);            \
  FUNC(5);            \
  FUNC(6);
#define SETZERO(k) \
  for (int j = 0; j < VN - k; ++j) row##k[j] = 0;
  REPEAT7(SETZERO);

  float4 sjv[2];
  if (scaling && (pd || md)) {
    sjv[0] = tex1Dfetch(tex_jacobian_sj, (cam << 1));
    sjv[1] = tex1Dfetch(tex_jacobian_sj, (cam << 1) + 1);
  }

  // loop to read the index of the projection.
  // so to get the location to read the jacobian
  for (int i = idx1 + threadIdx.x; i < idx2; i += 32) {
    /////////////////////////////////////////
    int index = tex1Dfetch(tex_jtjd_cmlist, i);
    int2 proj = tex1Dfetch(tex_jacobian_idx, index);

    ///////////////////////////////////////////////
    jacobian_camera_internal<md, pd>(cam << 2, proj.y, index, rp, jic, jxc,
                                     jyc);

    if (scaling && (pd || md)) {
      float* sj = (float*)sjv;  // 32 threads...64 values
      for (int j = 0; j < VN; ++j) {
        jxc[j] *= sj[j];
        jyc[j] *= sj[j];
      }
    }

////////////////////////////////////////////////
#define ADDROW(k)              \
  for (int j = k; j < VN; ++j) \
  row##k[j - k] += (jxc[k] * jxc[j] + jyc[k] * jyc[j])

    ///////////////
    REPEAT7(ADDROW);
    if (VN == 8) {
      ADDROW(7);
    }
  }

////////////////////////////////////
// make the matrix..//add up the 32 * 8 matrix
#define JTJDSUM8_V1()                                          \
  buffer[threadIdx.x] =                                        \
      (buffer[threadIdx.x] + buffer[threadIdx.x + 32] +        \
       buffer[threadIdx.x + 64] + buffer[threadIdx.x + 96] +   \
       buffer[threadIdx.x + 128] + buffer[threadIdx.x + 160] + \
       buffer[threadIdx.x + 192] + buffer[threadIdx.x + 224]);

#define JTJDSUM8_V2()                                             \
  buffer[threadIdx.x] =                                           \
      (((buffer[threadIdx.x] + buffer[threadIdx.x + 128]) +       \
        (buffer[threadIdx.x + 64] + buffer[threadIdx.x + 192])) + \
       ((buffer[threadIdx.x + 32] + buffer[threadIdx.x + 160]) +  \
        (buffer[threadIdx.x + 96] + buffer[threadIdx.x + 224])));

#define STORE_ROWS(k)                                                        \
  for (int i = 0; i < (VN - k); ++i) bufi[i] = row##k[i];                    \
  JTJDSUM8_V2();                                                             \
  if (threadIdx.x < 16 - k) buffer[threadIdx.x] += buffer[threadIdx.x + 16]; \
  if (threadIdx.x < 8 - k)                                                   \
    value[threadIdx.x + k * 9] = buffer[threadIdx.x] + buffer[threadIdx.x + 8];

  float* bufi = buffer + threadIdx.x * 8;
  REPEAT7(STORE_ROWS);
  if (VN == 8) {
    STORE_ROWS(7);
  }

  /////////////////////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////    (8 * i + j) -> (8 * j + i)
  //#define COPYSYM(i) if(threadIdx.x < VN - i - 1) value[threadIdx.x * 8 +  i *
  //9 + 8] = value[threadIdx.x +  i * 9 + 1];
  if (threadIdx.x < VN - 1) value[threadIdx.x * 8 + 8] = value[threadIdx.x + 1];
  if (threadIdx.x < VN - 2)
    value[threadIdx.x * 8 + 17] = value[threadIdx.x + 10];
  if (threadIdx.x < VN - 3)
    value[threadIdx.x * 8 + 26] = value[threadIdx.x + 19];
  if (threadIdx.x < VN - 4)
    value[threadIdx.x * 8 + 35] = value[threadIdx.x + 28];
  if (threadIdx.x < VN - 5)
    value[threadIdx.x * 8 + 44] = value[threadIdx.x + 37];
  if (threadIdx.x < VN - 6)
    value[threadIdx.x * 8 + 53] = value[threadIdx.x + 46];
  if (VN == 8 && threadIdx.x < VN - 7)
    value[threadIdx.x * 8 + 62] = value[threadIdx.x + 55];

  if (scaling && !pd && !md) {
    float4 sjv[2];
    float* sj = (float*)sjv;  // 32 threads...64 values
    sjv[0] = tex1Dfetch(tex_jacobian_sj, (cam << 1));
    sjv[1] = tex1Dfetch(tex_jacobian_sj, (cam << 1) + 1);
    float sji = sj[threadIdx.x & 0x07];
    value[threadIdx.x] *= (sji * sj[threadIdx.x / 8]);
    value[threadIdx.x + 32] *= (sji * sj[4 + threadIdx.x / 8]);
  }

  bool zero = ((threadIdx.x & 0x7) == VN);

  ///////////write back
  if (threadIdx.x < 8) {
    float* dp = value + threadIdx.x * 9;
    float temp = zero ? 0 : dp[0];
    int didx = threadIdx.x + (cam << 3);
    if (add_existing_diagc) temp += diag[didx];
    diag[didx] = temp;
    dp[0] = lambda1 + lambda2 * temp;
  }
  int wpos = cam * (8 * VN) + threadIdx.x;
  blocks[wpos] = zero ? 0 : value[threadIdx.x];
  if (threadIdx.x < VN * 8 - 32)
    blocks[wpos + 32] = zero ? 0 : value[threadIdx.x + 32];
}

template <int KW, bool pd, bool scaling>
__global__ void jtjd_point_block_noj_kernel(int num, int rowsz, float lambda1,
                                            float lambda2, float4* diag,
                                            float4* blocks, int ptx) {
  ////////////////////////////
  int index = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * rowsz;
  if (index >= num) return;

  __shared__ float value[KW * 9];
  int idx1 = tex1Dfetch(tex_jtjd_pmp, index);  // first
  int idx2 = tex1Dfetch(tex_jtjd_pmp, index + 1);  // last + 1

  float M00 = 0, M01 = 0, M02 = 0, M11 = 0, M12 = 0, M22 = 0;
  float jxp[3], jyp[3];
  float* rp = value + threadIdx.x * 9;

  float4 sj;
  if (scaling && pd) sj = tex1Dfetch(tex_jacobian_sj, index + ptx);

  for (int i = idx1; i < idx2; ++i) {
    int2 proj = tex1Dfetch(tex_jacobian_idx, i);
    jacobian_point_internal<pd>(proj.x << 1, proj.y, i, rp, jxp, jyp);

    if (scaling && pd) {
      jxp[0] *= sj.x;
      jxp[1] *= sj.y;
      jxp[2] *= sj.z;
      jyp[0] *= sj.x;
      jyp[1] *= sj.y;
      jyp[2] *= sj.z;
    }
    M00 += (jxp[0] * jxp[0] + jyp[0] * jyp[0]);
    M01 += (jxp[0] * jxp[1] + jyp[0] * jyp[1]);
    M02 += (jxp[0] * jxp[2] + jyp[0] * jyp[2]);
    M11 += (jxp[1] * jxp[1] + jyp[1] * jyp[1]);
    M12 += (jxp[1] * jxp[2] + jyp[1] * jyp[2]);
    M22 += (jxp[2] * jxp[2] + jyp[2] * jyp[2]);
  }

  if (scaling && !pd) {
    sj = tex1Dfetch(tex_jacobian_sj, index + ptx);
    M00 *= (sj.x * sj.x);
    M01 *= (sj.x * sj.y);
    M02 *= (sj.x * sj.z);
    M11 *= (sj.y * sj.y);
    M12 *= (sj.y * sj.z);
    M22 *= (sj.z * sj.z);
  }

  diag[index] = make_float4(M00, M11, M22, 0);

  M00 = lambda2 * M00 + lambda1;
  M11 = lambda2 * M11 + lambda1;
  M22 = lambda2 * M22 + lambda1;

  // invert the 3x3 matrix.
  float det = (M00 * M11 - M01 * M01) * M22 + 2.0 * M01 * M12 * M02 -
              M02 * M02 * M11 - M12 * M12 * M00;
  if (det >= FLT_MAX || det <= FLT_MIN * 2.0f) {
    int write_pos = index * 3;
    blocks[write_pos] = make_float4(0, 0, 0, 0);
    blocks[write_pos + 1] = make_float4(0, 0, 0, 0);
    blocks[write_pos + 2] = make_float4(0, 0, 0, 0);
  } else {
    float m00 = (M11 * M22 - M12 * M12) / det;
    float m01 = -(M01 * M22 - M12 * M02) / det;
    float m02 = (M01 * M12 - M02 * M11) / det;
    int write_pos = index * 3;
    blocks[write_pos] = make_float4(m00, m01, m02, 0);

    float m11 = (M00 * M22 - M02 * M02) / det;
    float m12 = -(M00 * M12 - M01 * M02) / det;
    blocks[write_pos + 1] = make_float4(m01, m11, m12, 0);

    float m22 = (M00 * M11 - M01 * M01) / det;
    blocks[write_pos + 2] = make_float4(m02, m12, m22, 0);
  }
}

void ProgramCU::ComputeDiagonalBlock_(
    float lambda, bool dampd, CuTexImage& camera, CuTexImage& point,
    CuTexImage& meas, CuTexImage& cmap, CuTexImage& cmlist, CuTexImage& pmap,
    CuTexImage& jmap, CuTexImage& jp, CuTexImage& sj, CuTexImage& diag,
    CuTexImage& blocks, bool intrinsic_fixed, int radial_distortion,
    bool add_existing_diagc, int mode) {
  float lambda1 = dampd ? 0.0f : lambda;
  float lambda2 = dampd ? (1.0f + lambda) : 1.0f;
  float jfc = intrinsic_fixed ? 0.0f : 1.0f;

  //////////////////////////////////
  jmap.BindTexture(tex_jacobian_idx);
  camera.BindTexture(tex_jacobian_cam);
  point.BindTexture(tex_jacobian_pts);
  cmap.BindTexture(tex_jtjd_cmp);
  cmlist.BindTexture(tex_jtjd_cmlist);

  ////////////////////////////////////////////////////
  const unsigned int bsize1 = 32;
  const unsigned int bheight1 = 2;
  unsigned int ncam = camera.GetImgWidth();  // how many cameras
  unsigned int nblock = (ncam + bheight1 - 1) / bheight1;
  unsigned int bw, bh;
  GetBlockConfiguration(nblock, bw, bh);
  dim3 block1(bsize1, bheight1), grid1(bw, bh);

  ///////////////////////////////////////////////////
  if (radial_distortion == -1) meas.BindTexture(tex_jacobian_meas);
  if (mode == 2) {
    // skip the camera part.
  } else if (sj.IsValid()) {
    sj.BindTexture(tex_jacobian_sj);
    if (radial_distortion == -1)
      jtjd_cam_block_noj_kernel<bheight1, true, false, true><<<grid1, block1>>>(
          ncam, bw * bheight1, lambda1, lambda2, jfc, diag.data(),
          blocks.data(), add_existing_diagc);
    else if (radial_distortion)
      jtjd_cam_block_noj_kernel<bheight1, false, true, true><<<grid1, block1>>>(
          ncam, bw * bheight1, lambda1, lambda2, jfc, diag.data(),
          blocks.data(), add_existing_diagc);
    else
      jtjd_cam_block_noj_kernel<bheight1, false, false,
                                true><<<grid1, block1>>>(
          ncam, bw * bheight1, lambda1, lambda2, jfc, diag.data(),
          blocks.data(), add_existing_diagc);
  } else {
    if (radial_distortion == -1)
      jtjd_cam_block_noj_kernel<bheight1, true, false,
                                false><<<grid1, block1>>>(
          ncam, bw * bheight1, lambda1, lambda2, jfc, diag.data(),
          blocks.data(), add_existing_diagc);
    else if (radial_distortion)
      jtjd_cam_block_noj_kernel<bheight1, false, true,
                                false><<<grid1, block1>>>(
          ncam, bw * bheight1, lambda1, lambda2, jfc, diag.data(),
          blocks.data(), add_existing_diagc);
    else
      jtjd_cam_block_noj_kernel<bheight1, false, false,
                                false><<<grid1, block1>>>(
          ncam, bw * bheight1, lambda1, lambda2, jfc, diag.data(),
          blocks.data(), add_existing_diagc);
  }
  CheckErrorCUDA("ComputeDiagonalBlock_<Camera>");

  ////////////////////////////////////////////////////
  const unsigned int bsize2 = 64;
  unsigned int npoint = point.GetImgWidth();
  unsigned int len2 = npoint;
  unsigned int nblock2 = (len2 + bsize2 - 1) / bsize2;
  unsigned int offsetd = 2 * ncam;
  unsigned int offsetb = (radial_distortion ? 16 : 14) * ncam;
  GetBlockConfiguration(nblock2, bw, bh);
  dim3 grid2(bw, bh), block2(bsize2);
  pmap.BindTexture(tex_jtjd_pmp);

  if (mode == 1) {
  } else if (jp.IsValid()) {
    jp.BindTexture2(tex_jtjd_jp, tex_jtjd_jp2);
    if (jp.GetDataSize() > MAX_TEXSIZE)
      jtjd_point_block_kernel<2><<<grid2, block2>>>(
          len2, (bw * bsize2), lambda1, lambda2,
          ((float4*)diag.data()) + offsetd, ((float4*)blocks.data()) + offsetb);
    else
      jtjd_point_block_kernel<1><<<grid2, block2>>>(
          len2, (bw * bsize2), lambda1, lambda2,
          ((float4*)diag.data()) + offsetd, ((float4*)blocks.data()) + offsetb);
  } else {
    if (sj.IsValid()) {
      sj.BindTexture(tex_jacobian_sj);
      if (radial_distortion && radial_distortion != -1)
        jtjd_point_block_noj_kernel<bsize2, true, true><<<grid2, block2>>>(
            len2, (bw * bsize2), lambda1, lambda2,
            ((float4*)diag.data()) + offsetd,
            ((float4*)blocks.data()) + offsetb, offsetd);
      else
        jtjd_point_block_noj_kernel<bsize2, false, true><<<grid2, block2>>>(
            len2, (bw * bsize2), lambda1, lambda2,
            ((float4*)diag.data()) + offsetd,
            ((float4*)blocks.data()) + offsetb, offsetd);
    } else {
      if (radial_distortion && radial_distortion != -1)
        jtjd_point_block_noj_kernel<bsize2, true, false><<<grid2, block2>>>(
            len2, (bw * bsize2), lambda1, lambda2,
            ((float4*)diag.data()) + offsetd,
            ((float4*)blocks.data()) + offsetb, 0);
      else
        jtjd_point_block_noj_kernel<bsize2, false, false><<<grid2, block2>>>(
            len2, (bw * bsize2), lambda1, lambda2,
            ((float4*)diag.data()) + offsetd,
            ((float4*)blocks.data()) + offsetb, 0);
    }
  }
  CheckErrorCUDA("ComputeDiagonalBlock_<Point>");

  ////////////////////////////////////////////////////
  if (mode != 2) {
    const unsigned int bsize3 = JTJD_BLOCK_CAM_INVERT_KWIDTH;
    unsigned int len3 = ncam * 8;
    unsigned int nblock3 = (len3 + bsize3 - 1) / bsize3;
    dim3 grid3(nblock3), block3(bsize3);
    if (radial_distortion)
      jtjd_cam_block_invert_kernel<8><<<grid3, block3>>>(
          len3, (float4*)blocks.data());
    else
      jtjd_cam_block_invert_kernel<7><<<grid3, block3>>>(
          len3, (float4*)blocks.data());
    CheckErrorCUDA("ComputeDiagonalBlockInverse<Camera>");
  }
}

__global__ void projection_q_kernel(int nproj, int rowsz, float2* pj) {
  ////////////////////////////////
  int tidx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * rowsz;
  if (tidx >= nproj) return;
  int2 proj = tex1Dfetch(tex_projection_idx, tidx);
  float2 wq = tex1Dfetch(tex_projection_mea, tidx);
  ///////////////////////////////////
  float f1 = tex1Dfetch(tex_projection_cam, proj.x * 4).x;
  float r1 = tex1Dfetch(tex_projection_cam, proj.x * 4 + 3).w;
  float f2 = tex1Dfetch(tex_projection_cam, proj.y * 4).x;
  float r2 = tex1Dfetch(tex_projection_cam, proj.y * 4 + 3).w;
  pj[tidx] = make_float2(-wq.x * (f1 - f2), -wq.y * (r1 - r2));
}

void ProgramCU::ComputeProjectionQ(CuTexImage& camera, CuTexImage& qmap,
                                   CuTexImage& qw, CuTexImage& proj,
                                   int offset) {
  ///////////////////////////////////////
  unsigned int len = qmap.GetImgWidth();
  unsigned int bsize = PROJECTION_FRT_KWIDTH;
  unsigned int nblock = (len + bsize - 1) / bsize;
  unsigned int bw, bh;
  GetBlockConfiguration(nblock, bw, bh);
  dim3 grid(bw, bh), block(bsize);

  ///////////////////////////////////////////
  camera.BindTexture(tex_projection_cam);
  qmap.BindTexture(tex_projection_idx);
  qw.BindTexture(tex_projection_mea);

  //////////////////////////////
  projection_q_kernel<<<grid, block>>>(len, bw * bsize,
                                       ((float2*)proj.data()) + offset);
}

template <bool SJ>
__global__ void multiply_jqx_kernel(int num, int bwidth, float2* result) {
  int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * bwidth;
  if (index >= num) return;
  ////////////////////////////////////////////
  int2 proj = tex1Dfetch(tex_mjx_idx, index);
  float2 wq = tex1Dfetch(tex_jacobian_meas, index);
  int idx1 = proj.x * 2, idx2 = proj.y * 2;
  float x11 = tex1Dfetch(tex_mjx_x, idx1).x;
  float x17 = tex1Dfetch(tex_mjx_x, idx1 + 1).w;
  float x21 = tex1Dfetch(tex_mjx_x, idx2).x;
  float x27 = tex1Dfetch(tex_mjx_x, idx2 + 1).w;

  if (SJ) {
    float s11 = tex1Dfetch(tex_jacobian_sj, idx1).x;
    float s17 = tex1Dfetch(tex_jacobian_sj, idx1 + 1).w;
    float s21 = tex1Dfetch(tex_jacobian_sj, idx2).x;
    float s27 = tex1Dfetch(tex_jacobian_sj, idx2 + 1).w;
    result[index] = make_float2((x11 * s11 - x21 * s21) * wq.x,
                                (x17 * s17 - x27 * s27) * wq.y);
  } else {
    result[index] = make_float2((x11 - x21) * wq.x, (x17 - x27) * wq.y);
  }
}

void ProgramCU::ComputeJQX(CuTexImage& x, CuTexImage& qmap, CuTexImage& wq,
                           CuTexImage& sj, CuTexImage& jx, int offset) {
  unsigned int nproj = qmap.GetImgWidth();
  unsigned int len = nproj;
  unsigned int bsize = 64;
  unsigned int nblock = (len + bsize - 1) / bsize;
  unsigned int bw, bh;

  /////////////////////////////
  qmap.BindTexture(tex_mjx_idx);
  x.BindTexture(tex_mjx_x);
  wq.BindTexture(tex_jacobian_meas);

  ///////////////////////////////////
  GetBlockConfiguration(nblock, bw, bh);
  dim3 grid(bw, bh), block(bsize);

  if (sj.IsValid()) {
    sj.BindTexture(tex_jacobian_sj);
    multiply_jqx_kernel<true><<<grid, block>>>(len, (bw * bsize),
                                               ((float2*)jx.data()) + offset);
  } else {
    multiply_jqx_kernel<false><<<grid, block>>>(len, (bw * bsize),
                                                ((float2*)jx.data()) + offset);
  }
}

texture<int2, 1, cudaReadModeElementType> tex_jte_q_idx;
texture<float2, 1, cudaReadModeElementType> tex_jte_q_w;

template <bool SJ>
__global__ void jte_cam_q_kernel(int num, int bwidth, float* jte) {
  // int cam = blockIdx.x * KH + threadIdx.y + blockIdx.y * rowsz ;
  int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * bwidth;
  if (index >= num) return;
  int2 indexp = tex1Dfetch(tex_jte_q_idx, index);
  if (indexp.x == -1) return;
  float2 wq = tex1Dfetch(tex_jte_q_w, index);
  float2 e1 = tex1Dfetch(tex_jte_pe, indexp.x);
  float2 e2 = tex1Dfetch(tex_jte_pe, indexp.y);
  int index8 = index << 3;
  if (SJ) {
    float s1 = tex1Dfetch(tex_jacobian_sj, index * 2).x;
    jte[index8] += s1 * wq.x * (e1.x - e2.x);
    float s7 = tex1Dfetch(tex_jacobian_sj, index * 2 + 1).w;
    jte[index8 + 7] += s7 * wq.y * (e1.y - e2.y);
  } else {
    jte[index8] += wq.x * (e1.x - e2.x);
    jte[index8 + 7] += wq.y * (e1.y - e2.y);
  }
}

void ProgramCU::ComputeJQtEC(CuTexImage& pe, CuTexImage& qlist, CuTexImage& wq,
                             CuTexImage& sj, CuTexImage& jte) {
  int ncam = qlist.GetImgWidth();
  const int bsize = 32;
  int nblock = (ncam + bsize - 1) / bsize;
  unsigned int bw, bh;
  GetBlockConfiguration(nblock, bw, bh);
  dim3 grid(bw, bh), block(bsize);

  pe.BindTexture(tex_jte_pe);
  qlist.BindTexture(tex_jte_q_idx);
  wq.BindTexture(tex_jte_q_w);

  if (sj.IsValid()) {
    sj.BindTexture(tex_jacobian_sj);
    jte_cam_q_kernel<true><<<grid, block>>>(ncam, (bw * bsize), jte.data());
  } else {
    jte_cam_q_kernel<false><<<grid, block>>>(ncam, (bw * bsize), jte.data());
  }
}

}  // namespace pba
