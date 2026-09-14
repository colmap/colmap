#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_OpenCVCalib_update_r_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) OpenCVCalibUpdateRFirstKernel(
    float *OpenCVCalib_r_k, unsigned int OpenCVCalib_r_k_num_alloc,
    float *OpenCVCalib_w, unsigned int OpenCVCalib_w_num_alloc,
    const float *const negalpha, float *out_OpenCVCalib_r_kp1,
    unsigned int out_OpenCVCalib_r_kp1_num_alloc,
    float *const out_OpenCVCalib_r_0_norm2_tot,
    float *const out_OpenCVCalib_r_kp1_norm2_tot, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_OpenCVCalib_r_0_norm2_tot_local[1];

  __shared__ float out_OpenCVCalib_r_kp1_norm2_tot_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_r_k,
                                         0 * OpenCVCalib_r_k_num_alloc,
                                         global_thread_idx, r0, r1, r2, r3);
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_w,
                                         0 * OpenCVCalib_w_num_alloc,
                                         global_thread_idx, r4, r5, r6, r7);
  };
  LoadUnique<1, float, float>(negalpha, 0, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float *)inout_shared, 0, r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r4 = fmaf(r4, r8, r0);
    r5 = fmaf(r5, r8, r1);
    r6 = fmaf(r6, r8, r2);
    r7 = fmaf(r7, r8, r3);
    WriteIdx4<1024, float, float, float4>(out_OpenCVCalib_r_kp1,
                                          0 * out_OpenCVCalib_r_kp1_num_alloc,
                                          global_thread_idx, r4, r5, r6, r7);
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_r_k,
                                         4 * OpenCVCalib_r_k_num_alloc,
                                         global_thread_idx, r9, r10, r11, r12);
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_w,
                                         4 * OpenCVCalib_w_num_alloc,
                                         global_thread_idx, r13, r14, r15, r16);
    r13 = fmaf(r13, r8, r9);
    r14 = fmaf(r14, r8, r10);
    r15 = fmaf(r15, r8, r11);
    r8 = fmaf(r16, r8, r12);
    WriteIdx4<1024, float, float, float4>(out_OpenCVCalib_r_kp1,
                                          4 * out_OpenCVCalib_r_kp1_num_alloc,
                                          global_thread_idx, r13, r14, r15, r8);
    r12 = fmaf(r12, r12, r11 * r11);
    r12 = fmaf(r1, r1, r12);
    r12 = fmaf(r0, r0, r12);
    r12 = fmaf(r3, r3, r12);
    r12 = fmaf(r2, r2, r12);
    r12 = fmaf(r10, r10, r12);
    r12 = fmaf(r9, r9, r12);
  };
  SumStore<float>(out_OpenCVCalib_r_0_norm2_tot_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r12);
  if (global_thread_idx < problem_size) {
    r7 = fmaf(r7, r7, r4 * r4);
    r7 = fmaf(r6, r6, r7);
    r7 = fmaf(r13, r13, r7);
    r7 = fmaf(r5, r5, r7);
    r7 = fmaf(r15, r15, r7);
    r7 = fmaf(r8, r8, r7);
    r7 = fmaf(r14, r14, r7);
  };
  SumStore<float>(out_OpenCVCalib_r_kp1_norm2_tot_local, (float *)inout_shared,
                  0, global_thread_idx < problem_size, r7);
  SumFlushFinal<float>(out_OpenCVCalib_r_0_norm2_tot_local,
                       out_OpenCVCalib_r_0_norm2_tot, 1);
  SumFlushFinal<float>(out_OpenCVCalib_r_kp1_norm2_tot_local,
                       out_OpenCVCalib_r_kp1_norm2_tot, 1);
}

void OpenCVCalibUpdateRFirst(
    float *OpenCVCalib_r_k, unsigned int OpenCVCalib_r_k_num_alloc,
    float *OpenCVCalib_w, unsigned int OpenCVCalib_w_num_alloc,
    const float *const negalpha, float *out_OpenCVCalib_r_kp1,
    unsigned int out_OpenCVCalib_r_kp1_num_alloc,
    float *const out_OpenCVCalib_r_0_norm2_tot,
    float *const out_OpenCVCalib_r_kp1_norm2_tot, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpenCVCalibUpdateRFirstKernel<<<n_blocks, 1024>>>(
      OpenCVCalib_r_k, OpenCVCalib_r_k_num_alloc, OpenCVCalib_w,
      OpenCVCalib_w_num_alloc, negalpha, out_OpenCVCalib_r_kp1,
      out_OpenCVCalib_r_kp1_num_alloc, out_OpenCVCalib_r_0_norm2_tot,
      out_OpenCVCalib_r_kp1_norm2_tot, problem_size);
}

} // namespace caspar