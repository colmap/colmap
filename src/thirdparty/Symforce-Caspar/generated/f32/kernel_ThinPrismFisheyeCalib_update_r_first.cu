#include "kernel_ThinPrismFisheyeCalib_update_r_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeCalibUpdateRFirstKernel(
        float* ThinPrismFisheyeCalib_r_k,
        unsigned int ThinPrismFisheyeCalib_r_k_num_alloc,
        float* ThinPrismFisheyeCalib_w,
        unsigned int ThinPrismFisheyeCalib_w_num_alloc,
        const float* const negalpha,
        float* out_ThinPrismFisheyeCalib_r_kp1,
        unsigned int out_ThinPrismFisheyeCalib_r_kp1_num_alloc,
        float* const out_ThinPrismFisheyeCalib_r_0_norm2_tot,
        float* const out_ThinPrismFisheyeCalib_r_kp1_norm2_tot,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_ThinPrismFisheyeCalib_r_0_norm2_tot_local[1];

  __shared__ float out_ThinPrismFisheyeCalib_r_kp1_norm2_tot_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_r_k,
        0 * ThinPrismFisheyeCalib_r_k_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r3);
    ReadIdx4<1024, float, float, float4>(ThinPrismFisheyeCalib_w,
                                         0 * ThinPrismFisheyeCalib_w_num_alloc,
                                         global_thread_idx,
                                         r4,
                                         r5,
                                         r6,
                                         r7);
  };
  LoadUnique<1, float, float>(negalpha, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r4 = fmaf(r4, r8, r0);
    r5 = fmaf(r5, r8, r1);
    r6 = fmaf(r6, r8, r2);
    r7 = fmaf(r7, r8, r3);
    WriteIdx4<1024, float, float, float4>(
        out_ThinPrismFisheyeCalib_r_kp1,
        0 * out_ThinPrismFisheyeCalib_r_kp1_num_alloc,
        global_thread_idx,
        r4,
        r5,
        r6,
        r7);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_r_k,
        4 * ThinPrismFisheyeCalib_r_k_num_alloc,
        global_thread_idx,
        r9,
        r10,
        r11,
        r12);
    ReadIdx4<1024, float, float, float4>(ThinPrismFisheyeCalib_w,
                                         4 * ThinPrismFisheyeCalib_w_num_alloc,
                                         global_thread_idx,
                                         r13,
                                         r14,
                                         r15,
                                         r16);
    r13 = fmaf(r13, r8, r9);
    r14 = fmaf(r14, r8, r10);
    r15 = fmaf(r15, r8, r11);
    r16 = fmaf(r16, r8, r12);
    WriteIdx4<1024, float, float, float4>(
        out_ThinPrismFisheyeCalib_r_kp1,
        4 * out_ThinPrismFisheyeCalib_r_kp1_num_alloc,
        global_thread_idx,
        r13,
        r14,
        r15,
        r16);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_r_k,
        8 * ThinPrismFisheyeCalib_r_k_num_alloc,
        global_thread_idx,
        r17,
        r18,
        r19,
        r20);
    ReadIdx4<1024, float, float, float4>(ThinPrismFisheyeCalib_w,
                                         8 * ThinPrismFisheyeCalib_w_num_alloc,
                                         global_thread_idx,
                                         r21,
                                         r22,
                                         r23,
                                         r24);
    r21 = fmaf(r21, r8, r17);
    r22 = fmaf(r22, r8, r18);
    r23 = fmaf(r23, r8, r19);
    r8 = fmaf(r24, r8, r20);
    WriteIdx4<1024, float, float, float4>(
        out_ThinPrismFisheyeCalib_r_kp1,
        8 * out_ThinPrismFisheyeCalib_r_kp1_num_alloc,
        global_thread_idx,
        r21,
        r22,
        r23,
        r8);
    r11 = fmaf(r11, r11, r12 * r12);
    r11 = fmaf(r3, r3, r11);
    r11 = fmaf(r2, r2, r11);
    r11 = fmaf(r1, r1, r11);
    r11 = fmaf(r0, r0, r11);
    r11 = fmaf(r18, r18, r11);
    r11 = fmaf(r17, r17, r11);
    r11 = fmaf(r9, r9, r11);
    r11 = fmaf(r10, r10, r11);
    r11 = fmaf(r20, r20, r11);
    r11 = fmaf(r19, r19, r11);
  };
  SumStore<float>(out_ThinPrismFisheyeCalib_r_0_norm2_tot_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r11);
  if (global_thread_idx < problem_size) {
    r4 = fmaf(r4, r4, r22 * r22);
    r4 = fmaf(r15, r15, r4);
    r4 = fmaf(r14, r14, r4);
    r4 = fmaf(r5, r5, r4);
    r4 = fmaf(r21, r21, r4);
    r4 = fmaf(r16, r16, r4);
    r4 = fmaf(r7, r7, r4);
    r4 = fmaf(r6, r6, r4);
    r4 = fmaf(r13, r13, r4);
    r4 = fmaf(r23, r23, r4);
    r4 = fmaf(r8, r8, r4);
  };
  SumStore<float>(out_ThinPrismFisheyeCalib_r_kp1_norm2_tot_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r4);
  SumFlushFinal<float>(out_ThinPrismFisheyeCalib_r_0_norm2_tot_local,
                       out_ThinPrismFisheyeCalib_r_0_norm2_tot,
                       1);
  SumFlushFinal<float>(out_ThinPrismFisheyeCalib_r_kp1_norm2_tot_local,
                       out_ThinPrismFisheyeCalib_r_kp1_norm2_tot,
                       1);
}

void ThinPrismFisheyeCalibUpdateRFirst(
    float* ThinPrismFisheyeCalib_r_k,
    unsigned int ThinPrismFisheyeCalib_r_k_num_alloc,
    float* ThinPrismFisheyeCalib_w,
    unsigned int ThinPrismFisheyeCalib_w_num_alloc,
    const float* const negalpha,
    float* out_ThinPrismFisheyeCalib_r_kp1,
    unsigned int out_ThinPrismFisheyeCalib_r_kp1_num_alloc,
    float* const out_ThinPrismFisheyeCalib_r_0_norm2_tot,
    float* const out_ThinPrismFisheyeCalib_r_kp1_norm2_tot,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeCalibUpdateRFirstKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyeCalib_r_k,
      ThinPrismFisheyeCalib_r_k_num_alloc,
      ThinPrismFisheyeCalib_w,
      ThinPrismFisheyeCalib_w_num_alloc,
      negalpha,
      out_ThinPrismFisheyeCalib_r_kp1,
      out_ThinPrismFisheyeCalib_r_kp1_num_alloc,
      out_ThinPrismFisheyeCalib_r_0_norm2_tot,
      out_ThinPrismFisheyeCalib_r_kp1_norm2_tot,
      problem_size);
}

}  // namespace caspar