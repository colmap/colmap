#include "kernel_ThinPrismFisheyePose_retract.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) ThinPrismFisheyePoseRetractKernel(
    float* ThinPrismFisheyePose,
    unsigned int ThinPrismFisheyePose_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_ThinPrismFisheyePose_retracted,
    unsigned int out_ThinPrismFisheyePose_retracted_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;

  if (global_thread_idx < problem_size) {
    r0 = -1.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(ThinPrismFisheyePose,
                                         0 * ThinPrismFisheyePose_num_alloc,
                                         global_thread_idx,
                                         r1,
                                         r2,
                                         r3,
                                         r4);
    ReadIdx4<1024, float, float, float4>(
        delta, 0 * delta_num_alloc, global_thread_idx, r5, r6, r7, r8);
    r9 = 5.00000000000000000e-01;
    r10 = 9.99999999999999980e-13;
    r10 = fmaf(r7, r7, r10);
    r10 = fmaf(r6, r6, r10);
    r10 = fmaf(r5, r5, r10);
    r11 = sqrtf(r10);
    r11 = r9 * r11;
    r9 = sinf(r11);
    r10 = rsqrtf(r10);
    r10 = r9 * r10;
    r5 = r5 * r10;
    r9 = r2 * r6;
    r9 = fmaf(r10, r9, r1 * r5);
    r12 = r3 * r7;
    r9 = fmaf(r10, r12, r9);
    r11 = cosf(r11);
    r9 = fmaf(r4, r11, r0 * r9);
    r12 = fmaf(r4, r5, r1 * r11);
    r13 = r3 * r6;
    r13 = r13 * r0;
    r12 = fmaf(r10, r13, r12);
    r14 = r2 * r7;
    r12 = fmaf(r10, r14, r12);
    r14 = fmaf(r3, r5, r2 * r11);
    r13 = r4 * r6;
    r14 = fmaf(r10, r13, r14);
    r15 = r1 * r7;
    r15 = r15 * r0;
    r14 = fmaf(r10, r15, r14);
    r15 = r2 * r0;
    r15 = fmaf(r5, r15, r3 * r11);
    r11 = r1 * r6;
    r15 = fmaf(r10, r11, r15);
    r5 = r4 * r7;
    r15 = fmaf(r10, r5, r15);
    WriteIdx4<1024, float, float, float4>(
        out_ThinPrismFisheyePose_retracted,
        0 * out_ThinPrismFisheyePose_retracted_num_alloc,
        global_thread_idx,
        r12,
        r14,
        r15,
        r9);
    ReadIdx3<1024, float, float, float4>(ThinPrismFisheyePose,
                                         4 * ThinPrismFisheyePose_num_alloc,
                                         global_thread_idx,
                                         r9,
                                         r15,
                                         r14);
    r8 = r9 + r8;
    ReadIdx2<1024, float, float, float2>(
        delta, 4 * delta_num_alloc, global_thread_idx, r9, r12);
    r9 = r15 + r9;
    r12 = r14 + r12;
    WriteIdx3<1024, float, float, float4>(
        out_ThinPrismFisheyePose_retracted,
        4 * out_ThinPrismFisheyePose_retracted_num_alloc,
        global_thread_idx,
        r8,
        r9,
        r12);
  };
}

void ThinPrismFisheyePoseRetract(
    float* ThinPrismFisheyePose,
    unsigned int ThinPrismFisheyePose_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_ThinPrismFisheyePose_retracted,
    unsigned int out_ThinPrismFisheyePose_retracted_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyePoseRetractKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyePose,
      ThinPrismFisheyePose_num_alloc,
      delta,
      delta_num_alloc,
      out_ThinPrismFisheyePose_retracted,
      out_ThinPrismFisheyePose_retracted_num_alloc,
      problem_size);
}

}  // namespace caspar