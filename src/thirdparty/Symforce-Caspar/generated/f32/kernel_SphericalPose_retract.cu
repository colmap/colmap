#include "kernel_SphericalPose_retract.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SphericalPoseRetractKernel(
    float* SphericalPose,
    unsigned int SphericalPose_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_SphericalPose_retracted,
    unsigned int out_SphericalPose_retracted_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;

  if (global_thread_idx < problem_size) {
    r0 = -1.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(SphericalPose,
                                         0 * SphericalPose_num_alloc,
                                         global_thread_idx,
                                         r1,
                                         r2,
                                         r3,
                                         r4);
    ReadIdx4<1024, float, float, float4>(
        delta, 0 * delta_num_alloc, global_thread_idx, r5, r6, r7, r8);
    r9 = r2 * r6;
    r10 = 5.00000000000000000e-01;
    r11 = 9.99999999999999980e-13;
    r11 = fmaf(r6, r6, r11);
    r11 = fmaf(r5, r5, r11);
    r11 = fmaf(r7, r7, r11);
    r12 = sqrtf(r11);
    r12 = r10 * r12;
    r10 = sinf(r12);
    r11 = rsqrtf(r11);
    r11 = r10 * r11;
    r7 = r7 * r11;
    r9 = fmaf(r3, r7, r11 * r9);
    r10 = r1 * r5;
    r9 = fmaf(r11, r10, r9);
    r12 = cosf(r12);
    r9 = fmaf(r4, r12, r0 * r9);
    r10 = fmaf(r2, r7, r1 * r12);
    r13 = r3 * r6;
    r13 = r13 * r0;
    r10 = fmaf(r11, r13, r10);
    r14 = r4 * r5;
    r10 = fmaf(r11, r14, r10);
    r14 = r3 * r5;
    r14 = fmaf(r11, r14, r2 * r12);
    r13 = r1 * r0;
    r14 = fmaf(r7, r13, r14);
    r15 = r4 * r6;
    r14 = fmaf(r11, r15, r14);
    r15 = r2 * r5;
    r15 = r15 * r0;
    r15 = fmaf(r11, r15, r3 * r12);
    r12 = r1 * r6;
    r15 = fmaf(r11, r12, r15);
    r15 = fmaf(r4, r7, r15);
    WriteIdx4<1024, float, float, float4>(
        out_SphericalPose_retracted,
        0 * out_SphericalPose_retracted_num_alloc,
        global_thread_idx,
        r10,
        r14,
        r15,
        r9);
    ReadIdx3<1024, float, float, float4>(SphericalPose,
                                         4 * SphericalPose_num_alloc,
                                         global_thread_idx,
                                         r9,
                                         r15,
                                         r14);
    r8 = r9 + r8;
    ReadIdx2<1024, float, float, float2>(
        delta, 4 * delta_num_alloc, global_thread_idx, r9, r10);
    r9 = r15 + r9;
    r10 = r14 + r10;
    WriteIdx3<1024, float, float, float4>(
        out_SphericalPose_retracted,
        4 * out_SphericalPose_retracted_num_alloc,
        global_thread_idx,
        r8,
        r9,
        r10);
  };
}

void SphericalPoseRetract(float* SphericalPose,
                          unsigned int SphericalPose_num_alloc,
                          float* delta,
                          unsigned int delta_num_alloc,
                          float* out_SphericalPose_retracted,
                          unsigned int out_SphericalPose_retracted_num_alloc,
                          size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SphericalPoseRetractKernel<<<n_blocks, 1024>>>(
      SphericalPose,
      SphericalPose_num_alloc,
      delta,
      delta_num_alloc,
      out_SphericalPose_retracted,
      out_SphericalPose_retracted_num_alloc,
      problem_size);
}

}  // namespace caspar