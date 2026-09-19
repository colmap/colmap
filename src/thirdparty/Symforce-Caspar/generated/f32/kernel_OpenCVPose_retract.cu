#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_OpenCVPose_retract.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) OpenCVPoseRetractKernel(
    float *OpenCVPose, unsigned int OpenCVPose_num_alloc, float *delta,
    unsigned int delta_num_alloc, float *out_OpenCVPose_retracted,
    unsigned int out_OpenCVPose_retracted_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(OpenCVPose, 0 * OpenCVPose_num_alloc,
                                         global_thread_idx, r0, r1, r2, r3);
    r4 = 5.00000000000000000e-01;
    r5 = 9.99999999999999980e-13;
    ReadIdx4<1024, float, float, float4>(delta, 0 * delta_num_alloc,
                                         global_thread_idx, r6, r7, r8, r9);
    r5 = fmaf(r8, r8, r5);
    r5 = fmaf(r7, r7, r5);
    r5 = fmaf(r6, r6, r5);
    r10 = sqrtf(r5);
    r10 = r4 * r10;
    r4 = cosf(r10);
    r11 = -1.00000000000000000e+00;
    r12 = r1 * r7;
    r10 = sinf(r10);
    r5 = rsqrtf(r5);
    r5 = r10 * r5;
    r10 = r0 * r6;
    r10 = fmaf(r5, r10, r5 * r12);
    r8 = r8 * r5;
    r10 = fmaf(r2, r8, r10);
    r10 = fmaf(r11, r10, r3 * r4);
    r12 = fmaf(r1, r8, r0 * r4);
    r13 = r3 * r6;
    r12 = fmaf(r5, r13, r12);
    r14 = r2 * r7;
    r14 = r14 * r11;
    r12 = fmaf(r5, r14, r12);
    r14 = r3 * r7;
    r14 = fmaf(r5, r14, r1 * r4);
    r13 = r0 * r11;
    r14 = fmaf(r8, r13, r14);
    r15 = r2 * r6;
    r14 = fmaf(r5, r15, r14);
    r15 = r1 * r6;
    r15 = r15 * r11;
    r15 = fmaf(r5, r15, r2 * r4);
    r4 = r0 * r7;
    r15 = fmaf(r5, r4, r15);
    r15 = fmaf(r3, r8, r15);
    WriteIdx4<1024, float, float, float4>(
        out_OpenCVPose_retracted, 0 * out_OpenCVPose_retracted_num_alloc,
        global_thread_idx, r12, r14, r15, r10);
    ReadIdx3<1024, float, float, float4>(OpenCVPose, 4 * OpenCVPose_num_alloc,
                                         global_thread_idx, r10, r15, r14);
    r9 = r10 + r9;
    ReadIdx2<1024, float, float, float2>(delta, 4 * delta_num_alloc,
                                         global_thread_idx, r10, r12);
    r10 = r15 + r10;
    r12 = r14 + r12;
    WriteIdx3<1024, float, float, float4>(
        out_OpenCVPose_retracted, 4 * out_OpenCVPose_retracted_num_alloc,
        global_thread_idx, r9, r10, r12);
  };
}

void OpenCVPoseRetract(float *OpenCVPose, unsigned int OpenCVPose_num_alloc,
                       float *delta, unsigned int delta_num_alloc,
                       float *out_OpenCVPose_retracted,
                       unsigned int out_OpenCVPose_retracted_num_alloc,
                       size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpenCVPoseRetractKernel<<<n_blocks, 1024>>>(
      OpenCVPose, OpenCVPose_num_alloc, delta, delta_num_alloc,
      out_OpenCVPose_retracted, out_OpenCVPose_retracted_num_alloc,
      problem_size);
}

} // namespace caspar