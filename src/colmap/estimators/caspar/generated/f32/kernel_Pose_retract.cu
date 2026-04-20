#include "kernel_Pose_retract.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    Pose_retract_kernel(float* Pose,
                        unsigned int Pose_num_alloc,
                        float* delta,
                        unsigned int delta_num_alloc,
                        float* out_Pose_retracted,
                        unsigned int out_Pose_retracted_num_alloc,
                        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        Pose, 0 * Pose_num_alloc, global_thread_idx, r0, r1, r2, r3);
    r4 = 5.00000000000000000e-01;
    r5 = 9.99999999999999980e-13;
    read_idx_4<1024, float, float, float4>(
        delta, 0 * delta_num_alloc, global_thread_idx, r6, r7, r8, r9);
    r5 = fmaf(r8, r8, r5);
    r5 = fmaf(r7, r7, r5);
    r5 = fmaf(r6, r6, r5);
    r10 = sqrtf(r5);
    r10 = r4 * r10;
    r4 = cosf(r10);
    r11 = -1.00000000000000000e+00;
    r10 = sinf(r10);
    r5 = rsqrtf(r5);
    r5 = r10 * r5;
    r7 = r7 * r5;
    r10 = r2 * r8;
    r10 = fmaf(r5, r10, r1 * r7);
    r12 = r0 * r6;
    r10 = fmaf(r5, r12, r10);
    r10 = fmaf(r11, r10, r3 * r4);
    r12 = r2 * r11;
    r12 = fmaf(r7, r12, r0 * r4);
    r13 = r1 * r8;
    r12 = fmaf(r5, r13, r12);
    r14 = r3 * r6;
    r12 = fmaf(r5, r14, r12);
    r14 = fmaf(r3, r7, r1 * r4);
    r13 = r0 * r8;
    r13 = r13 * r11;
    r14 = fmaf(r5, r13, r14);
    r15 = r2 * r6;
    r14 = fmaf(r5, r15, r14);
    r7 = fmaf(r0, r7, r2 * r4);
    r4 = r3 * r8;
    r7 = fmaf(r5, r4, r7);
    r15 = r1 * r6;
    r15 = r15 * r11;
    r7 = fmaf(r5, r15, r7);
    write_idx_4<1024, float, float, float4>(out_Pose_retracted,
                                            0 * out_Pose_retracted_num_alloc,
                                            global_thread_idx,
                                            r12,
                                            r14,
                                            r7,
                                            r10);
    read_idx_3<1024, float, float, float4>(
        Pose, 4 * Pose_num_alloc, global_thread_idx, r10, r7, r14);
    r9 = r10 + r9;
    read_idx_2<1024, float, float, float2>(
        delta, 4 * delta_num_alloc, global_thread_idx, r10, r12);
    r10 = r7 + r10;
    r12 = r14 + r12;
    write_idx_3<1024, float, float, float4>(out_Pose_retracted,
                                            4 * out_Pose_retracted_num_alloc,
                                            global_thread_idx,
                                            r9,
                                            r10,
                                            r12);
  };
}

void Pose_retract(float* Pose,
                  unsigned int Pose_num_alloc,
                  float* delta,
                  unsigned int delta_num_alloc,
                  float* out_Pose_retracted,
                  unsigned int out_Pose_retracted_num_alloc,
                  size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  Pose_retract_kernel<<<n_blocks, 1024>>>(Pose,
                                          Pose_num_alloc,
                                          delta,
                                          delta_num_alloc,
                                          out_Pose_retracted,
                                          out_Pose_retracted_num_alloc,
                                          problem_size);
}

}  // namespace caspar