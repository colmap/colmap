#include "kernel_pinhole_fixed_principal_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_principal_point_score_kernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* const out_rTr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(principal_point,
                                           0 * principal_point_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r1);
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
  };
  load_shared<2, float, float>(focal_and_extra,
                               0 * focal_and_extra_num_alloc,
                               focal_and_extra_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>((float*)inout_shared,
                         focal_and_extra_indices_loc[threadIdx.x].target,
                         r0,
                         r5);
  };
  __syncthreads();
  load_shared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>(
        (float*)inout_shared, pose_indices_loc[threadIdx.x].target, r6, r7, r8);
  };
  __syncthreads();
  load_shared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         point_indices_loc[threadIdx.x].target,
                         r9,
                         r10,
                         r11);
  };
  __syncthreads();
  load_shared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         pose_indices_loc[threadIdx.x].target,
                         r12,
                         r13,
                         r14,
                         r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = -2.00000000000000000e+00;
    r17 = r15 * r16;
    r18 = 2.00000000000000000e+00;
    r19 = r12 * r18;
    r20 = r13 * r19;
    r21 = fmaf(r14, r17, r20);
    r21 = fmaf(r10, r21, r6);
    r6 = r13 * r15;
    r22 = r14 * r19;
    r6 = fmaf(r18, r6, r22);
    r23 = r14 * r14;
    r23 = r16 * r23;
    r24 = 1.00000000000000000e+00;
    r25 = r13 * r13;
    r25 = fmaf(r16, r25, r24);
    r26 = r23 + r25;
    r21 = fmaf(r11, r6, r21);
    r21 = fmaf(r9, r26, r21);
    r26 = r0 * r21;
    r6 = 9.99999999999999955e-07;
    r27 = r13 * r14;
    r27 = r27 * r18;
    r19 = fmaf(r15, r19, r27);
    r19 = fmaf(r10, r19, r8);
    r22 = fmaf(r13, r17, r22);
    r8 = r12 * r12;
    r8 = r8 * r16;
    r25 = r8 + r25;
    r19 = fmaf(r9, r22, r19);
    r19 = fmaf(r11, r25, r19);
    r25 = copysign(1.0, r19);
    r25 = fmaf(r6, r25, r19);
    r25 = 1.0 / r25;
    r2 = fmaf(r25, r26, r2);
    r4 = fmaf(r3, r4, r1);
    r3 = r14 * r15;
    r3 = fmaf(r18, r3, r20);
    r3 = fmaf(r9, r3, r7);
    r17 = fmaf(r12, r17, r27);
    r23 = r24 + r23;
    r23 = r23 + r8;
    r3 = fmaf(r11, r17, r3);
    r3 = fmaf(r10, r23, r3);
    r23 = r5 * r3;
    r4 = fmaf(r25, r23, r4);
    r4 = fmaf(r4, r4, r2 * r2);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r4);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void pinhole_fixed_principal_point_score(float* pose,
                                         unsigned int pose_num_alloc,
                                         SharedIndex* pose_indices,
                                         float* focal_and_extra,
                                         unsigned int focal_and_extra_num_alloc,
                                         SharedIndex* focal_and_extra_indices,
                                         float* point,
                                         unsigned int point_num_alloc,
                                         SharedIndex* point_indices,
                                         float* pixel,
                                         unsigned int pixel_num_alloc,
                                         float* principal_point,
                                         unsigned int principal_point_num_alloc,
                                         float* const out_rTr,
                                         size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_principal_point_score_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      focal_and_extra,
      focal_and_extra_num_alloc,
      focal_and_extra_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      principal_point,
      principal_point_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar