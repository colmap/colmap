#include "kernel_simple_radial_fixed_focal_fixed_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_focal_fixed_point_score_kernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* extra_calib,
        unsigned int extra_calib_num_alloc,
        SharedIndex* extra_calib_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* const out_rTr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex extra_calib_indices_loc[1024];
  extra_calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? extra_calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26;
  load_shared<3, float, float>(extra_calib,
                               0 * extra_calib_num_alloc,
                               extra_calib_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         extra_calib_indices_loc[threadIdx.x].target,
                         r0,
                         r1,
                         r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r3, r4);
    r5 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r5, r1);
  };
  load_shared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>(
        (float*)inout_shared, pose_indices_loc[threadIdx.x].target, r1, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9, r10);
  };
  load_shared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         pose_indices_loc[threadIdx.x].target,
                         r11,
                         r12,
                         r13,
                         r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = r11 * r12;
    r16 = 2.00000000000000000e+00;
    r15 = r15 * r16;
    r17 = r13 * r16;
    r18 = fmaf(r14, r17, r15);
    r18 = fmaf(r8, r18, r6);
    r6 = r12 * r17;
    r19 = -2.00000000000000000e+00;
    r20 = r14 * r19;
    r21 = fmaf(r11, r20, r6);
    r22 = r13 * r13;
    r22 = r22 * r19;
    r23 = 1.00000000000000000e+00;
    r24 = r11 * r11;
    r24 = fmaf(r19, r24, r23);
    r25 = r22 + r24;
    r18 = fmaf(r10, r21, r18);
    r18 = fmaf(r9, r25, r18);
    read_idx_1<1024, float, float, float>(
        focal, 0 * focal_num_alloc, global_thread_idx, r25);
    r21 = 9.99999999999999955e-07;
    r26 = r11 * r14;
    r26 = fmaf(r16, r26, r6);
    r26 = fmaf(r9, r26, r7);
    r17 = r11 * r17;
    r7 = fmaf(r12, r20, r17);
    r6 = r12 * r12;
    r6 = r19 * r6;
    r24 = r6 + r24;
    r26 = fmaf(r8, r7, r26);
    r26 = fmaf(r10, r24, r26);
    r24 = copysign(1.0, r26);
    r24 = fmaf(r21, r24, r26);
    r21 = r24 * r24;
    r21 = 1.0 / r21;
    r26 = r18 * r18;
    r20 = fmaf(r13, r20, r15);
    r20 = fmaf(r9, r20, r1);
    r9 = r12 * r14;
    r9 = fmaf(r16, r9, r17);
    r22 = r23 + r22;
    r22 = r22 + r6;
    r20 = fmaf(r10, r9, r20);
    r20 = fmaf(r8, r22, r20);
    r22 = r20 * r20;
    r22 = fmaf(r21, r22, r21 * r26);
    r22 = fmaf(r2, r22, r23);
    r22 = r25 * r22;
    r24 = 1.0 / r24;
    r22 = r22 * r24;
    r4 = fmaf(r18, r22, r4);
    r5 = fmaf(r3, r5, r0);
    r5 = fmaf(r20, r22, r5);
    r5 = fmaf(r5, r5, r4 * r4);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r5);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_focal_fixed_point_score(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* extra_calib,
    unsigned int extra_calib_num_alloc,
    SharedIndex* extra_calib_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* const out_rTr,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_focal_fixed_point_score_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      extra_calib,
      extra_calib_num_alloc,
      extra_calib_indices,
      pixel,
      pixel_num_alloc,
      focal,
      focal_num_alloc,
      point,
      point_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar