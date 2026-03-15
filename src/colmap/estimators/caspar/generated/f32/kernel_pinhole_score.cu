#include "kernel_pinhole_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_score_kernel(float* pose,
                         unsigned int pose_num_alloc,
                         SharedIndex* pose_indices,
                         float* calib,
                         unsigned int calib_num_alloc,
                         SharedIndex* calib_indices,
                         float* point,
                         unsigned int point_num_alloc,
                         SharedIndex* point_indices,
                         float* pixel,
                         unsigned int pixel_num_alloc,
                         float* const out_rTr,
                         size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27;
  load_shared<4, float, float>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         calib_indices_loc[threadIdx.x].target,
                         r0,
                         r1,
                         r2,
                         r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r5 = fmaf(r5, r6, r3);
  };
  load_shared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>(
        (float*)inout_shared, pose_indices_loc[threadIdx.x].target, r3, r7, r8);
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
    r16 = r12 * r13;
    r17 = 2.00000000000000000e+00;
    r16 = r16 * r17;
    r18 = r14 * r17;
    r19 = fmaf(r15, r18, r16);
    r19 = fmaf(r9, r19, r7);
    r7 = r13 * r18;
    r20 = -2.00000000000000000e+00;
    r21 = r15 * r20;
    r22 = fmaf(r12, r21, r7);
    r23 = r14 * r14;
    r23 = r23 * r20;
    r24 = 1.00000000000000000e+00;
    r25 = r12 * r12;
    r25 = fmaf(r20, r25, r24);
    r26 = r23 + r25;
    r19 = fmaf(r11, r22, r19);
    r19 = fmaf(r10, r26, r19);
    r26 = r1 * r19;
    r22 = 9.99999999999999955e-07;
    r27 = r12 * r15;
    r27 = fmaf(r17, r27, r7);
    r27 = fmaf(r10, r27, r8);
    r18 = r12 * r18;
    r8 = fmaf(r13, r21, r18);
    r7 = r13 * r13;
    r7 = r20 * r7;
    r25 = r7 + r25;
    r27 = fmaf(r9, r8, r27);
    r27 = fmaf(r11, r25, r27);
    r25 = copysign(1.0, r27);
    r25 = fmaf(r22, r25, r27);
    r25 = 1.0 / r25;
    r5 = fmaf(r25, r26, r5);
    r6 = fmaf(r4, r6, r2);
    r21 = fmaf(r14, r21, r16);
    r21 = fmaf(r10, r21, r3);
    r10 = r13 * r15;
    r10 = fmaf(r17, r10, r18);
    r23 = r24 + r23;
    r23 = r23 + r7;
    r21 = fmaf(r11, r10, r21);
    r21 = fmaf(r9, r23, r21);
    r23 = r0 * r21;
    r6 = fmaf(r25, r23, r6);
    r6 = fmaf(r6, r6, r5 * r5);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r6);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void pinhole_score(float* pose,
                   unsigned int pose_num_alloc,
                   SharedIndex* pose_indices,
                   float* calib,
                   unsigned int calib_num_alloc,
                   SharedIndex* calib_indices,
                   float* point,
                   unsigned int point_num_alloc,
                   SharedIndex* point_indices,
                   float* pixel,
                   unsigned int pixel_num_alloc,
                   float* const out_rTr,
                   size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_score_kernel<<<n_blocks, 1024>>>(pose,
                                           pose_num_alloc,
                                           pose_indices,
                                           calib,
                                           calib_num_alloc,
                                           calib_indices,
                                           point,
                                           point_num_alloc,
                                           point_indices,
                                           pixel,
                                           pixel_num_alloc,
                                           out_rTr,
                                           problem_size);
}

}  // namespace caspar