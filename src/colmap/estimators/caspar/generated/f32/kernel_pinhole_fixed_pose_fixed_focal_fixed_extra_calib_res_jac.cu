#include "kernel_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_pose_fixed_focal_fixed_extra_calib_res_jac_kernel(
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* extra_calib,
        unsigned int extra_calib_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        float* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        extra_calib, 0 * extra_calib_num_alloc, global_thread_idx, r0, r1);
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    r0 = 9.99999999999999955e-07;
    read_idx_3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r5, r6, r7);
  };
  load_shared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         point_indices_loc[threadIdx.x].target,
                         r8,
                         r9,
                         r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r11, r12, r13, r14);
    r15 = r12 * r13;
    r16 = 2.00000000000000000e+00;
    r15 = r15 * r16;
    r17 = r11 * r16;
    r18 = fmaf(r14, r17, r15);
    r7 = fmaf(r9, r18, r7);
    r19 = r13 * r17;
    r20 = -2.00000000000000000e+00;
    r21 = r14 * r20;
    r22 = fmaf(r12, r21, r19);
    r23 = r11 * r11;
    r23 = r23 * r20;
    r24 = 1.00000000000000000e+00;
    r25 = r12 * r12;
    r25 = fmaf(r20, r25, r24);
    r26 = r23 + r25;
    r7 = fmaf(r8, r22, r7);
    r7 = fmaf(r10, r26, r7);
    r27 = copysign(1.0, r7);
    r27 = fmaf(r0, r27, r7);
    r0 = 1.0 / r27;
    read_idx_2<1024, float, float, float2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r7, r28);
    r17 = r12 * r17;
    r29 = fmaf(r13, r21, r17);
    r5 = fmaf(r9, r29, r5);
    r30 = r12 * r14;
    r30 = fmaf(r16, r30, r19);
    r19 = r13 * r13;
    r19 = r20 * r19;
    r25 = r19 + r25;
    r5 = fmaf(r10, r30, r5);
    r5 = fmaf(r8, r25, r5);
    r5 = r7 * r5;
    r2 = fmaf(r0, r5, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r13 * r14;
    r1 = fmaf(r16, r1, r17);
    r8 = fmaf(r8, r1, r6);
    r21 = fmaf(r11, r21, r15);
    r19 = r24 + r19;
    r19 = r19 + r23;
    r8 = fmaf(r10, r21, r8);
    r8 = fmaf(r9, r19, r8);
    r8 = r28 * r8;
    r3 = fmaf(r0, r8, r3);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r9 = r4 * r2;
    r10 = r7 * r25;
    r27 = r27 * r27;
    r27 = 1.0 / r27;
    r27 = r4 * r27;
    r5 = r27 * r5;
    r10 = fmaf(r22, r5, r0 * r10);
    r23 = r4 * r3;
    r24 = r22 * r27;
    r11 = r28 * r1;
    r11 = fmaf(r0, r11, r8 * r24);
    r23 = fmaf(r11, r23, r10 * r9);
    r9 = r4 * r2;
    r24 = r7 * r29;
    r24 = fmaf(r0, r24, r18 * r5);
    r15 = r4 * r3;
    r6 = r28 * r19;
    r17 = r18 * r27;
    r17 = fmaf(r8, r17, r0 * r6);
    r15 = fmaf(r17, r15, r24 * r9);
    r9 = r4 * r3;
    r6 = r26 * r27;
    r16 = r28 * r21;
    r16 = fmaf(r0, r16, r8 * r6);
    r6 = r4 * r2;
    r8 = r7 * r30;
    r8 = fmaf(r0, r8, r26 * r5);
    r6 = fmaf(r8, r6, r16 * r9);
    write_sum_3<float, float>((float*)inout_shared, r23, r15, r6);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fmaf(r10, r10, r11 * r11);
    r15 = fmaf(r17, r17, r24 * r24);
    r23 = fmaf(r8, r8, r16 * r16);
    write_sum_3<float, float>((float*)inout_shared, r6, r15, r23);
  };
  flush_sum_shared<3, float>(out_point_precond_diag,
                             0 * out_point_precond_diag_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fmaf(r11, r17, r10 * r24);
    r11 = fmaf(r11, r16, r10 * r8);
    r8 = fmaf(r24, r8, r17 * r16);
    write_sum_3<float, float>((float*)inout_shared, r23, r11, r8);
  };
  flush_sum_shared<3, float>(out_point_precond_tril,
                             0 * out_point_precond_tril_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
}

void pinhole_fixed_pose_fixed_focal_fixed_extra_calib_res_jac(
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* extra_calib,
    unsigned int extra_calib_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    float* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_pose_fixed_focal_fixed_extra_calib_res_jac_kernel<<<n_blocks,
                                                                    1024>>>(
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      focal,
      focal_num_alloc,
      extra_calib,
      extra_calib_num_alloc,
      out_res,
      out_res_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar