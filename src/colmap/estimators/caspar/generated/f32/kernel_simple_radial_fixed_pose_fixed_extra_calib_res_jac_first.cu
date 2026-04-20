#include "kernel_simple_radial_fixed_pose_fixed_extra_calib_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_pose_fixed_extra_calib_res_jac_first_kernel(
        float* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* extra_calib,
        unsigned int extra_calib_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* out_focal_jac,
        unsigned int out_focal_jac_num_alloc,
        float* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        float* const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        float* const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc,
        float* out_point_jac,
        unsigned int out_point_jac_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        float* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex focal_indices_loc[1024];
  focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36;

  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        extra_calib, 0 * extra_calib_num_alloc, global_thread_idx, r0, r1, r2);
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r3, r4);
    r5 = -1.00000000000000000e+00;
    r3 = fmaf(r3, r5, r0);
    read_idx_3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r0, r6, r7);
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
    r11 = -2.00000000000000000e+00;
    read_idx_4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r12, r13, r14, r15);
    r16 = r14 * r15;
    r17 = 2.00000000000000000e+00;
    r18 = r12 * r17;
    r19 = r13 * r18;
    r20 = fmaf(r11, r16, r19);
    r0 = fmaf(r9, r20, r0);
    r21 = r13 * r15;
    r22 = r14 * r18;
    r21 = fmaf(r17, r21, r22);
    r23 = r14 * r14;
    r23 = r23 * r11;
    r24 = 1.00000000000000000e+00;
    r25 = r13 * r13;
    r25 = fmaf(r11, r25, r24);
    r26 = r23 + r25;
    r0 = fmaf(r10, r21, r0);
    r0 = fmaf(r8, r26, r0);
    r27 = 9.99999999999999955e-07;
    r14 = r13 * r14;
    r14 = r14 * r17;
    r18 = fmaf(r15, r18, r14);
    r7 = fmaf(r9, r18, r7);
    r28 = r13 * r15;
    r28 = fmaf(r11, r28, r22);
    r22 = r12 * r12;
    r22 = r22 * r11;
    r25 = r22 + r25;
    r7 = fmaf(r8, r28, r7);
    r7 = fmaf(r10, r25, r7);
    r29 = copysign(1.0, r7);
    r29 = fmaf(r27, r29, r7);
    r27 = 1.0 / r29;
  };
  load_shared<1, float, float>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>(
        (float*)inout_shared, focal_indices_loc[threadIdx.x].target, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r30 = r29 * r29;
    r31 = 1.0 / r30;
    r32 = r0 * r0;
    r33 = r31 * r32;
    r16 = fmaf(r17, r16, r19);
    r8 = fmaf(r8, r16, r6);
    r6 = r12 * r15;
    r6 = fmaf(r11, r6, r14);
    r23 = r24 + r23;
    r23 = r23 + r22;
    r8 = fmaf(r10, r6, r8);
    r8 = fmaf(r9, r23, r8);
    r9 = r8 * r31;
    r10 = fmaf(r8, r9, r33);
    r10 = fmaf(r2, r10, r24);
    r24 = r7 * r10;
    r22 = r27 * r24;
    r3 = fmaf(r0, r22, r3);
    r4 = fmaf(r4, r5, r1);
    r4 = fmaf(r8, r22, r4);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r3, r4);
    r1 = fmaf(r3, r3, r4 * r4);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r1);
  if (global_thread_idx < problem_size) {
    r1 = r0 * r10;
    r1 = r1 * r27;
    r14 = r8 * r10;
    r14 = r14 * r27;
    write_idx_2<1024, float, float, float2>(
        out_focal_jac, 0 * out_focal_jac_num_alloc, global_thread_idx, r1, r14);
    r14 = r8 * r10;
    r4 = r5 * r4;
    r14 = r14 * r27;
    r1 = r0 * r10;
    r1 = r1 * r5;
    r1 = r1 * r3;
    r1 = fmaf(r27, r1, r4 * r14);
    write_sum_1<float, float>((float*)inout_shared, r1);
  };
  flush_sum_shared<1, float>(out_focal_njtr,
                             0 * out_focal_njtr_num_alloc,
                             focal_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r8 * r9;
    r14 = r10 * r10;
    r33 = fmaf(r14, r33, r14 * r1);
    write_sum_1<float, float>((float*)inout_shared, r33);
  };
  flush_sum_shared<1, float>(out_focal_precond_diag,
                             0 * out_focal_precond_diag_num_alloc,
                             focal_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = r7 * r2;
    r14 = r28 * r32;
    r30 = r29 * r30;
    r30 = 1.0 / r30;
    r30 = r11 * r30;
    r11 = r17 * r26;
    r11 = r11 * r0;
    r11 = fmaf(r31, r11, r30 * r14);
    r14 = r8 * r8;
    r14 = r14 * r30;
    r29 = r17 * r16;
    r11 = fmaf(r9, r29, r11);
    r11 = fmaf(r28, r14, r11);
    r33 = r33 * r11;
    r33 = r33 * r27;
    r11 = fmaf(r26, r22, r0 * r33);
    r29 = r0 * r28;
    r29 = r29 * r5;
    r29 = r29 * r31;
    r11 = fmaf(r24, r29, r11);
    r29 = r28 * r5;
    r29 = r29 * r24;
    r33 = fmaf(r8, r33, r9 * r29);
    r33 = fmaf(r16, r22, r33);
    r29 = r7 * r2;
    r1 = r17 * r20;
    r1 = r1 * r0;
    r19 = r18 * r32;
    r19 = fmaf(r30, r19, r31 * r1);
    r1 = r17 * r23;
    r19 = fmaf(r9, r1, r19);
    r19 = fmaf(r18, r14, r19);
    r29 = r29 * r0;
    r29 = r29 * r19;
    r29 = fmaf(r20, r22, r27 * r29);
    r1 = r0 * r18;
    r1 = r1 * r5;
    r1 = r1 * r31;
    r29 = fmaf(r24, r1, r29);
    r1 = r18 * r5;
    r1 = r1 * r24;
    r1 = fmaf(r9, r1, r23 * r22);
    r34 = r7 * r2;
    r34 = r34 * r8;
    r34 = r34 * r19;
    r1 = fmaf(r27, r34, r1);
    write_idx_4<1024, float, float, float4>(out_point_jac,
                                            0 * out_point_jac_num_alloc,
                                            global_thread_idx,
                                            r11,
                                            r33,
                                            r29,
                                            r1);
    r34 = r0 * r25;
    r34 = r34 * r5;
    r34 = r34 * r31;
    r34 = fmaf(r24, r34, r21 * r22);
    r19 = r7 * r2;
    r35 = r17 * r21;
    r35 = r35 * r0;
    r36 = r25 * r32;
    r36 = fmaf(r30, r36, r31 * r35);
    r35 = r17 * r6;
    r36 = fmaf(r9, r35, r36);
    r36 = fmaf(r25, r14, r36);
    r19 = r19 * r0;
    r19 = r19 * r36;
    r34 = fmaf(r27, r19, r34);
    r19 = r25 * r5;
    r19 = r19 * r24;
    r22 = fmaf(r6, r22, r9 * r19);
    r19 = r7 * r2;
    r19 = r19 * r8;
    r19 = r19 * r36;
    r22 = fmaf(r27, r19, r22);
    write_idx_2<1024, float, float, float2>(out_point_jac,
                                            4 * out_point_jac_num_alloc,
                                            global_thread_idx,
                                            r34,
                                            r22);
    r19 = r5 * r3;
    r19 = fmaf(r33, r4, r11 * r19);
    r27 = r5 * r3;
    r27 = fmaf(r1, r4, r29 * r27);
    r36 = r5 * r3;
    r4 = fmaf(r22, r4, r34 * r36);
    write_sum_3<float, float>((float*)inout_shared, r19, r27, r4);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = fmaf(r33, r33, r11 * r11);
    r27 = fmaf(r1, r1, r29 * r29);
    r19 = fmaf(r22, r22, r34 * r34);
    write_sum_3<float, float>((float*)inout_shared, r4, r27, r19);
  };
  flush_sum_shared<3, float>(out_point_precond_diag,
                             0 * out_point_precond_diag_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fmaf(r33, r1, r11 * r29);
    r33 = fmaf(r33, r22, r11 * r34);
    r22 = fmaf(r1, r22, r29 * r34);
    write_sum_3<float, float>((float*)inout_shared, r19, r33, r22);
  };
  flush_sum_shared<3, float>(out_point_precond_tril,
                             0 * out_point_precond_tril_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_pose_fixed_extra_calib_res_jac_first(
    float* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* extra_calib,
    unsigned int extra_calib_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* out_focal_jac,
    unsigned int out_focal_jac_num_alloc,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    float* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    float* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    float* out_point_jac,
    unsigned int out_point_jac_num_alloc,
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
  simple_radial_fixed_pose_fixed_extra_calib_res_jac_first_kernel<<<n_blocks,
                                                                    1024>>>(
      focal,
      focal_num_alloc,
      focal_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      extra_calib,
      extra_calib_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_focal_jac,
      out_focal_jac_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_focal_precond_diag,
      out_focal_precond_diag_num_alloc,
      out_focal_precond_tril,
      out_focal_precond_tril_num_alloc,
      out_point_jac,
      out_point_jac_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar