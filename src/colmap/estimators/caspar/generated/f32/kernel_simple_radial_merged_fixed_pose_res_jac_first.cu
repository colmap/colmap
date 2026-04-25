#include "kernel_simple_radial_merged_fixed_pose_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_merged_fixed_pose_res_jac_first_kernel(
        float* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* out_calib_jac,
        unsigned int out_calib_jac_num_alloc,
        float* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        float* const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        float* const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc,
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
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40;
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
    r4 = fmaf(r4, r6, r2);
    read_idx_3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r2, r7, r8);
  };
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
  if (global_thread_idx < problem_size) {
    r12 = -2.00000000000000000e+00;
    read_idx_4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r13, r14, r15, r16);
    r17 = r15 * r16;
    r18 = 2.00000000000000000e+00;
    r19 = r13 * r18;
    r20 = r14 * r19;
    r21 = fmaf(r12, r17, r20);
    r2 = fmaf(r10, r21, r2);
    r22 = r14 * r16;
    r23 = r15 * r19;
    r22 = fmaf(r18, r22, r23);
    r24 = r15 * r15;
    r24 = r24 * r12;
    r25 = 1.00000000000000000e+00;
    r26 = r14 * r14;
    r26 = fmaf(r12, r26, r25);
    r27 = r24 + r26;
    r2 = fmaf(r11, r22, r2);
    r2 = fmaf(r9, r27, r2);
    r17 = fmaf(r18, r17, r20);
    r7 = fmaf(r9, r17, r7);
    r15 = r14 * r15;
    r15 = r15 * r18;
    r20 = r13 * r16;
    r20 = fmaf(r12, r20, r15);
    r24 = r25 + r24;
    r28 = r13 * r13;
    r28 = r28 * r12;
    r24 = r24 + r28;
    r7 = fmaf(r11, r20, r7);
    r7 = fmaf(r10, r24, r7);
    r29 = r7 * r7;
    r30 = 9.99999999999999955e-07;
    r19 = fmaf(r16, r19, r15);
    r10 = fmaf(r10, r19, r8);
    r8 = r14 * r16;
    r8 = fmaf(r12, r8, r23);
    r26 = r28 + r26;
    r10 = fmaf(r9, r8, r10);
    r10 = fmaf(r11, r26, r10);
    r11 = copysign(1.0, r10);
    r11 = fmaf(r30, r11, r10);
    r30 = r11 * r11;
    r10 = 1.0 / r30;
    r29 = r29 * r10;
    r9 = r2 * r2;
    r28 = r10 * r9;
    r23 = r29 + r28;
    r15 = fmaf(r1, r23, r25);
    r31 = r2 * r15;
    r32 = 1.0 / r11;
    r33 = r0 * r32;
    r4 = fmaf(r33, r31, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r7 * r15;
    r5 = fmaf(r33, r3, r5);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r3 = fmaf(r5, r5, r4 * r4);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r3);
  if (global_thread_idx < problem_size) {
    r3 = r2 * r15;
    r3 = r3 * r32;
    r31 = r7 * r15;
    r31 = r31 * r32;
    r34 = r2 * r23;
    r34 = r34 * r33;
    r35 = r7 * r23;
    r35 = r35 * r33;
    write_idx_4<1024, float, float, float4>(out_calib_jac,
                                            0 * out_calib_jac_num_alloc,
                                            global_thread_idx,
                                            r3,
                                            r31,
                                            r34,
                                            r35);
    r36 = r6 * r4;
    r37 = r6 * r5;
    r38 = r2 * r4;
    r39 = r15 * r6;
    r38 = r38 * r32;
    r40 = r7 * r5;
    r40 = r40 * r32;
    r40 = fmaf(r39, r40, r39 * r38);
    r38 = r2 * r23;
    r38 = r38 * r6;
    r38 = r38 * r4;
    r32 = r7 * r23;
    r32 = r32 * r6;
    r32 = r32 * r5;
    r32 = fmaf(r33, r32, r33 * r38);
    write_sum_4<float, float>((float*)inout_shared, r40, r32, r36, r37);
  };
  flush_sum_shared<4, float>(out_calib_njtr,
                             0 * out_calib_njtr_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r15 * r15;
    r37 = fmaf(r28, r37, r29 * r37);
    r36 = r0 * r23;
    r32 = r0 * r23;
    r36 = r36 * r32;
    r36 = fmaf(r28, r36, r29 * r36);
    write_sum_4<float, float>((float*)inout_shared, r37, r36, r25, r25);
  };
  flush_sum_shared<4, float>(out_calib_precond_diag,
                             0 * out_calib_precond_diag_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r15 * r28;
    r36 = r15 * r29;
    r36 = fmaf(r32, r36, r32 * r25);
    write_sum_4<float, float>((float*)inout_shared, r36, r3, r31, r34);
  };
  flush_sum_shared<4, float>(out_calib_precond_tril,
                             0 * out_calib_precond_tril_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = 0.00000000000000000e+00;
    write_sum_2<float, float>((float*)inout_shared, r35, r34);
  };
  flush_sum_shared<2, float>(out_calib_precond_tril,
                             4 * out_calib_precond_tril_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r27 * r15;
    r35 = r2 * r10;
    r39 = r0 * r39;
    r35 = r35 * r39;
    r34 = fmaf(r8, r35, r33 * r34);
    r0 = r8 * r9;
    r30 = r11 * r30;
    r30 = 1.0 / r30;
    r30 = r12 * r30;
    r12 = r18 * r27;
    r12 = r12 * r2;
    r12 = fmaf(r10, r12, r30 * r0);
    r0 = r7 * r7;
    r0 = r0 * r8;
    r12 = fmaf(r30, r0, r12);
    r11 = r18 * r17;
    r31 = r7 * r10;
    r12 = fmaf(r31, r11, r12);
    r1 = r1 * r33;
    r12 = r12 * r1;
    r34 = fmaf(r2, r12, r34);
    r11 = r8 * r31;
    r11 = fmaf(r39, r11, r7 * r12);
    r12 = r17 * r15;
    r11 = fmaf(r33, r12, r11);
    r12 = r21 * r15;
    r12 = fmaf(r33, r12, r19 * r35);
    r0 = r18 * r21;
    r0 = r0 * r2;
    r3 = r19 * r30;
    r0 = fmaf(r9, r3, r10 * r0);
    r36 = r7 * r7;
    r0 = fmaf(r3, r36, r0);
    r3 = r18 * r24;
    r0 = fmaf(r31, r3, r0);
    r3 = r2 * r0;
    r12 = fmaf(r1, r3, r12);
    r3 = r19 * r31;
    r36 = r7 * r0;
    r36 = fmaf(r1, r36, r39 * r3);
    r3 = r24 * r15;
    r36 = fmaf(r33, r3, r36);
    write_idx_4<1024, float, float, float4>(out_point_jac,
                                            0 * out_point_jac_num_alloc,
                                            global_thread_idx,
                                            r34,
                                            r11,
                                            r12,
                                            r36);
    r3 = r22 * r15;
    r3 = fmaf(r33, r3, r26 * r35);
    r35 = r18 * r22;
    r35 = r35 * r2;
    r25 = r26 * r9;
    r25 = fmaf(r30, r25, r10 * r35);
    r35 = r18 * r20;
    r25 = fmaf(r31, r35, r25);
    r10 = r7 * r7;
    r10 = r10 * r26;
    r25 = fmaf(r30, r10, r25);
    r10 = r2 * r25;
    r3 = fmaf(r1, r10, r3);
    r10 = r26 * r31;
    r35 = r20 * r15;
    r35 = fmaf(r33, r35, r39 * r10);
    r10 = r7 * r25;
    r35 = fmaf(r1, r10, r35);
    write_idx_2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r3, r35);
    r10 = r6 * r5;
    r1 = r6 * r4;
    r1 = fmaf(r34, r1, r11 * r10);
    r10 = r6 * r4;
    r33 = r6 * r5;
    r33 = fmaf(r36, r33, r12 * r10);
    r10 = r6 * r4;
    r39 = r6 * r5;
    r39 = fmaf(r35, r39, r3 * r10);
    write_sum_3<float, float>((float*)inout_shared, r1, r33, r39);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fmaf(r34, r34, r11 * r11);
    r33 = fmaf(r12, r12, r36 * r36);
    r1 = fmaf(r35, r35, r3 * r3);
    write_sum_3<float, float>((float*)inout_shared, r39, r33, r1);
  };
  flush_sum_shared<3, float>(out_point_precond_diag,
                             0 * out_point_precond_diag_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fmaf(r11, r36, r34 * r12);
    r11 = fmaf(r11, r35, r34 * r3);
    r35 = fmaf(r36, r35, r12 * r3);
    write_sum_3<float, float>((float*)inout_shared, r1, r11, r35);
  };
  flush_sum_shared<3, float>(out_point_precond_tril,
                             0 * out_point_precond_tril_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void simple_radial_merged_fixed_pose_res_jac_first(
    float* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* out_calib_jac,
    unsigned int out_calib_jac_num_alloc,
    float* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    float* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
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
  simple_radial_merged_fixed_pose_res_jac_first_kernel<<<n_blocks, 1024>>>(
      calib,
      calib_num_alloc,
      calib_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_calib_jac,
      out_calib_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
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