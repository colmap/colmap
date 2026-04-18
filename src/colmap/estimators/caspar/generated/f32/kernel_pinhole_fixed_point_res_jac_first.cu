#include "kernel_pinhole_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_point_res_jac_first_kernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        float* extra_calib,
        unsigned int extra_calib_num_alloc,
        SharedIndex* extra_calib_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* out_pose_jac,
        unsigned int out_pose_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        float* out_focal_jac,
        unsigned int out_focal_jac_num_alloc,
        float* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        float* const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        float* const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc,
        float* out_extra_calib_jac,
        unsigned int out_extra_calib_jac_num_alloc,
        float* const out_extra_calib_njtr,
        unsigned int out_extra_calib_njtr_num_alloc,
        float* const out_extra_calib_precond_diag,
        unsigned int out_extra_calib_precond_diag_num_alloc,
        float* const out_extra_calib_precond_tril,
        unsigned int out_extra_calib_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex focal_indices_loc[1024];
  focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex extra_calib_indices_loc[1024];
  extra_calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? extra_calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46;
  load_shared<2, float, float>(extra_calib,
                               0 * extra_calib_num_alloc,
                               extra_calib_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>((float*)inout_shared,
                         extra_calib_indices_loc[threadIdx.x].target,
                         r0,
                         r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    r0 = 9.99999999999999955e-07;
  };
  load_shared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>(
        (float*)inout_shared, pose_indices_loc[threadIdx.x].target, r5, r6, r7);
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
    r15 = r12 * r13;
    r16 = 2.00000000000000000e+00;
    r15 = r15 * r16;
    r17 = r11 * r16;
    r18 = r14 * r17;
    r19 = r15 + r18;
    r7 = fmaf(r9, r19, r7);
    r20 = r12 * r14;
    r21 = -2.00000000000000000e+00;
    r20 = r20 * r21;
    r22 = r13 * r17;
    r23 = r20 + r22;
    r24 = r11 * r11;
    r24 = r24 * r21;
    r25 = 1.00000000000000000e+00;
    r26 = r12 * r12;
    r27 = fmaf(r21, r26, r25);
    r28 = r24 + r27;
    r7 = fmaf(r8, r23, r7);
    r7 = fmaf(r10, r28, r7);
    r28 = copysign(1.0, r7);
    r28 = fmaf(r0, r28, r7);
    r0 = 1.0 / r28;
  };
  load_shared<2, float, float>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>(
        (float*)inout_shared, focal_indices_loc[threadIdx.x].target, r7, r29);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r30 = r13 * r21;
    r31 = r14 * r30;
    r17 = r12 * r17;
    r32 = r31 + r17;
    r5 = fmaf(r9, r32, r5);
    r33 = r12 * r14;
    r33 = r33 * r16;
    r22 = r33 + r22;
    r34 = r13 * r30;
    r27 = r34 + r27;
    r5 = fmaf(r10, r22, r5);
    r5 = fmaf(r8, r27, r5);
    r27 = r7 * r5;
    r2 = fmaf(r0, r27, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r13 * r14;
    r1 = r1 * r16;
    r17 = r1 + r17;
    r6 = fmaf(r8, r17, r6);
    r16 = r11 * r14;
    r16 = r16 * r21;
    r15 = r15 + r16;
    r34 = r25 + r34;
    r34 = r34 + r24;
    r6 = fmaf(r10, r15, r6);
    r6 = fmaf(r9, r34, r6);
    r34 = r29 * r6;
    r3 = fmaf(r0, r34, r3);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r24 = fmaf(r2, r2, r3 * r3);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r24);
  if (global_thread_idx < problem_size) {
    r24 = r11 * r12;
    r24 = r24 * r21;
    r1 = r1 + r24;
    r22 = fmaf(r9, r22, r10 * r1);
    r1 = r7 * r22;
    r12 = r12 * r30;
    r16 = r16 + r12;
    r21 = r13 * r13;
    r35 = r4 * r26;
    r36 = r21 + r35;
    r14 = r14 * r14;
    r37 = r11 * r11;
    r37 = r37 * r4;
    r38 = r14 + r37;
    r39 = r36 + r38;
    r39 = fmaf(r9, r39, r10 * r16);
    r16 = r7 * r5;
    r40 = r28 * r28;
    r41 = 1.0 / r40;
    r16 = r16 * r4;
    r16 = r16 * r41;
    r1 = fmaf(r39, r16, r0 * r1);
    r42 = r11 * r11;
    r43 = r4 * r14;
    r44 = r42 + r43;
    r36 = r36 + r44;
    r36 = fmaf(r10, r36, r9 * r15);
    r15 = r29 * r36;
    r45 = r4 * r41;
    r46 = r39 * r45;
    r46 = fmaf(r34, r46, r0 * r15);
    r30 = r11 * r30;
    r20 = r20 + r30;
    r14 = r42 + r14;
    r13 = r13 * r13;
    r13 = r13 * r4;
    r14 = r14 + r35;
    r14 = r14 + r13;
    r14 = fmaf(r10, r14, r8 * r20);
    r20 = r7 * r14;
    r13 = r26 + r13;
    r44 = r44 + r13;
    r44 = fmaf(r8, r44, r10 * r23);
    r20 = fmaf(r44, r16, r0 * r20);
    r12 = r18 + r12;
    r12 = fmaf(r8, r12, r10 * r17);
    r17 = r29 * r12;
    r10 = r44 * r45;
    r10 = fmaf(r34, r10, r0 * r17);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            0 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r1,
                                            r46,
                                            r20,
                                            r10);
    r17 = r7 * r0;
    r18 = r29 * r0;
    r21 = r26 + r21;
    r21 = r21 + r37;
    r21 = r21 + r43;
    r21 = fmaf(r9, r21, r8 * r32);
    r32 = r7 * r21;
    r30 = r33 + r30;
    r30 = fmaf(r9, r30, r8 * r19);
    r32 = fmaf(r30, r16, r0 * r32);
    r19 = r30 * r45;
    r31 = r24 + r31;
    r13 = r38 + r13;
    r13 = fmaf(r8, r13, r9 * r31);
    r8 = r29 * r13;
    r8 = fmaf(r0, r8, r34 * r19);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            4 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r32,
                                            r8,
                                            r17,
                                            r18);
    r18 = r45 * r34;
    write_idx_2<1024, float, float, float2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r16, r18);
    r18 = r4 * r2;
    r17 = r4 * r3;
    r17 = fmaf(r46, r17, r1 * r18);
    r18 = r4 * r2;
    r19 = r4 * r3;
    r19 = fmaf(r10, r19, r20 * r18);
    r18 = r4 * r3;
    r31 = r4 * r2;
    r31 = fmaf(r32, r31, r8 * r18);
    r18 = r4 * r2;
    r18 = r18 * r0;
    r9 = r7 * r18;
    write_sum_4<float, float>((float*)inout_shared, r17, r19, r31, r9);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r29 * r4;
    r9 = r9 * r3;
    r9 = r9 * r0;
    r31 = r2 * r41;
    r19 = r3 * r41;
    r19 = fmaf(r34, r19, r27 * r31);
    write_sum_2<float, float>((float*)inout_shared, r9, r19);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r7 * r7;
    r19 = r19 * r41;
    r9 = fmaf(r1, r1, r46 * r46);
    r31 = fmaf(r20, r20, r10 * r10);
    r17 = fmaf(r32, r32, r8 * r8);
    write_sum_4<float, float>((float*)inout_shared, r9, r31, r17, r19);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r29 * r29;
    r19 = r19 * r41;
    r17 = r7 * r5;
    r40 = r28 * r40;
    r28 = r28 * r40;
    r28 = 1.0 / r28;
    r17 = r17 * r28;
    r31 = r6 * r28;
    r9 = r29 * r34;
    r31 = fmaf(r9, r31, r27 * r17);
    write_sum_2<float, float>((float*)inout_shared, r19, r31);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r46, r10, r1 * r20);
    r19 = fmaf(r46, r8, r1 * r32);
    r17 = r7 * r1;
    r17 = r17 * r0;
    r38 = r29 * r46;
    r38 = r38 * r0;
    write_sum_4<float, float>((float*)inout_shared, r31, r19, r17, r38);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fmaf(r20, r32, r10 * r8);
    r17 = r7 * r20;
    r17 = r17 * r0;
    r19 = r29 * r10;
    r19 = r19 * r0;
    r31 = r46 * r45;
    r31 = fmaf(r34, r31, r1 * r16);
    write_sum_4<float, float>((float*)inout_shared, r31, r38, r17, r19);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r7 * r32;
    r19 = r19 * r0;
    r17 = r29 * r8;
    r17 = r17 * r0;
    r38 = r10 * r45;
    r38 = fmaf(r34, r38, r20 * r16);
    r20 = r8 * r45;
    r20 = fmaf(r34, r20, r32 * r16);
    write_sum_4<float, float>((float*)inout_shared, r38, r19, r17, r20);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = 0.00000000000000000e+00;
    r17 = r7 * r4;
    r40 = 1.0 / r40;
    r17 = r17 * r40;
    r17 = r17 * r27;
    r40 = r4 * r40;
    r40 = r40 * r9;
    write_sum_3<float, float>((float*)inout_shared, r20, r17, r40);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r5 * r0;
    r17 = r6 * r0;
    write_idx_2<1024, float, float, float2>(out_focal_jac,
                                            0 * out_focal_jac_num_alloc,
                                            global_thread_idx,
                                            r40,
                                            r17);
    r18 = r5 * r18;
    r17 = r4 * r6;
    r17 = r17 * r3;
    r17 = r17 * r0;
    write_sum_2<float, float>((float*)inout_shared, r18, r17);
  };
  flush_sum_shared<2, float>(out_focal_njtr,
                             0 * out_focal_njtr_num_alloc,
                             focal_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r5 * r5;
    r17 = r17 * r41;
    r18 = r6 * r6;
    r18 = r18 * r41;
    write_sum_2<float, float>((float*)inout_shared, r17, r18);
  };
  flush_sum_shared<2, float>(out_focal_precond_diag,
                             0 * out_focal_precond_diag_num_alloc,
                             focal_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = r4 * r2;
    r17 = r4 * r3;
    write_sum_2<float, float>((float*)inout_shared, r18, r17);
  };
  flush_sum_shared<2, float>(out_extra_calib_njtr,
                             0 * out_extra_calib_njtr_num_alloc,
                             extra_calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<float, float>((float*)inout_shared, r25, r25);
  };
  flush_sum_shared<2, float>(out_extra_calib_precond_diag,
                             0 * out_extra_calib_precond_diag_num_alloc,
                             extra_calib_indices_loc,
                             (float*)inout_shared);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void pinhole_fixed_point_res_jac_first(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    float* extra_calib,
    unsigned int extra_calib_num_alloc,
    SharedIndex* extra_calib_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float* out_focal_jac,
    unsigned int out_focal_jac_num_alloc,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    float* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    float* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    float* out_extra_calib_jac,
    unsigned int out_extra_calib_jac_num_alloc,
    float* const out_extra_calib_njtr,
    unsigned int out_extra_calib_njtr_num_alloc,
    float* const out_extra_calib_precond_diag,
    unsigned int out_extra_calib_precond_diag_num_alloc,
    float* const out_extra_calib_precond_tril,
    unsigned int out_extra_calib_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_point_res_jac_first_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      focal,
      focal_num_alloc,
      focal_indices,
      extra_calib,
      extra_calib_num_alloc,
      extra_calib_indices,
      pixel,
      pixel_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      out_focal_jac,
      out_focal_jac_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_focal_precond_diag,
      out_focal_precond_diag_num_alloc,
      out_focal_precond_tril,
      out_focal_precond_tril_num_alloc,
      out_extra_calib_jac,
      out_extra_calib_jac_num_alloc,
      out_extra_calib_njtr,
      out_extra_calib_njtr_num_alloc,
      out_extra_calib_precond_diag,
      out_extra_calib_precond_diag_num_alloc,
      out_extra_calib_precond_tril,
      out_extra_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar