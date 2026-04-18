#include "kernel_pinhole_fixed_extra_calib_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_extra_calib_res_jac_first_kernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* extra_calib,
        unsigned int extra_calib_num_alloc,
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
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        extra_calib, 0 * extra_calib_num_alloc, global_thread_idx, r0, r1);
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
    r29 = copysign(1.0, r7);
    r29 = fmaf(r0, r29, r7);
    r0 = 1.0 / r29;
  };
  load_shared<2, float, float>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>(
        (float*)inout_shared, focal_indices_loc[threadIdx.x].target, r7, r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r31 = r13 * r21;
    r32 = r14 * r31;
    r17 = r12 * r17;
    r33 = r32 + r17;
    r5 = fmaf(r9, r33, r5);
    r34 = r12 * r14;
    r34 = r34 * r16;
    r22 = r34 + r22;
    r35 = r13 * r31;
    r27 = r35 + r27;
    r5 = fmaf(r10, r22, r5);
    r5 = fmaf(r8, r27, r5);
    r36 = r7 * r5;
    r2 = fmaf(r0, r36, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r13 * r14;
    r1 = r1 * r16;
    r17 = r1 + r17;
    r6 = fmaf(r8, r17, r6);
    r16 = r11 * r14;
    r16 = r16 * r21;
    r15 = r15 + r16;
    r35 = r25 + r35;
    r35 = r35 + r24;
    r6 = fmaf(r10, r15, r6);
    r6 = fmaf(r9, r35, r6);
    r24 = r30 * r6;
    r3 = fmaf(r0, r24, r3);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r25 = fmaf(r2, r2, r3 * r3);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r25);
  if (global_thread_idx < problem_size) {
    r25 = r11 * r12;
    r25 = r25 * r21;
    r1 = r1 + r25;
    r1 = fmaf(r9, r22, r10 * r1);
    r21 = r7 * r1;
    r12 = r12 * r31;
    r16 = r16 + r12;
    r37 = r13 * r13;
    r38 = r4 * r26;
    r39 = r37 + r38;
    r14 = r14 * r14;
    r40 = r11 * r11;
    r40 = r40 * r4;
    r41 = r14 + r40;
    r42 = r39 + r41;
    r42 = fmaf(r9, r42, r10 * r16);
    r16 = r7 * r5;
    r43 = r29 * r29;
    r44 = 1.0 / r43;
    r16 = r16 * r4;
    r16 = r16 * r44;
    r21 = fmaf(r42, r16, r0 * r21);
    r45 = r11 * r11;
    r46 = r4 * r14;
    r47 = r45 + r46;
    r39 = r39 + r47;
    r39 = fmaf(r10, r39, r9 * r15);
    r48 = r30 * r39;
    r49 = r4 * r44;
    r50 = r42 * r49;
    r50 = fmaf(r24, r50, r0 * r48);
    r31 = r11 * r31;
    r20 = r20 + r31;
    r14 = r45 + r14;
    r13 = r13 * r13;
    r13 = r13 * r4;
    r14 = r14 + r38;
    r14 = r14 + r13;
    r14 = fmaf(r10, r14, r8 * r20);
    r20 = r7 * r14;
    r13 = r26 + r13;
    r47 = r47 + r13;
    r47 = fmaf(r8, r47, r10 * r23);
    r20 = fmaf(r47, r16, r0 * r20);
    r12 = r18 + r12;
    r12 = fmaf(r8, r12, r10 * r17);
    r10 = r30 * r12;
    r18 = r47 * r49;
    r18 = fmaf(r24, r18, r0 * r10);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            0 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r21,
                                            r50,
                                            r20,
                                            r18);
    r10 = r7 * r0;
    r38 = r30 * r0;
    r37 = r26 + r37;
    r37 = r37 + r40;
    r37 = r37 + r46;
    r37 = fmaf(r9, r37, r8 * r33);
    r46 = r7 * r37;
    r31 = r34 + r31;
    r31 = fmaf(r9, r31, r8 * r19);
    r46 = fmaf(r31, r16, r0 * r46);
    r34 = r31 * r49;
    r32 = r25 + r32;
    r13 = r41 + r13;
    r13 = fmaf(r8, r13, r9 * r32);
    r8 = r30 * r13;
    r8 = fmaf(r0, r8, r24 * r34);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            4 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r46,
                                            r8,
                                            r10,
                                            r38);
    r38 = r49 * r24;
    write_idx_2<1024, float, float, float2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r16, r38);
    r38 = r4 * r2;
    r10 = r4 * r3;
    r10 = fmaf(r50, r10, r21 * r38);
    r38 = r4 * r2;
    r34 = r4 * r3;
    r34 = fmaf(r18, r34, r20 * r38);
    r38 = r4 * r3;
    r32 = r4 * r2;
    r32 = fmaf(r46, r32, r8 * r38);
    r38 = r4 * r2;
    r38 = r38 * r0;
    r9 = r7 * r38;
    write_sum_4<float, float>((float*)inout_shared, r10, r34, r32, r9);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r30 * r4;
    r9 = r9 * r3;
    r9 = r9 * r0;
    r32 = r2 * r44;
    r34 = r3 * r44;
    r34 = fmaf(r24, r34, r36 * r32);
    write_sum_2<float, float>((float*)inout_shared, r9, r34);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r7 * r7;
    r34 = r34 * r44;
    r9 = fmaf(r21, r21, r50 * r50);
    r32 = fmaf(r20, r20, r18 * r18);
    r10 = fmaf(r46, r46, r8 * r8);
    write_sum_4<float, float>((float*)inout_shared, r9, r32, r10, r34);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r30 * r30;
    r34 = r34 * r44;
    r10 = r7 * r5;
    r43 = r29 * r43;
    r29 = r29 * r43;
    r29 = 1.0 / r29;
    r10 = r10 * r29;
    r32 = r6 * r29;
    r9 = r30 * r24;
    r32 = fmaf(r9, r32, r36 * r10);
    write_sum_2<float, float>((float*)inout_shared, r34, r32);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fmaf(r50, r18, r21 * r20);
    r34 = fmaf(r50, r8, r21 * r46);
    r10 = r7 * r21;
    r10 = r10 * r0;
    r41 = r30 * r50;
    r41 = r41 * r0;
    write_sum_4<float, float>((float*)inout_shared, r32, r34, r10, r41);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fmaf(r20, r46, r18 * r8);
    r10 = r7 * r20;
    r10 = r10 * r0;
    r34 = r30 * r18;
    r34 = r34 * r0;
    r32 = r50 * r49;
    r32 = fmaf(r24, r32, r21 * r16);
    write_sum_4<float, float>((float*)inout_shared, r32, r41, r10, r34);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r7 * r46;
    r34 = r34 * r0;
    r10 = r30 * r8;
    r10 = r10 * r0;
    r41 = r18 * r49;
    r41 = fmaf(r24, r41, r20 * r16);
    r20 = r8 * r49;
    r20 = fmaf(r24, r20, r46 * r16);
    write_sum_4<float, float>((float*)inout_shared, r41, r34, r10, r20);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = 0.00000000000000000e+00;
    r10 = r7 * r4;
    r43 = 1.0 / r43;
    r10 = r10 * r43;
    r10 = r10 * r36;
    r43 = r4 * r43;
    r43 = r43 * r9;
    write_sum_3<float, float>((float*)inout_shared, r20, r10, r43);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r5 * r0;
    r10 = r6 * r0;
    write_idx_2<1024, float, float, float2>(out_focal_jac,
                                            0 * out_focal_jac_num_alloc,
                                            global_thread_idx,
                                            r43,
                                            r10);
    r38 = r5 * r38;
    r10 = r4 * r6;
    r10 = r10 * r3;
    r10 = r10 * r0;
    write_sum_2<float, float>((float*)inout_shared, r38, r10);
  };
  flush_sum_shared<2, float>(out_focal_njtr,
                             0 * out_focal_njtr_num_alloc,
                             focal_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = r5 * r5;
    r10 = r10 * r44;
    r38 = r6 * r6;
    r38 = r38 * r44;
    write_sum_2<float, float>((float*)inout_shared, r10, r38);
  };
  flush_sum_shared<2, float>(out_focal_precond_diag,
                             0 * out_focal_precond_diag_num_alloc,
                             focal_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r7 * r27;
    r38 = fmaf(r23, r16, r0 * r38);
    r10 = r23 * r49;
    r43 = r30 * r17;
    r43 = fmaf(r0, r43, r24 * r10);
    r10 = r7 * r33;
    r10 = fmaf(r0, r10, r19 * r16);
    r20 = r30 * r35;
    r9 = r19 * r49;
    r9 = fmaf(r24, r9, r0 * r20);
    write_idx_4<1024, float, float, float4>(out_point_jac,
                                            0 * out_point_jac_num_alloc,
                                            global_thread_idx,
                                            r38,
                                            r43,
                                            r10,
                                            r9);
    r20 = r7 * r22;
    r20 = fmaf(r0, r20, r28 * r16);
    r16 = r28 * r49;
    r36 = r30 * r15;
    r36 = fmaf(r0, r36, r24 * r16);
    write_idx_2<1024, float, float, float2>(out_point_jac,
                                            4 * out_point_jac_num_alloc,
                                            global_thread_idx,
                                            r20,
                                            r36);
    r16 = r4 * r2;
    r0 = r4 * r3;
    r0 = fmaf(r43, r0, r38 * r16);
    r16 = r4 * r2;
    r24 = r4 * r3;
    r24 = fmaf(r9, r24, r10 * r16);
    r16 = r4 * r3;
    r34 = r4 * r2;
    r34 = fmaf(r20, r34, r36 * r16);
    write_sum_3<float, float>((float*)inout_shared, r0, r24, r34);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fmaf(r38, r38, r43 * r43);
    r24 = fmaf(r9, r9, r10 * r10);
    r0 = fmaf(r20, r20, r36 * r36);
    write_sum_3<float, float>((float*)inout_shared, r34, r24, r0);
  };
  flush_sum_shared<3, float>(out_point_precond_diag,
                             0 * out_point_precond_diag_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fmaf(r43, r9, r38 * r10);
    r43 = fmaf(r43, r36, r38 * r20);
    r20 = fmaf(r10, r20, r9 * r36);
    write_sum_3<float, float>((float*)inout_shared, r0, r43, r20);
  };
  flush_sum_shared<3, float>(out_point_precond_tril,
                             0 * out_point_precond_tril_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void pinhole_fixed_extra_calib_res_jac_first(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* extra_calib,
    unsigned int extra_calib_num_alloc,
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
  pinhole_fixed_extra_calib_res_jac_first_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      focal,
      focal_num_alloc,
      focal_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      extra_calib,
      extra_calib_num_alloc,
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