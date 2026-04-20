#include "kernel_simple_radial_fixed_focal_fixed_extra_calib_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_focal_fixed_extra_calib_res_jac_first_kernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
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
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52;

  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        extra_calib, 0 * extra_calib_num_alloc, global_thread_idx, r0, r1, r2);
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r3, r4);
    r5 = -1.00000000000000000e+00;
    r3 = fmaf(r3, r5, r0);
    read_idx_1<1024, float, float, float>(
        focal, 0 * focal_num_alloc, global_thread_idx, r0);
  };
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
    r16 = r14 * r15;
    r17 = -2.00000000000000000e+00;
    r16 = r16 * r17;
    r18 = r12 * r13;
    r19 = 2.00000000000000000e+00;
    r18 = r18 * r19;
    r20 = r16 + r18;
    r6 = fmaf(r10, r20, r6);
    r21 = r12 * r14;
    r21 = r21 * r19;
    r22 = r13 * r15;
    r22 = r22 * r19;
    r23 = r21 + r22;
    r24 = r14 * r14;
    r25 = r17 * r24;
    r26 = 1.00000000000000000e+00;
    r27 = r13 * r13;
    r28 = fmaf(r17, r27, r26);
    r29 = r25 + r28;
    r6 = fmaf(r11, r23, r6);
    r6 = fmaf(r9, r29, r6);
    r30 = r0 * r6;
    r31 = r14 * r15;
    r31 = r31 * r19;
    r18 = r18 + r31;
    r7 = fmaf(r9, r18, r7);
    r32 = r13 * r14;
    r32 = r32 * r19;
    r33 = r12 * r15;
    r33 = r33 * r17;
    r34 = r32 + r33;
    r25 = r26 + r25;
    r35 = r12 * r12;
    r36 = r17 * r35;
    r25 = r25 + r36;
    r7 = fmaf(r11, r34, r7);
    r7 = fmaf(r10, r25, r7);
    r37 = r7 * r7;
    r38 = 9.99999999999999955e-07;
    r39 = r12 * r15;
    r39 = r39 * r19;
    r32 = r32 + r39;
    r8 = fmaf(r10, r32, r8);
    r40 = r13 * r15;
    r40 = r40 * r17;
    r21 = r21 + r40;
    r28 = r36 + r28;
    r8 = fmaf(r9, r21, r8);
    r8 = fmaf(r11, r28, r8);
    r36 = copysign(1.0, r8);
    r36 = fmaf(r38, r36, r8);
    r38 = r36 * r36;
    r8 = 1.0 / r38;
    r41 = r6 * r6;
    r41 = fmaf(r8, r41, r8 * r37);
    r41 = fmaf(r2, r41, r26);
    r26 = 1.0 / r36;
    r37 = r41 * r26;
    r3 = fmaf(r30, r37, r3);
    r4 = fmaf(r4, r5, r1);
    r1 = r0 * r41;
    r1 = r1 * r26;
    r4 = fmaf(r7, r1, r4);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r3, r4);
    r37 = fmaf(r3, r3, r4 * r4);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r37);
  if (global_thread_idx < problem_size) {
    r37 = r13 * r14;
    r37 = r37 * r17;
    r33 = r33 + r37;
    r42 = r5 * r27;
    r43 = r24 + r42;
    r15 = r15 * r15;
    r44 = r5 * r35;
    r45 = r15 + r44;
    r46 = r43 + r45;
    r46 = fmaf(r10, r46, r11 * r33);
    r33 = r46 * r30;
    r41 = r41 * r5;
    r41 = r41 * r8;
    r47 = r19 * r7;
    r48 = r5 * r15;
    r49 = r35 + r48;
    r43 = r43 + r49;
    r43 = fmaf(r11, r43, r10 * r34);
    r47 = r47 * r43;
    r50 = r7 * r7;
    r50 = r17 * r50;
    r38 = r36 * r38;
    r36 = 1.0 / r38;
    r50 = r50 * r36;
    r47 = fmaf(r46, r50, r8 * r47);
    r51 = r6 * r6;
    r51 = r17 * r51;
    r51 = r51 * r36;
    r52 = r19 * r6;
    r13 = r12 * r13;
    r13 = r13 * r17;
    r31 = r31 + r13;
    r31 = fmaf(r10, r23, r11 * r31);
    r52 = r52 * r31;
    r47 = fmaf(r8, r52, r47);
    r47 = fmaf(r46, r51, r47);
    r52 = r2 * r47;
    r52 = r52 * r26;
    r52 = fmaf(r30, r52, r41 * r33);
    r52 = fmaf(r31, r1, r52);
    r31 = r0 * r47;
    r33 = r2 * r7;
    r31 = r31 * r26;
    r43 = fmaf(r43, r1, r33 * r31);
    r31 = r0 * r7;
    r31 = r31 * r46;
    r43 = fmaf(r41, r31, r43);
    r31 = r19 * r7;
    r37 = r39 + r37;
    r37 = fmaf(r9, r37, r11 * r18);
    r31 = r31 * r37;
    r39 = r19 * r6;
    r14 = r12 * r14;
    r14 = r14 * r17;
    r40 = r40 + r14;
    r15 = r35 + r15;
    r35 = r5 * r24;
    r15 = r15 + r42;
    r15 = r15 + r35;
    r15 = fmaf(r11, r15, r9 * r40);
    r39 = r39 * r15;
    r39 = fmaf(r8, r39, r8 * r31);
    r35 = r27 + r35;
    r49 = r49 + r35;
    r49 = fmaf(r9, r49, r11 * r21);
    r39 = fmaf(r49, r51, r39);
    r39 = fmaf(r49, r50, r39);
    r11 = r2 * r39;
    r11 = r11 * r26;
    r15 = fmaf(r15, r1, r30 * r11);
    r11 = r49 * r30;
    r15 = fmaf(r41, r11, r15);
    r11 = r0 * r39;
    r11 = r11 * r26;
    r31 = r0 * r7;
    r31 = r31 * r49;
    r31 = fmaf(r41, r31, r33 * r11);
    r31 = fmaf(r37, r1, r31);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            0 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r52,
                                            r43,
                                            r15,
                                            r31);
    r37 = r0 * r2;
    r37 = r37 * r19;
    r37 = r37 * r6;
    r37 = r37 * r7;
    r37 = r37 * r36;
    r14 = r22 + r14;
    r14 = fmaf(r10, r14, r9 * r32);
    r22 = r14 * r30;
    r11 = r19 * r6;
    r24 = r27 + r24;
    r24 = r24 + r44;
    r24 = r24 + r48;
    r24 = fmaf(r10, r24, r9 * r20);
    r11 = r11 * r24;
    r11 = fmaf(r14, r50, r8 * r11);
    r48 = r19 * r7;
    r13 = r16 + r13;
    r35 = r45 + r35;
    r35 = fmaf(r9, r35, r10 * r13);
    r48 = r48 * r35;
    r11 = fmaf(r8, r48, r11);
    r11 = fmaf(r14, r51, r11);
    r48 = r2 * r11;
    r48 = r48 * r26;
    r48 = fmaf(r30, r48, r41 * r22);
    r48 = fmaf(r24, r1, r48);
    r24 = r0 * r7;
    r24 = r24 * r14;
    r22 = r0 * r11;
    r22 = r22 * r26;
    r22 = fmaf(r33, r22, r41 * r24);
    r22 = fmaf(r35, r1, r22);
    r35 = r2 * r19;
    r35 = r35 * r6;
    r35 = r35 * r36;
    r35 = fmaf(r30, r35, r1);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            4 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r48,
                                            r22,
                                            r35,
                                            r37);
    r24 = r0 * r19;
    r24 = r24 * r7;
    r24 = r24 * r36;
    r24 = fmaf(r33, r24, r1);
    r9 = r50 + r51;
    r13 = r2 * r9;
    r13 = r13 * r26;
    r13 = fmaf(r30, r13, r30 * r41);
    r10 = r0 * r9;
    r10 = r10 * r26;
    r45 = r0 * r7;
    r45 = fmaf(r41, r45, r33 * r10);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            8 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r37,
                                            r24,
                                            r13,
                                            r45);
    r10 = r5 * r4;
    r16 = r5 * r3;
    r16 = fmaf(r52, r16, r43 * r10);
    r10 = r5 * r3;
    r44 = r5 * r4;
    r44 = fmaf(r31, r44, r15 * r10);
    r10 = r5 * r3;
    r27 = r5 * r4;
    r27 = fmaf(r22, r27, r48 * r10);
    r10 = r5 * r3;
    r36 = r17 * r36;
    r17 = r4 * r36;
    r40 = r30 * r33;
    r17 = fmaf(r40, r17, r35 * r10);
    write_sum_4<float, float>((float*)inout_shared, r16, r44, r27, r17);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r5 * r4;
    r27 = r5 * r3;
    r27 = fmaf(r13, r27, r45 * r17);
    r17 = r5 * r4;
    r44 = r3 * r36;
    r44 = fmaf(r40, r44, r24 * r17);
    write_sum_2<float, float>((float*)inout_shared, r44, r27);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fmaf(r43, r43, r52 * r52);
    r44 = fmaf(r15, r15, r31 * r31);
    r17 = fmaf(r48, r48, r22 * r22);
    r16 = r0 * r2;
    r10 = 4.00000000000000000e+00;
    r38 = r38 * r38;
    r38 = 1.0 / r38;
    r16 = r16 * r6;
    r16 = r16 * r7;
    r16 = r16 * r10;
    r16 = r16 * r38;
    r16 = r16 * r40;
    r40 = fmaf(r35, r35, r16);
    write_sum_4<float, float>((float*)inout_shared, r27, r44, r17, r40);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = fmaf(r24, r24, r16);
    r40 = fmaf(r45, r45, r13 * r13);
    write_sum_2<float, float>((float*)inout_shared, r16, r40);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fmaf(r52, r15, r43 * r31);
    r16 = fmaf(r43, r22, r52 * r48);
    r17 = fmaf(r43, r37, r52 * r35);
    r44 = fmaf(r52, r37, r43 * r24);
    write_sum_4<float, float>((float*)inout_shared, r40, r16, r17, r44);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fmaf(r52, r13, r43 * r45);
    r43 = fmaf(r15, r48, r31 * r22);
    r44 = fmaf(r31, r37, r15 * r35);
    r17 = fmaf(r15, r37, r31 * r24);
    write_sum_4<float, float>((float*)inout_shared, r52, r43, r44, r17);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r31, r45, r15 * r13);
    r15 = fmaf(r22, r45, r48 * r13);
    r17 = fmaf(r22, r37, r48 * r35);
    r48 = fmaf(r48, r37, r22 * r24);
    write_sum_4<float, float>((float*)inout_shared, r31, r17, r48, r15);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r35, r37, r24 * r37);
    r35 = fmaf(r45, r37, r35 * r13);
    r37 = fmaf(r13, r37, r24 * r45);
    write_sum_3<float, float>((float*)inout_shared, r15, r35, r37);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r19 * r29;
    r37 = r37 * r6;
    r37 = fmaf(r8, r37, r21 * r51);
    r35 = r19 * r18;
    r35 = r35 * r7;
    r37 = fmaf(r8, r35, r37);
    r37 = fmaf(r21, r50, r37);
    r35 = r2 * r37;
    r35 = r35 * r26;
    r35 = fmaf(r29, r1, r30 * r35);
    r21 = r21 * r41;
    r35 = fmaf(r30, r21, r35);
    r15 = r0 * r7;
    r13 = r0 * r37;
    r13 = r13 * r26;
    r13 = fmaf(r33, r13, r21 * r15);
    r13 = fmaf(r18, r1, r13);
    r15 = r19 * r20;
    r15 = r15 * r6;
    r15 = fmaf(r32, r51, r8 * r15);
    r21 = r19 * r25;
    r21 = r21 * r7;
    r15 = fmaf(r8, r21, r15);
    r15 = fmaf(r32, r50, r15);
    r21 = r2 * r15;
    r21 = r21 * r26;
    r21 = fmaf(r20, r1, r30 * r21);
    r45 = r32 * r30;
    r21 = fmaf(r41, r45, r21);
    r45 = r0 * r7;
    r45 = r45 * r32;
    r45 = fmaf(r41, r45, r25 * r1);
    r24 = r0 * r15;
    r24 = r24 * r26;
    r45 = fmaf(r33, r24, r45);
    write_idx_4<1024, float, float, float4>(out_point_jac,
                                            0 * out_point_jac_num_alloc,
                                            global_thread_idx,
                                            r35,
                                            r13,
                                            r21,
                                            r45);
    r24 = r28 * r30;
    r24 = fmaf(r41, r24, r23 * r1);
    r48 = r19 * r23;
    r48 = r48 * r6;
    r51 = fmaf(r28, r51, r8 * r48);
    r48 = r19 * r34;
    r48 = r48 * r7;
    r51 = fmaf(r8, r48, r51);
    r51 = fmaf(r28, r50, r51);
    r50 = r2 * r51;
    r50 = r50 * r26;
    r24 = fmaf(r30, r50, r24);
    r50 = r0 * r7;
    r50 = r50 * r28;
    r1 = fmaf(r34, r1, r41 * r50);
    r50 = r0 * r51;
    r50 = r50 * r26;
    r1 = fmaf(r33, r50, r1);
    write_idx_2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r24, r1);
    r50 = r5 * r3;
    r33 = r5 * r4;
    r33 = fmaf(r13, r33, r35 * r50);
    r50 = r5 * r3;
    r26 = r5 * r4;
    r26 = fmaf(r45, r26, r21 * r50);
    r50 = r5 * r3;
    r41 = r5 * r4;
    r41 = fmaf(r1, r41, r24 * r50);
    write_sum_3<float, float>((float*)inout_shared, r33, r26, r41);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fmaf(r13, r13, r35 * r35);
    r26 = fmaf(r45, r45, r21 * r21);
    r33 = fmaf(r1, r1, r24 * r24);
    write_sum_3<float, float>((float*)inout_shared, r41, r26, r33);
  };
  flush_sum_shared<3, float>(out_point_precond_diag,
                             0 * out_point_precond_diag_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fmaf(r13, r45, r35 * r21);
    r13 = fmaf(r13, r1, r35 * r24);
    r1 = fmaf(r45, r1, r21 * r24);
    write_sum_3<float, float>((float*)inout_shared, r33, r13, r1);
  };
  flush_sum_shared<3, float>(out_point_precond_tril,
                             0 * out_point_precond_tril_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_focal_fixed_extra_calib_res_jac_first(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
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
  simple_radial_fixed_focal_fixed_extra_calib_res_jac_first_kernel<<<n_blocks,
                                                                     1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      focal,
      focal_num_alloc,
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