#include "kernel_simple_radial_fixed_focal_and_extra_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_focal_and_extra_fixed_principal_point_res_jac_first_kernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
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
    read_idx_2<1024, float, float, float2>(principal_point,
                                           0 * principal_point_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r1);
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    read_idx_2<1024, float, float, float2>(focal_and_extra,
                                           0 * focal_and_extra_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r5);
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
    r41 = fmaf(r5, r41, r26);
    r26 = 1.0 / r36;
    r37 = r41 * r26;
    r2 = fmaf(r30, r37, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r0 * r41;
    r1 = r1 * r26;
    r3 = fmaf(r7, r1, r3);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r37 = fmaf(r3, r3, r2 * r2);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r37);
  if (global_thread_idx < problem_size) {
    r37 = r12 * r13;
    r37 = r37 * r17;
    r31 = r31 + r37;
    r31 = fmaf(r10, r23, r11 * r31);
    r13 = r13 * r14;
    r13 = r13 * r17;
    r33 = r33 + r13;
    r42 = r4 * r27;
    r43 = r24 + r42;
    r15 = r15 * r15;
    r44 = r4 * r35;
    r45 = r15 + r44;
    r46 = r43 + r45;
    r46 = fmaf(r10, r46, r11 * r33);
    r33 = r46 * r30;
    r41 = r41 * r4;
    r41 = r41 * r8;
    r33 = fmaf(r41, r33, r31 * r1);
    r47 = r19 * r7;
    r48 = r4 * r15;
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
    r52 = r52 * r31;
    r47 = fmaf(r8, r52, r47);
    r47 = fmaf(r46, r51, r47);
    r52 = r5 * r47;
    r52 = r52 * r26;
    r33 = fmaf(r30, r52, r33);
    r52 = r0 * r47;
    r31 = r5 * r7;
    r52 = r52 * r26;
    r43 = fmaf(r43, r1, r31 * r52);
    r52 = r0 * r7;
    r52 = r52 * r46;
    r43 = fmaf(r41, r52, r43);
    r52 = r19 * r7;
    r13 = r39 + r13;
    r13 = fmaf(r9, r13, r11 * r18);
    r52 = r52 * r13;
    r39 = r19 * r6;
    r14 = r12 * r14;
    r14 = r14 * r17;
    r40 = r40 + r14;
    r15 = r35 + r15;
    r35 = r4 * r24;
    r15 = r15 + r42;
    r15 = r15 + r35;
    r15 = fmaf(r11, r15, r9 * r40);
    r39 = r39 * r15;
    r39 = fmaf(r8, r39, r8 * r52);
    r35 = r27 + r35;
    r49 = r49 + r35;
    r49 = fmaf(r9, r49, r11 * r21);
    r39 = fmaf(r49, r51, r39);
    r39 = fmaf(r49, r50, r39);
    r11 = r5 * r39;
    r11 = r11 * r26;
    r52 = r49 * r30;
    r52 = fmaf(r41, r52, r30 * r11);
    r52 = fmaf(r15, r1, r52);
    r15 = r0 * r39;
    r15 = r15 * r26;
    r15 = fmaf(r31, r15, r13 * r1);
    r13 = r0 * r7;
    r13 = r13 * r49;
    r15 = fmaf(r41, r13, r15);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            0 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r33,
                                            r43,
                                            r52,
                                            r15);
    r13 = r0 * r5;
    r13 = r13 * r19;
    r13 = r13 * r6;
    r13 = r13 * r7;
    r13 = r13 * r36;
    r24 = r27 + r24;
    r24 = r24 + r44;
    r24 = r24 + r48;
    r24 = fmaf(r10, r24, r9 * r20);
    r14 = r22 + r14;
    r14 = fmaf(r10, r14, r9 * r32);
    r22 = r14 * r30;
    r22 = fmaf(r41, r22, r24 * r1);
    r48 = r19 * r6;
    r48 = r48 * r24;
    r48 = fmaf(r14, r50, r8 * r48);
    r24 = r19 * r7;
    r37 = r16 + r37;
    r35 = r45 + r35;
    r35 = fmaf(r9, r35, r10 * r37);
    r24 = r24 * r35;
    r48 = fmaf(r8, r24, r48);
    r48 = fmaf(r14, r51, r48);
    r24 = r5 * r48;
    r24 = r24 * r26;
    r22 = fmaf(r30, r24, r22);
    r24 = r0 * r48;
    r24 = r24 * r26;
    r24 = fmaf(r31, r24, r35 * r1);
    r35 = r0 * r7;
    r35 = r35 * r14;
    r24 = fmaf(r41, r35, r24);
    r35 = r5 * r19;
    r35 = r35 * r6;
    r35 = r35 * r36;
    r35 = fmaf(r30, r35, r1);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            4 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r22,
                                            r24,
                                            r35,
                                            r13);
    r9 = r0 * r19;
    r9 = r9 * r7;
    r9 = r9 * r36;
    r9 = fmaf(r31, r9, r1);
    r37 = r50 + r51;
    r10 = r5 * r37;
    r10 = r10 * r26;
    r10 = fmaf(r30, r10, r30 * r41);
    r45 = r0 * r7;
    r16 = r0 * r37;
    r16 = r16 * r26;
    r16 = fmaf(r31, r16, r41 * r45);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            8 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r13,
                                            r9,
                                            r10,
                                            r16);
    r45 = r4 * r2;
    r44 = r4 * r3;
    r44 = fmaf(r43, r44, r33 * r45);
    r45 = r4 * r2;
    r27 = r4 * r3;
    r27 = fmaf(r15, r27, r52 * r45);
    r45 = r4 * r3;
    r11 = r4 * r2;
    r11 = fmaf(r22, r11, r24 * r45);
    r45 = r4 * r2;
    r36 = r17 * r36;
    r17 = r3 * r36;
    r40 = r30 * r31;
    r17 = fmaf(r40, r17, r35 * r45);
    write_sum_4<float, float>((float*)inout_shared, r44, r27, r11, r17);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r4 * r3;
    r11 = r4 * r2;
    r11 = fmaf(r10, r11, r16 * r17);
    r17 = r4 * r3;
    r27 = r2 * r36;
    r27 = fmaf(r40, r27, r9 * r17);
    write_sum_2<float, float>((float*)inout_shared, r27, r11);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r43, r43, r33 * r33);
    r27 = fmaf(r15, r15, r52 * r52);
    r17 = fmaf(r22, r22, r24 * r24);
    r44 = r0 * r5;
    r45 = 4.00000000000000000e+00;
    r38 = r38 * r38;
    r38 = 1.0 / r38;
    r44 = r44 * r6;
    r44 = r44 * r7;
    r44 = r44 * r45;
    r44 = r44 * r38;
    r44 = r44 * r40;
    r40 = fmaf(r35, r35, r44);
    write_sum_4<float, float>((float*)inout_shared, r11, r27, r17, r40);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fmaf(r9, r9, r44);
    r40 = fmaf(r10, r10, r16 * r16);
    write_sum_2<float, float>((float*)inout_shared, r44, r40);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fmaf(r33, r52, r43 * r15);
    r44 = fmaf(r33, r22, r43 * r24);
    r17 = fmaf(r43, r13, r33 * r35);
    r27 = fmaf(r33, r13, r43 * r9);
    write_sum_4<float, float>((float*)inout_shared, r40, r44, r17, r27);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fmaf(r33, r10, r43 * r16);
    r43 = fmaf(r15, r24, r52 * r22);
    r27 = fmaf(r15, r13, r52 * r35);
    r17 = fmaf(r52, r13, r15 * r9);
    write_sum_4<float, float>((float*)inout_shared, r33, r43, r27, r17);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r15, r16, r52 * r10);
    r52 = fmaf(r24, r16, r22 * r10);
    r17 = fmaf(r24, r13, r22 * r35);
    r22 = fmaf(r22, r13, r24 * r9);
    write_sum_4<float, float>((float*)inout_shared, r15, r17, r22, r52);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fmaf(r35, r13, r9 * r13);
    r35 = fmaf(r16, r13, r35 * r10);
    r13 = fmaf(r10, r13, r9 * r16);
    write_sum_3<float, float>((float*)inout_shared, r52, r35, r13);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r21 * r41;
    r35 = r19 * r29;
    r35 = r35 * r6;
    r35 = fmaf(r8, r35, r21 * r51);
    r52 = r19 * r18;
    r52 = r52 * r7;
    r35 = fmaf(r8, r52, r35);
    r35 = fmaf(r21, r50, r35);
    r52 = r5 * r35;
    r52 = r52 * r26;
    r52 = fmaf(r30, r52, r30 * r13);
    r52 = fmaf(r29, r1, r52);
    r21 = r0 * r7;
    r10 = r0 * r35;
    r10 = r10 * r26;
    r10 = fmaf(r31, r10, r13 * r21);
    r10 = fmaf(r18, r1, r10);
    r21 = r32 * r30;
    r21 = fmaf(r20, r1, r41 * r21);
    r13 = r19 * r20;
    r13 = r13 * r6;
    r13 = fmaf(r32, r51, r8 * r13);
    r16 = r19 * r25;
    r16 = r16 * r7;
    r13 = fmaf(r8, r16, r13);
    r13 = fmaf(r32, r50, r13);
    r16 = r5 * r13;
    r16 = r16 * r26;
    r21 = fmaf(r30, r16, r21);
    r16 = r0 * r13;
    r16 = r16 * r26;
    r9 = r0 * r7;
    r9 = r9 * r32;
    r9 = fmaf(r41, r9, r31 * r16);
    r9 = fmaf(r25, r1, r9);
    write_idx_4<1024, float, float, float4>(out_point_jac,
                                            0 * out_point_jac_num_alloc,
                                            global_thread_idx,
                                            r52,
                                            r10,
                                            r21,
                                            r9);
    r16 = r28 * r30;
    r22 = r19 * r23;
    r22 = r22 * r6;
    r51 = fmaf(r28, r51, r8 * r22);
    r22 = r19 * r34;
    r22 = r22 * r7;
    r51 = fmaf(r8, r22, r51);
    r51 = fmaf(r28, r50, r51);
    r50 = r5 * r51;
    r50 = r50 * r26;
    r50 = fmaf(r30, r50, r41 * r16);
    r50 = fmaf(r23, r1, r50);
    r16 = r0 * r7;
    r16 = r16 * r28;
    r22 = r0 * r51;
    r22 = r22 * r26;
    r22 = fmaf(r31, r22, r41 * r16);
    r22 = fmaf(r34, r1, r22);
    write_idx_2<1024, float, float, float2>(out_point_jac,
                                            4 * out_point_jac_num_alloc,
                                            global_thread_idx,
                                            r50,
                                            r22);
    r1 = r4 * r3;
    r16 = r4 * r2;
    r16 = fmaf(r52, r16, r10 * r1);
    r1 = r4 * r3;
    r31 = r4 * r2;
    r31 = fmaf(r21, r31, r9 * r1);
    r1 = r4 * r3;
    r41 = r4 * r2;
    r41 = fmaf(r50, r41, r22 * r1);
    write_sum_3<float, float>((float*)inout_shared, r16, r31, r41);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fmaf(r52, r52, r10 * r10);
    r31 = fmaf(r9, r9, r21 * r21);
    r16 = fmaf(r50, r50, r22 * r22);
    write_sum_3<float, float>((float*)inout_shared, r41, r31, r16);
  };
  flush_sum_shared<3, float>(out_point_precond_diag,
                             0 * out_point_precond_diag_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = fmaf(r10, r9, r52 * r21);
    r10 = fmaf(r10, r22, r52 * r50);
    r22 = fmaf(r9, r22, r21 * r50);
    write_sum_3<float, float>((float*)inout_shared, r16, r10, r22);
  };
  flush_sum_shared<3, float>(out_point_precond_tril,
                             0 * out_point_precond_tril_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_focal_and_extra_fixed_principal_point_res_jac_first(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
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
  simple_radial_fixed_focal_and_extra_fixed_principal_point_res_jac_first_kernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              point,
              point_num_alloc,
              point_indices,
              pixel,
              pixel_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              principal_point,
              principal_point_num_alloc,
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