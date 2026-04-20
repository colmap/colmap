#include "kernel_simple_radial_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_point_res_jac_first_kernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
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
        float* out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        float* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        float* const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        float* const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
        float* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        float* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
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
  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51;
  load_shared<2, float, float>(principal_point,
                               0 * principal_point_num_alloc,
                               principal_point_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>((float*)inout_shared,
                         principal_point_indices_loc[threadIdx.x].target,
                         r0,
                         r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r9, r10, r11);
  };
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
    r18 = -2.00000000000000000e+00;
    r19 = r14 * r18;
    r20 = r15 * r19;
    r21 = r16 + r20;
    r6 = fmaf(r10, r21, r6);
    r22 = r12 * r14;
    r22 = r22 * r17;
    r23 = r13 * r15;
    r23 = r23 * r17;
    r24 = r22 + r23;
    r25 = r14 * r19;
    r26 = 1.00000000000000000e+00;
    r27 = r13 * r13;
    r28 = fmaf(r18, r27, r26);
    r29 = r25 + r28;
    r6 = fmaf(r11, r24, r6);
    r6 = fmaf(r9, r29, r6);
    r29 = r0 * r6;
    r30 = r14 * r15;
    r30 = r30 * r17;
    r16 = r16 + r30;
    r7 = fmaf(r9, r16, r7);
    r31 = r13 * r14;
    r31 = r31 * r17;
    r32 = r12 * r15;
    r32 = r32 * r18;
    r33 = r31 + r32;
    r25 = r26 + r25;
    r34 = r12 * r12;
    r35 = r18 * r34;
    r25 = r25 + r35;
    r7 = fmaf(r11, r33, r7);
    r7 = fmaf(r10, r25, r7);
    r25 = r7 * r7;
    r36 = 9.99999999999999955e-07;
    r37 = r12 * r15;
    r37 = r37 * r17;
    r31 = r31 + r37;
    r8 = fmaf(r10, r31, r8);
    r38 = r13 * r15;
    r38 = r38 * r18;
    r22 = r22 + r38;
    r28 = r35 + r28;
    r8 = fmaf(r9, r22, r8);
    r8 = fmaf(r11, r28, r8);
    r28 = copysign(1.0, r8);
    r28 = fmaf(r36, r28, r8);
    r36 = r28 * r28;
    r8 = 1.0 / r36;
    r35 = r6 * r6;
    r35 = fmaf(r8, r35, r8 * r25);
    r25 = fmaf(r5, r35, r26);
    r39 = 1.0 / r28;
    r40 = r25 * r39;
    r2 = fmaf(r29, r40, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r0 * r25;
    r1 = r1 * r39;
    r3 = fmaf(r7, r1, r3);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r41 = fmaf(r3, r3, r2 * r2);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r41);
  if (global_thread_idx < problem_size) {
    r41 = r12 * r13;
    r41 = r41 * r18;
    r30 = r30 + r41;
    r24 = fmaf(r10, r24, r11 * r30);
    r13 = r13 * r19;
    r32 = r32 + r13;
    r30 = r14 * r14;
    r42 = r4 * r27;
    r43 = r30 + r42;
    r15 = r15 * r15;
    r44 = r4 * r34;
    r45 = r15 + r44;
    r46 = r43 + r45;
    r46 = fmaf(r10, r46, r11 * r32);
    r32 = r46 * r29;
    r47 = r25 * r4;
    r47 = r47 * r8;
    r32 = fmaf(r47, r32, r24 * r1);
    r48 = r17 * r7;
    r49 = r4 * r15;
    r50 = r34 + r49;
    r43 = r43 + r50;
    r43 = fmaf(r11, r43, r10 * r33);
    r48 = r48 * r43;
    r33 = r18 * r7;
    r36 = r28 * r36;
    r28 = 1.0 / r36;
    r33 = r33 * r7;
    r33 = r33 * r46;
    r33 = fmaf(r28, r33, r8 * r48);
    r48 = r18 * r6;
    r48 = r48 * r6;
    r48 = r48 * r46;
    r33 = fmaf(r28, r48, r33);
    r51 = r17 * r6;
    r51 = r51 * r24;
    r33 = fmaf(r8, r51, r33);
    r51 = r5 * r33;
    r51 = r51 * r39;
    r32 = fmaf(r29, r51, r32);
    r51 = r0 * r33;
    r48 = r5 * r7;
    r51 = r51 * r39;
    r43 = fmaf(r43, r1, r48 * r51);
    r51 = r0 * r7;
    r51 = r51 * r8;
    r51 = r51 * r25;
    r51 = r51 * r4;
    r43 = fmaf(r46, r51, r43);
    r24 = r17 * r7;
    r13 = r37 + r13;
    r13 = fmaf(r9, r13, r11 * r16);
    r24 = r24 * r13;
    r16 = r17 * r6;
    r19 = r12 * r19;
    r38 = r38 + r19;
    r15 = r34 + r15;
    r14 = r14 * r14;
    r14 = r14 * r4;
    r15 = r15 + r42;
    r15 = r15 + r14;
    r15 = fmaf(r11, r15, r9 * r38);
    r16 = r16 * r15;
    r16 = fmaf(r8, r16, r8 * r24);
    r24 = r18 * r6;
    r14 = r27 + r14;
    r50 = r50 + r14;
    r50 = fmaf(r9, r50, r11 * r22);
    r24 = r24 * r6;
    r24 = r24 * r50;
    r16 = fmaf(r28, r24, r16);
    r22 = r18 * r7;
    r22 = r22 * r7;
    r22 = r22 * r50;
    r16 = fmaf(r28, r22, r16);
    r22 = r5 * r16;
    r22 = r22 * r39;
    r24 = r50 * r29;
    r24 = fmaf(r47, r24, r29 * r22);
    r24 = fmaf(r15, r1, r24);
    r15 = r0 * r16;
    r15 = r15 * r39;
    r15 = fmaf(r48, r15, r13 * r1);
    r15 = fmaf(r50, r51, r15);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            0 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r32,
                                            r43,
                                            r24,
                                            r15);
    r13 = r0 * r5;
    r13 = r13 * r17;
    r13 = r13 * r6;
    r13 = r13 * r7;
    r13 = r13 * r28;
    r30 = r27 + r30;
    r30 = r30 + r44;
    r30 = r30 + r49;
    r30 = fmaf(r10, r30, r9 * r21);
    r19 = r23 + r19;
    r19 = fmaf(r10, r19, r9 * r31);
    r31 = r19 * r29;
    r31 = fmaf(r47, r31, r30 * r1);
    r23 = r17 * r6;
    r23 = r23 * r30;
    r30 = r18 * r7;
    r30 = r30 * r7;
    r30 = r30 * r19;
    r30 = fmaf(r28, r30, r8 * r23);
    r23 = r18 * r6;
    r23 = r23 * r6;
    r23 = r23 * r19;
    r30 = fmaf(r28, r23, r30);
    r21 = r17 * r7;
    r20 = r41 + r20;
    r14 = r45 + r14;
    r14 = fmaf(r9, r14, r10 * r20);
    r21 = r21 * r14;
    r30 = fmaf(r8, r21, r30);
    r21 = r5 * r30;
    r21 = r21 * r39;
    r31 = fmaf(r29, r21, r31);
    r21 = r0 * r30;
    r21 = r21 * r39;
    r21 = fmaf(r48, r21, r14 * r1);
    r21 = fmaf(r19, r51, r21);
    r14 = r28 * r29;
    r23 = r17 * r14;
    r9 = r5 * r6;
    r23 = fmaf(r9, r23, r1);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            4 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r31,
                                            r21,
                                            r23,
                                            r13);
    r20 = r0 * r17;
    r20 = r20 * r7;
    r20 = r20 * r28;
    r20 = fmaf(r48, r20, r1);
    r1 = r18 * r7;
    r1 = r1 * r7;
    r10 = r18 * r6;
    r10 = r10 * r6;
    r10 = fmaf(r28, r10, r28 * r1);
    r1 = r5 * r10;
    r1 = r1 * r39;
    r1 = fmaf(r29, r1, r29 * r47);
    r47 = r0 * r10;
    r47 = r47 * r39;
    r47 = fmaf(r48, r47, r51);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            8 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r13,
                                            r20,
                                            r1,
                                            r47);
    r51 = r4 * r2;
    r28 = r4 * r3;
    r28 = fmaf(r43, r28, r32 * r51);
    r51 = r4 * r2;
    r45 = r4 * r3;
    r45 = fmaf(r15, r45, r24 * r51);
    r51 = r4 * r3;
    r41 = r4 * r2;
    r41 = fmaf(r31, r41, r21 * r51);
    r51 = r4 * r2;
    r49 = r18 * r3;
    r44 = r48 * r14;
    r49 = fmaf(r44, r49, r23 * r51);
    write_sum_4<float, float>((float*)inout_shared, r28, r45, r41, r49);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r4 * r3;
    r41 = r4 * r2;
    r41 = fmaf(r1, r41, r47 * r49);
    r49 = r4 * r3;
    r45 = r18 * r2;
    r45 = fmaf(r44, r45, r20 * r49);
    write_sum_2<float, float>((float*)inout_shared, r45, r41);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fmaf(r43, r43, r32 * r32);
    r45 = fmaf(r15, r15, r24 * r24);
    r49 = fmaf(r31, r31, r21 * r21);
    r44 = 4.00000000000000000e+00;
    r36 = r36 * r36;
    r36 = 1.0 / r36;
    r36 = r44 * r36;
    r44 = r0 * r7;
    r36 = r36 * r29;
    r36 = r36 * r48;
    r36 = r36 * r9;
    r36 = r36 * r44;
    r9 = fmaf(r23, r23, r36);
    write_sum_4<float, float>((float*)inout_shared, r41, r45, r49, r9);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fmaf(r20, r20, r36);
    r9 = fmaf(r1, r1, r47 * r47);
    write_sum_2<float, float>((float*)inout_shared, r36, r9);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = fmaf(r32, r24, r43 * r15);
    r36 = fmaf(r32, r31, r43 * r21);
    r49 = fmaf(r43, r13, r32 * r23);
    r45 = fmaf(r32, r13, r43 * r20);
    write_sum_4<float, float>((float*)inout_shared, r9, r36, r49, r45);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fmaf(r32, r1, r43 * r47);
    r43 = fmaf(r15, r21, r24 * r31);
    r45 = fmaf(r15, r13, r24 * r23);
    r49 = fmaf(r24, r13, r15 * r20);
    write_sum_4<float, float>((float*)inout_shared, r32, r43, r45, r49);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r15, r47, r24 * r1);
    r24 = fmaf(r21, r47, r31 * r1);
    r49 = fmaf(r21, r13, r31 * r23);
    r31 = fmaf(r31, r13, r21 * r20);
    write_sum_4<float, float>((float*)inout_shared, r15, r49, r31, r24);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fmaf(r23, r13, r20 * r13);
    r23 = fmaf(r47, r13, r23 * r1);
    r13 = fmaf(r1, r13, r20 * r47);
    write_sum_3<float, float>((float*)inout_shared, r24, r23, r13);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r6 * r40;
    r23 = r7 * r40;
    r24 = r35 * r39;
    r24 = r24 * r29;
    r1 = r0 * r7;
    r1 = r1 * r35;
    r1 = r1 * r39;
    write_idx_4<1024, float, float, float4>(
        out_focal_and_extra_jac,
        0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r13,
        r23,
        r24,
        r1);
    r1 = r7 * r4;
    r1 = r1 * r3;
    r24 = r6 * r4;
    r24 = r24 * r2;
    r24 = fmaf(r40, r24, r40 * r1);
    r1 = r35 * r4;
    r1 = r1 * r2;
    r1 = r1 * r39;
    r40 = r0 * r7;
    r40 = r40 * r35;
    r40 = r40 * r4;
    r40 = r40 * r3;
    r40 = fmaf(r39, r40, r29 * r1);
    write_sum_2<float, float>((float*)inout_shared, r24, r40);
  };
  flush_sum_shared<2, float>(out_focal_and_extra_njtr,
                             0 * out_focal_and_extra_njtr_num_alloc,
                             focal_and_extra_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r7 * r7;
    r40 = r40 * r25;
    r40 = r40 * r25;
    r24 = r6 * r6;
    r24 = r24 * r25;
    r24 = r24 * r25;
    r24 = fmaf(r8, r24, r8 * r40);
    r40 = r6 * r29;
    r1 = r0 * r8;
    r39 = r35 * r35;
    r1 = r1 * r39;
    r39 = r7 * r1;
    r39 = fmaf(r44, r39, r1 * r40);
    write_sum_2<float, float>((float*)inout_shared, r24, r39);
  };
  flush_sum_shared<2, float>(out_focal_and_extra_precond_diag,
                             0 * out_focal_and_extra_precond_diag_num_alloc,
                             focal_and_extra_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r0 * r7;
    r39 = r39 * r7;
    r39 = r39 * r35;
    r39 = r39 * r25;
    r24 = r6 * r35;
    r24 = r24 * r25;
    r24 = r24 * r8;
    r24 = fmaf(r29, r24, r8 * r39);
    write_sum_1<float, float>((float*)inout_shared, r24);
  };
  flush_sum_shared<1, float>(out_focal_and_extra_precond_tril,
                             0 * out_focal_and_extra_precond_tril_num_alloc,
                             focal_and_extra_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r4 * r2;
    r39 = r4 * r3;
    write_sum_2<float, float>((float*)inout_shared, r24, r39);
  };
  flush_sum_shared<2, float>(out_principal_point_njtr,
                             0 * out_principal_point_njtr_num_alloc,
                             principal_point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<float, float>((float*)inout_shared, r26, r26);
  };
  flush_sum_shared<2, float>(out_principal_point_precond_diag,
                             0 * out_principal_point_precond_diag_num_alloc,
                             principal_point_indices_loc,
                             (float*)inout_shared);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_point_res_jac_first(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
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
    float* out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    float* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    float* const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    float* const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
    float* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_point_res_jac_first_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      focal_and_extra,
      focal_and_extra_num_alloc,
      focal_and_extra_indices,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
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
      out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc,
      out_principal_point_jac,
      out_principal_point_jac_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar