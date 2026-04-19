#include "kernel_simple_radial_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_principal_point_res_jac_kernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
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
  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55;

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
    r26 = fmaf(r5, r41, r26);
    r37 = 1.0 / r36;
    r42 = r26 * r37;
    r2 = fmaf(r30, r42, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r0 * r26;
    r1 = r1 * r37;
    r3 = fmaf(r7, r1, r3);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r43 = r12 * r13;
    r43 = r43 * r17;
    r31 = r31 + r43;
    r31 = fmaf(r10, r23, r11 * r31);
    r13 = r13 * r14;
    r13 = r13 * r17;
    r33 = r33 + r13;
    r44 = r4 * r27;
    r45 = r24 + r44;
    r15 = r15 * r15;
    r46 = r4 * r35;
    r47 = r15 + r46;
    r48 = r45 + r47;
    r48 = fmaf(r10, r48, r11 * r33);
    r33 = r48 * r30;
    r49 = r26 * r4;
    r49 = r49 * r8;
    r33 = fmaf(r49, r33, r31 * r1);
    r50 = r19 * r7;
    r51 = r4 * r15;
    r52 = r35 + r51;
    r45 = r45 + r52;
    r45 = fmaf(r11, r45, r10 * r34);
    r50 = r50 * r45;
    r53 = r7 * r7;
    r53 = r17 * r53;
    r38 = r36 * r38;
    r36 = 1.0 / r38;
    r53 = r53 * r36;
    r50 = fmaf(r48, r53, r8 * r50);
    r54 = r6 * r6;
    r54 = r17 * r54;
    r54 = r54 * r36;
    r55 = r19 * r6;
    r55 = r55 * r31;
    r50 = fmaf(r8, r55, r50);
    r50 = fmaf(r48, r54, r50);
    r55 = r5 * r50;
    r55 = r55 * r37;
    r33 = fmaf(r30, r55, r33);
    r55 = r0 * r50;
    r31 = r5 * r7;
    r55 = r55 * r37;
    r45 = fmaf(r45, r1, r31 * r55);
    r55 = r0 * r7;
    r55 = r55 * r48;
    r45 = fmaf(r49, r55, r45);
    r55 = r19 * r7;
    r13 = r39 + r13;
    r13 = fmaf(r9, r13, r11 * r18);
    r55 = r55 * r13;
    r39 = r19 * r6;
    r14 = r12 * r14;
    r14 = r14 * r17;
    r40 = r40 + r14;
    r15 = r35 + r15;
    r35 = r4 * r24;
    r15 = r15 + r44;
    r15 = r15 + r35;
    r15 = fmaf(r11, r15, r9 * r40);
    r39 = r39 * r15;
    r39 = fmaf(r8, r39, r8 * r55);
    r35 = r27 + r35;
    r52 = r52 + r35;
    r52 = fmaf(r9, r52, r11 * r21);
    r39 = fmaf(r52, r54, r39);
    r39 = fmaf(r52, r53, r39);
    r11 = r5 * r39;
    r11 = r11 * r37;
    r55 = r52 * r30;
    r55 = fmaf(r49, r55, r30 * r11);
    r55 = fmaf(r15, r1, r55);
    r15 = r0 * r39;
    r15 = r15 * r37;
    r15 = fmaf(r31, r15, r13 * r1);
    r13 = r0 * r7;
    r13 = r13 * r52;
    r15 = fmaf(r49, r13, r15);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            0 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r33,
                                            r45,
                                            r55,
                                            r15);
    r13 = r0 * r5;
    r13 = r13 * r19;
    r13 = r13 * r6;
    r13 = r13 * r7;
    r13 = r13 * r36;
    r24 = r27 + r24;
    r24 = r24 + r46;
    r24 = r24 + r51;
    r24 = fmaf(r10, r24, r9 * r20);
    r14 = r22 + r14;
    r14 = fmaf(r10, r14, r9 * r32);
    r22 = r14 * r30;
    r22 = fmaf(r49, r22, r24 * r1);
    r51 = r19 * r6;
    r51 = r51 * r24;
    r51 = fmaf(r14, r53, r8 * r51);
    r24 = r19 * r7;
    r43 = r16 + r43;
    r35 = r47 + r35;
    r35 = fmaf(r9, r35, r10 * r43);
    r24 = r24 * r35;
    r51 = fmaf(r8, r24, r51);
    r51 = fmaf(r14, r54, r51);
    r24 = r5 * r51;
    r24 = r24 * r37;
    r22 = fmaf(r30, r24, r22);
    r24 = r0 * r51;
    r24 = r24 * r37;
    r24 = fmaf(r31, r24, r35 * r1);
    r35 = r0 * r7;
    r35 = r35 * r14;
    r24 = fmaf(r49, r35, r24);
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
    r43 = r53 + r54;
    r10 = r5 * r43;
    r10 = r10 * r37;
    r10 = fmaf(r30, r10, r30 * r49);
    r47 = r0 * r7;
    r16 = r0 * r43;
    r16 = r16 * r37;
    r16 = fmaf(r31, r16, r49 * r47);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            8 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r13,
                                            r9,
                                            r10,
                                            r16);
    r47 = r4 * r2;
    r46 = r4 * r3;
    r46 = fmaf(r45, r46, r33 * r47);
    r47 = r4 * r2;
    r27 = r4 * r3;
    r27 = fmaf(r15, r27, r55 * r47);
    r47 = r4 * r3;
    r11 = r4 * r2;
    r11 = fmaf(r22, r11, r24 * r47);
    r47 = r4 * r2;
    r36 = r17 * r36;
    r17 = r3 * r36;
    r40 = r30 * r31;
    r17 = fmaf(r40, r17, r35 * r47);
    write_sum_4<float, float>((float*)inout_shared, r46, r27, r11, r17);
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
    r11 = fmaf(r45, r45, r33 * r33);
    r27 = fmaf(r15, r15, r55 * r55);
    r17 = fmaf(r22, r22, r24 * r24);
    r46 = r0 * r5;
    r47 = 4.00000000000000000e+00;
    r38 = r38 * r38;
    r38 = 1.0 / r38;
    r46 = r46 * r6;
    r46 = r46 * r7;
    r46 = r46 * r47;
    r46 = r46 * r38;
    r46 = r46 * r40;
    r40 = fmaf(r35, r35, r46);
    write_sum_4<float, float>((float*)inout_shared, r11, r27, r17, r40);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fmaf(r9, r9, r46);
    r40 = fmaf(r10, r10, r16 * r16);
    write_sum_2<float, float>((float*)inout_shared, r46, r40);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fmaf(r33, r55, r45 * r15);
    r46 = fmaf(r33, r22, r45 * r24);
    r17 = fmaf(r45, r13, r33 * r35);
    r27 = fmaf(r33, r13, r45 * r9);
    write_sum_4<float, float>((float*)inout_shared, r40, r46, r17, r27);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fmaf(r33, r10, r45 * r16);
    r45 = fmaf(r15, r24, r55 * r22);
    r27 = fmaf(r15, r13, r55 * r35);
    r17 = fmaf(r55, r13, r15 * r9);
    write_sum_4<float, float>((float*)inout_shared, r33, r45, r27, r17);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r15, r16, r55 * r10);
    r55 = fmaf(r24, r16, r22 * r10);
    r17 = fmaf(r24, r13, r22 * r35);
    r22 = fmaf(r22, r13, r24 * r9);
    write_sum_4<float, float>((float*)inout_shared, r15, r17, r22, r55);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fmaf(r35, r13, r9 * r13);
    r35 = fmaf(r16, r13, r35 * r10);
    r13 = fmaf(r10, r13, r9 * r16);
    write_sum_3<float, float>((float*)inout_shared, r55, r35, r13);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r6 * r42;
    r35 = r7 * r42;
    r55 = r41 * r37;
    r55 = r55 * r30;
    r10 = r0 * r7;
    r10 = r10 * r41;
    r10 = r10 * r37;
    write_idx_4<1024, float, float, float4>(
        out_focal_and_extra_jac,
        0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r13,
        r35,
        r55,
        r10);
    r10 = r7 * r4;
    r10 = r10 * r3;
    r55 = r6 * r4;
    r55 = r55 * r2;
    r55 = fmaf(r42, r55, r42 * r10);
    r10 = r41 * r4;
    r10 = r10 * r2;
    r10 = r10 * r37;
    r42 = r0 * r7;
    r42 = r42 * r41;
    r42 = r42 * r4;
    r42 = r42 * r3;
    r42 = fmaf(r37, r42, r30 * r10);
    write_sum_2<float, float>((float*)inout_shared, r55, r42);
  };
  flush_sum_shared<2, float>(out_focal_and_extra_njtr,
                             0 * out_focal_and_extra_njtr_num_alloc,
                             focal_and_extra_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r7 * r7;
    r42 = r42 * r26;
    r42 = r42 * r26;
    r55 = r6 * r6;
    r55 = r55 * r26;
    r55 = r55 * r26;
    r55 = fmaf(r8, r55, r8 * r42);
    r42 = r6 * r30;
    r10 = r0 * r8;
    r35 = r41 * r41;
    r10 = r10 * r35;
    r35 = r7 * r10;
    r13 = r0 * r7;
    r35 = fmaf(r13, r35, r10 * r42);
    write_sum_2<float, float>((float*)inout_shared, r55, r35);
  };
  flush_sum_shared<2, float>(out_focal_and_extra_precond_diag,
                             0 * out_focal_and_extra_precond_diag_num_alloc,
                             focal_and_extra_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r0 * r7;
    r35 = r35 * r7;
    r35 = r35 * r41;
    r35 = r35 * r26;
    r55 = r6 * r41;
    r55 = r55 * r26;
    r55 = r55 * r8;
    r55 = fmaf(r30, r55, r8 * r35);
    write_sum_1<float, float>((float*)inout_shared, r55);
  };
  flush_sum_shared<1, float>(out_focal_and_extra_precond_tril,
                             0 * out_focal_and_extra_precond_tril_num_alloc,
                             focal_and_extra_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = r21 * r30;
    r35 = r19 * r29;
    r35 = r35 * r6;
    r35 = fmaf(r8, r35, r21 * r54);
    r26 = r19 * r18;
    r26 = r26 * r7;
    r35 = fmaf(r8, r26, r35);
    r35 = fmaf(r21, r53, r35);
    r26 = r5 * r35;
    r26 = r26 * r37;
    r26 = fmaf(r30, r26, r49 * r55);
    r26 = fmaf(r29, r1, r26);
    r55 = r0 * r7;
    r55 = r55 * r21;
    r42 = r0 * r35;
    r42 = r42 * r37;
    r42 = fmaf(r31, r42, r49 * r55);
    r42 = fmaf(r18, r1, r42);
    r55 = r32 * r30;
    r55 = fmaf(r20, r1, r49 * r55);
    r16 = r19 * r20;
    r16 = r16 * r6;
    r16 = fmaf(r32, r54, r8 * r16);
    r9 = r19 * r25;
    r9 = r9 * r7;
    r16 = fmaf(r8, r9, r16);
    r16 = fmaf(r32, r53, r16);
    r9 = r5 * r16;
    r9 = r9 * r37;
    r55 = fmaf(r30, r9, r55);
    r9 = r0 * r16;
    r9 = r9 * r37;
    r22 = r0 * r7;
    r22 = r22 * r32;
    r22 = fmaf(r49, r22, r31 * r9);
    r22 = fmaf(r25, r1, r22);
    write_idx_4<1024, float, float, float4>(out_point_jac,
                                            0 * out_point_jac_num_alloc,
                                            global_thread_idx,
                                            r26,
                                            r42,
                                            r55,
                                            r22);
    r49 = r28 * r49;
    r9 = r19 * r23;
    r9 = r9 * r6;
    r54 = fmaf(r28, r54, r8 * r9);
    r9 = r19 * r34;
    r9 = r9 * r7;
    r54 = fmaf(r8, r9, r54);
    r54 = fmaf(r28, r53, r54);
    r53 = r5 * r54;
    r53 = r53 * r37;
    r53 = fmaf(r30, r53, r30 * r49);
    r53 = fmaf(r23, r1, r53);
    r28 = r0 * r54;
    r28 = r28 * r37;
    r28 = fmaf(r31, r28, r49 * r13);
    r28 = fmaf(r34, r1, r28);
    write_idx_2<1024, float, float, float2>(out_point_jac,
                                            4 * out_point_jac_num_alloc,
                                            global_thread_idx,
                                            r53,
                                            r28);
    r1 = r4 * r3;
    r13 = r4 * r2;
    r13 = fmaf(r26, r13, r42 * r1);
    r1 = r4 * r3;
    r49 = r4 * r2;
    r49 = fmaf(r55, r49, r22 * r1);
    r1 = r4 * r3;
    r31 = r4 * r2;
    r31 = fmaf(r53, r31, r28 * r1);
    write_sum_3<float, float>((float*)inout_shared, r13, r49, r31);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r26, r26, r42 * r42);
    r49 = fmaf(r22, r22, r55 * r55);
    r13 = fmaf(r53, r53, r28 * r28);
    write_sum_3<float, float>((float*)inout_shared, r31, r49, r13);
  };
  flush_sum_shared<3, float>(out_point_precond_diag,
                             0 * out_point_precond_diag_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fmaf(r42, r22, r26 * r55);
    r42 = fmaf(r42, r28, r26 * r53);
    r28 = fmaf(r22, r28, r55 * r53);
    write_sum_3<float, float>((float*)inout_shared, r13, r42, r28);
  };
  flush_sum_shared<3, float>(out_point_precond_tril,
                             0 * out_point_precond_tril_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_principal_point_res_jac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
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
  simple_radial_fixed_principal_point_res_jac_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      focal_and_extra,
      focal_and_extra_num_alloc,
      focal_and_extra_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      principal_point,
      principal_point_num_alloc,
      out_res,
      out_res_num_alloc,
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