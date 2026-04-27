#include "kernel_simple_radial_merged_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_merged_res_jac_first_kernel(
        float* pose,
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
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56;
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
  };
  load_shared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>(
        (float*)inout_shared, pose_indices_loc[threadIdx.x].target, r2, r7, r8);
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
    r2 = fmaf(r10, r20, r2);
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
    r2 = fmaf(r11, r23, r2);
    r2 = fmaf(r9, r29, r2);
    r30 = r0 * r2;
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
    r41 = r2 * r2;
    r41 = fmaf(r8, r41, r8 * r37);
    r37 = fmaf(r1, r41, r26);
    r42 = 1.0 / r36;
    r43 = r37 * r42;
    r4 = fmaf(r30, r43, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r0 * r37;
    r3 = r3 * r42;
    r5 = fmaf(r7, r3, r5);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r44 = fmaf(r5, r5, r4 * r4);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r44);
  if (global_thread_idx < problem_size) {
    r44 = r19 * r7;
    r15 = r15 * r15;
    r45 = r6 * r15;
    r46 = r24 + r45;
    r47 = r6 * r27;
    r48 = r35 + r47;
    r49 = r46 + r48;
    r49 = fmaf(r11, r49, r10 * r34);
    r44 = r44 * r49;
    r50 = r13 * r14;
    r50 = r50 * r17;
    r33 = r33 + r50;
    r47 = r24 + r47;
    r51 = r6 * r35;
    r52 = r15 + r51;
    r47 = r47 + r52;
    r47 = fmaf(r10, r47, r11 * r33);
    r33 = r7 * r7;
    r33 = r17 * r33;
    r38 = r36 * r38;
    r36 = 1.0 / r38;
    r33 = r33 * r36;
    r44 = fmaf(r47, r33, r8 * r44);
    r53 = r2 * r2;
    r53 = r17 * r53;
    r53 = r53 * r36;
    r54 = r19 * r2;
    r13 = r12 * r13;
    r13 = r13 * r17;
    r31 = r31 + r13;
    r31 = fmaf(r10, r23, r11 * r31);
    r54 = r54 * r31;
    r44 = fmaf(r8, r54, r44);
    r44 = fmaf(r47, r53, r44);
    r54 = r1 * r44;
    r54 = r54 * r42;
    r55 = r47 * r30;
    r56 = r37 * r6;
    r56 = r56 * r8;
    r55 = fmaf(r56, r55, r30 * r54);
    r55 = fmaf(r31, r3, r55);
    r31 = r0 * r44;
    r54 = r1 * r7;
    r31 = r31 * r42;
    r31 = fmaf(r54, r31, r49 * r3);
    r49 = r0 * r7;
    r49 = r49 * r47;
    r31 = fmaf(r56, r49, r31);
    r49 = r19 * r7;
    r50 = r39 + r50;
    r50 = fmaf(r9, r50, r11 * r18);
    r49 = r49 * r50;
    r39 = r19 * r2;
    r14 = r12 * r14;
    r14 = r14 * r17;
    r40 = r40 + r14;
    r24 = r6 * r24;
    r15 = r15 + r24;
    r15 = r15 + r48;
    r15 = fmaf(r11, r15, r9 * r40);
    r39 = r39 * r15;
    r39 = fmaf(r8, r39, r8 * r49);
    r45 = r35 + r45;
    r24 = r27 + r24;
    r45 = r45 + r24;
    r45 = fmaf(r9, r45, r11 * r21);
    r39 = fmaf(r45, r53, r39);
    r39 = fmaf(r45, r33, r39);
    r11 = r1 * r39;
    r11 = r11 * r42;
    r35 = r45 * r30;
    r35 = fmaf(r56, r35, r30 * r11);
    r35 = fmaf(r15, r3, r35);
    r15 = r0 * r39;
    r15 = r15 * r42;
    r11 = r0 * r7;
    r11 = r11 * r45;
    r11 = fmaf(r56, r11, r54 * r15);
    r11 = fmaf(r50, r3, r11);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            0 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r55,
                                            r31,
                                            r35,
                                            r11);
    r50 = r0 * r1;
    r50 = r50 * r19;
    r50 = r50 * r2;
    r50 = r50 * r7;
    r50 = r50 * r36;
    r51 = r27 + r51;
    r51 = r51 + r46;
    r51 = fmaf(r10, r51, r9 * r20);
    r14 = r22 + r14;
    r14 = fmaf(r10, r14, r9 * r32);
    r22 = r14 * r30;
    r22 = fmaf(r56, r22, r51 * r3);
    r46 = r19 * r2;
    r46 = r46 * r51;
    r46 = fmaf(r14, r33, r8 * r46);
    r51 = r19 * r7;
    r13 = r16 + r13;
    r24 = r52 + r24;
    r24 = fmaf(r9, r24, r10 * r13);
    r51 = r51 * r24;
    r46 = fmaf(r8, r51, r46);
    r46 = fmaf(r14, r53, r46);
    r51 = r1 * r46;
    r51 = r51 * r42;
    r22 = fmaf(r30, r51, r22);
    r51 = r0 * r46;
    r51 = r51 * r42;
    r9 = r0 * r7;
    r9 = r9 * r14;
    r9 = fmaf(r56, r9, r54 * r51);
    r9 = fmaf(r24, r3, r9);
    r24 = r1 * r19;
    r24 = r24 * r2;
    r24 = r24 * r36;
    r24 = fmaf(r30, r24, r3);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            4 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r22,
                                            r9,
                                            r24,
                                            r50);
    r51 = r0 * r19;
    r51 = r51 * r7;
    r51 = r51 * r36;
    r51 = fmaf(r54, r51, r3);
    r13 = r33 + r53;
    r10 = r1 * r13;
    r10 = r10 * r42;
    r10 = fmaf(r30, r10, r30 * r56);
    r52 = r0 * r7;
    r16 = r0 * r13;
    r16 = r16 * r42;
    r16 = fmaf(r54, r16, r56 * r52);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            8 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r50,
                                            r51,
                                            r10,
                                            r16);
    r52 = r6 * r4;
    r27 = r6 * r5;
    r27 = fmaf(r31, r27, r55 * r52);
    r52 = r6 * r5;
    r15 = r6 * r4;
    r15 = fmaf(r35, r15, r11 * r52);
    r52 = r6 * r5;
    r49 = r6 * r4;
    r49 = fmaf(r22, r49, r9 * r52);
    r52 = r6 * r4;
    r36 = r17 * r36;
    r17 = r5 * r36;
    r40 = r30 * r54;
    r17 = fmaf(r40, r17, r24 * r52);
    write_sum_4<float, float>((float*)inout_shared, r27, r15, r49, r17);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r6 * r5;
    r49 = r6 * r4;
    r49 = fmaf(r10, r49, r16 * r17);
    r17 = r6 * r5;
    r15 = r4 * r36;
    r15 = fmaf(r40, r15, r51 * r17);
    write_sum_2<float, float>((float*)inout_shared, r15, r49);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fmaf(r31, r31, r55 * r55);
    r15 = fmaf(r11, r11, r35 * r35);
    r17 = fmaf(r9, r9, r22 * r22);
    r27 = r0 * r1;
    r52 = 4.00000000000000000e+00;
    r38 = r38 * r38;
    r38 = 1.0 / r38;
    r27 = r27 * r2;
    r27 = r27 * r7;
    r27 = r27 * r52;
    r27 = r27 * r38;
    r27 = r27 * r40;
    r40 = fmaf(r24, r24, r27);
    write_sum_4<float, float>((float*)inout_shared, r49, r15, r17, r40);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fmaf(r51, r51, r27);
    r40 = fmaf(r16, r16, r10 * r10);
    write_sum_2<float, float>((float*)inout_shared, r27, r40);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fmaf(r31, r11, r55 * r35);
    r27 = fmaf(r31, r9, r55 * r22);
    r17 = fmaf(r31, r50, r55 * r24);
    r15 = fmaf(r55, r50, r31 * r51);
    write_sum_4<float, float>((float*)inout_shared, r40, r27, r17, r15);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r31, r16, r55 * r10);
    r55 = fmaf(r35, r22, r11 * r9);
    r15 = fmaf(r11, r50, r35 * r24);
    r17 = fmaf(r35, r50, r11 * r51);
    write_sum_4<float, float>((float*)inout_shared, r31, r55, r15, r17);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r11, r16, r35 * r10);
    r35 = fmaf(r22, r10, r9 * r16);
    r17 = fmaf(r9, r50, r22 * r24);
    r22 = fmaf(r22, r50, r9 * r51);
    write_sum_4<float, float>((float*)inout_shared, r11, r17, r22, r35);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fmaf(r24, r50, r51 * r50);
    r24 = fmaf(r16, r50, r24 * r10);
    r50 = fmaf(r10, r50, r51 * r16);
    write_sum_3<float, float>((float*)inout_shared, r35, r24, r50);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r2 * r43;
    r24 = r7 * r43;
    r35 = r41 * r42;
    r35 = r35 * r30;
    r10 = r0 * r7;
    r10 = r10 * r41;
    r10 = r10 * r42;
    write_idx_4<1024, float, float, float4>(out_calib_jac,
                                            0 * out_calib_jac_num_alloc,
                                            global_thread_idx,
                                            r50,
                                            r24,
                                            r35,
                                            r10);
    r16 = r6 * r4;
    r51 = r6 * r5;
    r22 = r2 * r6;
    r22 = r22 * r4;
    r17 = r7 * r6;
    r17 = r17 * r5;
    r17 = fmaf(r43, r17, r43 * r22);
    r22 = r41 * r6;
    r22 = r22 * r4;
    r22 = r22 * r42;
    r43 = r0 * r7;
    r43 = r43 * r41;
    r43 = r43 * r6;
    r43 = r43 * r5;
    r43 = fmaf(r42, r43, r30 * r22);
    write_sum_4<float, float>((float*)inout_shared, r17, r43, r16, r51);
  };
  flush_sum_shared<4, float>(out_calib_njtr,
                             0 * out_calib_njtr_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = r7 * r7;
    r51 = r51 * r37;
    r51 = r51 * r37;
    r16 = r2 * r2;
    r16 = r16 * r37;
    r16 = r16 * r37;
    r16 = fmaf(r8, r16, r8 * r51);
    r51 = r0 * r8;
    r43 = r41 * r41;
    r51 = r51 * r43;
    r43 = r7 * r51;
    r17 = r0 * r7;
    r22 = r2 * r30;
    r22 = fmaf(r51, r22, r17 * r43);
    write_sum_4<float, float>((float*)inout_shared, r16, r22, r26, r26);
  };
  flush_sum_shared<4, float>(out_calib_precond_diag,
                             0 * out_calib_precond_diag_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = r2 * r41;
    r26 = r26 * r37;
    r26 = r26 * r8;
    r22 = r0 * r7;
    r22 = r22 * r7;
    r22 = r22 * r41;
    r22 = r22 * r37;
    r22 = fmaf(r8, r22, r30 * r26);
    write_sum_4<float, float>((float*)inout_shared, r22, r50, r24, r35);
  };
  flush_sum_shared<4, float>(out_calib_precond_tril,
                             0 * out_calib_precond_tril_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = 0.00000000000000000e+00;
    write_sum_2<float, float>((float*)inout_shared, r10, r35);
  };
  flush_sum_shared<2, float>(out_calib_precond_tril,
                             4 * out_calib_precond_tril_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r21 * r30;
    r35 = fmaf(r56, r35, r29 * r3);
    r10 = r19 * r29;
    r10 = r10 * r2;
    r10 = fmaf(r8, r10, r21 * r53);
    r24 = r19 * r18;
    r24 = r24 * r7;
    r10 = fmaf(r8, r24, r10);
    r10 = fmaf(r21, r33, r10);
    r24 = r1 * r10;
    r24 = r24 * r42;
    r35 = fmaf(r30, r24, r35);
    r24 = r0 * r10;
    r24 = r24 * r42;
    r50 = r0 * r7;
    r50 = r50 * r21;
    r50 = fmaf(r56, r50, r54 * r24);
    r50 = fmaf(r18, r3, r50);
    r24 = r32 * r56;
    r22 = fmaf(r20, r3, r30 * r24);
    r26 = r19 * r20;
    r26 = r26 * r2;
    r26 = fmaf(r32, r53, r8 * r26);
    r37 = r19 * r25;
    r37 = r37 * r7;
    r26 = fmaf(r8, r37, r26);
    r26 = fmaf(r32, r33, r26);
    r37 = r1 * r26;
    r37 = r37 * r42;
    r22 = fmaf(r30, r37, r22);
    r37 = r0 * r26;
    r37 = r37 * r42;
    r37 = fmaf(r54, r37, r24 * r17);
    r37 = fmaf(r25, r3, r37);
    write_idx_4<1024, float, float, float4>(out_point_jac,
                                            0 * out_point_jac_num_alloc,
                                            global_thread_idx,
                                            r35,
                                            r50,
                                            r22,
                                            r37);
    r17 = r28 * r30;
    r17 = fmaf(r23, r3, r56 * r17);
    r24 = r19 * r23;
    r24 = r24 * r2;
    r53 = fmaf(r28, r53, r8 * r24);
    r24 = r19 * r34;
    r24 = r24 * r7;
    r53 = fmaf(r8, r24, r53);
    r53 = fmaf(r28, r33, r53);
    r33 = r1 * r53;
    r33 = r33 * r42;
    r17 = fmaf(r30, r33, r17);
    r33 = r0 * r7;
    r33 = r33 * r28;
    r3 = fmaf(r34, r3, r56 * r33);
    r33 = r0 * r53;
    r33 = r33 * r42;
    r3 = fmaf(r54, r33, r3);
    write_idx_2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r17, r3);
    r33 = r6 * r5;
    r54 = r6 * r4;
    r54 = fmaf(r35, r54, r50 * r33);
    r33 = r6 * r4;
    r42 = r6 * r5;
    r42 = fmaf(r37, r42, r22 * r33);
    r33 = r6 * r4;
    r56 = r6 * r5;
    r56 = fmaf(r3, r56, r17 * r33);
    write_sum_3<float, float>((float*)inout_shared, r54, r42, r56);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = fmaf(r35, r35, r50 * r50);
    r42 = fmaf(r22, r22, r37 * r37);
    r54 = fmaf(r3, r3, r17 * r17);
    write_sum_3<float, float>((float*)inout_shared, r56, r42, r54);
  };
  flush_sum_shared<3, float>(out_point_precond_diag,
                             0 * out_point_precond_diag_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fmaf(r50, r37, r35 * r22);
    r50 = fmaf(r50, r3, r35 * r17);
    r3 = fmaf(r37, r3, r22 * r17);
    write_sum_3<float, float>((float*)inout_shared, r54, r50, r3);
  };
  flush_sum_shared<3, float>(out_point_precond_tril,
                             0 * out_point_precond_tril_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void simple_radial_merged_res_jac_first(
    float* pose,
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
  simple_radial_merged_res_jac_first_kernel<<<n_blocks, 1024>>>(
      pose,
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