#include "kernel_simple_radial_fixed_calib_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_calib_res_jac_first_kernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* calib,
        unsigned int calib_num_alloc,
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
      r46, r47, r48, r49, r50, r51, r52, r53;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        calib, 0 * calib_num_alloc, global_thread_idx, r0, r1, r2, r3);
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r6, r1);
  };
  load_shared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>(
        (float*)inout_shared, pose_indices_loc[threadIdx.x].target, r1, r7, r8);
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
    r1 = fmaf(r10, r20, r1);
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
    r1 = fmaf(r11, r23, r1);
    r1 = fmaf(r9, r29, r1);
    r30 = r0 * r1;
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
    r41 = r1 * r1;
    r41 = fmaf(r8, r41, r8 * r37);
    r41 = fmaf(r3, r41, r26);
    r26 = 1.0 / r36;
    r37 = r41 * r26;
    r4 = fmaf(r30, r37, r4);
    r5 = fmaf(r5, r6, r2);
    r2 = r0 * r41;
    r2 = r2 * r26;
    r5 = fmaf(r7, r2, r5);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r37 = fmaf(r5, r5, r4 * r4);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r37);
  if (global_thread_idx < problem_size) {
    r37 = r19 * r7;
    r15 = r15 * r15;
    r42 = r6 * r15;
    r43 = r24 + r42;
    r44 = r6 * r27;
    r45 = r35 + r44;
    r46 = r43 + r45;
    r46 = fmaf(r11, r46, r10 * r34);
    r37 = r37 * r46;
    r47 = r13 * r14;
    r47 = r47 * r17;
    r33 = r33 + r47;
    r44 = r24 + r44;
    r48 = r6 * r35;
    r49 = r15 + r48;
    r44 = r44 + r49;
    r44 = fmaf(r10, r44, r11 * r33);
    r33 = r7 * r7;
    r33 = r17 * r33;
    r38 = r36 * r38;
    r36 = 1.0 / r38;
    r33 = r33 * r36;
    r37 = fmaf(r44, r33, r8 * r37);
    r50 = r1 * r1;
    r50 = r17 * r50;
    r50 = r50 * r36;
    r51 = r19 * r1;
    r13 = r12 * r13;
    r13 = r13 * r17;
    r31 = r31 + r13;
    r31 = fmaf(r10, r23, r11 * r31);
    r51 = r51 * r31;
    r37 = fmaf(r8, r51, r37);
    r37 = fmaf(r44, r50, r37);
    r51 = r3 * r37;
    r51 = r51 * r26;
    r31 = fmaf(r31, r2, r30 * r51);
    r51 = r44 * r30;
    r41 = r41 * r6;
    r41 = r41 * r8;
    r31 = fmaf(r41, r51, r31);
    r51 = r0 * r37;
    r52 = r3 * r7;
    r51 = r51 * r26;
    r53 = r0 * r7;
    r53 = r53 * r44;
    r53 = fmaf(r41, r53, r52 * r51);
    r53 = fmaf(r46, r2, r53);
    r46 = r19 * r7;
    r47 = r39 + r47;
    r47 = fmaf(r9, r47, r11 * r18);
    r46 = r46 * r47;
    r39 = r19 * r1;
    r14 = r12 * r14;
    r14 = r14 * r17;
    r40 = r40 + r14;
    r24 = r6 * r24;
    r15 = r15 + r24;
    r15 = r15 + r45;
    r15 = fmaf(r11, r15, r9 * r40);
    r39 = r39 * r15;
    r39 = fmaf(r8, r39, r8 * r46);
    r42 = r35 + r42;
    r24 = r27 + r24;
    r42 = r42 + r24;
    r42 = fmaf(r9, r42, r11 * r21);
    r39 = fmaf(r42, r50, r39);
    r39 = fmaf(r42, r33, r39);
    r11 = r3 * r39;
    r11 = r11 * r26;
    r35 = r42 * r30;
    r35 = fmaf(r41, r35, r30 * r11);
    r35 = fmaf(r15, r2, r35);
    r15 = r0 * r39;
    r15 = r15 * r26;
    r11 = r0 * r7;
    r11 = r11 * r42;
    r11 = fmaf(r41, r11, r52 * r15);
    r11 = fmaf(r47, r2, r11);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            0 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r31,
                                            r53,
                                            r35,
                                            r11);
    r47 = r0 * r3;
    r47 = r47 * r19;
    r47 = r47 * r1;
    r47 = r47 * r7;
    r47 = r47 * r36;
    r15 = r19 * r1;
    r48 = r27 + r48;
    r48 = r48 + r43;
    r48 = fmaf(r10, r48, r9 * r20);
    r15 = r15 * r48;
    r14 = r22 + r14;
    r14 = fmaf(r10, r14, r9 * r32);
    r15 = fmaf(r14, r33, r8 * r15);
    r22 = r19 * r7;
    r13 = r16 + r13;
    r24 = r49 + r24;
    r24 = fmaf(r9, r24, r10 * r13);
    r22 = r22 * r24;
    r15 = fmaf(r8, r22, r15);
    r15 = fmaf(r14, r50, r15);
    r22 = r3 * r15;
    r22 = r22 * r26;
    r48 = fmaf(r48, r2, r30 * r22);
    r14 = r14 * r41;
    r48 = fmaf(r30, r14, r48);
    r22 = r0 * r15;
    r22 = r22 * r26;
    r22 = fmaf(r52, r22, r24 * r2);
    r24 = r0 * r7;
    r22 = fmaf(r14, r24, r22);
    r24 = r3 * r19;
    r24 = r24 * r1;
    r24 = r24 * r36;
    r24 = fmaf(r30, r24, r2);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            4 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r48,
                                            r22,
                                            r24,
                                            r47);
    r14 = r0 * r19;
    r14 = r14 * r7;
    r14 = r14 * r36;
    r14 = fmaf(r52, r14, r2);
    r9 = r33 + r50;
    r13 = r3 * r9;
    r13 = r13 * r26;
    r13 = fmaf(r30, r41, r30 * r13);
    r10 = r0 * r7;
    r49 = r0 * r9;
    r49 = r49 * r26;
    r49 = fmaf(r52, r49, r41 * r10);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            8 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r47,
                                            r14,
                                            r13,
                                            r49);
    r10 = r6 * r4;
    r16 = r6 * r5;
    r16 = fmaf(r53, r16, r31 * r10);
    r10 = r6 * r4;
    r43 = r6 * r5;
    r43 = fmaf(r11, r43, r35 * r10);
    r10 = r6 * r4;
    r27 = r6 * r5;
    r27 = fmaf(r22, r27, r48 * r10);
    r10 = r6 * r4;
    r36 = r17 * r36;
    r17 = r5 * r36;
    r46 = r30 * r52;
    r17 = fmaf(r46, r17, r24 * r10);
    write_sum_4<float, float>((float*)inout_shared, r16, r43, r27, r17);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r6 * r5;
    r27 = r6 * r4;
    r27 = fmaf(r13, r27, r49 * r17);
    r17 = r6 * r5;
    r43 = r4 * r36;
    r43 = fmaf(r46, r43, r14 * r17);
    write_sum_2<float, float>((float*)inout_shared, r43, r27);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fmaf(r53, r53, r31 * r31);
    r43 = fmaf(r11, r11, r35 * r35);
    r17 = fmaf(r22, r22, r48 * r48);
    r16 = r0 * r3;
    r10 = 4.00000000000000000e+00;
    r38 = r38 * r38;
    r38 = 1.0 / r38;
    r16 = r16 * r1;
    r16 = r16 * r7;
    r16 = r16 * r10;
    r16 = r16 * r38;
    r16 = r16 * r46;
    r46 = fmaf(r24, r24, r16);
    write_sum_4<float, float>((float*)inout_shared, r27, r43, r17, r46);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = fmaf(r14, r14, r16);
    r46 = fmaf(r13, r13, r49 * r49);
    write_sum_2<float, float>((float*)inout_shared, r16, r46);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fmaf(r31, r35, r53 * r11);
    r16 = fmaf(r53, r22, r31 * r48);
    r17 = fmaf(r53, r47, r31 * r24);
    r43 = fmaf(r31, r47, r53 * r14);
    write_sum_4<float, float>((float*)inout_shared, r46, r16, r17, r43);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = fmaf(r53, r49, r31 * r13);
    r31 = fmaf(r35, r48, r11 * r22);
    r43 = fmaf(r11, r47, r35 * r24);
    r17 = fmaf(r35, r47, r11 * r14);
    write_sum_4<float, float>((float*)inout_shared, r53, r31, r43, r17);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fmaf(r35, r13, r11 * r49);
    r11 = fmaf(r22, r49, r48 * r13);
    r17 = fmaf(r22, r47, r48 * r24);
    r48 = fmaf(r48, r47, r22 * r14);
    write_sum_4<float, float>((float*)inout_shared, r35, r17, r48, r11);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r14, r47, r24 * r47);
    r24 = fmaf(r49, r47, r24 * r13);
    r47 = fmaf(r13, r47, r14 * r49);
    write_sum_3<float, float>((float*)inout_shared, r11, r24, r47);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = r21 * r30;
    r47 = fmaf(r29, r2, r41 * r47);
    r24 = r19 * r29;
    r24 = r24 * r1;
    r24 = fmaf(r8, r24, r21 * r50);
    r11 = r19 * r18;
    r11 = r11 * r7;
    r24 = fmaf(r8, r11, r24);
    r24 = fmaf(r21, r33, r24);
    r11 = r3 * r24;
    r11 = r11 * r26;
    r47 = fmaf(r30, r11, r47);
    r11 = r0 * r24;
    r11 = r11 * r26;
    r13 = r0 * r7;
    r13 = r13 * r21;
    r13 = fmaf(r41, r13, r52 * r11);
    r13 = fmaf(r18, r2, r13);
    r11 = r32 * r30;
    r11 = fmaf(r20, r2, r41 * r11);
    r49 = r19 * r20;
    r49 = r49 * r1;
    r49 = fmaf(r32, r50, r8 * r49);
    r14 = r19 * r25;
    r14 = r14 * r7;
    r49 = fmaf(r8, r14, r49);
    r49 = fmaf(r32, r33, r49);
    r14 = r3 * r49;
    r14 = r14 * r26;
    r11 = fmaf(r30, r14, r11);
    r14 = r0 * r7;
    r14 = r14 * r32;
    r48 = r0 * r49;
    r48 = r48 * r26;
    r48 = fmaf(r52, r48, r41 * r14);
    r48 = fmaf(r25, r2, r48);
    write_idx_4<1024, float, float, float4>(out_point_jac,
                                            0 * out_point_jac_num_alloc,
                                            global_thread_idx,
                                            r47,
                                            r13,
                                            r11,
                                            r48);
    r14 = r19 * r23;
    r14 = r14 * r1;
    r50 = fmaf(r28, r50, r8 * r14);
    r14 = r19 * r34;
    r14 = r14 * r7;
    r50 = fmaf(r8, r14, r50);
    r50 = fmaf(r28, r33, r50);
    r33 = r3 * r50;
    r33 = r33 * r26;
    r14 = r28 * r30;
    r14 = fmaf(r41, r14, r30 * r33);
    r14 = fmaf(r23, r2, r14);
    r33 = r0 * r7;
    r33 = r33 * r28;
    r2 = fmaf(r34, r2, r41 * r33);
    r33 = r0 * r50;
    r33 = r33 * r26;
    r2 = fmaf(r52, r33, r2);
    write_idx_2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r14, r2);
    r33 = r6 * r4;
    r52 = r6 * r5;
    r52 = fmaf(r13, r52, r47 * r33);
    r33 = r6 * r5;
    r26 = r6 * r4;
    r26 = fmaf(r11, r26, r48 * r33);
    r33 = r6 * r4;
    r41 = r6 * r5;
    r41 = fmaf(r2, r41, r14 * r33);
    write_sum_3<float, float>((float*)inout_shared, r52, r26, r41);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fmaf(r47, r47, r13 * r13);
    r26 = fmaf(r11, r11, r48 * r48);
    r52 = fmaf(r2, r2, r14 * r14);
    write_sum_3<float, float>((float*)inout_shared, r41, r26, r52);
  };
  flush_sum_shared<3, float>(out_point_precond_diag,
                             0 * out_point_precond_diag_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fmaf(r47, r11, r13 * r48);
    r47 = fmaf(r47, r14, r13 * r2);
    r2 = fmaf(r48, r2, r11 * r14);
    write_sum_3<float, float>((float*)inout_shared, r52, r47, r2);
  };
  flush_sum_shared<3, float>(out_point_precond_tril,
                             0 * out_point_precond_tril_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_calib_res_jac_first(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* calib,
    unsigned int calib_num_alloc,
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
  simple_radial_fixed_calib_res_jac_first_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      calib,
      calib_num_alloc,
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