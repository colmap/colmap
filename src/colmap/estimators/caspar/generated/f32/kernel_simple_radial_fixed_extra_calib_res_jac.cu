#include "kernel_simple_radial_fixed_extra_calib_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_extra_calib_res_jac_kernel(
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54;

  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        extra_calib, 0 * extra_calib_num_alloc, global_thread_idx, r0, r1, r2);
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r3, r4);
    r5 = -1.00000000000000000e+00;
    r3 = fmaf(r3, r5, r0);
  };
  load_shared<1, float, float>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>(
        (float*)inout_shared, focal_indices_loc[threadIdx.x].target, r0);
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
    r42 = r13 * r14;
    r42 = r42 * r17;
    r33 = r33 + r42;
    r43 = r5 * r27;
    r44 = r24 + r43;
    r15 = r15 * r15;
    r45 = r5 * r35;
    r46 = r15 + r45;
    r47 = r44 + r46;
    r47 = fmaf(r10, r47, r11 * r33);
    r33 = r47 * r30;
    r48 = r41 * r5;
    r48 = r48 * r8;
    r49 = r19 * r7;
    r50 = r5 * r15;
    r51 = r35 + r50;
    r44 = r44 + r51;
    r44 = fmaf(r11, r44, r10 * r34);
    r49 = r49 * r44;
    r52 = r7 * r7;
    r52 = r17 * r52;
    r38 = r36 * r38;
    r36 = 1.0 / r38;
    r52 = r52 * r36;
    r49 = fmaf(r47, r52, r8 * r49);
    r53 = r6 * r6;
    r53 = r17 * r53;
    r53 = r53 * r36;
    r54 = r19 * r6;
    r13 = r12 * r13;
    r13 = r13 * r17;
    r31 = r31 + r13;
    r31 = fmaf(r10, r23, r11 * r31);
    r54 = r54 * r31;
    r49 = fmaf(r8, r54, r49);
    r49 = fmaf(r47, r53, r49);
    r54 = r2 * r49;
    r54 = r54 * r26;
    r54 = fmaf(r30, r54, r48 * r33);
    r54 = fmaf(r31, r1, r54);
    r31 = r0 * r49;
    r33 = r2 * r7;
    r31 = r31 * r26;
    r44 = fmaf(r44, r1, r33 * r31);
    r31 = r0 * r7;
    r31 = r31 * r47;
    r44 = fmaf(r48, r31, r44);
    r31 = r19 * r7;
    r42 = r39 + r42;
    r42 = fmaf(r9, r42, r11 * r18);
    r31 = r31 * r42;
    r39 = r19 * r6;
    r14 = r12 * r14;
    r14 = r14 * r17;
    r40 = r40 + r14;
    r15 = r35 + r15;
    r35 = r5 * r24;
    r15 = r15 + r43;
    r15 = r15 + r35;
    r15 = fmaf(r11, r15, r9 * r40);
    r39 = r39 * r15;
    r39 = fmaf(r8, r39, r8 * r31);
    r35 = r27 + r35;
    r51 = r51 + r35;
    r51 = fmaf(r9, r51, r11 * r21);
    r39 = fmaf(r51, r53, r39);
    r39 = fmaf(r51, r52, r39);
    r11 = r2 * r39;
    r11 = r11 * r26;
    r15 = fmaf(r15, r1, r30 * r11);
    r11 = r51 * r30;
    r15 = fmaf(r48, r11, r15);
    r11 = r0 * r39;
    r11 = r11 * r26;
    r31 = r0 * r7;
    r31 = r31 * r51;
    r31 = fmaf(r48, r31, r33 * r11);
    r31 = fmaf(r42, r1, r31);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            0 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r54,
                                            r44,
                                            r15,
                                            r31);
    r42 = r0 * r2;
    r42 = r42 * r19;
    r42 = r42 * r6;
    r42 = r42 * r7;
    r42 = r42 * r36;
    r14 = r22 + r14;
    r14 = fmaf(r10, r14, r9 * r32);
    r22 = r14 * r48;
    r11 = r19 * r6;
    r24 = r27 + r24;
    r24 = r24 + r45;
    r24 = r24 + r50;
    r24 = fmaf(r10, r24, r9 * r20);
    r11 = r11 * r24;
    r11 = fmaf(r14, r52, r8 * r11);
    r50 = r19 * r7;
    r13 = r16 + r13;
    r35 = r46 + r35;
    r35 = fmaf(r9, r35, r10 * r13);
    r50 = r50 * r35;
    r11 = fmaf(r8, r50, r11);
    r11 = fmaf(r14, r53, r11);
    r50 = r2 * r11;
    r50 = r50 * r26;
    r50 = fmaf(r30, r50, r30 * r22);
    r50 = fmaf(r24, r1, r50);
    r24 = r0 * r7;
    r14 = r0 * r11;
    r14 = r14 * r26;
    r14 = fmaf(r33, r14, r22 * r24);
    r14 = fmaf(r35, r1, r14);
    r35 = r2 * r19;
    r35 = r35 * r6;
    r35 = r35 * r36;
    r35 = fmaf(r30, r35, r1);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            4 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r50,
                                            r14,
                                            r35,
                                            r42);
    r24 = r0 * r19;
    r24 = r24 * r7;
    r24 = r24 * r36;
    r24 = fmaf(r33, r24, r1);
    r22 = r52 + r53;
    r9 = r2 * r22;
    r9 = r9 * r26;
    r9 = fmaf(r30, r9, r30 * r48);
    r13 = r0 * r22;
    r13 = r13 * r26;
    r10 = r0 * r7;
    r10 = fmaf(r48, r10, r33 * r13);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            8 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r42,
                                            r24,
                                            r9,
                                            r10);
    r13 = r5 * r4;
    r46 = r5 * r3;
    r46 = fmaf(r54, r46, r44 * r13);
    r13 = r5 * r3;
    r16 = r5 * r4;
    r16 = fmaf(r31, r16, r15 * r13);
    r13 = r5 * r3;
    r45 = r5 * r4;
    r45 = fmaf(r14, r45, r50 * r13);
    r13 = r5 * r3;
    r36 = r17 * r36;
    r17 = r4 * r36;
    r27 = r30 * r33;
    r17 = fmaf(r27, r17, r35 * r13);
    write_sum_4<float, float>((float*)inout_shared, r46, r16, r45, r17);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r5 * r4;
    r45 = r5 * r3;
    r45 = fmaf(r9, r45, r10 * r17);
    r17 = r5 * r4;
    r16 = r3 * r36;
    r16 = fmaf(r27, r16, r24 * r17);
    write_sum_2<float, float>((float*)inout_shared, r16, r45);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fmaf(r44, r44, r54 * r54);
    r16 = fmaf(r15, r15, r31 * r31);
    r17 = fmaf(r50, r50, r14 * r14);
    r46 = r0 * r2;
    r13 = 4.00000000000000000e+00;
    r38 = r38 * r38;
    r38 = 1.0 / r38;
    r46 = r46 * r6;
    r46 = r46 * r7;
    r46 = r46 * r13;
    r46 = r46 * r38;
    r46 = r46 * r27;
    r27 = fmaf(r35, r35, r46);
    write_sum_4<float, float>((float*)inout_shared, r45, r16, r17, r27);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fmaf(r24, r24, r46);
    r27 = fmaf(r10, r10, r9 * r9);
    write_sum_2<float, float>((float*)inout_shared, r46, r27);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fmaf(r54, r15, r44 * r31);
    r46 = fmaf(r44, r14, r54 * r50);
    r17 = fmaf(r44, r42, r54 * r35);
    r16 = fmaf(r54, r42, r44 * r24);
    write_sum_4<float, float>((float*)inout_shared, r27, r46, r17, r16);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fmaf(r54, r9, r44 * r10);
    r44 = fmaf(r15, r50, r31 * r14);
    r16 = fmaf(r31, r42, r15 * r35);
    r17 = fmaf(r15, r42, r31 * r24);
    write_sum_4<float, float>((float*)inout_shared, r54, r44, r16, r17);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r31, r10, r15 * r9);
    r15 = fmaf(r14, r10, r50 * r9);
    r17 = fmaf(r14, r42, r50 * r35);
    r50 = fmaf(r50, r42, r14 * r24);
    write_sum_4<float, float>((float*)inout_shared, r31, r17, r50, r15);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r35, r42, r24 * r42);
    r35 = fmaf(r10, r42, r35 * r9);
    r42 = fmaf(r9, r42, r24 * r10);
    write_sum_3<float, float>((float*)inout_shared, r15, r35, r42);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r6 * r37;
    r35 = r7 * r37;
    write_idx_2<1024, float, float, float2>(out_focal_jac,
                                            0 * out_focal_jac_num_alloc,
                                            global_thread_idx,
                                            r42,
                                            r35);
    r35 = r7 * r5;
    r35 = r35 * r4;
    r42 = r6 * r5;
    r42 = r42 * r3;
    r42 = fmaf(r37, r42, r37 * r35);
    write_sum_1<float, float>((float*)inout_shared, r42);
  };
  flush_sum_shared<1, float>(out_focal_njtr,
                             0 * out_focal_njtr_num_alloc,
                             focal_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r7 * r7;
    r42 = r42 * r41;
    r42 = r42 * r41;
    r35 = r6 * r6;
    r35 = r35 * r41;
    r35 = r35 * r41;
    r35 = fmaf(r8, r35, r8 * r42);
    write_sum_1<float, float>((float*)inout_shared, r35);
  };
  flush_sum_shared<1, float>(out_focal_precond_diag,
                             0 * out_focal_precond_diag_num_alloc,
                             focal_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r19 * r29;
    r35 = r35 * r6;
    r35 = fmaf(r8, r35, r21 * r53);
    r42 = r19 * r18;
    r42 = r42 * r7;
    r35 = fmaf(r8, r42, r35);
    r35 = fmaf(r21, r52, r35);
    r42 = r2 * r35;
    r42 = r42 * r26;
    r42 = fmaf(r29, r1, r30 * r42);
    r41 = r21 * r30;
    r42 = fmaf(r48, r41, r42);
    r41 = r0 * r7;
    r41 = r41 * r21;
    r37 = r0 * r35;
    r37 = r37 * r26;
    r37 = fmaf(r33, r37, r48 * r41);
    r37 = fmaf(r18, r1, r37);
    r41 = r19 * r20;
    r41 = r41 * r6;
    r41 = fmaf(r32, r53, r8 * r41);
    r15 = r19 * r25;
    r15 = r15 * r7;
    r41 = fmaf(r8, r15, r41);
    r41 = fmaf(r32, r52, r41);
    r15 = r2 * r41;
    r15 = r15 * r26;
    r15 = fmaf(r20, r1, r30 * r15);
    r9 = r32 * r30;
    r15 = fmaf(r48, r9, r15);
    r9 = r0 * r7;
    r9 = r9 * r32;
    r9 = fmaf(r48, r9, r25 * r1);
    r10 = r0 * r41;
    r10 = r10 * r26;
    r9 = fmaf(r33, r10, r9);
    write_idx_4<1024, float, float, float4>(out_point_jac,
                                            0 * out_point_jac_num_alloc,
                                            global_thread_idx,
                                            r42,
                                            r37,
                                            r15,
                                            r9);
    r10 = r28 * r30;
    r10 = fmaf(r48, r10, r23 * r1);
    r24 = r19 * r23;
    r24 = r24 * r6;
    r53 = fmaf(r28, r53, r8 * r24);
    r24 = r19 * r34;
    r24 = r24 * r7;
    r53 = fmaf(r8, r24, r53);
    r53 = fmaf(r28, r52, r53);
    r52 = r2 * r53;
    r52 = r52 * r26;
    r10 = fmaf(r30, r52, r10);
    r52 = r0 * r7;
    r52 = r52 * r28;
    r1 = fmaf(r34, r1, r48 * r52);
    r52 = r0 * r53;
    r52 = r52 * r26;
    r1 = fmaf(r33, r52, r1);
    write_idx_2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r10, r1);
    r52 = r5 * r3;
    r33 = r5 * r4;
    r33 = fmaf(r37, r33, r42 * r52);
    r52 = r5 * r3;
    r26 = r5 * r4;
    r26 = fmaf(r9, r26, r15 * r52);
    r52 = r5 * r3;
    r48 = r5 * r4;
    r48 = fmaf(r1, r48, r10 * r52);
    write_sum_3<float, float>((float*)inout_shared, r33, r26, r48);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = fmaf(r37, r37, r42 * r42);
    r26 = fmaf(r9, r9, r15 * r15);
    r33 = fmaf(r1, r1, r10 * r10);
    write_sum_3<float, float>((float*)inout_shared, r48, r26, r33);
  };
  flush_sum_shared<3, float>(out_point_precond_diag,
                             0 * out_point_precond_diag_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fmaf(r37, r9, r42 * r15);
    r37 = fmaf(r37, r1, r42 * r10);
    r1 = fmaf(r9, r1, r15 * r10);
    write_sum_3<float, float>((float*)inout_shared, r33, r37, r1);
  };
  flush_sum_shared<3, float>(out_point_precond_tril,
                             0 * out_point_precond_tril_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_extra_calib_res_jac(
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
  simple_radial_fixed_extra_calib_res_jac_kernel<<<n_blocks, 1024>>>(
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