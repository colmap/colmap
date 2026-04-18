#include "kernel_simple_radial_fixed_focal_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_focal_res_jac_first_kernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* extra_calib,
        unsigned int extra_calib_num_alloc,
        SharedIndex* extra_calib_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
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
        float* out_extra_calib_jac,
        unsigned int out_extra_calib_jac_num_alloc,
        float* const out_extra_calib_njtr,
        unsigned int out_extra_calib_njtr_num_alloc,
        float* const out_extra_calib_precond_diag,
        unsigned int out_extra_calib_precond_diag_num_alloc,
        float* const out_extra_calib_precond_tril,
        unsigned int out_extra_calib_precond_tril_num_alloc,
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
  __shared__ SharedIndex extra_calib_indices_loc[1024];
  extra_calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? extra_calib_indices[global_thread_idx]
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
      r46, r47, r48, r49, r50, r51, r52, r53, r54;
  load_shared<3, float, float>(extra_calib,
                               0 * extra_calib_num_alloc,
                               extra_calib_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         extra_calib_indices_loc[threadIdx.x].target,
                         r0,
                         r1,
                         r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r37 = fmaf(r2, r41, r26);
    r42 = 1.0 / r36;
    r43 = r37 * r42;
    r3 = fmaf(r30, r43, r3);
    r4 = fmaf(r4, r5, r1);
    r1 = r0 * r37;
    r1 = r1 * r42;
    r4 = fmaf(r7, r1, r4);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r3, r4);
    r43 = fmaf(r3, r3, r4 * r4);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r43);
  if (global_thread_idx < problem_size) {
    r43 = r13 * r14;
    r43 = r43 * r17;
    r33 = r33 + r43;
    r44 = r5 * r27;
    r45 = r24 + r44;
    r15 = r15 * r15;
    r46 = r5 * r35;
    r47 = r15 + r46;
    r48 = r45 + r47;
    r48 = fmaf(r10, r48, r11 * r33);
    r33 = r48 * r30;
    r37 = r37 * r5;
    r37 = r37 * r8;
    r49 = r19 * r7;
    r50 = r5 * r15;
    r51 = r35 + r50;
    r45 = r45 + r51;
    r45 = fmaf(r11, r45, r10 * r34);
    r49 = r49 * r45;
    r52 = r7 * r7;
    r52 = r17 * r52;
    r38 = r36 * r38;
    r36 = 1.0 / r38;
    r52 = r52 * r36;
    r49 = fmaf(r48, r52, r8 * r49);
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
    r49 = fmaf(r48, r53, r49);
    r54 = r2 * r49;
    r54 = r54 * r42;
    r54 = fmaf(r30, r54, r37 * r33);
    r54 = fmaf(r31, r1, r54);
    r31 = r0 * r49;
    r33 = r2 * r7;
    r31 = r31 * r42;
    r45 = fmaf(r45, r1, r33 * r31);
    r31 = r0 * r7;
    r31 = r31 * r48;
    r45 = fmaf(r37, r31, r45);
    r31 = r19 * r7;
    r43 = r39 + r43;
    r43 = fmaf(r9, r43, r11 * r18);
    r31 = r31 * r43;
    r39 = r19 * r6;
    r14 = r12 * r14;
    r14 = r14 * r17;
    r40 = r40 + r14;
    r15 = r35 + r15;
    r35 = r5 * r24;
    r15 = r15 + r44;
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
    r11 = r11 * r42;
    r15 = fmaf(r15, r1, r30 * r11);
    r11 = r51 * r30;
    r15 = fmaf(r37, r11, r15);
    r11 = r0 * r39;
    r11 = r11 * r42;
    r31 = r0 * r7;
    r31 = r31 * r51;
    r31 = fmaf(r37, r31, r33 * r11);
    r31 = fmaf(r43, r1, r31);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            0 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r54,
                                            r45,
                                            r15,
                                            r31);
    r43 = r2 * r0;
    r43 = r43 * r19;
    r43 = r43 * r6;
    r43 = r43 * r7;
    r43 = r43 * r36;
    r14 = r22 + r14;
    r14 = fmaf(r10, r14, r9 * r32);
    r22 = r14 * r30;
    r11 = r19 * r6;
    r24 = r27 + r24;
    r24 = r24 + r46;
    r24 = r24 + r50;
    r24 = fmaf(r10, r24, r9 * r20);
    r11 = r11 * r24;
    r11 = fmaf(r14, r52, r8 * r11);
    r50 = r19 * r7;
    r13 = r16 + r13;
    r35 = r47 + r35;
    r35 = fmaf(r9, r35, r10 * r13);
    r50 = r50 * r35;
    r11 = fmaf(r8, r50, r11);
    r11 = fmaf(r14, r53, r11);
    r50 = r2 * r11;
    r50 = r50 * r42;
    r50 = fmaf(r30, r50, r37 * r22);
    r50 = fmaf(r24, r1, r50);
    r24 = r0 * r7;
    r24 = r24 * r14;
    r22 = r0 * r11;
    r22 = r22 * r42;
    r22 = fmaf(r33, r22, r37 * r24);
    r22 = fmaf(r35, r1, r22);
    r35 = r2 * r19;
    r35 = r35 * r6;
    r35 = r35 * r36;
    r35 = fmaf(r30, r35, r1);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            4 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r50,
                                            r22,
                                            r35,
                                            r43);
    r24 = r0 * r19;
    r24 = r24 * r7;
    r24 = r24 * r36;
    r24 = fmaf(r33, r24, r1);
    r9 = r52 + r53;
    r13 = r2 * r9;
    r13 = r13 * r42;
    r13 = fmaf(r30, r13, r30 * r37);
    r10 = r0 * r9;
    r10 = r10 * r42;
    r47 = r0 * r7;
    r47 = fmaf(r37, r47, r33 * r10);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            8 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r43,
                                            r24,
                                            r13,
                                            r47);
    r10 = r5 * r4;
    r16 = r5 * r3;
    r16 = fmaf(r54, r16, r45 * r10);
    r10 = r5 * r3;
    r46 = r5 * r4;
    r46 = fmaf(r31, r46, r15 * r10);
    r10 = r5 * r3;
    r27 = r5 * r4;
    r27 = fmaf(r22, r27, r50 * r10);
    r10 = r5 * r3;
    r36 = r17 * r36;
    r17 = r4 * r36;
    r40 = r30 * r33;
    r17 = fmaf(r40, r17, r35 * r10);
    write_sum_4<float, float>((float*)inout_shared, r16, r46, r27, r17);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r5 * r4;
    r27 = r5 * r3;
    r27 = fmaf(r13, r27, r47 * r17);
    r17 = r5 * r4;
    r46 = r3 * r36;
    r46 = fmaf(r40, r46, r24 * r17);
    write_sum_2<float, float>((float*)inout_shared, r46, r27);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fmaf(r45, r45, r54 * r54);
    r46 = fmaf(r15, r15, r31 * r31);
    r17 = fmaf(r50, r50, r22 * r22);
    r16 = r2 * r0;
    r10 = 4.00000000000000000e+00;
    r38 = r38 * r38;
    r38 = 1.0 / r38;
    r16 = r16 * r6;
    r16 = r16 * r7;
    r16 = r16 * r10;
    r16 = r16 * r38;
    r16 = r16 * r40;
    r40 = fmaf(r35, r35, r16);
    write_sum_4<float, float>((float*)inout_shared, r27, r46, r17, r40);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = fmaf(r24, r24, r16);
    r40 = fmaf(r47, r47, r13 * r13);
    write_sum_2<float, float>((float*)inout_shared, r16, r40);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fmaf(r54, r15, r45 * r31);
    r16 = fmaf(r45, r22, r54 * r50);
    r17 = fmaf(r45, r43, r54 * r35);
    r46 = fmaf(r54, r43, r45 * r24);
    write_sum_4<float, float>((float*)inout_shared, r40, r16, r17, r46);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fmaf(r54, r13, r45 * r47);
    r45 = fmaf(r15, r50, r31 * r22);
    r46 = fmaf(r31, r43, r15 * r35);
    r17 = fmaf(r15, r43, r31 * r24);
    write_sum_4<float, float>((float*)inout_shared, r54, r45, r46, r17);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r31, r47, r15 * r13);
    r15 = fmaf(r22, r47, r50 * r13);
    r17 = fmaf(r22, r43, r50 * r35);
    r50 = fmaf(r50, r43, r22 * r24);
    write_sum_4<float, float>((float*)inout_shared, r31, r17, r50, r15);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r35, r43, r24 * r43);
    r35 = fmaf(r47, r43, r35 * r13);
    r43 = fmaf(r13, r43, r24 * r47);
    write_sum_3<float, float>((float*)inout_shared, r15, r35, r43);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r41 * r42;
    r43 = r43 * r30;
    r35 = r0 * r7;
    r35 = r35 * r41;
    r35 = r35 * r42;
    write_idx_2<1024, float, float, float2>(out_extra_calib_jac,
                                            0 * out_extra_calib_jac_num_alloc,
                                            global_thread_idx,
                                            r43,
                                            r35);
    r15 = r5 * r3;
    r13 = r5 * r4;
    r47 = r0 * r7;
    r47 = r47 * r41;
    r47 = r47 * r5;
    r47 = r47 * r4;
    r24 = r41 * r5;
    r24 = r24 * r3;
    r24 = r24 * r42;
    r24 = fmaf(r30, r24, r42 * r47);
    write_sum_3<float, float>((float*)inout_shared, r15, r13, r24);
  };
  flush_sum_shared<3, float>(out_extra_calib_njtr,
                             0 * out_extra_calib_njtr_num_alloc,
                             extra_calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r0 * r8;
    r13 = r41 * r41;
    r24 = r24 * r13;
    r13 = r7 * r24;
    r15 = r0 * r7;
    r47 = r6 * r30;
    r47 = fmaf(r24, r47, r15 * r13);
    write_sum_3<float, float>((float*)inout_shared, r26, r26, r47);
  };
  flush_sum_shared<3, float>(out_extra_calib_precond_diag,
                             0 * out_extra_calib_precond_diag_num_alloc,
                             extra_calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = 0.00000000000000000e+00;
    write_sum_3<float, float>((float*)inout_shared, r47, r43, r35);
  };
  flush_sum_shared<3, float>(out_extra_calib_precond_tril,
                             0 * out_extra_calib_precond_tril_num_alloc,
                             extra_calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r19 * r29;
    r35 = r35 * r6;
    r35 = fmaf(r8, r35, r21 * r53);
    r43 = r19 * r18;
    r43 = r43 * r7;
    r35 = fmaf(r8, r43, r35);
    r35 = fmaf(r21, r52, r35);
    r43 = r2 * r35;
    r43 = r43 * r42;
    r43 = fmaf(r29, r1, r30 * r43);
    r47 = r21 * r30;
    r43 = fmaf(r37, r47, r43);
    r47 = r0 * r7;
    r47 = r47 * r21;
    r26 = r0 * r35;
    r26 = r26 * r42;
    r26 = fmaf(r33, r26, r37 * r47);
    r26 = fmaf(r18, r1, r26);
    r47 = r19 * r20;
    r47 = r47 * r6;
    r47 = fmaf(r32, r53, r8 * r47);
    r13 = r19 * r25;
    r13 = r13 * r7;
    r47 = fmaf(r8, r13, r47);
    r47 = fmaf(r32, r52, r47);
    r13 = r2 * r47;
    r13 = r13 * r42;
    r13 = fmaf(r20, r1, r30 * r13);
    r50 = r32 * r30;
    r13 = fmaf(r37, r50, r13);
    r50 = r0 * r7;
    r50 = r50 * r32;
    r50 = fmaf(r37, r50, r25 * r1);
    r17 = r0 * r47;
    r17 = r17 * r42;
    r50 = fmaf(r33, r17, r50);
    write_idx_4<1024, float, float, float4>(out_point_jac,
                                            0 * out_point_jac_num_alloc,
                                            global_thread_idx,
                                            r43,
                                            r26,
                                            r13,
                                            r50);
    r37 = r28 * r37;
    r17 = fmaf(r30, r37, r23 * r1);
    r31 = r19 * r23;
    r31 = r31 * r6;
    r53 = fmaf(r28, r53, r8 * r31);
    r31 = r19 * r34;
    r31 = r31 * r7;
    r53 = fmaf(r8, r31, r53);
    r53 = fmaf(r28, r52, r53);
    r52 = r2 * r53;
    r52 = r52 * r42;
    r17 = fmaf(r30, r52, r17);
    r1 = fmaf(r34, r1, r37 * r15);
    r15 = r0 * r53;
    r15 = r15 * r42;
    r1 = fmaf(r33, r15, r1);
    write_idx_2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r17, r1);
    r15 = r5 * r3;
    r33 = r5 * r4;
    r33 = fmaf(r26, r33, r43 * r15);
    r15 = r5 * r3;
    r42 = r5 * r4;
    r42 = fmaf(r50, r42, r13 * r15);
    r15 = r5 * r3;
    r37 = r5 * r4;
    r37 = fmaf(r1, r37, r17 * r15);
    write_sum_3<float, float>((float*)inout_shared, r33, r42, r37);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fmaf(r26, r26, r43 * r43);
    r42 = fmaf(r50, r50, r13 * r13);
    r33 = fmaf(r1, r1, r17 * r17);
    write_sum_3<float, float>((float*)inout_shared, r37, r42, r33);
  };
  flush_sum_shared<3, float>(out_point_precond_diag,
                             0 * out_point_precond_diag_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fmaf(r26, r50, r43 * r13);
    r26 = fmaf(r26, r1, r43 * r17);
    r1 = fmaf(r50, r1, r13 * r17);
    write_sum_3<float, float>((float*)inout_shared, r33, r26, r1);
  };
  flush_sum_shared<3, float>(out_point_precond_tril,
                             0 * out_point_precond_tril_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_focal_res_jac_first(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* extra_calib,
    unsigned int extra_calib_num_alloc,
    SharedIndex* extra_calib_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
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
    float* out_extra_calib_jac,
    unsigned int out_extra_calib_jac_num_alloc,
    float* const out_extra_calib_njtr,
    unsigned int out_extra_calib_njtr_num_alloc,
    float* const out_extra_calib_precond_diag,
    unsigned int out_extra_calib_precond_diag_num_alloc,
    float* const out_extra_calib_precond_tril,
    unsigned int out_extra_calib_precond_tril_num_alloc,
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
  simple_radial_fixed_focal_res_jac_first_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      extra_calib,
      extra_calib_num_alloc,
      extra_calib_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      focal,
      focal_num_alloc,
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
      out_extra_calib_jac,
      out_extra_calib_jac_num_alloc,
      out_extra_calib_njtr,
      out_extra_calib_njtr_num_alloc,
      out_extra_calib_precond_diag,
      out_extra_calib_precond_diag_num_alloc,
      out_extra_calib_precond_tril,
      out_extra_calib_precond_tril_num_alloc,
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