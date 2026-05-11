#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_split_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPrincipalPointResJacFirstKernel(
        float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        float *focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc,
        SharedIndex *focal_and_distortion_indices, float *point,
        unsigned int point_num_alloc, SharedIndex *point_indices, float *pixel,
        unsigned int pixel_num_alloc, float *principal_point,
        unsigned int principal_point_num_alloc, float *out_res,
        unsigned int out_res_num_alloc, float *const out_rTr,
        float *out_pose_jac, unsigned int out_pose_jac_num_alloc,
        float *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
        float *const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float *const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        float *out_focal_and_distortion_jac,
        unsigned int out_focal_and_distortion_jac_num_alloc,
        float *const out_focal_and_distortion_njtr,
        unsigned int out_focal_and_distortion_njtr_num_alloc,
        float *const out_focal_and_distortion_precond_diag,
        unsigned int out_focal_and_distortion_precond_diag_num_alloc,
        float *const out_focal_and_distortion_precond_tril,
        unsigned int out_focal_and_distortion_precond_tril_num_alloc,
        float *out_point_jac, unsigned int out_point_jac_num_alloc,
        float *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
        float *const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float *const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex focal_and_distortion_indices_loc[1024];
  focal_and_distortion_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_distortion_indices[global_thread_idx]
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
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx, r0, r1);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
  };
  LoadShared<2, float, float>(
      focal_and_distortion, 0 * focal_and_distortion_num_alloc,
      focal_and_distortion_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float *)inout_shared,
                       focal_and_distortion_indices_loc[threadIdx.x].target, r0,
                       r5);
  };
  __syncthreads();
  LoadShared<3, float, float>(pose, 4 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r6, r7, r8);
  };
  __syncthreads();
  LoadShared<3, float, float>(point, 0 * point_num_alloc, point_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       point_indices_loc[threadIdx.x].target, r9, r10, r11);
  };
  __syncthreads();
  LoadShared<4, float, float>(pose, 0 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r12, r13, r14,
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
    r1 = r0 * r37;
    r1 = r1 * r26;
    r3 = fmaf(r7, r1, r3);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r2, r3);
    r43 = fmaf(r3, r3, r2 * r2);
  };
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r43);
  if (global_thread_idx < problem_size) {
    r43 = r12 * r13;
    r43 = r43 * r17;
    r31 = r31 + r43;
    r31 = fmaf(r10, r23, r11 * r31);
    r44 = r19 * r7;
    r15 = r15 * r15;
    r45 = r4 * r15;
    r46 = r24 + r45;
    r47 = r4 * r27;
    r48 = r35 + r47;
    r49 = r46 + r48;
    r49 = fmaf(r11, r49, r10 * r34);
    r44 = r44 * r49;
    r13 = r13 * r14;
    r13 = r13 * r17;
    r33 = r33 + r13;
    r47 = r24 + r47;
    r50 = r4 * r35;
    r51 = r15 + r50;
    r47 = r47 + r51;
    r47 = fmaf(r10, r47, r11 * r33);
    r33 = r7 * r7;
    r33 = r17 * r33;
    r38 = r36 * r38;
    r36 = 1.0 / r38;
    r33 = r33 * r36;
    r44 = fmaf(r47, r33, r8 * r44);
    r52 = r6 * r6;
    r52 = r17 * r52;
    r52 = r52 * r36;
    r53 = r19 * r6;
    r53 = r53 * r31;
    r44 = fmaf(r8, r53, r44);
    r44 = fmaf(r47, r52, r44);
    r53 = r5 * r44;
    r53 = r53 * r37;
    r53 = fmaf(r30, r53, r31 * r1);
    r31 = r47 * r30;
    r54 = r4 * r26;
    r54 = r54 * r8;
    r53 = fmaf(r54, r31, r53);
    r31 = r0 * r44;
    r55 = r5 * r7;
    r31 = r31 * r37;
    r31 = fmaf(r55, r31, r49 * r1);
    r49 = r0 * r7;
    r49 = r49 * r47;
    r31 = fmaf(r54, r49, r31);
    r14 = r12 * r14;
    r14 = r14 * r17;
    r40 = r40 + r14;
    r24 = r4 * r24;
    r15 = r15 + r24;
    r15 = r15 + r48;
    r15 = fmaf(r11, r15, r9 * r40);
    r45 = r35 + r45;
    r24 = r27 + r24;
    r45 = r45 + r24;
    r45 = fmaf(r9, r45, r11 * r21);
    r35 = r45 * r30;
    r35 = fmaf(r54, r35, r15 * r1);
    r40 = r19 * r7;
    r13 = r39 + r13;
    r13 = fmaf(r9, r13, r11 * r18);
    r40 = r40 * r13;
    r11 = r19 * r6;
    r11 = r11 * r15;
    r11 = fmaf(r8, r11, r8 * r40);
    r11 = fmaf(r45, r52, r11);
    r11 = fmaf(r45, r33, r11);
    r40 = r5 * r11;
    r40 = r40 * r37;
    r35 = fmaf(r30, r40, r35);
    r40 = r0 * r7;
    r40 = r40 * r45;
    r13 = fmaf(r13, r1, r54 * r40);
    r40 = r0 * r11;
    r40 = r40 * r37;
    r13 = fmaf(r55, r40, r13);
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r53, r31,
        r35, r13);
    r40 = r0 * r5;
    r40 = r40 * r19;
    r40 = r40 * r6;
    r40 = r40 * r7;
    r40 = r40 * r36;
    r15 = r19 * r6;
    r50 = r27 + r50;
    r50 = r50 + r46;
    r50 = fmaf(r10, r50, r9 * r20);
    r15 = r15 * r50;
    r14 = r22 + r14;
    r14 = fmaf(r10, r14, r9 * r32);
    r15 = fmaf(r14, r33, r8 * r15);
    r22 = r19 * r7;
    r43 = r16 + r43;
    r24 = r51 + r24;
    r24 = fmaf(r9, r24, r10 * r43);
    r22 = r22 * r24;
    r15 = fmaf(r8, r22, r15);
    r15 = fmaf(r14, r52, r15);
    r22 = r5 * r15;
    r22 = r22 * r37;
    r50 = fmaf(r50, r1, r30 * r22);
    r22 = r14 * r30;
    r50 = fmaf(r54, r22, r50);
    r22 = r0 * r7;
    r22 = r22 * r14;
    r24 = fmaf(r24, r1, r54 * r22);
    r22 = r0 * r15;
    r22 = r22 * r37;
    r24 = fmaf(r55, r22, r24);
    r22 = r5 * r19;
    r22 = r22 * r6;
    r22 = r22 * r36;
    r22 = fmaf(r30, r22, r1);
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r50, r24,
        r22, r40);
    r9 = r0 * r19;
    r9 = r9 * r7;
    r9 = r9 * r36;
    r9 = fmaf(r55, r9, r1);
    r43 = r33 + r52;
    r10 = r5 * r43;
    r10 = r10 * r37;
    r10 = fmaf(r30, r54, r30 * r10);
    r51 = r0 * r43;
    r51 = r51 * r37;
    r16 = r0 * r7;
    r16 = fmaf(r54, r16, r55 * r51);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx, r40, r9, r10, r16);
    r51 = r4 * r3;
    r46 = r4 * r2;
    r46 = fmaf(r53, r46, r31 * r51);
    r51 = r4 * r2;
    r27 = r4 * r3;
    r27 = fmaf(r13, r27, r35 * r51);
    r51 = r4 * r3;
    r39 = r4 * r2;
    r39 = fmaf(r50, r39, r24 * r51);
    r51 = r4 * r2;
    r36 = r17 * r36;
    r17 = r3 * r36;
    r48 = r30 * r55;
    r17 = fmaf(r48, r17, r22 * r51);
    WriteSum4<float, float>((float *)inout_shared, r46, r27, r39, r17);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r4 * r3;
    r39 = r4 * r2;
    r39 = fmaf(r10, r39, r16 * r17);
    r17 = r4 * r3;
    r27 = r2 * r36;
    r27 = fmaf(r48, r27, r9 * r17);
    WriteSum2<float, float>((float *)inout_shared, r27, r39);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fmaf(r53, r53, r31 * r31);
    r27 = fmaf(r13, r13, r35 * r35);
    r17 = fmaf(r50, r50, r24 * r24);
    r46 = r0 * r5;
    r51 = 4.00000000000000000e+00;
    r38 = r38 * r38;
    r38 = 1.0 / r38;
    r46 = r46 * r6;
    r46 = r46 * r7;
    r46 = r46 * r51;
    r46 = r46 * r38;
    r46 = r46 * r48;
    r48 = fmaf(r22, r22, r46);
    WriteSum4<float, float>((float *)inout_shared, r39, r27, r17, r48);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fmaf(r9, r9, r46);
    r48 = fmaf(r16, r16, r10 * r10);
    WriteSum2<float, float>((float *)inout_shared, r46, r48);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = fmaf(r31, r13, r53 * r35);
    r46 = fmaf(r53, r50, r31 * r24);
    r17 = fmaf(r31, r40, r53 * r22);
    r27 = fmaf(r53, r40, r31 * r9);
    WriteSum4<float, float>((float *)inout_shared, r48, r46, r17, r27);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r31, r16, r53 * r10);
    r53 = fmaf(r13, r24, r35 * r50);
    r27 = fmaf(r13, r40, r35 * r22);
    r17 = fmaf(r35, r40, r13 * r9);
    WriteSum4<float, float>((float *)inout_shared, r31, r53, r27, r17);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fmaf(r35, r10, r13 * r16);
    r13 = fmaf(r50, r10, r24 * r16);
    r17 = fmaf(r24, r40, r50 * r22);
    r50 = fmaf(r50, r40, r24 * r9);
    WriteSum4<float, float>((float *)inout_shared, r35, r17, r50, r13);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fmaf(r9, r40, r22 * r40);
    r22 = fmaf(r16, r40, r22 * r10);
    r40 = fmaf(r10, r40, r9 * r16);
    WriteSum3<float, float>((float *)inout_shared, r13, r22, r40);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r6 * r42;
    r22 = r7 * r42;
    r13 = r41 * r37;
    r13 = r13 * r30;
    r10 = r0 * r7;
    r10 = r10 * r41;
    r10 = r10 * r37;
    WriteIdx4<1024, float, float, float4>(
        out_focal_and_distortion_jac,
        0 * out_focal_and_distortion_jac_num_alloc, global_thread_idx, r40, r22,
        r13, r10);
    r10 = r4 * r7;
    r10 = r10 * r3;
    r13 = r6 * r4;
    r13 = r13 * r2;
    r13 = fmaf(r42, r13, r42 * r10);
    r10 = r0 * r4;
    r10 = r10 * r7;
    r10 = r10 * r41;
    r10 = r10 * r3;
    r42 = r4 * r41;
    r42 = r42 * r2;
    r42 = r42 * r37;
    r42 = fmaf(r30, r42, r37 * r10);
    WriteSum2<float, float>((float *)inout_shared, r13, r42);
  };
  FlushSumShared<2, float>(out_focal_and_distortion_njtr,
                           0 * out_focal_and_distortion_njtr_num_alloc,
                           focal_and_distortion_indices_loc,
                           (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r7 * r7;
    r42 = r42 * r26;
    r42 = r42 * r26;
    r13 = r6 * r6;
    r13 = r13 * r26;
    r13 = r13 * r26;
    r13 = fmaf(r8, r13, r8 * r42);
    r42 = r0 * r8;
    r10 = r41 * r41;
    r42 = r42 * r10;
    r10 = r7 * r42;
    r22 = r0 * r7;
    r40 = r6 * r30;
    r40 = fmaf(r42, r40, r22 * r10);
    WriteSum2<float, float>((float *)inout_shared, r13, r40);
  };
  FlushSumShared<2, float>(out_focal_and_distortion_precond_diag,
                           0 * out_focal_and_distortion_precond_diag_num_alloc,
                           focal_and_distortion_indices_loc,
                           (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r6 * r41;
    r40 = r40 * r26;
    r40 = r40 * r8;
    r13 = r0 * r7;
    r13 = r13 * r7;
    r13 = r13 * r41;
    r13 = r13 * r26;
    r13 = fmaf(r8, r13, r30 * r40);
    WriteSum1<float, float>((float *)inout_shared, r13);
  };
  FlushSumShared<1, float>(out_focal_and_distortion_precond_tril,
                           0 * out_focal_and_distortion_precond_tril_num_alloc,
                           focal_and_distortion_indices_loc,
                           (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r21 * r30;
    r40 = r19 * r29;
    r40 = r40 * r6;
    r40 = fmaf(r8, r40, r21 * r52);
    r26 = r19 * r18;
    r26 = r26 * r7;
    r40 = fmaf(r8, r26, r40);
    r40 = fmaf(r21, r33, r40);
    r26 = r5 * r40;
    r26 = r26 * r37;
    r26 = fmaf(r30, r26, r54 * r13);
    r26 = fmaf(r29, r1, r26);
    r13 = r0 * r40;
    r13 = r13 * r37;
    r10 = r0 * r21;
    r10 = r10 * r7;
    r10 = fmaf(r54, r10, r55 * r13);
    r10 = fmaf(r18, r1, r10);
    r13 = r32 * r54;
    r16 = fmaf(r20, r1, r30 * r13);
    r9 = r19 * r20;
    r9 = r9 * r6;
    r9 = fmaf(r32, r52, r8 * r9);
    r50 = r19 * r25;
    r50 = r50 * r7;
    r9 = fmaf(r8, r50, r9);
    r9 = fmaf(r32, r33, r9);
    r50 = r5 * r9;
    r50 = r50 * r37;
    r16 = fmaf(r30, r50, r16);
    r50 = r0 * r9;
    r50 = r50 * r37;
    r50 = fmaf(r55, r50, r13 * r22);
    r50 = fmaf(r25, r1, r50);
    WriteIdx4<1024, float, float, float4>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r26, r10,
        r16, r50);
    r22 = r19 * r23;
    r22 = r22 * r6;
    r52 = fmaf(r28, r52, r8 * r22);
    r22 = r19 * r34;
    r22 = r22 * r7;
    r52 = fmaf(r8, r22, r52);
    r52 = fmaf(r28, r33, r52);
    r33 = r5 * r52;
    r33 = r33 * r37;
    r33 = fmaf(r30, r33, r23 * r1);
    r22 = r28 * r30;
    r33 = fmaf(r54, r22, r33);
    r22 = r0 * r28;
    r22 = r22 * r7;
    r22 = fmaf(r54, r22, r34 * r1);
    r1 = r0 * r52;
    r1 = r1 * r37;
    r22 = fmaf(r55, r1, r22);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx, r33, r22);
    r1 = r4 * r2;
    r55 = r4 * r3;
    r55 = fmaf(r10, r55, r26 * r1);
    r1 = r4 * r2;
    r37 = r4 * r3;
    r37 = fmaf(r50, r37, r16 * r1);
    r1 = r4 * r3;
    r54 = r4 * r2;
    r54 = fmaf(r33, r54, r22 * r1);
    WriteSum3<float, float>((float *)inout_shared, r55, r37, r54);
  };
  FlushSumShared<3, float>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fmaf(r26, r26, r10 * r10);
    r37 = fmaf(r50, r50, r16 * r16);
    r55 = fmaf(r33, r33, r22 * r22);
    WriteSum3<float, float>((float *)inout_shared, r54, r37, r55);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fmaf(r10, r50, r26 * r16);
    r10 = fmaf(r10, r22, r26 * r33);
    r22 = fmaf(r50, r22, r16 * r33);
    WriteSum3<float, float>((float *)inout_shared, r55, r10, r22);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedPrincipalPointResJacFirst(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *focal_and_distortion, unsigned int focal_and_distortion_num_alloc,
    SharedIndex *focal_and_distortion_indices, float *point,
    unsigned int point_num_alloc, SharedIndex *point_indices, float *pixel,
    unsigned int pixel_num_alloc, float *principal_point,
    unsigned int principal_point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *const out_rTr, float *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float *out_focal_and_distortion_jac,
    unsigned int out_focal_and_distortion_jac_num_alloc,
    float *const out_focal_and_distortion_njtr,
    unsigned int out_focal_and_distortion_njtr_num_alloc,
    float *const out_focal_and_distortion_precond_diag,
    unsigned int out_focal_and_distortion_precond_diag_num_alloc,
    float *const out_focal_and_distortion_precond_tril,
    unsigned int out_focal_and_distortion_precond_tril_num_alloc,
    float *out_point_jac, unsigned int out_point_jac_num_alloc,
    float *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
    float *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedPrincipalPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, focal_and_distortion,
      focal_and_distortion_num_alloc, focal_and_distortion_indices, point,
      point_num_alloc, point_indices, pixel, pixel_num_alloc, principal_point,
      principal_point_num_alloc, out_res, out_res_num_alloc, out_rTr,
      out_pose_jac, out_pose_jac_num_alloc, out_pose_njtr,
      out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, out_focal_and_distortion_jac,
      out_focal_and_distortion_jac_num_alloc, out_focal_and_distortion_njtr,
      out_focal_and_distortion_njtr_num_alloc,
      out_focal_and_distortion_precond_diag,
      out_focal_and_distortion_precond_diag_num_alloc,
      out_focal_and_distortion_precond_tril,
      out_focal_and_distortion_precond_tril_num_alloc, out_point_jac,
      out_point_jac_num_alloc, out_point_njtr, out_point_njtr_num_alloc,
      out_point_precond_diag, out_point_precond_diag_num_alloc,
      out_point_precond_tril, out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar