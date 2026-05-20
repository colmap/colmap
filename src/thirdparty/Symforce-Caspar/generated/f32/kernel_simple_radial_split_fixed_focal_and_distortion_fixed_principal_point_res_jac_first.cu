#include "kernel_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointResJacFirstKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc,
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
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    ReadIdx2<1024, float, float, float2>(focal_and_distortion,
                                         0 * focal_and_distortion_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r5);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>(
        (float*)inout_shared, pose_indices_loc[threadIdx.x].target, r6, r7, r8);
  };
  __syncthreads();
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r9,
                       r10,
                       r11);
  };
  __syncthreads();
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
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
    r1 = r0 * r26;
    r1 = r1 * r41;
    r3 = fmaf(r7, r1, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r37 = fmaf(r3, r3, r2 * r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r37);
  if (global_thread_idx < problem_size) {
    r37 = r12 * r13;
    r37 = r37 * r17;
    r31 = r31 + r37;
    r31 = fmaf(r10, r23, r11 * r31);
    r42 = r19 * r7;
    r15 = r15 * r15;
    r43 = r4 * r15;
    r44 = r24 + r43;
    r45 = r4 * r27;
    r46 = r35 + r45;
    r47 = r44 + r46;
    r47 = fmaf(r11, r47, r10 * r34);
    r42 = r42 * r47;
    r13 = r13 * r14;
    r13 = r13 * r17;
    r33 = r33 + r13;
    r45 = r24 + r45;
    r48 = r4 * r35;
    r49 = r15 + r48;
    r45 = r45 + r49;
    r45 = fmaf(r10, r45, r11 * r33);
    r33 = r7 * r7;
    r33 = r17 * r33;
    r38 = r36 * r38;
    r36 = 1.0 / r38;
    r33 = r33 * r36;
    r42 = fmaf(r45, r33, r8 * r42);
    r50 = r6 * r6;
    r50 = r17 * r50;
    r50 = r50 * r36;
    r51 = r19 * r6;
    r51 = r51 * r31;
    r42 = fmaf(r8, r51, r42);
    r42 = fmaf(r45, r50, r42);
    r51 = r5 * r42;
    r51 = r51 * r26;
    r51 = fmaf(r30, r51, r31 * r1);
    r31 = r45 * r30;
    r41 = r4 * r41;
    r41 = r41 * r8;
    r51 = fmaf(r41, r31, r51);
    r31 = r0 * r42;
    r52 = r5 * r7;
    r31 = r31 * r26;
    r31 = fmaf(r52, r31, r47 * r1);
    r47 = r0 * r7;
    r47 = r47 * r45;
    r31 = fmaf(r41, r47, r31);
    r14 = r12 * r14;
    r14 = r14 * r17;
    r40 = r40 + r14;
    r24 = r4 * r24;
    r15 = r15 + r24;
    r15 = r15 + r46;
    r15 = fmaf(r11, r15, r9 * r40);
    r43 = r35 + r43;
    r24 = r27 + r24;
    r43 = r43 + r24;
    r43 = fmaf(r9, r43, r11 * r21);
    r35 = r43 * r41;
    r40 = fmaf(r30, r35, r15 * r1);
    r46 = r19 * r7;
    r13 = r39 + r13;
    r13 = fmaf(r9, r13, r11 * r18);
    r46 = r46 * r13;
    r11 = r19 * r6;
    r11 = r11 * r15;
    r11 = fmaf(r8, r11, r8 * r46);
    r11 = fmaf(r43, r50, r11);
    r11 = fmaf(r43, r33, r11);
    r43 = r5 * r11;
    r43 = r43 * r26;
    r40 = fmaf(r30, r43, r40);
    r43 = r0 * r7;
    r13 = fmaf(r13, r1, r35 * r43);
    r43 = r0 * r11;
    r43 = r43 * r26;
    r13 = fmaf(r52, r43, r13);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r51,
                                          r31,
                                          r40,
                                          r13);
    r43 = r0 * r5;
    r43 = r43 * r19;
    r43 = r43 * r6;
    r43 = r43 * r7;
    r43 = r43 * r36;
    r35 = r19 * r6;
    r48 = r27 + r48;
    r48 = r48 + r44;
    r48 = fmaf(r10, r48, r9 * r20);
    r35 = r35 * r48;
    r14 = r22 + r14;
    r14 = fmaf(r10, r14, r9 * r32);
    r35 = fmaf(r14, r33, r8 * r35);
    r22 = r19 * r7;
    r37 = r16 + r37;
    r24 = r49 + r24;
    r24 = fmaf(r9, r24, r10 * r37);
    r22 = r22 * r24;
    r35 = fmaf(r8, r22, r35);
    r35 = fmaf(r14, r50, r35);
    r22 = r5 * r35;
    r22 = r22 * r26;
    r48 = fmaf(r48, r1, r30 * r22);
    r22 = r14 * r30;
    r48 = fmaf(r41, r22, r48);
    r22 = r0 * r7;
    r22 = r22 * r14;
    r24 = fmaf(r24, r1, r41 * r22);
    r22 = r0 * r35;
    r22 = r22 * r26;
    r24 = fmaf(r52, r22, r24);
    r22 = r5 * r19;
    r22 = r22 * r6;
    r22 = r22 * r36;
    r22 = fmaf(r30, r22, r1);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r48,
                                          r24,
                                          r22,
                                          r43);
    r9 = r0 * r19;
    r9 = r9 * r7;
    r9 = r9 * r36;
    r9 = fmaf(r52, r9, r1);
    r37 = r33 + r50;
    r10 = r5 * r37;
    r10 = r10 * r26;
    r10 = fmaf(r30, r41, r30 * r10);
    r49 = r0 * r37;
    r49 = r49 * r26;
    r16 = r0 * r7;
    r16 = fmaf(r41, r16, r52 * r49);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r43,
                                          r9,
                                          r10,
                                          r16);
    r49 = r4 * r3;
    r44 = r4 * r2;
    r44 = fmaf(r51, r44, r31 * r49);
    r49 = r4 * r2;
    r27 = r4 * r3;
    r27 = fmaf(r13, r27, r40 * r49);
    r49 = r4 * r3;
    r46 = r4 * r2;
    r46 = fmaf(r48, r46, r24 * r49);
    r49 = r4 * r2;
    r36 = r17 * r36;
    r17 = r3 * r36;
    r15 = r30 * r52;
    r17 = fmaf(r15, r17, r22 * r49);
    WriteSum4<float, float>((float*)inout_shared, r44, r27, r46, r17);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r4 * r3;
    r46 = r4 * r2;
    r46 = fmaf(r10, r46, r16 * r17);
    r17 = r4 * r3;
    r27 = r2 * r36;
    r27 = fmaf(r15, r27, r9 * r17);
    WriteSum2<float, float>((float*)inout_shared, r27, r46);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fmaf(r51, r51, r31 * r31);
    r27 = fmaf(r13, r13, r40 * r40);
    r17 = fmaf(r48, r48, r24 * r24);
    r44 = r0 * r5;
    r49 = 4.00000000000000000e+00;
    r38 = r38 * r38;
    r38 = 1.0 / r38;
    r44 = r44 * r6;
    r44 = r44 * r7;
    r44 = r44 * r49;
    r44 = r44 * r38;
    r44 = r44 * r15;
    r15 = fmaf(r22, r22, r44);
    WriteSum4<float, float>((float*)inout_shared, r46, r27, r17, r15);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fmaf(r9, r9, r44);
    r15 = fmaf(r16, r16, r10 * r10);
    WriteSum2<float, float>((float*)inout_shared, r44, r15);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r31, r13, r51 * r40);
    r44 = fmaf(r51, r48, r31 * r24);
    r17 = fmaf(r31, r43, r51 * r22);
    r27 = fmaf(r51, r43, r31 * r9);
    WriteSum4<float, float>((float*)inout_shared, r15, r44, r17, r27);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r31, r16, r51 * r10);
    r51 = fmaf(r13, r24, r40 * r48);
    r27 = fmaf(r13, r43, r40 * r22);
    r17 = fmaf(r40, r43, r13 * r9);
    WriteSum4<float, float>((float*)inout_shared, r31, r51, r27, r17);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fmaf(r40, r10, r13 * r16);
    r13 = fmaf(r48, r10, r24 * r16);
    r17 = fmaf(r24, r43, r48 * r22);
    r48 = fmaf(r48, r43, r24 * r9);
    WriteSum4<float, float>((float*)inout_shared, r40, r17, r48, r13);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fmaf(r9, r43, r22 * r43);
    r22 = fmaf(r16, r43, r22 * r10);
    r43 = fmaf(r10, r43, r9 * r16);
    WriteSum3<float, float>((float*)inout_shared, r13, r22, r43);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r21 * r30;
    r22 = r19 * r29;
    r22 = r22 * r6;
    r22 = fmaf(r8, r22, r21 * r50);
    r13 = r19 * r18;
    r13 = r13 * r7;
    r22 = fmaf(r8, r13, r22);
    r22 = fmaf(r21, r33, r22);
    r13 = r5 * r22;
    r13 = r13 * r26;
    r13 = fmaf(r30, r13, r41 * r43);
    r13 = fmaf(r29, r1, r13);
    r43 = r0 * r22;
    r43 = r43 * r26;
    r10 = r0 * r21;
    r10 = r10 * r7;
    r10 = fmaf(r41, r10, r52 * r43);
    r10 = fmaf(r18, r1, r10);
    r43 = r32 * r30;
    r43 = fmaf(r20, r1, r41 * r43);
    r16 = r19 * r20;
    r16 = r16 * r6;
    r16 = fmaf(r32, r50, r8 * r16);
    r9 = r19 * r25;
    r9 = r9 * r7;
    r16 = fmaf(r8, r9, r16);
    r16 = fmaf(r32, r33, r16);
    r9 = r5 * r16;
    r9 = r9 * r26;
    r43 = fmaf(r30, r9, r43);
    r9 = r0 * r32;
    r9 = r9 * r7;
    r48 = r0 * r16;
    r48 = r48 * r26;
    r48 = fmaf(r52, r48, r41 * r9);
    r48 = fmaf(r25, r1, r48);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r13,
                                          r10,
                                          r43,
                                          r48);
    r9 = r19 * r23;
    r9 = r9 * r6;
    r50 = fmaf(r28, r50, r8 * r9);
    r9 = r19 * r34;
    r9 = r9 * r7;
    r50 = fmaf(r8, r9, r50);
    r50 = fmaf(r28, r33, r50);
    r33 = r5 * r50;
    r33 = r33 * r26;
    r33 = fmaf(r30, r33, r23 * r1);
    r9 = r28 * r30;
    r33 = fmaf(r41, r9, r33);
    r9 = r0 * r28;
    r9 = r9 * r7;
    r9 = fmaf(r41, r9, r34 * r1);
    r1 = r0 * r50;
    r1 = r1 * r26;
    r9 = fmaf(r52, r1, r9);
    WriteIdx2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r33, r9);
    r1 = r4 * r2;
    r52 = r4 * r3;
    r52 = fmaf(r10, r52, r13 * r1);
    r1 = r4 * r2;
    r26 = r4 * r3;
    r26 = fmaf(r48, r26, r43 * r1);
    r1 = r4 * r3;
    r41 = r4 * r2;
    r41 = fmaf(r33, r41, r9 * r1);
    WriteSum3<float, float>((float*)inout_shared, r52, r26, r41);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fmaf(r13, r13, r10 * r10);
    r26 = fmaf(r48, r48, r43 * r43);
    r52 = fmaf(r33, r33, r9 * r9);
    WriteSum3<float, float>((float*)inout_shared, r41, r26, r52);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fmaf(r10, r48, r13 * r43);
    r10 = fmaf(r10, r9, r13 * r33);
    r9 = fmaf(r48, r9, r43 * r33);
    WriteSum3<float, float>((float*)inout_shared, r52, r10, r9);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc,
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
  SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointResJacFirstKernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              point,
              point_num_alloc,
              point_indices,
              pixel,
              pixel_num_alloc,
              focal_and_distortion,
              focal_and_distortion_num_alloc,
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