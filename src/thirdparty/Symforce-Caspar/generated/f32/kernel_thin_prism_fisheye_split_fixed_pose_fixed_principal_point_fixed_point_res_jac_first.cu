#include "kernel_thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointResJacFirstKernel(
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        float* const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        float* const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58;

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
  };
  LoadShared<4, float, float>(focal_and_extra,
                              0 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r0,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  LoadShared<2, float, float>(focal_and_extra,
                              8 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r8,
                       r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r10,
                                         r11,
                                         r12);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r13, r14, r15);
    r16 = 2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r17,
                                         r18,
                                         r19,
                                         r20);
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r21, r22, r23, r24);
    r25 = fmaf(r17, r24, r20 * r21);
    r26 = r19 * r22;
    r25 = fmaf(r4, r26, r25);
    r25 = fmaf(r18, r23, r25);
    r26 = r16 * r25;
    r27 = r17 * r23;
    r27 = fmaf(r4, r27, r20 * r22);
    r27 = fmaf(r18, r24, r27);
    r27 = fmaf(r19, r21, r27);
    r26 = r26 * r27;
    r28 = fmaf(r17, r22, r20 * r23);
    r29 = r18 * r21;
    r28 = fmaf(r4, r29, r28);
    r28 = fmaf(r19, r24, r28);
    r29 = r16 * r28;
    r30 = fmaf(r18, r22, r17 * r21);
    r30 = fmaf(r19, r23, r30);
    r30 = fmaf(r4, r30, r20 * r24);
    r29 = fmaf(r30, r29, r26);
    r29 = fmaf(r13, r29, r11);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r11, r24, r31);
    r32 = r17 * r18;
    r32 = r32 * r16;
    r33 = r19 * r20;
    r33 = fmaf(r16, r33, r32);
    r34 = -2.00000000000000000e+00;
    r35 = r17 * r17;
    r35 = r34 * r35;
    r36 = 1.00000000000000000e+00;
    r37 = r19 * r19;
    r37 = fmaf(r34, r37, r36);
    r38 = r35 + r37;
    r39 = r18 * r19;
    r39 = r39 * r16;
    r40 = r17 * r20;
    r40 = fmaf(r34, r40, r39);
    r41 = r16 * r28;
    r41 = r41 * r27;
    r42 = r34 * r30;
    r43 = fmaf(r25, r42, r41);
    r44 = r28 * r28;
    r44 = r34 * r44;
    r45 = r36 + r44;
    r46 = r25 * r25;
    r46 = r34 * r46;
    r45 = r45 + r46;
    r29 = fmaf(r11, r33, r29);
    r29 = fmaf(r24, r38, r29);
    r29 = fmaf(r31, r40, r29);
    r29 = fmaf(r15, r43, r29);
    r29 = fmaf(r14, r45, r29);
    r45 = r29 * r29;
    r43 = r27 * r27;
    r43 = r34 * r43;
    r40 = r36 + r43;
    r40 = r40 + r44;
    r40 = fmaf(r13, r40, r10);
    r26 = fmaf(r28, r42, r26);
    r10 = r16 * r28;
    r10 = r10 * r25;
    r44 = r16 * r27;
    r44 = fmaf(r30, r44, r10);
    r38 = r17 * r19;
    r38 = r38 * r16;
    r33 = r18 * r20;
    r47 = fmaf(r16, r33, r38);
    r48 = r19 * r20;
    r48 = fmaf(r34, r48, r32);
    r32 = r18 * r18;
    r32 = r32 * r34;
    r37 = r32 + r37;
    r40 = fmaf(r14, r26, r40);
    r40 = fmaf(r15, r44, r40);
    r40 = fmaf(r31, r47, r40);
    r40 = fmaf(r24, r48, r40);
    r40 = fmaf(r11, r37, r40);
    r37 = r40 * r40;
    r48 = 9.99999999999999955e-07;
    r42 = fmaf(r27, r42, r10);
    r42 = fmaf(r13, r42, r12);
    r33 = fmaf(r34, r33, r38);
    r32 = r36 + r32;
    r32 = r32 + r35;
    r35 = r17 * r20;
    r35 = fmaf(r16, r35, r39);
    r39 = r16 * r25;
    r39 = fmaf(r30, r39, r41);
    r43 = r36 + r43;
    r43 = r43 + r46;
    r42 = fmaf(r11, r33, r42);
    r42 = fmaf(r31, r32, r42);
    r42 = fmaf(r24, r35, r42);
    r42 = fmaf(r14, r39, r42);
    r42 = fmaf(r15, r43, r42);
    r43 = copysign(1.0, r42);
    r43 = fmaf(r48, r43, r42);
    r42 = r43 * r43;
    r15 = 1.0 / r42;
    r39 = r29 * r29;
    r39 = fmaf(r15, r39, r15 * r37);
    r39 = sqrtf(r39);
    r37 = atanf(r39);
    r14 = copysign(1.0, r39);
    r14 = fmaf(r48, r14, r39);
    r48 = r14 * r14;
    r39 = 1.0 / r48;
    r39 = r15 * r39;
    r15 = r37 * r39;
    r45 = r45 * r37;
    r45 = r45 * r15;
    r35 = r40 * r37;
    r24 = r40 * r35;
    r24 = r24 * r15;
    r32 = r45 + r24;
  };
  LoadShared<4, float, float>(focal_and_extra,
                              4 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r31,
                       r33,
                       r11,
                       r46);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r36 = 3.00000000000000000e+00;
    r41 = r40 * r36;
    r41 = r41 * r35;
    r41 = fmaf(r15, r41, r45);
    r8 = fmaf(r33, r41, r8 * r32);
    r45 = r29 * r37;
    r30 = r31 * r45;
    r38 = r16 * r35;
    r13 = r39 * r38;
    r8 = fmaf(r13, r30, r8);
    r12 = r32 * r32;
    r10 = r32 * r12;
    r11 = fmaf(r11, r10, r6 * r32);
    r6 = r12 * r12;
    r11 = fmaf(r46, r6, r11);
    r11 = fmaf(r7, r12, r11);
    r7 = r11 * r35;
    r46 = 1.0 / r43;
    r47 = 1.0 / r14;
    r44 = r46 * r47;
    r8 = fmaf(r44, r7, r8);
    r8 = fmaf(r35, r44, r8);
    r2 = fmaf(r0, r8, r2);
    r7 = r29 * r29;
    r7 = r7 * r37;
    r7 = r7 * r36;
    r7 = fmaf(r15, r7, r24);
    r9 = fmaf(r31, r7, r9 * r32);
    r24 = r11 * r44;
    r9 = fmaf(r45, r24, r9);
    r30 = r33 * r45;
    r9 = fmaf(r13, r30, r9);
    r9 = fmaf(r44, r45, r9);
    r1 = fmaf(r5, r9, r1);
    r1 = fmaf(r3, r4, r1);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r1);
    r3 = fmaf(r1, r1, r2 * r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r3);
  if (global_thread_idx < problem_size) {
    r3 = r4 * r9;
    r3 = r3 * r1;
    r30 = r4 * r2;
    r24 = r8 * r30;
    r26 = r0 * r35;
    r49 = r26 * r30;
    r50 = r44 * r49;
    r51 = r4 * r32;
    r52 = r5 * r45;
    r51 = r51 * r1;
    r51 = r51 * r44;
    r51 = fmaf(r52, r51, r32 * r50);
    r53 = r12 * r44;
    r54 = r4 * r1;
    r54 = r54 * r52;
    r54 = fmaf(r53, r54, r53 * r49);
    WriteSum4<float, float>((float*)inout_shared, r24, r3, r51, r54);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           0 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = r5 * r4;
    r54 = r54 * r7;
    r51 = r34 * r2;
    r51 = r51 * r45;
    r51 = r51 * r26;
    r51 = fmaf(r39, r51, r1 * r54);
    r54 = r34 * r1;
    r54 = r54 * r35;
    r54 = r54 * r52;
    r3 = r0 * r41;
    r3 = fmaf(r30, r3, r39 * r54);
    r54 = r4 * r1;
    r54 = r54 * r10;
    r54 = r54 * r44;
    r54 = fmaf(r52, r54, r10 * r50);
    r24 = r4 * r1;
    r24 = r24 * r44;
    r24 = r24 * r52;
    r24 = fmaf(r6, r24, r6 * r50);
    WriteSum4<float, float>((float*)inout_shared, r51, r3, r54, r24);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           4 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r5 * r4;
    r24 = r24 * r32;
    r24 = r24 * r1;
    r54 = r0 * r32;
    r54 = r54 * r30;
    WriteSum2<float, float>((float*)inout_shared, r54, r24);
  };
  FlushSumShared<2, float>(out_focal_and_extra_njtr,
                           8 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r8 * r8;
    r54 = r9 * r9;
    r30 = r29 * r29;
    r3 = r5 * r5;
    r30 = r30 * r37;
    r30 = r30 * r12;
    r30 = r30 * r15;
    r51 = r0 * r26;
    r50 = r40 * r51;
    r49 = r15 * r50;
    r30 = fmaf(r12, r49, r3 * r30);
    r55 = r5 * r52;
    r56 = r29 * r55;
    r57 = r6 * r56;
    r58 = fmaf(r6, r49, r15 * r57);
    WriteSum4<float, float>((float*)inout_shared, r24, r54, r30, r58);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_diag,
                           0 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = r29 * r45;
    r54 = r37 * r37;
    r24 = 4.00000000000000000e+00;
    r42 = r43 * r42;
    r43 = r43 * r42;
    r43 = 1.0 / r43;
    r48 = r14 * r48;
    r14 = r14 * r48;
    r14 = 1.0 / r14;
    r54 = r54 * r24;
    r54 = r54 * r43;
    r54 = r54 * r14;
    r30 = r30 * r50;
    r14 = r7 * r7;
    r3 = fmaf(r14, r3, r54 * r30);
    r14 = r40 * r35;
    r14 = r14 * r56;
    r30 = r0 * r0;
    r43 = r41 * r41;
    r30 = fmaf(r43, r30, r54 * r14);
    r14 = r15 * r56;
    r43 = r10 * r10;
    r14 = fmaf(r43, r49, r43 * r14);
    r54 = r15 * r56;
    r24 = r6 * r6;
    r24 = fmaf(r49, r24, r24 * r54);
    WriteSum4<float, float>((float*)inout_shared, r3, r30, r14, r24);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_diag,
                           4 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r0 * r0;
    r24 = r24 * r12;
    r30 = r5 * r5;
    r30 = r30 * r12;
    WriteSum2<float, float>((float*)inout_shared, r24, r30);
  };
  FlushSumShared<2, float>(out_focal_and_extra_precond_diag,
                           8 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = 0.00000000000000000e+00;
    r24 = r32 * r8;
    r24 = r24 * r44;
    r24 = r24 * r26;
    r3 = r8 * r26;
    r3 = r3 * r53;
    r54 = r16 * r8;
    r54 = r54 * r45;
    r54 = r54 * r26;
    r54 = r54 * r39;
    WriteSum4<float, float>((float*)inout_shared, r30, r24, r3, r54);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           0 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = r0 * r41;
    r54 = r54 * r8;
    r3 = r0 * r32;
    r3 = r3 * r8;
    r24 = r8 * r10;
    r24 = r24 * r44;
    r24 = r24 * r26;
    r8 = r8 * r44;
    r8 = r8 * r26;
    r8 = r8 * r6;
    WriteSum4<float, float>((float*)inout_shared, r54, r24, r8, r3);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           4 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r5 * r7;
    r3 = r3 * r9;
    r8 = r32 * r9;
    r8 = r8 * r44;
    r8 = r8 * r52;
    r24 = r9 * r52;
    r54 = r53 * r24;
    WriteSum4<float, float>((float*)inout_shared, r30, r8, r54, r3);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           8 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r13 * r24;
    r3 = r9 * r10;
    r3 = r3 * r44;
    r3 = r3 * r52;
    r54 = r9 * r44;
    r54 = r54 * r52;
    r54 = r54 * r6;
    WriteSum4<float, float>((float*)inout_shared, r24, r3, r54, r30);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           12 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = r5 * r32;
    r30 = r30 * r9;
    r9 = r10 * r15;
    r9 = fmaf(r10, r49, r56 * r9);
    r54 = r16 * r45;
    r42 = 1.0 / r42;
    r42 = r37 * r42;
    r48 = 1.0 / r48;
    r42 = r42 * r48;
    r54 = r54 * r50;
    r54 = r54 * r42;
    r50 = r32 * r7;
    r50 = r50 * r44;
    r50 = fmaf(r55, r50, r32 * r54);
    r48 = r32 * r41;
    r48 = r48 * r44;
    r3 = r32 * r56;
    r3 = r3 * r38;
    r3 = fmaf(r42, r3, r51 * r48);
    WriteSum4<float, float>((float*)inout_shared, r30, r9, r50, r3);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           16 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r40 * r37;
    r3 = r0 * r0;
    r37 = r37 * r12;
    r37 = r37 * r46;
    r37 = r37 * r47;
    r37 = r37 * r3;
    r3 = r55 * r53;
    r47 = r32 * r6;
    r46 = r15 * r47;
    r46 = fmaf(r47, r49, r56 * r46);
    WriteSum4<float, float>((float*)inout_shared, r58, r46, r37, r3);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           20 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r7 * r55;
    r3 = fmaf(r53, r3, r12 * r54);
    r53 = r12 * r56;
    r53 = r53 * r38;
    r53 = fmaf(r42, r53, r41 * r37);
    WriteSum4<float, float>((float*)inout_shared, r3, r53, r46, r14);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           24 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = r10 * r44;
    r14 = r14 * r51;
    r46 = r10 * r44;
    r46 = r46 * r55;
    r13 = r55 * r13;
    r53 = r16 * r41;
    r53 = r53 * r45;
    r53 = r53 * r39;
    r53 = fmaf(r51, r53, r7 * r13);
    r3 = r7 * r10;
    r3 = r3 * r44;
    r3 = fmaf(r55, r3, r10 * r54);
    WriteSum4<float, float>((float*)inout_shared, r14, r46, r53, r3);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           28 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r16 * r32;
    r3 = r3 * r45;
    r3 = r3 * r39;
    r3 = r3 * r51;
    r39 = r5 * r5;
    r39 = r39 * r32;
    r39 = r39 * r7;
    r53 = r7 * r44;
    r53 = r53 * r6;
    r53 = fmaf(r55, r53, r6 * r54);
    r54 = r41 * r10;
    r54 = r54 * r44;
    r46 = r10 * r56;
    r46 = r46 * r38;
    r46 = fmaf(r42, r46, r51 * r54);
    WriteSum4<float, float>((float*)inout_shared, r53, r3, r39, r46);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           32 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r0 * r0;
    r46 = r46 * r32;
    r46 = r46 * r41;
    r13 = r32 * r13;
    r39 = r41 * r44;
    r39 = r39 * r6;
    r3 = r38 * r42;
    r3 = fmaf(r57, r3, r51 * r39);
    r39 = r15 * r56;
    r43 = r32 * r43;
    r43 = fmaf(r49, r43, r43 * r39);
    WriteSum4<float, float>((float*)inout_shared, r3, r46, r13, r43);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           36 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r44 * r6;
    r43 = r43 * r51;
    r6 = r44 * r6;
    r6 = r6 * r55;
    r51 = r44 * r51;
    r51 = r51 * r47;
    r13 = r44 * r55;
    r13 = r13 * r47;
    WriteSum4<float, float>((float*)inout_shared, r43, r6, r51, r13);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           40 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointResJacFirst(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    float* const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    float* const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointResJacFirstKernel<<<
      n_blocks,
      1024>>>(sensor_from_rig,
              sensor_from_rig_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              focal_and_extra_indices,
              pixel,
              pixel_num_alloc,
              pose,
              pose_num_alloc,
              principal_point,
              principal_point_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_rTr,
              out_focal_and_extra_njtr,
              out_focal_and_extra_njtr_num_alloc,
              out_focal_and_extra_precond_diag,
              out_focal_and_extra_precond_diag_num_alloc,
              out_focal_and_extra_precond_tril,
              out_focal_and_extra_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar