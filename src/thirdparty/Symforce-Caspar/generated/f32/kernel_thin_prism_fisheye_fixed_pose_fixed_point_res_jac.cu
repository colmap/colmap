#include "kernel_thin_prism_fisheye_fixed_pose_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFixedPoseFixedPointResJacKernel(
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        float* const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        float* const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59;
  LoadShared<4, float, float>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r0,
                       r1,
                       r2,
                       r3);
  };
  __syncthreads();
  LoadShared<4, float, float>(
      calib, 4 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r4,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r8,
                                         r9,
                                         r10);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r11, r12, r13);
    r14 = 2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r15,
                                         r16,
                                         r17,
                                         r18);
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r19, r20, r21, r22);
    r23 = fmaf(r15, r22, r18 * r19);
    r24 = r17 * r20;
    r25 = -1.00000000000000000e+00;
    r23 = fmaf(r25, r24, r23);
    r23 = fmaf(r16, r21, r23);
    r24 = r14 * r23;
    r26 = r15 * r21;
    r26 = fmaf(r25, r26, r18 * r20);
    r26 = fmaf(r16, r22, r26);
    r26 = fmaf(r17, r19, r26);
    r24 = r24 * r26;
    r27 = fmaf(r15, r20, r18 * r21);
    r28 = r16 * r19;
    r27 = fmaf(r25, r28, r27);
    r27 = fmaf(r17, r22, r27);
    r28 = r14 * r27;
    r29 = fmaf(r16, r20, r15 * r19);
    r29 = fmaf(r17, r21, r29);
    r29 = fmaf(r25, r29, r18 * r22);
    r28 = fmaf(r29, r28, r24);
    r28 = fmaf(r11, r28, r9);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r9, r22, r30);
    r31 = r15 * r16;
    r31 = r31 * r14;
    r32 = r17 * r18;
    r32 = fmaf(r14, r32, r31);
    r33 = -2.00000000000000000e+00;
    r34 = r17 * r17;
    r34 = r33 * r34;
    r35 = 1.00000000000000000e+00;
    r36 = r15 * r15;
    r36 = fmaf(r33, r36, r35);
    r37 = r34 + r36;
    r38 = r16 * r17;
    r38 = r38 * r14;
    r39 = r15 * r18;
    r39 = fmaf(r33, r39, r38);
    r40 = r14 * r27;
    r40 = r40 * r26;
    r41 = r33 * r29;
    r42 = fmaf(r23, r41, r40);
    r43 = r23 * r23;
    r43 = r33 * r43;
    r44 = r35 + r43;
    r45 = r27 * r27;
    r45 = r33 * r45;
    r44 = r44 + r45;
    r28 = fmaf(r9, r32, r28);
    r28 = fmaf(r22, r37, r28);
    r28 = fmaf(r30, r39, r28);
    r28 = fmaf(r13, r42, r28);
    r28 = fmaf(r12, r44, r28);
    r44 = r26 * r26;
    r44 = r33 * r44;
    r42 = r35 + r44;
    r42 = r42 + r45;
    r42 = fmaf(r11, r42, r8);
    r24 = fmaf(r27, r41, r24);
    r8 = r14 * r27;
    r8 = r8 * r23;
    r45 = r14 * r26;
    r45 = fmaf(r29, r45, r8);
    r39 = r15 * r17;
    r39 = r39 * r14;
    r37 = r16 * r18;
    r32 = fmaf(r14, r37, r39);
    r46 = r17 * r18;
    r46 = fmaf(r33, r46, r31);
    r31 = r16 * r16;
    r31 = r31 * r33;
    r47 = r35 + r31;
    r47 = r47 + r34;
    r42 = fmaf(r12, r24, r42);
    r42 = fmaf(r13, r45, r42);
    r42 = fmaf(r30, r32, r42);
    r42 = fmaf(r22, r46, r42);
    r42 = fmaf(r9, r47, r42);
    r47 = r42 * r42;
    r46 = 9.99999999999999955e-07;
    r41 = fmaf(r26, r41, r8);
    r41 = fmaf(r11, r41, r10);
    r37 = fmaf(r33, r37, r39);
    r36 = r31 + r36;
    r31 = r15 * r18;
    r31 = fmaf(r14, r31, r38);
    r38 = r14 * r23;
    r38 = fmaf(r29, r38, r40);
    r44 = r35 + r44;
    r44 = r44 + r43;
    r41 = fmaf(r9, r37, r41);
    r41 = fmaf(r30, r36, r41);
    r41 = fmaf(r22, r31, r41);
    r41 = fmaf(r12, r38, r41);
    r41 = fmaf(r13, r44, r41);
    r44 = copysign(1.0, r41);
    r44 = fmaf(r46, r44, r41);
    r41 = r44 * r44;
    r13 = 1.0 / r41;
    r38 = r28 * r28;
    r38 = fmaf(r13, r38, r13 * r47);
    r38 = sqrtf(r38);
    r47 = atanf(r38);
    r12 = r28 * r47;
    r31 = r28 * r12;
    r22 = copysign(1.0, r38);
    r22 = fmaf(r46, r22, r38);
    r46 = r22 * r22;
    r38 = 1.0 / r46;
    r38 = r13 * r38;
    r13 = r47 * r38;
    r31 = r31 * r13;
    r36 = r42 * r42;
    r30 = 3.00000000000000000e+00;
    r36 = r36 * r47;
    r36 = r36 * r30;
    r36 = fmaf(r13, r36, r31);
  };
  LoadShared<4, float, float>(
      calib, 8 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r37,
                       r9,
                       r43,
                       r40);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r29 = r42 * r42;
    r29 = r29 * r47;
    r29 = r29 * r13;
    r31 = r31 + r29;
    r43 = fmaf(r43, r31, r7 * r36);
    r39 = r42 * r47;
    r11 = r6 * r39;
    r10 = r14 * r12;
    r8 = r38 * r10;
    r43 = fmaf(r8, r11, r43);
    r32 = r31 * r31;
    r5 = fmaf(r5, r32, r4 * r31);
    r4 = r32 * r32;
    r45 = r31 * r32;
    r5 = fmaf(r9, r4, r5);
    r5 = fmaf(r37, r45, r5);
    r37 = 1.0 / r44;
    r9 = 1.0 / r22;
    r24 = r37 * r9;
    r34 = r5 * r24;
    r43 = fmaf(r39, r34, r43);
    r43 = fmaf(r24, r39, r43);
    r2 = fmaf(r0, r43, r2);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r34, r11);
    r2 = fmaf(r34, r25, r2);
    r34 = r28 * r30;
    r34 = r34 * r12;
    r34 = fmaf(r13, r34, r29);
    r40 = fmaf(r40, r31, r6 * r34);
    r29 = r7 * r39;
    r40 = fmaf(r8, r29, r40);
    r48 = r5 * r12;
    r40 = fmaf(r24, r48, r40);
    r40 = fmaf(r12, r24, r40);
    r3 = fmaf(r1, r40, r3);
    r3 = fmaf(r11, r25, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r11 = r25 * r40;
    r11 = r11 * r3;
    r48 = r25 * r2;
    r29 = r25 * r3;
    r49 = r43 * r48;
    WriteSum4<float, float>((float*)inout_shared, r49, r11, r48, r29);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = r25 * r31;
    r11 = r1 * r12;
    r29 = r29 * r3;
    r29 = r29 * r24;
    r49 = r0 * r39;
    r50 = r49 * r48;
    r51 = r24 * r50;
    r29 = fmaf(r31, r51, r11 * r29);
    r52 = r25 * r3;
    r53 = r1 * r28;
    r53 = r53 * r47;
    r53 = r53 * r32;
    r53 = r53 * r37;
    r53 = r53 * r9;
    r9 = r32 * r24;
    r50 = fmaf(r9, r50, r53 * r52);
    r52 = r1 * r25;
    r52 = r52 * r34;
    r37 = r33 * r2;
    r37 = r37 * r12;
    r37 = r37 * r49;
    r37 = fmaf(r38, r37, r3 * r52);
    r52 = r0 * r36;
    r54 = r33 * r3;
    r54 = r54 * r39;
    r54 = r54 * r11;
    r54 = fmaf(r38, r54, r48 * r52);
    WriteSum4<float, float>((float*)inout_shared, r29, r50, r37, r54);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           4 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = r1 * r25;
    r54 = r54 * r31;
    r54 = r54 * r3;
    r37 = r0 * r31;
    r37 = r37 * r48;
    r48 = r25 * r3;
    r48 = r48 * r45;
    r48 = r48 * r24;
    r48 = fmaf(r45, r51, r11 * r48);
    r50 = r25 * r3;
    r50 = r50 * r24;
    r50 = r50 * r11;
    r51 = fmaf(r4, r51, r4 * r50);
    WriteSum4<float, float>((float*)inout_shared, r48, r51, r37, r54);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           8 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = r43 * r43;
    r37 = r40 * r40;
    WriteSum4<float, float>((float*)inout_shared, r54, r37, r35, r35);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r42 * r42;
    r37 = r0 * r0;
    r35 = r35 * r47;
    r35 = r35 * r32;
    r35 = r35 * r13;
    r54 = r1 * r11;
    r51 = r28 * r54;
    r48 = r13 * r51;
    r35 = fmaf(r32, r48, r37 * r35);
    r50 = r4 * r13;
    r29 = r0 * r49;
    r52 = r42 * r29;
    r50 = fmaf(r4, r48, r52 * r50);
    r55 = r1 * r1;
    r56 = r34 * r34;
    r57 = r28 * r12;
    r58 = r47 * r47;
    r59 = 4.00000000000000000e+00;
    r41 = r44 * r41;
    r44 = r44 * r41;
    r44 = 1.0 / r44;
    r46 = r22 * r46;
    r22 = r22 * r46;
    r22 = 1.0 / r22;
    r58 = r58 * r59;
    r58 = r58 * r44;
    r58 = r58 * r22;
    r57 = r57 * r52;
    r57 = fmaf(r58, r57, r56 * r55);
    r55 = r36 * r36;
    r56 = r42 * r39;
    r56 = r56 * r51;
    r56 = fmaf(r58, r56, r55 * r37);
    WriteSum4<float, float>((float*)inout_shared, r35, r50, r57, r56);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           4 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = r0 * r0;
    r56 = r56 * r32;
    r57 = r1 * r1;
    r57 = r57 * r32;
    r35 = r13 * r52;
    r37 = r45 * r45;
    r35 = fmaf(r37, r48, r37 * r35);
    r55 = r13 * r52;
    r58 = r4 * r4;
    r58 = fmaf(r48, r58, r58 * r55);
    WriteSum4<float, float>((float*)inout_shared, r35, r58, r56, r57);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           8 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = 0.00000000000000000e+00;
    r56 = r31 * r43;
    r56 = r56 * r24;
    r56 = r56 * r49;
    WriteSum4<float, float>((float*)inout_shared, r57, r43, r57, r56);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = r0 * r36;
    r56 = r56 * r43;
    r58 = r43 * r49;
    r58 = r58 * r9;
    r55 = r43 * r49;
    r55 = r55 * r8;
    r22 = r43 * r45;
    r22 = r22 * r24;
    r22 = r22 * r49;
    WriteSum4<float, float>((float*)inout_shared, r58, r55, r56, r22);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = r0 * r31;
    r22 = r22 * r43;
    r43 = r43 * r24;
    r43 = r43 * r49;
    r43 = r43 * r4;
    WriteSum4<float, float>((float*)inout_shared, r43, r22, r57, r57);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           8 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = r1 * r34;
    r22 = r22 * r40;
    r43 = r31 * r40;
    r43 = r43 * r24;
    r43 = r43 * r11;
    r56 = r40 * r53;
    WriteSum4<float, float>((float*)inout_shared, r40, r43, r56, r22);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           12 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = r14 * r40;
    r22 = r22 * r39;
    r22 = r22 * r11;
    r22 = r22 * r38;
    r56 = r40 * r45;
    r56 = r56 * r24;
    r56 = r56 * r11;
    r43 = r40 * r24;
    r43 = r43 * r11;
    r43 = r43 * r4;
    WriteSum4<float, float>((float*)inout_shared, r22, r56, r43, r57);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           16 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r1 * r31;
    r43 = r43 * r40;
    r40 = r31 * r24;
    r40 = r40 * r49;
    r56 = r49 * r9;
    WriteSum4<float, float>((float*)inout_shared, r43, r57, r40, r56);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           20 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = r0 * r36;
    r40 = r49 * r8;
    r43 = r45 * r24;
    r43 = r43 * r49;
    r49 = r24 * r49;
    r49 = r49 * r4;
    WriteSum4<float, float>((float*)inout_shared, r40, r56, r43, r49);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           24 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r0 * r31;
    r43 = r31 * r24;
    r43 = r43 * r11;
    WriteSum4<float, float>((float*)inout_shared, r49, r57, r43, r53);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           28 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r1 * r34;
    r43 = r14 * r39;
    r43 = r43 * r11;
    r43 = r43 * r38;
    r49 = r45 * r24;
    r49 = r49 * r11;
    r11 = r24 * r11;
    r11 = r11 * r4;
    WriteSum4<float, float>((float*)inout_shared, r53, r43, r49, r11);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           32 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = r1 * r31;
    r49 = r45 * r52;
    r43 = fmaf(r45, r48, r13 * r49);
    r53 = r31 * r52;
    r41 = 1.0 / r41;
    r41 = r47 * r41;
    r46 = 1.0 / r46;
    r41 = r41 * r46;
    r53 = r53 * r10;
    r46 = r31 * r34;
    r46 = r46 * r24;
    r46 = fmaf(r54, r46, r41 * r53);
    WriteSum4<float, float>((float*)inout_shared, r57, r11, r43, r46);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           36 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r29 * r9;
    r43 = r36 * r31;
    r43 = r43 * r24;
    r11 = r14 * r39;
    r11 = r11 * r51;
    r11 = r11 * r41;
    r43 = fmaf(r31, r11, r29 * r43);
    r51 = r31 * r4;
    r53 = r13 * r51;
    r53 = fmaf(r51, r48, r52 * r53);
    WriteSum4<float, float>((float*)inout_shared, r43, r50, r53, r46);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           40 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r54 * r9;
    r50 = r32 * r52;
    r50 = r50 * r10;
    r43 = r34 * r54;
    r43 = fmaf(r9, r43, r41 * r50);
    r50 = r36 * r29;
    r50 = fmaf(r32, r11, r9 * r50);
    WriteSum4<float, float>((float*)inout_shared, r46, r43, r50, r53);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           44 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r45 * r24;
    r53 = r53 * r29;
    r50 = r45 * r24;
    r50 = r50 * r54;
    r8 = r29 * r8;
    r43 = r14 * r34;
    r43 = r43 * r39;
    r43 = r43 * r38;
    r43 = fmaf(r54, r43, r36 * r8);
    WriteSum4<float, float>((float*)inout_shared, r35, r53, r50, r43);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           48 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = r31 * r8;
    r43 = r1 * r1;
    r43 = r43 * r31;
    r43 = r43 * r34;
    r50 = r10 * r41;
    r53 = r34 * r45;
    r53 = r53 * r24;
    r53 = fmaf(r54, r53, r49 * r50);
    r50 = r4 * r52;
    r50 = r50 * r10;
    r49 = r34 * r24;
    r49 = r49 * r4;
    r49 = fmaf(r54, r49, r41 * r50);
    WriteSum4<float, float>((float*)inout_shared, r53, r49, r8, r43);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           52 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r0 * r0;
    r43 = r43 * r36;
    r43 = r43 * r31;
    r8 = r14 * r31;
    r8 = r8 * r39;
    r8 = r8 * r38;
    r8 = r8 * r54;
    r38 = r36 * r45;
    r38 = r38 * r24;
    r38 = fmaf(r45, r11, r29 * r38);
    r49 = r36 * r24;
    r49 = r49 * r4;
    r11 = fmaf(r4, r11, r29 * r49);
    WriteSum4<float, float>((float*)inout_shared, r38, r11, r43, r8);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           56 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = r24 * r4;
    r8 = r8 * r29;
    r43 = r24 * r4;
    r43 = r43 * r54;
    r11 = r24 * r29;
    r11 = r11 * r51;
    r38 = r13 * r52;
    r37 = r31 * r37;
    r37 = fmaf(r48, r37, r37 * r38);
    WriteSum4<float, float>((float*)inout_shared, r37, r8, r43, r11);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           60 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = r24 * r54;
    r11 = r11 * r51;
    WriteSum2<float, float>((float*)inout_shared, r11, r57);
  };
  FlushSumShared<2, float>(out_calib_precond_tril,
                           64 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
}

void ThinPrismFisheyeFixedPoseFixedPointResJac(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    float* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFixedPoseFixedPointResJacKernel<<<n_blocks, 1024>>>(
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar