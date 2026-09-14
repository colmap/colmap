#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_fixed_pose_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) OpencvFixedPoseResJacFirstKernel(
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    float *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    float *pixel, unsigned int pixel_num_alloc, float *pose,
    unsigned int pose_num_alloc, float *out_res, unsigned int out_res_num_alloc,
    float *const out_rTr, float *out_calib_jac,
    unsigned int out_calib_jac_num_alloc, float *const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc, float *const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float *const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc, float *out_point_jac,
    unsigned int out_point_jac_num_alloc, float *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, float *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

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
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66;
  LoadShared<4, float, float>(calib, 4 * calib_num_alloc, calib_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       calib_indices_loc[threadIdx.x].target, r0, r1, r2, r3);
  };
  __syncthreads();
  LoadShared<4, float, float>(calib, 0 * calib_num_alloc, calib_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       calib_indices_loc[threadIdx.x].target, r4, r5, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r8, r9, r10);
  };
  LoadShared<3, float, float>(point, 0 * point_num_alloc, point_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       point_indices_loc[threadIdx.x].target, r11, r12, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r14 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r15, r16, r17, r18);
    ReadIdx4<1024, float, float, float4>(pose, 0 * pose_num_alloc,
                                         global_thread_idx, r19, r20, r21, r22);
    r23 = fmaf(r15, r20, r18 * r21);
    r24 = r16 * r19;
    r25 = -1.00000000000000000e+00;
    r23 = fmaf(r25, r24, r23);
    r23 = fmaf(r17, r22, r23);
    r24 = r23 * r23;
    r24 = r14 * r24;
    r26 = 1.00000000000000000e+00;
    r27 = r15 * r21;
    r27 = fmaf(r25, r27, r18 * r20);
    r27 = fmaf(r16, r22, r27);
    r27 = fmaf(r17, r19, r27);
    r28 = r27 * r27;
    r28 = fmaf(r14, r28, r26);
    r29 = r24 + r28;
    r8 = fmaf(r11, r29, r8);
    r30 = 2.00000000000000000e+00;
    r31 = fmaf(r15, r22, r18 * r19);
    r32 = r17 * r20;
    r31 = fmaf(r25, r32, r31);
    r31 = fmaf(r16, r21, r31);
    r32 = r30 * r31;
    r32 = r32 * r27;
    r33 = fmaf(r16, r20, r15 * r19);
    r33 = fmaf(r17, r21, r33);
    r33 = fmaf(r25, r33, r18 * r22);
    r22 = r14 * r33;
    r34 = fmaf(r23, r22, r32);
    r35 = r30 * r23;
    r35 = r35 * r31;
    r36 = r30 * r27;
    r36 = fmaf(r33, r36, r35);
    ReadIdx3<1024, float, float, float4>(pose, 4 * pose_num_alloc,
                                         global_thread_idx, r37, r38, r39);
    r40 = r15 * r17;
    r40 = r40 * r30;
    r41 = r16 * r18;
    r42 = fmaf(r30, r41, r40);
    r43 = r17 * r18;
    r44 = r15 * r16;
    r44 = r44 * r30;
    r43 = fmaf(r14, r43, r44);
    r45 = r16 * r16;
    r45 = r45 * r14;
    r46 = r26 + r45;
    r47 = r17 * r17;
    r47 = r14 * r47;
    r46 = r46 + r47;
    r8 = fmaf(r12, r34, r8);
    r8 = fmaf(r13, r36, r8);
    r8 = fmaf(r39, r42, r8);
    r8 = fmaf(r38, r43, r8);
    r8 = fmaf(r37, r46, r8);
    r46 = 3.00000000000000000e+00;
    r43 = r8 * r46;
    r42 = 9.99999999999999955e-07;
    r35 = fmaf(r27, r22, r35);
    r10 = fmaf(r11, r35, r10);
    r41 = fmaf(r14, r41, r40);
    r45 = r26 + r45;
    r40 = r15 * r15;
    r40 = r14 * r40;
    r45 = r45 + r40;
    r48 = r16 * r17;
    r48 = r48 * r30;
    r49 = r15 * r18;
    r49 = fmaf(r30, r49, r48);
    r50 = r30 * r23;
    r50 = r50 * r27;
    r51 = r30 * r31;
    r51 = fmaf(r33, r51, r50);
    r52 = r31 * r31;
    r52 = r14 * r52;
    r28 = r52 + r28;
    r10 = fmaf(r37, r41, r10);
    r10 = fmaf(r39, r45, r10);
    r10 = fmaf(r38, r49, r10);
    r10 = fmaf(r12, r51, r10);
    r10 = fmaf(r13, r28, r10);
    r49 = copysign(1.0, r10);
    r49 = fmaf(r42, r49, r10);
    r42 = r49 * r49;
    r10 = 1.0 / r42;
    r45 = r8 * r10;
    r41 = r30 * r23;
    r41 = fmaf(r33, r41, r32);
    r11 = fmaf(r11, r41, r9);
    r9 = r17 * r18;
    r9 = fmaf(r30, r9, r44);
    r47 = r26 + r47;
    r47 = r47 + r40;
    r40 = r15 * r18;
    r40 = fmaf(r14, r40, r48);
    r22 = fmaf(r31, r22, r50);
    r24 = r26 + r24;
    r24 = r24 + r52;
    r11 = fmaf(r37, r9, r11);
    r11 = fmaf(r38, r47, r11);
    r11 = fmaf(r39, r40, r11);
    r11 = fmaf(r13, r22, r11);
    r11 = fmaf(r12, r24, r11);
    r12 = r11 * r11;
    r12 = r12 * r10;
    r43 = fmaf(r45, r43, r12);
    r13 = 1.0 / r49;
    r40 = fmaf(r8, r13, r1 * r43);
    r39 = r8 * r45;
    r12 = r12 + r39;
    r47 = r12 * r12;
    r38 = fmaf(r7, r47, r6 * r12);
    r9 = r38 * r13;
    r37 = r30 * r11;
    r52 = r0 * r45;
    r40 = fmaf(r8, r9, r40);
    r40 = fmaf(r37, r52, r40);
    r2 = fmaf(r4, r40, r2);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r50, r48);
    r2 = fmaf(r50, r25, r2);
    r50 = r46 * r11;
    r50 = r50 * r11;
    r50 = fmaf(r10, r50, r39);
    r39 = fmaf(r11, r13, r0 * r50);
    r44 = r1 * r45;
    r39 = fmaf(r37, r44, r39);
    r39 = fmaf(r11, r9, r39);
    r3 = fmaf(r5, r39, r3);
    r3 = fmaf(r48, r25, r3);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r2, r3);
    r48 = fmaf(r3, r3, r2 * r2);
  };
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r48);
  if (global_thread_idx < problem_size) {
    r48 = r4 * r8;
    r48 = r48 * r12;
    r48 = r48 * r13;
    r44 = r5 * r11;
    r44 = r44 * r12;
    r44 = r44 * r13;
    WriteIdx4<1024, float, float, float4>(
        out_calib_jac, 0 * out_calib_jac_num_alloc, global_thread_idx, r40, r39,
        r48, r44);
    r32 = r5 * r50;
    r33 = r4 * r13;
    r53 = r8 * r47;
    r33 = r33 * r53;
    r54 = r5 * r11;
    r54 = r54 * r13;
    r54 = r54 * r47;
    r55 = r4 * r45;
    r55 = r55 * r37;
    WriteIdx4<1024, float, float, float4>(
        out_calib_jac, 4 * out_calib_jac_num_alloc, global_thread_idx, r33, r54,
        r55, r32);
    r56 = r4 * r43;
    r57 = r5 * r45;
    r57 = r57 * r37;
    WriteIdx2<1024, float, float, float2>(out_calib_jac,
                                          8 * out_calib_jac_num_alloc,
                                          global_thread_idx, r56, r57);
    r58 = r25 * r39;
    r58 = r58 * r3;
    r59 = r25 * r2;
    r60 = r40 * r59;
    r61 = r5 * r25;
    r61 = r61 * r11;
    r61 = r61 * r12;
    r61 = r61 * r3;
    r62 = r8 * r12;
    r63 = r4 * r59;
    r64 = r13 * r63;
    r62 = fmaf(r64, r62, r13 * r61);
    r61 = r5 * r25;
    r61 = r61 * r11;
    r61 = r61 * r3;
    r61 = r61 * r13;
    r64 = fmaf(r53, r64, r47 * r61);
    WriteSum4<float, float>((float *)inout_shared, r60, r58, r62, r64);
  };
  FlushSumShared<4, float>(out_calib_njtr, 0 * out_calib_njtr_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = r25 * r3;
    r62 = r5 * r25;
    r62 = r62 * r50;
    r58 = r4 * r14;
    r58 = r58 * r11;
    r58 = r58 * r2;
    r58 = fmaf(r45, r58, r3 * r62);
    r62 = r5 * r14;
    r62 = r62 * r11;
    r62 = r62 * r3;
    r62 = fmaf(r43, r63, r45 * r62);
    WriteSum4<float, float>((float *)inout_shared, r58, r62, r59, r64);
  };
  FlushSumShared<4, float>(out_calib_njtr, 4 * out_calib_njtr_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = r40 * r40;
    r59 = r39 * r39;
    r62 = r5 * r5;
    r58 = r11 * r62;
    r2 = r11 * r10;
    r2 = r2 * r47;
    r60 = r4 * r4;
    r61 = r60 * r53;
    r2 = fmaf(r45, r61, r58 * r2);
    r65 = r11 * r11;
    r65 = r10 * r65;
    r66 = r12 * r47;
    r65 = r65 * r62;
    r65 = r65 * r66;
    r66 = r12 * r12;
    r66 = r66 * r45;
    r66 = fmaf(r61, r66, r12 * r65);
    WriteSum4<float, float>((float *)inout_shared, r64, r59, r2, r66);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = r50 * r50;
    r2 = r11 * r60;
    r59 = r8 * r8;
    r64 = 4.00000000000000000e+00;
    r42 = r49 * r42;
    r49 = r49 * r42;
    r49 = 1.0 / r49;
    r59 = r59 * r11;
    r59 = r59 * r64;
    r59 = r59 * r49;
    r2 = fmaf(r59, r2, r62 * r66);
    r66 = r43 * r43;
    r59 = fmaf(r58, r59, r60 * r66);
    WriteSum4<float, float>((float *)inout_shared, r2, r59, r26, r26);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           4 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = 0.00000000000000000e+00;
    r59 = r4 * r8;
    r59 = r59 * r12;
    r59 = r59 * r40;
    r59 = r59 * r13;
    r2 = r4 * r40;
    r2 = r2 * r13;
    r2 = r2 * r53;
    r66 = r4 * r40;
    r66 = r66 * r45;
    r66 = r66 * r37;
    WriteSum4<float, float>((float *)inout_shared, r26, r59, r2, r66);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = r4 * r43;
    r66 = r66 * r40;
    r2 = r5 * r11;
    r2 = r2 * r12;
    r2 = r2 * r39;
    r2 = r2 * r13;
    WriteSum4<float, float>((float *)inout_shared, r66, r40, r26, r2);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r5 * r50;
    r2 = r2 * r39;
    r40 = r5 * r11;
    r40 = r40 * r39;
    r40 = r40 * r13;
    r40 = r40 * r47;
    r66 = r5 * r39;
    r66 = r66 * r45;
    r66 = r66 * r37;
    WriteSum4<float, float>((float *)inout_shared, r40, r2, r66, r26);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           8 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = r12 * r45;
    r66 = fmaf(r61, r66, r65);
    r65 = r8 * r8;
    r42 = 1.0 / r42;
    r65 = r65 * r12;
    r65 = r65 * r42;
    r65 = r65 * r37;
    r2 = r50 * r13;
    r40 = r12 * r58;
    r2 = fmaf(r40, r2, r60 * r65);
    r65 = r8 * r42;
    r65 = r65 * r37;
    r59 = r8 * r43;
    r59 = r59 * r12;
    r59 = r59 * r13;
    r59 = fmaf(r60, r59, r40 * r65);
    WriteSum4<float, float>((float *)inout_shared, r39, r66, r2, r59);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           12 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = r8 * r42;
    r59 = r59 * r37;
    r2 = r50 * r13;
    r2 = r2 * r47;
    r2 = fmaf(r58, r2, r61 * r59);
    r59 = r42 * r37;
    r59 = r59 * r53;
    r53 = r43 * r13;
    r53 = fmaf(r61, r53, r58 * r59);
    WriteSum4<float, float>((float *)inout_shared, r48, r44, r2, r53);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           16 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r50 * r45;
    r53 = r53 * r37;
    r2 = r43 * r45;
    r2 = r2 * r37;
    r2 = fmaf(r60, r2, r62 * r53);
    WriteSum4<float, float>((float *)inout_shared, r33, r54, r2, r55);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           20 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float *)inout_shared, r32, r56, r57, r26);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           24 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = 6.00000000000000000e+00;
    r57 = r29 * r26;
    r56 = r8 * r8;
    r32 = -6.00000000000000000e+00;
    r55 = r35 * r32;
    r55 = r55 * r42;
    r56 = fmaf(r55, r56, r45 * r57);
    r57 = r10 * r37;
    r2 = r14 * r35;
    r2 = r2 * r11;
    r2 = r2 * r11;
    r2 = fmaf(r42, r2, r41 * r57);
    r56 = r56 + r2;
    r56 = fmaf(r29, r13, r1 * r56);
    r54 = r0 * r35;
    r33 = r8 * r11;
    r53 = -4.00000000000000000e+00;
    r33 = r33 * r53;
    r33 = r33 * r42;
    r56 = fmaf(r33, r54, r56);
    r53 = r0 * r29;
    r56 = fmaf(r57, r53, r56);
    r44 = r25 * r35;
    r56 = fmaf(r45, r44, r56);
    r48 = r25 * r35;
    r48 = r48 * r38;
    r56 = fmaf(r45, r48, r56);
    r59 = r30 * r29;
    r61 = r14 * r35;
    r61 = r61 * r8;
    r61 = r61 * r8;
    r61 = fmaf(r42, r61, r45 * r59);
    r2 = r2 + r61;
    r59 = r7 * r30;
    r59 = r59 * r12;
    r59 = fmaf(r2, r59, r6 * r2);
    r2 = r8 * r59;
    r56 = fmaf(r13, r2, r56);
    r52 = r30 * r52;
    r56 = fmaf(r29, r9, r56);
    r56 = fmaf(r41, r52, r56);
    r2 = r4 * r56;
    r48 = r41 * r11;
    r48 = r48 * r26;
    r44 = r11 * r11;
    r44 = fmaf(r55, r44, r10 * r48);
    r44 = r44 + r61;
    r61 = r1 * r35;
    r61 = fmaf(r33, r61, r0 * r44);
    r44 = r25 * r35;
    r44 = r44 * r11;
    r44 = r44 * r38;
    r61 = fmaf(r10, r44, r61);
    r48 = r11 * r59;
    r61 = fmaf(r13, r48, r61);
    r55 = r1 * r57;
    r53 = r25 * r35;
    r53 = r53 * r11;
    r61 = fmaf(r10, r53, r61);
    r54 = r1 * r30;
    r54 = r54 * r41;
    r61 = fmaf(r45, r54, r61);
    r61 = fmaf(r41, r13, r61);
    r61 = fmaf(r29, r55, r61);
    r61 = fmaf(r41, r9, r61);
    r54 = r5 * r61;
    r53 = r34 * r26;
    r48 = r51 * r8;
    r48 = r48 * r8;
    r48 = r48 * r32;
    r48 = fmaf(r42, r48, r45 * r53);
    r53 = r14 * r51;
    r53 = r53 * r11;
    r53 = r53 * r11;
    r53 = fmaf(r24, r57, r42 * r53);
    r48 = r48 + r53;
    r48 = fmaf(r24, r52, r1 * r48);
    r44 = r30 * r34;
    r58 = r14 * r51;
    r58 = r58 * r8;
    r58 = r58 * r8;
    r58 = fmaf(r42, r58, r45 * r44);
    r53 = r53 + r58;
    r44 = r7 * r30;
    r44 = r44 * r12;
    r44 = fmaf(r53, r44, r6 * r53);
    r53 = r8 * r44;
    r48 = fmaf(r13, r53, r48);
    r47 = r0 * r34;
    r48 = fmaf(r57, r47, r48);
    r66 = r51 * r33;
    r39 = r25 * r51;
    r39 = r39 * r38;
    r48 = fmaf(r45, r39, r48);
    r65 = r25 * r51;
    r48 = fmaf(r45, r65, r48);
    r48 = fmaf(r34, r13, r48);
    r48 = fmaf(r34, r9, r48);
    r48 = fmaf(r0, r66, r48);
    r65 = r4 * r48;
    r39 = r51 * r11;
    r39 = r39 * r11;
    r39 = r39 * r32;
    r47 = r24 * r11;
    r47 = r47 * r26;
    r47 = fmaf(r10, r47, r42 * r39);
    r47 = r47 + r58;
    r58 = r25 * r51;
    r58 = r58 * r11;
    r58 = r58 * r38;
    r58 = fmaf(r10, r58, r0 * r47);
    r47 = r25 * r51;
    r47 = r47 * r11;
    r58 = fmaf(r10, r47, r58);
    r39 = r1 * r30;
    r39 = r39 * r24;
    r58 = fmaf(r45, r39, r58);
    r53 = r11 * r44;
    r58 = fmaf(r13, r53, r58);
    r58 = fmaf(r24, r13, r58);
    r58 = fmaf(r24, r9, r58);
    r58 = fmaf(r34, r55, r58);
    r58 = fmaf(r1, r66, r58);
    r53 = r5 * r58;
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx, r2, r54, r65, r53);
    r53 = r28 * r8;
    r53 = r53 * r8;
    r53 = r53 * r32;
    r65 = r36 * r26;
    r65 = fmaf(r45, r65, r42 * r53);
    r53 = r14 * r28;
    r53 = r53 * r11;
    r53 = r53 * r11;
    r53 = fmaf(r42, r53, r22 * r57);
    r65 = r65 + r53;
    r54 = r14 * r28;
    r54 = r54 * r8;
    r54 = r54 * r8;
    r2 = r30 * r36;
    r2 = fmaf(r45, r2, r42 * r54);
    r53 = r53 + r2;
    r54 = r7 * r30;
    r54 = r54 * r12;
    r54 = fmaf(r53, r54, r6 * r53);
    r53 = r8 * r54;
    r53 = fmaf(r13, r53, r1 * r65);
    r65 = r25 * r28;
    r53 = fmaf(r45, r65, r53);
    r6 = r25 * r28;
    r6 = r6 * r38;
    r53 = fmaf(r45, r6, r53);
    r66 = r0 * r28;
    r53 = fmaf(r33, r66, r53);
    r39 = r0 * r36;
    r53 = fmaf(r57, r39, r53);
    r53 = fmaf(r22, r52, r53);
    r53 = fmaf(r36, r13, r53);
    r53 = fmaf(r36, r9, r53);
    r39 = r4 * r53;
    r66 = r22 * r11;
    r66 = r66 * r26;
    r6 = r28 * r11;
    r6 = r6 * r11;
    r6 = r6 * r32;
    r6 = fmaf(r42, r6, r10 * r66);
    r6 = r6 + r2;
    r6 = fmaf(r22, r13, r0 * r6);
    r2 = r1 * r30;
    r2 = r2 * r22;
    r6 = fmaf(r45, r2, r6);
    r66 = r25 * r28;
    r66 = r66 * r11;
    r6 = fmaf(r10, r66, r6);
    r32 = r25 * r28;
    r32 = r32 * r11;
    r32 = r32 * r38;
    r6 = fmaf(r10, r32, r6);
    r10 = r1 * r28;
    r6 = fmaf(r33, r10, r6);
    r33 = r11 * r54;
    r6 = fmaf(r13, r33, r6);
    r6 = fmaf(r22, r9, r6);
    r6 = fmaf(r36, r55, r6);
    r55 = r5 * r6;
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx, r39, r55);
    r55 = r5 * r25;
    r55 = r55 * r3;
    r55 = fmaf(r56, r63, r61 * r55);
    r39 = r5 * r25;
    r39 = r39 * r3;
    r39 = fmaf(r48, r63, r58 * r39);
    r33 = r5 * r25;
    r33 = r33 * r3;
    r63 = fmaf(r53, r63, r6 * r33);
    WriteSum3<float, float>((float *)inout_shared, r55, r39, r63);
  };
  FlushSumShared<3, float>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = r56 * r56;
    r39 = r61 * r61;
    r39 = fmaf(r62, r39, r60 * r63);
    r63 = r48 * r48;
    r55 = r58 * r58;
    r55 = fmaf(r62, r55, r60 * r63);
    r63 = r53 * r53;
    r33 = r6 * r6;
    r33 = fmaf(r62, r33, r60 * r63);
    WriteSum3<float, float>((float *)inout_shared, r39, r55, r33);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = r61 * r58;
    r55 = r56 * r48;
    r55 = fmaf(r60, r55, r62 * r33);
    r33 = r56 * r53;
    r39 = r61 * r6;
    r39 = fmaf(r62, r39, r60 * r33);
    r33 = r48 * r53;
    r63 = r58 * r6;
    r63 = fmaf(r62, r63, r60 * r33);
    WriteSum3<float, float>((float *)inout_shared, r55, r39, r63);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvFixedPoseResJacFirst(
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    float *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    float *pixel, unsigned int pixel_num_alloc, float *pose,
    unsigned int pose_num_alloc, float *out_res, unsigned int out_res_num_alloc,
    float *const out_rTr, float *out_calib_jac,
    unsigned int out_calib_jac_num_alloc, float *const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc, float *const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float *const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc, float *out_point_jac,
    unsigned int out_point_jac_num_alloc, float *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, float *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvFixedPoseResJacFirstKernel<<<n_blocks, 1024>>>(
      sensor_from_rig, sensor_from_rig_num_alloc, calib, calib_num_alloc,
      calib_indices, point, point_num_alloc, point_indices, pixel,
      pixel_num_alloc, pose, pose_num_alloc, out_res, out_res_num_alloc,
      out_rTr, out_calib_jac, out_calib_jac_num_alloc, out_calib_njtr,
      out_calib_njtr_num_alloc, out_calib_precond_diag,
      out_calib_precond_diag_num_alloc, out_calib_precond_tril,
      out_calib_precond_tril_num_alloc, out_point_jac, out_point_jac_num_alloc,
      out_point_njtr, out_point_njtr_num_alloc, out_point_precond_diag,
      out_point_precond_diag_num_alloc, out_point_precond_tril,
      out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar