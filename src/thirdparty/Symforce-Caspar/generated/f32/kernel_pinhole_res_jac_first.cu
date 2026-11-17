#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_pinhole_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) PinholeResJacFirstKernel(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    float *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    float *pixel, unsigned int pixel_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *const out_rTr, float *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, float *out_calib_jac,
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
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84;
  LoadShared<4, float, float>(calib, 0 * calib_num_alloc, calib_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       calib_indices_loc[threadIdx.x].target, r0, r1, r2, r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r6, r2);
    r2 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r7, r8, r9);
  };
  LoadShared<3, float, float>(point, 0 * point_num_alloc, point_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       point_indices_loc[threadIdx.x].target, r10, r11, r12);
  };
  __syncthreads();
  LoadShared<4, float, float>(pose, 0 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r13, r14, r15,
                       r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r17, r18, r19, r20);
    r21 = fmaf(r14, r17, r15 * r20);
    r22 = r13 * r18;
    r21 = fmaf(r6, r22, r21);
    r21 = fmaf(r16, r19, r21);
    r22 = 2.00000000000000000e+00;
    r23 = fmaf(r16, r17, r13 * r20);
    r24 = r14 * r19;
    r23 = fmaf(r6, r24, r23);
    r23 = fmaf(r15, r18, r23);
    r24 = r22 * r23;
    r25 = r21 * r24;
    r26 = -2.00000000000000000e+00;
    r27 = fmaf(r14, r18, r13 * r17);
    r27 = fmaf(r15, r19, r27);
    r27 = fmaf(r6, r27, r16 * r20);
    r28 = r26 * r27;
    r29 = r16 * r18;
    r30 = fmaf(r14, r20, r29);
    r31 = r13 * r19;
    r32 = r15 * r17;
    r30 = r30 + r31;
    r30 = fmaf(r6, r32, r30);
    r33 = fmaf(r30, r28, r25);
    r9 = fmaf(r10, r33, r9);
  };
  LoadShared<3, float, float>(pose, 4 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r34, r35, r36);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r37 = r17 * r19;
    r37 = r37 * r22;
    r38 = r18 * r20;
    r38 = fmaf(r26, r38, r37);
    r39 = r17 * r17;
    r39 = r39 * r26;
    r40 = 1.00000000000000000e+00;
    r41 = r18 * r18;
    r41 = fmaf(r26, r41, r40);
    r42 = r39 + r41;
    r43 = r18 * r19;
    r43 = r43 * r22;
    r44 = r17 * r20;
    r44 = fmaf(r22, r44, r43);
    r45 = r22 * r21;
    r45 = r45 * r30;
    r46 = fmaf(r27, r24, r45);
    r47 = r26 * r30;
    r47 = r47 * r30;
    r48 = r40 + r47;
    r49 = r23 * r23;
    r49 = r49 * r26;
    r48 = r48 + r49;
    r9 = fmaf(r34, r38, r9);
    r9 = fmaf(r36, r42, r9);
    r9 = fmaf(r35, r44, r9);
    r9 = fmaf(r11, r46, r9);
    r9 = fmaf(r12, r48, r9);
    r50 = copysign(1.0, r9);
    r50 = fmaf(r2, r50, r9);
    r2 = 1.0 / r50;
    r47 = r40 + r47;
    r9 = r21 * r21;
    r9 = r9 * r26;
    r47 = r47 + r9;
    r7 = fmaf(r10, r47, r7);
    r51 = r30 * r24;
    r52 = fmaf(r21, r28, r51);
    r53 = r22 * r30;
    r53 = fmaf(r27, r53, r25);
    r25 = r18 * r20;
    r25 = fmaf(r22, r25, r37);
    r37 = r19 * r20;
    r54 = r17 * r18;
    r54 = r54 * r22;
    r37 = fmaf(r26, r37, r54);
    r55 = r19 * r19;
    r55 = r55 * r26;
    r41 = r55 + r41;
    r7 = fmaf(r11, r52, r7);
    r7 = fmaf(r12, r53, r7);
    r7 = fmaf(r36, r25, r7);
    r7 = fmaf(r35, r37, r7);
    r7 = fmaf(r34, r41, r7);
    r56 = r0 * r7;
    r4 = fmaf(r2, r56, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r22 * r21;
    r3 = fmaf(r27, r3, r51);
    r8 = fmaf(r10, r3, r8);
    r51 = r19 * r20;
    r51 = fmaf(r22, r51, r54);
    r55 = r40 + r55;
    r55 = r55 + r39;
    r39 = r17 * r20;
    r39 = fmaf(r26, r39, r43);
    r45 = fmaf(r23, r28, r45);
    r9 = r40 + r9;
    r9 = r9 + r49;
    r8 = fmaf(r34, r51, r8);
    r8 = fmaf(r35, r55, r8);
    r8 = fmaf(r36, r39, r8);
    r8 = fmaf(r12, r45, r8);
    r8 = fmaf(r11, r9, r8);
    r36 = r1 * r8;
    r5 = fmaf(r2, r36, r5);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r4, r5);
    r35 = fmaf(r5, r5, r4 * r4);
  };
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r35);
  if (global_thread_idx < problem_size) {
    r35 = r22 * r30;
    r34 = -5.00000000000000000e-01;
    r49 = r14 * r34;
    r43 = 5.00000000000000000e-01;
    r54 = fmaf(r43, r32, r20 * r49);
    r54 = fmaf(r34, r29, r54);
    r54 = fmaf(r34, r31, r54);
    r35 = r35 * r54;
    r57 = r22 * r21;
    r58 = r14 * r17;
    r59 = r13 * r18;
    r59 = fmaf(r34, r59, r43 * r58);
    r58 = r16 * r19;
    r59 = fmaf(r43, r58, r59);
    r60 = r20 * r43;
    r59 = fmaf(r15, r60, r59);
    r57 = fmaf(r59, r57, r35);
    r58 = r22 * r27;
    r61 = r13 * r17;
    r62 = r15 * r19;
    r62 = fmaf(r34, r62, r34 * r61);
    r62 = fmaf(r16, r60, r62);
    r62 = fmaf(r18, r49, r62);
    r58 = r58 * r62;
    r61 = r13 * r20;
    r63 = r16 * r17;
    r63 = fmaf(r34, r63, r34 * r61);
    r61 = r15 * r18;
    r63 = fmaf(r34, r61, r63);
    r64 = r14 * r19;
    r63 = fmaf(r43, r64, r63);
    r64 = r63 * r24;
    r61 = r58 + r64;
    r65 = r57 + r61;
    r66 = r30 * r63;
    r67 = fmaf(r59, r28, r26 * r66);
    r68 = r22 * r21;
    r68 = r68 * r62;
    r69 = fmaf(r54, r24, r68);
    r67 = r67 + r69;
    r67 = fmaf(r10, r67, r11 * r65);
    r65 = r30 * r59;
    r70 = -4.00000000000000000e+00;
    r65 = r65 * r70;
    r71 = r62 * r70;
    r72 = r23 * r71;
    r73 = r65 + r72;
    r67 = fmaf(r12, r73, r67);
    r50 = r50 * r50;
    r50 = 1.0 / r50;
    r73 = r6 * r50;
    r56 = r73 * r56;
    r74 = r22 * r27;
    r74 = fmaf(r22, r66, r59 * r74);
    r74 = r74 + r69;
    r75 = r22 * r30;
    r75 = r75 * r62;
    r76 = r21 * r26;
    r76 = fmaf(r63, r76, r75);
    r59 = r59 * r24;
    r76 = r76 + r59;
    r76 = fmaf(r54, r28, r76);
    r76 = fmaf(r11, r76, r12 * r74);
    r74 = r21 * r54;
    r77 = r70 * r74;
    r65 = r65 + r77;
    r76 = fmaf(r10, r65, r76);
    r65 = r0 * r76;
    r65 = fmaf(r2, r65, r67 * r56);
    r78 = r23 * r26;
    r79 = r62 * r28;
    r78 = fmaf(r63, r78, r79);
    r78 = r78 + r57;
    r77 = r72 + r77;
    r77 = fmaf(r11, r77, r12 * r78);
    r78 = r22 * r27;
    r78 = fmaf(r54, r78, r59);
    r59 = r22 * r21;
    r59 = fmaf(r63, r59, r75);
    r78 = r78 + r59;
    r77 = fmaf(r10, r78, r77);
    r78 = r1 * r77;
    r75 = r67 * r73;
    r75 = fmaf(r36, r75, r2 * r78);
    r58 = r35 + r58;
    r35 = r22 * r21;
    r78 = r15 * r20;
    r72 = r13 * r18;
    r72 = fmaf(r43, r72, r34 * r78);
    r78 = r16 * r19;
    r72 = fmaf(r34, r78, r72);
    r72 = fmaf(r17, r49, r72);
    r35 = r35 * r72;
    r78 = r16 * r17;
    r57 = r15 * r18;
    r57 = fmaf(r43, r57, r43 * r78);
    r57 = fmaf(r13, r60, r57);
    r57 = fmaf(r19, r49, r57);
    r49 = fmaf(r57, r24, r35);
    r58 = r58 + r49;
    r78 = r21 * r70;
    r78 = r78 * r57;
    r80 = r30 * r71;
    r81 = r78 + r80;
    r81 = fmaf(r10, r81, r12 * r58);
    r58 = fmaf(r57, r28, r26 * r74);
    r82 = r22 * r30;
    r62 = r62 * r24;
    r82 = fmaf(r72, r82, r62);
    r58 = r58 + r82;
    r81 = fmaf(r11, r58, r81);
    r58 = r0 * r81;
    r83 = r26 * r30;
    r83 = fmaf(r54, r83, r79);
    r83 = r83 + r49;
    r49 = r22 * r30;
    r49 = r49 * r57;
    r84 = r22 * r27;
    r84 = fmaf(r72, r84, r49);
    r84 = r84 + r69;
    r84 = fmaf(r11, r84, r10 * r83);
    r83 = r23 * r70;
    r83 = r83 * r72;
    r80 = r83 + r80;
    r84 = fmaf(r12, r80, r84);
    r58 = fmaf(r84, r56, r2 * r58);
    r80 = r84 * r73;
    r49 = r68 + r49;
    r68 = r23 * r26;
    r49 = fmaf(r54, r68, r49);
    r49 = fmaf(r72, r28, r49);
    r68 = r22 * r27;
    r74 = fmaf(r22, r74, r57 * r68);
    r74 = r74 + r82;
    r74 = fmaf(r10, r74, r12 * r49);
    r83 = r78 + r83;
    r74 = fmaf(r11, r83, r74);
    r83 = r1 * r74;
    r83 = fmaf(r2, r83, r36 * r80);
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r65, r75,
        r58, r83);
    r80 = r23 * r70;
    r32 = fmaf(r34, r32, r14 * r60);
    r32 = fmaf(r43, r29, r32);
    r32 = fmaf(r43, r31, r32);
    r80 = r80 * r32;
    r66 = r70 * r66;
    r70 = r80 + r66;
    r31 = r22 * r21;
    r31 = r31 * r32;
    r43 = r26 * r30;
    r43 = fmaf(r72, r43, r31);
    r43 = r43 + r62;
    r43 = fmaf(r63, r28, r43);
    r43 = fmaf(r10, r43, r12 * r70);
    r70 = r22 * r27;
    r24 = fmaf(r72, r24, r32 * r70);
    r24 = r24 + r59;
    r43 = fmaf(r11, r24, r43);
    r24 = r22 * r30;
    r24 = r24 * r32;
    r70 = r21 * r26;
    r70 = fmaf(r72, r70, r24);
    r70 = r70 + r64;
    r70 = r70 + r79;
    r71 = r21 * r71;
    r66 = r66 + r71;
    r66 = fmaf(r10, r66, r11 * r70);
    r70 = r22 * r27;
    r70 = fmaf(r63, r70, r31);
    r70 = r70 + r82;
    r66 = fmaf(r12, r70, r66);
    r70 = r0 * r66;
    r70 = fmaf(r2, r70, r43 * r56);
    r82 = r43 * r73;
    r24 = r35 + r24;
    r24 = r24 + r61;
    r61 = r23 * r26;
    r28 = fmaf(r32, r28, r72 * r61);
    r28 = r28 + r59;
    r28 = fmaf(r12, r28, r10 * r24);
    r71 = r80 + r71;
    r28 = fmaf(r11, r71, r28);
    r71 = r1 * r28;
    r71 = fmaf(r2, r71, r36 * r82);
    r82 = r0 * r41;
    r82 = fmaf(r38, r56, r2 * r82);
    r11 = r1 * r51;
    r80 = r38 * r73;
    r80 = fmaf(r36, r80, r2 * r11);
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r70, r71,
        r82, r80);
    r11 = r0 * r37;
    r11 = fmaf(r44, r56, r2 * r11);
    r12 = r1 * r55;
    r24 = r44 * r73;
    r24 = fmaf(r36, r24, r2 * r12);
    r12 = r0 * r25;
    r12 = fmaf(r42, r56, r2 * r12);
    r10 = r42 * r73;
    r59 = r1 * r39;
    r59 = fmaf(r2, r59, r36 * r10);
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r11, r24,
        r12, r59);
    r10 = r6 * r5;
    r32 = r6 * r4;
    r32 = fmaf(r65, r32, r75 * r10);
    r10 = r6 * r5;
    r61 = r6 * r4;
    r61 = fmaf(r58, r61, r83 * r10);
    r10 = r6 * r4;
    r72 = r6 * r5;
    r72 = fmaf(r71, r72, r70 * r10);
    r10 = r6 * r4;
    r35 = r6 * r5;
    r35 = fmaf(r80, r35, r82 * r10);
    WriteSum4<float, float>((float *)inout_shared, r32, r61, r72, r35);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r6 * r5;
    r72 = r6 * r4;
    r72 = fmaf(r11, r72, r24 * r35);
    r35 = r6 * r4;
    r61 = r6 * r5;
    r61 = fmaf(r59, r61, r12 * r35);
    WriteSum2<float, float>((float *)inout_shared, r72, r61);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fmaf(r65, r65, r75 * r75);
    r72 = fmaf(r58, r58, r83 * r83);
    r35 = fmaf(r70, r70, r71 * r71);
    r32 = fmaf(r82, r82, r80 * r80);
    WriteSum4<float, float>((float *)inout_shared, r61, r72, r35, r32);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fmaf(r11, r11, r24 * r24);
    r35 = fmaf(r59, r59, r12 * r12);
    WriteSum2<float, float>((float *)inout_shared, r32, r35);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fmaf(r75, r83, r65 * r58);
    r32 = fmaf(r65, r70, r75 * r71);
    r72 = fmaf(r75, r80, r65 * r82);
    r61 = fmaf(r65, r11, r75 * r24);
    WriteSum4<float, float>((float *)inout_shared, r35, r32, r72, r61);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = fmaf(r75, r59, r65 * r12);
    r65 = fmaf(r83, r71, r58 * r70);
    r61 = fmaf(r58, r82, r83 * r80);
    r72 = fmaf(r58, r11, r83 * r24);
    WriteSum4<float, float>((float *)inout_shared, r75, r65, r61, r72);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = fmaf(r58, r12, r83 * r59);
    r83 = fmaf(r71, r80, r70 * r82);
    r72 = fmaf(r71, r24, r70 * r11);
    r70 = fmaf(r70, r12, r71 * r59);
    WriteSum4<float, float>((float *)inout_shared, r58, r83, r72, r70);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fmaf(r80, r24, r82 * r11);
    r80 = fmaf(r80, r59, r82 * r12);
    r12 = fmaf(r11, r12, r24 * r59);
    WriteSum3<float, float>((float *)inout_shared, r70, r80, r12);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = r7 * r2;
    r80 = r8 * r2;
    WriteIdx2<1024, float, float, float2>(out_calib_jac,
                                          0 * out_calib_jac_num_alloc,
                                          global_thread_idx, r12, r80);
    r70 = r6 * r4;
    r11 = r6 * r5;
    r59 = r6 * r7;
    r59 = r59 * r4;
    r59 = r59 * r2;
    r24 = r6 * r8;
    r24 = r24 * r5;
    r24 = r24 * r2;
    WriteSum4<float, float>((float *)inout_shared, r59, r24, r70, r11);
  };
  FlushSumShared<4, float>(out_calib_njtr, 0 * out_calib_njtr_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r7 * r7;
    r7 = r7 * r50;
    r8 = r8 * r8;
    r8 = r8 * r50;
    WriteSum4<float, float>((float *)inout_shared, r7, r8, r40, r40);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = 0.00000000000000000e+00;
    WriteSum4<float, float>((float *)inout_shared, r40, r12, r40, r40);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float *)inout_shared, r80, r40);
  };
  FlushSumShared<2, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r0 * r47;
    r40 = fmaf(r2, r40, r33 * r56);
    r80 = r1 * r3;
    r12 = r33 * r73;
    r12 = fmaf(r36, r12, r2 * r80);
    r80 = r0 * r52;
    r80 = fmaf(r2, r80, r46 * r56);
    r8 = r1 * r9;
    r7 = r46 * r73;
    r7 = fmaf(r36, r7, r2 * r8);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx, r40, r12, r80, r7);
    r8 = r0 * r53;
    r56 = fmaf(r48, r56, r2 * r8);
    r8 = r1 * r45;
    r50 = r48 * r73;
    r50 = fmaf(r36, r50, r2 * r8);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx, r56, r50);
    r8 = r6 * r5;
    r36 = r6 * r4;
    r36 = fmaf(r40, r36, r12 * r8);
    r8 = r6 * r5;
    r2 = r6 * r4;
    r2 = fmaf(r80, r2, r7 * r8);
    r8 = r6 * r5;
    r11 = r6 * r4;
    r11 = fmaf(r56, r11, r50 * r8);
    WriteSum3<float, float>((float *)inout_shared, r36, r2, r11);
  };
  FlushSumShared<3, float>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r40, r40, r12 * r12);
    r2 = fmaf(r7, r7, r80 * r80);
    r36 = fmaf(r56, r56, r50 * r50);
    WriteSum3<float, float>((float *)inout_shared, r11, r2, r36);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fmaf(r40, r80, r12 * r7);
    r12 = fmaf(r12, r50, r40 * r56);
    r50 = fmaf(r7, r50, r80 * r56);
    WriteSum3<float, float>((float *)inout_shared, r36, r12, r50);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeResJacFirst(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    float *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    float *pixel, unsigned int pixel_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *const out_rTr, float *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, float *out_calib_jac,
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
  PinholeResJacFirstKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, calib, calib_num_alloc, calib_indices, point,
      point_num_alloc, point_indices, pixel, pixel_num_alloc, out_res,
      out_res_num_alloc, out_rTr, out_pose_jac, out_pose_jac_num_alloc,
      out_pose_njtr, out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, out_calib_jac, out_calib_jac_num_alloc,
      out_calib_njtr, out_calib_njtr_num_alloc, out_calib_precond_diag,
      out_calib_precond_diag_num_alloc, out_calib_precond_tril,
      out_calib_precond_tril_num_alloc, out_point_jac, out_point_jac_num_alloc,
      out_point_njtr, out_point_njtr_num_alloc, out_point_precond_diag,
      out_point_precond_diag_num_alloc, out_point_precond_tril,
      out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar