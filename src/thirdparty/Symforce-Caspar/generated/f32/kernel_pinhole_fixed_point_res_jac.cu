#include "kernel_pinhole_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedPointResJacKernel(float* pose,
                                  unsigned int pose_num_alloc,
                                  SharedIndex* pose_indices,
                                  float* sensor_from_rig,
                                  unsigned int sensor_from_rig_num_alloc,
                                  float* calib,
                                  unsigned int calib_num_alloc,
                                  SharedIndex* calib_indices,
                                  float* pixel,
                                  unsigned int pixel_num_alloc,
                                  float* point,
                                  unsigned int point_num_alloc,
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
                                  float* out_calib_jac,
                                  unsigned int out_calib_jac_num_alloc,
                                  float* const out_calib_njtr,
                                  unsigned int out_calib_njtr_num_alloc,
                                  float* const out_calib_precond_diag,
                                  unsigned int out_calib_precond_diag_num_alloc,
                                  float* const out_calib_precond_tril,
                                  unsigned int out_calib_precond_tril_num_alloc,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74;
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
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r6, r2);
    r2 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r7,
                                         r8,
                                         r9);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r10, r11, r12);
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r13,
                       r14,
                       r15,
                       r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r17,
                                         r18,
                                         r19,
                                         r20);
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
    r33 = fmaf(r10, r33, r9);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r9,
                       r34,
                       r35);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r36 = r17 * r19;
    r36 = r36 * r22;
    r37 = r18 * r20;
    r37 = fmaf(r26, r37, r36);
    r38 = r17 * r17;
    r38 = r38 * r26;
    r39 = 1.00000000000000000e+00;
    r40 = r18 * r18;
    r40 = fmaf(r26, r40, r39);
    r41 = r38 + r40;
    r42 = r18 * r19;
    r42 = r42 * r22;
    r43 = r17 * r20;
    r43 = fmaf(r22, r43, r42);
    r44 = r22 * r21;
    r44 = r44 * r30;
    r45 = fmaf(r27, r24, r44);
    r46 = r26 * r30;
    r46 = r46 * r30;
    r47 = r39 + r46;
    r48 = r23 * r23;
    r48 = r48 * r26;
    r47 = r47 + r48;
    r33 = fmaf(r9, r37, r33);
    r33 = fmaf(r35, r41, r33);
    r33 = fmaf(r34, r43, r33);
    r33 = fmaf(r11, r45, r33);
    r33 = fmaf(r12, r47, r33);
    r47 = copysign(1.0, r33);
    r47 = fmaf(r2, r47, r33);
    r2 = 1.0 / r47;
    r46 = r39 + r46;
    r33 = r21 * r21;
    r33 = r33 * r26;
    r46 = r46 + r33;
    r46 = fmaf(r10, r46, r7);
    r7 = r30 * r24;
    r45 = fmaf(r21, r28, r7);
    r49 = r22 * r30;
    r49 = fmaf(r27, r49, r25);
    r25 = r18 * r20;
    r25 = fmaf(r22, r25, r36);
    r36 = r19 * r20;
    r50 = r17 * r18;
    r50 = r50 * r22;
    r36 = fmaf(r26, r36, r50);
    r51 = r19 * r19;
    r51 = r51 * r26;
    r40 = r51 + r40;
    r46 = fmaf(r11, r45, r46);
    r46 = fmaf(r12, r49, r46);
    r46 = fmaf(r35, r25, r46);
    r46 = fmaf(r34, r36, r46);
    r46 = fmaf(r9, r40, r46);
    r49 = r0 * r46;
    r4 = fmaf(r2, r49, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r22 * r21;
    r3 = fmaf(r27, r3, r7);
    r3 = fmaf(r10, r3, r8);
    r8 = r19 * r20;
    r8 = fmaf(r22, r8, r50);
    r51 = r39 + r51;
    r51 = r51 + r38;
    r38 = r17 * r20;
    r38 = fmaf(r26, r38, r42);
    r44 = fmaf(r23, r28, r44);
    r33 = r39 + r33;
    r33 = r33 + r48;
    r3 = fmaf(r9, r8, r3);
    r3 = fmaf(r34, r51, r3);
    r3 = fmaf(r35, r38, r3);
    r3 = fmaf(r12, r44, r3);
    r3 = fmaf(r11, r33, r3);
    r33 = r1 * r3;
    r5 = fmaf(r2, r33, r5);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r44 = r22 * r30;
    r35 = -5.00000000000000000e-01;
    r34 = r14 * r35;
    r9 = 5.00000000000000000e-01;
    r48 = fmaf(r9, r32, r20 * r34);
    r48 = fmaf(r35, r29, r48);
    r48 = fmaf(r35, r31, r48);
    r44 = r44 * r48;
    r42 = r22 * r21;
    r50 = r14 * r17;
    r7 = r13 * r18;
    r7 = fmaf(r35, r7, r9 * r50);
    r50 = r16 * r19;
    r7 = fmaf(r9, r50, r7);
    r45 = r20 * r9;
    r7 = fmaf(r15, r45, r7);
    r42 = fmaf(r7, r42, r44);
    r50 = r22 * r27;
    r52 = r13 * r17;
    r53 = r15 * r19;
    r53 = fmaf(r35, r53, r35 * r52);
    r53 = fmaf(r16, r45, r53);
    r53 = fmaf(r18, r34, r53);
    r50 = r50 * r53;
    r52 = r13 * r20;
    r54 = r16 * r17;
    r54 = fmaf(r35, r54, r35 * r52);
    r52 = r15 * r18;
    r54 = fmaf(r35, r52, r54);
    r55 = r14 * r19;
    r54 = fmaf(r9, r55, r54);
    r55 = r54 * r24;
    r52 = r50 + r55;
    r56 = r42 + r52;
    r57 = r30 * r54;
    r58 = fmaf(r7, r28, r26 * r57);
    r59 = r22 * r21;
    r59 = r59 * r53;
    r60 = fmaf(r48, r24, r59);
    r58 = r58 + r60;
    r58 = fmaf(r10, r58, r11 * r56);
    r56 = r30 * r7;
    r61 = -4.00000000000000000e+00;
    r56 = r56 * r61;
    r62 = r53 * r61;
    r63 = r23 * r62;
    r64 = r56 + r63;
    r58 = fmaf(r12, r64, r58);
    r47 = r47 * r47;
    r47 = 1.0 / r47;
    r64 = r6 * r47;
    r49 = r64 * r49;
    r65 = r22 * r27;
    r65 = fmaf(r22, r57, r7 * r65);
    r65 = r65 + r60;
    r66 = r22 * r30;
    r66 = r66 * r53;
    r67 = r21 * r26;
    r67 = fmaf(r54, r67, r66);
    r7 = r7 * r24;
    r67 = r67 + r7;
    r67 = fmaf(r48, r28, r67);
    r67 = fmaf(r11, r67, r12 * r65);
    r65 = r21 * r48;
    r68 = r61 * r65;
    r56 = r56 + r68;
    r67 = fmaf(r10, r56, r67);
    r56 = r0 * r67;
    r56 = fmaf(r2, r56, r58 * r49);
    r69 = r58 * r64;
    r70 = r23 * r26;
    r71 = r53 * r28;
    r70 = fmaf(r54, r70, r71);
    r70 = r70 + r42;
    r68 = r63 + r68;
    r68 = fmaf(r11, r68, r12 * r70);
    r70 = r22 * r27;
    r70 = fmaf(r48, r70, r7);
    r7 = r22 * r21;
    r7 = fmaf(r54, r7, r66);
    r70 = r70 + r7;
    r68 = fmaf(r10, r70, r68);
    r70 = r1 * r68;
    r70 = fmaf(r2, r70, r33 * r69);
    r69 = r26 * r30;
    r69 = fmaf(r48, r69, r71);
    r66 = r22 * r21;
    r63 = r15 * r20;
    r42 = r13 * r18;
    r42 = fmaf(r9, r42, r35 * r63);
    r63 = r16 * r19;
    r42 = fmaf(r35, r63, r42);
    r42 = fmaf(r17, r34, r42);
    r66 = r66 * r42;
    r63 = r16 * r17;
    r72 = r15 * r18;
    r72 = fmaf(r9, r72, r9 * r63);
    r72 = fmaf(r13, r45, r72);
    r72 = fmaf(r19, r34, r72);
    r34 = fmaf(r72, r24, r66);
    r69 = r69 + r34;
    r63 = r22 * r30;
    r63 = r63 * r72;
    r73 = r22 * r27;
    r73 = fmaf(r42, r73, r63);
    r73 = r73 + r60;
    r73 = fmaf(r11, r73, r10 * r69);
    r69 = r23 * r61;
    r69 = r69 * r42;
    r60 = r30 * r62;
    r74 = r69 + r60;
    r73 = fmaf(r12, r74, r73);
    r50 = r44 + r50;
    r50 = r50 + r34;
    r34 = r21 * r61;
    r34 = r34 * r72;
    r60 = r34 + r60;
    r60 = fmaf(r10, r60, r12 * r50);
    r50 = fmaf(r72, r28, r26 * r65);
    r44 = r22 * r30;
    r53 = r53 * r24;
    r44 = fmaf(r42, r44, r53);
    r50 = r50 + r44;
    r60 = fmaf(r11, r50, r60);
    r50 = r0 * r60;
    r50 = fmaf(r2, r50, r73 * r49);
    r74 = r73 * r64;
    r63 = r59 + r63;
    r59 = r23 * r26;
    r63 = fmaf(r48, r59, r63);
    r63 = fmaf(r42, r28, r63);
    r59 = r22 * r27;
    r65 = fmaf(r22, r65, r72 * r59);
    r65 = r65 + r44;
    r65 = fmaf(r10, r65, r12 * r63);
    r34 = r69 + r34;
    r65 = fmaf(r11, r34, r65);
    r34 = r1 * r65;
    r34 = fmaf(r2, r34, r33 * r74);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r56,
                                          r70,
                                          r50,
                                          r34);
    r74 = r23 * r61;
    r32 = fmaf(r35, r32, r14 * r45);
    r32 = fmaf(r9, r29, r32);
    r32 = fmaf(r9, r31, r32);
    r74 = r74 * r32;
    r57 = r61 * r57;
    r61 = r74 + r57;
    r31 = r22 * r21;
    r31 = r31 * r32;
    r9 = r26 * r30;
    r9 = fmaf(r42, r9, r31);
    r9 = r9 + r53;
    r9 = fmaf(r54, r28, r9);
    r9 = fmaf(r10, r9, r12 * r61);
    r61 = r22 * r27;
    r24 = fmaf(r42, r24, r32 * r61);
    r24 = r24 + r7;
    r9 = fmaf(r11, r24, r9);
    r24 = r22 * r30;
    r24 = r24 * r32;
    r61 = r21 * r26;
    r61 = fmaf(r42, r61, r24);
    r61 = r61 + r55;
    r61 = r61 + r71;
    r62 = r21 * r62;
    r57 = r57 + r62;
    r57 = fmaf(r10, r57, r11 * r61);
    r61 = r22 * r27;
    r61 = fmaf(r54, r61, r31);
    r61 = r61 + r44;
    r57 = fmaf(r12, r61, r57);
    r61 = r0 * r57;
    r61 = fmaf(r2, r61, r9 * r49);
    r24 = r66 + r24;
    r24 = r24 + r52;
    r52 = r23 * r26;
    r28 = fmaf(r32, r28, r42 * r52);
    r28 = r28 + r7;
    r28 = fmaf(r12, r28, r10 * r24);
    r62 = r74 + r62;
    r28 = fmaf(r11, r62, r28);
    r62 = r1 * r28;
    r11 = r9 * r64;
    r11 = fmaf(r33, r11, r2 * r62);
    r62 = r0 * r40;
    r62 = fmaf(r37, r49, r2 * r62);
    r74 = r1 * r8;
    r12 = r37 * r64;
    r12 = fmaf(r33, r12, r2 * r74);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r61,
                                          r11,
                                          r62,
                                          r12);
    r74 = r0 * r36;
    r74 = fmaf(r43, r49, r2 * r74);
    r24 = r1 * r51;
    r10 = r43 * r64;
    r10 = fmaf(r33, r10, r2 * r24);
    r24 = r0 * r25;
    r49 = fmaf(r41, r49, r2 * r24);
    r24 = r41 * r64;
    r7 = r1 * r38;
    r7 = fmaf(r2, r7, r33 * r24);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r74,
                                          r10,
                                          r49,
                                          r7);
    r24 = r6 * r4;
    r33 = r6 * r5;
    r33 = fmaf(r70, r33, r56 * r24);
    r24 = r6 * r4;
    r32 = r6 * r5;
    r32 = fmaf(r34, r32, r50 * r24);
    r24 = r6 * r5;
    r52 = r6 * r4;
    r52 = fmaf(r61, r52, r11 * r24);
    r24 = r6 * r4;
    r42 = r6 * r5;
    r42 = fmaf(r12, r42, r62 * r24);
    WriteSum4<float, float>((float*)inout_shared, r33, r32, r52, r42);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r6 * r5;
    r52 = r6 * r4;
    r52 = fmaf(r74, r52, r10 * r42);
    r42 = r6 * r5;
    r32 = r6 * r4;
    r32 = fmaf(r49, r32, r7 * r42);
    WriteSum2<float, float>((float*)inout_shared, r52, r32);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fmaf(r70, r70, r56 * r56);
    r52 = fmaf(r50, r50, r34 * r34);
    r42 = fmaf(r11, r11, r61 * r61);
    r33 = fmaf(r12, r12, r62 * r62);
    WriteSum4<float, float>((float*)inout_shared, r32, r52, r42, r33);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fmaf(r74, r74, r10 * r10);
    r42 = fmaf(r7, r7, r49 * r49);
    WriteSum2<float, float>((float*)inout_shared, r33, r42);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fmaf(r56, r50, r70 * r34);
    r33 = fmaf(r70, r11, r56 * r61);
    r52 = fmaf(r56, r62, r70 * r12);
    r32 = fmaf(r56, r74, r70 * r10);
    WriteSum4<float, float>((float*)inout_shared, r42, r33, r52, r32);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = fmaf(r56, r49, r70 * r7);
    r70 = fmaf(r50, r61, r34 * r11);
    r32 = fmaf(r50, r62, r34 * r12);
    r52 = fmaf(r34, r10, r50 * r74);
    WriteSum4<float, float>((float*)inout_shared, r56, r70, r32, r52);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fmaf(r34, r7, r50 * r49);
    r50 = fmaf(r11, r12, r61 * r62);
    r52 = fmaf(r11, r10, r61 * r74);
    r61 = fmaf(r61, r49, r11 * r7);
    WriteSum4<float, float>((float*)inout_shared, r34, r50, r52, r61);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fmaf(r12, r10, r62 * r74);
    r62 = fmaf(r62, r49, r12 * r7);
    r7 = fmaf(r10, r7, r74 * r49);
    WriteSum3<float, float>((float*)inout_shared, r61, r62, r7);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r46 * r2;
    r62 = r3 * r2;
    WriteIdx2<1024, float, float, float2>(
        out_calib_jac, 0 * out_calib_jac_num_alloc, global_thread_idx, r7, r62);
    r61 = r6 * r4;
    r10 = r6 * r5;
    r49 = r6 * r46;
    r49 = r49 * r4;
    r49 = r49 * r2;
    r74 = r6 * r3;
    r74 = r74 * r5;
    r74 = r74 * r2;
    WriteSum4<float, float>((float*)inout_shared, r49, r74, r61, r10);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r46 * r46;
    r46 = r46 * r47;
    r3 = r3 * r3;
    r3 = r3 * r47;
    WriteSum4<float, float>((float*)inout_shared, r46, r3, r39, r39);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = 0.00000000000000000e+00;
    WriteSum4<float, float>((float*)inout_shared, r39, r7, r39, r39);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float*)inout_shared, r62, r39);
  };
  FlushSumShared<2, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
}

void PinholeFixedPointResJac(float* pose,
                             unsigned int pose_num_alloc,
                             SharedIndex* pose_indices,
                             float* sensor_from_rig,
                             unsigned int sensor_from_rig_num_alloc,
                             float* calib,
                             unsigned int calib_num_alloc,
                             SharedIndex* calib_indices,
                             float* pixel,
                             unsigned int pixel_num_alloc,
                             float* point,
                             unsigned int point_num_alloc,
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
                             float* out_calib_jac,
                             unsigned int out_calib_jac_num_alloc,
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
  PinholeFixedPointResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      point,
      point_num_alloc,
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
      out_calib_jac,
      out_calib_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar