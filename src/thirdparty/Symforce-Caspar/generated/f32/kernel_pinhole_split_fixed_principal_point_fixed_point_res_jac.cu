#include "kernel_pinhole_split_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPrincipalPointFixedPointResJacKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
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
        float* out_focal_jac,
        unsigned int out_focal_jac_num_alloc,
        float* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        float* const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        float* const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73;

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
    r0 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r5,
                                         r6,
                                         r7);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9, r10);
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r11,
                       r12,
                       r13,
                       r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r15,
                                         r16,
                                         r17,
                                         r18);
    r19 = fmaf(r12, r15, r13 * r18);
    r20 = r11 * r16;
    r19 = fmaf(r4, r20, r19);
    r19 = fmaf(r14, r17, r19);
    r20 = 2.00000000000000000e+00;
    r21 = fmaf(r14, r15, r11 * r18);
    r22 = r12 * r17;
    r21 = fmaf(r4, r22, r21);
    r21 = fmaf(r13, r16, r21);
    r22 = r20 * r21;
    r23 = r19 * r22;
    r24 = -2.00000000000000000e+00;
    r25 = fmaf(r12, r16, r11 * r15);
    r25 = fmaf(r13, r17, r25);
    r25 = fmaf(r4, r25, r14 * r18);
    r26 = r24 * r25;
    r27 = fmaf(r14, r16, r12 * r18);
    r28 = r11 * r17;
    r29 = r13 * r15;
    r27 = r27 + r28;
    r27 = fmaf(r4, r29, r27);
    r30 = fmaf(r27, r26, r23);
    r30 = fmaf(r8, r30, r7);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r7,
                       r31,
                       r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r33 = r15 * r17;
    r33 = r33 * r20;
    r34 = r16 * r18;
    r35 = fmaf(r24, r34, r33);
    r36 = r15 * r15;
    r36 = r36 * r24;
    r37 = 1.00000000000000000e+00;
    r38 = r16 * r16;
    r38 = fmaf(r24, r38, r37);
    r39 = r36 + r38;
    r40 = r16 * r17;
    r40 = r40 * r20;
    r41 = r15 * r18;
    r41 = fmaf(r20, r41, r40);
    r42 = r20 * r19;
    r42 = r42 * r27;
    r43 = fmaf(r25, r22, r42);
    r44 = r24 * r27;
    r44 = r44 * r27;
    r45 = r37 + r44;
    r46 = r21 * r21;
    r46 = r46 * r24;
    r45 = r45 + r46;
    r30 = fmaf(r7, r35, r30);
    r30 = fmaf(r32, r39, r30);
    r30 = fmaf(r31, r41, r30);
    r30 = fmaf(r9, r43, r30);
    r30 = fmaf(r10, r45, r30);
    r45 = copysign(1.0, r30);
    r45 = fmaf(r0, r45, r30);
    r0 = 1.0 / r45;
  };
  LoadShared<2, float, float>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>(
        (float*)inout_shared, focal_indices_loc[threadIdx.x].target, r30, r43);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r44 = r37 + r44;
    r47 = r19 * r19;
    r47 = r47 * r24;
    r44 = r44 + r47;
    r44 = fmaf(r8, r44, r5);
    r5 = r27 * r22;
    r48 = fmaf(r19, r26, r5);
    r49 = r20 * r27;
    r49 = fmaf(r25, r49, r23);
    r34 = fmaf(r20, r34, r33);
    r33 = r17 * r18;
    r23 = r15 * r16;
    r23 = r23 * r20;
    r33 = fmaf(r24, r33, r23);
    r50 = r17 * r17;
    r50 = r50 * r24;
    r38 = r50 + r38;
    r44 = fmaf(r9, r48, r44);
    r44 = fmaf(r10, r49, r44);
    r44 = fmaf(r32, r34, r44);
    r44 = fmaf(r31, r33, r44);
    r44 = fmaf(r7, r38, r44);
    r49 = r30 * r44;
    r2 = fmaf(r0, r49, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r20 * r19;
    r1 = fmaf(r25, r1, r5);
    r1 = fmaf(r8, r1, r6);
    r6 = r17 * r18;
    r6 = fmaf(r20, r6, r23);
    r50 = r37 + r50;
    r50 = r50 + r36;
    r36 = r15 * r18;
    r36 = fmaf(r24, r36, r40);
    r42 = fmaf(r21, r26, r42);
    r47 = r37 + r47;
    r47 = r47 + r46;
    r1 = fmaf(r7, r6, r1);
    r1 = fmaf(r31, r50, r1);
    r1 = fmaf(r32, r36, r1);
    r1 = fmaf(r10, r42, r1);
    r1 = fmaf(r9, r47, r1);
    r47 = r43 * r1;
    r3 = fmaf(r0, r47, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r42 = r20 * r25;
    r32 = r13 * r18;
    r31 = 5.00000000000000000e-01;
    r7 = r12 * r15;
    r7 = fmaf(r31, r7, r31 * r32);
    r32 = r11 * r16;
    r46 = -5.00000000000000000e-01;
    r7 = fmaf(r46, r32, r7);
    r37 = r14 * r31;
    r7 = fmaf(r17, r37, r7);
    r32 = r11 * r18;
    r40 = r14 * r15;
    r40 = fmaf(r46, r40, r46 * r32);
    r32 = r13 * r16;
    r40 = fmaf(r46, r32, r40);
    r23 = r12 * r17;
    r40 = fmaf(r31, r23, r40);
    r23 = r27 * r40;
    r42 = fmaf(r20, r23, r7 * r42);
    r32 = r20 * r19;
    r5 = r11 * r15;
    r48 = r13 * r17;
    r48 = fmaf(r46, r48, r46 * r5);
    r5 = r12 * r46;
    r48 = fmaf(r18, r37, r48);
    r48 = fmaf(r16, r5, r48);
    r32 = r32 * r48;
    r51 = r14 * r16;
    r51 = fmaf(r18, r5, r46 * r51);
    r51 = fmaf(r31, r29, r51);
    r51 = fmaf(r46, r28, r51);
    r52 = fmaf(r51, r22, r32);
    r42 = r42 + r52;
    r53 = r20 * r27;
    r53 = r53 * r48;
    r54 = r19 * r24;
    r54 = fmaf(r40, r54, r53);
    r55 = r7 * r22;
    r54 = r54 + r55;
    r54 = fmaf(r51, r26, r54);
    r54 = fmaf(r9, r54, r10 * r42);
    r42 = r27 * r7;
    r56 = -4.00000000000000000e+00;
    r42 = r42 * r56;
    r57 = r19 * r51;
    r58 = r56 * r57;
    r59 = r42 + r58;
    r54 = fmaf(r8, r59, r54);
    r59 = r30 * r54;
    r60 = r20 * r27;
    r60 = r60 * r51;
    r61 = r20 * r19;
    r61 = fmaf(r7, r61, r60);
    r62 = r20 * r25;
    r62 = r62 * r48;
    r63 = r40 * r22;
    r64 = r62 + r63;
    r65 = r61 + r64;
    r7 = fmaf(r7, r26, r24 * r23);
    r7 = r7 + r52;
    r7 = fmaf(r8, r7, r9 * r65);
    r65 = r48 * r56;
    r66 = r21 * r65;
    r42 = r42 + r66;
    r7 = fmaf(r10, r42, r7);
    r45 = r45 * r45;
    r45 = 1.0 / r45;
    r42 = r4 * r45;
    r49 = r42 * r49;
    r59 = fmaf(r7, r49, r0 * r59);
    r67 = r21 * r24;
    r68 = r48 * r26;
    r67 = fmaf(r40, r67, r68);
    r67 = r67 + r61;
    r66 = r58 + r66;
    r66 = fmaf(r9, r66, r10 * r67);
    r67 = r20 * r25;
    r67 = fmaf(r51, r67, r55);
    r55 = r20 * r19;
    r55 = fmaf(r40, r55, r53);
    r67 = r67 + r55;
    r66 = fmaf(r8, r67, r66);
    r67 = r43 * r66;
    r53 = r7 * r42;
    r53 = fmaf(r47, r53, r0 * r67);
    r62 = r60 + r62;
    r60 = r20 * r19;
    r67 = r13 * r18;
    r58 = r11 * r16;
    r58 = fmaf(r31, r58, r46 * r67);
    r67 = r14 * r17;
    r58 = fmaf(r46, r67, r58);
    r58 = fmaf(r15, r5, r58);
    r60 = r60 * r58;
    r67 = r11 * r18;
    r61 = r13 * r16;
    r61 = fmaf(r31, r61, r31 * r67);
    r61 = fmaf(r15, r37, r61);
    r61 = fmaf(r17, r5, r61);
    r5 = fmaf(r61, r22, r60);
    r62 = r62 + r5;
    r67 = r19 * r56;
    r67 = r67 * r61;
    r69 = r27 * r65;
    r70 = r67 + r69;
    r70 = fmaf(r8, r70, r10 * r62);
    r62 = fmaf(r61, r26, r24 * r57);
    r71 = r20 * r27;
    r48 = r48 * r22;
    r71 = fmaf(r58, r71, r48);
    r62 = r62 + r71;
    r70 = fmaf(r9, r62, r70);
    r62 = r30 * r70;
    r72 = r24 * r27;
    r72 = fmaf(r51, r72, r68);
    r72 = r72 + r5;
    r5 = r20 * r27;
    r5 = r5 * r61;
    r73 = r20 * r25;
    r73 = fmaf(r58, r73, r5);
    r73 = r73 + r52;
    r73 = fmaf(r9, r73, r8 * r72);
    r72 = r21 * r56;
    r72 = r72 * r58;
    r69 = r72 + r69;
    r73 = fmaf(r10, r69, r73);
    r62 = fmaf(r73, r49, r0 * r62);
    r69 = r73 * r42;
    r5 = r32 + r5;
    r32 = r21 * r24;
    r5 = fmaf(r51, r32, r5);
    r5 = fmaf(r58, r26, r5);
    r32 = r20 * r25;
    r57 = fmaf(r20, r57, r61 * r32);
    r57 = r57 + r71;
    r57 = fmaf(r8, r57, r10 * r5);
    r72 = r67 + r72;
    r57 = fmaf(r9, r72, r57);
    r72 = r43 * r57;
    r72 = fmaf(r0, r72, r47 * r69);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r59,
                                          r53,
                                          r62,
                                          r72);
    r69 = r20 * r27;
    r67 = r12 * r18;
    r29 = fmaf(r46, r29, r31 * r67);
    r29 = fmaf(r16, r37, r29);
    r29 = fmaf(r31, r28, r29);
    r69 = r69 * r29;
    r28 = r19 * r24;
    r28 = fmaf(r58, r28, r69);
    r28 = r28 + r63;
    r28 = r28 + r68;
    r65 = r19 * r65;
    r23 = r56 * r23;
    r68 = r65 + r23;
    r68 = fmaf(r8, r68, r9 * r28);
    r28 = r20 * r19;
    r28 = r28 * r29;
    r63 = r20 * r25;
    r63 = fmaf(r40, r63, r28);
    r63 = r63 + r71;
    r68 = fmaf(r10, r63, r68);
    r63 = r30 * r68;
    r56 = r21 * r56;
    r56 = r56 * r29;
    r23 = r56 + r23;
    r71 = r24 * r27;
    r71 = fmaf(r58, r71, r28);
    r71 = r71 + r48;
    r71 = fmaf(r40, r26, r71);
    r71 = fmaf(r8, r71, r10 * r23);
    r23 = r20 * r25;
    r22 = fmaf(r58, r22, r29 * r23);
    r22 = r22 + r55;
    r71 = fmaf(r9, r22, r71);
    r63 = fmaf(r71, r49, r0 * r63);
    r22 = r71 * r42;
    r69 = r60 + r69;
    r69 = r69 + r64;
    r64 = r21 * r24;
    r26 = fmaf(r29, r26, r58 * r64);
    r26 = r26 + r55;
    r26 = fmaf(r10, r26, r8 * r69);
    r65 = r56 + r65;
    r26 = fmaf(r9, r65, r26);
    r65 = r43 * r26;
    r65 = fmaf(r0, r65, r47 * r22);
    r22 = r30 * r38;
    r22 = fmaf(r35, r49, r0 * r22);
    r9 = r43 * r6;
    r56 = r35 * r42;
    r56 = fmaf(r47, r56, r0 * r9);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r63,
                                          r65,
                                          r22,
                                          r56);
    r9 = r30 * r33;
    r9 = fmaf(r0, r9, r41 * r49);
    r10 = r41 * r42;
    r69 = r43 * r50;
    r69 = fmaf(r0, r69, r47 * r10);
    r10 = r30 * r34;
    r49 = fmaf(r39, r49, r0 * r10);
    r10 = r43 * r36;
    r8 = r39 * r42;
    r8 = fmaf(r47, r8, r0 * r10);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r9,
                                          r69,
                                          r49,
                                          r8);
    r10 = r4 * r2;
    r47 = r4 * r3;
    r47 = fmaf(r53, r47, r59 * r10);
    r10 = r4 * r3;
    r55 = r4 * r2;
    r55 = fmaf(r62, r55, r72 * r10);
    r10 = r4 * r3;
    r29 = r4 * r2;
    r29 = fmaf(r63, r29, r65 * r10);
    r10 = r4 * r3;
    r64 = r4 * r2;
    r64 = fmaf(r22, r64, r56 * r10);
    WriteSum4<float, float>((float*)inout_shared, r47, r55, r29, r64);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = r4 * r3;
    r29 = r4 * r2;
    r29 = fmaf(r9, r29, r69 * r64);
    r64 = r4 * r3;
    r55 = r4 * r2;
    r55 = fmaf(r49, r55, r8 * r64);
    WriteSum2<float, float>((float*)inout_shared, r29, r55);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fmaf(r53, r53, r59 * r59);
    r29 = fmaf(r72, r72, r62 * r62);
    r64 = fmaf(r63, r63, r65 * r65);
    r47 = fmaf(r22, r22, r56 * r56);
    WriteSum4<float, float>((float*)inout_shared, r55, r29, r64, r47);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fmaf(r9, r9, r69 * r69);
    r64 = fmaf(r49, r49, r8 * r8);
    WriteSum2<float, float>((float*)inout_shared, r47, r64);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = fmaf(r59, r62, r53 * r72);
    r47 = fmaf(r53, r65, r59 * r63);
    r29 = fmaf(r59, r22, r53 * r56);
    r55 = fmaf(r53, r69, r59 * r9);
    WriteSum4<float, float>((float*)inout_shared, r64, r47, r29, r55);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = fmaf(r53, r8, r59 * r49);
    r59 = fmaf(r62, r63, r72 * r65);
    r55 = fmaf(r72, r56, r62 * r22);
    r29 = fmaf(r72, r69, r62 * r9);
    WriteSum4<float, float>((float*)inout_shared, r53, r59, r55, r29);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = fmaf(r62, r49, r72 * r8);
    r72 = fmaf(r63, r22, r65 * r56);
    r29 = fmaf(r65, r69, r63 * r9);
    r63 = fmaf(r63, r49, r65 * r8);
    WriteSum4<float, float>((float*)inout_shared, r62, r72, r29, r63);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fmaf(r56, r69, r22 * r9);
    r22 = fmaf(r22, r49, r56 * r8);
    r49 = fmaf(r9, r49, r69 * r8);
    WriteSum3<float, float>((float*)inout_shared, r63, r22, r49);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r44 * r0;
    r22 = r1 * r0;
    WriteIdx2<1024, float, float, float2>(out_focal_jac,
                                          0 * out_focal_jac_num_alloc,
                                          global_thread_idx,
                                          r49,
                                          r22);
    r22 = r4 * r44;
    r22 = r22 * r2;
    r22 = r22 * r0;
    r49 = r4 * r1;
    r49 = r49 * r3;
    r49 = r49 * r0;
    WriteSum2<float, float>((float*)inout_shared, r22, r49);
  };
  FlushSumShared<2, float>(out_focal_njtr,
                           0 * out_focal_njtr_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = r44 * r44;
    r44 = r44 * r45;
    r1 = r1 * r1;
    r1 = r1 * r45;
    WriteSum2<float, float>((float*)inout_shared, r44, r1);
  };
  FlushSumShared<2, float>(out_focal_precond_diag,
                           0 * out_focal_precond_diag_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
}

void PinholeSplitFixedPrincipalPointFixedPointResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
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
    float* out_focal_jac,
    unsigned int out_focal_jac_num_alloc,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    float* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    float* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPrincipalPointFixedPointResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      focal,
      focal_num_alloc,
      focal_indices,
      pixel,
      pixel_num_alloc,
      principal_point,
      principal_point_num_alloc,
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
      out_focal_jac,
      out_focal_jac_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_focal_precond_diag,
      out_focal_precond_diag_num_alloc,
      out_focal_precond_tril,
      out_focal_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar