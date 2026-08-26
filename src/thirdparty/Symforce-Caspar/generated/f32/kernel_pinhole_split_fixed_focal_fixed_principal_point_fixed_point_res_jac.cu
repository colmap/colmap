#include "kernel_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedFocalFixedPrincipalPointFixedPointResJacKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
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
    r27 = r14 * r16;
    r28 = fmaf(r12, r18, r27);
    r29 = r11 * r17;
    r30 = r13 * r15;
    r28 = r28 + r29;
    r28 = fmaf(r4, r30, r28);
    r31 = fmaf(r28, r26, r23);
    r31 = fmaf(r8, r31, r7);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r7,
                       r32,
                       r33);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r34 = r15 * r17;
    r34 = r34 * r20;
    r35 = r16 * r18;
    r35 = fmaf(r24, r35, r34);
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
    r42 = r42 * r28;
    r43 = fmaf(r25, r22, r42);
    r44 = r24 * r28;
    r44 = r44 * r28;
    r45 = r37 + r44;
    r46 = r21 * r21;
    r46 = r46 * r24;
    r45 = r45 + r46;
    r31 = fmaf(r7, r35, r31);
    r31 = fmaf(r33, r39, r31);
    r31 = fmaf(r32, r41, r31);
    r31 = fmaf(r9, r43, r31);
    r31 = fmaf(r10, r45, r31);
    r45 = copysign(1.0, r31);
    r45 = fmaf(r0, r45, r31);
    r0 = 1.0 / r45;
    ReadIdx2<1024, float, float, float2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r31, r43);
    r44 = r37 + r44;
    r47 = r19 * r19;
    r47 = r47 * r24;
    r44 = r44 + r47;
    r44 = fmaf(r8, r44, r5);
    r5 = r28 * r22;
    r48 = fmaf(r19, r26, r5);
    r49 = r20 * r28;
    r49 = fmaf(r25, r49, r23);
    r23 = r16 * r18;
    r23 = fmaf(r20, r23, r34);
    r34 = r17 * r18;
    r50 = r15 * r16;
    r50 = r50 * r20;
    r34 = fmaf(r24, r34, r50);
    r51 = r17 * r17;
    r51 = r51 * r24;
    r38 = r51 + r38;
    r44 = fmaf(r9, r48, r44);
    r44 = fmaf(r10, r49, r44);
    r44 = fmaf(r33, r23, r44);
    r44 = fmaf(r32, r34, r44);
    r44 = fmaf(r7, r38, r44);
    r44 = r31 * r44;
    r2 = fmaf(r0, r44, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r20 * r19;
    r1 = fmaf(r25, r1, r5);
    r1 = fmaf(r8, r1, r6);
    r6 = r17 * r18;
    r6 = fmaf(r20, r6, r50);
    r51 = r37 + r51;
    r51 = r51 + r36;
    r36 = r15 * r18;
    r36 = fmaf(r24, r36, r40);
    r42 = fmaf(r21, r26, r42);
    r47 = r37 + r47;
    r47 = r47 + r46;
    r1 = fmaf(r7, r6, r1);
    r1 = fmaf(r32, r51, r1);
    r1 = fmaf(r33, r36, r1);
    r1 = fmaf(r10, r42, r1);
    r1 = fmaf(r9, r47, r1);
    r1 = r43 * r1;
    r3 = fmaf(r0, r1, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r47 = r4 * r3;
    r42 = r20 * r28;
    r33 = -5.00000000000000000e-01;
    r32 = r12 * r33;
    r7 = 5.00000000000000000e-01;
    r46 = fmaf(r7, r30, r18 * r32);
    r46 = fmaf(r33, r27, r46);
    r46 = fmaf(r33, r29, r46);
    r42 = r42 * r46;
    r37 = r20 * r19;
    r40 = r12 * r15;
    r50 = r11 * r16;
    r50 = fmaf(r33, r50, r7 * r40);
    r40 = r14 * r17;
    r50 = fmaf(r7, r40, r50);
    r5 = r18 * r7;
    r50 = fmaf(r13, r5, r50);
    r37 = fmaf(r50, r37, r42);
    r40 = r20 * r25;
    r49 = r11 * r15;
    r48 = r13 * r17;
    r48 = fmaf(r33, r48, r33 * r49);
    r48 = fmaf(r14, r5, r48);
    r48 = fmaf(r16, r32, r48);
    r40 = r40 * r48;
    r49 = r11 * r18;
    r52 = r14 * r15;
    r52 = fmaf(r33, r52, r33 * r49);
    r49 = r13 * r16;
    r52 = fmaf(r33, r49, r52);
    r53 = r12 * r17;
    r52 = fmaf(r7, r53, r52);
    r53 = r52 * r22;
    r49 = r40 + r53;
    r54 = r37 + r49;
    r55 = r28 * r52;
    r56 = fmaf(r50, r26, r24 * r55);
    r57 = r20 * r19;
    r57 = r57 * r48;
    r58 = fmaf(r46, r22, r57);
    r56 = r56 + r58;
    r56 = fmaf(r8, r56, r9 * r54);
    r54 = r28 * r50;
    r59 = -4.00000000000000000e+00;
    r54 = r54 * r59;
    r60 = r48 * r59;
    r61 = r21 * r60;
    r62 = r54 + r61;
    r56 = fmaf(r10, r62, r56);
    r45 = r45 * r45;
    r45 = 1.0 / r45;
    r45 = r4 * r45;
    r62 = r56 * r45;
    r63 = r21 * r24;
    r64 = r48 * r26;
    r63 = fmaf(r52, r63, r64);
    r63 = r63 + r37;
    r37 = r19 * r46;
    r65 = r59 * r37;
    r61 = r61 + r65;
    r61 = fmaf(r9, r61, r10 * r63);
    r63 = r20 * r25;
    r66 = r50 * r22;
    r63 = fmaf(r46, r63, r66);
    r67 = r20 * r28;
    r67 = r67 * r48;
    r68 = r20 * r19;
    r68 = fmaf(r52, r68, r67);
    r63 = r63 + r68;
    r61 = fmaf(r8, r63, r61);
    r63 = r43 * r61;
    r63 = fmaf(r0, r63, r1 * r62);
    r62 = r4 * r2;
    r69 = r20 * r25;
    r69 = fmaf(r20, r55, r50 * r69);
    r69 = r69 + r58;
    r50 = r19 * r24;
    r50 = fmaf(r52, r50, r67);
    r50 = r50 + r66;
    r50 = fmaf(r46, r26, r50);
    r50 = fmaf(r9, r50, r10 * r69);
    r65 = r54 + r65;
    r50 = fmaf(r8, r65, r50);
    r65 = r31 * r50;
    r44 = r45 * r44;
    r65 = fmaf(r56, r44, r0 * r65);
    r62 = fmaf(r65, r62, r63 * r47);
    r47 = r4 * r2;
    r54 = r24 * r28;
    r54 = fmaf(r46, r54, r64);
    r69 = r20 * r19;
    r66 = r13 * r18;
    r67 = r11 * r16;
    r67 = fmaf(r7, r67, r33 * r66);
    r66 = r14 * r17;
    r67 = fmaf(r33, r66, r67);
    r67 = fmaf(r15, r32, r67);
    r69 = r69 * r67;
    r66 = r14 * r15;
    r70 = r13 * r16;
    r70 = fmaf(r7, r70, r7 * r66);
    r70 = fmaf(r11, r5, r70);
    r70 = fmaf(r17, r32, r70);
    r32 = fmaf(r70, r22, r69);
    r54 = r54 + r32;
    r66 = r20 * r28;
    r66 = r66 * r70;
    r71 = r20 * r25;
    r71 = fmaf(r67, r71, r66);
    r71 = r71 + r58;
    r71 = fmaf(r9, r71, r8 * r54);
    r54 = r21 * r59;
    r54 = r54 * r67;
    r58 = r28 * r60;
    r72 = r54 + r58;
    r71 = fmaf(r10, r72, r71);
    r40 = r42 + r40;
    r40 = r40 + r32;
    r32 = r19 * r59;
    r32 = r32 * r70;
    r58 = r32 + r58;
    r58 = fmaf(r8, r58, r10 * r40);
    r40 = fmaf(r70, r26, r24 * r37);
    r42 = r20 * r28;
    r48 = r48 * r22;
    r42 = fmaf(r67, r42, r48);
    r40 = r40 + r42;
    r58 = fmaf(r9, r40, r58);
    r40 = r31 * r58;
    r40 = fmaf(r0, r40, r71 * r44);
    r72 = r4 * r3;
    r73 = r71 * r45;
    r66 = r57 + r66;
    r57 = r21 * r24;
    r66 = fmaf(r46, r57, r66);
    r66 = fmaf(r67, r26, r66);
    r57 = r20 * r25;
    r37 = fmaf(r20, r37, r70 * r57);
    r37 = r37 + r42;
    r37 = fmaf(r8, r37, r10 * r66);
    r32 = r54 + r32;
    r37 = fmaf(r9, r32, r37);
    r32 = r43 * r37;
    r32 = fmaf(r0, r32, r1 * r73);
    r72 = fmaf(r32, r72, r40 * r47);
    r47 = r4 * r2;
    r73 = r21 * r59;
    r30 = fmaf(r33, r30, r12 * r5);
    r30 = fmaf(r7, r27, r30);
    r30 = fmaf(r7, r29, r30);
    r73 = r73 * r30;
    r55 = r59 * r55;
    r59 = r73 + r55;
    r29 = r20 * r19;
    r29 = r29 * r30;
    r7 = r24 * r28;
    r7 = fmaf(r67, r7, r29);
    r7 = r7 + r48;
    r7 = fmaf(r52, r26, r7);
    r7 = fmaf(r8, r7, r10 * r59);
    r59 = r20 * r25;
    r22 = fmaf(r67, r22, r30 * r59);
    r22 = r22 + r68;
    r7 = fmaf(r9, r22, r7);
    r22 = r20 * r28;
    r22 = r22 * r30;
    r59 = r19 * r24;
    r59 = fmaf(r67, r59, r22);
    r59 = r59 + r53;
    r59 = r59 + r64;
    r60 = r19 * r60;
    r55 = r55 + r60;
    r55 = fmaf(r8, r55, r9 * r59);
    r59 = r20 * r25;
    r59 = fmaf(r52, r59, r29);
    r59 = r59 + r42;
    r55 = fmaf(r10, r59, r55);
    r59 = r31 * r55;
    r59 = fmaf(r0, r59, r7 * r44);
    r42 = r4 * r3;
    r22 = r69 + r22;
    r22 = r22 + r49;
    r49 = r21 * r24;
    r26 = fmaf(r30, r26, r67 * r49);
    r26 = r26 + r68;
    r26 = fmaf(r10, r26, r8 * r22);
    r60 = r73 + r60;
    r26 = fmaf(r9, r60, r26);
    r60 = r43 * r26;
    r9 = r7 * r45;
    r9 = fmaf(r1, r9, r0 * r60);
    r42 = fmaf(r9, r42, r59 * r47);
    r47 = r4 * r2;
    r60 = r31 * r38;
    r60 = fmaf(r35, r44, r0 * r60);
    r73 = r4 * r3;
    r10 = r43 * r6;
    r22 = r35 * r45;
    r22 = fmaf(r1, r22, r0 * r10);
    r73 = fmaf(r22, r73, r60 * r47);
    WriteSum4<float, float>((float*)inout_shared, r62, r72, r42, r73);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r4 * r2;
    r42 = r31 * r34;
    r42 = fmaf(r0, r42, r41 * r44);
    r72 = r4 * r3;
    r62 = r41 * r45;
    r47 = r43 * r51;
    r47 = fmaf(r0, r47, r1 * r62);
    r72 = fmaf(r47, r72, r42 * r73);
    r73 = r4 * r3;
    r62 = r43 * r36;
    r10 = r39 * r45;
    r10 = fmaf(r1, r10, r0 * r62);
    r62 = r4 * r2;
    r1 = r31 * r23;
    r44 = fmaf(r39, r44, r0 * r1);
    r62 = fmaf(r44, r62, r10 * r73);
    WriteSum2<float, float>((float*)inout_shared, r72, r62);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = fmaf(r65, r65, r63 * r63);
    r72 = fmaf(r32, r32, r40 * r40);
    r73 = fmaf(r9, r9, r59 * r59);
    r1 = fmaf(r22, r22, r60 * r60);
    WriteSum4<float, float>((float*)inout_shared, r62, r72, r73, r1);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fmaf(r47, r47, r42 * r42);
    r73 = fmaf(r44, r44, r10 * r10);
    WriteSum2<float, float>((float*)inout_shared, r1, r73);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fmaf(r65, r40, r63 * r32);
    r1 = fmaf(r65, r59, r63 * r9);
    r72 = fmaf(r63, r22, r65 * r60);
    r62 = fmaf(r65, r42, r63 * r47);
    WriteSum4<float, float>((float*)inout_shared, r73, r1, r72, r62);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fmaf(r63, r10, r65 * r44);
    r65 = fmaf(r40, r59, r32 * r9);
    r62 = fmaf(r32, r22, r40 * r60);
    r72 = fmaf(r40, r42, r32 * r47);
    WriteSum4<float, float>((float*)inout_shared, r63, r65, r62, r72);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fmaf(r32, r10, r40 * r44);
    r40 = fmaf(r59, r60, r9 * r22);
    r72 = fmaf(r9, r47, r59 * r42);
    r59 = fmaf(r59, r44, r9 * r10);
    WriteSum4<float, float>((float*)inout_shared, r32, r40, r72, r59);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fmaf(r60, r42, r22 * r47);
    r60 = fmaf(r60, r44, r22 * r10);
    r10 = fmaf(r47, r10, r42 * r44);
    WriteSum3<float, float>((float*)inout_shared, r59, r60, r10);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
}

void PinholeSplitFixedFocalFixedPrincipalPointFixedPointResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedFocalFixedPrincipalPointFixedPointResJacKernel<<<n_blocks,
                                                                    1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      pixel,
      pixel_num_alloc,
      focal,
      focal_num_alloc,
      principal_point,
      principal_point_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar