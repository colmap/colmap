#include "kernel_simple_radial_split_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPrincipalPointFixedPointResJacKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
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
        float* out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        float* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        float* const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        float* const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81;

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
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r5,
                                         r6);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r7, r8, r9);
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r10,
                       r11,
                       r12,
                       r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r14,
                                         r15,
                                         r16,
                                         r17);
    r18 = fmaf(r11, r14, r12 * r17);
    r19 = r10 * r15;
    r18 = fmaf(r4, r19, r18);
    r18 = fmaf(r13, r16, r18);
    r19 = r18 * r18;
    r20 = -2.00000000000000000e+00;
    r19 = r19 * r20;
    r21 = 1.00000000000000000e+00;
    r22 = r13 * r15;
    r23 = fmaf(r11, r17, r22);
    r24 = r10 * r16;
    r25 = r12 * r14;
    r23 = r23 + r24;
    r23 = fmaf(r4, r25, r23);
    r26 = r20 * r23;
    r26 = fmaf(r23, r26, r21);
    r27 = r19 + r26;
    r27 = fmaf(r7, r27, r0);
    r0 = 2.00000000000000000e+00;
    r28 = fmaf(r13, r14, r10 * r17);
    r29 = r11 * r16;
    r28 = fmaf(r4, r29, r28);
    r28 = fmaf(r12, r15, r28);
    r29 = r0 * r28;
    r29 = r29 * r23;
    r30 = r18 * r20;
    r31 = fmaf(r11, r15, r10 * r14);
    r31 = fmaf(r12, r16, r31);
    r31 = fmaf(r4, r31, r13 * r17);
    r30 = fmaf(r31, r30, r29);
    r32 = r0 * r18;
    r32 = r32 * r28;
    r33 = r0 * r31;
    r34 = fmaf(r23, r33, r32);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r35,
                       r36,
                       r37);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r38 = r14 * r16;
    r38 = r38 * r0;
    r39 = r15 * r17;
    r39 = fmaf(r0, r39, r38);
    r40 = r16 * r17;
    r41 = r14 * r15;
    r41 = r41 * r0;
    r40 = fmaf(r20, r40, r41);
    r42 = r15 * r15;
    r42 = r42 * r20;
    r43 = r21 + r42;
    r44 = r16 * r16;
    r44 = r44 * r20;
    r43 = r43 + r44;
    r27 = fmaf(r8, r30, r27);
    r27 = fmaf(r9, r34, r27);
    r27 = fmaf(r37, r39, r27);
    r27 = fmaf(r36, r40, r27);
    r27 = fmaf(r35, r43, r27);
  };
  LoadShared<2, float, float>(focal_and_extra,
                              0 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r34,
                       r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r45 = 9.99999999999999955e-07;
    r46 = r20 * r23;
    r46 = fmaf(r31, r46, r32);
    r46 = fmaf(r7, r46, r6);
    r6 = r15 * r17;
    r6 = fmaf(r20, r6, r38);
    r42 = r21 + r42;
    r38 = r14 * r14;
    r38 = r38 * r20;
    r42 = r42 + r38;
    r32 = r15 * r16;
    r32 = r32 * r0;
    r47 = r14 * r17;
    r47 = fmaf(r0, r47, r32);
    r48 = r0 * r18;
    r48 = r48 * r23;
    r49 = fmaf(r28, r33, r48);
    r50 = r28 * r28;
    r50 = r50 * r20;
    r26 = r50 + r26;
    r46 = fmaf(r35, r6, r46);
    r46 = fmaf(r37, r42, r46);
    r46 = fmaf(r36, r47, r46);
    r46 = fmaf(r8, r49, r46);
    r46 = fmaf(r9, r26, r46);
    r26 = copysign(1.0, r46);
    r26 = fmaf(r45, r26, r46);
    r45 = r26 * r26;
    r46 = 1.0 / r45;
    r49 = r27 * r46;
    r29 = fmaf(r18, r33, r29);
    r29 = fmaf(r7, r29, r5);
    r5 = r16 * r17;
    r5 = fmaf(r0, r5, r41);
    r44 = r21 + r44;
    r44 = r44 + r38;
    r38 = r14 * r17;
    r38 = fmaf(r20, r38, r32);
    r32 = r28 * r20;
    r32 = fmaf(r31, r32, r48);
    r19 = r21 + r19;
    r19 = r19 + r50;
    r29 = fmaf(r35, r5, r29);
    r29 = fmaf(r36, r44, r29);
    r29 = fmaf(r37, r38, r29);
    r29 = fmaf(r9, r32, r29);
    r29 = fmaf(r8, r19, r29);
    r19 = r29 * r29;
    r32 = fmaf(r46, r19, r27 * r49);
    r21 = fmaf(r30, r32, r21);
    r37 = r27 * r21;
    r36 = 1.0 / r26;
    r35 = r34 * r36;
    r2 = fmaf(r35, r37, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r29 * r21;
    r3 = fmaf(r35, r1, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r0 * r23;
    r37 = -5.00000000000000000e-01;
    r50 = r11 * r37;
    r48 = 5.00000000000000000e-01;
    r41 = fmaf(r48, r25, r17 * r50);
    r41 = fmaf(r37, r22, r41);
    r41 = fmaf(r37, r24, r41);
    r1 = r1 * r41;
    r51 = r0 * r18;
    r52 = r11 * r14;
    r53 = r10 * r15;
    r53 = fmaf(r37, r53, r48 * r52);
    r52 = r13 * r16;
    r53 = fmaf(r48, r52, r53);
    r54 = r17 * r48;
    r53 = fmaf(r12, r54, r53);
    r51 = fmaf(r53, r51, r1);
    r52 = r0 * r28;
    r55 = r10 * r17;
    r56 = r13 * r14;
    r56 = fmaf(r37, r56, r37 * r55);
    r55 = r12 * r15;
    r56 = fmaf(r37, r55, r56);
    r57 = r11 * r16;
    r56 = fmaf(r48, r57, r56);
    r52 = r52 * r56;
    r57 = r10 * r14;
    r55 = r12 * r16;
    r55 = fmaf(r37, r55, r37 * r57);
    r55 = fmaf(r13, r54, r55);
    r55 = fmaf(r15, r50, r55);
    r57 = r55 * r33;
    r58 = r52 + r57;
    r59 = r51 + r58;
    r60 = r20 * r31;
    r61 = r23 * r56;
    r60 = fmaf(r20, r61, r53 * r60);
    r62 = r0 * r28;
    r63 = r0 * r18;
    r63 = r63 * r55;
    r62 = fmaf(r41, r62, r63);
    r60 = r60 + r62;
    r60 = fmaf(r7, r60, r8 * r59);
    r59 = r23 * r53;
    r64 = -4.00000000000000000e+00;
    r59 = r59 * r64;
    r65 = r28 * r64;
    r66 = r55 * r65;
    r67 = r59 + r66;
    r60 = fmaf(r9, r67, r60);
    r67 = r60 * r49;
    r68 = r4 * r21;
    r69 = r34 * r68;
    r70 = r0 * r29;
    r71 = r28 * r20;
    r72 = r20 * r31;
    r72 = r72 * r55;
    r71 = fmaf(r56, r71, r72);
    r71 = r71 + r51;
    r51 = r18 * r41;
    r73 = r64 * r51;
    r66 = r66 + r73;
    r66 = fmaf(r8, r66, r9 * r71);
    r71 = r0 * r28;
    r71 = r71 * r53;
    r74 = fmaf(r41, r33, r71);
    r75 = r0 * r23;
    r75 = r75 * r55;
    r76 = r0 * r18;
    r76 = fmaf(r56, r76, r75);
    r74 = r74 + r76;
    r66 = fmaf(r7, r74, r66);
    r70 = r70 * r66;
    r53 = fmaf(r53, r33, r0 * r61);
    r53 = r53 + r62;
    r71 = r75 + r71;
    r75 = r18 * r20;
    r71 = fmaf(r56, r75, r71);
    r74 = r20 * r31;
    r71 = fmaf(r41, r74, r71);
    r71 = fmaf(r8, r71, r9 * r53);
    r73 = r59 + r73;
    r71 = fmaf(r7, r73, r71);
    r73 = r0 * r71;
    r73 = fmaf(r49, r73, r46 * r70);
    r45 = r26 * r45;
    r45 = 1.0 / r45;
    r45 = r20 * r45;
    r26 = r60 * r45;
    r73 = fmaf(r19, r26, r73);
    r70 = r27 * r27;
    r70 = r70 * r45;
    r73 = fmaf(r60, r70, r73);
    r30 = r30 * r35;
    r73 = r73 * r30;
    r67 = fmaf(r27, r73, r69 * r67);
    r26 = r21 * r71;
    r67 = fmaf(r35, r26, r67);
    r26 = r29 * r46;
    r26 = r26 * r69;
    r73 = fmaf(r29, r73, r60 * r26);
    r59 = r21 * r66;
    r73 = fmaf(r35, r59, r73);
    r57 = r1 + r57;
    r1 = r0 * r18;
    r59 = r12 * r17;
    r53 = r10 * r15;
    r53 = fmaf(r48, r53, r37 * r59);
    r59 = r13 * r16;
    r53 = fmaf(r37, r59, r53);
    r53 = fmaf(r14, r50, r53);
    r1 = r1 * r53;
    r59 = r0 * r28;
    r74 = r13 * r14;
    r75 = r12 * r15;
    r75 = fmaf(r48, r75, r48 * r74);
    r75 = fmaf(r10, r54, r75);
    r75 = fmaf(r16, r50, r75);
    r59 = fmaf(r75, r59, r1);
    r57 = r57 + r59;
    r50 = r23 * r55;
    r50 = r50 * r64;
    r74 = r18 * r64;
    r74 = r74 * r75;
    r77 = r50 + r74;
    r77 = fmaf(r7, r77, r9 * r57);
    r57 = r20 * r31;
    r57 = fmaf(r20, r51, r75 * r57);
    r78 = r0 * r28;
    r78 = r78 * r55;
    r79 = r0 * r23;
    r79 = fmaf(r53, r79, r78);
    r57 = r57 + r79;
    r77 = fmaf(r8, r57, r77);
    r57 = r0 * r77;
    r80 = r20 * r23;
    r80 = fmaf(r41, r80, r72);
    r80 = r80 + r59;
    r59 = r0 * r23;
    r59 = r59 * r75;
    r81 = fmaf(r53, r33, r59);
    r81 = r81 + r62;
    r81 = fmaf(r8, r81, r7 * r80);
    r80 = r53 * r65;
    r50 = r50 + r80;
    r81 = fmaf(r9, r50, r81);
    r57 = fmaf(r81, r70, r49 * r57);
    r50 = r81 * r45;
    r57 = fmaf(r19, r50, r57);
    r62 = r0 * r29;
    r59 = r63 + r59;
    r63 = r28 * r20;
    r59 = fmaf(r41, r63, r59);
    r41 = r20 * r31;
    r59 = fmaf(r53, r41, r59);
    r75 = fmaf(r75, r33, r0 * r51);
    r75 = r75 + r79;
    r75 = fmaf(r7, r75, r9 * r59);
    r80 = r74 + r80;
    r75 = fmaf(r8, r80, r75);
    r62 = r62 * r75;
    r57 = fmaf(r46, r62, r57);
    r62 = r27 * r57;
    r50 = r21 * r77;
    r50 = fmaf(r35, r50, r30 * r62);
    r62 = r81 * r49;
    r50 = fmaf(r69, r62, r50);
    r62 = r21 * r75;
    r62 = fmaf(r81, r26, r35 * r62);
    r80 = r29 * r57;
    r62 = fmaf(r30, r80, r62);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r67,
                                          r73,
                                          r50,
                                          r62);
    r25 = fmaf(r37, r25, r11 * r54);
    r25 = fmaf(r48, r22, r25);
    r25 = fmaf(r48, r24, r25);
    r65 = r25 * r65;
    r61 = r64 * r61;
    r24 = r65 + r61;
    r48 = r0 * r18;
    r48 = r48 * r25;
    r78 = r78 + r48;
    r22 = r20 * r23;
    r78 = fmaf(r53, r22, r78);
    r37 = r20 * r31;
    r78 = fmaf(r56, r37, r78);
    r78 = fmaf(r7, r78, r9 * r24);
    r24 = r0 * r28;
    r24 = fmaf(r25, r33, r53 * r24);
    r24 = r24 + r76;
    r78 = fmaf(r8, r24, r78);
    r24 = r78 * r49;
    r72 = r52 + r72;
    r52 = r0 * r23;
    r52 = r52 * r25;
    r37 = r18 * r20;
    r72 = fmaf(r53, r37, r72);
    r72 = r72 + r52;
    r55 = r18 * r55;
    r55 = r55 * r64;
    r61 = r55 + r61;
    r61 = fmaf(r7, r61, r8 * r72);
    r33 = fmaf(r56, r33, r48);
    r33 = r33 + r79;
    r61 = fmaf(r9, r33, r61);
    r33 = r21 * r61;
    r33 = fmaf(r35, r33, r69 * r24);
    r24 = r0 * r61;
    r24 = fmaf(r49, r24, r78 * r70);
    r79 = r78 * r45;
    r24 = fmaf(r19, r79, r24);
    r56 = r0 * r29;
    r52 = r1 + r52;
    r52 = r52 + r58;
    r58 = r28 * r20;
    r1 = r20 * r31;
    r1 = fmaf(r25, r1, r53 * r58);
    r1 = r1 + r76;
    r1 = fmaf(r9, r1, r7 * r52);
    r65 = r55 + r65;
    r1 = fmaf(r8, r65, r1);
    r56 = r56 * r1;
    r24 = fmaf(r46, r56, r24);
    r56 = r27 * r24;
    r33 = fmaf(r30, r56, r33);
    r56 = r29 * r24;
    r56 = fmaf(r78, r26, r30 * r56);
    r79 = r21 * r1;
    r56 = fmaf(r35, r79, r56);
    r79 = r43 * r21;
    r65 = r6 * r45;
    r8 = r0 * r5;
    r8 = r8 * r29;
    r8 = fmaf(r46, r8, r19 * r65);
    r65 = r0 * r43;
    r8 = fmaf(r49, r65, r8);
    r8 = fmaf(r6, r70, r8);
    r65 = r27 * r8;
    r65 = fmaf(r30, r65, r35 * r79);
    r79 = r6 * r49;
    r65 = fmaf(r69, r79, r65);
    r79 = r5 * r21;
    r79 = fmaf(r35, r79, r6 * r26);
    r55 = r29 * r8;
    r79 = fmaf(r30, r55, r79);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r33,
                                          r56,
                                          r65,
                                          r79);
    r55 = r47 * r49;
    r9 = r47 * r45;
    r52 = r0 * r44;
    r52 = r52 * r29;
    r52 = fmaf(r46, r52, r19 * r9);
    r9 = r0 * r40;
    r52 = fmaf(r49, r9, r52);
    r52 = fmaf(r47, r70, r52);
    r9 = r27 * r52;
    r9 = fmaf(r30, r9, r69 * r55);
    r55 = r40 * r21;
    r9 = fmaf(r35, r55, r9);
    r55 = r29 * r52;
    r7 = r44 * r21;
    r7 = fmaf(r35, r7, r30 * r55);
    r7 = fmaf(r47, r26, r7);
    r55 = r42 * r49;
    r76 = r39 * r21;
    r76 = fmaf(r35, r76, r69 * r55);
    r55 = r0 * r38;
    r55 = r55 * r29;
    r69 = r42 * r45;
    r69 = fmaf(r19, r69, r46 * r55);
    r55 = r0 * r39;
    r69 = fmaf(r49, r55, r69);
    r69 = fmaf(r42, r70, r69);
    r70 = r27 * r69;
    r76 = fmaf(r30, r70, r76);
    r70 = r38 * r21;
    r55 = r29 * r69;
    r55 = fmaf(r30, r55, r35 * r70);
    r55 = fmaf(r42, r26, r55);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r9,
                                          r7,
                                          r76,
                                          r55);
    r26 = r4 * r2;
    r70 = r4 * r3;
    r70 = fmaf(r73, r70, r67 * r26);
    r26 = r4 * r2;
    r30 = r4 * r3;
    r30 = fmaf(r62, r30, r50 * r26);
    r26 = r4 * r2;
    r58 = r4 * r3;
    r58 = fmaf(r56, r58, r33 * r26);
    r26 = r4 * r3;
    r25 = r4 * r2;
    r25 = fmaf(r65, r25, r79 * r26);
    WriteSum4<float, float>((float*)inout_shared, r70, r30, r58, r25);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r4 * r2;
    r58 = r4 * r3;
    r58 = fmaf(r7, r58, r9 * r25);
    r25 = r4 * r3;
    r30 = r4 * r2;
    r30 = fmaf(r76, r30, r55 * r25);
    WriteSum2<float, float>((float*)inout_shared, r58, r30);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fmaf(r67, r67, r73 * r73);
    r58 = fmaf(r50, r50, r62 * r62);
    r25 = fmaf(r56, r56, r33 * r33);
    r70 = fmaf(r65, r65, r79 * r79);
    WriteSum4<float, float>((float*)inout_shared, r30, r58, r25, r70);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fmaf(r9, r9, r7 * r7);
    r25 = fmaf(r76, r76, r55 * r55);
    WriteSum2<float, float>((float*)inout_shared, r70, r25);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = fmaf(r73, r62, r67 * r50);
    r70 = fmaf(r73, r56, r67 * r33);
    r58 = fmaf(r67, r65, r73 * r79);
    r30 = fmaf(r67, r9, r73 * r7);
    WriteSum4<float, float>((float*)inout_shared, r25, r70, r58, r30);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fmaf(r73, r55, r67 * r76);
    r67 = fmaf(r62, r56, r50 * r33);
    r30 = fmaf(r50, r65, r62 * r79);
    r58 = fmaf(r50, r9, r62 * r7);
    WriteSum4<float, float>((float*)inout_shared, r73, r67, r30, r58);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = fmaf(r50, r76, r62 * r55);
    r62 = fmaf(r33, r65, r56 * r79);
    r58 = fmaf(r33, r9, r56 * r7);
    r33 = fmaf(r33, r76, r56 * r55);
    WriteSum4<float, float>((float*)inout_shared, r50, r62, r58, r33);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fmaf(r65, r9, r79 * r7);
    r79 = fmaf(r79, r55, r65 * r76);
    r55 = fmaf(r7, r55, r9 * r76);
    WriteSum3<float, float>((float*)inout_shared, r33, r79, r55);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = r27 * r21;
    r55 = r55 * r36;
    r79 = r29 * r21;
    r79 = r79 * r36;
    r33 = r27 * r32;
    r33 = r33 * r35;
    r7 = r29 * r32;
    r7 = r7 * r35;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          0 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r55,
                                          r79,
                                          r33,
                                          r7);
    r7 = r29 * r3;
    r7 = r7 * r36;
    r33 = r27 * r2;
    r33 = r33 * r36;
    r33 = fmaf(r68, r33, r68 * r7);
    r7 = r4 * r29;
    r7 = r7 * r32;
    r7 = r7 * r3;
    r68 = r4 * r27;
    r68 = r68 * r32;
    r68 = r68 * r2;
    r68 = fmaf(r35, r68, r35 * r7);
    WriteSum2<float, float>((float*)inout_shared, r33, r68);
  };
  FlushSumShared<2, float>(out_focal_and_extra_njtr,
                           0 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r21 * r46;
    r68 = r68 * r19;
    r33 = r27 * r21;
    r33 = r33 * r21;
    r33 = fmaf(r49, r33, r21 * r68);
    r7 = r27 * r49;
    r35 = r34 * r34;
    r36 = r32 * r32;
    r35 = r35 * r36;
    r36 = r46 * r19;
    r36 = fmaf(r35, r36, r35 * r7);
    WriteSum2<float, float>((float*)inout_shared, r33, r36);
  };
  FlushSumShared<2, float>(out_focal_and_extra_precond_diag,
                           0 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = r34 * r27;
    r36 = r36 * r32;
    r36 = r36 * r21;
    r33 = r34 * r32;
    r33 = fmaf(r68, r33, r49 * r36);
    WriteSum1<float, float>((float*)inout_shared, r33);
  };
  FlushSumShared<1, float>(out_focal_and_extra_precond_tril,
                           0 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
}

void SimpleRadialSplitFixedPrincipalPointFixedPointResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
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
    float* out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
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
  SimpleRadialSplitFixedPrincipalPointFixedPointResJacKernel<<<n_blocks,
                                                               1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      focal_and_extra_indices,
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
      out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar