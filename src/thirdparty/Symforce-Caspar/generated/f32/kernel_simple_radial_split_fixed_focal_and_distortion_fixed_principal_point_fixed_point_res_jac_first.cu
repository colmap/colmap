#include "kernel_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointResJacFirstKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
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

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79;

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
    r34 = 9.99999999999999955e-07;
    r30 = r20 * r23;
    r30 = fmaf(r31, r30, r32);
    r30 = fmaf(r7, r30, r6);
    r6 = r15 * r17;
    r6 = fmaf(r20, r6, r38);
    r42 = r21 + r42;
    r38 = r14 * r14;
    r38 = r38 * r20;
    r42 = r42 + r38;
    r32 = r15 * r16;
    r32 = r32 * r0;
    r45 = r14 * r17;
    r45 = fmaf(r0, r45, r32);
    r46 = r0 * r18;
    r46 = r46 * r23;
    r47 = fmaf(r28, r33, r46);
    r48 = r28 * r28;
    r48 = r48 * r20;
    r26 = r48 + r26;
    r30 = fmaf(r35, r6, r30);
    r30 = fmaf(r37, r42, r30);
    r30 = fmaf(r36, r45, r30);
    r30 = fmaf(r8, r47, r30);
    r30 = fmaf(r9, r26, r30);
    r26 = copysign(1.0, r30);
    r26 = fmaf(r34, r26, r30);
    r34 = 1.0 / r26;
    ReadIdx2<1024, float, float, float2>(focal_and_distortion,
                                         0 * focal_and_distortion_num_alloc,
                                         global_thread_idx,
                                         r30,
                                         r47);
    r49 = r26 * r26;
    r50 = 1.0 / r49;
    r51 = r27 * r50;
    r29 = fmaf(r18, r33, r29);
    r29 = fmaf(r7, r29, r5);
    r5 = r16 * r17;
    r5 = fmaf(r0, r5, r41);
    r44 = r21 + r44;
    r44 = r44 + r38;
    r38 = r14 * r17;
    r38 = fmaf(r20, r38, r32);
    r32 = r28 * r20;
    r32 = fmaf(r31, r32, r46);
    r19 = r21 + r19;
    r19 = r19 + r48;
    r29 = fmaf(r35, r5, r29);
    r29 = fmaf(r36, r44, r29);
    r29 = fmaf(r37, r38, r29);
    r29 = fmaf(r9, r32, r29);
    r29 = fmaf(r8, r19, r29);
    r19 = r29 * r29;
    r32 = fmaf(r50, r19, r27 * r51);
    r32 = fmaf(r47, r32, r21);
    r32 = r30 * r32;
    r21 = r34 * r32;
    r2 = fmaf(r27, r21, r2);
    r3 = fmaf(r3, r4, r1);
    r3 = fmaf(r29, r21, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = fmaf(r3, r3, r2 * r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r1);
  if (global_thread_idx < problem_size) {
    r1 = r4 * r3;
    r37 = r28 * r20;
    r36 = r10 * r17;
    r35 = -5.00000000000000000e-01;
    r48 = r13 * r14;
    r48 = fmaf(r35, r48, r35 * r36);
    r36 = r12 * r15;
    r48 = fmaf(r35, r36, r48);
    r46 = r11 * r16;
    r41 = 5.00000000000000000e-01;
    r48 = fmaf(r41, r46, r48);
    r46 = r20 * r31;
    r36 = r10 * r14;
    r52 = r12 * r16;
    r52 = fmaf(r35, r52, r35 * r36);
    r36 = r17 * r41;
    r53 = r11 * r35;
    r52 = fmaf(r13, r36, r52);
    r52 = fmaf(r15, r53, r52);
    r46 = r46 * r52;
    r37 = fmaf(r48, r37, r46);
    r54 = r0 * r23;
    r55 = fmaf(r41, r25, r17 * r53);
    r55 = fmaf(r35, r22, r55);
    r55 = fmaf(r35, r24, r55);
    r54 = r54 * r55;
    r56 = r0 * r18;
    r57 = r11 * r14;
    r58 = r10 * r15;
    r58 = fmaf(r35, r58, r41 * r57);
    r57 = r13 * r16;
    r58 = fmaf(r41, r57, r58);
    r58 = fmaf(r12, r36, r58);
    r56 = fmaf(r58, r56, r54);
    r37 = r37 + r56;
    r57 = -4.00000000000000000e+00;
    r59 = r28 * r57;
    r60 = r52 * r59;
    r61 = r18 * r55;
    r62 = r57 * r61;
    r63 = r60 + r62;
    r63 = fmaf(r8, r63, r9 * r37);
    r37 = r0 * r28;
    r37 = r37 * r58;
    r64 = fmaf(r55, r33, r37);
    r65 = r0 * r23;
    r65 = r65 * r52;
    r66 = r0 * r18;
    r66 = fmaf(r48, r66, r65);
    r64 = r64 + r66;
    r63 = fmaf(r7, r64, r63);
    r64 = r4 * r29;
    r67 = r0 * r28;
    r67 = r67 * r48;
    r68 = r52 * r33;
    r69 = r67 + r68;
    r56 = r56 + r69;
    r70 = r20 * r31;
    r71 = r23 * r48;
    r70 = fmaf(r20, r71, r58 * r70);
    r72 = r0 * r28;
    r73 = r0 * r18;
    r73 = r73 * r52;
    r72 = fmaf(r55, r72, r73);
    r70 = r70 + r72;
    r70 = fmaf(r7, r70, r8 * r56);
    r56 = r23 * r58;
    r56 = r56 * r57;
    r60 = r56 + r60;
    r70 = fmaf(r9, r60, r70);
    r64 = r64 * r70;
    r64 = r64 * r50;
    r64 = fmaf(r32, r64, r63 * r21);
    r60 = r30 * r47;
    r74 = r0 * r29;
    r74 = r74 * r63;
    r58 = fmaf(r58, r33, r0 * r71);
    r58 = r58 + r72;
    r37 = r65 + r37;
    r65 = r18 * r20;
    r37 = fmaf(r48, r65, r37);
    r63 = r20 * r31;
    r37 = fmaf(r55, r63, r37);
    r37 = fmaf(r8, r37, r9 * r58);
    r62 = r56 + r62;
    r37 = fmaf(r7, r62, r37);
    r62 = r0 * r37;
    r62 = fmaf(r51, r62, r50 * r74);
    r49 = r26 * r49;
    r49 = 1.0 / r49;
    r49 = r20 * r49;
    r26 = r70 * r49;
    r62 = fmaf(r19, r26, r62);
    r74 = r27 * r27;
    r74 = r74 * r49;
    r62 = fmaf(r70, r74, r62);
    r60 = r60 * r62;
    r60 = r60 * r34;
    r64 = fmaf(r29, r60, r64);
    r62 = r4 * r2;
    r26 = r4 * r32;
    r26 = r26 * r51;
    r56 = fmaf(r70, r26, r37 * r21);
    r56 = fmaf(r27, r60, r56);
    r62 = fmaf(r56, r62, r64 * r1);
    r1 = r4 * r3;
    r60 = r28 * r20;
    r60 = fmaf(r55, r60, r73);
    r73 = r0 * r23;
    r58 = r13 * r14;
    r63 = r12 * r15;
    r63 = fmaf(r41, r63, r41 * r58);
    r63 = fmaf(r10, r36, r63);
    r63 = fmaf(r16, r53, r63);
    r73 = r73 * r63;
    r58 = r20 * r31;
    r65 = r12 * r17;
    r75 = r10 * r15;
    r75 = fmaf(r41, r75, r35 * r65);
    r65 = r13 * r16;
    r75 = fmaf(r35, r65, r75);
    r75 = fmaf(r14, r53, r75);
    r60 = fmaf(r75, r58, r60);
    r60 = r60 + r73;
    r58 = fmaf(r63, r33, r0 * r61);
    r53 = r0 * r28;
    r53 = r53 * r52;
    r65 = r0 * r23;
    r65 = fmaf(r75, r65, r53);
    r58 = r58 + r65;
    r58 = fmaf(r7, r58, r9 * r60);
    r60 = r18 * r57;
    r60 = r60 * r63;
    r76 = r75 * r59;
    r77 = r60 + r76;
    r58 = fmaf(r8, r77, r58);
    r77 = r30 * r47;
    r68 = r54 + r68;
    r54 = r0 * r18;
    r54 = r54 * r75;
    r78 = r0 * r28;
    r78 = fmaf(r63, r78, r54);
    r68 = r68 + r78;
    r79 = r23 * r52;
    r79 = r79 * r57;
    r60 = r60 + r79;
    r60 = fmaf(r7, r60, r9 * r68);
    r68 = r20 * r31;
    r61 = fmaf(r20, r61, r63 * r68);
    r61 = r61 + r65;
    r60 = fmaf(r8, r61, r60);
    r61 = r0 * r60;
    r68 = r20 * r23;
    r68 = fmaf(r55, r68, r46);
    r68 = r68 + r78;
    r73 = fmaf(r75, r33, r73);
    r73 = r73 + r72;
    r73 = fmaf(r8, r73, r7 * r68);
    r76 = r79 + r76;
    r73 = fmaf(r9, r76, r73);
    r61 = fmaf(r73, r74, r51 * r61);
    r76 = r73 * r49;
    r61 = fmaf(r19, r76, r61);
    r79 = r0 * r29;
    r79 = r79 * r58;
    r61 = fmaf(r50, r79, r61);
    r77 = r77 * r29;
    r77 = r77 * r61;
    r77 = fmaf(r34, r77, r58 * r21);
    r58 = r4 * r29;
    r58 = r58 * r73;
    r58 = r58 * r50;
    r77 = fmaf(r32, r58, r77);
    r58 = r4 * r2;
    r79 = fmaf(r60, r21, r73 * r26);
    r76 = r30 * r47;
    r76 = r76 * r27;
    r76 = r76 * r61;
    r79 = fmaf(r34, r76, r79);
    r58 = fmaf(r79, r58, r77 * r1);
    r1 = r4 * r3;
    r76 = r30 * r47;
    r25 = fmaf(r35, r25, r11 * r36);
    r25 = fmaf(r41, r22, r25);
    r25 = fmaf(r41, r24, r25);
    r59 = r25 * r59;
    r71 = r57 * r71;
    r24 = r59 + r71;
    r41 = r0 * r18;
    r41 = r41 * r25;
    r53 = r53 + r41;
    r22 = r20 * r23;
    r53 = fmaf(r75, r22, r53);
    r35 = r20 * r31;
    r53 = fmaf(r48, r35, r53);
    r53 = fmaf(r7, r53, r9 * r24);
    r24 = r0 * r28;
    r24 = fmaf(r25, r33, r75 * r24);
    r24 = r24 + r66;
    r53 = fmaf(r8, r24, r53);
    r67 = r46 + r67;
    r46 = r0 * r23;
    r46 = r46 * r25;
    r24 = r18 * r20;
    r67 = fmaf(r75, r24, r67);
    r67 = r67 + r46;
    r52 = r18 * r52;
    r52 = r52 * r57;
    r71 = r52 + r71;
    r71 = fmaf(r7, r71, r8 * r67);
    r33 = fmaf(r48, r33, r41);
    r33 = r33 + r65;
    r71 = fmaf(r9, r33, r71);
    r33 = r0 * r71;
    r33 = fmaf(r51, r33, r53 * r74);
    r65 = r53 * r49;
    r33 = fmaf(r19, r65, r33);
    r48 = r0 * r29;
    r46 = r54 + r46;
    r46 = r46 + r69;
    r69 = r28 * r20;
    r54 = r20 * r31;
    r54 = fmaf(r25, r54, r75 * r69);
    r54 = r54 + r66;
    r54 = fmaf(r9, r54, r7 * r46);
    r59 = r52 + r59;
    r54 = fmaf(r8, r59, r54);
    r48 = r48 * r54;
    r33 = fmaf(r50, r48, r33);
    r76 = r76 * r29;
    r76 = r76 * r33;
    r48 = r4 * r29;
    r48 = r48 * r53;
    r48 = r48 * r50;
    r48 = fmaf(r32, r48, r34 * r76);
    r48 = fmaf(r54, r21, r48);
    r54 = r4 * r2;
    r76 = fmaf(r71, r21, r53 * r26);
    r65 = r30 * r47;
    r65 = r65 * r27;
    r65 = r65 * r33;
    r76 = fmaf(r34, r65, r76);
    r54 = fmaf(r76, r54, r48 * r1);
    r1 = r4 * r2;
    r65 = r30 * r47;
    r33 = r6 * r49;
    r59 = r0 * r5;
    r59 = r59 * r29;
    r59 = fmaf(r50, r59, r19 * r33);
    r33 = r0 * r43;
    r59 = fmaf(r51, r33, r59);
    r59 = fmaf(r6, r74, r59);
    r65 = r65 * r27;
    r65 = r65 * r59;
    r65 = fmaf(r43, r21, r34 * r65);
    r65 = fmaf(r6, r26, r65);
    r33 = r4 * r3;
    r8 = r30 * r47;
    r8 = r8 * r29;
    r8 = r8 * r59;
    r8 = fmaf(r34, r8, r5 * r21);
    r59 = r4 * r6;
    r59 = r59 * r29;
    r59 = r59 * r50;
    r8 = fmaf(r32, r59, r8);
    r33 = fmaf(r8, r33, r65 * r1);
    WriteSum4<float, float>((float*)inout_shared, r62, r58, r54, r33);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = r4 * r2;
    r54 = r30 * r47;
    r58 = r45 * r49;
    r62 = r0 * r44;
    r62 = r62 * r29;
    r62 = fmaf(r50, r62, r19 * r58);
    r58 = r0 * r40;
    r62 = fmaf(r51, r58, r62);
    r62 = fmaf(r45, r74, r62);
    r54 = r54 * r27;
    r54 = r54 * r62;
    r54 = fmaf(r34, r54, r45 * r26);
    r54 = fmaf(r40, r21, r54);
    r58 = r4 * r3;
    r1 = r30 * r47;
    r1 = r1 * r29;
    r1 = r1 * r62;
    r1 = fmaf(r44, r21, r34 * r1);
    r62 = r4 * r45;
    r62 = r62 * r29;
    r62 = r62 * r50;
    r1 = fmaf(r32, r62, r1);
    r58 = fmaf(r1, r58, r54 * r33);
    r33 = r4 * r2;
    r62 = r30 * r47;
    r59 = r0 * r38;
    r59 = r59 * r29;
    r52 = r42 * r49;
    r52 = fmaf(r19, r52, r50 * r59);
    r59 = r0 * r39;
    r52 = fmaf(r51, r59, r52);
    r52 = fmaf(r42, r74, r52);
    r62 = r62 * r27;
    r62 = r62 * r52;
    r62 = fmaf(r39, r21, r34 * r62);
    r62 = fmaf(r42, r26, r62);
    r26 = r4 * r3;
    r27 = r30 * r47;
    r27 = r27 * r29;
    r27 = r27 * r52;
    r21 = fmaf(r38, r21, r34 * r27);
    r27 = r4 * r42;
    r27 = r27 * r29;
    r27 = r27 * r50;
    r21 = fmaf(r32, r27, r21);
    r26 = fmaf(r21, r26, r62 * r33);
    WriteSum2<float, float>((float*)inout_shared, r58, r26);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fmaf(r64, r64, r56 * r56);
    r58 = fmaf(r77, r77, r79 * r79);
    r33 = fmaf(r76, r76, r48 * r48);
    r27 = fmaf(r65, r65, r8 * r8);
    WriteSum4<float, float>((float*)inout_shared, r26, r58, r33, r27);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fmaf(r54, r54, r1 * r1);
    r33 = fmaf(r62, r62, r21 * r21);
    WriteSum2<float, float>((float*)inout_shared, r27, r33);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fmaf(r56, r79, r64 * r77);
    r27 = fmaf(r64, r48, r56 * r76);
    r58 = fmaf(r56, r65, r64 * r8);
    r26 = fmaf(r56, r54, r64 * r1);
    WriteSum4<float, float>((float*)inout_shared, r33, r27, r58, r26);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = fmaf(r56, r62, r64 * r21);
    r64 = fmaf(r77, r48, r79 * r76);
    r26 = fmaf(r79, r65, r77 * r8);
    r58 = fmaf(r77, r1, r79 * r54);
    WriteSum4<float, float>((float*)inout_shared, r56, r64, r26, r58);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = fmaf(r77, r21, r79 * r62);
    r79 = fmaf(r48, r8, r76 * r65);
    r58 = fmaf(r48, r1, r76 * r54);
    r48 = fmaf(r48, r21, r76 * r62);
    WriteSum4<float, float>((float*)inout_shared, r77, r79, r58, r48);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = fmaf(r65, r54, r8 * r1);
    r8 = fmaf(r8, r21, r65 * r62);
    r62 = fmaf(r54, r62, r1 * r21);
    WriteSum3<float, float>((float*)inout_shared, r48, r8, r62);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
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
  SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointResJacFirstKernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              sensor_from_rig,
              sensor_from_rig_num_alloc,
              pixel,
              pixel_num_alloc,
              focal_and_distortion,
              focal_and_distortion_num_alloc,
              principal_point,
              principal_point_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_rTr,
              out_pose_njtr,
              out_pose_njtr_num_alloc,
              out_pose_precond_diag,
              out_pose_precond_diag_num_alloc,
              out_pose_precond_tril,
              out_pose_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar