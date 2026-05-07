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
        float* focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc,
        SharedIndex* focal_and_distortion_indices,
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
        float* out_focal_and_distortion_jac,
        unsigned int out_focal_and_distortion_jac_num_alloc,
        float* const out_focal_and_distortion_njtr,
        unsigned int out_focal_and_distortion_njtr_num_alloc,
        float* const out_focal_and_distortion_precond_diag,
        unsigned int out_focal_and_distortion_precond_diag_num_alloc,
        float* const out_focal_and_distortion_precond_tril,
        unsigned int out_focal_and_distortion_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex focal_and_distortion_indices_loc[1024];
  focal_and_distortion_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_distortion_indices[global_thread_idx]
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
    r22 = fmaf(r13, r15, r11 * r17);
    r23 = r10 * r16;
    r24 = r12 * r14;
    r22 = r22 + r23;
    r22 = fmaf(r4, r24, r22);
    r25 = r20 * r22;
    r25 = fmaf(r22, r25, r21);
    r26 = r19 + r25;
    r26 = fmaf(r7, r26, r0);
    r0 = 2.00000000000000000e+00;
    r27 = fmaf(r13, r14, r10 * r17);
    r28 = r11 * r16;
    r27 = fmaf(r4, r28, r27);
    r27 = fmaf(r12, r15, r27);
    r28 = r0 * r27;
    r28 = r28 * r22;
    r29 = r18 * r20;
    r30 = fmaf(r11, r15, r10 * r14);
    r30 = fmaf(r12, r16, r30);
    r30 = fmaf(r4, r30, r13 * r17);
    r29 = fmaf(r30, r29, r28);
    r31 = r0 * r18;
    r31 = r31 * r27;
    r32 = r0 * r30;
    r33 = fmaf(r22, r32, r31);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r34,
                       r35,
                       r36);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r37 = r14 * r16;
    r37 = r37 * r0;
    r38 = r15 * r17;
    r39 = fmaf(r0, r38, r37);
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
    r26 = fmaf(r8, r29, r26);
    r26 = fmaf(r9, r33, r26);
    r26 = fmaf(r36, r39, r26);
    r26 = fmaf(r35, r40, r26);
    r26 = fmaf(r34, r43, r26);
  };
  LoadShared<2, float, float>(focal_and_distortion,
                              0 * focal_and_distortion_num_alloc,
                              focal_and_distortion_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_and_distortion_indices_loc[threadIdx.x].target,
                       r33,
                       r29);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r45 = 9.99999999999999955e-07;
    r46 = r20 * r22;
    r46 = fmaf(r30, r46, r31);
    r46 = fmaf(r7, r46, r6);
    r38 = fmaf(r20, r38, r37);
    r42 = r21 + r42;
    r37 = r14 * r14;
    r37 = r37 * r20;
    r42 = r42 + r37;
    r6 = r15 * r16;
    r6 = r6 * r0;
    r31 = r14 * r17;
    r31 = fmaf(r0, r31, r6);
    r47 = r0 * r18;
    r47 = r47 * r22;
    r48 = fmaf(r27, r32, r47);
    r49 = r27 * r27;
    r49 = r49 * r20;
    r25 = r49 + r25;
    r46 = fmaf(r34, r38, r46);
    r46 = fmaf(r36, r42, r46);
    r46 = fmaf(r35, r31, r46);
    r46 = fmaf(r8, r48, r46);
    r46 = fmaf(r9, r25, r46);
    r25 = copysign(1.0, r46);
    r25 = fmaf(r45, r25, r46);
    r45 = r25 * r25;
    r46 = 1.0 / r45;
    r48 = r26 * r46;
    r28 = fmaf(r18, r32, r28);
    r28 = fmaf(r7, r28, r5);
    r5 = r16 * r17;
    r5 = fmaf(r0, r5, r41);
    r44 = r21 + r44;
    r44 = r44 + r37;
    r37 = r14 * r17;
    r37 = fmaf(r20, r37, r6);
    r6 = r27 * r20;
    r6 = fmaf(r30, r6, r47);
    r19 = r21 + r19;
    r19 = r19 + r49;
    r28 = fmaf(r34, r5, r28);
    r28 = fmaf(r35, r44, r28);
    r28 = fmaf(r36, r37, r28);
    r28 = fmaf(r9, r6, r28);
    r28 = fmaf(r8, r19, r28);
    r19 = r28 * r28;
    r6 = fmaf(r46, r19, r26 * r48);
    r21 = fmaf(r29, r6, r21);
    r36 = r26 * r21;
    r35 = 1.0 / r25;
    r34 = r33 * r35;
    r2 = fmaf(r34, r36, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r28 * r21;
    r3 = fmaf(r34, r1, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r10 * r17;
    r36 = -5.00000000000000000e-01;
    r49 = r13 * r14;
    r49 = fmaf(r36, r49, r36 * r1);
    r1 = r12 * r15;
    r49 = fmaf(r36, r1, r49);
    r47 = r11 * r16;
    r41 = 5.00000000000000000e-01;
    r49 = fmaf(r41, r47, r49);
    r47 = r22 * r49;
    r1 = r12 * r17;
    r50 = r11 * r14;
    r50 = fmaf(r41, r50, r41 * r1);
    r1 = r10 * r15;
    r50 = fmaf(r36, r1, r50);
    r51 = r13 * r41;
    r50 = fmaf(r16, r51, r50);
    r1 = fmaf(r50, r32, r0 * r47);
    r52 = r0 * r27;
    r53 = r13 * r15;
    r54 = r11 * r36;
    r53 = fmaf(r17, r54, r36 * r53);
    r53 = fmaf(r41, r24, r53);
    r53 = fmaf(r36, r23, r53);
    r55 = r0 * r18;
    r56 = r10 * r14;
    r57 = r12 * r16;
    r57 = fmaf(r36, r57, r36 * r56);
    r57 = fmaf(r17, r51, r57);
    r57 = fmaf(r15, r54, r57);
    r55 = r55 * r57;
    r52 = fmaf(r53, r52, r55);
    r1 = r1 + r52;
    r56 = r0 * r22;
    r56 = r56 * r57;
    r58 = r0 * r27;
    r58 = r58 * r50;
    r59 = r56 + r58;
    r60 = r18 * r20;
    r59 = fmaf(r49, r60, r59);
    r61 = r20 * r30;
    r59 = fmaf(r53, r61, r59);
    r59 = fmaf(r8, r59, r9 * r1);
    r1 = r22 * r50;
    r61 = -4.00000000000000000e+00;
    r1 = r1 * r61;
    r60 = r18 * r53;
    r62 = r61 * r60;
    r63 = r1 + r62;
    r59 = fmaf(r7, r63, r59);
    r63 = r21 * r59;
    r64 = r0 * r22;
    r64 = r64 * r53;
    r65 = r0 * r18;
    r65 = fmaf(r50, r65, r64);
    r66 = r0 * r27;
    r66 = r66 * r49;
    r67 = r57 * r32;
    r68 = r66 + r67;
    r69 = r65 + r68;
    r70 = r20 * r30;
    r70 = fmaf(r20, r47, r50 * r70);
    r70 = r70 + r52;
    r70 = fmaf(r7, r70, r8 * r69);
    r69 = r27 * r61;
    r50 = r57 * r69;
    r1 = r1 + r50;
    r70 = fmaf(r9, r1, r70);
    r1 = r70 * r48;
    r71 = r4 * r21;
    r72 = r33 * r71;
    r1 = fmaf(r72, r1, r34 * r63);
    r63 = r0 * r28;
    r73 = r27 * r20;
    r74 = r20 * r30;
    r74 = r74 * r57;
    r73 = fmaf(r49, r73, r74);
    r73 = r73 + r65;
    r50 = r62 + r50;
    r50 = fmaf(r8, r50, r9 * r73);
    r58 = fmaf(r53, r32, r58);
    r73 = r0 * r18;
    r73 = fmaf(r49, r73, r56);
    r58 = r58 + r73;
    r50 = fmaf(r7, r58, r50);
    r63 = r63 * r50;
    r58 = r0 * r59;
    r58 = fmaf(r48, r58, r46 * r63);
    r45 = r25 * r45;
    r45 = 1.0 / r45;
    r45 = r20 * r45;
    r25 = r70 * r45;
    r58 = fmaf(r19, r25, r58);
    r63 = r26 * r26;
    r63 = r63 * r45;
    r58 = fmaf(r70, r63, r58);
    r29 = r29 * r34;
    r58 = r58 * r29;
    r1 = fmaf(r26, r58, r1);
    r25 = r21 * r50;
    r56 = r28 * r46;
    r56 = r56 * r72;
    r25 = fmaf(r70, r56, r34 * r25);
    r25 = fmaf(r28, r58, r25);
    r58 = r20 * r22;
    r58 = fmaf(r53, r58, r74);
    r62 = r0 * r18;
    r65 = r12 * r17;
    r75 = r10 * r15;
    r75 = fmaf(r41, r75, r36 * r65);
    r65 = r13 * r16;
    r75 = fmaf(r36, r65, r75);
    r75 = fmaf(r14, r54, r75);
    r62 = r62 * r75;
    r65 = r0 * r27;
    r76 = r10 * r17;
    r77 = r12 * r15;
    r77 = fmaf(r41, r77, r41 * r76);
    r77 = fmaf(r14, r51, r77);
    r77 = fmaf(r16, r54, r77);
    r65 = fmaf(r77, r65, r62);
    r58 = r58 + r65;
    r54 = r0 * r22;
    r54 = r54 * r77;
    r76 = fmaf(r75, r32, r54);
    r76 = r76 + r52;
    r76 = fmaf(r8, r76, r7 * r58);
    r58 = r22 * r57;
    r58 = r58 * r61;
    r52 = r75 * r69;
    r78 = r58 + r52;
    r76 = fmaf(r9, r78, r76);
    r78 = r76 * r48;
    r67 = r64 + r67;
    r67 = r67 + r65;
    r65 = r18 * r61;
    r65 = r65 * r77;
    r58 = r58 + r65;
    r58 = fmaf(r7, r58, r9 * r67);
    r67 = r20 * r30;
    r67 = fmaf(r20, r60, r77 * r67);
    r64 = r0 * r27;
    r64 = r64 * r57;
    r79 = r0 * r22;
    r79 = fmaf(r75, r79, r64);
    r67 = r67 + r79;
    r58 = fmaf(r8, r67, r58);
    r67 = r21 * r58;
    r67 = fmaf(r34, r67, r72 * r78);
    r78 = r0 * r58;
    r78 = fmaf(r76, r63, r48 * r78);
    r80 = r76 * r45;
    r78 = fmaf(r19, r80, r78);
    r81 = r0 * r28;
    r54 = r55 + r54;
    r55 = r27 * r20;
    r54 = fmaf(r53, r55, r54);
    r53 = r20 * r30;
    r54 = fmaf(r75, r53, r54);
    r77 = fmaf(r77, r32, r0 * r60);
    r77 = r77 + r79;
    r77 = fmaf(r7, r77, r9 * r54);
    r52 = r65 + r52;
    r77 = fmaf(r8, r52, r77);
    r81 = r81 * r77;
    r78 = fmaf(r46, r81, r78);
    r81 = r26 * r78;
    r67 = fmaf(r29, r81, r67);
    r81 = r21 * r77;
    r80 = r28 * r78;
    r80 = fmaf(r29, r80, r34 * r81);
    r80 = fmaf(r76, r56, r80);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r1,
                                          r25,
                                          r67,
                                          r80);
    r81 = r11 * r17;
    r24 = fmaf(r36, r24, r41 * r81);
    r24 = fmaf(r15, r51, r24);
    r24 = fmaf(r41, r23, r24);
    r69 = r24 * r69;
    r47 = r61 * r47;
    r23 = r69 + r47;
    r41 = r0 * r18;
    r41 = r41 * r24;
    r64 = r64 + r41;
    r51 = r20 * r22;
    r64 = fmaf(r75, r51, r64);
    r36 = r20 * r30;
    r64 = fmaf(r49, r36, r64);
    r64 = fmaf(r7, r64, r9 * r23);
    r23 = r0 * r27;
    r23 = fmaf(r24, r32, r75 * r23);
    r23 = r23 + r73;
    r64 = fmaf(r8, r23, r64);
    r23 = r64 * r48;
    r74 = r66 + r74;
    r66 = r0 * r22;
    r66 = r66 * r24;
    r36 = r18 * r20;
    r74 = fmaf(r75, r36, r74);
    r74 = r74 + r66;
    r57 = r18 * r57;
    r57 = r57 * r61;
    r47 = r57 + r47;
    r47 = fmaf(r7, r47, r8 * r74);
    r32 = fmaf(r49, r32, r41);
    r32 = r32 + r79;
    r47 = fmaf(r9, r32, r47);
    r32 = r21 * r47;
    r32 = fmaf(r34, r32, r72 * r23);
    r23 = r0 * r47;
    r23 = fmaf(r48, r23, r64 * r63);
    r79 = r64 * r45;
    r23 = fmaf(r19, r79, r23);
    r49 = r0 * r28;
    r66 = r62 + r66;
    r66 = r66 + r68;
    r68 = r27 * r20;
    r62 = r20 * r30;
    r62 = fmaf(r24, r62, r75 * r68);
    r62 = r62 + r73;
    r62 = fmaf(r9, r62, r7 * r66);
    r69 = r57 + r69;
    r62 = fmaf(r8, r69, r62);
    r49 = r49 * r62;
    r23 = fmaf(r46, r49, r23);
    r49 = r26 * r23;
    r32 = fmaf(r29, r49, r32);
    r49 = r28 * r23;
    r49 = fmaf(r64, r56, r29 * r49);
    r79 = r21 * r62;
    r49 = fmaf(r34, r79, r49);
    r79 = r38 * r45;
    r69 = r0 * r5;
    r69 = r69 * r28;
    r69 = fmaf(r46, r69, r19 * r79);
    r79 = r0 * r43;
    r69 = fmaf(r48, r79, r69);
    r69 = fmaf(r38, r63, r69);
    r79 = r26 * r69;
    r8 = r43 * r21;
    r8 = fmaf(r34, r8, r29 * r79);
    r79 = r38 * r48;
    r8 = fmaf(r72, r79, r8);
    r79 = r5 * r21;
    r57 = r28 * r69;
    r57 = fmaf(r29, r57, r34 * r79);
    r57 = fmaf(r38, r56, r57);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r32,
                                          r49,
                                          r8,
                                          r57);
    r79 = r31 * r48;
    r9 = r31 * r45;
    r66 = r0 * r44;
    r66 = r66 * r28;
    r66 = fmaf(r46, r66, r19 * r9);
    r9 = r0 * r40;
    r66 = fmaf(r48, r9, r66);
    r66 = fmaf(r31, r63, r66);
    r9 = r26 * r66;
    r9 = fmaf(r29, r9, r72 * r79);
    r79 = r40 * r21;
    r9 = fmaf(r34, r79, r9);
    r79 = r28 * r66;
    r7 = r44 * r21;
    r7 = fmaf(r34, r7, r29 * r79);
    r7 = fmaf(r31, r56, r7);
    r79 = r0 * r37;
    r79 = r79 * r28;
    r73 = r42 * r45;
    r73 = fmaf(r19, r73, r46 * r79);
    r79 = r0 * r39;
    r73 = fmaf(r48, r79, r73);
    r73 = fmaf(r42, r63, r73);
    r63 = r26 * r73;
    r79 = r39 * r21;
    r79 = fmaf(r34, r79, r29 * r63);
    r63 = r42 * r48;
    r79 = fmaf(r72, r63, r79);
    r63 = r28 * r73;
    r72 = r37 * r21;
    r72 = fmaf(r34, r72, r29 * r63);
    r72 = fmaf(r42, r56, r72);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r9,
                                          r7,
                                          r79,
                                          r72);
    r56 = r4 * r3;
    r63 = r4 * r2;
    r63 = fmaf(r1, r63, r25 * r56);
    r56 = r4 * r3;
    r29 = r4 * r2;
    r29 = fmaf(r67, r29, r80 * r56);
    r56 = r4 * r3;
    r68 = r4 * r2;
    r68 = fmaf(r32, r68, r49 * r56);
    r56 = r4 * r2;
    r24 = r4 * r3;
    r24 = fmaf(r57, r24, r8 * r56);
    WriteSum4<float, float>((float*)inout_shared, r63, r29, r68, r24);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r4 * r2;
    r68 = r4 * r3;
    r68 = fmaf(r7, r68, r9 * r24);
    r24 = r4 * r2;
    r29 = r4 * r3;
    r29 = fmaf(r72, r29, r79 * r24);
    WriteSum2<float, float>((float*)inout_shared, r68, r29);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fmaf(r25, r25, r1 * r1);
    r68 = fmaf(r80, r80, r67 * r67);
    r24 = fmaf(r32, r32, r49 * r49);
    r63 = fmaf(r8, r8, r57 * r57);
    WriteSum4<float, float>((float*)inout_shared, r29, r68, r24, r63);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fmaf(r9, r9, r7 * r7);
    r24 = fmaf(r79, r79, r72 * r72);
    WriteSum2<float, float>((float*)inout_shared, r63, r24);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fmaf(r1, r67, r25 * r80);
    r63 = fmaf(r25, r49, r1 * r32);
    r68 = fmaf(r1, r8, r25 * r57);
    r29 = fmaf(r1, r9, r25 * r7);
    WriteSum4<float, float>((float*)inout_shared, r24, r63, r68, r29);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fmaf(r1, r79, r25 * r72);
    r25 = fmaf(r80, r49, r67 * r32);
    r29 = fmaf(r67, r8, r80 * r57);
    r68 = fmaf(r80, r7, r67 * r9);
    WriteSum4<float, float>((float*)inout_shared, r1, r25, r29, r68);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fmaf(r80, r72, r67 * r79);
    r67 = fmaf(r49, r57, r32 * r8);
    r68 = fmaf(r49, r7, r32 * r9);
    r49 = fmaf(r49, r72, r32 * r79);
    WriteSum4<float, float>((float*)inout_shared, r80, r67, r68, r49);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fmaf(r8, r9, r57 * r7);
    r57 = fmaf(r57, r72, r8 * r79);
    r79 = fmaf(r9, r79, r7 * r72);
    WriteSum3<float, float>((float*)inout_shared, r49, r57, r79);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = r26 * r21;
    r79 = r79 * r35;
    r57 = r28 * r21;
    r57 = r57 * r35;
    r49 = r26 * r6;
    r49 = r49 * r34;
    r9 = r28 * r6;
    r9 = r9 * r34;
    WriteIdx4<1024, float, float, float4>(
        out_focal_and_distortion_jac,
        0 * out_focal_and_distortion_jac_num_alloc,
        global_thread_idx,
        r79,
        r57,
        r49,
        r9);
    r9 = r28 * r3;
    r9 = r9 * r35;
    r49 = r26 * r2;
    r49 = r49 * r35;
    r49 = fmaf(r71, r49, r71 * r9);
    r9 = r4 * r26;
    r9 = r9 * r6;
    r9 = r9 * r2;
    r71 = r4 * r28;
    r71 = r71 * r6;
    r71 = r71 * r3;
    r71 = fmaf(r34, r71, r34 * r9);
    WriteSum2<float, float>((float*)inout_shared, r49, r71);
  };
  FlushSumShared<2, float>(out_focal_and_distortion_njtr,
                           0 * out_focal_and_distortion_njtr_num_alloc,
                           focal_and_distortion_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = r21 * r46;
    r71 = r71 * r19;
    r49 = r26 * r21;
    r49 = r49 * r21;
    r49 = fmaf(r48, r49, r21 * r71);
    r9 = r26 * r48;
    r34 = r33 * r33;
    r35 = r6 * r6;
    r34 = r34 * r35;
    r35 = r46 * r19;
    r35 = fmaf(r34, r35, r34 * r9);
    WriteSum2<float, float>((float*)inout_shared, r49, r35);
  };
  FlushSumShared<2, float>(out_focal_and_distortion_precond_diag,
                           0 * out_focal_and_distortion_precond_diag_num_alloc,
                           focal_and_distortion_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r33 * r26;
    r35 = r35 * r6;
    r35 = r35 * r21;
    r49 = r33 * r6;
    r49 = fmaf(r71, r49, r48 * r35);
    WriteSum1<float, float>((float*)inout_shared, r49);
  };
  FlushSumShared<1, float>(out_focal_and_distortion_precond_tril,
                           0 * out_focal_and_distortion_precond_tril_num_alloc,
                           focal_and_distortion_indices_loc,
                           (float*)inout_shared);
}

void SimpleRadialSplitFixedPrincipalPointFixedPointResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc,
    SharedIndex* focal_and_distortion_indices,
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
    float* out_focal_and_distortion_jac,
    unsigned int out_focal_and_distortion_jac_num_alloc,
    float* const out_focal_and_distortion_njtr,
    unsigned int out_focal_and_distortion_njtr_num_alloc,
    float* const out_focal_and_distortion_precond_diag,
    unsigned int out_focal_and_distortion_precond_diag_num_alloc,
    float* const out_focal_and_distortion_precond_tril,
    unsigned int out_focal_and_distortion_precond_tril_num_alloc,
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
      focal_and_distortion,
      focal_and_distortion_num_alloc,
      focal_and_distortion_indices,
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
      out_focal_and_distortion_jac,
      out_focal_and_distortion_jac_num_alloc,
      out_focal_and_distortion_njtr,
      out_focal_and_distortion_njtr_num_alloc,
      out_focal_and_distortion_precond_diag,
      out_focal_and_distortion_precond_diag_num_alloc,
      out_focal_and_distortion_precond_tril,
      out_focal_and_distortion_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar