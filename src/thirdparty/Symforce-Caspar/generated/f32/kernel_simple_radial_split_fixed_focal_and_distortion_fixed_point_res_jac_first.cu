#include "kernel_simple_radial_split_fixed_focal_and_distortion_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndDistortionFixedPointResJacFirstKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* out_pose_jac,
        unsigned int out_pose_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        float* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        float* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79;
  LoadShared<2, float, float>(principal_point,
                              0 * principal_point_num_alloc,
                              principal_point_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       principal_point_indices_loc[threadIdx.x].target,
                       r0,
                       r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r33 = 9.99999999999999955e-07;
    r29 = r20 * r22;
    r29 = fmaf(r30, r29, r31);
    r29 = fmaf(r7, r29, r6);
    r38 = fmaf(r20, r38, r37);
    r42 = r21 + r42;
    r37 = r14 * r14;
    r37 = r37 * r20;
    r42 = r42 + r37;
    r6 = r15 * r16;
    r6 = r6 * r0;
    r31 = r14 * r17;
    r31 = fmaf(r0, r31, r6);
    r45 = r0 * r18;
    r45 = r45 * r22;
    r46 = fmaf(r27, r32, r45);
    r47 = r27 * r27;
    r47 = r47 * r20;
    r25 = r47 + r25;
    r29 = fmaf(r34, r38, r29);
    r29 = fmaf(r36, r42, r29);
    r29 = fmaf(r35, r31, r29);
    r29 = fmaf(r8, r46, r29);
    r29 = fmaf(r9, r25, r29);
    r25 = copysign(1.0, r29);
    r25 = fmaf(r33, r25, r29);
    r33 = 1.0 / r25;
    ReadIdx2<1024, float, float, float2>(focal_and_distortion,
                                         0 * focal_and_distortion_num_alloc,
                                         global_thread_idx,
                                         r29,
                                         r46);
    r48 = r25 * r25;
    r49 = 1.0 / r48;
    r50 = r26 * r49;
    r28 = fmaf(r18, r32, r28);
    r28 = fmaf(r7, r28, r5);
    r5 = r16 * r17;
    r5 = fmaf(r0, r5, r41);
    r44 = r21 + r44;
    r44 = r44 + r37;
    r37 = r14 * r17;
    r37 = fmaf(r20, r37, r6);
    r6 = r27 * r20;
    r6 = fmaf(r30, r6, r45);
    r19 = r21 + r19;
    r19 = r19 + r47;
    r28 = fmaf(r34, r5, r28);
    r28 = fmaf(r35, r44, r28);
    r28 = fmaf(r36, r37, r28);
    r28 = fmaf(r9, r6, r28);
    r28 = fmaf(r8, r19, r28);
    r19 = r28 * r28;
    r6 = fmaf(r49, r19, r26 * r50);
    r6 = fmaf(r46, r6, r21);
    r6 = r29 * r6;
    r36 = r33 * r6;
    r2 = fmaf(r26, r36, r2);
    r3 = fmaf(r3, r4, r1);
    r3 = fmaf(r28, r36, r3);
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
    r1 = r10 * r17;
    r35 = -5.00000000000000000e-01;
    r34 = r13 * r14;
    r34 = fmaf(r35, r34, r35 * r1);
    r1 = r12 * r15;
    r34 = fmaf(r35, r1, r34);
    r47 = r11 * r16;
    r45 = 5.00000000000000000e-01;
    r34 = fmaf(r45, r47, r34);
    r47 = r22 * r34;
    r1 = r12 * r17;
    r41 = r11 * r14;
    r41 = fmaf(r45, r41, r45 * r1);
    r1 = r10 * r15;
    r41 = fmaf(r35, r1, r41);
    r51 = r13 * r45;
    r41 = fmaf(r16, r51, r41);
    r1 = fmaf(r41, r32, r0 * r47);
    r52 = r0 * r27;
    r53 = r13 * r15;
    r54 = r11 * r35;
    r53 = fmaf(r17, r54, r35 * r53);
    r53 = fmaf(r45, r24, r53);
    r53 = fmaf(r35, r23, r53);
    r55 = r0 * r18;
    r56 = r10 * r14;
    r57 = r12 * r16;
    r57 = fmaf(r35, r57, r35 * r56);
    r57 = fmaf(r17, r51, r57);
    r57 = fmaf(r15, r54, r57);
    r55 = r55 * r57;
    r52 = fmaf(r53, r52, r55);
    r1 = r1 + r52;
    r56 = r0 * r22;
    r56 = r56 * r57;
    r58 = r0 * r27;
    r58 = r58 * r41;
    r59 = r56 + r58;
    r60 = r18 * r20;
    r59 = fmaf(r34, r60, r59);
    r61 = r20 * r30;
    r59 = fmaf(r53, r61, r59);
    r59 = fmaf(r8, r59, r9 * r1);
    r1 = r22 * r41;
    r61 = -4.00000000000000000e+00;
    r1 = r1 * r61;
    r60 = r18 * r53;
    r62 = r61 * r60;
    r63 = r1 + r62;
    r59 = fmaf(r7, r63, r59);
    r63 = r0 * r22;
    r63 = r63 * r53;
    r64 = r0 * r18;
    r64 = fmaf(r41, r64, r63);
    r65 = r0 * r27;
    r65 = r65 * r34;
    r66 = r57 * r32;
    r67 = r65 + r66;
    r68 = r64 + r67;
    r69 = r20 * r30;
    r69 = fmaf(r20, r47, r41 * r69);
    r69 = r69 + r52;
    r69 = fmaf(r7, r69, r8 * r68);
    r68 = r27 * r61;
    r41 = r57 * r68;
    r1 = r1 + r41;
    r69 = fmaf(r9, r1, r69);
    r1 = r4 * r6;
    r1 = r1 * r50;
    r70 = fmaf(r69, r1, r59 * r36);
    r71 = r29 * r46;
    r72 = r0 * r28;
    r73 = r27 * r20;
    r74 = r20 * r30;
    r74 = r74 * r57;
    r73 = fmaf(r34, r73, r74);
    r73 = r73 + r64;
    r41 = r62 + r41;
    r41 = fmaf(r8, r41, r9 * r73);
    r58 = fmaf(r53, r32, r58);
    r73 = r0 * r18;
    r73 = fmaf(r34, r73, r56);
    r58 = r58 + r73;
    r41 = fmaf(r7, r58, r41);
    r72 = r72 * r41;
    r58 = r0 * r59;
    r58 = fmaf(r50, r58, r49 * r72);
    r48 = r25 * r48;
    r48 = 1.0 / r48;
    r48 = r20 * r48;
    r25 = r69 * r48;
    r58 = fmaf(r19, r25, r58);
    r72 = r26 * r26;
    r72 = r72 * r48;
    r58 = fmaf(r69, r72, r58);
    r71 = r71 * r58;
    r71 = r71 * r33;
    r70 = fmaf(r26, r71, r70);
    r58 = r4 * r28;
    r58 = r58 * r69;
    r58 = r58 * r49;
    r58 = fmaf(r6, r58, r41 * r36);
    r58 = fmaf(r28, r71, r58);
    r71 = r20 * r22;
    r71 = fmaf(r53, r71, r74);
    r41 = r0 * r18;
    r25 = r12 * r17;
    r56 = r10 * r15;
    r56 = fmaf(r45, r56, r35 * r25);
    r25 = r13 * r16;
    r56 = fmaf(r35, r25, r56);
    r56 = fmaf(r14, r54, r56);
    r41 = r41 * r56;
    r25 = r0 * r27;
    r62 = r10 * r17;
    r64 = r12 * r15;
    r64 = fmaf(r45, r64, r45 * r62);
    r64 = fmaf(r14, r51, r64);
    r64 = fmaf(r16, r54, r64);
    r25 = fmaf(r64, r25, r41);
    r71 = r71 + r25;
    r54 = r0 * r22;
    r54 = r54 * r64;
    r62 = fmaf(r56, r32, r54);
    r62 = r62 + r52;
    r62 = fmaf(r8, r62, r7 * r71);
    r71 = r22 * r57;
    r71 = r71 * r61;
    r52 = r56 * r68;
    r75 = r71 + r52;
    r62 = fmaf(r9, r75, r62);
    r66 = r63 + r66;
    r66 = r66 + r25;
    r25 = r18 * r61;
    r25 = r25 * r64;
    r71 = r71 + r25;
    r71 = fmaf(r7, r71, r9 * r66);
    r66 = r20 * r30;
    r66 = fmaf(r20, r60, r64 * r66);
    r63 = r0 * r27;
    r63 = r63 * r57;
    r75 = r0 * r22;
    r75 = fmaf(r56, r75, r63);
    r66 = r66 + r75;
    r71 = fmaf(r8, r66, r71);
    r66 = fmaf(r71, r36, r62 * r1);
    r76 = r29 * r46;
    r77 = r0 * r71;
    r77 = fmaf(r62, r72, r50 * r77);
    r78 = r62 * r48;
    r77 = fmaf(r19, r78, r77);
    r79 = r0 * r28;
    r54 = r55 + r54;
    r55 = r27 * r20;
    r54 = fmaf(r53, r55, r54);
    r53 = r20 * r30;
    r54 = fmaf(r56, r53, r54);
    r64 = fmaf(r64, r32, r0 * r60);
    r64 = r64 + r75;
    r64 = fmaf(r7, r64, r9 * r54);
    r52 = r25 + r52;
    r64 = fmaf(r8, r52, r64);
    r79 = r79 * r64;
    r77 = fmaf(r49, r79, r77);
    r76 = r76 * r26;
    r76 = r76 * r77;
    r66 = fmaf(r33, r76, r66);
    r76 = r29 * r46;
    r76 = r76 * r28;
    r76 = r76 * r77;
    r76 = fmaf(r33, r76, r64 * r36);
    r64 = r4 * r28;
    r64 = r64 * r62;
    r64 = r64 * r49;
    r76 = fmaf(r6, r64, r76);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r70,
                                          r58,
                                          r66,
                                          r76);
    r64 = r11 * r17;
    r24 = fmaf(r35, r24, r45 * r64);
    r24 = fmaf(r15, r51, r24);
    r24 = fmaf(r45, r23, r24);
    r68 = r24 * r68;
    r47 = r61 * r47;
    r23 = r68 + r47;
    r45 = r0 * r18;
    r45 = r45 * r24;
    r63 = r63 + r45;
    r51 = r20 * r22;
    r63 = fmaf(r56, r51, r63);
    r35 = r20 * r30;
    r63 = fmaf(r34, r35, r63);
    r63 = fmaf(r7, r63, r9 * r23);
    r23 = r0 * r27;
    r23 = fmaf(r24, r32, r56 * r23);
    r23 = r23 + r73;
    r63 = fmaf(r8, r23, r63);
    r74 = r65 + r74;
    r65 = r0 * r22;
    r65 = r65 * r24;
    r23 = r18 * r20;
    r74 = fmaf(r56, r23, r74);
    r74 = r74 + r65;
    r57 = r18 * r57;
    r57 = r57 * r61;
    r47 = r57 + r47;
    r47 = fmaf(r7, r47, r8 * r74);
    r32 = fmaf(r34, r32, r45);
    r32 = r32 + r75;
    r47 = fmaf(r9, r32, r47);
    r32 = fmaf(r47, r36, r63 * r1);
    r75 = r29 * r46;
    r34 = r0 * r47;
    r34 = fmaf(r50, r34, r63 * r72);
    r45 = r63 * r48;
    r34 = fmaf(r19, r45, r34);
    r74 = r0 * r28;
    r65 = r41 + r65;
    r65 = r65 + r67;
    r67 = r27 * r20;
    r41 = r20 * r30;
    r41 = fmaf(r24, r41, r56 * r67);
    r41 = r41 + r73;
    r41 = fmaf(r9, r41, r7 * r65);
    r68 = r57 + r68;
    r41 = fmaf(r8, r68, r41);
    r74 = r74 * r41;
    r34 = fmaf(r49, r74, r34);
    r75 = r75 * r26;
    r75 = r75 * r34;
    r32 = fmaf(r33, r75, r32);
    r75 = r29 * r46;
    r75 = r75 * r28;
    r75 = r75 * r34;
    r34 = r4 * r28;
    r34 = r34 * r63;
    r34 = r34 * r49;
    r34 = fmaf(r6, r34, r33 * r75);
    r34 = fmaf(r41, r36, r34);
    r41 = r29 * r46;
    r75 = r38 * r48;
    r74 = r0 * r5;
    r74 = r74 * r28;
    r74 = fmaf(r49, r74, r19 * r75);
    r75 = r0 * r43;
    r74 = fmaf(r50, r75, r74);
    r74 = fmaf(r38, r72, r74);
    r41 = r41 * r26;
    r41 = r41 * r74;
    r41 = fmaf(r43, r36, r33 * r41);
    r41 = fmaf(r38, r1, r41);
    r75 = r29 * r46;
    r75 = r75 * r28;
    r75 = r75 * r74;
    r75 = fmaf(r33, r75, r5 * r36);
    r74 = r4 * r38;
    r74 = r74 * r28;
    r74 = r74 * r49;
    r75 = fmaf(r6, r74, r75);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r32,
                                          r34,
                                          r41,
                                          r75);
    r74 = r29 * r46;
    r45 = r31 * r48;
    r68 = r0 * r44;
    r68 = r68 * r28;
    r68 = fmaf(r49, r68, r19 * r45);
    r45 = r0 * r40;
    r68 = fmaf(r50, r45, r68);
    r68 = fmaf(r31, r72, r68);
    r74 = r74 * r26;
    r74 = r74 * r68;
    r74 = fmaf(r33, r74, r31 * r1);
    r74 = fmaf(r40, r36, r74);
    r45 = r29 * r46;
    r45 = r45 * r28;
    r45 = r45 * r68;
    r45 = fmaf(r44, r36, r33 * r45);
    r68 = r4 * r31;
    r68 = r68 * r28;
    r68 = r68 * r49;
    r45 = fmaf(r6, r68, r45);
    r68 = r29 * r46;
    r8 = r0 * r37;
    r8 = r8 * r28;
    r57 = r42 * r48;
    r57 = fmaf(r19, r57, r49 * r8);
    r8 = r0 * r39;
    r57 = fmaf(r50, r8, r57);
    r57 = fmaf(r42, r72, r57);
    r68 = r68 * r26;
    r68 = r68 * r57;
    r68 = fmaf(r39, r36, r33 * r68);
    r68 = fmaf(r42, r1, r68);
    r1 = r29 * r46;
    r1 = r1 * r28;
    r1 = r1 * r57;
    r36 = fmaf(r37, r36, r33 * r1);
    r1 = r4 * r42;
    r1 = r1 * r28;
    r1 = r1 * r49;
    r36 = fmaf(r6, r1, r36);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r74,
                                          r45,
                                          r68,
                                          r36);
    r1 = r4 * r2;
    r3 = r4 * r3;
    r1 = fmaf(r58, r3, r70 * r1);
    r6 = r4 * r2;
    r6 = fmaf(r76, r3, r66 * r6);
    r49 = r4 * r2;
    r49 = fmaf(r34, r3, r32 * r49);
    r33 = r4 * r2;
    r33 = fmaf(r75, r3, r41 * r33);
    WriteSum4<float, float>((float*)inout_shared, r1, r6, r49, r33);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = r4 * r2;
    r33 = fmaf(r45, r3, r74 * r33);
    r49 = r4 * r2;
    r49 = fmaf(r36, r3, r68 * r49);
    WriteSum2<float, float>((float*)inout_shared, r33, r49);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fmaf(r58, r58, r70 * r70);
    r33 = fmaf(r76, r76, r66 * r66);
    r6 = fmaf(r32, r32, r34 * r34);
    r1 = fmaf(r41, r41, r75 * r75);
    WriteSum4<float, float>((float*)inout_shared, r49, r33, r6, r1);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fmaf(r74, r74, r45 * r45);
    r6 = fmaf(r68, r68, r36 * r36);
    WriteSum2<float, float>((float*)inout_shared, r1, r6);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fmaf(r70, r66, r58 * r76);
    r1 = fmaf(r58, r34, r70 * r32);
    r33 = fmaf(r70, r41, r58 * r75);
    r49 = fmaf(r70, r74, r58 * r45);
    WriteSum4<float, float>((float*)inout_shared, r6, r1, r33, r49);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fmaf(r70, r68, r58 * r36);
    r58 = fmaf(r76, r34, r66 * r32);
    r49 = fmaf(r66, r41, r76 * r75);
    r33 = fmaf(r76, r45, r66 * r74);
    WriteSum4<float, float>((float*)inout_shared, r70, r58, r49, r33);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = fmaf(r76, r36, r66 * r68);
    r66 = fmaf(r34, r75, r32 * r41);
    r33 = fmaf(r34, r45, r32 * r74);
    r34 = fmaf(r34, r36, r32 * r68);
    WriteSum4<float, float>((float*)inout_shared, r76, r66, r33, r34);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fmaf(r41, r74, r75 * r45);
    r75 = fmaf(r75, r36, r41 * r68);
    r68 = fmaf(r74, r68, r45 * r36);
    WriteSum3<float, float>((float*)inout_shared, r34, r75, r68);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r4 * r2;
    WriteSum2<float, float>((float*)inout_shared, r68, r3);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float*)inout_shared, r21, r21);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedFocalAndDistortionFixedPointResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedFocalAndDistortionFixedPointResJacFirstKernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              sensor_from_rig,
              sensor_from_rig_num_alloc,
              principal_point,
              principal_point_num_alloc,
              principal_point_indices,
              pixel,
              pixel_num_alloc,
              focal_and_distortion,
              focal_and_distortion_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_rTr,
              out_pose_jac,
              out_pose_jac_num_alloc,
              out_pose_njtr,
              out_pose_njtr_num_alloc,
              out_pose_precond_diag,
              out_pose_precond_diag_num_alloc,
              out_pose_precond_tril,
              out_pose_precond_tril_num_alloc,
              out_principal_point_jac,
              out_principal_point_jac_num_alloc,
              out_principal_point_njtr,
              out_principal_point_njtr_num_alloc,
              out_principal_point_precond_diag,
              out_principal_point_precond_diag_num_alloc,
              out_principal_point_precond_tril,
              out_principal_point_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar