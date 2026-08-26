#include "kernel_thin_prism_fisheye_split_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPrincipalPointResJacFirstKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
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
        float* out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        float* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        float* const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        float* const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
        float* out_point_jac,
        unsigned int out_point_jac_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        float* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
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
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104,
      r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116,
      r117, r118, r119, r120, r121, r122, r123, r124, r125, r126, r127, r128,
      r129, r130, r131, r132, r133, r134, r135, r136, r137, r138, r139, r140,
      r141, r142, r143, r144, r145, r146, r147, r148, r149, r150;

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
  };
  LoadShared<4, float, float>(focal_and_extra,
                              0 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r0,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  LoadShared<2, float, float>(focal_and_extra,
                              8 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r8,
                       r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r10,
                                         r11,
                                         r12);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r13,
                       r14,
                       r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = 2.00000000000000000e+00;
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r17,
                       r18,
                       r19,
                       r20);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r21,
                                         r22,
                                         r23,
                                         r24);
    r25 = fmaf(r20, r21, r17 * r24);
    r26 = r18 * r23;
    r25 = fmaf(r4, r26, r25);
    r25 = fmaf(r19, r22, r25);
    r26 = r16 * r25;
    r27 = r18 * r24;
    r28 = r20 * r22;
    r29 = r27 + r28;
    r30 = r17 * r23;
    r31 = r19 * r21;
    r29 = r29 + r30;
    r29 = fmaf(r4, r31, r29);
    r26 = r26 * r29;
    r32 = fmaf(r18, r21, r19 * r24);
    r33 = r17 * r22;
    r32 = fmaf(r4, r33, r32);
    r32 = fmaf(r20, r23, r32);
    r33 = r16 * r32;
    r34 = fmaf(r18, r22, r17 * r21);
    r34 = fmaf(r19, r23, r34);
    r34 = fmaf(r4, r34, r20 * r24);
    r33 = fmaf(r34, r33, r26);
    r11 = fmaf(r13, r33, r11);
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
    r38 = r21 * r22;
    r38 = r38 * r16;
    r39 = r23 * r24;
    r39 = fmaf(r16, r39, r38);
    r40 = r21 * r21;
    r41 = -2.00000000000000000e+00;
    r40 = r40 * r41;
    r42 = 1.00000000000000000e+00;
    r43 = r23 * r23;
    r43 = fmaf(r41, r43, r42);
    r44 = r40 + r43;
    r45 = r22 * r23;
    r45 = r45 * r16;
    r46 = r21 * r24;
    r46 = fmaf(r41, r46, r45);
    r47 = r16 * r32;
    r47 = r47 * r29;
    r48 = r25 * r41;
    r48 = fmaf(r34, r48, r47);
    r49 = r32 * r32;
    r49 = r49 * r41;
    r50 = r42 + r49;
    r51 = r25 * r25;
    r51 = r51 * r41;
    r50 = r50 + r51;
    r11 = fmaf(r35, r39, r11);
    r11 = fmaf(r36, r44, r11);
    r11 = fmaf(r37, r46, r11);
    r11 = fmaf(r15, r48, r11);
    r11 = fmaf(r14, r50, r11);
    r52 = 9.99999999999999955e-07;
    r53 = r41 * r29;
    r53 = r53 * r29;
    r54 = r42 + r53;
    r54 = r54 + r49;
    r10 = fmaf(r13, r54, r10);
    r49 = r32 * r41;
    r49 = fmaf(r34, r49, r26);
    r26 = r16 * r32;
    r26 = r26 * r25;
    r55 = r16 * r29;
    r55 = fmaf(r34, r55, r26);
    r56 = r21 * r23;
    r56 = r56 * r16;
    r57 = r22 * r24;
    r57 = fmaf(r16, r57, r56);
    r58 = r23 * r24;
    r58 = fmaf(r41, r58, r38);
    r38 = r22 * r22;
    r38 = r38 * r41;
    r43 = r38 + r43;
    r10 = fmaf(r14, r49, r10);
    r10 = fmaf(r15, r55, r10);
    r10 = fmaf(r37, r57, r10);
    r10 = fmaf(r36, r58, r10);
    r10 = fmaf(r35, r43, r10);
    r59 = r10 * r10;
    r60 = r41 * r29;
    r60 = fmaf(r34, r60, r26);
    r12 = fmaf(r13, r60, r12);
    r26 = r22 * r24;
    r26 = fmaf(r41, r26, r56);
    r38 = r42 + r38;
    r38 = r38 + r40;
    r40 = r21 * r24;
    r40 = fmaf(r16, r40, r45);
    r45 = r16 * r25;
    r45 = fmaf(r34, r45, r47);
    r53 = r42 + r53;
    r53 = r53 + r51;
    r12 = fmaf(r35, r26, r12);
    r12 = fmaf(r37, r38, r12);
    r12 = fmaf(r36, r40, r12);
    r12 = fmaf(r14, r45, r12);
    r12 = fmaf(r15, r53, r12);
    r36 = copysign(1.0, r12);
    r36 = fmaf(r52, r36, r12);
    r12 = r36 * r36;
    r37 = 1.0 / r12;
    r35 = r11 * r11;
    r35 = fmaf(r37, r35, r37 * r59);
    r59 = sqrtf(r35);
    r51 = copysign(1.0, r59);
    r51 = fmaf(r52, r51, r59);
    r52 = r51 * r51;
    r47 = 1.0 / r52;
    r59 = atanf(r59);
    r56 = r59 * r37;
    r61 = r47 * r56;
    r62 = r11 * r61;
    r63 = r11 * r59;
    r62 = r62 * r63;
    r64 = r10 * r10;
    r64 = r64 * r59;
    r64 = r64 * r61;
    r65 = r62 + r64;
  };
  LoadShared<4, float, float>(focal_and_extra,
                              4 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r66,
                       r67,
                       r68,
                       r69);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r70 = r10 * r10;
    r71 = 3.00000000000000000e+00;
    r70 = r70 * r59;
    r70 = r70 * r71;
    r70 = fmaf(r61, r70, r62);
    r62 = fmaf(r67, r70, r8 * r65);
    r72 = r66 * r10;
    r73 = r16 * r61;
    r74 = r63 * r73;
    r62 = fmaf(r74, r72, r62);
    r75 = r10 * r59;
    r76 = r65 * r65;
    r77 = r65 * r76;
    r78 = fmaf(r68, r77, r6 * r65);
    r79 = r76 * r76;
    r78 = fmaf(r69, r79, r78);
    r78 = fmaf(r7, r76, r78);
    r80 = 1.0 / r36;
    r81 = 1.0 / r51;
    r82 = r80 * r81;
    r83 = r78 * r82;
    r62 = fmaf(r83, r75, r62);
    r84 = r10 * r59;
    r62 = fmaf(r82, r84, r62);
    r84 = r0 * r62;
    r2 = r2 + r84;
    r75 = r11 * r71;
    r75 = r75 * r61;
    r75 = fmaf(r63, r75, r64);
    r64 = fmaf(r66, r75, r9 * r65);
    r72 = r67 * r74;
    r64 = fmaf(r63, r83, r64);
    r64 = fmaf(r10, r72, r64);
    r64 = fmaf(r82, r63, r64);
    r1 = fmaf(r5, r64, r1);
    r1 = fmaf(r3, r4, r1);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r1);
    r3 = fmaf(r1, r1, r2 * r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r3);
  if (global_thread_idx < problem_size) {
    r3 = r10 * r59;
    r85 = r16 * r34;
    r86 = r19 * r24;
    r87 = 5.00000000000000000e-01;
    r88 = r18 * r21;
    r88 = fmaf(r87, r88, r87 * r86);
    r86 = r17 * r22;
    r89 = -5.00000000000000000e-01;
    r88 = fmaf(r89, r86, r88);
    r90 = r20 * r23;
    r88 = fmaf(r87, r90, r88);
    r90 = r17 * r24;
    r86 = r20 * r21;
    r86 = fmaf(r89, r86, r89 * r90);
    r90 = r19 * r22;
    r86 = fmaf(r89, r90, r86);
    r91 = r18 * r23;
    r86 = fmaf(r87, r91, r86);
    r91 = r29 * r86;
    r85 = fmaf(r16, r91, r88 * r85);
    r90 = r16 * r25;
    r92 = fmaf(r87, r31, r89 * r27);
    r92 = fmaf(r89, r28, r92);
    r92 = fmaf(r89, r30, r92);
    r93 = r16 * r32;
    r94 = r20 * r24;
    r95 = r17 * r21;
    r95 = fmaf(r89, r95, r87 * r94);
    r94 = r18 * r22;
    r95 = fmaf(r89, r94, r95);
    r96 = r19 * r23;
    r95 = fmaf(r89, r96, r95);
    r93 = r93 * r95;
    r90 = fmaf(r92, r90, r93);
    r85 = r85 + r90;
    r96 = r16 * r29;
    r96 = r96 * r95;
    r94 = r16 * r25;
    r94 = r94 * r88;
    r97 = r96 + r94;
    r98 = r32 * r41;
    r97 = fmaf(r86, r98, r97);
    r99 = r41 * r34;
    r97 = fmaf(r92, r99, r97);
    r97 = fmaf(r14, r97, r15 * r85);
    r85 = r29 * r88;
    r99 = -4.00000000000000000e+00;
    r85 = r85 * r99;
    r98 = r32 * r92;
    r100 = r99 * r98;
    r101 = r85 + r100;
    r97 = fmaf(r13, r101, r97);
    r101 = 6.00000000000000000e+00;
    r3 = r3 * r97;
    r3 = r3 * r101;
    r102 = r16 * r11;
    r103 = r25 * r41;
    r104 = r34 * r95;
    r105 = r41 * r104;
    r103 = fmaf(r86, r103, r105);
    r106 = r16 * r29;
    r106 = r106 * r92;
    r107 = r16 * r32;
    r107 = fmaf(r88, r107, r106);
    r103 = r103 + r107;
    r108 = r25 * r95;
    r108 = r108 * r99;
    r100 = r108 + r100;
    r100 = fmaf(r14, r100, r15 * r103);
    r103 = r16 * r34;
    r103 = fmaf(r92, r103, r94);
    r94 = r16 * r32;
    r94 = fmaf(r86, r94, r96);
    r103 = r103 + r94;
    r100 = fmaf(r13, r103, r100);
    r102 = r102 * r100;
    r103 = r16 * r10;
    r103 = r103 * r97;
    r103 = fmaf(r37, r103, r37 * r102);
    r102 = r11 * r11;
    r96 = r16 * r25;
    r96 = r96 * r86;
    r104 = r16 * r104;
    r109 = r96 + r104;
    r107 = r107 + r109;
    r110 = r41 * r34;
    r110 = fmaf(r41, r91, r88 * r110);
    r110 = r110 + r90;
    r110 = fmaf(r13, r110, r14 * r107);
    r85 = r108 + r85;
    r110 = fmaf(r15, r85, r110);
    r12 = r36 * r12;
    r85 = 1.0 / r12;
    r108 = r41 * r85;
    r102 = r102 * r110;
    r103 = fmaf(r108, r102, r103);
    r107 = r10 * r10;
    r107 = r107 * r110;
    r103 = fmaf(r108, r107, r103);
    r107 = r71 * r103;
    r42 = r42 + r35;
    r42 = 1.0 / r42;
    r35 = rsqrtf(r35);
    r102 = r10 * r35;
    r88 = r42 * r102;
    r111 = r10 * r61;
    r107 = r107 * r88;
    r107 = fmaf(r111, r107, r61 * r3);
    r3 = r10 * r10;
    r112 = r59 * r59;
    r113 = -6.00000000000000000e+00;
    r112 = r112 * r110;
    r112 = r112 * r113;
    r112 = r112 * r47;
    r112 = r112 * r85;
    r114 = -3.00000000000000000e+00;
    r115 = r10 * r59;
    r116 = r102 * r115;
    r52 = r51 * r52;
    r117 = 1.0 / r52;
    r118 = r117 * r56;
    r116 = r116 * r118;
    r119 = r114 * r116;
    r120 = r11 * r11;
    r120 = r120 * r103;
    r120 = r120 * r35;
    r120 = r120 * r42;
    r120 = fmaf(r61, r120, r100 * r74);
    r121 = r4 * r103;
    r122 = r63 * r118;
    r123 = r11 * r35;
    r122 = r122 * r123;
    r120 = fmaf(r122, r121, r120);
    r124 = r11 * r11;
    r125 = r47 * r108;
    r126 = r59 * r59;
    r125 = r125 * r126;
    r124 = r124 * r110;
    r120 = fmaf(r125, r124, r120);
    r107 = fmaf(r112, r3, r107);
    r107 = fmaf(r103, r119, r107);
    r107 = r107 + r120;
    r124 = r97 * r73;
    r121 = r103 * r88;
    r121 = fmaf(r111, r121, r115 * r124);
    r124 = r10 * r10;
    r124 = r124 * r110;
    r121 = fmaf(r125, r124, r121);
    r127 = r4 * r103;
    r121 = fmaf(r116, r127, r121);
    r120 = r120 + r121;
    r107 = fmaf(r8, r120, r67 * r107);
    r127 = r59 * r89;
    r127 = r127 * r47;
    r127 = r127 * r80;
    r127 = r127 * r102;
    r124 = r78 * r127;
    r128 = r66 * r11;
    r128 = r128 * r103;
    r128 = r128 * r73;
    r107 = fmaf(r88, r128, r107);
    r129 = r66 * r10;
    r129 = r129 * r11;
    r129 = r129 * r59;
    r129 = r129 * r59;
    r129 = r129 * r99;
    r129 = r129 * r47;
    r129 = r129 * r85;
    r130 = r103 * r83;
    r131 = r87 * r88;
    r107 = fmaf(r131, r130, r107);
    r132 = r66 * r97;
    r107 = fmaf(r74, r132, r107);
    r133 = r59 * r97;
    r107 = fmaf(r82, r133, r107);
    r134 = r4 * r10;
    r134 = r134 * r110;
    r134 = r134 * r81;
    r107 = fmaf(r56, r134, r107);
    r135 = r4 * r10;
    r135 = r135 * r78;
    r135 = r135 * r110;
    r135 = r135 * r81;
    r107 = fmaf(r56, r135, r107);
    r136 = r82 * r131;
    r137 = r66 * r41;
    r137 = r137 * r103;
    r137 = r137 * r63;
    r137 = r137 * r102;
    r107 = fmaf(r118, r137, r107);
    r138 = r10 * r59;
    r139 = r7 * r16;
    r139 = r139 * r65;
    r139 = fmaf(r120, r139, r6 * r120);
    r140 = 4.00000000000000000e+00;
    r69 = r69 * r140;
    r69 = r69 * r77;
    r68 = r68 * r71;
    r68 = r68 * r76;
    r139 = fmaf(r120, r69, r139);
    r139 = fmaf(r120, r68, r139);
    r138 = r138 * r139;
    r107 = fmaf(r82, r138, r107);
    r141 = r66 * r100;
    r141 = r141 * r73;
    r107 = fmaf(r115, r141, r107);
    r142 = r59 * r97;
    r107 = fmaf(r83, r142, r107);
    r107 = fmaf(r103, r124, r107);
    r107 = fmaf(r103, r127, r107);
    r107 = fmaf(r110, r129, r107);
    r107 = fmaf(r103, r136, r107);
    r142 = r0 * r107;
    r141 = r100 * r101;
    r141 = r141 * r61;
    r138 = r11 * r11;
    r138 = r138 * r71;
    r138 = r138 * r103;
    r138 = r138 * r35;
    r138 = r138 * r42;
    r138 = fmaf(r61, r138, r63 * r141);
    r141 = r114 * r103;
    r138 = fmaf(r122, r141, r138);
    r137 = r11 * r11;
    r138 = fmaf(r112, r137, r138);
    r138 = r138 + r121;
    r120 = fmaf(r9, r120, r66 * r138);
    r138 = r11 * r87;
    r138 = r138 * r103;
    r138 = r138 * r35;
    r138 = r138 * r42;
    r120 = fmaf(r82, r138, r120);
    r121 = r11 * r59;
    r121 = r121 * r78;
    r121 = r121 * r89;
    r121 = r121 * r103;
    r121 = r121 * r47;
    r121 = r121 * r80;
    r120 = fmaf(r35, r121, r120);
    r112 = r139 * r82;
    r120 = fmaf(r63, r112, r120);
    r141 = r4 * r11;
    r141 = r141 * r78;
    r141 = r141 * r110;
    r141 = r141 * r81;
    r120 = fmaf(r56, r141, r120);
    r135 = r67 * r10;
    r135 = r135 * r11;
    r135 = r135 * r59;
    r135 = r135 * r59;
    r135 = r135 * r99;
    r135 = r135 * r110;
    r135 = r135 * r47;
    r120 = fmaf(r85, r135, r120);
    r134 = r67 * r11;
    r134 = r134 * r103;
    r134 = r134 * r73;
    r120 = fmaf(r88, r134, r120);
    r133 = r4 * r11;
    r133 = r133 * r110;
    r133 = r133 * r81;
    r120 = fmaf(r56, r133, r120);
    r110 = r67 * r41;
    r110 = r110 * r103;
    r110 = r110 * r63;
    r110 = r110 * r102;
    r120 = fmaf(r118, r110, r120);
    r132 = r87 * r42;
    r132 = r132 * r83;
    r132 = r132 * r123;
    r123 = r67 * r100;
    r123 = r123 * r73;
    r120 = fmaf(r115, r123, r120);
    r130 = r11 * r59;
    r130 = r130 * r89;
    r130 = r130 * r103;
    r130 = r130 * r47;
    r130 = r130 * r80;
    r120 = fmaf(r35, r130, r120);
    r128 = r59 * r100;
    r120 = fmaf(r83, r128, r120);
    r143 = r59 * r100;
    r120 = fmaf(r82, r143, r120);
    r120 = fmaf(r97, r72, r120);
    r120 = fmaf(r103, r132, r120);
    r143 = r5 * r120;
    r128 = r16 * r10;
    r104 = r106 + r104;
    r106 = r16 * r32;
    r130 = r19 * r24;
    r123 = r18 * r21;
    r123 = fmaf(r89, r123, r89 * r130);
    r130 = r17 * r22;
    r123 = fmaf(r87, r130, r123);
    r110 = r20 * r23;
    r123 = fmaf(r89, r110, r123);
    r106 = r106 * r123;
    r110 = r16 * r25;
    r130 = r17 * r24;
    r133 = r20 * r21;
    r133 = fmaf(r87, r133, r87 * r130);
    r130 = r19 * r22;
    r133 = fmaf(r87, r130, r133);
    r134 = r18 * r23;
    r133 = fmaf(r89, r134, r133);
    r110 = fmaf(r133, r110, r106);
    r104 = r104 + r110;
    r134 = r32 * r99;
    r134 = r134 * r133;
    r130 = r29 * r95;
    r130 = r130 * r99;
    r135 = r134 + r130;
    r135 = fmaf(r13, r135, r15 * r104);
    r104 = r41 * r34;
    r104 = fmaf(r41, r98, r133 * r104);
    r141 = r16 * r25;
    r141 = r141 * r95;
    r112 = r16 * r29;
    r112 = fmaf(r123, r112, r141);
    r104 = r104 + r112;
    r135 = fmaf(r14, r104, r135);
    r128 = r128 * r135;
    r104 = r10 * r10;
    r121 = r41 * r29;
    r121 = fmaf(r92, r121, r105);
    r121 = r121 + r110;
    r110 = r16 * r29;
    r110 = r110 * r133;
    r138 = r16 * r34;
    r138 = fmaf(r123, r138, r110);
    r138 = r138 + r90;
    r138 = fmaf(r14, r138, r13 * r121);
    r121 = r25 * r123;
    r90 = r99 * r121;
    r130 = r130 + r90;
    r138 = fmaf(r15, r130, r138);
    r104 = r104 * r138;
    r104 = fmaf(r108, r104, r37 * r128);
    r128 = r11 * r11;
    r128 = r128 * r138;
    r104 = fmaf(r108, r128, r104);
    r130 = r16 * r11;
    r144 = r25 * r41;
    r144 = fmaf(r92, r144, r93);
    r93 = r41 * r34;
    r144 = fmaf(r123, r93, r144);
    r144 = r144 + r110;
    r93 = r16 * r34;
    r98 = fmaf(r16, r98, r133 * r93);
    r98 = r98 + r112;
    r98 = fmaf(r13, r98, r15 * r144);
    r90 = r134 + r90;
    r98 = fmaf(r14, r90, r98);
    r130 = r130 * r98;
    r104 = fmaf(r37, r130, r104);
    r130 = r104 * r88;
    r128 = r4 * r104;
    r128 = fmaf(r116, r128, r111 * r130);
    r130 = r10 * r10;
    r130 = r130 * r138;
    r128 = fmaf(r125, r130, r128);
    r90 = r135 * r73;
    r128 = fmaf(r115, r90, r128);
    r90 = r4 * r104;
    r90 = fmaf(r122, r90, r98 * r74);
    r130 = r11 * r11;
    r130 = r130 * r138;
    r90 = fmaf(r125, r130, r90);
    r134 = r11 * r11;
    r134 = r134 * r104;
    r134 = r134 * r35;
    r134 = r134 * r42;
    r90 = fmaf(r61, r134, r90);
    r134 = r128 + r90;
    r130 = r71 * r104;
    r130 = r130 * r88;
    r130 = fmaf(r104, r119, r111 * r130);
    r144 = r10 * r10;
    r144 = r144 * r59;
    r144 = r144 * r59;
    r144 = r144 * r113;
    r144 = r144 * r138;
    r144 = r144 * r47;
    r130 = fmaf(r85, r144, r130);
    r93 = r10 * r59;
    r93 = r93 * r101;
    r93 = r93 * r135;
    r130 = fmaf(r61, r93, r130);
    r130 = r130 + r90;
    r130 = fmaf(r67, r130, r8 * r134);
    r90 = r66 * r98;
    r90 = r90 * r73;
    r130 = fmaf(r115, r90, r130);
    r93 = r4 * r10;
    r93 = r93 * r138;
    r93 = r93 * r81;
    r130 = fmaf(r56, r93, r130);
    r144 = r104 * r83;
    r130 = fmaf(r131, r144, r130);
    r133 = r59 * r135;
    r130 = fmaf(r83, r133, r130);
    r110 = r66 * r41;
    r110 = r110 * r104;
    r110 = r110 * r63;
    r110 = r110 * r102;
    r130 = fmaf(r118, r110, r130);
    r92 = r66 * r11;
    r92 = r92 * r104;
    r92 = r92 * r73;
    r130 = fmaf(r88, r92, r130);
    r145 = r4 * r10;
    r145 = r145 * r78;
    r145 = r145 * r138;
    r145 = r145 * r81;
    r130 = fmaf(r56, r145, r130);
    r146 = r66 * r135;
    r130 = fmaf(r74, r146, r130);
    r147 = r10 * r59;
    r148 = r7 * r16;
    r148 = r148 * r65;
    r148 = fmaf(r134, r148, r6 * r134);
    r148 = fmaf(r134, r69, r148);
    r148 = fmaf(r134, r68, r148);
    r147 = r147 * r148;
    r130 = fmaf(r82, r147, r130);
    r149 = r59 * r135;
    r130 = fmaf(r82, r149, r130);
    r130 = fmaf(r138, r129, r130);
    r130 = fmaf(r104, r127, r130);
    r130 = fmaf(r104, r136, r130);
    r130 = fmaf(r104, r124, r130);
    r149 = r0 * r130;
    r147 = r101 * r98;
    r147 = r147 * r61;
    r146 = r114 * r104;
    r146 = fmaf(r122, r146, r63 * r147);
    r147 = r11 * r11;
    r147 = r147 * r59;
    r147 = r147 * r59;
    r147 = r147 * r113;
    r147 = r147 * r138;
    r147 = r147 * r47;
    r146 = fmaf(r85, r147, r146);
    r145 = r11 * r11;
    r145 = r145 * r71;
    r145 = r145 * r104;
    r145 = r145 * r35;
    r145 = r145 * r42;
    r146 = fmaf(r61, r145, r146);
    r146 = r146 + r128;
    r146 = fmaf(r66, r146, r9 * r134);
    r134 = r4 * r11;
    r134 = r134 * r138;
    r134 = r134 * r81;
    r146 = fmaf(r56, r134, r146);
    r128 = r67 * r98;
    r128 = r128 * r73;
    r146 = fmaf(r115, r128, r146);
    r145 = r67 * r10;
    r145 = r145 * r11;
    r145 = r145 * r59;
    r145 = r145 * r59;
    r145 = r145 * r99;
    r145 = r145 * r138;
    r145 = r145 * r47;
    r146 = fmaf(r85, r145, r146);
    r147 = r11 * r59;
    r147 = r147 * r78;
    r147 = r147 * r89;
    r147 = r147 * r104;
    r147 = r147 * r47;
    r147 = r147 * r80;
    r146 = fmaf(r35, r147, r146);
    r92 = r148 * r82;
    r146 = fmaf(r63, r92, r146);
    r110 = r11 * r87;
    r110 = r110 * r104;
    r110 = r110 * r35;
    r110 = r110 * r42;
    r146 = fmaf(r82, r110, r146);
    r133 = r11 * r59;
    r133 = r133 * r89;
    r133 = r133 * r104;
    r133 = r133 * r47;
    r133 = r133 * r80;
    r146 = fmaf(r35, r133, r146);
    r144 = r67 * r41;
    r144 = r144 * r104;
    r144 = r144 * r63;
    r144 = r144 * r102;
    r146 = fmaf(r118, r144, r146);
    r93 = r67 * r11;
    r93 = r93 * r104;
    r93 = r93 * r73;
    r146 = fmaf(r88, r93, r146);
    r90 = r59 * r98;
    r146 = fmaf(r83, r90, r146);
    r150 = r4 * r11;
    r150 = r150 * r78;
    r150 = r150 * r138;
    r150 = r150 * r81;
    r146 = fmaf(r56, r150, r146);
    r138 = r59 * r98;
    r146 = fmaf(r82, r138, r146);
    r146 = fmaf(r104, r132, r146);
    r146 = fmaf(r135, r72, r146);
    r138 = r5 * r146;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r142,
                                          r143,
                                          r149,
                                          r138);
    r138 = r11 * r11;
    r149 = r10 * r10;
    r143 = r25 * r99;
    r31 = fmaf(r89, r31, r87 * r27);
    r31 = fmaf(r87, r28, r31);
    r31 = fmaf(r87, r30, r31);
    r143 = r143 * r31;
    r91 = r99 * r91;
    r30 = r143 + r91;
    r28 = r16 * r32;
    r28 = r28 * r31;
    r141 = r141 + r28;
    r27 = r41 * r29;
    r141 = fmaf(r123, r27, r141);
    r142 = r41 * r34;
    r141 = fmaf(r86, r142, r141);
    r141 = fmaf(r13, r141, r15 * r30);
    r30 = r16 * r34;
    r30 = fmaf(r16, r121, r31 * r30);
    r30 = r30 + r94;
    r141 = fmaf(r14, r30, r141);
    r149 = r149 * r141;
    r30 = r16 * r10;
    r142 = r16 * r29;
    r142 = r142 * r31;
    r96 = r96 + r142;
    r27 = r32 * r41;
    r96 = fmaf(r123, r27, r96);
    r96 = r96 + r105;
    r95 = r32 * r95;
    r95 = r95 * r99;
    r91 = r95 + r91;
    r91 = fmaf(r13, r91, r14 * r96);
    r96 = r16 * r34;
    r96 = fmaf(r86, r96, r28);
    r96 = r96 + r112;
    r91 = fmaf(r15, r96, r91);
    r30 = r30 * r91;
    r30 = fmaf(r37, r30, r108 * r149);
    r149 = r11 * r11;
    r149 = r149 * r141;
    r30 = fmaf(r108, r149, r30);
    r96 = r16 * r11;
    r142 = r106 + r142;
    r142 = r142 + r109;
    r109 = r41 * r34;
    r121 = fmaf(r41, r121, r31 * r109);
    r121 = r121 + r94;
    r121 = fmaf(r15, r121, r13 * r142);
    r95 = r143 + r95;
    r121 = fmaf(r14, r95, r121);
    r96 = r96 * r121;
    r30 = fmaf(r37, r96, r30);
    r138 = r138 * r30;
    r138 = r138 * r35;
    r138 = r138 * r42;
    r138 = fmaf(r121, r74, r61 * r138);
    r96 = r11 * r11;
    r96 = r96 * r141;
    r138 = fmaf(r125, r96, r138);
    r149 = r4 * r30;
    r138 = fmaf(r122, r149, r138);
    r149 = r10 * r10;
    r149 = r149 * r141;
    r96 = r30 * r88;
    r96 = fmaf(r111, r96, r125 * r149);
    r149 = r4 * r30;
    r96 = fmaf(r116, r149, r96);
    r95 = r91 * r73;
    r96 = fmaf(r115, r95, r96);
    r95 = r138 + r96;
    r149 = r10 * r10;
    r149 = r149 * r59;
    r149 = r149 * r59;
    r149 = r149 * r113;
    r149 = r149 * r141;
    r149 = r149 * r47;
    r14 = r71 * r30;
    r14 = r14 * r88;
    r14 = fmaf(r111, r14, r85 * r149);
    r149 = r10 * r59;
    r149 = r149 * r101;
    r149 = r149 * r91;
    r14 = fmaf(r61, r149, r14);
    r14 = fmaf(r30, r119, r14);
    r14 = r14 + r138;
    r14 = fmaf(r67, r14, r8 * r95);
    r138 = r10 * r59;
    r149 = r7 * r16;
    r149 = r149 * r65;
    r149 = fmaf(r95, r149, r6 * r95);
    r149 = fmaf(r95, r68, r149);
    r149 = fmaf(r95, r69, r149);
    r138 = r138 * r149;
    r14 = fmaf(r82, r138, r14);
    r143 = r4 * r10;
    r143 = r143 * r141;
    r143 = r143 * r81;
    r14 = fmaf(r56, r143, r14);
    r15 = r66 * r41;
    r15 = r15 * r30;
    r15 = r15 * r63;
    r15 = r15 * r102;
    r14 = fmaf(r118, r15, r14);
    r142 = r59 * r91;
    r14 = fmaf(r83, r142, r14);
    r13 = r66 * r91;
    r14 = fmaf(r74, r13, r14);
    r94 = r30 * r83;
    r14 = fmaf(r131, r94, r14);
    r109 = r66 * r121;
    r109 = r109 * r73;
    r14 = fmaf(r115, r109, r14);
    r31 = r4 * r10;
    r31 = r31 * r78;
    r31 = r31 * r141;
    r31 = r31 * r81;
    r14 = fmaf(r56, r31, r14);
    r106 = r59 * r91;
    r14 = fmaf(r82, r106, r14);
    r112 = r66 * r11;
    r112 = r112 * r30;
    r112 = r112 * r73;
    r14 = fmaf(r88, r112, r14);
    r14 = fmaf(r141, r129, r14);
    r14 = fmaf(r30, r136, r14);
    r14 = fmaf(r30, r124, r14);
    r14 = fmaf(r30, r127, r14);
    r112 = r0 * r14;
    r106 = r11 * r11;
    r106 = r106 * r71;
    r106 = r106 * r30;
    r106 = r106 * r35;
    r106 = r106 * r42;
    r31 = r101 * r121;
    r31 = r31 * r61;
    r31 = fmaf(r63, r31, r61 * r106);
    r106 = r11 * r11;
    r106 = r106 * r59;
    r106 = r106 * r59;
    r106 = r106 * r113;
    r106 = r106 * r141;
    r106 = r106 * r47;
    r31 = fmaf(r85, r106, r31);
    r109 = r114 * r30;
    r31 = fmaf(r122, r109, r31);
    r31 = r31 + r96;
    r31 = fmaf(r66, r31, r9 * r95);
    r95 = r67 * r10;
    r95 = r95 * r11;
    r95 = r95 * r59;
    r95 = r95 * r59;
    r95 = r95 * r99;
    r95 = r95 * r141;
    r95 = r95 * r47;
    r31 = fmaf(r85, r95, r31);
    r96 = r11 * r59;
    r96 = r96 * r89;
    r96 = r96 * r30;
    r96 = r96 * r47;
    r96 = r96 * r80;
    r31 = fmaf(r35, r96, r31);
    r109 = r11 * r87;
    r109 = r109 * r30;
    r109 = r109 * r35;
    r109 = r109 * r42;
    r31 = fmaf(r82, r109, r31);
    r106 = r4 * r11;
    r106 = r106 * r78;
    r106 = r106 * r141;
    r106 = r106 * r81;
    r31 = fmaf(r56, r106, r31);
    r94 = r59 * r121;
    r31 = fmaf(r83, r94, r31);
    r13 = r67 * r41;
    r13 = r13 * r30;
    r13 = r13 * r63;
    r13 = r13 * r102;
    r31 = fmaf(r118, r13, r31);
    r142 = r11 * r59;
    r142 = r142 * r78;
    r142 = r142 * r89;
    r142 = r142 * r30;
    r142 = r142 * r47;
    r142 = r142 * r80;
    r31 = fmaf(r35, r142, r31);
    r15 = r67 * r121;
    r15 = r15 * r73;
    r31 = fmaf(r115, r15, r31);
    r143 = r4 * r11;
    r143 = r143 * r141;
    r143 = r143 * r81;
    r31 = fmaf(r56, r143, r31);
    r141 = r67 * r11;
    r141 = r141 * r30;
    r141 = r141 * r73;
    r31 = fmaf(r88, r141, r31);
    r138 = r59 * r121;
    r31 = fmaf(r82, r138, r31);
    r28 = r149 * r82;
    r31 = fmaf(r63, r28, r31);
    r31 = fmaf(r91, r72, r31);
    r31 = fmaf(r30, r132, r31);
    r28 = r5 * r31;
    r138 = r26 * r10;
    r138 = r138 * r10;
    r138 = r138 * r59;
    r138 = r138 * r59;
    r138 = r138 * r113;
    r138 = r138 * r47;
    r141 = r43 * r10;
    r141 = r141 * r59;
    r141 = r141 * r101;
    r141 = fmaf(r61, r141, r85 * r138);
    r138 = r26 * r11;
    r138 = r138 * r11;
    r143 = r16 * r39;
    r143 = r143 * r11;
    r143 = fmaf(r37, r143, r108 * r138);
    r138 = r26 * r10;
    r138 = r138 * r10;
    r143 = fmaf(r108, r138, r143);
    r15 = r16 * r43;
    r15 = r15 * r10;
    r143 = fmaf(r37, r15, r143);
    r15 = r71 * r143;
    r15 = r15 * r88;
    r141 = fmaf(r111, r15, r141);
    r138 = r26 * r11;
    r138 = r138 * r11;
    r138 = fmaf(r39, r74, r125 * r138);
    r142 = r4 * r143;
    r138 = fmaf(r122, r142, r138);
    r13 = r11 * r11;
    r13 = r13 * r143;
    r13 = r13 * r35;
    r13 = r13 * r42;
    r138 = fmaf(r61, r13, r138);
    r141 = fmaf(r143, r119, r141);
    r141 = r141 + r138;
    r15 = r26 * r10;
    r15 = r15 * r10;
    r13 = r43 * r73;
    r13 = fmaf(r115, r13, r125 * r15);
    r15 = r4 * r143;
    r13 = fmaf(r116, r15, r13);
    r142 = r143 * r88;
    r13 = fmaf(r111, r142, r13);
    r138 = r138 + r13;
    r141 = fmaf(r8, r138, r67 * r141);
    r142 = r4 * r26;
    r142 = r142 * r10;
    r142 = r142 * r81;
    r141 = fmaf(r56, r142, r141);
    r15 = r43 * r59;
    r141 = fmaf(r82, r15, r141);
    r94 = r66 * r41;
    r94 = r94 * r143;
    r94 = r94 * r63;
    r94 = r94 * r102;
    r141 = fmaf(r118, r94, r141);
    r106 = r43 * r59;
    r141 = fmaf(r83, r106, r141);
    r109 = r66 * r11;
    r109 = r109 * r143;
    r109 = r109 * r73;
    r141 = fmaf(r88, r109, r141);
    r96 = r66 * r43;
    r141 = fmaf(r74, r96, r141);
    r95 = r143 * r83;
    r141 = fmaf(r131, r95, r141);
    r86 = r66 * r39;
    r86 = r86 * r73;
    r141 = fmaf(r115, r86, r141);
    r105 = r10 * r59;
    r27 = r7 * r16;
    r27 = r27 * r65;
    r27 = fmaf(r6, r138, r138 * r27);
    r27 = fmaf(r138, r69, r27);
    r27 = fmaf(r138, r68, r27);
    r105 = r105 * r27;
    r141 = fmaf(r82, r105, r141);
    r123 = r4 * r26;
    r123 = r123 * r10;
    r123 = r123 * r78;
    r123 = r123 * r81;
    r141 = fmaf(r56, r123, r141);
    r141 = fmaf(r143, r124, r141);
    r141 = fmaf(r143, r127, r141);
    r141 = fmaf(r26, r129, r141);
    r141 = fmaf(r143, r136, r141);
    r123 = r0 * r141;
    r105 = r26 * r11;
    r105 = r105 * r11;
    r105 = r105 * r59;
    r105 = r105 * r59;
    r105 = r105 * r113;
    r105 = r105 * r47;
    r86 = r39 * r101;
    r86 = r86 * r61;
    r86 = fmaf(r63, r86, r85 * r105);
    r105 = r114 * r143;
    r86 = fmaf(r122, r105, r86);
    r95 = r11 * r11;
    r95 = r95 * r71;
    r95 = r95 * r143;
    r95 = r95 * r35;
    r95 = r95 * r42;
    r86 = fmaf(r61, r95, r86);
    r86 = r86 + r13;
    r138 = fmaf(r9, r138, r66 * r86);
    r86 = r4 * r26;
    r86 = r86 * r11;
    r86 = r86 * r81;
    r138 = fmaf(r56, r86, r138);
    r13 = r4 * r26;
    r13 = r13 * r11;
    r13 = r13 * r78;
    r13 = r13 * r81;
    r138 = fmaf(r56, r13, r138);
    r95 = r11 * r59;
    r95 = r95 * r89;
    r95 = r95 * r143;
    r95 = r95 * r47;
    r95 = r95 * r80;
    r138 = fmaf(r35, r95, r138);
    r105 = r11 * r59;
    r105 = r105 * r78;
    r105 = r105 * r89;
    r105 = r105 * r143;
    r105 = r105 * r47;
    r105 = r105 * r80;
    r138 = fmaf(r35, r105, r138);
    r96 = r39 * r59;
    r138 = fmaf(r82, r96, r138);
    r109 = r39 * r59;
    r138 = fmaf(r83, r109, r138);
    r106 = r67 * r41;
    r106 = r106 * r143;
    r106 = r106 * r63;
    r106 = r106 * r102;
    r138 = fmaf(r118, r106, r138);
    r94 = r67 * r11;
    r94 = r94 * r143;
    r94 = r94 * r73;
    r138 = fmaf(r88, r94, r138);
    r15 = r11 * r87;
    r15 = r15 * r143;
    r15 = r15 * r35;
    r15 = r15 * r42;
    r138 = fmaf(r82, r15, r138);
    r142 = r67 * r26;
    r142 = r142 * r10;
    r142 = r142 * r11;
    r142 = r142 * r59;
    r142 = r142 * r59;
    r142 = r142 * r99;
    r142 = r142 * r47;
    r138 = fmaf(r85, r142, r138);
    r150 = r67 * r39;
    r150 = r150 * r73;
    r138 = fmaf(r115, r150, r138);
    r90 = r27 * r82;
    r138 = fmaf(r63, r90, r138);
    r138 = fmaf(r43, r72, r138);
    r138 = fmaf(r143, r132, r138);
    r90 = r5 * r138;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r112,
                                          r28,
                                          r123,
                                          r90);
    r90 = r11 * r11;
    r123 = r40 * r11;
    r123 = r123 * r11;
    r28 = r16 * r44;
    r28 = r28 * r11;
    r28 = fmaf(r37, r28, r108 * r123);
    r123 = r40 * r10;
    r123 = r123 * r10;
    r28 = fmaf(r108, r123, r28);
    r112 = r16 * r58;
    r112 = r112 * r10;
    r28 = fmaf(r37, r112, r28);
    r90 = r90 * r28;
    r90 = r90 * r35;
    r90 = r90 * r42;
    r112 = r40 * r125;
    r137 = fmaf(r112, r137, r61 * r90);
    r90 = r4 * r28;
    r137 = fmaf(r122, r90, r137);
    r137 = fmaf(r44, r74, r137);
    r90 = r28 * r88;
    r3 = fmaf(r112, r3, r111 * r90);
    r112 = r58 * r73;
    r3 = fmaf(r115, r112, r3);
    r90 = r4 * r28;
    r3 = fmaf(r116, r90, r3);
    r90 = r137 + r3;
    r112 = r71 * r28;
    r112 = r112 * r88;
    r123 = r40 * r10;
    r123 = r123 * r10;
    r123 = r123 * r59;
    r123 = r123 * r59;
    r123 = r123 * r113;
    r123 = r123 * r47;
    r123 = fmaf(r85, r123, r111 * r112);
    r112 = r58 * r10;
    r112 = r112 * r59;
    r112 = r112 * r101;
    r123 = fmaf(r61, r112, r123);
    r123 = fmaf(r28, r119, r123);
    r123 = r123 + r137;
    r123 = fmaf(r67, r123, r8 * r90);
    r137 = r58 * r59;
    r123 = fmaf(r83, r137, r123);
    r112 = r10 * r59;
    r150 = r7 * r16;
    r150 = r150 * r65;
    r150 = fmaf(r90, r150, r6 * r90);
    r150 = fmaf(r90, r69, r150);
    r150 = fmaf(r90, r68, r150);
    r112 = r112 * r150;
    r123 = fmaf(r82, r112, r123);
    r142 = r66 * r44;
    r142 = r142 * r73;
    r123 = fmaf(r115, r142, r123);
    r15 = r58 * r59;
    r123 = fmaf(r82, r15, r123);
    r94 = r66 * r11;
    r94 = r94 * r28;
    r94 = r94 * r73;
    r123 = fmaf(r88, r94, r123);
    r106 = r66 * r41;
    r106 = r106 * r28;
    r106 = r106 * r63;
    r106 = r106 * r102;
    r123 = fmaf(r118, r106, r123);
    r109 = r28 * r83;
    r123 = fmaf(r131, r109, r123);
    r96 = r4 * r40;
    r96 = r96 * r10;
    r96 = r96 * r81;
    r123 = fmaf(r56, r96, r123);
    r105 = r66 * r58;
    r123 = fmaf(r74, r105, r123);
    r95 = r4 * r40;
    r95 = r95 * r10;
    r95 = r95 * r78;
    r95 = r95 * r81;
    r123 = fmaf(r56, r95, r123);
    r123 = fmaf(r28, r127, r123);
    r123 = fmaf(r28, r136, r123);
    r123 = fmaf(r40, r129, r123);
    r123 = fmaf(r28, r124, r123);
    r95 = r0 * r123;
    r105 = r11 * r11;
    r105 = r105 * r71;
    r105 = r105 * r28;
    r105 = r105 * r35;
    r105 = r105 * r42;
    r96 = r40 * r11;
    r96 = r96 * r11;
    r96 = r96 * r59;
    r96 = r96 * r59;
    r96 = r96 * r113;
    r96 = r96 * r47;
    r96 = fmaf(r85, r96, r61 * r105);
    r105 = r44 * r101;
    r105 = r105 * r61;
    r96 = fmaf(r63, r105, r96);
    r109 = r114 * r28;
    r96 = fmaf(r122, r109, r96);
    r96 = r96 + r3;
    r96 = fmaf(r66, r96, r9 * r90);
    r90 = r67 * r44;
    r90 = r90 * r73;
    r96 = fmaf(r115, r90, r96);
    r3 = r4 * r40;
    r3 = r3 * r11;
    r3 = r3 * r81;
    r96 = fmaf(r56, r3, r96);
    r109 = r11 * r87;
    r109 = r109 * r28;
    r109 = r109 * r35;
    r109 = r109 * r42;
    r96 = fmaf(r82, r109, r96);
    r105 = r67 * r11;
    r105 = r105 * r28;
    r105 = r105 * r73;
    r96 = fmaf(r88, r105, r96);
    r106 = r44 * r59;
    r96 = fmaf(r83, r106, r96);
    r94 = r67 * r40;
    r94 = r94 * r10;
    r94 = r94 * r11;
    r94 = r94 * r59;
    r94 = r94 * r59;
    r94 = r94 * r99;
    r94 = r94 * r47;
    r96 = fmaf(r85, r94, r96);
    r15 = r67 * r41;
    r15 = r15 * r28;
    r15 = r15 * r63;
    r15 = r15 * r102;
    r96 = fmaf(r118, r15, r96);
    r142 = r44 * r59;
    r96 = fmaf(r82, r142, r96);
    r112 = r4 * r40;
    r112 = r112 * r11;
    r112 = r112 * r78;
    r112 = r112 * r81;
    r96 = fmaf(r56, r112, r96);
    r137 = r11 * r59;
    r137 = r137 * r78;
    r137 = r137 * r89;
    r137 = r137 * r28;
    r137 = r137 * r47;
    r137 = r137 * r80;
    r96 = fmaf(r35, r137, r96);
    r13 = r150 * r82;
    r96 = fmaf(r63, r13, r96);
    r86 = r11 * r59;
    r86 = r86 * r89;
    r86 = r86 * r28;
    r86 = r86 * r47;
    r86 = r86 * r80;
    r96 = fmaf(r35, r86, r96);
    r96 = fmaf(r28, r132, r96);
    r96 = fmaf(r58, r72, r96);
    r86 = r5 * r96;
    r13 = r11 * r11;
    r137 = r16 * r46;
    r137 = r137 * r11;
    r112 = r38 * r11;
    r112 = r112 * r11;
    r112 = fmaf(r108, r112, r37 * r137);
    r137 = r16 * r57;
    r137 = r137 * r10;
    r112 = fmaf(r37, r137, r112);
    r142 = r38 * r10;
    r142 = r142 * r10;
    r112 = fmaf(r108, r142, r112);
    r13 = r13 * r112;
    r13 = r13 * r35;
    r13 = r13 * r42;
    r13 = fmaf(r46, r74, r61 * r13);
    r142 = r4 * r112;
    r13 = fmaf(r122, r142, r13);
    r137 = r38 * r11;
    r137 = r137 * r11;
    r13 = fmaf(r125, r137, r13);
    r137 = r38 * r10;
    r137 = r137 * r10;
    r142 = r4 * r112;
    r142 = fmaf(r116, r142, r125 * r137);
    r137 = r57 * r73;
    r142 = fmaf(r115, r137, r142);
    r15 = r112 * r88;
    r142 = fmaf(r111, r15, r142);
    r15 = r13 + r142;
    r137 = r38 * r10;
    r137 = r137 * r10;
    r137 = r137 * r59;
    r137 = r137 * r59;
    r137 = r137 * r113;
    r137 = r137 * r47;
    r137 = fmaf(r112, r119, r85 * r137);
    r94 = r57 * r10;
    r94 = r94 * r59;
    r94 = r94 * r101;
    r137 = fmaf(r61, r94, r137);
    r106 = r71 * r112;
    r106 = r106 * r88;
    r137 = fmaf(r111, r106, r137);
    r137 = r137 + r13;
    r137 = fmaf(r67, r137, r8 * r15);
    r13 = r4 * r38;
    r13 = r13 * r10;
    r13 = r13 * r78;
    r13 = r13 * r81;
    r137 = fmaf(r56, r13, r137);
    r106 = r57 * r59;
    r137 = fmaf(r82, r106, r137);
    r94 = r4 * r38;
    r94 = r94 * r10;
    r94 = r94 * r81;
    r137 = fmaf(r56, r94, r137);
    r105 = r57 * r59;
    r137 = fmaf(r83, r105, r137);
    r109 = r112 * r83;
    r137 = fmaf(r131, r109, r137);
    r3 = r66 * r41;
    r3 = r3 * r112;
    r3 = r3 * r63;
    r3 = r3 * r102;
    r137 = fmaf(r118, r3, r137);
    r90 = r66 * r11;
    r90 = r90 * r112;
    r90 = r90 * r73;
    r137 = fmaf(r88, r90, r137);
    r93 = r66 * r46;
    r93 = r93 * r73;
    r137 = fmaf(r115, r93, r137);
    r144 = r66 * r57;
    r137 = fmaf(r74, r144, r137);
    r133 = r10 * r59;
    r110 = r7 * r16;
    r110 = r110 * r65;
    r110 = fmaf(r6, r15, r15 * r110);
    r110 = fmaf(r15, r69, r110);
    r110 = fmaf(r15, r68, r110);
    r133 = r133 * r110;
    r137 = fmaf(r82, r133, r137);
    r137 = fmaf(r112, r127, r137);
    r137 = fmaf(r112, r124, r137);
    r137 = fmaf(r112, r136, r137);
    r137 = fmaf(r38, r129, r137);
    r133 = r0 * r137;
    r144 = r11 * r11;
    r144 = r144 * r71;
    r144 = r144 * r112;
    r144 = r144 * r35;
    r144 = r144 * r42;
    r93 = r46 * r101;
    r93 = r93 * r61;
    r93 = fmaf(r63, r93, r61 * r144);
    r144 = r114 * r112;
    r93 = fmaf(r122, r144, r93);
    r90 = r38 * r11;
    r90 = r90 * r11;
    r90 = r90 * r59;
    r90 = r90 * r59;
    r90 = r90 * r113;
    r90 = r90 * r47;
    r93 = fmaf(r85, r90, r93);
    r93 = r93 + r142;
    r93 = fmaf(r66, r93, r9 * r15);
    r15 = r11 * r87;
    r15 = r15 * r112;
    r15 = r15 * r35;
    r15 = r15 * r42;
    r93 = fmaf(r82, r15, r93);
    r142 = r46 * r59;
    r93 = fmaf(r83, r142, r93);
    r90 = r11 * r59;
    r90 = r90 * r78;
    r90 = r90 * r89;
    r90 = r90 * r112;
    r90 = r90 * r47;
    r90 = r90 * r80;
    r93 = fmaf(r35, r90, r93);
    r144 = r11 * r59;
    r144 = r144 * r89;
    r144 = r144 * r112;
    r144 = r144 * r47;
    r144 = r144 * r80;
    r93 = fmaf(r35, r144, r93);
    r3 = r67 * r41;
    r3 = r3 * r112;
    r3 = r3 * r63;
    r3 = r3 * r102;
    r93 = fmaf(r118, r3, r93);
    r109 = r67 * r11;
    r109 = r109 * r112;
    r109 = r109 * r73;
    r93 = fmaf(r88, r109, r93);
    r105 = r67 * r46;
    r105 = r105 * r73;
    r93 = fmaf(r115, r105, r93);
    r94 = r4 * r38;
    r94 = r94 * r11;
    r94 = r94 * r78;
    r94 = r94 * r81;
    r93 = fmaf(r56, r94, r93);
    r106 = r46 * r59;
    r93 = fmaf(r82, r106, r93);
    r13 = r4 * r38;
    r13 = r13 * r11;
    r13 = r13 * r81;
    r93 = fmaf(r56, r13, r93);
    r92 = r110 * r82;
    r93 = fmaf(r63, r92, r93);
    r147 = r67 * r38;
    r147 = r147 * r10;
    r147 = r147 * r11;
    r147 = r147 * r59;
    r147 = r147 * r59;
    r147 = r147 * r99;
    r147 = r147 * r47;
    r93 = fmaf(r85, r147, r93);
    r93 = fmaf(r112, r132, r93);
    r93 = fmaf(r57, r72, r93);
    r147 = r5 * r93;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r95,
                                          r86,
                                          r133,
                                          r147);
    r147 = r0 * r4;
    r147 = r147 * r2;
    r133 = r4 * r1;
    r86 = r5 * r133;
    r147 = fmaf(r120, r86, r107 * r147);
    r95 = r0 * r4;
    r95 = r95 * r2;
    r95 = fmaf(r146, r86, r130 * r95);
    r92 = r0 * r4;
    r92 = r92 * r2;
    r92 = fmaf(r31, r86, r14 * r92);
    r13 = r0 * r4;
    r13 = r13 * r2;
    r13 = fmaf(r138, r86, r141 * r13);
    WriteSum4<float, float>((float*)inout_shared, r147, r95, r92, r13);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r0 * r4;
    r13 = r13 * r2;
    r13 = fmaf(r96, r86, r123 * r13);
    r92 = r0 * r4;
    r92 = r92 * r2;
    r92 = fmaf(r93, r86, r137 * r92);
    WriteSum2<float, float>((float*)inout_shared, r13, r92);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r92 = r0 * r0;
    r13 = r107 * r107;
    r95 = r5 * r5;
    r147 = r120 * r120;
    r147 = fmaf(r95, r147, r92 * r13);
    r13 = r130 * r130;
    r106 = r146 * r146;
    r106 = fmaf(r95, r106, r92 * r13);
    r13 = r31 * r31;
    r94 = r14 * r14;
    r94 = fmaf(r92, r94, r95 * r13);
    r13 = r141 * r141;
    r105 = r138 * r138;
    r105 = fmaf(r95, r105, r92 * r13);
    WriteSum4<float, float>((float*)inout_shared, r147, r106, r94, r105);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r105 = r123 * r123;
    r94 = r96 * r96;
    r94 = fmaf(r95, r94, r92 * r105);
    r105 = r93 * r93;
    r106 = r137 * r137;
    r106 = fmaf(r92, r106, r95 * r105);
    WriteSum2<float, float>((float*)inout_shared, r94, r106);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r106 = r120 * r146;
    r94 = r107 * r130;
    r94 = fmaf(r92, r94, r95 * r106);
    r106 = r107 * r14;
    r105 = r120 * r31;
    r105 = fmaf(r95, r105, r92 * r106);
    r106 = r120 * r138;
    r147 = r107 * r141;
    r147 = fmaf(r92, r147, r95 * r106);
    r106 = r120 * r96;
    r13 = r107 * r123;
    r13 = fmaf(r92, r13, r95 * r106);
    WriteSum4<float, float>((float*)inout_shared, r94, r105, r147, r13);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r107 * r137;
    r147 = r120 * r93;
    r147 = fmaf(r95, r147, r92 * r13);
    r13 = r146 * r31;
    r105 = r130 * r14;
    r105 = fmaf(r92, r105, r95 * r13);
    r13 = r130 * r141;
    r94 = r146 * r138;
    r94 = fmaf(r95, r94, r92 * r13);
    r13 = r130 * r123;
    r106 = r146 * r96;
    r106 = fmaf(r95, r106, r92 * r13);
    WriteSum4<float, float>((float*)inout_shared, r147, r105, r94, r106);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r106 = r146 * r93;
    r94 = r130 * r137;
    r94 = fmaf(r92, r94, r95 * r106);
    r106 = r31 * r138;
    r105 = r14 * r141;
    r105 = fmaf(r92, r105, r95 * r106);
    r106 = r14 * r123;
    r147 = r31 * r96;
    r147 = fmaf(r95, r147, r92 * r106);
    r106 = r31 * r93;
    r13 = r14 * r137;
    r13 = fmaf(r92, r13, r95 * r106);
    WriteSum4<float, float>((float*)inout_shared, r94, r105, r147, r13);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r138 * r96;
    r147 = r141 * r123;
    r147 = fmaf(r92, r147, r95 * r13);
    r13 = r141 * r137;
    r105 = r138 * r93;
    r105 = fmaf(r95, r105, r92 * r13);
    r13 = r96 * r93;
    r94 = r123 * r137;
    r94 = fmaf(r92, r94, r95 * r13);
    WriteSum3<float, float>((float*)inout_shared, r147, r105, r94);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r94 = r0 * r10;
    r94 = r94 * r59;
    r94 = r94 * r65;
    r94 = r94 * r82;
    r105 = r5 * r65;
    r105 = r105 * r82;
    r105 = r105 * r63;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          0 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r62,
                                          r64,
                                          r94,
                                          r105);
    r105 = r5 * r75;
    r94 = r0 * r82;
    r94 = r94 * r76;
    r94 = r94 * r115;
    r147 = r5 * r82;
    r147 = r147 * r63;
    r147 = r147 * r76;
    r13 = r0 * r10;
    r13 = r13 * r74;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          4 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r94,
                                          r147,
                                          r13,
                                          r105);
    r105 = r0 * r70;
    r13 = r5 * r10;
    r13 = r13 * r74;
    r147 = r0 * r82;
    r147 = r147 * r115;
    r147 = r147 * r77;
    r94 = r5 * r82;
    r94 = r94 * r63;
    r94 = r94 * r77;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          8 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r105,
                                          r13,
                                          r147,
                                          r94);
    r94 = r0 * r65;
    r147 = r5 * r65;
    r13 = r0 * r82;
    r13 = r13 * r115;
    r13 = r13 * r79;
    r105 = r5 * r82;
    r105 = r105 * r63;
    r105 = r105 * r79;
    WriteIdx4<1024, float, float, float4>(
        out_focal_and_extra_jac,
        12 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r13,
        r105,
        r94,
        r147);
    r147 = r4 * r62;
    r147 = r147 * r2;
    r133 = r64 * r133;
    r94 = r0 * r4;
    r94 = r94 * r10;
    r94 = r94 * r59;
    r94 = r94 * r65;
    r94 = r94 * r2;
    r105 = r65 * r82;
    r105 = r105 * r63;
    r105 = fmaf(r86, r105, r82 * r94);
    r94 = r0 * r4;
    r94 = r94 * r2;
    r94 = r94 * r82;
    r94 = r94 * r76;
    r13 = r82 * r63;
    r13 = r13 * r76;
    r13 = fmaf(r86, r13, r115 * r94);
    WriteSum4<float, float>((float*)inout_shared, r147, r133, r105, r13);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           0 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r0 * r41;
    r13 = r13 * r10;
    r13 = r13 * r2;
    r13 = r13 * r61;
    r13 = fmaf(r63, r13, r75 * r86);
    r105 = r0 * r4;
    r105 = r105 * r70;
    r133 = r5 * r41;
    r133 = r133 * r10;
    r133 = r133 * r1;
    r133 = r133 * r61;
    r133 = fmaf(r63, r133, r2 * r105);
    r105 = r0 * r4;
    r105 = r105 * r2;
    r105 = r105 * r82;
    r105 = r105 * r115;
    r1 = r82 * r63;
    r1 = r1 * r77;
    r1 = fmaf(r86, r1, r77 * r105);
    r105 = r0 * r4;
    r105 = r105 * r2;
    r105 = r105 * r82;
    r105 = r105 * r115;
    r147 = r82 * r63;
    r147 = r147 * r79;
    r147 = fmaf(r86, r147, r79 * r105);
    WriteSum4<float, float>((float*)inout_shared, r13, r133, r1, r147);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           4 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r147 = r0 * r4;
    r147 = r147 * r65;
    r147 = r147 * r2;
    r1 = r65 * r86;
    WriteSum2<float, float>((float*)inout_shared, r147, r1);
  };
  FlushSumShared<2, float>(out_focal_and_extra_njtr,
                           8 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r62 * r62;
    r147 = r64 * r64;
    r133 = r11 * r61;
    r133 = r133 * r63;
    r133 = r133 * r76;
    r13 = r76 * r115;
    r13 = r13 * r92;
    r13 = fmaf(r111, r13, r95 * r133);
    r133 = r11 * r61;
    r133 = r133 * r63;
    r133 = r133 * r95;
    r105 = r115 * r92;
    r105 = r105 * r79;
    r105 = fmaf(r111, r105, r79 * r133);
    WriteSum4<float, float>((float*)inout_shared, r1, r147, r13, r105);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_diag,
                           0 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r10 * r10;
    r12 = r36 * r12;
    r12 = 1.0 / r12;
    r52 = r51 * r52;
    r52 = 1.0 / r52;
    r13 = r13 * r11;
    r13 = r13 * r11;
    r13 = r13 * r59;
    r13 = r13 * r59;
    r13 = r13 * r140;
    r13 = r13 * r12;
    r13 = r13 * r52;
    r13 = r13 * r126;
    r52 = r75 * r95;
    r12 = fmaf(r75, r52, r92 * r13);
    r140 = r70 * r70;
    r140 = fmaf(r92, r140, r95 * r13);
    r13 = r77 * r77;
    r51 = r11 * r61;
    r51 = r51 * r63;
    r51 = r51 * r95;
    r36 = r115 * r92;
    r147 = r111 * r36;
    r1 = fmaf(r13, r147, r13 * r51);
    r133 = r79 * r79;
    r133 = fmaf(r147, r133, r51 * r133);
    WriteSum4<float, float>((float*)inout_shared, r12, r140, r1, r133);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_diag,
                           4 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r133 = r76 * r92;
    r140 = r76 * r95;
    WriteSum2<float, float>((float*)inout_shared, r133, r140);
  };
  FlushSumShared<2, float>(out_focal_and_extra_precond_diag,
                           8 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r140 = 0.00000000000000000e+00;
    r133 = r0 * r10;
    r133 = r133 * r59;
    r133 = r133 * r65;
    r133 = r133 * r62;
    r133 = r133 * r82;
    r12 = r0 * r62;
    r12 = r12 * r82;
    r12 = r12 * r76;
    r12 = r12 * r115;
    r94 = r10 * r74;
    r94 = r94 * r84;
    WriteSum4<float, float>((float*)inout_shared, r140, r133, r12, r94);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           0 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r94 = r0 * r70;
    r94 = r94 * r62;
    r12 = r0 * r65;
    r12 = r12 * r62;
    r62 = r0 * r62;
    r62 = r62 * r82;
    r62 = r62 * r115;
    r62 = r62 * r77;
    r133 = r82 * r115;
    r133 = r133 * r79;
    r133 = r133 * r84;
    WriteSum4<float, float>((float*)inout_shared, r94, r62, r133, r12);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           4 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = r5 * r75;
    r75 = r75 * r64;
    r12 = r5 * r65;
    r12 = r12 * r64;
    r12 = r12 * r82;
    r12 = r12 * r63;
    r133 = r5 * r64;
    r133 = r133 * r82;
    r133 = r133 * r63;
    r133 = r133 * r76;
    WriteSum4<float, float>((float*)inout_shared, r140, r12, r133, r75);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           8 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = r5 * r10;
    r75 = r75 * r64;
    r75 = r75 * r74;
    r133 = r5 * r64;
    r133 = r133 * r82;
    r133 = r133 * r63;
    r133 = r133 * r77;
    r12 = r5 * r64;
    r12 = r12 * r82;
    r12 = r12 * r63;
    r12 = r12 * r79;
    WriteSum4<float, float>((float*)inout_shared, r75, r133, r12, r140);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           12 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r140 = r5 * r65;
    r140 = r140 * r64;
    r64 = r11 * r61;
    r64 = r64 * r63;
    r64 = r64 * r77;
    r12 = r115 * r77;
    r12 = r12 * r92;
    r12 = fmaf(r111, r12, r95 * r64);
    r64 = r10 * r59;
    r133 = r16 * r11;
    r133 = r133 * r85;
    r133 = r133 * r117;
    r133 = r133 * r126;
    r126 = r10 * r133;
    r64 = r64 * r65;
    r64 = r64 * r92;
    r117 = r82 * r63;
    r117 = r117 * r52;
    r64 = fmaf(r65, r117, r126 * r64);
    r75 = r10 * r59;
    r75 = r75 * r65;
    r75 = r75 * r70;
    r75 = r75 * r82;
    r62 = r11 * r59;
    r62 = r62 * r65;
    r62 = r62 * r95;
    r62 = fmaf(r126, r62, r92 * r75);
    WriteSum4<float, float>((float*)inout_shared, r140, r12, r64, r62);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           16 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = r82 * r76;
    r62 = r62 * r115;
    r62 = r62 * r92;
    r64 = r82 * r63;
    r64 = r64 * r76;
    r64 = r64 * r95;
    r12 = r65 * r79;
    r140 = fmaf(r12, r147, r12 * r51);
    WriteSum4<float, float>((float*)inout_shared, r105, r140, r62, r64);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           20 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = r76 * r36;
    r64 = fmaf(r76, r117, r126 * r64);
    r62 = r70 * r82;
    r62 = r62 * r76;
    r62 = r62 * r115;
    r105 = r11 * r76;
    r105 = r105 * r115;
    r105 = r105 * r95;
    r105 = fmaf(r133, r105, r92 * r62);
    WriteSum4<float, float>((float*)inout_shared, r64, r105, r140, r1);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           24 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r82 * r115;
    r1 = r1 * r77;
    r1 = r1 * r92;
    r140 = r82 * r63;
    r140 = r140 * r77;
    r140 = r140 * r95;
    r105 = r10 * r74;
    r64 = r10 * r70;
    r64 = r64 * r92;
    r64 = fmaf(r74, r64, r52 * r105);
    r105 = r77 * r36;
    r105 = fmaf(r77, r117, r126 * r105);
    WriteSum4<float, float>((float*)inout_shared, r1, r140, r64, r105);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           28 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r105 = r10 * r65;
    r105 = r105 * r92;
    r105 = r105 * r74;
    r52 = r65 * r52;
    r64 = r79 * r36;
    r117 = fmaf(r79, r117, r126 * r64);
    r64 = r70 * r82;
    r64 = r64 * r115;
    r64 = r64 * r77;
    r126 = r11 * r115;
    r126 = r126 * r77;
    r126 = r126 * r95;
    r126 = fmaf(r133, r126, r92 * r64);
    WriteSum4<float, float>((float*)inout_shared, r117, r105, r52, r126);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           32 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r65 * r70;
    r126 = r126 * r92;
    r52 = r10 * r65;
    r52 = r52 * r95;
    r52 = r52 * r74;
    r105 = r70 * r82;
    r105 = r105 * r115;
    r105 = r105 * r92;
    r117 = r11 * r115;
    r117 = r117 * r95;
    r117 = r117 * r79;
    r117 = fmaf(r133, r117, r79 * r105);
    r13 = r65 * r13;
    r147 = fmaf(r13, r147, r13 * r51);
    WriteSum4<float, float>((float*)inout_shared, r117, r126, r52, r147);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           36 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r147 = r82 * r115;
    r147 = r147 * r92;
    r147 = r147 * r79;
    r52 = r82 * r63;
    r52 = r52 * r95;
    r52 = r52 * r79;
    r126 = r82 * r12;
    r126 = r126 * r36;
    r117 = r82 * r63;
    r117 = r117 * r95;
    r117 = r117 * r12;
    WriteSum4<float, float>((float*)inout_shared, r147, r52, r126, r117);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           40 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r117 = r54 * r10;
    r117 = r117 * r59;
    r117 = r117 * r101;
    r126 = r60 * r10;
    r126 = r126 * r10;
    r126 = r126 * r59;
    r126 = r126 * r59;
    r126 = r126 * r113;
    r126 = r126 * r47;
    r126 = fmaf(r85, r126, r61 * r117);
    r117 = r16 * r33;
    r117 = r117 * r11;
    r52 = r16 * r54;
    r52 = r52 * r10;
    r52 = fmaf(r37, r52, r37 * r117);
    r117 = r60 * r10;
    r117 = r117 * r10;
    r52 = fmaf(r108, r117, r52);
    r147 = r60 * r11;
    r147 = r147 * r11;
    r52 = fmaf(r108, r147, r52);
    r147 = r71 * r52;
    r147 = r147 * r88;
    r126 = fmaf(r111, r147, r126);
    r117 = r4 * r52;
    r12 = r11 * r11;
    r12 = r12 * r52;
    r12 = r12 * r35;
    r12 = r12 * r42;
    r12 = fmaf(r61, r12, r122 * r117);
    r117 = r60 * r11;
    r117 = r117 * r11;
    r12 = fmaf(r125, r117, r12);
    r12 = fmaf(r33, r74, r12);
    r126 = fmaf(r52, r119, r126);
    r126 = r126 + r12;
    r147 = r54 * r73;
    r117 = r60 * r10;
    r117 = r117 * r10;
    r117 = fmaf(r125, r117, r115 * r147);
    r147 = r52 * r88;
    r117 = fmaf(r111, r147, r117);
    r13 = r4 * r52;
    r117 = fmaf(r116, r13, r117);
    r12 = r12 + r117;
    r126 = fmaf(r8, r12, r67 * r126);
    r13 = r54 * r59;
    r126 = fmaf(r83, r13, r126);
    r147 = r4 * r60;
    r147 = r147 * r10;
    r147 = r147 * r78;
    r147 = r147 * r81;
    r126 = fmaf(r56, r147, r126);
    r51 = r4 * r60;
    r51 = r51 * r10;
    r51 = r51 * r81;
    r126 = fmaf(r56, r51, r126);
    r105 = r66 * r33;
    r105 = r105 * r73;
    r126 = fmaf(r115, r105, r126);
    r133 = r10 * r59;
    r64 = r7 * r16;
    r64 = r64 * r65;
    r64 = fmaf(r6, r12, r12 * r64);
    r64 = fmaf(r12, r69, r64);
    r64 = fmaf(r12, r68, r64);
    r133 = r133 * r64;
    r126 = fmaf(r82, r133, r126);
    r140 = r52 * r83;
    r126 = fmaf(r131, r140, r126);
    r1 = r54 * r59;
    r126 = fmaf(r82, r1, r126);
    r62 = r66 * r41;
    r62 = r62 * r52;
    r62 = r62 * r63;
    r62 = r62 * r102;
    r126 = fmaf(r118, r62, r126);
    r75 = r66 * r54;
    r126 = fmaf(r74, r75, r126);
    r94 = r66 * r11;
    r94 = r94 * r52;
    r94 = r94 * r73;
    r126 = fmaf(r88, r94, r126);
    r126 = fmaf(r52, r124, r126);
    r126 = fmaf(r52, r127, r126);
    r126 = fmaf(r60, r129, r126);
    r126 = fmaf(r52, r136, r126);
    r94 = r0 * r126;
    r75 = r114 * r52;
    r62 = r11 * r11;
    r62 = r62 * r71;
    r62 = r62 * r52;
    r62 = r62 * r35;
    r62 = r62 * r42;
    r62 = fmaf(r61, r62, r122 * r75);
    r75 = r60 * r11;
    r75 = r75 * r11;
    r75 = r75 * r59;
    r75 = r75 * r59;
    r75 = r75 * r113;
    r75 = r75 * r47;
    r62 = fmaf(r85, r75, r62);
    r1 = r33 * r101;
    r1 = r1 * r61;
    r62 = fmaf(r63, r1, r62);
    r62 = r62 + r117;
    r12 = fmaf(r9, r12, r66 * r62);
    r62 = r11 * r59;
    r62 = r62 * r78;
    r62 = r62 * r89;
    r62 = r62 * r52;
    r62 = r62 * r47;
    r62 = r62 * r80;
    r12 = fmaf(r35, r62, r12);
    r117 = r4 * r60;
    r117 = r117 * r11;
    r117 = r117 * r78;
    r117 = r117 * r81;
    r12 = fmaf(r56, r117, r12);
    r1 = r67 * r33;
    r1 = r1 * r73;
    r12 = fmaf(r115, r1, r12);
    r75 = r67 * r60;
    r75 = r75 * r10;
    r75 = r75 * r11;
    r75 = r75 * r59;
    r75 = r75 * r59;
    r75 = r75 * r99;
    r75 = r75 * r47;
    r12 = fmaf(r85, r75, r12);
    r140 = r11 * r59;
    r140 = r140 * r89;
    r140 = r140 * r52;
    r140 = r140 * r47;
    r140 = r140 * r80;
    r12 = fmaf(r35, r140, r12);
    r133 = r67 * r41;
    r133 = r133 * r52;
    r133 = r133 * r63;
    r133 = r133 * r102;
    r12 = fmaf(r118, r133, r12);
    r105 = r11 * r87;
    r105 = r105 * r52;
    r105 = r105 * r35;
    r105 = r105 * r42;
    r12 = fmaf(r82, r105, r12);
    r51 = r33 * r59;
    r12 = fmaf(r83, r51, r12);
    r147 = r4 * r60;
    r147 = r147 * r11;
    r147 = r147 * r81;
    r12 = fmaf(r56, r147, r12);
    r13 = r67 * r11;
    r13 = r13 * r52;
    r13 = r13 * r73;
    r12 = fmaf(r88, r13, r12);
    r84 = r33 * r59;
    r12 = fmaf(r82, r84, r12);
    r106 = r64 * r82;
    r12 = fmaf(r63, r106, r12);
    r12 = fmaf(r52, r132, r12);
    r12 = fmaf(r54, r72, r12);
    r106 = r5 * r12;
    r84 = r45 * r11;
    r84 = r84 * r11;
    r13 = r16 * r49;
    r13 = r13 * r10;
    r13 = fmaf(r37, r13, r108 * r84);
    r84 = r16 * r50;
    r84 = r84 * r11;
    r13 = fmaf(r37, r84, r13);
    r147 = r45 * r10;
    r147 = r147 * r10;
    r13 = fmaf(r108, r147, r13);
    r147 = r13 * r88;
    r84 = r45 * r10;
    r84 = r84 * r10;
    r84 = fmaf(r125, r84, r111 * r147);
    r147 = r4 * r13;
    r84 = fmaf(r116, r147, r84);
    r51 = r49 * r73;
    r84 = fmaf(r115, r51, r84);
    r51 = r13 * r122;
    r147 = fmaf(r50, r74, r4 * r51);
    r105 = r11 * r11;
    r105 = r105 * r13;
    r105 = r105 * r35;
    r105 = r105 * r42;
    r147 = fmaf(r61, r105, r147);
    r133 = r45 * r11;
    r133 = r133 * r11;
    r147 = fmaf(r125, r133, r147);
    r133 = r84 + r147;
    r105 = r71 * r13;
    r105 = r105 * r88;
    r140 = r45 * r10;
    r140 = r140 * r10;
    r140 = r140 * r59;
    r140 = r140 * r59;
    r140 = r140 * r113;
    r140 = r140 * r47;
    r140 = fmaf(r85, r140, r111 * r105);
    r105 = r49 * r10;
    r105 = r105 * r59;
    r105 = r105 * r101;
    r140 = fmaf(r61, r105, r140);
    r140 = fmaf(r13, r119, r140);
    r140 = r140 + r147;
    r140 = fmaf(r67, r140, r8 * r133);
    r147 = r66 * r49;
    r140 = fmaf(r74, r147, r140);
    r105 = r66 * r41;
    r105 = r105 * r13;
    r105 = r105 * r63;
    r105 = r105 * r102;
    r140 = fmaf(r118, r105, r140);
    r75 = r4 * r45;
    r75 = r75 * r10;
    r75 = r75 * r81;
    r140 = fmaf(r56, r75, r140);
    r1 = r4 * r45;
    r1 = r1 * r10;
    r1 = r1 * r78;
    r1 = r1 * r81;
    r140 = fmaf(r56, r1, r140);
    r117 = r10 * r59;
    r62 = r7 * r16;
    r62 = r62 * r65;
    r62 = fmaf(r133, r62, r6 * r133);
    r62 = fmaf(r133, r69, r62);
    r62 = fmaf(r133, r68, r62);
    r117 = r117 * r62;
    r140 = fmaf(r82, r117, r140);
    r109 = r13 * r83;
    r140 = fmaf(r131, r109, r140);
    r3 = r49 * r59;
    r140 = fmaf(r83, r3, r140);
    r144 = r66 * r50;
    r144 = r144 * r73;
    r140 = fmaf(r115, r144, r140);
    r90 = r49 * r59;
    r140 = fmaf(r82, r90, r140);
    r142 = r66 * r11;
    r142 = r142 * r13;
    r142 = r142 * r73;
    r140 = fmaf(r88, r142, r140);
    r140 = fmaf(r13, r127, r140);
    r140 = fmaf(r13, r124, r140);
    r140 = fmaf(r13, r136, r140);
    r140 = fmaf(r45, r129, r140);
    r142 = r0 * r140;
    r90 = r50 * r101;
    r90 = r90 * r61;
    r90 = fmaf(r63, r90, r114 * r51);
    r51 = r11 * r11;
    r51 = r51 * r71;
    r51 = r51 * r13;
    r51 = r51 * r35;
    r51 = r51 * r42;
    r90 = fmaf(r61, r51, r90);
    r144 = r45 * r11;
    r144 = r144 * r11;
    r144 = r144 * r59;
    r144 = r144 * r59;
    r144 = r144 * r113;
    r144 = r144 * r47;
    r90 = fmaf(r85, r144, r90);
    r90 = r90 + r84;
    r90 = fmaf(r66, r90, r9 * r133);
    r133 = r67 * r11;
    r133 = r133 * r13;
    r133 = r133 * r73;
    r90 = fmaf(r88, r133, r90);
    r84 = r67 * r41;
    r84 = r84 * r13;
    r84 = r84 * r63;
    r84 = r84 * r102;
    r90 = fmaf(r118, r84, r90);
    r144 = r11 * r59;
    r144 = r144 * r78;
    r144 = r144 * r89;
    r144 = r144 * r13;
    r144 = r144 * r47;
    r144 = r144 * r80;
    r90 = fmaf(r35, r144, r90);
    r51 = r11 * r59;
    r51 = r51 * r89;
    r51 = r51 * r13;
    r51 = r51 * r47;
    r51 = r51 * r80;
    r90 = fmaf(r35, r51, r90);
    r3 = r62 * r82;
    r90 = fmaf(r63, r3, r90);
    r109 = r50 * r59;
    r90 = fmaf(r82, r109, r90);
    r117 = r4 * r45;
    r117 = r117 * r11;
    r117 = r117 * r78;
    r117 = r117 * r81;
    r90 = fmaf(r56, r117, r90);
    r1 = r50 * r59;
    r90 = fmaf(r83, r1, r90);
    r75 = r11 * r87;
    r75 = r75 * r13;
    r75 = r75 * r35;
    r75 = r75 * r42;
    r90 = fmaf(r82, r75, r90);
    r105 = r67 * r50;
    r105 = r105 * r73;
    r90 = fmaf(r115, r105, r90);
    r147 = r4 * r45;
    r147 = r147 * r11;
    r147 = r147 * r81;
    r90 = fmaf(r56, r147, r90);
    r15 = r67 * r45;
    r15 = r15 * r10;
    r15 = r15 * r11;
    r15 = r15 * r59;
    r15 = r15 * r59;
    r15 = r15 * r99;
    r15 = r15 * r47;
    r90 = fmaf(r85, r15, r90);
    r90 = fmaf(r49, r72, r90);
    r90 = fmaf(r13, r132, r90);
    r15 = r5 * r90;
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r94,
                                          r106,
                                          r142,
                                          r15);
    r15 = r16 * r48;
    r15 = r15 * r11;
    r142 = r53 * r11;
    r142 = r142 * r11;
    r142 = fmaf(r108, r142, r37 * r15);
    r15 = r53 * r10;
    r15 = r15 * r10;
    r142 = fmaf(r108, r15, r142);
    r108 = r16 * r55;
    r108 = r108 * r10;
    r142 = fmaf(r37, r108, r142);
    r108 = r4 * r142;
    r108 = fmaf(r48, r74, r122 * r108);
    r15 = r11 * r11;
    r15 = r15 * r142;
    r15 = r15 * r35;
    r15 = r15 * r42;
    r108 = fmaf(r61, r15, r108);
    r37 = r53 * r11;
    r37 = r37 * r11;
    r108 = fmaf(r125, r37, r108);
    r37 = r55 * r73;
    r15 = r142 * r88;
    r15 = fmaf(r111, r15, r115 * r37);
    r37 = r53 * r10;
    r37 = r37 * r10;
    r15 = fmaf(r125, r37, r15);
    r125 = r4 * r142;
    r15 = fmaf(r116, r125, r15);
    r125 = r108 + r15;
    r37 = r55 * r10;
    r37 = r37 * r59;
    r37 = r37 * r101;
    r116 = r71 * r142;
    r116 = r116 * r88;
    r116 = fmaf(r111, r116, r61 * r37);
    r37 = r53 * r10;
    r37 = r37 * r10;
    r37 = r37 * r59;
    r37 = r37 * r59;
    r37 = r37 * r113;
    r37 = r37 * r47;
    r116 = fmaf(r85, r37, r116);
    r116 = fmaf(r142, r119, r116);
    r116 = r116 + r108;
    r116 = fmaf(r67, r116, r8 * r125);
    r8 = r66 * r48;
    r8 = r8 * r73;
    r116 = fmaf(r115, r8, r116);
    r108 = r66 * r11;
    r108 = r108 * r142;
    r108 = r108 * r73;
    r116 = fmaf(r88, r108, r116);
    r119 = r4 * r53;
    r119 = r119 * r10;
    r119 = r119 * r78;
    r119 = r119 * r81;
    r116 = fmaf(r56, r119, r116);
    r37 = r10 * r59;
    r111 = r7 * r16;
    r111 = r111 * r65;
    r111 = fmaf(r125, r111, r6 * r125);
    r111 = fmaf(r125, r69, r111);
    r111 = fmaf(r125, r68, r111);
    r37 = r37 * r111;
    r116 = fmaf(r82, r37, r116);
    r68 = r142 * r83;
    r116 = fmaf(r131, r68, r116);
    r131 = r4 * r53;
    r131 = r131 * r10;
    r131 = r131 * r81;
    r116 = fmaf(r56, r131, r116);
    r69 = r55 * r59;
    r116 = fmaf(r82, r69, r116);
    r6 = r55 * r59;
    r116 = fmaf(r83, r6, r116);
    r106 = r66 * r41;
    r106 = r106 * r142;
    r106 = r106 * r63;
    r106 = r106 * r102;
    r116 = fmaf(r118, r106, r116);
    r94 = r66 * r55;
    r116 = fmaf(r74, r94, r116);
    r116 = fmaf(r142, r124, r116);
    r116 = fmaf(r142, r127, r116);
    r116 = fmaf(r142, r136, r116);
    r116 = fmaf(r53, r129, r116);
    r94 = r0 * r116;
    r129 = r114 * r142;
    r136 = r48 * r101;
    r136 = r136 * r61;
    r136 = fmaf(r63, r136, r122 * r129);
    r129 = r11 * r11;
    r129 = r129 * r71;
    r129 = r129 * r142;
    r129 = r129 * r35;
    r129 = r129 * r42;
    r136 = fmaf(r61, r129, r136);
    r122 = r53 * r11;
    r122 = r122 * r11;
    r122 = r122 * r59;
    r122 = r122 * r59;
    r122 = r122 * r113;
    r122 = r122 * r47;
    r136 = fmaf(r85, r122, r136);
    r136 = r136 + r15;
    r136 = fmaf(r66, r136, r9 * r125);
    r125 = r67 * r48;
    r125 = r125 * r73;
    r136 = fmaf(r115, r125, r136);
    r9 = r4 * r53;
    r9 = r9 * r11;
    r9 = r9 * r78;
    r9 = r9 * r81;
    r136 = fmaf(r56, r9, r136);
    r15 = r11 * r59;
    r15 = r15 * r89;
    r15 = r15 * r142;
    r15 = r15 * r47;
    r15 = r15 * r80;
    r136 = fmaf(r35, r15, r136);
    r122 = r67 * r11;
    r122 = r122 * r142;
    r122 = r122 * r73;
    r136 = fmaf(r88, r122, r136);
    r129 = r48 * r59;
    r136 = fmaf(r82, r129, r136);
    r113 = r111 * r82;
    r136 = fmaf(r63, r113, r136);
    r106 = r67 * r41;
    r106 = r106 * r142;
    r106 = r106 * r63;
    r106 = r106 * r102;
    r136 = fmaf(r118, r106, r136);
    r118 = r67 * r53;
    r118 = r118 * r10;
    r118 = r118 * r11;
    r118 = r118 * r59;
    r118 = r118 * r59;
    r118 = r118 * r99;
    r118 = r118 * r47;
    r136 = fmaf(r85, r118, r136);
    r85 = r11 * r59;
    r85 = r85 * r78;
    r85 = r85 * r89;
    r85 = r85 * r142;
    r85 = r85 * r47;
    r85 = r85 * r80;
    r136 = fmaf(r35, r85, r136);
    r80 = r48 * r59;
    r136 = fmaf(r83, r80, r136);
    r47 = r4 * r53;
    r47 = r47 * r11;
    r47 = r47 * r81;
    r136 = fmaf(r56, r47, r136);
    r56 = r11 * r87;
    r56 = r56 * r142;
    r56 = r56 * r35;
    r56 = r56 * r42;
    r136 = fmaf(r82, r56, r136);
    r136 = fmaf(r142, r132, r136);
    r136 = fmaf(r55, r72, r136);
    r56 = r5 * r136;
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r94,
                                          r56);
    r56 = r0 * r4;
    r56 = r56 * r2;
    r56 = fmaf(r12, r86, r126 * r56);
    r94 = r0 * r4;
    r94 = r94 * r2;
    r94 = fmaf(r90, r86, r140 * r94);
    r47 = r0 * r4;
    r47 = r47 * r2;
    r86 = fmaf(r136, r86, r116 * r47);
    WriteSum3<float, float>((float*)inout_shared, r56, r94, r86);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = r126 * r126;
    r94 = r12 * r12;
    r94 = fmaf(r95, r94, r92 * r86);
    r86 = r90 * r90;
    r56 = r140 * r140;
    r56 = fmaf(r92, r56, r95 * r86);
    r86 = r136 * r136;
    r47 = r116 * r116;
    r47 = fmaf(r92, r47, r95 * r86);
    WriteSum3<float, float>((float*)inout_shared, r94, r56, r47);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = r12 * r90;
    r56 = r126 * r140;
    r56 = fmaf(r92, r56, r95 * r47);
    r47 = r126 * r116;
    r94 = r12 * r136;
    r94 = fmaf(r95, r94, r92 * r47);
    r47 = r90 * r136;
    r86 = r140 * r116;
    r86 = fmaf(r92, r86, r95 * r47);
    WriteSum3<float, float>((float*)inout_shared, r56, r94, r86);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void ThinPrismFisheyeSplitFixedPrincipalPointResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
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
    float* out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    float* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    float* const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    float* const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
    float* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    float* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeSplitFixedPrincipalPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      focal_and_extra_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      principal_point,
      principal_point_num_alloc,
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
      out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc,
      out_point_jac,
      out_point_jac_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar