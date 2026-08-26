#include "kernel_thin_prism_fisheye_split_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPrincipalPointResJacKernel(
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
    r52 = r11 * r11;
    r53 = 9.99999999999999955e-07;
    r54 = r41 * r29;
    r54 = r54 * r29;
    r55 = r42 + r54;
    r55 = r55 + r49;
    r10 = fmaf(r13, r55, r10);
    r49 = r32 * r41;
    r49 = fmaf(r34, r49, r26);
    r26 = r16 * r32;
    r26 = r26 * r25;
    r56 = r16 * r29;
    r56 = fmaf(r34, r56, r26);
    r57 = r21 * r23;
    r57 = r57 * r16;
    r58 = r22 * r24;
    r58 = fmaf(r16, r58, r57);
    r59 = r23 * r24;
    r59 = fmaf(r41, r59, r38);
    r38 = r22 * r22;
    r38 = r38 * r41;
    r43 = r38 + r43;
    r10 = fmaf(r14, r49, r10);
    r10 = fmaf(r15, r56, r10);
    r10 = fmaf(r37, r58, r10);
    r10 = fmaf(r36, r59, r10);
    r10 = fmaf(r35, r43, r10);
    r60 = r10 * r10;
    r61 = r41 * r29;
    r61 = fmaf(r34, r61, r26);
    r12 = fmaf(r13, r61, r12);
    r26 = r22 * r24;
    r26 = fmaf(r41, r26, r57);
    r38 = r42 + r38;
    r38 = r38 + r40;
    r40 = r21 * r24;
    r40 = fmaf(r16, r40, r45);
    r45 = r16 * r25;
    r45 = fmaf(r34, r45, r47);
    r54 = r42 + r54;
    r54 = r54 + r51;
    r12 = fmaf(r35, r26, r12);
    r12 = fmaf(r37, r38, r12);
    r12 = fmaf(r36, r40, r12);
    r12 = fmaf(r14, r45, r12);
    r12 = fmaf(r15, r54, r12);
    r36 = copysign(1.0, r12);
    r36 = fmaf(r53, r36, r12);
    r12 = r36 * r36;
    r37 = 1.0 / r12;
    r35 = r11 * r11;
    r35 = fmaf(r37, r35, r37 * r60);
    r60 = sqrtf(r35);
    r51 = copysign(1.0, r60);
    r51 = fmaf(r53, r51, r60);
    r53 = r51 * r51;
    r47 = 1.0 / r53;
    r60 = atanf(r60);
    r57 = r60 * r37;
    r62 = r60 * r57;
    r52 = r52 * r47;
    r52 = r52 * r62;
    r63 = r10 * r47;
    r64 = r10 * r63;
    r65 = r62 * r64;
    r66 = r52 + r65;
  };
  LoadShared<4, float, float>(focal_and_extra,
                              4 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r67,
                       r68,
                       r69,
                       r70);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r71 = 3.00000000000000000e+00;
    r72 = r71 * r62;
    r72 = fmaf(r64, r72, r52);
    r52 = fmaf(r68, r72, r8 * r66);
    r73 = 1.0 / r36;
    r74 = 1.0 / r51;
    r75 = r73 * r74;
    r76 = r60 * r75;
    r77 = r10 * r76;
    r78 = r67 * r11;
    r79 = r16 * r62;
    r78 = r78 * r63;
    r52 = fmaf(r79, r78, r52);
    r80 = r66 * r66;
    r81 = r66 * r80;
    r82 = fmaf(r69, r81, r6 * r66);
    r83 = r80 * r80;
    r82 = fmaf(r70, r83, r82);
    r82 = fmaf(r7, r80, r82);
    r84 = r82 * r76;
    r52 = r52 + r77;
    r52 = fmaf(r10, r84, r52);
    r2 = fmaf(r0, r52, r2);
    r78 = r11 * r11;
    r78 = r78 * r71;
    r78 = r78 * r47;
    r78 = fmaf(r62, r78, r65);
    r65 = fmaf(r67, r78, r9 * r66);
    r85 = r68 * r11;
    r85 = r85 * r63;
    r65 = fmaf(r79, r85, r65);
    r65 = fmaf(r11, r84, r65);
    r65 = fmaf(r11, r76, r65);
    r1 = fmaf(r5, r65, r1);
    r1 = fmaf(r3, r4, r1);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r1);
    r3 = r16 * r34;
    r85 = r19 * r24;
    r86 = 5.00000000000000000e-01;
    r87 = r18 * r21;
    r87 = fmaf(r86, r87, r86 * r85);
    r85 = r17 * r22;
    r88 = -5.00000000000000000e-01;
    r87 = fmaf(r88, r85, r87);
    r89 = r20 * r23;
    r87 = fmaf(r86, r89, r87);
    r89 = r17 * r24;
    r85 = r20 * r21;
    r85 = fmaf(r88, r85, r88 * r89);
    r89 = r19 * r22;
    r85 = fmaf(r88, r89, r85);
    r90 = r18 * r23;
    r85 = fmaf(r86, r90, r85);
    r90 = r29 * r85;
    r3 = fmaf(r16, r90, r87 * r3);
    r89 = r16 * r25;
    r91 = fmaf(r86, r31, r88 * r27);
    r91 = fmaf(r88, r28, r91);
    r91 = fmaf(r88, r30, r91);
    r92 = r16 * r32;
    r93 = r20 * r24;
    r94 = r17 * r21;
    r94 = fmaf(r88, r94, r86 * r93);
    r93 = r18 * r22;
    r94 = fmaf(r88, r93, r94);
    r95 = r19 * r23;
    r94 = fmaf(r88, r95, r94);
    r92 = r92 * r94;
    r89 = fmaf(r91, r89, r92);
    r3 = r3 + r89;
    r95 = r16 * r29;
    r95 = r95 * r94;
    r93 = r16 * r25;
    r93 = r93 * r87;
    r96 = r95 + r93;
    r97 = r32 * r41;
    r96 = fmaf(r85, r97, r96);
    r98 = r41 * r34;
    r96 = fmaf(r91, r98, r96);
    r96 = fmaf(r14, r96, r15 * r3);
    r3 = r29 * r87;
    r98 = -4.00000000000000000e+00;
    r3 = r3 * r98;
    r97 = r32 * r91;
    r99 = r98 * r97;
    r100 = r3 + r99;
    r96 = fmaf(r13, r100, r96);
    r100 = 6.00000000000000000e+00;
    r101 = r96 * r100;
    r101 = r101 * r62;
    r102 = r16 * r11;
    r103 = r25 * r41;
    r104 = r34 * r94;
    r105 = r41 * r104;
    r103 = fmaf(r85, r103, r105);
    r106 = r16 * r29;
    r106 = r106 * r91;
    r107 = r16 * r32;
    r107 = fmaf(r87, r107, r106);
    r103 = r103 + r107;
    r108 = r25 * r94;
    r108 = r108 * r98;
    r99 = r108 + r99;
    r99 = fmaf(r14, r99, r15 * r103);
    r103 = r16 * r34;
    r103 = fmaf(r91, r103, r93);
    r93 = r16 * r32;
    r93 = fmaf(r85, r93, r95);
    r103 = r103 + r93;
    r99 = fmaf(r13, r103, r99);
    r102 = r102 * r99;
    r103 = r16 * r10;
    r103 = r103 * r96;
    r103 = fmaf(r37, r103, r37 * r102);
    r102 = r16 * r25;
    r102 = r102 * r85;
    r104 = r16 * r104;
    r95 = r102 + r104;
    r107 = r107 + r95;
    r109 = r41 * r34;
    r109 = fmaf(r41, r90, r87 * r109);
    r109 = r109 + r89;
    r109 = fmaf(r13, r109, r14 * r107);
    r3 = r108 + r3;
    r109 = fmaf(r15, r3, r109);
    r12 = r36 * r12;
    r3 = 1.0 / r12;
    r108 = r41 * r3;
    r107 = r109 * r108;
    r87 = r11 * r11;
    r103 = fmaf(r87, r107, r103);
    r110 = r10 * r10;
    r110 = r110 * r109;
    r103 = fmaf(r108, r110, r103);
    r110 = r71 * r103;
    r107 = rsqrtf(r35);
    r35 = r42 + r35;
    r35 = 1.0 / r35;
    r42 = r35 * r57;
    r111 = r107 * r42;
    r111 = r111 * r64;
    r110 = fmaf(r111, r110, r63 * r101);
    r101 = r60 * r60;
    r112 = r64 * r101;
    r113 = -6.00000000000000000e+00;
    r114 = r109 * r113;
    r114 = r114 * r3;
    r110 = fmaf(r114, r112, r110);
    r115 = r10 * r10;
    r116 = -3.00000000000000000e+00;
    r53 = r51 * r53;
    r117 = 1.0 / r53;
    r115 = r115 * r116;
    r115 = r115 * r103;
    r115 = r115 * r107;
    r115 = r115 * r117;
    r110 = fmaf(r62, r115, r110);
    r118 = r11 * r47;
    r119 = r79 * r118;
    r120 = r11 * r107;
    r121 = r103 * r120;
    r121 = r121 * r42;
    r121 = fmaf(r118, r121, r99 * r119);
    r122 = r4 * r11;
    r123 = r117 * r62;
    r123 = r123 * r120;
    r122 = r122 * r103;
    r121 = fmaf(r123, r122, r121);
    r124 = r47 * r87;
    r125 = r108 * r101;
    r124 = r124 * r125;
    r121 = fmaf(r109, r124, r121);
    r110 = r110 + r121;
    r115 = r96 * r63;
    r115 = fmaf(r103, r111, r79 * r115);
    r112 = r109 * r64;
    r115 = fmaf(r125, r112, r115);
    r122 = r4 * r10;
    r122 = r122 * r10;
    r122 = r122 * r103;
    r122 = r122 * r107;
    r122 = r122 * r117;
    r115 = fmaf(r62, r122, r115);
    r121 = r121 + r115;
    r110 = fmaf(r8, r121, r68 * r110);
    r122 = r10 * r86;
    r122 = r122 * r73;
    r122 = r122 * r74;
    r122 = r122 * r107;
    r122 = r122 * r103;
    r122 = r122 * r35;
    r112 = r60 * r82;
    r112 = r112 * r88;
    r112 = r112 * r103;
    r112 = r112 * r73;
    r112 = r112 * r107;
    r110 = fmaf(r63, r112, r110);
    r126 = r67 * r103;
    r127 = r16 * r63;
    r127 = r127 * r120;
    r127 = r127 * r42;
    r110 = fmaf(r127, r126, r110);
    r128 = r60 * r88;
    r128 = r128 * r103;
    r128 = r128 * r73;
    r128 = r128 * r107;
    r110 = fmaf(r63, r128, r110);
    r129 = r67 * r11;
    r129 = r129 * r60;
    r129 = r129 * r60;
    r129 = r129 * r98;
    r129 = r129 * r3;
    r129 = r129 * r63;
    r130 = r67 * r96;
    r110 = fmaf(r119, r130, r110);
    r131 = r4 * r10;
    r131 = r131 * r109;
    r131 = r131 * r74;
    r110 = fmaf(r57, r131, r110);
    r132 = r4 * r10;
    r132 = r132 * r82;
    r132 = r132 * r109;
    r132 = r132 * r74;
    r110 = fmaf(r57, r132, r110);
    r133 = r67 * r103;
    r134 = r41 * r10;
    r134 = r134 * r123;
    r110 = fmaf(r134, r133, r110);
    r135 = r7 * r16;
    r135 = r135 * r66;
    r135 = fmaf(r121, r135, r6 * r121);
    r136 = 4.00000000000000000e+00;
    r70 = r70 * r136;
    r70 = r70 * r81;
    r69 = r69 * r71;
    r69 = r69 * r80;
    r135 = fmaf(r121, r70, r135);
    r135 = fmaf(r121, r69, r135);
    r137 = r10 * r135;
    r110 = fmaf(r76, r137, r110);
    r138 = r67 * r99;
    r138 = r138 * r63;
    r110 = fmaf(r79, r138, r110);
    r110 = r110 + r122;
    r110 = fmaf(r109, r129, r110);
    r110 = fmaf(r82, r122, r110);
    r110 = fmaf(r96, r76, r110);
    r110 = fmaf(r96, r84, r110);
    r138 = r0 * r110;
    r137 = r11 * r99;
    r137 = r137 * r100;
    r137 = r137 * r47;
    r133 = r71 * r103;
    r133 = r133 * r120;
    r133 = r133 * r42;
    r133 = fmaf(r118, r133, r62 * r137);
    r137 = r11 * r116;
    r137 = r137 * r103;
    r133 = fmaf(r123, r137, r133);
    r132 = r11 * r11;
    r132 = r132 * r60;
    r132 = r132 * r60;
    r132 = r132 * r47;
    r133 = fmaf(r114, r132, r133);
    r133 = r133 + r115;
    r121 = fmaf(r9, r121, r67 * r133);
    r133 = r86 * r103;
    r133 = r133 * r35;
    r133 = r133 * r75;
    r115 = r60 * r88;
    r115 = r115 * r47;
    r115 = r115 * r73;
    r115 = r115 * r120;
    r132 = r82 * r115;
    r137 = r11 * r135;
    r121 = fmaf(r76, r137, r121);
    r114 = r4 * r11;
    r114 = r114 * r82;
    r114 = r114 * r109;
    r114 = r114 * r74;
    r121 = fmaf(r57, r114, r121);
    r131 = r68 * r11;
    r131 = r131 * r60;
    r131 = r131 * r60;
    r131 = r131 * r98;
    r131 = r131 * r109;
    r131 = r131 * r3;
    r121 = fmaf(r63, r131, r121);
    r130 = r68 * r119;
    r122 = r68 * r103;
    r121 = fmaf(r127, r122, r121);
    r128 = r4 * r11;
    r128 = r128 * r109;
    r128 = r128 * r74;
    r121 = fmaf(r57, r128, r121);
    r126 = r68 * r103;
    r121 = fmaf(r134, r126, r121);
    r112 = r82 * r120;
    r121 = fmaf(r133, r112, r121);
    r139 = r68 * r99;
    r139 = r139 * r63;
    r121 = fmaf(r79, r139, r121);
    r121 = fmaf(r120, r133, r121);
    r121 = fmaf(r103, r132, r121);
    r121 = fmaf(r96, r130, r121);
    r121 = fmaf(r103, r115, r121);
    r121 = fmaf(r99, r84, r121);
    r121 = fmaf(r99, r76, r121);
    r139 = r5 * r121;
    r112 = r16 * r10;
    r104 = r106 + r104;
    r106 = r16 * r32;
    r126 = r19 * r24;
    r128 = r18 * r21;
    r128 = fmaf(r88, r128, r88 * r126);
    r126 = r17 * r22;
    r128 = fmaf(r86, r126, r128);
    r122 = r20 * r23;
    r128 = fmaf(r88, r122, r128);
    r106 = r106 * r128;
    r122 = r16 * r25;
    r126 = r17 * r24;
    r131 = r20 * r21;
    r131 = fmaf(r86, r131, r86 * r126);
    r126 = r19 * r22;
    r131 = fmaf(r86, r126, r131);
    r114 = r18 * r23;
    r131 = fmaf(r88, r114, r131);
    r122 = fmaf(r131, r122, r106);
    r104 = r104 + r122;
    r114 = r32 * r98;
    r114 = r114 * r131;
    r126 = r29 * r94;
    r126 = r126 * r98;
    r137 = r114 + r126;
    r137 = fmaf(r13, r137, r15 * r104);
    r104 = r41 * r34;
    r104 = fmaf(r41, r97, r131 * r104);
    r133 = r16 * r25;
    r133 = r133 * r94;
    r140 = r16 * r29;
    r140 = fmaf(r128, r140, r133);
    r104 = r104 + r140;
    r137 = fmaf(r14, r104, r137);
    r112 = r112 * r137;
    r104 = r10 * r10;
    r141 = r41 * r29;
    r141 = fmaf(r91, r141, r105);
    r141 = r141 + r122;
    r122 = r16 * r29;
    r122 = r122 * r131;
    r142 = r16 * r34;
    r142 = fmaf(r128, r142, r122);
    r142 = r142 + r89;
    r142 = fmaf(r14, r142, r13 * r141);
    r141 = r25 * r128;
    r89 = r98 * r141;
    r126 = r126 + r89;
    r142 = fmaf(r15, r126, r142);
    r104 = r104 * r142;
    r104 = fmaf(r108, r104, r37 * r112);
    r112 = r142 * r108;
    r104 = fmaf(r87, r112, r104);
    r126 = r16 * r11;
    r143 = r25 * r41;
    r143 = fmaf(r91, r143, r92);
    r92 = r41 * r34;
    r143 = fmaf(r128, r92, r143);
    r143 = r143 + r122;
    r92 = r16 * r34;
    r97 = fmaf(r16, r97, r131 * r92);
    r97 = r97 + r140;
    r97 = fmaf(r13, r97, r15 * r143);
    r89 = r114 + r89;
    r97 = fmaf(r14, r89, r97);
    r126 = r126 * r97;
    r104 = fmaf(r37, r126, r104);
    r126 = r4 * r10;
    r126 = r126 * r10;
    r126 = r126 * r104;
    r126 = r126 * r107;
    r126 = r126 * r117;
    r126 = fmaf(r62, r126, r104 * r111);
    r112 = r142 * r64;
    r126 = fmaf(r125, r112, r126);
    r89 = r137 * r63;
    r126 = fmaf(r79, r89, r126);
    r89 = r4 * r11;
    r89 = r89 * r104;
    r89 = fmaf(r123, r89, r97 * r119);
    r112 = r104 * r120;
    r112 = r112 * r42;
    r89 = fmaf(r118, r112, r89);
    r89 = fmaf(r142, r124, r89);
    r112 = r126 + r89;
    r114 = r71 * r104;
    r143 = r10 * r10;
    r92 = r116 * r104;
    r143 = r143 * r107;
    r143 = r143 * r117;
    r143 = r143 * r62;
    r143 = fmaf(r92, r143, r111 * r114);
    r114 = r113 * r142;
    r114 = r114 * r3;
    r114 = r114 * r64;
    r143 = fmaf(r101, r114, r143);
    r131 = r100 * r137;
    r131 = r131 * r62;
    r143 = fmaf(r63, r131, r143);
    r143 = r143 + r89;
    r143 = fmaf(r68, r143, r8 * r112);
    r89 = r67 * r97;
    r89 = r89 * r63;
    r143 = fmaf(r79, r89, r143);
    r131 = r60 * r88;
    r131 = r131 * r104;
    r131 = r131 * r73;
    r131 = r131 * r107;
    r143 = fmaf(r63, r131, r143);
    r114 = r4 * r10;
    r114 = r114 * r142;
    r114 = r114 * r74;
    r143 = fmaf(r57, r114, r143);
    r122 = r10 * r86;
    r122 = r122 * r82;
    r122 = r122 * r104;
    r122 = r122 * r107;
    r122 = r122 * r35;
    r143 = fmaf(r75, r122, r143);
    r91 = r67 * r104;
    r143 = fmaf(r134, r91, r143);
    r144 = r67 * r104;
    r143 = fmaf(r127, r144, r143);
    r145 = r10 * r86;
    r145 = r145 * r104;
    r145 = r145 * r107;
    r145 = r145 * r35;
    r143 = fmaf(r75, r145, r143);
    r146 = r4 * r10;
    r146 = r146 * r82;
    r146 = r146 * r142;
    r146 = r146 * r74;
    r143 = fmaf(r57, r146, r143);
    r147 = r67 * r137;
    r143 = fmaf(r119, r147, r143);
    r148 = r7 * r16;
    r148 = r148 * r66;
    r148 = fmaf(r112, r148, r6 * r112);
    r148 = fmaf(r112, r70, r148);
    r148 = fmaf(r112, r69, r148);
    r149 = r10 * r148;
    r143 = fmaf(r76, r149, r143);
    r150 = r60 * r82;
    r150 = r150 * r88;
    r150 = r150 * r104;
    r150 = r150 * r73;
    r150 = r150 * r107;
    r143 = fmaf(r63, r150, r143);
    r143 = fmaf(r142, r129, r143);
    r143 = fmaf(r137, r84, r143);
    r143 = fmaf(r137, r76, r143);
    r150 = r0 * r143;
    r149 = r11 * r100;
    r149 = r149 * r97;
    r149 = r149 * r47;
    r147 = r11 * r123;
    r147 = fmaf(r92, r147, r62 * r149);
    r149 = r11 * r11;
    r149 = r149 * r60;
    r149 = r149 * r60;
    r149 = r149 * r113;
    r149 = r149 * r142;
    r149 = r149 * r47;
    r147 = fmaf(r3, r149, r147);
    r92 = r71 * r104;
    r92 = r92 * r120;
    r92 = r92 * r42;
    r147 = fmaf(r118, r92, r147);
    r147 = r147 + r126;
    r147 = fmaf(r67, r147, r9 * r112);
    r112 = r4 * r11;
    r112 = r112 * r142;
    r112 = r112 * r74;
    r147 = fmaf(r57, r112, r147);
    r126 = r68 * r97;
    r126 = r126 * r63;
    r147 = fmaf(r79, r126, r147);
    r92 = r68 * r11;
    r92 = r92 * r60;
    r92 = r92 * r60;
    r92 = r92 * r98;
    r92 = r92 * r142;
    r92 = r92 * r3;
    r147 = fmaf(r63, r92, r147);
    r149 = r86 * r82;
    r149 = r149 * r104;
    r149 = r149 * r35;
    r149 = r149 * r75;
    r147 = fmaf(r120, r149, r147);
    r146 = r11 * r148;
    r147 = fmaf(r76, r146, r147);
    r145 = r86 * r104;
    r145 = r145 * r35;
    r145 = r145 * r75;
    r147 = fmaf(r120, r145, r147);
    r144 = r68 * r104;
    r147 = fmaf(r134, r144, r147);
    r91 = r68 * r104;
    r147 = fmaf(r127, r91, r147);
    r122 = r4 * r11;
    r122 = r122 * r82;
    r122 = r122 * r142;
    r122 = r122 * r74;
    r147 = fmaf(r57, r122, r147);
    r147 = fmaf(r104, r132, r147);
    r147 = fmaf(r104, r115, r147);
    r147 = fmaf(r137, r130, r147);
    r147 = fmaf(r97, r84, r147);
    r147 = fmaf(r97, r76, r147);
    r122 = r5 * r147;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r138,
                                          r139,
                                          r150,
                                          r122);
    r122 = r10 * r10;
    r150 = r25 * r98;
    r31 = fmaf(r88, r31, r86 * r27);
    r31 = fmaf(r86, r28, r31);
    r31 = fmaf(r86, r30, r31);
    r150 = r150 * r31;
    r90 = r98 * r90;
    r30 = r150 + r90;
    r28 = r16 * r32;
    r28 = r28 * r31;
    r133 = r133 + r28;
    r27 = r41 * r29;
    r133 = fmaf(r128, r27, r133);
    r139 = r41 * r34;
    r133 = fmaf(r85, r139, r133);
    r133 = fmaf(r13, r133, r15 * r30);
    r30 = r16 * r34;
    r30 = fmaf(r16, r141, r31 * r30);
    r30 = r30 + r93;
    r133 = fmaf(r14, r30, r133);
    r122 = r122 * r133;
    r30 = r16 * r10;
    r139 = r16 * r29;
    r139 = r139 * r31;
    r102 = r102 + r139;
    r27 = r32 * r41;
    r102 = fmaf(r128, r27, r102);
    r102 = r102 + r105;
    r94 = r32 * r94;
    r94 = r94 * r98;
    r90 = r94 + r90;
    r90 = fmaf(r13, r90, r14 * r102);
    r102 = r16 * r34;
    r102 = fmaf(r85, r102, r28);
    r102 = r102 + r140;
    r90 = fmaf(r15, r102, r90);
    r30 = r30 * r90;
    r30 = fmaf(r37, r30, r108 * r122);
    r122 = r133 * r108;
    r30 = fmaf(r87, r122, r30);
    r102 = r16 * r11;
    r139 = r106 + r139;
    r139 = r139 + r95;
    r95 = r41 * r34;
    r141 = fmaf(r41, r141, r31 * r95);
    r141 = r141 + r93;
    r141 = fmaf(r15, r141, r13 * r139);
    r94 = r150 + r94;
    r141 = fmaf(r14, r94, r141);
    r102 = r102 * r141;
    r30 = fmaf(r37, r102, r30);
    r102 = r30 * r120;
    r102 = r102 * r42;
    r102 = fmaf(r141, r119, r118 * r102);
    r122 = r4 * r11;
    r122 = r122 * r30;
    r102 = fmaf(r123, r122, r102);
    r102 = fmaf(r133, r124, r102);
    r122 = r133 * r64;
    r122 = fmaf(r30, r111, r125 * r122);
    r94 = r4 * r10;
    r94 = r94 * r10;
    r94 = r94 * r30;
    r94 = r94 * r107;
    r94 = r94 * r117;
    r122 = fmaf(r62, r94, r122);
    r14 = r90 * r63;
    r122 = fmaf(r79, r14, r122);
    r14 = r102 + r122;
    r94 = r113 * r133;
    r94 = r94 * r3;
    r94 = r94 * r64;
    r150 = r71 * r30;
    r150 = fmaf(r111, r150, r101 * r94);
    r94 = r10 * r10;
    r94 = r94 * r116;
    r94 = r94 * r30;
    r94 = r94 * r107;
    r94 = r94 * r117;
    r150 = fmaf(r62, r94, r150);
    r15 = r100 * r90;
    r15 = r15 * r62;
    r150 = fmaf(r63, r15, r150);
    r150 = r150 + r102;
    r150 = fmaf(r68, r150, r8 * r14);
    r102 = r7 * r16;
    r102 = r102 * r66;
    r102 = fmaf(r14, r102, r6 * r14);
    r102 = fmaf(r14, r69, r102);
    r102 = fmaf(r14, r70, r102);
    r15 = r10 * r102;
    r150 = fmaf(r76, r15, r150);
    r94 = r10 * r86;
    r94 = r94 * r30;
    r94 = r94 * r107;
    r94 = r94 * r35;
    r150 = fmaf(r75, r94, r150);
    r139 = r60 * r82;
    r139 = r139 * r88;
    r139 = r139 * r30;
    r139 = r139 * r73;
    r139 = r139 * r107;
    r150 = fmaf(r63, r139, r150);
    r13 = r4 * r10;
    r13 = r13 * r133;
    r13 = r13 * r74;
    r150 = fmaf(r57, r13, r150);
    r93 = r67 * r30;
    r150 = fmaf(r134, r93, r150);
    r95 = r67 * r90;
    r150 = fmaf(r119, r95, r150);
    r31 = r10 * r86;
    r31 = r31 * r82;
    r31 = r31 * r30;
    r31 = r31 * r107;
    r31 = r31 * r35;
    r150 = fmaf(r75, r31, r150);
    r106 = r67 * r141;
    r106 = r106 * r63;
    r150 = fmaf(r79, r106, r150);
    r140 = r60 * r88;
    r140 = r140 * r30;
    r140 = r140 * r73;
    r140 = r140 * r107;
    r150 = fmaf(r63, r140, r150);
    r28 = r4 * r10;
    r28 = r28 * r82;
    r28 = r28 * r133;
    r28 = r28 * r74;
    r150 = fmaf(r57, r28, r150);
    r85 = r67 * r30;
    r150 = fmaf(r127, r85, r150);
    r150 = fmaf(r133, r129, r150);
    r150 = fmaf(r90, r84, r150);
    r150 = fmaf(r90, r76, r150);
    r85 = r0 * r150;
    r28 = r71 * r30;
    r28 = r28 * r120;
    r28 = r28 * r42;
    r140 = r11 * r100;
    r140 = r140 * r141;
    r140 = r140 * r47;
    r140 = fmaf(r62, r140, r118 * r28);
    r28 = r11 * r11;
    r28 = r28 * r60;
    r28 = r28 * r60;
    r28 = r28 * r113;
    r28 = r28 * r133;
    r28 = r28 * r47;
    r140 = fmaf(r3, r28, r140);
    r106 = r11 * r116;
    r106 = r106 * r30;
    r140 = fmaf(r123, r106, r140);
    r140 = r140 + r122;
    r140 = fmaf(r67, r140, r9 * r14);
    r14 = r68 * r11;
    r14 = r14 * r60;
    r14 = r14 * r60;
    r14 = r14 * r98;
    r14 = r14 * r133;
    r14 = r14 * r3;
    r140 = fmaf(r63, r14, r140);
    r122 = r86 * r30;
    r122 = r122 * r35;
    r122 = r122 * r75;
    r140 = fmaf(r120, r122, r140);
    r106 = r4 * r11;
    r106 = r106 * r82;
    r106 = r106 * r133;
    r106 = r106 * r74;
    r140 = fmaf(r57, r106, r140);
    r28 = r68 * r30;
    r140 = fmaf(r134, r28, r140);
    r31 = r68 * r141;
    r31 = r31 * r63;
    r140 = fmaf(r79, r31, r140);
    r95 = r4 * r11;
    r95 = r95 * r133;
    r95 = r95 * r74;
    r140 = fmaf(r57, r95, r140);
    r93 = r68 * r30;
    r140 = fmaf(r127, r93, r140);
    r13 = r11 * r102;
    r140 = fmaf(r76, r13, r140);
    r139 = r86 * r82;
    r139 = r139 * r30;
    r139 = r139 * r35;
    r139 = r139 * r75;
    r140 = fmaf(r120, r139, r140);
    r140 = fmaf(r30, r115, r140);
    r140 = fmaf(r141, r84, r140);
    r140 = fmaf(r90, r130, r140);
    r140 = fmaf(r30, r132, r140);
    r140 = fmaf(r141, r76, r140);
    r139 = r5 * r140;
    r13 = r26 * r113;
    r13 = r13 * r3;
    r13 = r13 * r64;
    r93 = r43 * r100;
    r93 = r93 * r62;
    r93 = fmaf(r63, r93, r101 * r13);
    r13 = r10 * r10;
    r95 = r26 * r108;
    r31 = r16 * r39;
    r31 = r31 * r11;
    r31 = fmaf(r37, r31, r87 * r95);
    r95 = r26 * r10;
    r95 = r95 * r10;
    r31 = fmaf(r108, r95, r31);
    r28 = r16 * r43;
    r28 = r28 * r10;
    r31 = fmaf(r37, r28, r31);
    r13 = r13 * r116;
    r13 = r13 * r31;
    r13 = r13 * r107;
    r13 = r13 * r117;
    r93 = fmaf(r62, r13, r93);
    r28 = r71 * r31;
    r93 = fmaf(r111, r28, r93);
    r95 = fmaf(r39, r119, r26 * r124);
    r106 = r4 * r11;
    r106 = r106 * r31;
    r95 = fmaf(r123, r106, r95);
    r122 = r31 * r120;
    r122 = r122 * r42;
    r95 = fmaf(r118, r122, r95);
    r93 = r93 + r95;
    r28 = r26 * r64;
    r13 = r43 * r63;
    r13 = fmaf(r79, r13, r125 * r28);
    r28 = r4 * r10;
    r28 = r28 * r10;
    r28 = r28 * r31;
    r28 = r28 * r107;
    r28 = r28 * r117;
    r13 = fmaf(r62, r28, r13);
    r13 = fmaf(r31, r111, r13);
    r95 = r95 + r13;
    r93 = fmaf(r8, r95, r68 * r93);
    r28 = r4 * r26;
    r28 = r28 * r10;
    r28 = r28 * r74;
    r93 = fmaf(r57, r28, r93);
    r122 = r60 * r82;
    r122 = r122 * r88;
    r122 = r122 * r31;
    r122 = r122 * r73;
    r122 = r122 * r107;
    r93 = fmaf(r63, r122, r93);
    r106 = r67 * r31;
    r93 = fmaf(r134, r106, r93);
    r14 = r31 * r127;
    r94 = r67 * r43;
    r93 = fmaf(r119, r94, r93);
    r15 = r60 * r88;
    r15 = r15 * r31;
    r15 = r15 * r73;
    r15 = r15 * r107;
    r93 = fmaf(r63, r15, r93);
    r105 = r10 * r86;
    r105 = r105 * r82;
    r105 = r105 * r31;
    r105 = r105 * r107;
    r105 = r105 * r35;
    r93 = fmaf(r75, r105, r93);
    r27 = r10 * r86;
    r27 = r27 * r31;
    r27 = r27 * r107;
    r27 = r27 * r35;
    r93 = fmaf(r75, r27, r93);
    r128 = r67 * r39;
    r128 = r128 * r63;
    r93 = fmaf(r79, r128, r93);
    r138 = r7 * r16;
    r138 = r138 * r66;
    r138 = fmaf(r6, r95, r95 * r138);
    r138 = fmaf(r95, r70, r138);
    r138 = fmaf(r95, r69, r138);
    r91 = r10 * r138;
    r93 = fmaf(r76, r91, r93);
    r144 = r4 * r26;
    r144 = r144 * r10;
    r144 = r144 * r82;
    r144 = r144 * r74;
    r93 = fmaf(r57, r144, r93);
    r93 = fmaf(r43, r76, r93);
    r93 = fmaf(r43, r84, r93);
    r93 = fmaf(r67, r14, r93);
    r93 = fmaf(r26, r129, r93);
    r144 = r0 * r93;
    r91 = r26 * r11;
    r91 = r91 * r11;
    r91 = r91 * r60;
    r91 = r91 * r60;
    r91 = r91 * r113;
    r91 = r91 * r47;
    r128 = r39 * r11;
    r128 = r128 * r100;
    r128 = r128 * r47;
    r128 = fmaf(r62, r128, r3 * r91);
    r91 = r11 * r116;
    r91 = r91 * r31;
    r128 = fmaf(r123, r91, r128);
    r27 = r71 * r31;
    r27 = r27 * r120;
    r27 = r27 * r42;
    r128 = fmaf(r118, r27, r128);
    r128 = r128 + r13;
    r95 = fmaf(r9, r95, r67 * r128);
    r128 = r4 * r26;
    r128 = r128 * r11;
    r128 = r128 * r74;
    r95 = fmaf(r57, r128, r95);
    r13 = r4 * r26;
    r13 = r13 * r11;
    r13 = r13 * r82;
    r13 = r13 * r74;
    r95 = fmaf(r57, r13, r95);
    r27 = r68 * r31;
    r95 = fmaf(r134, r27, r95);
    r91 = r86 * r31;
    r91 = r91 * r35;
    r91 = r91 * r75;
    r95 = fmaf(r120, r91, r95);
    r105 = r68 * r26;
    r105 = r105 * r11;
    r105 = r105 * r60;
    r105 = r105 * r60;
    r105 = r105 * r98;
    r105 = r105 * r3;
    r95 = fmaf(r63, r105, r95);
    r15 = r68 * r39;
    r15 = r15 * r63;
    r95 = fmaf(r79, r15, r95);
    r94 = r86 * r82;
    r94 = r94 * r31;
    r94 = r94 * r35;
    r94 = r94 * r75;
    r95 = fmaf(r120, r94, r95);
    r106 = r11 * r138;
    r95 = fmaf(r76, r106, r95);
    r95 = fmaf(r31, r115, r95);
    r95 = fmaf(r31, r132, r95);
    r95 = fmaf(r39, r76, r95);
    r95 = fmaf(r39, r84, r95);
    r95 = fmaf(r68, r14, r95);
    r95 = fmaf(r43, r130, r95);
    r106 = r5 * r95;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r85,
                                          r139,
                                          r144,
                                          r106);
    r106 = r40 * r108;
    r144 = r16 * r44;
    r144 = r144 * r11;
    r144 = fmaf(r37, r144, r87 * r106);
    r106 = r40 * r10;
    r106 = r106 * r10;
    r144 = fmaf(r108, r106, r144);
    r139 = r16 * r59;
    r139 = r139 * r10;
    r144 = fmaf(r37, r139, r144);
    r139 = r144 * r120;
    r139 = r139 * r42;
    r139 = fmaf(r40, r124, r118 * r139);
    r106 = r4 * r11;
    r106 = r106 * r144;
    r139 = fmaf(r123, r106, r139);
    r139 = fmaf(r44, r119, r139);
    r106 = r40 * r64;
    r106 = fmaf(r125, r106, r144 * r111);
    r85 = r59 * r63;
    r106 = fmaf(r79, r85, r106);
    r94 = r4 * r10;
    r94 = r94 * r10;
    r94 = r94 * r144;
    r94 = r94 * r107;
    r94 = r94 * r117;
    r106 = fmaf(r62, r94, r106);
    r94 = r139 + r106;
    r85 = r71 * r144;
    r15 = r40 * r113;
    r15 = r15 * r3;
    r15 = r15 * r64;
    r15 = fmaf(r101, r15, r111 * r85);
    r85 = r59 * r100;
    r85 = r85 * r62;
    r15 = fmaf(r63, r85, r15);
    r105 = r10 * r10;
    r105 = r105 * r116;
    r105 = r105 * r144;
    r105 = r105 * r107;
    r105 = r105 * r117;
    r15 = fmaf(r62, r105, r15);
    r15 = r15 + r139;
    r15 = fmaf(r68, r15, r8 * r94);
    r139 = r7 * r16;
    r139 = r139 * r66;
    r139 = fmaf(r94, r139, r6 * r94);
    r139 = fmaf(r94, r70, r139);
    r139 = fmaf(r94, r69, r139);
    r105 = r10 * r139;
    r15 = fmaf(r76, r105, r15);
    r85 = r67 * r44;
    r85 = r85 * r63;
    r15 = fmaf(r79, r85, r15);
    r91 = r60 * r88;
    r91 = r91 * r144;
    r91 = r91 * r73;
    r91 = r91 * r107;
    r15 = fmaf(r63, r91, r15);
    r14 = r67 * r144;
    r15 = fmaf(r127, r14, r15);
    r27 = r10 * r86;
    r27 = r27 * r144;
    r27 = r27 * r107;
    r27 = r27 * r35;
    r15 = fmaf(r75, r27, r15);
    r13 = r67 * r144;
    r15 = fmaf(r134, r13, r15);
    r128 = r60 * r82;
    r128 = r128 * r88;
    r128 = r128 * r144;
    r128 = r128 * r73;
    r128 = r128 * r107;
    r15 = fmaf(r63, r128, r15);
    r122 = r10 * r86;
    r122 = r122 * r82;
    r122 = r122 * r144;
    r122 = r122 * r107;
    r122 = r122 * r35;
    r15 = fmaf(r75, r122, r15);
    r28 = r4 * r40;
    r28 = r28 * r10;
    r28 = r28 * r74;
    r15 = fmaf(r57, r28, r15);
    r145 = r67 * r59;
    r15 = fmaf(r119, r145, r15);
    r146 = r4 * r40;
    r146 = r146 * r10;
    r146 = r146 * r82;
    r146 = r146 * r74;
    r15 = fmaf(r57, r146, r15);
    r15 = fmaf(r59, r84, r15);
    r15 = fmaf(r59, r76, r15);
    r15 = fmaf(r40, r129, r15);
    r146 = r0 * r15;
    r145 = r71 * r144;
    r145 = r145 * r120;
    r145 = r145 * r42;
    r28 = r40 * r11;
    r28 = r28 * r11;
    r28 = r28 * r60;
    r28 = r28 * r60;
    r28 = r28 * r113;
    r28 = r28 * r47;
    r28 = fmaf(r3, r28, r118 * r145);
    r145 = r44 * r11;
    r145 = r145 * r100;
    r145 = r145 * r47;
    r28 = fmaf(r62, r145, r28);
    r122 = r11 * r116;
    r122 = r122 * r144;
    r28 = fmaf(r123, r122, r28);
    r28 = r28 + r106;
    r28 = fmaf(r67, r28, r9 * r94);
    r94 = r68 * r44;
    r94 = r94 * r63;
    r28 = fmaf(r79, r94, r28);
    r106 = r4 * r40;
    r106 = r106 * r11;
    r106 = r106 * r74;
    r28 = fmaf(r57, r106, r28);
    r122 = r86 * r144;
    r122 = r122 * r35;
    r122 = r122 * r75;
    r28 = fmaf(r120, r122, r28);
    r145 = r68 * r144;
    r28 = fmaf(r127, r145, r28);
    r128 = r86 * r82;
    r128 = r128 * r144;
    r128 = r128 * r35;
    r128 = r128 * r75;
    r28 = fmaf(r120, r128, r28);
    r13 = r68 * r40;
    r13 = r13 * r11;
    r13 = r13 * r60;
    r13 = r13 * r60;
    r13 = r13 * r98;
    r13 = r13 * r3;
    r28 = fmaf(r63, r13, r28);
    r27 = r68 * r144;
    r28 = fmaf(r134, r27, r28);
    r14 = r4 * r40;
    r14 = r14 * r11;
    r14 = r14 * r82;
    r14 = r14 * r74;
    r28 = fmaf(r57, r14, r28);
    r91 = r11 * r139;
    r28 = fmaf(r76, r91, r28);
    r28 = fmaf(r44, r84, r28);
    r28 = fmaf(r44, r76, r28);
    r28 = fmaf(r59, r130, r28);
    r28 = fmaf(r144, r132, r28);
    r28 = fmaf(r144, r115, r28);
    r91 = r5 * r28;
    r14 = r16 * r46;
    r14 = r14 * r11;
    r27 = r38 * r108;
    r27 = fmaf(r87, r27, r37 * r14);
    r14 = r16 * r58;
    r14 = r14 * r10;
    r27 = fmaf(r37, r14, r27);
    r13 = r38 * r10;
    r13 = r13 * r10;
    r27 = fmaf(r108, r13, r27);
    r13 = r27 * r120;
    r13 = r13 * r42;
    r13 = fmaf(r46, r119, r118 * r13);
    r14 = r4 * r11;
    r14 = r14 * r27;
    r13 = fmaf(r123, r14, r13);
    r13 = fmaf(r38, r124, r13);
    r14 = r38 * r64;
    r128 = r4 * r10;
    r128 = r128 * r10;
    r128 = r128 * r27;
    r128 = r128 * r107;
    r128 = r128 * r117;
    r128 = fmaf(r62, r128, r125 * r14);
    r14 = r58 * r63;
    r128 = fmaf(r79, r14, r128);
    r128 = fmaf(r27, r111, r128);
    r14 = r13 + r128;
    r145 = r38 * r113;
    r145 = r145 * r3;
    r145 = r145 * r64;
    r122 = r10 * r10;
    r122 = r122 * r116;
    r122 = r122 * r27;
    r122 = r122 * r107;
    r122 = r122 * r117;
    r122 = fmaf(r62, r122, r101 * r145);
    r145 = r58 * r100;
    r145 = r145 * r62;
    r122 = fmaf(r63, r145, r122);
    r106 = r71 * r27;
    r122 = fmaf(r111, r106, r122);
    r122 = r122 + r13;
    r122 = fmaf(r68, r122, r8 * r14);
    r13 = r4 * r38;
    r13 = r13 * r10;
    r13 = r13 * r82;
    r13 = r13 * r74;
    r122 = fmaf(r57, r13, r122);
    r106 = r4 * r38;
    r106 = r106 * r10;
    r106 = r106 * r74;
    r122 = fmaf(r57, r106, r122);
    r145 = r10 * r86;
    r145 = r145 * r82;
    r145 = r145 * r27;
    r145 = r145 * r107;
    r145 = r145 * r35;
    r122 = fmaf(r75, r145, r122);
    r94 = r60 * r88;
    r94 = r94 * r27;
    r94 = r94 * r73;
    r94 = r94 * r107;
    r122 = fmaf(r63, r94, r122);
    r85 = r67 * r27;
    r122 = fmaf(r134, r85, r122);
    r105 = r67 * r27;
    r122 = fmaf(r127, r105, r122);
    r149 = r67 * r46;
    r149 = r149 * r63;
    r122 = fmaf(r79, r149, r122);
    r92 = r67 * r58;
    r122 = fmaf(r119, r92, r122);
    r126 = r60 * r82;
    r126 = r126 * r88;
    r126 = r126 * r27;
    r126 = r126 * r73;
    r126 = r126 * r107;
    r122 = fmaf(r63, r126, r122);
    r112 = r10 * r86;
    r112 = r112 * r27;
    r112 = r112 * r107;
    r112 = r112 * r35;
    r122 = fmaf(r75, r112, r122);
    r114 = r7 * r16;
    r114 = r114 * r66;
    r114 = fmaf(r6, r14, r14 * r114);
    r114 = fmaf(r14, r70, r114);
    r114 = fmaf(r14, r69, r114);
    r131 = r10 * r114;
    r122 = fmaf(r76, r131, r122);
    r122 = fmaf(r58, r76, r122);
    r122 = fmaf(r58, r84, r122);
    r122 = fmaf(r38, r129, r122);
    r131 = r0 * r122;
    r112 = r71 * r27;
    r112 = r112 * r120;
    r112 = r112 * r42;
    r126 = r46 * r11;
    r126 = r126 * r100;
    r126 = r126 * r47;
    r126 = fmaf(r62, r126, r118 * r112);
    r112 = r11 * r116;
    r112 = r112 * r27;
    r126 = fmaf(r123, r112, r126);
    r92 = r38 * r11;
    r92 = r92 * r11;
    r92 = r92 * r60;
    r92 = r92 * r60;
    r92 = r92 * r113;
    r92 = r92 * r47;
    r126 = fmaf(r3, r92, r126);
    r126 = r126 + r128;
    r126 = fmaf(r67, r126, r9 * r14);
    r14 = r86 * r82;
    r14 = r14 * r27;
    r14 = r14 * r35;
    r14 = r14 * r75;
    r126 = fmaf(r120, r14, r126);
    r128 = r86 * r27;
    r128 = r128 * r35;
    r128 = r128 * r75;
    r126 = fmaf(r120, r128, r126);
    r92 = r68 * r27;
    r126 = fmaf(r134, r92, r126);
    r112 = r68 * r27;
    r126 = fmaf(r127, r112, r126);
    r149 = r68 * r46;
    r149 = r149 * r63;
    r126 = fmaf(r79, r149, r126);
    r105 = r4 * r38;
    r105 = r105 * r11;
    r105 = r105 * r82;
    r105 = r105 * r74;
    r126 = fmaf(r57, r105, r126);
    r85 = r4 * r38;
    r85 = r85 * r11;
    r85 = r85 * r74;
    r126 = fmaf(r57, r85, r126);
    r94 = r11 * r114;
    r126 = fmaf(r76, r94, r126);
    r145 = r68 * r38;
    r145 = r145 * r11;
    r145 = r145 * r60;
    r145 = r145 * r60;
    r145 = r145 * r98;
    r145 = r145 * r3;
    r126 = fmaf(r63, r145, r126);
    r126 = fmaf(r46, r84, r126);
    r126 = fmaf(r27, r132, r126);
    r126 = fmaf(r27, r115, r126);
    r126 = fmaf(r58, r130, r126);
    r126 = fmaf(r46, r76, r126);
    r145 = r5 * r126;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r146,
                                          r91,
                                          r131,
                                          r145);
    r145 = r0 * r4;
    r145 = r145 * r2;
    r131 = r4 * r1;
    r91 = r5 * r131;
    r145 = fmaf(r121, r91, r110 * r145);
    r146 = r0 * r4;
    r146 = r146 * r2;
    r146 = fmaf(r147, r91, r143 * r146);
    r94 = r0 * r4;
    r94 = r94 * r2;
    r94 = fmaf(r140, r91, r150 * r94);
    r85 = r0 * r4;
    r85 = r85 * r2;
    r85 = fmaf(r95, r91, r93 * r85);
    WriteSum4<float, float>((float*)inout_shared, r145, r146, r94, r85);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r0 * r4;
    r85 = r85 * r2;
    r85 = fmaf(r28, r91, r15 * r85);
    r94 = r0 * r4;
    r94 = r94 * r2;
    r94 = fmaf(r126, r91, r122 * r94);
    WriteSum2<float, float>((float*)inout_shared, r85, r94);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r94 = r0 * r0;
    r85 = r110 * r110;
    r146 = r5 * r5;
    r145 = r121 * r121;
    r145 = fmaf(r146, r145, r94 * r85);
    r85 = r143 * r143;
    r105 = r147 * r147;
    r105 = fmaf(r146, r105, r94 * r85);
    r85 = r140 * r140;
    r149 = r150 * r150;
    r149 = fmaf(r94, r149, r146 * r85);
    r85 = r93 * r93;
    r112 = r95 * r95;
    r112 = fmaf(r146, r112, r94 * r85);
    WriteSum4<float, float>((float*)inout_shared, r145, r105, r149, r112);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r112 = r15 * r15;
    r149 = r28 * r28;
    r149 = fmaf(r146, r149, r94 * r112);
    r112 = r126 * r126;
    r105 = r122 * r122;
    r105 = fmaf(r94, r105, r146 * r112);
    WriteSum2<float, float>((float*)inout_shared, r149, r105);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r105 = r121 * r147;
    r149 = r110 * r143;
    r149 = fmaf(r94, r149, r146 * r105);
    r105 = r110 * r150;
    r112 = r121 * r140;
    r112 = fmaf(r146, r112, r94 * r105);
    r105 = r121 * r95;
    r145 = r110 * r93;
    r145 = fmaf(r94, r145, r146 * r105);
    r105 = r121 * r28;
    r85 = r110 * r15;
    r85 = fmaf(r94, r85, r146 * r105);
    WriteSum4<float, float>((float*)inout_shared, r149, r112, r145, r85);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r110 * r122;
    r145 = r121 * r126;
    r145 = fmaf(r146, r145, r94 * r85);
    r85 = r147 * r140;
    r112 = r143 * r150;
    r112 = fmaf(r94, r112, r146 * r85);
    r85 = r143 * r93;
    r149 = r147 * r95;
    r149 = fmaf(r146, r149, r94 * r85);
    r85 = r143 * r15;
    r105 = r147 * r28;
    r105 = fmaf(r146, r105, r94 * r85);
    WriteSum4<float, float>((float*)inout_shared, r145, r112, r149, r105);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r105 = r147 * r126;
    r149 = r143 * r122;
    r149 = fmaf(r94, r149, r146 * r105);
    r105 = r140 * r95;
    r112 = r150 * r93;
    r112 = fmaf(r94, r112, r146 * r105);
    r105 = r150 * r15;
    r145 = r140 * r28;
    r145 = fmaf(r146, r145, r94 * r105);
    r105 = r140 * r126;
    r85 = r150 * r122;
    r85 = fmaf(r94, r85, r146 * r105);
    WriteSum4<float, float>((float*)inout_shared, r149, r112, r145, r85);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r95 * r28;
    r145 = r93 * r15;
    r145 = fmaf(r94, r145, r146 * r85);
    r85 = r93 * r122;
    r112 = r95 * r126;
    r112 = fmaf(r146, r112, r94 * r85);
    r85 = r28 * r126;
    r149 = r15 * r122;
    r149 = fmaf(r94, r149, r146 * r85);
    WriteSum3<float, float>((float*)inout_shared, r145, r112, r149);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r149 = r0 * r10;
    r149 = r149 * r66;
    r149 = r149 * r76;
    r112 = r5 * r11;
    r112 = r112 * r66;
    r112 = r112 * r76;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          0 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r52,
                                          r65,
                                          r149,
                                          r112);
    r112 = r5 * r78;
    r149 = r0 * r10;
    r149 = r149 * r76;
    r149 = r149 * r80;
    r145 = r5 * r11;
    r145 = r145 * r76;
    r145 = r145 * r80;
    r85 = r0 * r11;
    r85 = r85 * r63;
    r85 = r85 * r79;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          4 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r149,
                                          r145,
                                          r85,
                                          r112);
    r112 = r0 * r72;
    r85 = r5 * r11;
    r85 = r85 * r63;
    r85 = r85 * r79;
    r145 = r0 * r10;
    r145 = r145 * r76;
    r145 = r145 * r81;
    r149 = r5 * r11;
    r149 = r149 * r76;
    r149 = r149 * r81;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          8 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r112,
                                          r85,
                                          r145,
                                          r149);
    r149 = r0 * r66;
    r145 = r5 * r66;
    r85 = r0 * r10;
    r85 = r85 * r76;
    r85 = r85 * r83;
    r112 = r5 * r11;
    r112 = r112 * r76;
    r112 = r112 * r83;
    WriteIdx4<1024, float, float, float4>(
        out_focal_and_extra_jac,
        12 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r85,
        r112,
        r149,
        r145);
    r145 = r4 * r52;
    r145 = r145 * r2;
    r131 = r65 * r131;
    r149 = r0 * r4;
    r149 = r149 * r10;
    r149 = r149 * r66;
    r149 = r149 * r2;
    r112 = r11 * r66;
    r112 = r112 * r76;
    r112 = fmaf(r91, r112, r76 * r149);
    r149 = r0 * r4;
    r149 = r149 * r10;
    r149 = r149 * r2;
    r149 = r149 * r76;
    r85 = r11 * r76;
    r85 = r85 * r80;
    r85 = fmaf(r91, r85, r80 * r149);
    WriteSum4<float, float>((float*)inout_shared, r145, r131, r112, r85);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           0 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r0 * r41;
    r85 = r85 * r11;
    r85 = r85 * r2;
    r85 = r85 * r62;
    r85 = fmaf(r63, r85, r78 * r91);
    r112 = r0 * r4;
    r112 = r112 * r72;
    r131 = r5 * r41;
    r131 = r131 * r11;
    r131 = r131 * r1;
    r131 = r131 * r62;
    r131 = fmaf(r63, r131, r2 * r112);
    r112 = r0 * r4;
    r112 = r112 * r10;
    r112 = r112 * r2;
    r112 = r112 * r76;
    r1 = r11 * r76;
    r1 = r1 * r81;
    r1 = fmaf(r91, r1, r81 * r112);
    r112 = r0 * r4;
    r112 = r112 * r10;
    r112 = r112 * r2;
    r112 = r112 * r76;
    r145 = r11 * r76;
    r145 = r145 * r83;
    r145 = fmaf(r91, r145, r83 * r112);
    WriteSum4<float, float>((float*)inout_shared, r85, r131, r1, r145);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           4 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r145 = r0 * r4;
    r145 = r145 * r66;
    r145 = r145 * r2;
    r1 = r66 * r91;
    WriteSum2<float, float>((float*)inout_shared, r145, r1);
  };
  FlushSumShared<2, float>(out_focal_and_extra_njtr,
                           8 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r52 * r52;
    r145 = r65 * r65;
    r131 = r11 * r62;
    r131 = r131 * r80;
    r131 = r131 * r146;
    r85 = r62 * r80;
    r85 = r85 * r94;
    r85 = fmaf(r64, r85, r118 * r131);
    r131 = r11 * r62;
    r131 = r131 * r146;
    r131 = r131 * r118;
    r112 = r62 * r94;
    r112 = r112 * r64;
    r112 = fmaf(r83, r112, r83 * r131);
    WriteSum4<float, float>((float*)inout_shared, r1, r145, r85, r112);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_diag,
                           0 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r10 * r10;
    r12 = r36 * r12;
    r12 = 1.0 / r12;
    r53 = r51 * r53;
    r53 = 1.0 / r53;
    r85 = r85 * r60;
    r85 = r85 * r60;
    r85 = r85 * r136;
    r85 = r85 * r12;
    r85 = r85 * r53;
    r85 = r85 * r101;
    r85 = r85 * r87;
    r53 = r78 * r146;
    r12 = fmaf(r78, r53, r94 * r85);
    r136 = r72 * r94;
    r85 = fmaf(r72, r136, r146 * r85);
    r51 = r81 * r81;
    r36 = r11 * r62;
    r36 = r36 * r146;
    r36 = r36 * r118;
    r145 = r62 * r94;
    r145 = r145 * r64;
    r145 = fmaf(r51, r145, r51 * r36);
    r1 = r83 * r83;
    r131 = r62 * r94;
    r131 = r131 * r64;
    r1 = fmaf(r1, r131, r36 * r1);
    WriteSum4<float, float>((float*)inout_shared, r12, r85, r145, r1);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_diag,
                           4 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r80 * r94;
    r85 = r80 * r146;
    WriteSum2<float, float>((float*)inout_shared, r1, r85);
  };
  FlushSumShared<2, float>(out_focal_and_extra_precond_diag,
                           8 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = 0.00000000000000000e+00;
    r1 = r0 * r10;
    r1 = r1 * r66;
    r1 = r1 * r52;
    r1 = r1 * r76;
    r12 = r0 * r10;
    r12 = r12 * r52;
    r12 = r12 * r76;
    r12 = r12 * r80;
    r149 = r0 * r11;
    r149 = r149 * r52;
    r149 = r149 * r63;
    r149 = r149 * r79;
    WriteSum4<float, float>((float*)inout_shared, r85, r1, r12, r149);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           0 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r0 * r72;
    r72 = r72 * r52;
    r149 = r0 * r66;
    r149 = r149 * r52;
    r12 = r0 * r10;
    r12 = r12 * r52;
    r12 = r12 * r76;
    r12 = r12 * r81;
    r1 = r0 * r10;
    r1 = r1 * r52;
    r1 = r1 * r76;
    r1 = r1 * r83;
    WriteSum4<float, float>((float*)inout_shared, r72, r12, r1, r149);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           4 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = r5 * r78;
    r78 = r78 * r65;
    r149 = r5 * r11;
    r149 = r149 * r66;
    r149 = r149 * r65;
    r149 = r149 * r76;
    r1 = r5 * r11;
    r1 = r1 * r65;
    r1 = r1 * r76;
    r1 = r1 * r80;
    WriteSum4<float, float>((float*)inout_shared, r85, r149, r1, r78);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           8 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = r5 * r11;
    r78 = r78 * r65;
    r78 = r78 * r63;
    r78 = r78 * r79;
    r1 = r5 * r11;
    r1 = r1 * r65;
    r1 = r1 * r76;
    r1 = r1 * r81;
    r149 = r5 * r11;
    r149 = r149 * r65;
    r149 = r149 * r76;
    r149 = r149 * r83;
    WriteSum4<float, float>((float*)inout_shared, r78, r1, r149, r85);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           12 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r5 * r66;
    r85 = r85 * r65;
    r65 = r11 * r62;
    r65 = r65 * r81;
    r65 = r65 * r146;
    r149 = r62 * r81;
    r149 = r149 * r94;
    r149 = fmaf(r64, r149, r118 * r65);
    r65 = r16 * r10;
    r65 = r65 * r10;
    r65 = r65 * r11;
    r65 = r65 * r60;
    r65 = r65 * r66;
    r65 = r65 * r3;
    r65 = r65 * r117;
    r65 = r65 * r94;
    r1 = r11 * r53;
    r78 = r76 * r1;
    r65 = fmaf(r66, r78, r101 * r65);
    r12 = r136 * r77;
    r72 = r16 * r10;
    r72 = r72 * r60;
    r72 = r72 * r66;
    r72 = r72 * r3;
    r72 = r72 * r117;
    r72 = r72 * r146;
    r72 = r72 * r101;
    r72 = fmaf(r87, r72, r66 * r12);
    WriteSum4<float, float>((float*)inout_shared, r85, r149, r65, r72);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           16 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r10 * r76;
    r72 = r72 * r80;
    r72 = r72 * r94;
    r65 = r11 * r76;
    r65 = r65 * r80;
    r65 = r65 * r146;
    r149 = r66 * r83;
    r85 = r62 * r94;
    r85 = r85 * r64;
    r85 = fmaf(r149, r85, r149 * r36);
    WriteSum4<float, float>((float*)inout_shared, r112, r85, r72, r65);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           20 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r16 * r10;
    r65 = r65 * r10;
    r65 = r65 * r11;
    r65 = r65 * r60;
    r65 = r65 * r3;
    r65 = r65 * r117;
    r65 = r65 * r80;
    r65 = r65 * r94;
    r65 = fmaf(r80, r78, r101 * r65);
    r72 = r16 * r10;
    r72 = r72 * r60;
    r72 = r72 * r3;
    r72 = r72 * r117;
    r72 = r72 * r80;
    r72 = r72 * r146;
    r72 = r72 * r101;
    r72 = fmaf(r87, r72, r80 * r12);
    WriteSum4<float, float>((float*)inout_shared, r65, r72, r85, r145);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           24 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r145 = r10 * r76;
    r145 = r145 * r81;
    r145 = r145 * r94;
    r85 = r11 * r76;
    r85 = r85 * r81;
    r85 = r85 * r146;
    r72 = r63 * r79;
    r65 = r11 * r136;
    r65 = fmaf(r72, r65, r1 * r72);
    r72 = r16 * r10;
    r72 = r72 * r10;
    r72 = r72 * r11;
    r72 = r72 * r60;
    r72 = r72 * r3;
    r72 = r72 * r117;
    r72 = r72 * r81;
    r72 = r72 * r94;
    r72 = fmaf(r81, r78, r101 * r72);
    WriteSum4<float, float>((float*)inout_shared, r145, r85, r65, r72);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           28 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r11 * r66;
    r72 = r72 * r63;
    r72 = r72 * r79;
    r72 = r72 * r94;
    r53 = r66 * r53;
    r65 = r16 * r10;
    r65 = r65 * r10;
    r65 = r65 * r11;
    r65 = r65 * r60;
    r65 = r65 * r3;
    r65 = r65 * r117;
    r65 = r65 * r94;
    r65 = r65 * r101;
    r78 = fmaf(r83, r78, r83 * r65);
    r65 = r16 * r10;
    r65 = r65 * r60;
    r65 = r65 * r3;
    r65 = r65 * r117;
    r65 = r65 * r81;
    r65 = r65 * r146;
    r65 = r65 * r101;
    r65 = fmaf(r87, r65, r81 * r12);
    WriteSum4<float, float>((float*)inout_shared, r78, r72, r53, r65);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           32 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r66 * r136;
    r53 = r11 * r66;
    r53 = r53 * r63;
    r53 = r53 * r79;
    r53 = r53 * r146;
    r72 = r16 * r10;
    r72 = r72 * r60;
    r72 = r72 * r3;
    r72 = r72 * r117;
    r72 = r72 * r146;
    r72 = r72 * r101;
    r72 = r72 * r83;
    r72 = fmaf(r87, r72, r83 * r12);
    r51 = r66 * r51;
    r131 = fmaf(r51, r131, r51 * r36);
    WriteSum4<float, float>((float*)inout_shared, r72, r65, r53, r131);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           36 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r131 = r10 * r76;
    r131 = r131 * r94;
    r131 = r131 * r83;
    r53 = r11 * r76;
    r53 = r53 * r146;
    r53 = r53 * r83;
    r83 = r94 * r149;
    r83 = r83 * r77;
    r77 = r11 * r76;
    r77 = r77 * r146;
    r77 = r77 * r149;
    WriteSum4<float, float>((float*)inout_shared, r131, r53, r83, r77);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           40 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = r55 * r100;
    r77 = r77 * r62;
    r83 = r61 * r113;
    r83 = r83 * r3;
    r83 = r83 * r64;
    r83 = fmaf(r101, r83, r63 * r77);
    r77 = r16 * r33;
    r77 = r77 * r11;
    r53 = r16 * r55;
    r53 = r53 * r10;
    r53 = fmaf(r37, r53, r37 * r77);
    r77 = r61 * r10;
    r77 = r77 * r10;
    r53 = fmaf(r108, r77, r53);
    r131 = r61 * r108;
    r53 = fmaf(r87, r131, r53);
    r131 = r71 * r53;
    r83 = fmaf(r111, r131, r83);
    r77 = r10 * r10;
    r77 = r77 * r116;
    r77 = r77 * r53;
    r77 = r77 * r107;
    r77 = r77 * r117;
    r83 = fmaf(r62, r77, r83);
    r149 = r4 * r11;
    r149 = r149 * r53;
    r65 = r53 * r120;
    r65 = r65 * r42;
    r65 = fmaf(r118, r65, r123 * r149);
    r65 = fmaf(r61, r124, r65);
    r65 = fmaf(r33, r119, r65);
    r83 = r83 + r65;
    r77 = r55 * r63;
    r131 = r61 * r64;
    r131 = fmaf(r125, r131, r79 * r77);
    r77 = r4 * r10;
    r77 = r77 * r10;
    r77 = r77 * r53;
    r77 = r77 * r107;
    r77 = r77 * r117;
    r131 = fmaf(r62, r77, r131);
    r131 = fmaf(r53, r111, r131);
    r65 = r65 + r131;
    r83 = fmaf(r8, r65, r68 * r83);
    r77 = r60 * r82;
    r77 = r77 * r88;
    r77 = r77 * r53;
    r77 = r77 * r73;
    r77 = r77 * r107;
    r83 = fmaf(r63, r77, r83);
    r149 = r4 * r61;
    r149 = r149 * r10;
    r149 = r149 * r82;
    r149 = r149 * r74;
    r83 = fmaf(r57, r149, r83);
    r72 = r4 * r61;
    r72 = r72 * r10;
    r72 = r72 * r74;
    r83 = fmaf(r57, r72, r83);
    r51 = r67 * r33;
    r51 = r51 * r63;
    r83 = fmaf(r79, r51, r83);
    r36 = r7 * r16;
    r36 = r36 * r66;
    r36 = fmaf(r6, r65, r65 * r36);
    r36 = fmaf(r65, r70, r36);
    r36 = fmaf(r65, r69, r36);
    r12 = r10 * r36;
    r83 = fmaf(r76, r12, r83);
    r78 = r10 * r86;
    r78 = r78 * r82;
    r78 = r78 * r53;
    r78 = r78 * r107;
    r78 = r78 * r35;
    r83 = fmaf(r75, r78, r83);
    r85 = r60 * r88;
    r85 = r85 * r53;
    r85 = r85 * r73;
    r85 = r85 * r107;
    r83 = fmaf(r63, r85, r83);
    r145 = r53 * r134;
    r1 = r67 * r55;
    r83 = fmaf(r119, r1, r83);
    r112 = r67 * r53;
    r83 = fmaf(r127, r112, r83);
    r52 = r10 * r86;
    r52 = r52 * r53;
    r52 = r52 * r107;
    r52 = r52 * r35;
    r83 = fmaf(r75, r52, r83);
    r83 = fmaf(r55, r84, r83);
    r83 = fmaf(r55, r76, r83);
    r83 = fmaf(r61, r129, r83);
    r83 = fmaf(r67, r145, r83);
    r52 = r0 * r83;
    r112 = r11 * r116;
    r112 = r112 * r53;
    r1 = r71 * r53;
    r1 = r1 * r120;
    r1 = r1 * r42;
    r1 = fmaf(r118, r1, r123 * r112);
    r112 = r61 * r11;
    r112 = r112 * r11;
    r112 = r112 * r60;
    r112 = r112 * r60;
    r112 = r112 * r113;
    r112 = r112 * r47;
    r1 = fmaf(r3, r112, r1);
    r85 = r33 * r11;
    r85 = r85 * r100;
    r85 = r85 * r47;
    r1 = fmaf(r62, r85, r1);
    r1 = r1 + r131;
    r65 = fmaf(r9, r65, r67 * r1);
    r1 = r86 * r82;
    r1 = r1 * r53;
    r1 = r1 * r35;
    r1 = r1 * r75;
    r65 = fmaf(r120, r1, r65);
    r131 = r4 * r61;
    r131 = r131 * r11;
    r131 = r131 * r82;
    r131 = r131 * r74;
    r65 = fmaf(r57, r131, r65);
    r85 = r68 * r33;
    r85 = r85 * r63;
    r65 = fmaf(r79, r85, r65);
    r112 = r68 * r61;
    r112 = r112 * r11;
    r112 = r112 * r60;
    r112 = r112 * r60;
    r112 = r112 * r98;
    r112 = r112 * r3;
    r65 = fmaf(r63, r112, r65);
    r78 = r86 * r53;
    r78 = r78 * r35;
    r78 = r78 * r75;
    r65 = fmaf(r120, r78, r65);
    r12 = r4 * r61;
    r12 = r12 * r11;
    r12 = r12 * r74;
    r65 = fmaf(r57, r12, r65);
    r51 = r68 * r53;
    r65 = fmaf(r127, r51, r65);
    r72 = r11 * r36;
    r65 = fmaf(r76, r72, r65);
    r65 = fmaf(r53, r132, r65);
    r65 = fmaf(r53, r115, r65);
    r65 = fmaf(r68, r145, r65);
    r65 = fmaf(r55, r130, r65);
    r65 = fmaf(r33, r84, r65);
    r65 = fmaf(r33, r76, r65);
    r72 = r5 * r65;
    r51 = r45 * r108;
    r12 = r16 * r49;
    r12 = r12 * r10;
    r12 = fmaf(r37, r12, r87 * r51);
    r51 = r16 * r50;
    r51 = r51 * r11;
    r12 = fmaf(r37, r51, r12);
    r78 = r45 * r10;
    r78 = r78 * r10;
    r12 = fmaf(r108, r78, r12);
    r78 = r45 * r64;
    r78 = fmaf(r125, r78, r12 * r111);
    r51 = r4 * r10;
    r51 = r51 * r10;
    r51 = r51 * r12;
    r51 = r51 * r107;
    r51 = r51 * r117;
    r78 = fmaf(r62, r51, r78);
    r145 = r49 * r63;
    r78 = fmaf(r79, r145, r78);
    r145 = r4 * r11;
    r145 = r145 * r12;
    r145 = fmaf(r50, r119, r123 * r145);
    r51 = r12 * r120;
    r51 = r51 * r42;
    r145 = fmaf(r118, r51, r145);
    r145 = fmaf(r45, r124, r145);
    r51 = r78 + r145;
    r112 = r71 * r12;
    r85 = r45 * r113;
    r85 = r85 * r3;
    r85 = r85 * r64;
    r85 = fmaf(r101, r85, r111 * r112);
    r112 = r10 * r10;
    r112 = r112 * r116;
    r112 = r112 * r12;
    r112 = r112 * r107;
    r112 = r112 * r117;
    r85 = fmaf(r62, r112, r85);
    r111 = r49 * r100;
    r111 = r111 * r62;
    r85 = fmaf(r63, r111, r85);
    r85 = r85 + r145;
    r85 = fmaf(r68, r85, r8 * r51);
    r145 = r67 * r49;
    r85 = fmaf(r119, r145, r85);
    r111 = r67 * r12;
    r85 = fmaf(r134, r111, r85);
    r112 = r60 * r88;
    r112 = r112 * r12;
    r112 = r112 * r73;
    r112 = r112 * r107;
    r85 = fmaf(r63, r112, r85);
    r131 = r4 * r45;
    r131 = r131 * r10;
    r131 = r131 * r74;
    r85 = fmaf(r57, r131, r85);
    r1 = r4 * r45;
    r1 = r1 * r10;
    r1 = r1 * r82;
    r1 = r1 * r74;
    r85 = fmaf(r57, r1, r85);
    r149 = r7 * r16;
    r149 = r149 * r66;
    r149 = fmaf(r51, r149, r6 * r51);
    r149 = fmaf(r51, r70, r149);
    r149 = fmaf(r51, r69, r149);
    r77 = r10 * r149;
    r85 = fmaf(r76, r77, r85);
    r105 = r60 * r82;
    r105 = r105 * r88;
    r105 = r105 * r12;
    r105 = r105 * r73;
    r105 = r105 * r107;
    r85 = fmaf(r63, r105, r85);
    r92 = r10 * r86;
    r92 = r92 * r12;
    r92 = r92 * r107;
    r92 = r92 * r35;
    r85 = fmaf(r75, r92, r85);
    r128 = r10 * r86;
    r128 = r128 * r82;
    r128 = r128 * r12;
    r128 = r128 * r107;
    r128 = r128 * r35;
    r85 = fmaf(r75, r128, r85);
    r14 = r67 * r50;
    r14 = r14 * r63;
    r85 = fmaf(r79, r14, r85);
    r106 = r67 * r12;
    r85 = fmaf(r127, r106, r85);
    r85 = fmaf(r49, r84, r85);
    r85 = fmaf(r49, r76, r85);
    r85 = fmaf(r45, r129, r85);
    r106 = r0 * r85;
    r14 = r11 * r116;
    r14 = r14 * r12;
    r128 = r50 * r11;
    r128 = r128 * r100;
    r128 = r128 * r47;
    r128 = fmaf(r62, r128, r123 * r14);
    r14 = r71 * r12;
    r14 = r14 * r120;
    r14 = r14 * r42;
    r128 = fmaf(r118, r14, r128);
    r92 = r45 * r11;
    r92 = r92 * r11;
    r92 = r92 * r60;
    r92 = r92 * r60;
    r92 = r92 * r113;
    r92 = r92 * r47;
    r128 = fmaf(r3, r92, r128);
    r128 = r128 + r78;
    r128 = fmaf(r67, r128, r9 * r51);
    r51 = r68 * r12;
    r128 = fmaf(r127, r51, r128);
    r78 = r68 * r12;
    r128 = fmaf(r134, r78, r128);
    r92 = r11 * r149;
    r128 = fmaf(r76, r92, r128);
    r14 = r4 * r45;
    r14 = r14 * r11;
    r14 = r14 * r82;
    r14 = r14 * r74;
    r128 = fmaf(r57, r14, r128);
    r105 = r86 * r12;
    r105 = r105 * r35;
    r105 = r105 * r75;
    r128 = fmaf(r120, r105, r128);
    r77 = r68 * r50;
    r77 = r77 * r63;
    r128 = fmaf(r79, r77, r128);
    r1 = r86 * r82;
    r1 = r1 * r12;
    r1 = r1 * r35;
    r1 = r1 * r75;
    r128 = fmaf(r120, r1, r128);
    r131 = r4 * r45;
    r131 = r131 * r11;
    r131 = r131 * r74;
    r128 = fmaf(r57, r131, r128);
    r112 = r68 * r45;
    r112 = r112 * r11;
    r112 = r112 * r60;
    r112 = r112 * r60;
    r112 = r112 * r98;
    r112 = r112 * r3;
    r128 = fmaf(r63, r112, r128);
    r128 = fmaf(r49, r130, r128);
    r128 = fmaf(r12, r132, r128);
    r128 = fmaf(r12, r115, r128);
    r128 = fmaf(r50, r76, r128);
    r128 = fmaf(r50, r84, r128);
    r112 = r5 * r128;
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r52,
                                          r72,
                                          r106,
                                          r112);
    r112 = r10 * r10;
    r112 = r37 * r112;
    r106 = r16 * r48;
    r106 = r106 * r11;
    r72 = r54 * r108;
    r72 = fmaf(r87, r72, r37 * r106);
    r106 = r54 * r10;
    r106 = r106 * r10;
    r72 = fmaf(r108, r106, r72);
    r87 = r16 * r56;
    r87 = r87 * r10;
    r72 = fmaf(r37, r87, r72);
    r112 = r112 * r47;
    r112 = r112 * r60;
    r112 = r112 * r107;
    r112 = r112 * r35;
    r112 = r112 * r72;
    r87 = r56 * r63;
    r87 = fmaf(r79, r87, r112);
    r106 = r54 * r64;
    r87 = fmaf(r125, r106, r87);
    r125 = r4 * r10;
    r125 = r125 * r10;
    r125 = r125 * r72;
    r125 = r125 * r107;
    r125 = r125 * r117;
    r87 = fmaf(r62, r125, r87);
    r125 = r4 * r11;
    r125 = r125 * r72;
    r125 = fmaf(r48, r119, r123 * r125);
    r106 = r72 * r120;
    r106 = r106 * r42;
    r125 = fmaf(r118, r106, r125);
    r125 = fmaf(r54, r124, r125);
    r124 = r87 + r125;
    r106 = r56 * r100;
    r106 = r106 * r62;
    r112 = fmaf(r71, r112, r63 * r106);
    r106 = r54 * r113;
    r106 = r106 * r3;
    r106 = r106 * r64;
    r112 = fmaf(r101, r106, r112);
    r37 = r10 * r10;
    r37 = r37 * r116;
    r37 = r37 * r72;
    r37 = r37 * r107;
    r37 = r37 * r117;
    r112 = fmaf(r62, r37, r112);
    r112 = r112 + r125;
    r112 = fmaf(r68, r112, r8 * r124);
    r8 = r67 * r48;
    r8 = r8 * r63;
    r112 = fmaf(r79, r8, r112);
    r125 = r67 * r72;
    r112 = fmaf(r127, r125, r112);
    r37 = r60 * r82;
    r37 = r37 * r88;
    r37 = r37 * r72;
    r37 = r37 * r73;
    r37 = r37 * r107;
    r112 = fmaf(r63, r37, r112);
    r106 = r4 * r54;
    r106 = r106 * r10;
    r106 = r106 * r82;
    r106 = r106 * r74;
    r112 = fmaf(r57, r106, r112);
    r117 = r7 * r16;
    r117 = r117 * r66;
    r117 = fmaf(r124, r117, r6 * r124);
    r117 = fmaf(r124, r70, r117);
    r117 = fmaf(r124, r69, r117);
    r69 = r10 * r117;
    r112 = fmaf(r76, r69, r112);
    r70 = r10 * r86;
    r70 = r70 * r82;
    r70 = r70 * r72;
    r70 = r70 * r107;
    r70 = r70 * r35;
    r112 = fmaf(r75, r70, r112);
    r6 = r4 * r54;
    r6 = r6 * r10;
    r6 = r6 * r74;
    r112 = fmaf(r57, r6, r112);
    r52 = r60 * r88;
    r52 = r52 * r72;
    r52 = r52 * r73;
    r52 = r52 * r107;
    r112 = fmaf(r63, r52, r112);
    r73 = r67 * r72;
    r112 = fmaf(r134, r73, r112);
    r131 = r10 * r86;
    r131 = r131 * r72;
    r131 = r131 * r107;
    r131 = r131 * r35;
    r112 = fmaf(r75, r131, r112);
    r107 = r67 * r56;
    r112 = fmaf(r119, r107, r112);
    r112 = fmaf(r56, r76, r112);
    r112 = fmaf(r56, r84, r112);
    r112 = fmaf(r54, r129, r112);
    r107 = r0 * r112;
    r129 = r11 * r116;
    r129 = r129 * r72;
    r131 = r48 * r11;
    r131 = r131 * r100;
    r131 = r131 * r47;
    r131 = fmaf(r62, r131, r123 * r129);
    r129 = r71 * r72;
    r129 = r129 * r120;
    r129 = r129 * r42;
    r131 = fmaf(r118, r129, r131);
    r118 = r54 * r11;
    r118 = r118 * r11;
    r118 = r118 * r60;
    r118 = r118 * r60;
    r118 = r118 * r113;
    r118 = r118 * r47;
    r131 = fmaf(r3, r118, r131);
    r131 = r131 + r87;
    r131 = fmaf(r67, r131, r9 * r124);
    r124 = r68 * r48;
    r124 = r124 * r63;
    r131 = fmaf(r79, r124, r131);
    r79 = r4 * r54;
    r79 = r79 * r11;
    r79 = r79 * r82;
    r79 = r79 * r74;
    r131 = fmaf(r57, r79, r131);
    r9 = r68 * r72;
    r131 = fmaf(r127, r9, r131);
    r127 = r11 * r117;
    r131 = fmaf(r76, r127, r131);
    r87 = r68 * r72;
    r131 = fmaf(r134, r87, r131);
    r134 = r68 * r54;
    r134 = r134 * r11;
    r134 = r134 * r60;
    r134 = r134 * r60;
    r134 = r134 * r98;
    r134 = r134 * r3;
    r131 = fmaf(r63, r134, r131);
    r3 = r86 * r82;
    r3 = r3 * r72;
    r3 = r3 * r35;
    r3 = r3 * r75;
    r131 = fmaf(r120, r3, r131);
    r98 = r4 * r54;
    r98 = r98 * r11;
    r98 = r98 * r74;
    r131 = fmaf(r57, r98, r131);
    r57 = r86 * r72;
    r57 = r57 * r35;
    r57 = r57 * r75;
    r131 = fmaf(r120, r57, r131);
    r131 = fmaf(r72, r115, r131);
    r131 = fmaf(r48, r76, r131);
    r131 = fmaf(r72, r132, r131);
    r131 = fmaf(r48, r84, r131);
    r131 = fmaf(r56, r130, r131);
    r57 = r5 * r131;
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r107,
                                          r57);
    r57 = r0 * r4;
    r57 = r57 * r2;
    r57 = fmaf(r65, r91, r83 * r57);
    r107 = r0 * r4;
    r107 = r107 * r2;
    r107 = fmaf(r128, r91, r85 * r107);
    r98 = r0 * r4;
    r98 = r98 * r2;
    r91 = fmaf(r131, r91, r112 * r98);
    WriteSum3<float, float>((float*)inout_shared, r57, r107, r91);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = r83 * r83;
    r107 = r65 * r65;
    r107 = fmaf(r146, r107, r94 * r91);
    r91 = r128 * r128;
    r57 = r85 * r85;
    r57 = fmaf(r94, r57, r146 * r91);
    r91 = r131 * r131;
    r98 = r112 * r112;
    r98 = fmaf(r94, r98, r146 * r91);
    WriteSum3<float, float>((float*)inout_shared, r107, r57, r98);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r98 = r65 * r128;
    r57 = r83 * r85;
    r57 = fmaf(r94, r57, r146 * r98);
    r98 = r83 * r112;
    r107 = r65 * r131;
    r107 = fmaf(r146, r107, r94 * r98);
    r98 = r128 * r131;
    r91 = r85 * r112;
    r91 = fmaf(r94, r91, r146 * r98);
    WriteSum3<float, float>((float*)inout_shared, r57, r107, r91);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void ThinPrismFisheyeSplitFixedPrincipalPointResJac(
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
  ThinPrismFisheyeSplitFixedPrincipalPointResJacKernel<<<n_blocks, 1024>>>(
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