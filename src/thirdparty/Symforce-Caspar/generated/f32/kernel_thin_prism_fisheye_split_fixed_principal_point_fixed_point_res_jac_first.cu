#include "kernel_thin_prism_fisheye_split_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPrincipalPointFixedPointResJacFirstKernel(
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
      r129, r130, r131, r132, r133, r134, r135, r136, r137, r138, r139;

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
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r13, r14, r15);
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
    r33 = fmaf(r13, r33, r11);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r11,
                       r35,
                       r36);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r37 = r21 * r22;
    r37 = r37 * r16;
    r38 = r23 * r24;
    r38 = fmaf(r16, r38, r37);
    r39 = r21 * r21;
    r40 = -2.00000000000000000e+00;
    r39 = r39 * r40;
    r41 = 1.00000000000000000e+00;
    r42 = r23 * r23;
    r42 = fmaf(r40, r42, r41);
    r43 = r39 + r42;
    r44 = r22 * r23;
    r44 = r44 * r16;
    r45 = r21 * r24;
    r45 = fmaf(r40, r45, r44);
    r46 = r16 * r32;
    r46 = r46 * r29;
    r47 = r25 * r40;
    r47 = fmaf(r34, r47, r46);
    r48 = r32 * r32;
    r48 = r48 * r40;
    r49 = r41 + r48;
    r50 = r25 * r25;
    r50 = r50 * r40;
    r49 = r49 + r50;
    r33 = fmaf(r11, r38, r33);
    r33 = fmaf(r35, r43, r33);
    r33 = fmaf(r36, r45, r33);
    r33 = fmaf(r15, r47, r33);
    r33 = fmaf(r14, r49, r33);
    r49 = r40 * r29;
    r49 = r49 * r29;
    r47 = r41 + r49;
    r47 = r47 + r48;
    r47 = fmaf(r13, r47, r10);
    r10 = r32 * r40;
    r10 = fmaf(r34, r10, r26);
    r26 = r16 * r32;
    r26 = r26 * r25;
    r48 = r16 * r29;
    r48 = fmaf(r34, r48, r26);
    r51 = r21 * r23;
    r51 = r51 * r16;
    r52 = r22 * r24;
    r52 = fmaf(r16, r52, r51);
    r53 = r23 * r24;
    r53 = fmaf(r40, r53, r37);
    r37 = r22 * r22;
    r37 = r37 * r40;
    r42 = r37 + r42;
    r47 = fmaf(r14, r10, r47);
    r47 = fmaf(r15, r48, r47);
    r47 = fmaf(r36, r52, r47);
    r47 = fmaf(r35, r53, r47);
    r47 = fmaf(r11, r42, r47);
    r48 = r47 * r47;
    r10 = 9.99999999999999955e-07;
    r54 = r40 * r29;
    r54 = fmaf(r34, r54, r26);
    r54 = fmaf(r13, r54, r12);
    r12 = r22 * r24;
    r12 = fmaf(r40, r12, r51);
    r37 = r41 + r37;
    r37 = r37 + r39;
    r39 = r21 * r24;
    r39 = fmaf(r16, r39, r44);
    r44 = r16 * r25;
    r44 = fmaf(r34, r44, r46);
    r49 = r41 + r49;
    r49 = r49 + r50;
    r54 = fmaf(r11, r12, r54);
    r54 = fmaf(r36, r37, r54);
    r54 = fmaf(r35, r39, r54);
    r54 = fmaf(r14, r44, r54);
    r54 = fmaf(r15, r49, r54);
    r49 = copysign(1.0, r54);
    r49 = fmaf(r10, r49, r54);
    r54 = r49 * r49;
    r44 = 1.0 / r54;
    r35 = r33 * r33;
    r35 = fmaf(r44, r35, r44 * r48);
    r48 = sqrtf(r35);
    r36 = atanf(r48);
    r11 = r33 * r36;
    r50 = copysign(1.0, r48);
    r50 = fmaf(r10, r50, r48);
    r10 = r50 * r50;
    r48 = 1.0 / r10;
    r46 = r44 * r48;
    r51 = r33 * r36;
    r11 = r11 * r46;
    r11 = r11 * r51;
    r26 = r47 * r36;
    r55 = r26 * r46;
    r56 = r47 * r55;
    r57 = r36 * r56;
    r58 = r11 + r57;
  };
  LoadShared<4, float, float>(focal_and_extra,
                              4 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r59,
                       r60,
                       r61,
                       r62);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r63 = 3.00000000000000000e+00;
    r64 = r36 * r63;
    r64 = fmaf(r56, r64, r11);
    r11 = fmaf(r60, r64, r8 * r58);
    r65 = r59 * r16;
    r65 = r65 * r51;
    r11 = fmaf(r55, r65, r11);
    r66 = r58 * r58;
    r67 = r58 * r66;
    r68 = fmaf(r61, r67, r6 * r58);
    r69 = r66 * r66;
    r68 = fmaf(r62, r69, r68);
    r68 = fmaf(r7, r66, r68);
    r70 = 1.0 / r49;
    r71 = 1.0 / r50;
    r72 = r70 * r71;
    r73 = r68 * r72;
    r11 = fmaf(r26, r73, r11);
    r11 = fmaf(r26, r72, r11);
    r2 = fmaf(r0, r11, r2);
    r65 = r33 * r36;
    r65 = r65 * r63;
    r65 = r65 * r46;
    r65 = fmaf(r51, r65, r57);
    r57 = fmaf(r59, r65, r9 * r58);
    r74 = r60 * r16;
    r74 = r74 * r51;
    r57 = fmaf(r55, r74, r57);
    r57 = fmaf(r51, r73, r57);
    r57 = fmaf(r72, r51, r57);
    r1 = fmaf(r5, r57, r1);
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
    r3 = r16 * r34;
    r74 = r19 * r24;
    r75 = 5.00000000000000000e-01;
    r76 = r18 * r21;
    r76 = fmaf(r75, r76, r75 * r74);
    r74 = r17 * r22;
    r77 = -5.00000000000000000e-01;
    r76 = fmaf(r77, r74, r76);
    r78 = r20 * r23;
    r76 = fmaf(r75, r78, r76);
    r78 = r17 * r24;
    r74 = r20 * r21;
    r74 = fmaf(r77, r74, r77 * r78);
    r78 = r19 * r22;
    r74 = fmaf(r77, r78, r74);
    r79 = r18 * r23;
    r74 = fmaf(r75, r79, r74);
    r79 = r29 * r74;
    r3 = fmaf(r16, r79, r76 * r3);
    r78 = r16 * r25;
    r80 = fmaf(r75, r31, r77 * r27);
    r80 = fmaf(r77, r28, r80);
    r80 = fmaf(r77, r30, r80);
    r81 = r16 * r32;
    r82 = r20 * r24;
    r83 = r17 * r21;
    r83 = fmaf(r77, r83, r75 * r82);
    r82 = r18 * r22;
    r83 = fmaf(r77, r82, r83);
    r84 = r19 * r23;
    r83 = fmaf(r77, r84, r83);
    r81 = r81 * r83;
    r78 = fmaf(r80, r78, r81);
    r3 = r3 + r78;
    r84 = r16 * r29;
    r84 = r84 * r83;
    r82 = r16 * r25;
    r82 = r82 * r76;
    r85 = r84 + r82;
    r86 = r32 * r40;
    r85 = fmaf(r74, r86, r85);
    r87 = r40 * r34;
    r85 = fmaf(r80, r87, r85);
    r85 = fmaf(r14, r85, r15 * r3);
    r3 = r29 * r76;
    r87 = -4.00000000000000000e+00;
    r3 = r3 * r87;
    r86 = r32 * r80;
    r88 = r87 * r86;
    r89 = r3 + r88;
    r85 = fmaf(r13, r89, r85);
    r89 = r36 * r85;
    r90 = 6.00000000000000000e+00;
    r89 = r89 * r90;
    r91 = r16 * r33;
    r92 = r25 * r40;
    r93 = r34 * r83;
    r94 = r40 * r93;
    r92 = fmaf(r74, r92, r94);
    r95 = r16 * r29;
    r95 = r95 * r80;
    r96 = r16 * r32;
    r96 = fmaf(r76, r96, r95);
    r92 = r92 + r96;
    r97 = r25 * r83;
    r97 = r97 * r87;
    r88 = r97 + r88;
    r88 = fmaf(r14, r88, r15 * r92);
    r92 = r16 * r34;
    r92 = fmaf(r80, r92, r82);
    r82 = r16 * r32;
    r82 = fmaf(r74, r82, r84);
    r92 = r92 + r82;
    r88 = fmaf(r13, r92, r88);
    r91 = r91 * r88;
    r92 = r16 * r47;
    r92 = r92 * r85;
    r92 = fmaf(r44, r92, r44 * r91);
    r91 = r16 * r25;
    r91 = r91 * r74;
    r93 = r16 * r93;
    r84 = r91 + r93;
    r96 = r96 + r84;
    r98 = r40 * r34;
    r98 = fmaf(r40, r79, r76 * r98);
    r98 = r98 + r78;
    r98 = fmaf(r13, r98, r14 * r96);
    r3 = r97 + r3;
    r98 = fmaf(r15, r3, r98);
    r3 = r33 * r33;
    r54 = r49 * r54;
    r97 = 1.0 / r54;
    r96 = r40 * r97;
    r3 = r3 * r96;
    r76 = r47 * r47;
    r76 = r76 * r98;
    r92 = fmaf(r96, r76, r92);
    r92 = fmaf(r98, r3, r92);
    r76 = r63 * r92;
    r99 = rsqrtf(r35);
    r35 = r41 + r35;
    r35 = 1.0 / r35;
    r41 = r99 * r35;
    r100 = r41 * r56;
    r76 = fmaf(r100, r76, r55 * r89);
    r89 = r47 * r26;
    r101 = r36 * r98;
    r102 = -6.00000000000000000e+00;
    r101 = r101 * r102;
    r101 = r101 * r48;
    r101 = r101 * r97;
    r76 = fmaf(r101, r89, r76);
    r103 = r47 * r26;
    r104 = -3.00000000000000000e+00;
    r104 = r36 * r104;
    r10 = r50 * r10;
    r105 = 1.0 / r10;
    r104 = r104 * r44;
    r104 = r104 * r99;
    r104 = r104 * r105;
    r103 = r103 * r104;
    r106 = r16 * r36;
    r107 = r46 * r51;
    r108 = r106 * r107;
    r109 = r92 * r107;
    r110 = r33 * r41;
    r109 = fmaf(r110, r109, r88 * r108);
    r111 = r4 * r33;
    r111 = r111 * r33;
    r111 = r111 * r36;
    r111 = r111 * r36;
    r111 = r111 * r92;
    r111 = r111 * r44;
    r111 = r111 * r99;
    r109 = fmaf(r105, r111, r109);
    r112 = r36 * r3;
    r113 = r36 * r48;
    r113 = r113 * r96;
    r113 = r112 * r113;
    r109 = fmaf(r98, r113, r109);
    r76 = fmaf(r92, r103, r76);
    r76 = r76 + r109;
    r89 = r85 * r55;
    r89 = fmaf(r92, r100, r106 * r89);
    r111 = r47 * r36;
    r111 = r111 * r98;
    r111 = r111 * r48;
    r111 = r111 * r26;
    r89 = fmaf(r96, r111, r89);
    r114 = r4 * r47;
    r114 = r114 * r36;
    r114 = r114 * r92;
    r114 = r114 * r44;
    r114 = r114 * r99;
    r114 = r114 * r105;
    r89 = fmaf(r26, r114, r89);
    r109 = r109 + r89;
    r76 = fmaf(r8, r109, r60 * r76);
    r114 = r77 * r92;
    r114 = r114 * r48;
    r114 = r114 * r70;
    r114 = r114 * r99;
    r111 = r68 * r114;
    r115 = r59 * r92;
    r116 = r16 * r55;
    r116 = r116 * r110;
    r76 = fmaf(r116, r115, r76);
    r117 = r87 * r48;
    r117 = r117 * r97;
    r117 = r117 * r26;
    r117 = r117 * r51;
    r118 = r59 * r117;
    r119 = r47 * r92;
    r120 = r75 * r73;
    r119 = r119 * r41;
    r76 = fmaf(r120, r119, r76);
    r121 = r59 * r85;
    r76 = fmaf(r108, r121, r76);
    r122 = r36 * r85;
    r76 = fmaf(r72, r122, r76);
    r123 = r4 * r98;
    r123 = r123 * r44;
    r123 = r123 * r71;
    r76 = fmaf(r26, r123, r76);
    r124 = r4 * r68;
    r124 = r124 * r98;
    r124 = r124 * r44;
    r124 = r124 * r71;
    r76 = fmaf(r26, r124, r76);
    r125 = r47 * r75;
    r125 = r125 * r92;
    r125 = r125 * r72;
    r76 = fmaf(r41, r125, r76);
    r126 = r59 * r40;
    r126 = r126 * r92;
    r126 = r126 * r44;
    r126 = r126 * r99;
    r126 = r126 * r105;
    r126 = r126 * r26;
    r76 = fmaf(r51, r126, r76);
    r127 = r7 * r16;
    r127 = r127 * r58;
    r127 = fmaf(r109, r127, r6 * r109);
    r128 = 4.00000000000000000e+00;
    r62 = r62 * r128;
    r62 = r62 * r67;
    r61 = r61 * r63;
    r61 = r61 * r66;
    r127 = fmaf(r109, r62, r127);
    r127 = fmaf(r109, r61, r127);
    r129 = r127 * r26;
    r76 = fmaf(r72, r129, r76);
    r130 = r59 * r88;
    r130 = r130 * r55;
    r76 = fmaf(r106, r130, r76);
    r131 = r36 * r85;
    r76 = fmaf(r73, r131, r76);
    r76 = fmaf(r26, r111, r76);
    r76 = fmaf(r26, r114, r76);
    r76 = fmaf(r98, r118, r76);
    r131 = r0 * r76;
    r130 = r36 * r88;
    r130 = r130 * r90;
    r130 = r130 * r46;
    r129 = r63 * r92;
    r129 = r129 * r107;
    r129 = fmaf(r110, r129, r51 * r130);
    r130 = r92 * r104;
    r129 = fmaf(r112, r130, r129);
    r126 = r33 * r33;
    r126 = r126 * r36;
    r129 = fmaf(r101, r126, r129);
    r129 = r129 + r89;
    r109 = fmaf(r9, r109, r59 * r129);
    r129 = r75 * r92;
    r129 = r129 * r72;
    r109 = fmaf(r110, r129, r109);
    r89 = r127 * r72;
    r109 = fmaf(r51, r89, r109);
    r126 = r4 * r33;
    r126 = r126 * r36;
    r126 = r126 * r68;
    r126 = r126 * r98;
    r126 = r126 * r44;
    r109 = fmaf(r71, r126, r109);
    r130 = r60 * r98;
    r109 = fmaf(r117, r130, r109);
    r101 = r60 * r108;
    r125 = r60 * r92;
    r109 = fmaf(r116, r125, r109);
    r124 = r4 * r33;
    r124 = r124 * r36;
    r124 = r124 * r98;
    r124 = r124 * r44;
    r109 = fmaf(r71, r124, r109);
    r123 = r60 * r40;
    r123 = r123 * r92;
    r123 = r123 * r44;
    r123 = r123 * r99;
    r123 = r123 * r105;
    r123 = r123 * r26;
    r109 = fmaf(r51, r123, r109);
    r122 = r110 * r120;
    r121 = r60 * r88;
    r121 = r121 * r55;
    r109 = fmaf(r106, r121, r109);
    r119 = r36 * r88;
    r109 = fmaf(r73, r119, r109);
    r115 = r36 * r88;
    r109 = fmaf(r72, r115, r109);
    r109 = fmaf(r51, r111, r109);
    r109 = fmaf(r85, r101, r109);
    r109 = fmaf(r92, r122, r109);
    r109 = fmaf(r114, r51, r109);
    r115 = r5 * r109;
    r119 = r47 * r47;
    r119 = r44 * r119;
    r114 = r16 * r47;
    r93 = r95 + r93;
    r95 = r16 * r32;
    r121 = r19 * r24;
    r123 = r18 * r21;
    r123 = fmaf(r77, r123, r77 * r121);
    r121 = r17 * r22;
    r123 = fmaf(r75, r121, r123);
    r124 = r20 * r23;
    r123 = fmaf(r77, r124, r123);
    r95 = r95 * r123;
    r124 = r16 * r25;
    r121 = r17 * r24;
    r125 = r20 * r21;
    r125 = fmaf(r75, r125, r75 * r121);
    r121 = r19 * r22;
    r125 = fmaf(r75, r121, r125);
    r130 = r18 * r23;
    r125 = fmaf(r77, r130, r125);
    r124 = fmaf(r125, r124, r95);
    r93 = r93 + r124;
    r130 = r32 * r87;
    r130 = r130 * r125;
    r121 = r29 * r83;
    r121 = r121 * r87;
    r126 = r130 + r121;
    r126 = fmaf(r13, r126, r15 * r93);
    r93 = r40 * r34;
    r93 = fmaf(r40, r86, r125 * r93);
    r89 = r16 * r25;
    r89 = r89 * r83;
    r111 = r16 * r29;
    r111 = fmaf(r123, r111, r89);
    r93 = r93 + r111;
    r126 = fmaf(r14, r93, r126);
    r114 = r114 * r126;
    r93 = r47 * r47;
    r129 = r40 * r29;
    r129 = fmaf(r80, r129, r94);
    r129 = r129 + r124;
    r124 = r16 * r29;
    r124 = r124 * r125;
    r132 = r16 * r34;
    r132 = fmaf(r123, r132, r124);
    r132 = r132 + r78;
    r132 = fmaf(r14, r132, r13 * r129);
    r129 = r25 * r123;
    r78 = r87 * r129;
    r121 = r121 + r78;
    r132 = fmaf(r15, r121, r132);
    r93 = r93 * r132;
    r93 = fmaf(r96, r93, r44 * r114);
    r114 = r16 * r33;
    r121 = r25 * r40;
    r121 = fmaf(r80, r121, r81);
    r81 = r40 * r34;
    r121 = fmaf(r123, r81, r121);
    r121 = r121 + r124;
    r81 = r16 * r34;
    r86 = fmaf(r16, r86, r125 * r81);
    r86 = r86 + r111;
    r86 = fmaf(r13, r86, r15 * r121);
    r78 = r130 + r78;
    r86 = fmaf(r14, r78, r86);
    r114 = r114 * r86;
    r93 = fmaf(r44, r114, r93);
    r93 = fmaf(r132, r3, r93);
    r119 = r119 * r48;
    r119 = r119 * r36;
    r119 = r119 * r99;
    r119 = r119 * r35;
    r119 = r119 * r93;
    r35 = r4 * r47;
    r35 = r35 * r36;
    r35 = r35 * r93;
    r35 = r35 * r44;
    r35 = r35 * r99;
    r35 = r35 * r105;
    r35 = fmaf(r26, r35, r119);
    r114 = r47 * r36;
    r114 = r114 * r132;
    r114 = r114 * r48;
    r114 = r114 * r26;
    r35 = fmaf(r96, r114, r35);
    r78 = r126 * r55;
    r35 = fmaf(r106, r78, r35);
    r78 = r4 * r33;
    r78 = r78 * r33;
    r78 = r78 * r36;
    r78 = r78 * r36;
    r78 = r78 * r93;
    r78 = r78 * r44;
    r78 = r78 * r99;
    r78 = fmaf(r105, r78, r86 * r108);
    r114 = r93 * r107;
    r78 = fmaf(r110, r114, r78);
    r78 = fmaf(r132, r113, r78);
    r114 = r35 + r78;
    r119 = fmaf(r93, r103, r63 * r119);
    r130 = r47 * r36;
    r130 = r130 * r102;
    r130 = r130 * r132;
    r130 = r130 * r48;
    r130 = r130 * r97;
    r119 = fmaf(r26, r130, r119);
    r121 = r36 * r90;
    r121 = r121 * r126;
    r119 = fmaf(r55, r121, r119);
    r119 = r119 + r78;
    r119 = fmaf(r60, r119, r8 * r114);
    r78 = r59 * r86;
    r78 = r78 * r55;
    r119 = fmaf(r106, r78, r119);
    r121 = r77 * r93;
    r121 = r121 * r48;
    r121 = r121 * r70;
    r121 = r121 * r99;
    r119 = fmaf(r26, r121, r119);
    r130 = r4 * r132;
    r130 = r130 * r44;
    r130 = r130 * r71;
    r119 = fmaf(r26, r130, r119);
    r81 = r47 * r93;
    r81 = r81 * r41;
    r119 = fmaf(r120, r81, r119);
    r125 = r36 * r126;
    r119 = fmaf(r73, r125, r119);
    r124 = r59 * r40;
    r124 = r124 * r93;
    r124 = r124 * r44;
    r124 = r124 * r99;
    r124 = r124 * r105;
    r124 = r124 * r26;
    r119 = fmaf(r51, r124, r119);
    r80 = r59 * r93;
    r119 = fmaf(r116, r80, r119);
    r133 = r47 * r75;
    r133 = r133 * r93;
    r133 = r133 * r72;
    r119 = fmaf(r41, r133, r119);
    r134 = r4 * r68;
    r134 = r134 * r132;
    r134 = r134 * r44;
    r134 = r134 * r71;
    r119 = fmaf(r26, r134, r119);
    r135 = r59 * r126;
    r119 = fmaf(r108, r135, r119);
    r136 = r7 * r16;
    r136 = r136 * r58;
    r136 = fmaf(r114, r136, r6 * r114);
    r136 = fmaf(r114, r62, r136);
    r136 = fmaf(r114, r61, r136);
    r137 = r136 * r26;
    r119 = fmaf(r72, r137, r119);
    r138 = r68 * r77;
    r138 = r138 * r93;
    r138 = r138 * r48;
    r138 = r138 * r70;
    r138 = r138 * r99;
    r119 = fmaf(r26, r138, r119);
    r139 = r36 * r126;
    r119 = fmaf(r72, r139, r119);
    r119 = fmaf(r132, r118, r119);
    r139 = r0 * r119;
    r138 = r36 * r90;
    r138 = r138 * r86;
    r138 = r138 * r46;
    r137 = r93 * r104;
    r137 = fmaf(r112, r137, r51 * r138);
    r138 = r33 * r33;
    r138 = r138 * r36;
    r138 = r138 * r36;
    r138 = r138 * r102;
    r138 = r138 * r132;
    r138 = r138 * r48;
    r137 = fmaf(r97, r138, r137);
    r135 = r63 * r93;
    r135 = r135 * r107;
    r137 = fmaf(r110, r135, r137);
    r137 = r137 + r35;
    r137 = fmaf(r59, r137, r9 * r114);
    r114 = r4 * r33;
    r114 = r114 * r36;
    r114 = r114 * r132;
    r114 = r114 * r44;
    r137 = fmaf(r71, r114, r137);
    r35 = r60 * r86;
    r35 = r35 * r55;
    r137 = fmaf(r106, r35, r137);
    r135 = r60 * r132;
    r137 = fmaf(r117, r135, r137);
    r138 = r33 * r36;
    r138 = r138 * r68;
    r138 = r138 * r77;
    r138 = r138 * r93;
    r138 = r138 * r48;
    r138 = r138 * r70;
    r137 = fmaf(r99, r138, r137);
    r134 = r136 * r72;
    r137 = fmaf(r51, r134, r137);
    r133 = r75 * r93;
    r133 = r133 * r72;
    r137 = fmaf(r110, r133, r137);
    r80 = r33 * r36;
    r80 = r80 * r77;
    r80 = r80 * r93;
    r80 = r80 * r48;
    r80 = r80 * r70;
    r137 = fmaf(r99, r80, r137);
    r124 = r60 * r40;
    r124 = r124 * r93;
    r124 = r124 * r44;
    r124 = r124 * r99;
    r124 = r124 * r105;
    r124 = r124 * r26;
    r137 = fmaf(r51, r124, r137);
    r125 = r60 * r93;
    r137 = fmaf(r116, r125, r137);
    r81 = r36 * r86;
    r137 = fmaf(r73, r81, r137);
    r130 = r4 * r33;
    r130 = r130 * r36;
    r130 = r130 * r68;
    r130 = r130 * r132;
    r130 = r130 * r44;
    r137 = fmaf(r71, r130, r137);
    r121 = r36 * r86;
    r137 = fmaf(r72, r121, r137);
    r137 = fmaf(r93, r122, r137);
    r137 = fmaf(r126, r101, r137);
    r121 = r5 * r137;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r131,
                                          r115,
                                          r139,
                                          r121);
    r121 = r47 * r47;
    r139 = r25 * r87;
    r31 = fmaf(r77, r31, r75 * r27);
    r31 = fmaf(r75, r28, r31);
    r31 = fmaf(r75, r30, r31);
    r139 = r139 * r31;
    r79 = r87 * r79;
    r30 = r139 + r79;
    r28 = r16 * r32;
    r28 = r28 * r31;
    r89 = r89 + r28;
    r27 = r40 * r29;
    r89 = fmaf(r123, r27, r89);
    r115 = r40 * r34;
    r89 = fmaf(r74, r115, r89);
    r89 = fmaf(r13, r89, r15 * r30);
    r30 = r16 * r34;
    r30 = fmaf(r16, r129, r31 * r30);
    r30 = r30 + r82;
    r89 = fmaf(r14, r30, r89);
    r121 = r121 * r89;
    r30 = r16 * r47;
    r115 = r16 * r29;
    r115 = r115 * r31;
    r91 = r91 + r115;
    r27 = r32 * r40;
    r91 = fmaf(r123, r27, r91);
    r91 = r91 + r94;
    r83 = r32 * r83;
    r83 = r83 * r87;
    r79 = r83 + r79;
    r79 = fmaf(r13, r79, r14 * r91);
    r91 = r16 * r34;
    r91 = fmaf(r74, r91, r28);
    r91 = r91 + r111;
    r79 = fmaf(r15, r91, r79);
    r30 = r30 * r79;
    r30 = fmaf(r44, r30, r96 * r121);
    r121 = r16 * r33;
    r115 = r95 + r115;
    r115 = r115 + r84;
    r84 = r40 * r34;
    r129 = fmaf(r40, r129, r31 * r84);
    r129 = r129 + r82;
    r129 = fmaf(r15, r129, r13 * r115);
    r83 = r139 + r83;
    r129 = fmaf(r14, r83, r129);
    r121 = r121 * r129;
    r30 = fmaf(r44, r121, r30);
    r30 = fmaf(r89, r3, r30);
    r121 = r30 * r107;
    r121 = fmaf(r129, r108, r110 * r121);
    r83 = r4 * r33;
    r83 = r83 * r33;
    r83 = r83 * r36;
    r83 = r83 * r36;
    r83 = r83 * r30;
    r83 = r83 * r44;
    r83 = r83 * r99;
    r121 = fmaf(r105, r83, r121);
    r121 = fmaf(r89, r113, r121);
    r83 = r47 * r36;
    r83 = r83 * r89;
    r83 = r83 * r48;
    r83 = r83 * r26;
    r83 = fmaf(r30, r100, r96 * r83);
    r14 = r4 * r47;
    r14 = r14 * r36;
    r14 = r14 * r30;
    r14 = r14 * r44;
    r14 = r14 * r99;
    r14 = r14 * r105;
    r83 = fmaf(r26, r14, r83);
    r139 = r79 * r55;
    r83 = fmaf(r106, r139, r83);
    r139 = r121 + r83;
    r14 = r47 * r36;
    r14 = r14 * r102;
    r14 = r14 * r89;
    r14 = r14 * r48;
    r14 = r14 * r97;
    r15 = r63 * r30;
    r15 = fmaf(r100, r15, r26 * r14);
    r14 = r36 * r90;
    r14 = r14 * r79;
    r15 = fmaf(r55, r14, r15);
    r15 = fmaf(r30, r103, r15);
    r15 = r15 + r121;
    r15 = fmaf(r60, r15, r8 * r139);
    r121 = r7 * r16;
    r121 = r121 * r58;
    r121 = fmaf(r139, r121, r6 * r139);
    r121 = fmaf(r139, r61, r121);
    r121 = fmaf(r139, r62, r121);
    r14 = r121 * r26;
    r15 = fmaf(r72, r14, r15);
    r115 = r47 * r75;
    r115 = r115 * r30;
    r115 = r115 * r72;
    r15 = fmaf(r41, r115, r15);
    r13 = r68 * r77;
    r13 = r13 * r30;
    r13 = r13 * r48;
    r13 = r13 * r70;
    r13 = r13 * r99;
    r15 = fmaf(r26, r13, r15);
    r82 = r4 * r89;
    r82 = r82 * r44;
    r82 = r82 * r71;
    r15 = fmaf(r26, r82, r15);
    r84 = r59 * r40;
    r84 = r84 * r30;
    r84 = r84 * r44;
    r84 = r84 * r99;
    r84 = r84 * r105;
    r84 = r84 * r26;
    r15 = fmaf(r51, r84, r15);
    r31 = r36 * r79;
    r15 = fmaf(r73, r31, r15);
    r95 = r59 * r79;
    r15 = fmaf(r108, r95, r15);
    r91 = r47 * r30;
    r91 = r91 * r41;
    r15 = fmaf(r120, r91, r15);
    r111 = r59 * r129;
    r111 = r111 * r55;
    r15 = fmaf(r106, r111, r15);
    r28 = r77 * r30;
    r28 = r28 * r48;
    r28 = r28 * r70;
    r28 = r28 * r99;
    r15 = fmaf(r26, r28, r15);
    r74 = r4 * r68;
    r74 = r74 * r89;
    r74 = r74 * r44;
    r74 = r74 * r71;
    r15 = fmaf(r26, r74, r15);
    r87 = r36 * r79;
    r15 = fmaf(r72, r87, r15);
    r94 = r59 * r30;
    r15 = fmaf(r116, r94, r15);
    r15 = fmaf(r89, r118, r15);
    r94 = r0 * r15;
    r87 = r63 * r30;
    r87 = r87 * r107;
    r74 = r36 * r90;
    r74 = r74 * r129;
    r74 = r74 * r46;
    r74 = fmaf(r51, r74, r110 * r87);
    r87 = r33 * r33;
    r87 = r87 * r36;
    r87 = r87 * r36;
    r87 = r87 * r102;
    r87 = r87 * r89;
    r87 = r87 * r48;
    r74 = fmaf(r97, r87, r74);
    r28 = r30 * r104;
    r74 = fmaf(r112, r28, r74);
    r74 = r74 + r83;
    r74 = fmaf(r59, r74, r9 * r139);
    r139 = r60 * r89;
    r74 = fmaf(r117, r139, r74);
    r83 = r33 * r36;
    r83 = r83 * r77;
    r83 = r83 * r30;
    r83 = r83 * r48;
    r83 = r83 * r70;
    r74 = fmaf(r99, r83, r74);
    r28 = r75 * r30;
    r28 = r28 * r72;
    r74 = fmaf(r110, r28, r74);
    r87 = r4 * r33;
    r87 = r87 * r36;
    r87 = r87 * r68;
    r87 = r87 * r89;
    r87 = r87 * r44;
    r74 = fmaf(r71, r87, r74);
    r111 = r36 * r129;
    r74 = fmaf(r73, r111, r74);
    r91 = r60 * r40;
    r91 = r91 * r30;
    r91 = r91 * r44;
    r91 = r91 * r99;
    r91 = r91 * r105;
    r91 = r91 * r26;
    r74 = fmaf(r51, r91, r74);
    r95 = r33 * r36;
    r95 = r95 * r68;
    r95 = r95 * r77;
    r95 = r95 * r30;
    r95 = r95 * r48;
    r95 = r95 * r70;
    r74 = fmaf(r99, r95, r74);
    r31 = r60 * r129;
    r31 = r31 * r55;
    r74 = fmaf(r106, r31, r74);
    r84 = r4 * r33;
    r84 = r84 * r36;
    r84 = r84 * r89;
    r84 = r84 * r44;
    r74 = fmaf(r71, r84, r74);
    r82 = r60 * r30;
    r74 = fmaf(r116, r82, r74);
    r13 = r36 * r129;
    r74 = fmaf(r72, r13, r74);
    r115 = r121 * r72;
    r74 = fmaf(r51, r115, r74);
    r74 = fmaf(r79, r101, r74);
    r74 = fmaf(r30, r122, r74);
    r115 = r5 * r74;
    r13 = r12 * r47;
    r13 = r13 * r36;
    r13 = r13 * r102;
    r13 = r13 * r48;
    r13 = r13 * r97;
    r82 = r42 * r36;
    r82 = r82 * r90;
    r82 = fmaf(r55, r82, r26 * r13);
    r13 = r16 * r38;
    r13 = r13 * r33;
    r13 = fmaf(r44, r13, r12 * r3);
    r84 = r12 * r47;
    r84 = r84 * r47;
    r13 = fmaf(r96, r84, r13);
    r31 = r16 * r42;
    r31 = r31 * r47;
    r13 = fmaf(r44, r31, r13);
    r31 = r63 * r13;
    r82 = fmaf(r100, r31, r82);
    r84 = fmaf(r38, r108, r12 * r113);
    r95 = r4 * r33;
    r95 = r95 * r33;
    r95 = r95 * r36;
    r95 = r95 * r36;
    r95 = r95 * r13;
    r95 = r95 * r44;
    r95 = r95 * r99;
    r84 = fmaf(r105, r95, r84);
    r91 = r13 * r107;
    r84 = fmaf(r110, r91, r84);
    r82 = fmaf(r13, r103, r82);
    r82 = r82 + r84;
    r31 = r12 * r47;
    r31 = r31 * r36;
    r31 = r31 * r48;
    r31 = r31 * r26;
    r91 = r42 * r55;
    r91 = fmaf(r106, r91, r96 * r31);
    r31 = r4 * r47;
    r31 = r31 * r36;
    r31 = r31 * r13;
    r31 = r31 * r44;
    r31 = r31 * r99;
    r31 = r31 * r105;
    r91 = fmaf(r26, r31, r91);
    r91 = fmaf(r13, r100, r91);
    r84 = r84 + r91;
    r82 = fmaf(r8, r84, r60 * r82);
    r31 = r4 * r12;
    r31 = r31 * r44;
    r31 = r31 * r71;
    r82 = fmaf(r26, r31, r82);
    r95 = r42 * r36;
    r82 = fmaf(r72, r95, r82);
    r111 = r68 * r77;
    r111 = r111 * r13;
    r111 = r111 * r48;
    r111 = r111 * r70;
    r111 = r111 * r99;
    r82 = fmaf(r26, r111, r82);
    r87 = r59 * r40;
    r87 = r87 * r13;
    r87 = r87 * r44;
    r87 = r87 * r99;
    r87 = r87 * r105;
    r87 = r87 * r26;
    r82 = fmaf(r51, r87, r82);
    r28 = r42 * r36;
    r82 = fmaf(r73, r28, r82);
    r83 = r59 * r13;
    r82 = fmaf(r116, r83, r82);
    r139 = r59 * r42;
    r82 = fmaf(r108, r139, r82);
    r14 = r77 * r13;
    r14 = r14 * r48;
    r14 = r14 * r70;
    r14 = r14 * r99;
    r82 = fmaf(r26, r14, r82);
    r27 = r47 * r13;
    r27 = r27 * r41;
    r82 = fmaf(r120, r27, r82);
    r123 = r47 * r75;
    r123 = r123 * r13;
    r123 = r123 * r72;
    r82 = fmaf(r41, r123, r82);
    r131 = r59 * r38;
    r131 = r131 * r55;
    r82 = fmaf(r106, r131, r82);
    r130 = r7 * r16;
    r130 = r130 * r58;
    r130 = fmaf(r6, r84, r84 * r130);
    r130 = fmaf(r84, r62, r130);
    r130 = fmaf(r84, r61, r130);
    r81 = r130 * r26;
    r82 = fmaf(r72, r81, r82);
    r125 = r4 * r12;
    r125 = r125 * r68;
    r125 = r125 * r44;
    r125 = r125 * r71;
    r82 = fmaf(r26, r125, r82);
    r82 = fmaf(r12, r118, r82);
    r125 = r0 * r82;
    r81 = r12 * r33;
    r81 = r81 * r33;
    r81 = r81 * r36;
    r81 = r81 * r36;
    r81 = r81 * r102;
    r81 = r81 * r48;
    r131 = r38 * r36;
    r131 = r131 * r90;
    r131 = r131 * r46;
    r131 = fmaf(r51, r131, r97 * r81);
    r81 = r13 * r104;
    r131 = fmaf(r112, r81, r131);
    r123 = r63 * r13;
    r123 = r123 * r107;
    r131 = fmaf(r110, r123, r131);
    r131 = r131 + r91;
    r84 = fmaf(r9, r84, r59 * r131);
    r131 = r4 * r12;
    r131 = r131 * r33;
    r131 = r131 * r36;
    r131 = r131 * r44;
    r84 = fmaf(r71, r131, r84);
    r91 = r4 * r12;
    r91 = r91 * r33;
    r91 = r91 * r36;
    r91 = r91 * r68;
    r91 = r91 * r44;
    r84 = fmaf(r71, r91, r84);
    r123 = r33 * r36;
    r123 = r123 * r77;
    r123 = r123 * r13;
    r123 = r123 * r48;
    r123 = r123 * r70;
    r84 = fmaf(r99, r123, r84);
    r81 = r33 * r36;
    r81 = r81 * r68;
    r81 = r81 * r77;
    r81 = r81 * r13;
    r81 = r81 * r48;
    r81 = r81 * r70;
    r84 = fmaf(r99, r81, r84);
    r27 = r38 * r36;
    r84 = fmaf(r72, r27, r84);
    r14 = r38 * r36;
    r84 = fmaf(r73, r14, r84);
    r139 = r60 * r40;
    r139 = r139 * r13;
    r139 = r139 * r44;
    r139 = r139 * r99;
    r139 = r139 * r105;
    r139 = r139 * r26;
    r84 = fmaf(r51, r139, r84);
    r83 = r60 * r13;
    r84 = fmaf(r116, r83, r84);
    r28 = r75 * r13;
    r28 = r28 * r72;
    r84 = fmaf(r110, r28, r84);
    r87 = r60 * r12;
    r84 = fmaf(r117, r87, r84);
    r111 = r60 * r38;
    r111 = r111 * r55;
    r84 = fmaf(r106, r111, r84);
    r95 = r130 * r72;
    r84 = fmaf(r51, r95, r84);
    r84 = fmaf(r42, r101, r84);
    r84 = fmaf(r13, r122, r84);
    r95 = r5 * r84;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r94,
                                          r115,
                                          r125,
                                          r95);
    r95 = r16 * r43;
    r95 = r95 * r33;
    r95 = fmaf(r44, r95, r39 * r3);
    r125 = r39 * r47;
    r125 = r125 * r47;
    r95 = fmaf(r96, r125, r95);
    r115 = r16 * r53;
    r115 = r115 * r47;
    r95 = fmaf(r44, r115, r95);
    r115 = r95 * r107;
    r115 = fmaf(r39, r113, r110 * r115);
    r125 = r4 * r33;
    r125 = r125 * r33;
    r125 = r125 * r36;
    r125 = r125 * r36;
    r125 = r125 * r95;
    r125 = r125 * r44;
    r125 = r125 * r99;
    r115 = fmaf(r105, r125, r115);
    r115 = fmaf(r43, r108, r115);
    r125 = r39 * r47;
    r125 = r125 * r36;
    r125 = r125 * r48;
    r125 = r125 * r26;
    r125 = fmaf(r96, r125, r95 * r100);
    r94 = r53 * r55;
    r125 = fmaf(r106, r94, r125);
    r111 = r4 * r47;
    r111 = r111 * r36;
    r111 = r111 * r95;
    r111 = r111 * r44;
    r111 = r111 * r99;
    r111 = r111 * r105;
    r125 = fmaf(r26, r111, r125);
    r111 = r115 + r125;
    r94 = r63 * r95;
    r87 = r39 * r47;
    r87 = r87 * r36;
    r87 = r87 * r102;
    r87 = r87 * r48;
    r87 = r87 * r97;
    r87 = fmaf(r26, r87, r100 * r94);
    r94 = r53 * r36;
    r94 = r94 * r90;
    r87 = fmaf(r55, r94, r87);
    r87 = fmaf(r95, r103, r87);
    r87 = r87 + r115;
    r87 = fmaf(r60, r87, r8 * r111);
    r115 = r53 * r36;
    r87 = fmaf(r73, r115, r87);
    r94 = r7 * r16;
    r94 = r94 * r58;
    r94 = fmaf(r111, r94, r6 * r111);
    r94 = fmaf(r111, r62, r94);
    r94 = fmaf(r111, r61, r94);
    r28 = r94 * r26;
    r87 = fmaf(r72, r28, r87);
    r83 = r59 * r43;
    r83 = r83 * r55;
    r87 = fmaf(r106, r83, r87);
    r139 = r77 * r95;
    r139 = r139 * r48;
    r139 = r139 * r70;
    r139 = r139 * r99;
    r87 = fmaf(r26, r139, r87);
    r14 = r53 * r36;
    r87 = fmaf(r72, r14, r87);
    r27 = r59 * r95;
    r87 = fmaf(r116, r27, r87);
    r81 = r47 * r75;
    r81 = r81 * r95;
    r81 = r81 * r72;
    r87 = fmaf(r41, r81, r87);
    r123 = r59 * r40;
    r123 = r123 * r95;
    r123 = r123 * r44;
    r123 = r123 * r99;
    r123 = r123 * r105;
    r123 = r123 * r26;
    r87 = fmaf(r51, r123, r87);
    r91 = r68 * r77;
    r91 = r91 * r95;
    r91 = r91 * r48;
    r91 = r91 * r70;
    r91 = r91 * r99;
    r87 = fmaf(r26, r91, r87);
    r131 = r47 * r95;
    r131 = r131 * r41;
    r87 = fmaf(r120, r131, r87);
    r31 = r4 * r39;
    r31 = r31 * r44;
    r31 = r31 * r71;
    r87 = fmaf(r26, r31, r87);
    r124 = r59 * r53;
    r87 = fmaf(r108, r124, r87);
    r80 = r4 * r39;
    r80 = r80 * r68;
    r80 = r80 * r44;
    r80 = r80 * r71;
    r87 = fmaf(r26, r80, r87);
    r87 = fmaf(r39, r118, r87);
    r80 = r0 * r87;
    r124 = r63 * r95;
    r124 = r124 * r107;
    r31 = r39 * r33;
    r31 = r31 * r33;
    r31 = r31 * r36;
    r31 = r31 * r36;
    r31 = r31 * r102;
    r31 = r31 * r48;
    r31 = fmaf(r97, r31, r110 * r124);
    r124 = r43 * r36;
    r124 = r124 * r90;
    r124 = r124 * r46;
    r31 = fmaf(r51, r124, r31);
    r131 = r95 * r104;
    r31 = fmaf(r112, r131, r31);
    r31 = r31 + r125;
    r31 = fmaf(r59, r31, r9 * r111);
    r111 = r60 * r43;
    r111 = r111 * r55;
    r31 = fmaf(r106, r111, r31);
    r125 = r4 * r39;
    r125 = r125 * r33;
    r125 = r125 * r36;
    r125 = r125 * r44;
    r31 = fmaf(r71, r125, r31);
    r131 = r75 * r95;
    r131 = r131 * r72;
    r31 = fmaf(r110, r131, r31);
    r124 = r60 * r95;
    r31 = fmaf(r116, r124, r31);
    r91 = r43 * r36;
    r31 = fmaf(r73, r91, r31);
    r123 = r60 * r39;
    r31 = fmaf(r117, r123, r31);
    r81 = r60 * r40;
    r81 = r81 * r95;
    r81 = r81 * r44;
    r81 = r81 * r99;
    r81 = r81 * r105;
    r81 = r81 * r26;
    r31 = fmaf(r51, r81, r31);
    r27 = r43 * r36;
    r31 = fmaf(r72, r27, r31);
    r14 = r4 * r39;
    r14 = r14 * r33;
    r14 = r14 * r36;
    r14 = r14 * r68;
    r14 = r14 * r44;
    r31 = fmaf(r71, r14, r31);
    r139 = r33 * r36;
    r139 = r139 * r68;
    r139 = r139 * r77;
    r139 = r139 * r95;
    r139 = r139 * r48;
    r139 = r139 * r70;
    r31 = fmaf(r99, r139, r31);
    r83 = r94 * r72;
    r31 = fmaf(r51, r83, r31);
    r28 = r33 * r36;
    r28 = r28 * r77;
    r28 = r28 * r95;
    r28 = r28 * r48;
    r28 = r28 * r70;
    r31 = fmaf(r99, r28, r31);
    r31 = fmaf(r95, r122, r31);
    r31 = fmaf(r53, r101, r31);
    r28 = r5 * r31;
    r83 = r16 * r45;
    r83 = r83 * r33;
    r3 = fmaf(r37, r3, r44 * r83);
    r83 = r16 * r52;
    r83 = r83 * r47;
    r3 = fmaf(r44, r83, r3);
    r139 = r37 * r47;
    r139 = r139 * r47;
    r3 = fmaf(r96, r139, r3);
    r139 = r3 * r107;
    r139 = fmaf(r45, r108, r110 * r139);
    r83 = r4 * r33;
    r83 = r83 * r33;
    r83 = r83 * r36;
    r83 = r83 * r36;
    r83 = r83 * r3;
    r83 = r83 * r44;
    r83 = r83 * r99;
    r139 = fmaf(r105, r83, r139);
    r139 = fmaf(r37, r113, r139);
    r113 = r37 * r47;
    r113 = r113 * r36;
    r113 = r113 * r48;
    r113 = r113 * r26;
    r83 = r4 * r47;
    r83 = r83 * r36;
    r83 = r83 * r3;
    r83 = r83 * r44;
    r83 = r83 * r99;
    r83 = r83 * r105;
    r83 = fmaf(r26, r83, r96 * r113);
    r113 = r52 * r55;
    r83 = fmaf(r106, r113, r83);
    r83 = fmaf(r3, r100, r83);
    r113 = r139 + r83;
    r96 = r37 * r47;
    r96 = r96 * r36;
    r96 = r96 * r102;
    r96 = r96 * r48;
    r96 = r96 * r97;
    r103 = fmaf(r3, r103, r26 * r96);
    r96 = r52 * r36;
    r96 = r96 * r90;
    r103 = fmaf(r55, r96, r103);
    r14 = r63 * r3;
    r103 = fmaf(r100, r14, r103);
    r103 = r103 + r139;
    r103 = fmaf(r60, r103, r8 * r113);
    r8 = r4 * r37;
    r8 = r8 * r68;
    r8 = r8 * r44;
    r8 = r8 * r71;
    r103 = fmaf(r26, r8, r103);
    r139 = r52 * r36;
    r103 = fmaf(r72, r139, r103);
    r14 = r4 * r37;
    r14 = r14 * r44;
    r14 = r14 * r71;
    r103 = fmaf(r26, r14, r103);
    r96 = r52 * r36;
    r103 = fmaf(r73, r96, r103);
    r100 = r47 * r3;
    r100 = r100 * r41;
    r103 = fmaf(r120, r100, r103);
    r120 = r77 * r3;
    r120 = r120 * r48;
    r120 = r120 * r70;
    r120 = r120 * r99;
    r103 = fmaf(r26, r120, r103);
    r27 = r59 * r40;
    r27 = r27 * r3;
    r27 = r27 * r44;
    r27 = r27 * r99;
    r27 = r27 * r105;
    r27 = r27 * r26;
    r103 = fmaf(r51, r27, r103);
    r116 = r3 * r116;
    r81 = r59 * r45;
    r81 = r81 * r55;
    r103 = fmaf(r106, r81, r103);
    r123 = r59 * r52;
    r103 = fmaf(r108, r123, r103);
    r108 = r68 * r77;
    r108 = r108 * r3;
    r108 = r108 * r48;
    r108 = r108 * r70;
    r108 = r108 * r99;
    r103 = fmaf(r26, r108, r103);
    r91 = r47 * r75;
    r91 = r91 * r3;
    r91 = r91 * r72;
    r103 = fmaf(r41, r91, r103);
    r41 = r7 * r16;
    r41 = r41 * r58;
    r6 = fmaf(r6, r113, r113 * r41);
    r6 = fmaf(r113, r62, r6);
    r6 = fmaf(r113, r61, r6);
    r61 = r6 * r26;
    r103 = fmaf(r72, r61, r103);
    r103 = fmaf(r59, r116, r103);
    r103 = fmaf(r37, r118, r103);
    r118 = r0 * r103;
    r61 = r63 * r3;
    r61 = r61 * r107;
    r91 = r45 * r36;
    r91 = r91 * r90;
    r91 = r91 * r46;
    r91 = fmaf(r51, r91, r110 * r61);
    r61 = r3 * r104;
    r91 = fmaf(r112, r61, r91);
    r112 = r37 * r33;
    r112 = r112 * r33;
    r112 = r112 * r36;
    r112 = r112 * r36;
    r112 = r112 * r102;
    r112 = r112 * r48;
    r91 = fmaf(r97, r112, r91);
    r91 = r91 + r83;
    r91 = fmaf(r59, r91, r9 * r113);
    r113 = r75 * r3;
    r113 = r113 * r72;
    r91 = fmaf(r110, r113, r91);
    r110 = r45 * r36;
    r91 = fmaf(r73, r110, r91);
    r73 = r33 * r36;
    r73 = r73 * r68;
    r73 = r73 * r77;
    r73 = r73 * r3;
    r73 = r73 * r48;
    r73 = r73 * r70;
    r91 = fmaf(r99, r73, r91);
    r9 = r33 * r36;
    r9 = r9 * r77;
    r9 = r9 * r3;
    r9 = r9 * r48;
    r9 = r9 * r70;
    r91 = fmaf(r99, r9, r91);
    r70 = r60 * r40;
    r70 = r70 * r3;
    r70 = r70 * r44;
    r70 = r70 * r99;
    r70 = r70 * r105;
    r70 = r70 * r26;
    r91 = fmaf(r51, r70, r91);
    r99 = r60 * r45;
    r99 = r99 * r55;
    r91 = fmaf(r106, r99, r91);
    r48 = r4 * r37;
    r48 = r48 * r33;
    r48 = r48 * r36;
    r48 = r48 * r68;
    r48 = r48 * r44;
    r91 = fmaf(r71, r48, r91);
    r83 = r45 * r36;
    r91 = fmaf(r72, r83, r91);
    r112 = r4 * r37;
    r112 = r112 * r33;
    r112 = r112 * r36;
    r112 = r112 * r44;
    r91 = fmaf(r71, r112, r91);
    r71 = r6 * r72;
    r91 = fmaf(r51, r71, r91);
    r44 = r60 * r37;
    r91 = fmaf(r117, r44, r91);
    r91 = fmaf(r3, r122, r91);
    r91 = fmaf(r60, r116, r91);
    r91 = fmaf(r52, r101, r91);
    r44 = r5 * r91;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r80,
                                          r28,
                                          r118,
                                          r44);
    r44 = r0 * r4;
    r44 = r44 * r2;
    r118 = r4 * r1;
    r28 = r5 * r118;
    r44 = fmaf(r109, r28, r76 * r44);
    r80 = r0 * r4;
    r80 = r80 * r2;
    r80 = fmaf(r137, r28, r119 * r80);
    r71 = r0 * r4;
    r71 = r71 * r2;
    r71 = fmaf(r74, r28, r15 * r71);
    r112 = r0 * r4;
    r112 = r112 * r2;
    r112 = fmaf(r84, r28, r82 * r112);
    WriteSum4<float, float>((float*)inout_shared, r44, r80, r71, r112);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r112 = r0 * r4;
    r112 = r112 * r2;
    r112 = fmaf(r31, r28, r87 * r112);
    r71 = r0 * r4;
    r71 = r71 * r2;
    r71 = fmaf(r91, r28, r103 * r71);
    WriteSum2<float, float>((float*)inout_shared, r112, r71);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = r0 * r0;
    r112 = r76 * r76;
    r80 = r5 * r5;
    r44 = r109 * r109;
    r44 = fmaf(r80, r44, r71 * r112);
    r112 = r119 * r119;
    r83 = r137 * r137;
    r83 = fmaf(r80, r83, r71 * r112);
    r112 = r74 * r74;
    r101 = r15 * r15;
    r101 = fmaf(r71, r101, r80 * r112);
    r112 = r82 * r82;
    r48 = r84 * r84;
    r48 = fmaf(r80, r48, r71 * r112);
    WriteSum4<float, float>((float*)inout_shared, r44, r83, r101, r48);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r87 * r87;
    r101 = r31 * r31;
    r101 = fmaf(r80, r101, r71 * r48);
    r48 = r91 * r91;
    r83 = r103 * r103;
    r83 = fmaf(r71, r83, r80 * r48);
    WriteSum2<float, float>((float*)inout_shared, r101, r83);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = r109 * r137;
    r101 = r76 * r119;
    r101 = fmaf(r71, r101, r80 * r83);
    r83 = r76 * r15;
    r48 = r109 * r74;
    r48 = fmaf(r80, r48, r71 * r83);
    r83 = r109 * r84;
    r44 = r76 * r82;
    r44 = fmaf(r71, r44, r80 * r83);
    r83 = r109 * r31;
    r112 = r76 * r87;
    r112 = fmaf(r71, r112, r80 * r83);
    WriteSum4<float, float>((float*)inout_shared, r101, r48, r44, r112);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r112 = r76 * r103;
    r44 = r109 * r91;
    r44 = fmaf(r80, r44, r71 * r112);
    r112 = r137 * r74;
    r48 = r119 * r15;
    r48 = fmaf(r71, r48, r80 * r112);
    r112 = r119 * r82;
    r101 = r137 * r84;
    r101 = fmaf(r80, r101, r71 * r112);
    r112 = r119 * r87;
    r83 = r137 * r31;
    r83 = fmaf(r80, r83, r71 * r112);
    WriteSum4<float, float>((float*)inout_shared, r44, r48, r101, r83);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = r137 * r91;
    r101 = r119 * r103;
    r101 = fmaf(r71, r101, r80 * r83);
    r83 = r74 * r84;
    r48 = r15 * r82;
    r48 = fmaf(r71, r48, r80 * r83);
    r83 = r15 * r87;
    r44 = r74 * r31;
    r44 = fmaf(r80, r44, r71 * r83);
    r83 = r74 * r91;
    r112 = r15 * r103;
    r112 = fmaf(r71, r112, r80 * r83);
    WriteSum4<float, float>((float*)inout_shared, r101, r48, r44, r112);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r112 = r84 * r31;
    r44 = r82 * r87;
    r44 = fmaf(r71, r44, r80 * r112);
    r112 = r82 * r103;
    r48 = r84 * r91;
    r48 = fmaf(r80, r48, r71 * r112);
    r112 = r31 * r91;
    r101 = r87 * r103;
    r101 = fmaf(r71, r101, r80 * r112);
    WriteSum3<float, float>((float*)inout_shared, r44, r48, r101);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r101 = r0 * r58;
    r101 = r101 * r26;
    r101 = r101 * r72;
    r48 = r5 * r58;
    r48 = r48 * r72;
    r48 = r48 * r51;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          0 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r11,
                                          r57,
                                          r101,
                                          r48);
    r48 = r5 * r65;
    r101 = r0 * r26;
    r101 = r101 * r72;
    r101 = r101 * r66;
    r44 = r5 * r72;
    r44 = r44 * r51;
    r44 = r44 * r66;
    r112 = r0 * r16;
    r112 = r112 * r51;
    r112 = r112 * r55;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          4 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r101,
                                          r44,
                                          r112,
                                          r48);
    r48 = r0 * r64;
    r112 = r5 * r16;
    r112 = r112 * r51;
    r112 = r112 * r55;
    r44 = r0 * r26;
    r44 = r44 * r72;
    r44 = r44 * r67;
    r101 = r5 * r72;
    r101 = r101 * r51;
    r101 = r101 * r67;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          8 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r48,
                                          r112,
                                          r44,
                                          r101);
    r101 = r0 * r58;
    r44 = r5 * r58;
    r112 = r0 * r26;
    r112 = r112 * r72;
    r112 = r112 * r69;
    r48 = r5 * r72;
    r48 = r48 * r51;
    r48 = r48 * r69;
    WriteIdx4<1024, float, float, float4>(
        out_focal_and_extra_jac,
        12 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r112,
        r48,
        r101,
        r44);
    r44 = r4 * r11;
    r44 = r44 * r2;
    r118 = r57 * r118;
    r101 = r0 * r4;
    r101 = r101 * r58;
    r101 = r101 * r2;
    r101 = r101 * r26;
    r48 = r58 * r72;
    r48 = r48 * r51;
    r48 = fmaf(r28, r48, r72 * r101);
    r101 = r0 * r4;
    r101 = r101 * r2;
    r101 = r101 * r26;
    r101 = r101 * r72;
    r112 = r72 * r51;
    r112 = r112 * r66;
    r112 = fmaf(r28, r112, r66 * r101);
    WriteSum4<float, float>((float*)inout_shared, r44, r118, r48, r112);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           0 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r112 = r0 * r40;
    r112 = r112 * r2;
    r112 = r112 * r51;
    r112 = fmaf(r55, r112, r65 * r28);
    r48 = r0 * r4;
    r48 = r48 * r64;
    r118 = r5 * r40;
    r118 = r118 * r1;
    r118 = r118 * r51;
    r118 = fmaf(r55, r118, r2 * r48);
    r48 = r0 * r4;
    r48 = r48 * r2;
    r48 = r48 * r26;
    r48 = r48 * r72;
    r1 = r72 * r51;
    r1 = r1 * r67;
    r1 = fmaf(r28, r1, r67 * r48);
    r48 = r0 * r4;
    r48 = r48 * r2;
    r48 = r48 * r26;
    r48 = r48 * r72;
    r44 = r72 * r51;
    r44 = r44 * r69;
    r44 = fmaf(r28, r44, r69 * r48);
    WriteSum4<float, float>((float*)inout_shared, r112, r118, r1, r44);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           4 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = r0 * r4;
    r44 = r44 * r58;
    r44 = r44 * r2;
    r28 = r58 * r28;
    WriteSum2<float, float>((float*)inout_shared, r44, r28);
  };
  FlushSumShared<2, float>(out_focal_and_extra_njtr,
                           8 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = r11 * r11;
    r44 = r57 * r57;
    r2 = r33 * r36;
    r2 = r2 * r66;
    r2 = r2 * r80;
    r1 = r36 * r66;
    r1 = r1 * r71;
    r1 = fmaf(r56, r1, r107 * r2);
    r2 = r33 * r36;
    r2 = r2 * r80;
    r2 = r2 * r107;
    r118 = r36 * r71;
    r118 = r118 * r69;
    r118 = fmaf(r56, r118, r69 * r2);
    WriteSum4<float, float>((float*)inout_shared, r28, r44, r1, r118);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_diag,
                           0 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r26 * r71;
    r44 = r47 * r33;
    r54 = r49 * r54;
    r54 = 1.0 / r54;
    r10 = r50 * r10;
    r10 = 1.0 / r10;
    r44 = r44 * r36;
    r44 = r44 * r36;
    r44 = r44 * r128;
    r44 = r44 * r54;
    r44 = r44 * r10;
    r44 = r44 * r51;
    r10 = r65 * r80;
    r54 = fmaf(r65, r10, r1 * r44);
    r128 = r26 * r80;
    r50 = r64 * r64;
    r50 = fmaf(r71, r50, r44 * r128);
    r128 = r33 * r36;
    r44 = r67 * r67;
    r128 = r128 * r80;
    r128 = r128 * r107;
    r49 = r36 * r71;
    r49 = r49 * r56;
    r128 = fmaf(r44, r49, r44 * r128);
    r28 = r69 * r69;
    r2 = r80 * r107;
    r2 = r2 * r51;
    r28 = fmaf(r49, r28, r28 * r2);
    WriteSum4<float, float>((float*)inout_shared, r54, r50, r128, r28);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_diag,
                           4 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = r66 * r71;
    r50 = r66 * r80;
    WriteSum2<float, float>((float*)inout_shared, r28, r50);
  };
  FlushSumShared<2, float>(out_focal_and_extra_precond_diag,
                           8 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = 0.00000000000000000e+00;
    r28 = r0 * r58;
    r28 = r28 * r11;
    r28 = r28 * r26;
    r28 = r28 * r72;
    r54 = r0 * r11;
    r54 = r54 * r26;
    r54 = r54 * r72;
    r54 = r54 * r66;
    r112 = r0 * r16;
    r112 = r112 * r11;
    r112 = r112 * r51;
    r112 = r112 * r55;
    WriteSum4<float, float>((float*)inout_shared, r50, r28, r54, r112);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           0 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r112 = r0 * r64;
    r112 = r112 * r11;
    r54 = r0 * r58;
    r54 = r54 * r11;
    r28 = r0 * r11;
    r28 = r28 * r26;
    r28 = r28 * r72;
    r28 = r28 * r67;
    r11 = r0 * r11;
    r11 = r11 * r26;
    r11 = r11 * r72;
    r11 = r11 * r69;
    WriteSum4<float, float>((float*)inout_shared, r112, r28, r11, r54);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           4 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r5 * r65;
    r65 = r65 * r57;
    r54 = r5 * r58;
    r54 = r54 * r57;
    r54 = r54 * r72;
    r54 = r54 * r51;
    r11 = r5 * r57;
    r11 = r11 * r72;
    r11 = r11 * r51;
    r11 = r11 * r66;
    WriteSum4<float, float>((float*)inout_shared, r50, r54, r11, r65);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           8 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r5 * r16;
    r65 = r65 * r57;
    r65 = r65 * r51;
    r65 = r65 * r55;
    r11 = r5 * r57;
    r11 = r11 * r72;
    r11 = r11 * r51;
    r11 = r11 * r67;
    r54 = r5 * r57;
    r54 = r54 * r72;
    r54 = r54 * r51;
    r54 = r54 * r69;
    WriteSum4<float, float>((float*)inout_shared, r65, r11, r54, r50);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           12 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r5 * r58;
    r50 = r50 * r57;
    r57 = r33 * r36;
    r57 = r57 * r67;
    r57 = r57 * r80;
    r54 = r36 * r67;
    r54 = r54 * r71;
    r54 = fmaf(r56, r54, r107 * r57);
    r57 = r47 * r58;
    r57 = r57 * r97;
    r57 = r57 * r105;
    r57 = r57 * r51;
    r57 = r57 * r106;
    r11 = r51 * r10;
    r65 = r72 * r11;
    r57 = fmaf(r58, r65, r1 * r57);
    r28 = r72 * r1;
    r112 = r64 * r28;
    r48 = r33 * r58;
    r48 = r48 * r97;
    r48 = r48 * r105;
    r48 = r48 * r26;
    r48 = r48 * r51;
    r48 = r48 * r80;
    r48 = fmaf(r106, r48, r58 * r112);
    WriteSum4<float, float>((float*)inout_shared, r50, r54, r57, r48);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           16 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r66 * r28;
    r57 = r72 * r51;
    r57 = r57 * r66;
    r57 = r57 * r80;
    r54 = r33 * r36;
    r50 = r58 * r69;
    r54 = r54 * r80;
    r54 = r54 * r107;
    r101 = r36 * r71;
    r101 = r101 * r56;
    r101 = fmaf(r50, r101, r50 * r54);
    WriteSum4<float, float>((float*)inout_shared, r118, r101, r48, r57);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           20 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = r47 * r97;
    r57 = r57 * r105;
    r57 = r57 * r51;
    r57 = r57 * r66;
    r57 = r57 * r106;
    r57 = fmaf(r66, r65, r1 * r57);
    r48 = r33 * r97;
    r48 = r48 * r105;
    r48 = r48 * r26;
    r48 = r48 * r51;
    r48 = r48 * r66;
    r48 = r48 * r80;
    r48 = fmaf(r106, r48, r66 * r112);
    WriteSum4<float, float>((float*)inout_shared, r57, r48, r101, r128);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           24 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r128 = r67 * r28;
    r101 = r72 * r51;
    r101 = r101 * r67;
    r101 = r101 * r80;
    r48 = r16 * r55;
    r57 = r16 * r64;
    r57 = r57 * r51;
    r57 = r57 * r55;
    r57 = fmaf(r71, r57, r11 * r48);
    r48 = r47 * r97;
    r48 = r48 * r105;
    r48 = r48 * r51;
    r48 = r48 * r67;
    r48 = r48 * r106;
    r48 = fmaf(r67, r65, r1 * r48);
    WriteSum4<float, float>((float*)inout_shared, r128, r101, r57, r48);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           28 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r16 * r58;
    r48 = r48 * r51;
    r48 = r48 * r55;
    r48 = r48 * r71;
    r10 = r58 * r10;
    r57 = r47 * r97;
    r57 = r57 * r105;
    r57 = r57 * r51;
    r57 = r57 * r106;
    r57 = r57 * r69;
    r65 = fmaf(r69, r65, r1 * r57);
    r57 = r33 * r97;
    r57 = r57 * r105;
    r57 = r57 * r26;
    r57 = r57 * r51;
    r57 = r57 * r67;
    r57 = r57 * r80;
    r57 = fmaf(r106, r57, r67 * r112);
    WriteSum4<float, float>((float*)inout_shared, r65, r48, r10, r57);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           32 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = r58 * r64;
    r57 = r57 * r71;
    r10 = r16 * r58;
    r10 = r10 * r51;
    r10 = r10 * r55;
    r10 = r10 * r80;
    r48 = r33 * r97;
    r48 = r48 * r105;
    r48 = r48 * r26;
    r48 = r48 * r51;
    r48 = r48 * r80;
    r48 = r48 * r106;
    r48 = fmaf(r69, r48, r69 * r112);
    r44 = r58 * r44;
    r49 = fmaf(r44, r49, r44 * r2);
    WriteSum4<float, float>((float*)inout_shared, r48, r57, r10, r49);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           36 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r69 * r28;
    r10 = r72 * r51;
    r10 = r10 * r80;
    r10 = r10 * r69;
    r28 = r50 * r28;
    r69 = r72 * r51;
    r69 = r69 * r80;
    r69 = r69 * r50;
    WriteSum4<float, float>((float*)inout_shared, r49, r10, r28, r69);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           40 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void ThinPrismFisheyeSplitFixedPrincipalPointFixedPointResJacFirst(
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
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeSplitFixedPrincipalPointFixedPointResJacFirstKernel<<<
      n_blocks,
      1024>>>(pose,
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
              problem_size);
}

}  // namespace caspar