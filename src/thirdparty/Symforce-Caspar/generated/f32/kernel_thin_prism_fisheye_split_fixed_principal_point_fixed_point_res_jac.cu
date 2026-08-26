#include "kernel_thin_prism_fisheye_split_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPrincipalPointFixedPointResJacKernel(
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
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104,
      r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116,
      r117, r118, r119, r120, r121, r122, r123, r124, r125, r126, r127, r128,
      r129, r130, r131, r132, r133, r134, r135, r136, r137, r138, r139, r140,
      r141;

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
    r3 = r47 * r48;
    r74 = -5.00000000000000000e-01;
    r75 = rsqrtf(r35);
    r76 = r16 * r33;
    r77 = r25 * r40;
    r78 = r17 * r24;
    r79 = r20 * r21;
    r79 = fmaf(r74, r79, r74 * r78);
    r78 = r19 * r22;
    r79 = fmaf(r74, r78, r79);
    r80 = r18 * r23;
    r81 = 5.00000000000000000e-01;
    r79 = fmaf(r81, r80, r79);
    r80 = r20 * r24;
    r78 = r17 * r21;
    r78 = fmaf(r74, r78, r81 * r80);
    r80 = r18 * r22;
    r78 = fmaf(r74, r80, r78);
    r82 = r19 * r23;
    r78 = fmaf(r74, r82, r78);
    r82 = r34 * r78;
    r80 = r40 * r82;
    r77 = fmaf(r79, r77, r80);
    r83 = r16 * r29;
    r84 = fmaf(r81, r31, r74 * r27);
    r84 = fmaf(r74, r28, r84);
    r84 = fmaf(r74, r30, r84);
    r83 = r83 * r84;
    r85 = r16 * r32;
    r86 = r19 * r24;
    r87 = r18 * r21;
    r87 = fmaf(r81, r87, r81 * r86);
    r86 = r17 * r22;
    r87 = fmaf(r74, r86, r87);
    r88 = r20 * r23;
    r87 = fmaf(r81, r88, r87);
    r85 = fmaf(r87, r85, r83);
    r77 = r77 + r85;
    r88 = r25 * r78;
    r86 = -4.00000000000000000e+00;
    r88 = r88 * r86;
    r89 = r32 * r84;
    r90 = r86 * r89;
    r91 = r88 + r90;
    r91 = fmaf(r14, r91, r15 * r77);
    r77 = r16 * r25;
    r77 = r77 * r87;
    r92 = r16 * r34;
    r92 = fmaf(r84, r92, r77);
    r93 = r16 * r29;
    r93 = r93 * r78;
    r94 = r16 * r32;
    r94 = fmaf(r79, r94, r93);
    r92 = r92 + r94;
    r91 = fmaf(r13, r92, r91);
    r76 = r76 * r91;
    r92 = r16 * r47;
    r95 = r16 * r34;
    r96 = r29 * r79;
    r95 = fmaf(r16, r96, r87 * r95);
    r97 = r16 * r25;
    r98 = r16 * r32;
    r98 = r98 * r78;
    r97 = fmaf(r84, r97, r98);
    r95 = r95 + r97;
    r77 = r93 + r77;
    r93 = r32 * r40;
    r77 = fmaf(r79, r93, r77);
    r99 = r40 * r34;
    r77 = fmaf(r84, r99, r77);
    r77 = fmaf(r14, r77, r15 * r95);
    r95 = r29 * r87;
    r95 = r95 * r86;
    r90 = r95 + r90;
    r77 = fmaf(r13, r90, r77);
    r92 = r92 * r77;
    r92 = fmaf(r44, r92, r44 * r76);
    r76 = r16 * r25;
    r76 = r76 * r79;
    r82 = r16 * r82;
    r90 = r76 + r82;
    r85 = r85 + r90;
    r99 = r40 * r34;
    r99 = fmaf(r40, r96, r87 * r99);
    r99 = r99 + r97;
    r99 = fmaf(r13, r99, r14 * r85);
    r95 = r88 + r95;
    r99 = fmaf(r15, r95, r99);
    r95 = r33 * r33;
    r54 = r49 * r54;
    r88 = 1.0 / r54;
    r85 = r40 * r88;
    r95 = r95 * r85;
    r87 = r47 * r47;
    r87 = r87 * r99;
    r92 = fmaf(r85, r87, r92);
    r92 = fmaf(r99, r95, r92);
    r3 = r3 * r36;
    r3 = r3 * r70;
    r3 = r3 * r74;
    r3 = r3 * r75;
    r3 = r3 * r92;
    r87 = r36 * r77;
    r93 = 6.00000000000000000e+00;
    r87 = r87 * r93;
    r100 = r63 * r92;
    r35 = r41 + r35;
    r35 = 1.0 / r35;
    r41 = r75 * r35;
    r101 = r41 * r56;
    r100 = fmaf(r101, r100, r55 * r87);
    r87 = r47 * r26;
    r102 = r36 * r99;
    r103 = -6.00000000000000000e+00;
    r102 = r102 * r103;
    r102 = r102 * r48;
    r102 = r102 * r88;
    r100 = fmaf(r102, r87, r100);
    r104 = r47 * r26;
    r105 = -3.00000000000000000e+00;
    r105 = r36 * r105;
    r10 = r50 * r10;
    r106 = 1.0 / r10;
    r105 = r105 * r44;
    r105 = r105 * r75;
    r105 = r105 * r106;
    r104 = r104 * r105;
    r107 = r16 * r36;
    r108 = r46 * r51;
    r109 = r107 * r108;
    r110 = r92 * r108;
    r111 = r33 * r41;
    r110 = fmaf(r111, r110, r91 * r109);
    r112 = r4 * r33;
    r112 = r112 * r33;
    r112 = r112 * r36;
    r112 = r112 * r36;
    r112 = r112 * r92;
    r112 = r112 * r44;
    r112 = r112 * r75;
    r110 = fmaf(r106, r112, r110);
    r113 = r36 * r95;
    r114 = r36 * r48;
    r114 = r114 * r85;
    r114 = r113 * r114;
    r110 = fmaf(r99, r114, r110);
    r100 = fmaf(r92, r104, r100);
    r100 = r100 + r110;
    r100 = fmaf(r60, r100, r3);
    r87 = r77 * r55;
    r87 = fmaf(r92, r101, r107 * r87);
    r112 = r47 * r36;
    r112 = r112 * r99;
    r112 = r112 * r48;
    r112 = r112 * r26;
    r87 = fmaf(r85, r112, r87);
    r115 = r4 * r47;
    r115 = r115 * r36;
    r115 = r115 * r92;
    r115 = r115 * r44;
    r115 = r115 * r75;
    r115 = r115 * r106;
    r87 = fmaf(r26, r115, r87);
    r110 = r110 + r87;
    r115 = r59 * r92;
    r112 = r16 * r55;
    r112 = r112 * r111;
    r100 = fmaf(r112, r115, r100);
    r116 = r86 * r48;
    r116 = r116 * r88;
    r116 = r116 * r26;
    r116 = r116 * r51;
    r117 = r59 * r116;
    r118 = r47 * r92;
    r119 = r81 * r73;
    r118 = r118 * r41;
    r100 = fmaf(r119, r118, r100);
    r120 = r59 * r77;
    r100 = fmaf(r109, r120, r100);
    r121 = r36 * r77;
    r100 = fmaf(r72, r121, r100);
    r122 = r4 * r99;
    r122 = r122 * r44;
    r122 = r122 * r71;
    r100 = fmaf(r26, r122, r100);
    r123 = r4 * r68;
    r123 = r123 * r99;
    r123 = r123 * r44;
    r123 = r123 * r71;
    r100 = fmaf(r26, r123, r100);
    r124 = r47 * r81;
    r124 = r124 * r92;
    r124 = r124 * r72;
    r100 = fmaf(r41, r124, r100);
    r125 = r59 * r40;
    r125 = r125 * r92;
    r125 = r125 * r44;
    r125 = r125 * r75;
    r125 = r125 * r106;
    r125 = r125 * r26;
    r100 = fmaf(r51, r125, r100);
    r126 = r7 * r16;
    r126 = r126 * r58;
    r126 = fmaf(r110, r126, r6 * r110);
    r127 = 4.00000000000000000e+00;
    r62 = r62 * r127;
    r62 = r62 * r67;
    r61 = r61 * r63;
    r61 = r61 * r66;
    r126 = fmaf(r110, r62, r126);
    r126 = fmaf(r110, r61, r126);
    r128 = r126 * r26;
    r100 = fmaf(r72, r128, r100);
    r129 = r59 * r91;
    r129 = r129 * r55;
    r100 = fmaf(r107, r129, r100);
    r130 = r36 * r77;
    r100 = fmaf(r73, r130, r100);
    r100 = fmaf(r8, r110, r100);
    r100 = fmaf(r68, r3, r100);
    r100 = fmaf(r99, r117, r100);
    r130 = r0 * r100;
    r129 = r36 * r91;
    r129 = r129 * r93;
    r129 = r129 * r46;
    r128 = r63 * r92;
    r128 = r128 * r108;
    r128 = fmaf(r111, r128, r51 * r129);
    r129 = r92 * r105;
    r128 = fmaf(r113, r129, r128);
    r125 = r33 * r33;
    r125 = r125 * r36;
    r128 = fmaf(r102, r125, r128);
    r128 = r128 + r87;
    r110 = fmaf(r9, r110, r59 * r128);
    r128 = r81 * r92;
    r128 = r128 * r72;
    r110 = fmaf(r111, r128, r110);
    r87 = r74 * r92;
    r87 = r87 * r48;
    r87 = r87 * r70;
    r87 = r87 * r75;
    r125 = r68 * r87;
    r110 = fmaf(r51, r125, r110);
    r129 = r126 * r72;
    r110 = fmaf(r51, r129, r110);
    r102 = r4 * r33;
    r102 = r102 * r36;
    r102 = r102 * r68;
    r102 = r102 * r99;
    r102 = r102 * r44;
    r110 = fmaf(r71, r102, r110);
    r124 = r60 * r99;
    r110 = fmaf(r116, r124, r110);
    r123 = r60 * r109;
    r122 = r60 * r92;
    r110 = fmaf(r112, r122, r110);
    r121 = r4 * r33;
    r121 = r121 * r36;
    r121 = r121 * r99;
    r121 = r121 * r44;
    r110 = fmaf(r71, r121, r110);
    r120 = r60 * r40;
    r120 = r120 * r92;
    r120 = r120 * r44;
    r120 = r120 * r75;
    r120 = r120 * r106;
    r120 = r120 * r26;
    r110 = fmaf(r51, r120, r110);
    r118 = r111 * r119;
    r115 = r60 * r91;
    r115 = r115 * r55;
    r110 = fmaf(r107, r115, r110);
    r3 = r36 * r91;
    r110 = fmaf(r73, r3, r110);
    r131 = r36 * r91;
    r110 = fmaf(r72, r131, r110);
    r110 = fmaf(r77, r123, r110);
    r110 = fmaf(r92, r118, r110);
    r110 = fmaf(r87, r51, r110);
    r131 = r5 * r110;
    r3 = r16 * r47;
    r82 = r83 + r82;
    r83 = r16 * r32;
    r115 = r19 * r24;
    r120 = r18 * r21;
    r120 = fmaf(r74, r120, r74 * r115);
    r115 = r17 * r22;
    r120 = fmaf(r81, r115, r120);
    r121 = r20 * r23;
    r120 = fmaf(r74, r121, r120);
    r83 = r83 * r120;
    r121 = r16 * r25;
    r115 = r17 * r24;
    r122 = r20 * r21;
    r122 = fmaf(r81, r122, r81 * r115);
    r115 = r19 * r22;
    r122 = fmaf(r81, r115, r122);
    r124 = r18 * r23;
    r122 = fmaf(r74, r124, r122);
    r121 = fmaf(r122, r121, r83);
    r82 = r82 + r121;
    r124 = r32 * r86;
    r124 = r124 * r122;
    r115 = r29 * r78;
    r115 = r115 * r86;
    r102 = r124 + r115;
    r102 = fmaf(r13, r102, r15 * r82);
    r82 = r40 * r34;
    r82 = fmaf(r40, r89, r122 * r82);
    r129 = r16 * r25;
    r129 = r129 * r78;
    r125 = r16 * r29;
    r125 = fmaf(r120, r125, r129);
    r82 = r82 + r125;
    r102 = fmaf(r14, r82, r102);
    r3 = r3 * r102;
    r82 = r47 * r47;
    r128 = r40 * r29;
    r128 = fmaf(r84, r128, r80);
    r128 = r128 + r121;
    r121 = r16 * r29;
    r121 = r121 * r122;
    r132 = r16 * r34;
    r132 = fmaf(r120, r132, r121);
    r132 = r132 + r97;
    r132 = fmaf(r14, r132, r13 * r128);
    r128 = r25 * r120;
    r97 = r86 * r128;
    r115 = r115 + r97;
    r132 = fmaf(r15, r115, r132);
    r82 = r82 * r132;
    r82 = fmaf(r85, r82, r44 * r3);
    r3 = r16 * r33;
    r115 = r25 * r40;
    r115 = fmaf(r84, r115, r98);
    r98 = r40 * r34;
    r115 = fmaf(r120, r98, r115);
    r115 = r115 + r121;
    r98 = r16 * r34;
    r89 = fmaf(r16, r89, r122 * r98);
    r89 = r89 + r125;
    r89 = fmaf(r13, r89, r15 * r115);
    r97 = r124 + r97;
    r89 = fmaf(r14, r97, r89);
    r3 = r3 * r89;
    r82 = fmaf(r44, r3, r82);
    r82 = fmaf(r132, r95, r82);
    r3 = r4 * r47;
    r3 = r3 * r36;
    r3 = r3 * r82;
    r3 = r3 * r44;
    r3 = r3 * r75;
    r3 = r3 * r106;
    r3 = fmaf(r26, r3, r82 * r101);
    r97 = r47 * r36;
    r97 = r97 * r132;
    r97 = r97 * r48;
    r97 = r97 * r26;
    r3 = fmaf(r85, r97, r3);
    r124 = r102 * r55;
    r3 = fmaf(r107, r124, r3);
    r124 = r4 * r33;
    r124 = r124 * r33;
    r124 = r124 * r36;
    r124 = r124 * r36;
    r124 = r124 * r82;
    r124 = r124 * r44;
    r124 = r124 * r75;
    r124 = fmaf(r106, r124, r89 * r109);
    r97 = r82 * r108;
    r124 = fmaf(r111, r97, r124);
    r124 = fmaf(r132, r114, r124);
    r97 = r3 + r124;
    r115 = r63 * r82;
    r115 = fmaf(r82, r104, r101 * r115);
    r98 = r47 * r36;
    r98 = r98 * r103;
    r98 = r98 * r132;
    r98 = r98 * r48;
    r98 = r98 * r88;
    r115 = fmaf(r26, r98, r115);
    r122 = r36 * r93;
    r122 = r122 * r102;
    r115 = fmaf(r55, r122, r115);
    r115 = r115 + r124;
    r115 = fmaf(r60, r115, r8 * r97);
    r124 = r59 * r89;
    r124 = r124 * r55;
    r115 = fmaf(r107, r124, r115);
    r122 = r74 * r82;
    r122 = r122 * r48;
    r122 = r122 * r70;
    r122 = r122 * r75;
    r115 = fmaf(r26, r122, r115);
    r98 = r4 * r132;
    r98 = r98 * r44;
    r98 = r98 * r71;
    r115 = fmaf(r26, r98, r115);
    r121 = r47 * r82;
    r121 = r121 * r41;
    r115 = fmaf(r119, r121, r115);
    r84 = r36 * r102;
    r115 = fmaf(r73, r84, r115);
    r133 = r59 * r40;
    r133 = r133 * r82;
    r133 = r133 * r44;
    r133 = r133 * r75;
    r133 = r133 * r106;
    r133 = r133 * r26;
    r115 = fmaf(r51, r133, r115);
    r134 = r82 * r112;
    r135 = r47 * r81;
    r135 = r135 * r82;
    r135 = r135 * r72;
    r115 = fmaf(r41, r135, r115);
    r136 = r4 * r68;
    r136 = r136 * r132;
    r136 = r136 * r44;
    r136 = r136 * r71;
    r115 = fmaf(r26, r136, r115);
    r137 = r59 * r102;
    r115 = fmaf(r109, r137, r115);
    r138 = r7 * r16;
    r138 = r138 * r58;
    r138 = fmaf(r97, r138, r6 * r97);
    r138 = fmaf(r97, r62, r138);
    r138 = fmaf(r97, r61, r138);
    r139 = r138 * r26;
    r115 = fmaf(r72, r139, r115);
    r140 = r68 * r74;
    r140 = r140 * r82;
    r140 = r140 * r48;
    r140 = r140 * r70;
    r140 = r140 * r75;
    r115 = fmaf(r26, r140, r115);
    r141 = r36 * r102;
    r115 = fmaf(r72, r141, r115);
    r115 = fmaf(r132, r117, r115);
    r115 = fmaf(r59, r134, r115);
    r141 = r0 * r115;
    r140 = r36 * r93;
    r140 = r140 * r89;
    r140 = r140 * r46;
    r139 = r82 * r105;
    r139 = fmaf(r113, r139, r51 * r140);
    r140 = r33 * r33;
    r140 = r140 * r36;
    r140 = r140 * r36;
    r140 = r140 * r103;
    r140 = r140 * r132;
    r140 = r140 * r48;
    r139 = fmaf(r88, r140, r139);
    r137 = r63 * r82;
    r137 = r137 * r108;
    r139 = fmaf(r111, r137, r139);
    r139 = r139 + r3;
    r139 = fmaf(r59, r139, r9 * r97);
    r97 = r4 * r33;
    r97 = r97 * r36;
    r97 = r97 * r132;
    r97 = r97 * r44;
    r139 = fmaf(r71, r97, r139);
    r3 = r60 * r89;
    r3 = r3 * r55;
    r139 = fmaf(r107, r3, r139);
    r137 = r60 * r132;
    r139 = fmaf(r116, r137, r139);
    r140 = r33 * r36;
    r140 = r140 * r68;
    r140 = r140 * r74;
    r140 = r140 * r82;
    r140 = r140 * r48;
    r140 = r140 * r70;
    r139 = fmaf(r75, r140, r139);
    r136 = r138 * r72;
    r139 = fmaf(r51, r136, r139);
    r135 = r81 * r82;
    r135 = r135 * r72;
    r139 = fmaf(r111, r135, r139);
    r133 = r33 * r36;
    r133 = r133 * r74;
    r133 = r133 * r82;
    r133 = r133 * r48;
    r133 = r133 * r70;
    r139 = fmaf(r75, r133, r139);
    r84 = r60 * r40;
    r84 = r84 * r82;
    r84 = r84 * r44;
    r84 = r84 * r75;
    r84 = r84 * r106;
    r84 = r84 * r26;
    r139 = fmaf(r51, r84, r139);
    r121 = r36 * r89;
    r139 = fmaf(r73, r121, r139);
    r98 = r4 * r33;
    r98 = r98 * r36;
    r98 = r98 * r68;
    r98 = r98 * r132;
    r98 = r98 * r44;
    r139 = fmaf(r71, r98, r139);
    r122 = r36 * r89;
    r139 = fmaf(r72, r122, r139);
    r139 = fmaf(r82, r118, r139);
    r139 = fmaf(r60, r134, r139);
    r139 = fmaf(r102, r123, r139);
    r122 = r5 * r139;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r130,
                                          r131,
                                          r141,
                                          r122);
    r122 = r47 * r47;
    r141 = r25 * r86;
    r31 = fmaf(r74, r31, r81 * r27);
    r31 = fmaf(r81, r28, r31);
    r31 = fmaf(r81, r30, r31);
    r141 = r141 * r31;
    r96 = r86 * r96;
    r30 = r141 + r96;
    r28 = r16 * r32;
    r28 = r28 * r31;
    r129 = r129 + r28;
    r27 = r40 * r29;
    r129 = fmaf(r120, r27, r129);
    r131 = r40 * r34;
    r129 = fmaf(r79, r131, r129);
    r129 = fmaf(r13, r129, r15 * r30);
    r30 = r16 * r34;
    r30 = fmaf(r16, r128, r31 * r30);
    r30 = r30 + r94;
    r129 = fmaf(r14, r30, r129);
    r122 = r122 * r129;
    r30 = r16 * r47;
    r131 = r16 * r29;
    r131 = r131 * r31;
    r76 = r76 + r131;
    r27 = r32 * r40;
    r76 = fmaf(r120, r27, r76);
    r76 = r76 + r80;
    r78 = r32 * r78;
    r78 = r78 * r86;
    r96 = r78 + r96;
    r96 = fmaf(r13, r96, r14 * r76);
    r76 = r16 * r34;
    r76 = fmaf(r79, r76, r28);
    r76 = r76 + r125;
    r96 = fmaf(r15, r76, r96);
    r30 = r30 * r96;
    r30 = fmaf(r44, r30, r85 * r122);
    r122 = r16 * r33;
    r131 = r83 + r131;
    r131 = r131 + r90;
    r90 = r40 * r34;
    r128 = fmaf(r40, r128, r31 * r90);
    r128 = r128 + r94;
    r128 = fmaf(r15, r128, r13 * r131);
    r78 = r141 + r78;
    r128 = fmaf(r14, r78, r128);
    r122 = r122 * r128;
    r30 = fmaf(r44, r122, r30);
    r30 = fmaf(r129, r95, r30);
    r122 = r30 * r108;
    r122 = fmaf(r128, r109, r111 * r122);
    r78 = r4 * r33;
    r78 = r78 * r33;
    r78 = r78 * r36;
    r78 = r78 * r36;
    r78 = r78 * r30;
    r78 = r78 * r44;
    r78 = r78 * r75;
    r122 = fmaf(r106, r78, r122);
    r122 = fmaf(r129, r114, r122);
    r78 = r47 * r36;
    r78 = r78 * r129;
    r78 = r78 * r48;
    r78 = r78 * r26;
    r78 = fmaf(r30, r101, r85 * r78);
    r14 = r4 * r47;
    r14 = r14 * r36;
    r14 = r14 * r30;
    r14 = r14 * r44;
    r14 = r14 * r75;
    r14 = r14 * r106;
    r78 = fmaf(r26, r14, r78);
    r141 = r96 * r55;
    r78 = fmaf(r107, r141, r78);
    r141 = r122 + r78;
    r14 = r47 * r36;
    r14 = r14 * r103;
    r14 = r14 * r129;
    r14 = r14 * r48;
    r14 = r14 * r88;
    r15 = r63 * r30;
    r15 = fmaf(r101, r15, r26 * r14);
    r14 = r36 * r93;
    r14 = r14 * r96;
    r15 = fmaf(r55, r14, r15);
    r15 = fmaf(r30, r104, r15);
    r15 = r15 + r122;
    r15 = fmaf(r60, r15, r8 * r141);
    r122 = r7 * r16;
    r122 = r122 * r58;
    r122 = fmaf(r141, r122, r6 * r141);
    r122 = fmaf(r141, r61, r122);
    r122 = fmaf(r141, r62, r122);
    r14 = r122 * r26;
    r15 = fmaf(r72, r14, r15);
    r131 = r47 * r81;
    r131 = r131 * r30;
    r131 = r131 * r72;
    r15 = fmaf(r41, r131, r15);
    r13 = r68 * r74;
    r13 = r13 * r30;
    r13 = r13 * r48;
    r13 = r13 * r70;
    r13 = r13 * r75;
    r15 = fmaf(r26, r13, r15);
    r94 = r4 * r129;
    r94 = r94 * r44;
    r94 = r94 * r71;
    r15 = fmaf(r26, r94, r15);
    r90 = r59 * r40;
    r90 = r90 * r30;
    r90 = r90 * r44;
    r90 = r90 * r75;
    r90 = r90 * r106;
    r90 = r90 * r26;
    r15 = fmaf(r51, r90, r15);
    r31 = r36 * r96;
    r15 = fmaf(r73, r31, r15);
    r83 = r59 * r96;
    r15 = fmaf(r109, r83, r15);
    r76 = r47 * r30;
    r76 = r76 * r41;
    r15 = fmaf(r119, r76, r15);
    r125 = r59 * r128;
    r125 = r125 * r55;
    r15 = fmaf(r107, r125, r15);
    r28 = r74 * r30;
    r28 = r28 * r48;
    r28 = r28 * r70;
    r28 = r28 * r75;
    r15 = fmaf(r26, r28, r15);
    r79 = r4 * r68;
    r79 = r79 * r129;
    r79 = r79 * r44;
    r79 = r79 * r71;
    r15 = fmaf(r26, r79, r15);
    r86 = r36 * r96;
    r15 = fmaf(r72, r86, r15);
    r80 = r59 * r30;
    r15 = fmaf(r112, r80, r15);
    r15 = fmaf(r129, r117, r15);
    r80 = r0 * r15;
    r86 = r63 * r30;
    r86 = r86 * r108;
    r79 = r36 * r93;
    r79 = r79 * r128;
    r79 = r79 * r46;
    r79 = fmaf(r51, r79, r111 * r86);
    r86 = r33 * r33;
    r86 = r86 * r36;
    r86 = r86 * r36;
    r86 = r86 * r103;
    r86 = r86 * r129;
    r86 = r86 * r48;
    r79 = fmaf(r88, r86, r79);
    r28 = r30 * r105;
    r79 = fmaf(r113, r28, r79);
    r79 = r79 + r78;
    r79 = fmaf(r59, r79, r9 * r141);
    r141 = r60 * r129;
    r79 = fmaf(r116, r141, r79);
    r78 = r33 * r36;
    r78 = r78 * r74;
    r78 = r78 * r30;
    r78 = r78 * r48;
    r78 = r78 * r70;
    r79 = fmaf(r75, r78, r79);
    r28 = r81 * r30;
    r28 = r28 * r72;
    r79 = fmaf(r111, r28, r79);
    r86 = r4 * r33;
    r86 = r86 * r36;
    r86 = r86 * r68;
    r86 = r86 * r129;
    r86 = r86 * r44;
    r79 = fmaf(r71, r86, r79);
    r125 = r36 * r128;
    r79 = fmaf(r73, r125, r79);
    r76 = r60 * r40;
    r76 = r76 * r30;
    r76 = r76 * r44;
    r76 = r76 * r75;
    r76 = r76 * r106;
    r76 = r76 * r26;
    r79 = fmaf(r51, r76, r79);
    r83 = r33 * r36;
    r83 = r83 * r68;
    r83 = r83 * r74;
    r83 = r83 * r30;
    r83 = r83 * r48;
    r83 = r83 * r70;
    r79 = fmaf(r75, r83, r79);
    r31 = r60 * r128;
    r31 = r31 * r55;
    r79 = fmaf(r107, r31, r79);
    r90 = r4 * r33;
    r90 = r90 * r36;
    r90 = r90 * r129;
    r90 = r90 * r44;
    r79 = fmaf(r71, r90, r79);
    r94 = r60 * r30;
    r79 = fmaf(r112, r94, r79);
    r13 = r36 * r128;
    r79 = fmaf(r72, r13, r79);
    r131 = r122 * r72;
    r79 = fmaf(r51, r131, r79);
    r79 = fmaf(r96, r123, r79);
    r79 = fmaf(r30, r118, r79);
    r131 = r5 * r79;
    r13 = r12 * r47;
    r13 = r13 * r36;
    r13 = r13 * r103;
    r13 = r13 * r48;
    r13 = r13 * r88;
    r94 = r42 * r36;
    r94 = r94 * r93;
    r94 = fmaf(r55, r94, r26 * r13);
    r13 = r16 * r38;
    r13 = r13 * r33;
    r13 = fmaf(r44, r13, r12 * r95);
    r90 = r12 * r47;
    r90 = r90 * r47;
    r13 = fmaf(r85, r90, r13);
    r31 = r16 * r42;
    r31 = r31 * r47;
    r13 = fmaf(r44, r31, r13);
    r31 = r63 * r13;
    r94 = fmaf(r101, r31, r94);
    r90 = fmaf(r38, r109, r12 * r114);
    r83 = r4 * r33;
    r83 = r83 * r33;
    r83 = r83 * r36;
    r83 = r83 * r36;
    r83 = r83 * r13;
    r83 = r83 * r44;
    r83 = r83 * r75;
    r90 = fmaf(r106, r83, r90);
    r76 = r13 * r108;
    r90 = fmaf(r111, r76, r90);
    r94 = fmaf(r13, r104, r94);
    r94 = r94 + r90;
    r31 = r12 * r47;
    r31 = r31 * r36;
    r31 = r31 * r48;
    r31 = r31 * r26;
    r76 = r42 * r55;
    r76 = fmaf(r107, r76, r85 * r31);
    r31 = r4 * r47;
    r31 = r31 * r36;
    r31 = r31 * r13;
    r31 = r31 * r44;
    r31 = r31 * r75;
    r31 = r31 * r106;
    r76 = fmaf(r26, r31, r76);
    r76 = fmaf(r13, r101, r76);
    r90 = r90 + r76;
    r94 = fmaf(r8, r90, r60 * r94);
    r31 = r4 * r12;
    r31 = r31 * r44;
    r31 = r31 * r71;
    r94 = fmaf(r26, r31, r94);
    r83 = r42 * r36;
    r94 = fmaf(r72, r83, r94);
    r125 = r68 * r74;
    r125 = r125 * r13;
    r125 = r125 * r48;
    r125 = r125 * r70;
    r125 = r125 * r75;
    r94 = fmaf(r26, r125, r94);
    r86 = r59 * r40;
    r86 = r86 * r13;
    r86 = r86 * r44;
    r86 = r86 * r75;
    r86 = r86 * r106;
    r86 = r86 * r26;
    r94 = fmaf(r51, r86, r94);
    r28 = r42 * r36;
    r94 = fmaf(r73, r28, r94);
    r78 = r59 * r13;
    r94 = fmaf(r112, r78, r94);
    r141 = r59 * r42;
    r94 = fmaf(r109, r141, r94);
    r14 = r74 * r13;
    r14 = r14 * r48;
    r14 = r14 * r70;
    r14 = r14 * r75;
    r94 = fmaf(r26, r14, r94);
    r27 = r47 * r13;
    r27 = r27 * r41;
    r94 = fmaf(r119, r27, r94);
    r120 = r47 * r81;
    r120 = r120 * r13;
    r120 = r120 * r72;
    r94 = fmaf(r41, r120, r94);
    r130 = r59 * r38;
    r130 = r130 * r55;
    r94 = fmaf(r107, r130, r94);
    r98 = r7 * r16;
    r98 = r98 * r58;
    r98 = fmaf(r6, r90, r90 * r98);
    r98 = fmaf(r90, r62, r98);
    r98 = fmaf(r90, r61, r98);
    r121 = r98 * r26;
    r94 = fmaf(r72, r121, r94);
    r134 = r4 * r12;
    r134 = r134 * r68;
    r134 = r134 * r44;
    r134 = r134 * r71;
    r94 = fmaf(r26, r134, r94);
    r94 = fmaf(r12, r117, r94);
    r134 = r0 * r94;
    r121 = r12 * r33;
    r121 = r121 * r33;
    r121 = r121 * r36;
    r121 = r121 * r36;
    r121 = r121 * r103;
    r121 = r121 * r48;
    r130 = r38 * r36;
    r130 = r130 * r93;
    r130 = r130 * r46;
    r130 = fmaf(r51, r130, r88 * r121);
    r121 = r13 * r105;
    r130 = fmaf(r113, r121, r130);
    r120 = r63 * r13;
    r120 = r120 * r108;
    r130 = fmaf(r111, r120, r130);
    r130 = r130 + r76;
    r90 = fmaf(r9, r90, r59 * r130);
    r130 = r4 * r12;
    r130 = r130 * r33;
    r130 = r130 * r36;
    r130 = r130 * r44;
    r90 = fmaf(r71, r130, r90);
    r76 = r4 * r12;
    r76 = r76 * r33;
    r76 = r76 * r36;
    r76 = r76 * r68;
    r76 = r76 * r44;
    r90 = fmaf(r71, r76, r90);
    r120 = r33 * r36;
    r120 = r120 * r74;
    r120 = r120 * r13;
    r120 = r120 * r48;
    r120 = r120 * r70;
    r90 = fmaf(r75, r120, r90);
    r121 = r33 * r36;
    r121 = r121 * r68;
    r121 = r121 * r74;
    r121 = r121 * r13;
    r121 = r121 * r48;
    r121 = r121 * r70;
    r90 = fmaf(r75, r121, r90);
    r27 = r38 * r36;
    r90 = fmaf(r72, r27, r90);
    r14 = r38 * r36;
    r90 = fmaf(r73, r14, r90);
    r141 = r60 * r40;
    r141 = r141 * r13;
    r141 = r141 * r44;
    r141 = r141 * r75;
    r141 = r141 * r106;
    r141 = r141 * r26;
    r90 = fmaf(r51, r141, r90);
    r78 = r60 * r13;
    r90 = fmaf(r112, r78, r90);
    r28 = r81 * r13;
    r28 = r28 * r72;
    r90 = fmaf(r111, r28, r90);
    r86 = r60 * r12;
    r90 = fmaf(r116, r86, r90);
    r125 = r60 * r38;
    r125 = r125 * r55;
    r90 = fmaf(r107, r125, r90);
    r83 = r98 * r72;
    r90 = fmaf(r51, r83, r90);
    r90 = fmaf(r42, r123, r90);
    r90 = fmaf(r13, r118, r90);
    r83 = r5 * r90;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r80,
                                          r131,
                                          r134,
                                          r83);
    r83 = r16 * r43;
    r83 = r83 * r33;
    r83 = fmaf(r44, r83, r39 * r95);
    r134 = r39 * r47;
    r134 = r134 * r47;
    r83 = fmaf(r85, r134, r83);
    r131 = r16 * r53;
    r131 = r131 * r47;
    r83 = fmaf(r44, r131, r83);
    r131 = r83 * r108;
    r131 = fmaf(r39, r114, r111 * r131);
    r134 = r4 * r33;
    r134 = r134 * r33;
    r134 = r134 * r36;
    r134 = r134 * r36;
    r134 = r134 * r83;
    r134 = r134 * r44;
    r134 = r134 * r75;
    r131 = fmaf(r106, r134, r131);
    r131 = fmaf(r43, r109, r131);
    r134 = r39 * r47;
    r134 = r134 * r36;
    r134 = r134 * r48;
    r134 = r134 * r26;
    r134 = fmaf(r85, r134, r83 * r101);
    r80 = r53 * r55;
    r134 = fmaf(r107, r80, r134);
    r125 = r4 * r47;
    r125 = r125 * r36;
    r125 = r125 * r83;
    r125 = r125 * r44;
    r125 = r125 * r75;
    r125 = r125 * r106;
    r134 = fmaf(r26, r125, r134);
    r125 = r131 + r134;
    r80 = r63 * r83;
    r86 = r39 * r47;
    r86 = r86 * r36;
    r86 = r86 * r103;
    r86 = r86 * r48;
    r86 = r86 * r88;
    r86 = fmaf(r26, r86, r101 * r80);
    r80 = r53 * r36;
    r80 = r80 * r93;
    r86 = fmaf(r55, r80, r86);
    r86 = fmaf(r83, r104, r86);
    r86 = r86 + r131;
    r86 = fmaf(r60, r86, r8 * r125);
    r131 = r53 * r36;
    r86 = fmaf(r73, r131, r86);
    r80 = r7 * r16;
    r80 = r80 * r58;
    r80 = fmaf(r125, r80, r6 * r125);
    r80 = fmaf(r125, r62, r80);
    r80 = fmaf(r125, r61, r80);
    r101 = r80 * r26;
    r86 = fmaf(r72, r101, r86);
    r28 = r59 * r43;
    r28 = r28 * r55;
    r86 = fmaf(r107, r28, r86);
    r78 = r74 * r83;
    r78 = r78 * r48;
    r78 = r78 * r70;
    r78 = r78 * r75;
    r86 = fmaf(r26, r78, r86);
    r141 = r53 * r36;
    r86 = fmaf(r72, r141, r86);
    r14 = r59 * r83;
    r86 = fmaf(r112, r14, r86);
    r27 = r47 * r81;
    r27 = r27 * r83;
    r27 = r27 * r72;
    r86 = fmaf(r41, r27, r86);
    r121 = r59 * r40;
    r121 = r121 * r83;
    r121 = r121 * r44;
    r121 = r121 * r75;
    r121 = r121 * r106;
    r121 = r121 * r26;
    r86 = fmaf(r51, r121, r86);
    r120 = r68 * r74;
    r120 = r120 * r83;
    r120 = r120 * r48;
    r120 = r120 * r70;
    r120 = r120 * r75;
    r86 = fmaf(r26, r120, r86);
    r76 = r47 * r83;
    r76 = r76 * r41;
    r86 = fmaf(r119, r76, r86);
    r130 = r4 * r39;
    r130 = r130 * r44;
    r130 = r130 * r71;
    r86 = fmaf(r26, r130, r86);
    r31 = r59 * r53;
    r86 = fmaf(r109, r31, r86);
    r84 = r4 * r39;
    r84 = r84 * r68;
    r84 = r84 * r44;
    r84 = r84 * r71;
    r86 = fmaf(r26, r84, r86);
    r86 = fmaf(r39, r117, r86);
    r84 = r0 * r86;
    r31 = r63 * r83;
    r31 = r31 * r108;
    r130 = r39 * r33;
    r130 = r130 * r33;
    r130 = r130 * r36;
    r130 = r130 * r36;
    r130 = r130 * r103;
    r130 = r130 * r48;
    r130 = fmaf(r88, r130, r111 * r31);
    r31 = r43 * r36;
    r31 = r31 * r93;
    r31 = r31 * r46;
    r130 = fmaf(r51, r31, r130);
    r76 = r83 * r105;
    r130 = fmaf(r113, r76, r130);
    r130 = r130 + r134;
    r130 = fmaf(r59, r130, r9 * r125);
    r125 = r60 * r43;
    r125 = r125 * r55;
    r130 = fmaf(r107, r125, r130);
    r134 = r4 * r39;
    r134 = r134 * r33;
    r134 = r134 * r36;
    r134 = r134 * r44;
    r130 = fmaf(r71, r134, r130);
    r76 = r81 * r83;
    r76 = r76 * r72;
    r130 = fmaf(r111, r76, r130);
    r31 = r60 * r83;
    r130 = fmaf(r112, r31, r130);
    r120 = r43 * r36;
    r130 = fmaf(r73, r120, r130);
    r121 = r60 * r39;
    r130 = fmaf(r116, r121, r130);
    r27 = r60 * r40;
    r27 = r27 * r83;
    r27 = r27 * r44;
    r27 = r27 * r75;
    r27 = r27 * r106;
    r27 = r27 * r26;
    r130 = fmaf(r51, r27, r130);
    r14 = r43 * r36;
    r130 = fmaf(r72, r14, r130);
    r141 = r4 * r39;
    r141 = r141 * r33;
    r141 = r141 * r36;
    r141 = r141 * r68;
    r141 = r141 * r44;
    r130 = fmaf(r71, r141, r130);
    r78 = r33 * r36;
    r78 = r78 * r68;
    r78 = r78 * r74;
    r78 = r78 * r83;
    r78 = r78 * r48;
    r78 = r78 * r70;
    r130 = fmaf(r75, r78, r130);
    r28 = r80 * r72;
    r130 = fmaf(r51, r28, r130);
    r101 = r33 * r36;
    r101 = r101 * r74;
    r101 = r101 * r83;
    r101 = r101 * r48;
    r101 = r101 * r70;
    r130 = fmaf(r75, r101, r130);
    r130 = fmaf(r83, r118, r130);
    r130 = fmaf(r53, r123, r130);
    r101 = r5 * r130;
    r28 = r47 * r47;
    r28 = r44 * r28;
    r78 = r16 * r45;
    r78 = r78 * r33;
    r95 = fmaf(r37, r95, r44 * r78);
    r78 = r16 * r52;
    r78 = r78 * r47;
    r95 = fmaf(r44, r78, r95);
    r141 = r37 * r47;
    r141 = r141 * r47;
    r95 = fmaf(r85, r141, r95);
    r28 = r28 * r48;
    r28 = r28 * r36;
    r28 = r28 * r75;
    r28 = r28 * r35;
    r28 = r28 * r95;
    r35 = r37 * r47;
    r35 = r35 * r36;
    r35 = r35 * r48;
    r35 = r35 * r26;
    r35 = fmaf(r85, r35, r28);
    r85 = r4 * r47;
    r85 = r85 * r36;
    r85 = r85 * r95;
    r85 = r85 * r44;
    r85 = r85 * r75;
    r85 = r85 * r106;
    r35 = fmaf(r26, r85, r35);
    r141 = r52 * r55;
    r35 = fmaf(r107, r141, r35);
    r141 = r95 * r108;
    r141 = fmaf(r45, r109, r111 * r141);
    r85 = r4 * r33;
    r85 = r85 * r33;
    r85 = r85 * r36;
    r85 = r85 * r36;
    r85 = r85 * r95;
    r85 = r85 * r44;
    r85 = r85 * r75;
    r141 = fmaf(r106, r85, r141);
    r141 = fmaf(r37, r114, r141);
    r114 = r35 + r141;
    r85 = r37 * r47;
    r85 = r85 * r36;
    r85 = r85 * r103;
    r85 = r85 * r48;
    r85 = r85 * r88;
    r104 = fmaf(r95, r104, r26 * r85);
    r85 = r52 * r36;
    r85 = r85 * r93;
    r104 = fmaf(r55, r85, r104);
    r104 = fmaf(r63, r28, r104);
    r104 = r104 + r141;
    r104 = fmaf(r60, r104, r8 * r114);
    r8 = r4 * r37;
    r8 = r8 * r68;
    r8 = r8 * r44;
    r8 = r8 * r71;
    r104 = fmaf(r26, r8, r104);
    r141 = r52 * r36;
    r104 = fmaf(r72, r141, r104);
    r28 = r4 * r37;
    r28 = r28 * r44;
    r28 = r28 * r71;
    r104 = fmaf(r26, r28, r104);
    r85 = r52 * r36;
    r104 = fmaf(r73, r85, r104);
    r78 = r47 * r95;
    r78 = r78 * r41;
    r104 = fmaf(r119, r78, r104);
    r119 = r74 * r95;
    r119 = r119 * r48;
    r119 = r119 * r70;
    r119 = r119 * r75;
    r104 = fmaf(r26, r119, r104);
    r14 = r59 * r40;
    r14 = r14 * r95;
    r14 = r14 * r44;
    r14 = r14 * r75;
    r14 = r14 * r106;
    r14 = r14 * r26;
    r104 = fmaf(r51, r14, r104);
    r27 = r59 * r95;
    r104 = fmaf(r112, r27, r104);
    r121 = r59 * r45;
    r121 = r121 * r55;
    r104 = fmaf(r107, r121, r104);
    r120 = r59 * r52;
    r104 = fmaf(r109, r120, r104);
    r109 = r68 * r74;
    r109 = r109 * r95;
    r109 = r109 * r48;
    r109 = r109 * r70;
    r109 = r109 * r75;
    r104 = fmaf(r26, r109, r104);
    r31 = r47 * r81;
    r31 = r31 * r95;
    r31 = r31 * r72;
    r104 = fmaf(r41, r31, r104);
    r41 = r7 * r16;
    r41 = r41 * r58;
    r6 = fmaf(r6, r114, r114 * r41);
    r6 = fmaf(r114, r62, r6);
    r6 = fmaf(r114, r61, r6);
    r61 = r6 * r26;
    r104 = fmaf(r72, r61, r104);
    r104 = fmaf(r37, r117, r104);
    r117 = r0 * r104;
    r61 = r63 * r95;
    r61 = r61 * r108;
    r31 = r45 * r36;
    r31 = r31 * r93;
    r31 = r31 * r46;
    r31 = fmaf(r51, r31, r111 * r61);
    r61 = r95 * r105;
    r31 = fmaf(r113, r61, r31);
    r113 = r37 * r33;
    r113 = r113 * r33;
    r113 = r113 * r36;
    r113 = r113 * r36;
    r113 = r113 * r103;
    r113 = r113 * r48;
    r31 = fmaf(r88, r113, r31);
    r31 = r31 + r35;
    r31 = fmaf(r59, r31, r9 * r114);
    r114 = r81 * r95;
    r114 = r114 * r72;
    r31 = fmaf(r111, r114, r31);
    r111 = r45 * r36;
    r31 = fmaf(r73, r111, r31);
    r73 = r33 * r36;
    r73 = r73 * r68;
    r73 = r73 * r74;
    r73 = r73 * r95;
    r73 = r73 * r48;
    r73 = r73 * r70;
    r31 = fmaf(r75, r73, r31);
    r9 = r33 * r36;
    r9 = r9 * r74;
    r9 = r9 * r95;
    r9 = r9 * r48;
    r9 = r9 * r70;
    r31 = fmaf(r75, r9, r31);
    r70 = r60 * r40;
    r70 = r70 * r95;
    r70 = r70 * r44;
    r70 = r70 * r75;
    r70 = r70 * r106;
    r70 = r70 * r26;
    r31 = fmaf(r51, r70, r31);
    r75 = r60 * r95;
    r31 = fmaf(r112, r75, r31);
    r112 = r60 * r45;
    r112 = r112 * r55;
    r31 = fmaf(r107, r112, r31);
    r48 = r4 * r37;
    r48 = r48 * r33;
    r48 = r48 * r36;
    r48 = r48 * r68;
    r48 = r48 * r44;
    r31 = fmaf(r71, r48, r31);
    r35 = r45 * r36;
    r31 = fmaf(r72, r35, r31);
    r113 = r4 * r37;
    r113 = r113 * r33;
    r113 = r113 * r36;
    r113 = r113 * r44;
    r31 = fmaf(r71, r113, r31);
    r71 = r6 * r72;
    r31 = fmaf(r51, r71, r31);
    r44 = r60 * r37;
    r31 = fmaf(r116, r44, r31);
    r31 = fmaf(r95, r118, r31);
    r31 = fmaf(r52, r123, r31);
    r44 = r5 * r31;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r84,
                                          r101,
                                          r117,
                                          r44);
    r44 = r0 * r4;
    r44 = r44 * r2;
    r117 = r4 * r1;
    r101 = r5 * r117;
    r44 = fmaf(r110, r101, r100 * r44);
    r84 = r0 * r4;
    r84 = r84 * r2;
    r84 = fmaf(r139, r101, r115 * r84);
    r71 = r0 * r4;
    r71 = r71 * r2;
    r71 = fmaf(r79, r101, r15 * r71);
    r113 = r0 * r4;
    r113 = r113 * r2;
    r113 = fmaf(r90, r101, r94 * r113);
    WriteSum4<float, float>((float*)inout_shared, r44, r84, r71, r113);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r113 = r0 * r4;
    r113 = r113 * r2;
    r113 = fmaf(r130, r101, r86 * r113);
    r71 = r0 * r4;
    r71 = r71 * r2;
    r71 = fmaf(r31, r101, r104 * r71);
    WriteSum2<float, float>((float*)inout_shared, r113, r71);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = r0 * r0;
    r113 = r100 * r100;
    r84 = r5 * r5;
    r44 = r110 * r110;
    r44 = fmaf(r84, r44, r71 * r113);
    r113 = r115 * r115;
    r35 = r139 * r139;
    r35 = fmaf(r84, r35, r71 * r113);
    r113 = r79 * r79;
    r123 = r15 * r15;
    r123 = fmaf(r71, r123, r84 * r113);
    r113 = r94 * r94;
    r48 = r90 * r90;
    r48 = fmaf(r84, r48, r71 * r113);
    WriteSum4<float, float>((float*)inout_shared, r44, r35, r123, r48);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r86 * r86;
    r123 = r130 * r130;
    r123 = fmaf(r84, r123, r71 * r48);
    r48 = r31 * r31;
    r35 = r104 * r104;
    r35 = fmaf(r71, r35, r84 * r48);
    WriteSum2<float, float>((float*)inout_shared, r123, r35);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r110 * r139;
    r123 = r100 * r115;
    r123 = fmaf(r71, r123, r84 * r35);
    r35 = r100 * r15;
    r48 = r110 * r79;
    r48 = fmaf(r84, r48, r71 * r35);
    r35 = r110 * r90;
    r44 = r100 * r94;
    r44 = fmaf(r71, r44, r84 * r35);
    r35 = r110 * r130;
    r113 = r100 * r86;
    r113 = fmaf(r71, r113, r84 * r35);
    WriteSum4<float, float>((float*)inout_shared, r123, r48, r44, r113);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r113 = r100 * r104;
    r44 = r110 * r31;
    r44 = fmaf(r84, r44, r71 * r113);
    r113 = r139 * r79;
    r48 = r115 * r15;
    r48 = fmaf(r71, r48, r84 * r113);
    r113 = r115 * r94;
    r123 = r139 * r90;
    r123 = fmaf(r84, r123, r71 * r113);
    r113 = r115 * r86;
    r35 = r139 * r130;
    r35 = fmaf(r84, r35, r71 * r113);
    WriteSum4<float, float>((float*)inout_shared, r44, r48, r123, r35);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r139 * r31;
    r123 = r115 * r104;
    r123 = fmaf(r71, r123, r84 * r35);
    r35 = r79 * r90;
    r48 = r15 * r94;
    r48 = fmaf(r71, r48, r84 * r35);
    r35 = r15 * r86;
    r44 = r79 * r130;
    r44 = fmaf(r84, r44, r71 * r35);
    r35 = r79 * r31;
    r113 = r15 * r104;
    r113 = fmaf(r71, r113, r84 * r35);
    WriteSum4<float, float>((float*)inout_shared, r123, r48, r44, r113);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r113 = r90 * r130;
    r44 = r94 * r86;
    r44 = fmaf(r71, r44, r84 * r113);
    r113 = r94 * r104;
    r48 = r90 * r31;
    r48 = fmaf(r84, r48, r71 * r113);
    r113 = r130 * r31;
    r123 = r86 * r104;
    r123 = fmaf(r71, r123, r84 * r113);
    WriteSum3<float, float>((float*)inout_shared, r44, r48, r123);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r123 = r0 * r58;
    r123 = r123 * r26;
    r123 = r123 * r72;
    r48 = r5 * r58;
    r48 = r48 * r72;
    r48 = r48 * r51;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          0 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r11,
                                          r57,
                                          r123,
                                          r48);
    r48 = r5 * r65;
    r123 = r0 * r26;
    r123 = r123 * r72;
    r123 = r123 * r66;
    r44 = r5 * r72;
    r44 = r44 * r51;
    r44 = r44 * r66;
    r113 = r0 * r16;
    r113 = r113 * r51;
    r113 = r113 * r55;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          4 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r123,
                                          r44,
                                          r113,
                                          r48);
    r48 = r0 * r64;
    r113 = r5 * r16;
    r113 = r113 * r51;
    r113 = r113 * r55;
    r44 = r0 * r26;
    r44 = r44 * r72;
    r44 = r44 * r67;
    r123 = r5 * r72;
    r123 = r123 * r51;
    r123 = r123 * r67;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          8 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r48,
                                          r113,
                                          r44,
                                          r123);
    r123 = r0 * r58;
    r44 = r5 * r58;
    r113 = r0 * r26;
    r113 = r113 * r72;
    r113 = r113 * r69;
    r48 = r5 * r72;
    r48 = r48 * r51;
    r48 = r48 * r69;
    WriteIdx4<1024, float, float, float4>(
        out_focal_and_extra_jac,
        12 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r113,
        r48,
        r123,
        r44);
    r44 = r4 * r11;
    r44 = r44 * r2;
    r117 = r57 * r117;
    r123 = r0 * r4;
    r123 = r123 * r58;
    r123 = r123 * r2;
    r123 = r123 * r26;
    r48 = r58 * r72;
    r48 = r48 * r51;
    r48 = fmaf(r101, r48, r72 * r123);
    r123 = r0 * r4;
    r123 = r123 * r2;
    r123 = r123 * r26;
    r123 = r123 * r72;
    r113 = r72 * r51;
    r113 = r113 * r66;
    r113 = fmaf(r101, r113, r66 * r123);
    WriteSum4<float, float>((float*)inout_shared, r44, r117, r48, r113);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           0 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r113 = r0 * r40;
    r113 = r113 * r2;
    r113 = r113 * r51;
    r113 = fmaf(r55, r113, r65 * r101);
    r48 = r0 * r4;
    r48 = r48 * r64;
    r117 = r5 * r40;
    r117 = r117 * r1;
    r117 = r117 * r51;
    r117 = fmaf(r55, r117, r2 * r48);
    r48 = r0 * r4;
    r48 = r48 * r2;
    r48 = r48 * r26;
    r48 = r48 * r72;
    r1 = r72 * r51;
    r1 = r1 * r67;
    r1 = fmaf(r101, r1, r67 * r48);
    r48 = r0 * r4;
    r48 = r48 * r2;
    r48 = r48 * r26;
    r48 = r48 * r72;
    r44 = r72 * r51;
    r44 = r44 * r69;
    r44 = fmaf(r101, r44, r69 * r48);
    WriteSum4<float, float>((float*)inout_shared, r113, r117, r1, r44);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           4 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = r0 * r4;
    r44 = r44 * r58;
    r44 = r44 * r2;
    r101 = r58 * r101;
    WriteSum2<float, float>((float*)inout_shared, r44, r101);
  };
  FlushSumShared<2, float>(out_focal_and_extra_njtr,
                           8 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r101 = r11 * r11;
    r44 = r57 * r57;
    r2 = r33 * r36;
    r2 = r2 * r66;
    r2 = r2 * r84;
    r1 = r36 * r66;
    r1 = r1 * r71;
    r1 = fmaf(r56, r1, r108 * r2);
    r2 = r33 * r36;
    r2 = r2 * r84;
    r2 = r2 * r108;
    r117 = r36 * r71;
    r117 = r117 * r69;
    r117 = fmaf(r56, r117, r69 * r2);
    WriteSum4<float, float>((float*)inout_shared, r101, r44, r1, r117);
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
    r44 = r44 * r127;
    r44 = r44 * r54;
    r44 = r44 * r10;
    r44 = r44 * r51;
    r10 = r65 * r84;
    r54 = fmaf(r65, r10, r1 * r44);
    r127 = r26 * r84;
    r50 = r64 * r64;
    r50 = fmaf(r71, r50, r44 * r127);
    r127 = r33 * r36;
    r44 = r67 * r67;
    r127 = r127 * r84;
    r127 = r127 * r108;
    r49 = r36 * r71;
    r49 = r49 * r56;
    r127 = fmaf(r44, r49, r44 * r127);
    r101 = r69 * r69;
    r2 = r84 * r108;
    r2 = r2 * r51;
    r101 = fmaf(r49, r101, r101 * r2);
    WriteSum4<float, float>((float*)inout_shared, r54, r50, r127, r101);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_diag,
                           4 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r101 = r66 * r71;
    r50 = r66 * r84;
    WriteSum2<float, float>((float*)inout_shared, r101, r50);
  };
  FlushSumShared<2, float>(out_focal_and_extra_precond_diag,
                           8 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = 0.00000000000000000e+00;
    r101 = r0 * r58;
    r101 = r101 * r11;
    r101 = r101 * r26;
    r101 = r101 * r72;
    r54 = r0 * r11;
    r54 = r54 * r26;
    r54 = r54 * r72;
    r54 = r54 * r66;
    r113 = r0 * r16;
    r113 = r113 * r11;
    r113 = r113 * r51;
    r113 = r113 * r55;
    WriteSum4<float, float>((float*)inout_shared, r50, r101, r54, r113);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           0 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r113 = r0 * r64;
    r113 = r113 * r11;
    r54 = r0 * r58;
    r54 = r54 * r11;
    r101 = r0 * r11;
    r101 = r101 * r26;
    r101 = r101 * r72;
    r101 = r101 * r67;
    r11 = r0 * r11;
    r11 = r11 * r26;
    r11 = r11 * r72;
    r11 = r11 * r69;
    WriteSum4<float, float>((float*)inout_shared, r113, r101, r11, r54);
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
    r57 = r57 * r84;
    r54 = r36 * r67;
    r54 = r54 * r71;
    r54 = fmaf(r56, r54, r108 * r57);
    r57 = r47 * r58;
    r57 = r57 * r88;
    r57 = r57 * r106;
    r57 = r57 * r51;
    r57 = r57 * r107;
    r11 = r51 * r10;
    r65 = r72 * r11;
    r57 = fmaf(r58, r65, r1 * r57);
    r101 = r72 * r1;
    r113 = r64 * r101;
    r48 = r33 * r58;
    r48 = r48 * r88;
    r48 = r48 * r106;
    r48 = r48 * r26;
    r48 = r48 * r51;
    r48 = r48 * r84;
    r48 = fmaf(r107, r48, r58 * r113);
    WriteSum4<float, float>((float*)inout_shared, r50, r54, r57, r48);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           16 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r66 * r101;
    r57 = r72 * r51;
    r57 = r57 * r66;
    r57 = r57 * r84;
    r54 = r33 * r36;
    r50 = r58 * r69;
    r54 = r54 * r84;
    r54 = r54 * r108;
    r123 = r36 * r71;
    r123 = r123 * r56;
    r123 = fmaf(r50, r123, r50 * r54);
    WriteSum4<float, float>((float*)inout_shared, r117, r123, r48, r57);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           20 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = r47 * r88;
    r57 = r57 * r106;
    r57 = r57 * r51;
    r57 = r57 * r66;
    r57 = r57 * r107;
    r57 = fmaf(r66, r65, r1 * r57);
    r48 = r33 * r88;
    r48 = r48 * r106;
    r48 = r48 * r26;
    r48 = r48 * r51;
    r48 = r48 * r66;
    r48 = r48 * r84;
    r48 = fmaf(r107, r48, r66 * r113);
    WriteSum4<float, float>((float*)inout_shared, r57, r48, r123, r127);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           24 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r127 = r67 * r101;
    r123 = r72 * r51;
    r123 = r123 * r67;
    r123 = r123 * r84;
    r48 = r16 * r55;
    r57 = r16 * r64;
    r57 = r57 * r51;
    r57 = r57 * r55;
    r57 = fmaf(r71, r57, r11 * r48);
    r48 = r47 * r88;
    r48 = r48 * r106;
    r48 = r48 * r51;
    r48 = r48 * r67;
    r48 = r48 * r107;
    r48 = fmaf(r67, r65, r1 * r48);
    WriteSum4<float, float>((float*)inout_shared, r127, r123, r57, r48);
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
    r57 = r47 * r88;
    r57 = r57 * r106;
    r57 = r57 * r51;
    r57 = r57 * r107;
    r57 = r57 * r69;
    r65 = fmaf(r69, r65, r1 * r57);
    r57 = r33 * r88;
    r57 = r57 * r106;
    r57 = r57 * r26;
    r57 = r57 * r51;
    r57 = r57 * r67;
    r57 = r57 * r84;
    r57 = fmaf(r107, r57, r67 * r113);
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
    r10 = r10 * r84;
    r48 = r33 * r88;
    r48 = r48 * r106;
    r48 = r48 * r26;
    r48 = r48 * r51;
    r48 = r48 * r84;
    r48 = r48 * r107;
    r48 = fmaf(r69, r48, r69 * r113);
    r44 = r58 * r44;
    r49 = fmaf(r44, r49, r44 * r2);
    WriteSum4<float, float>((float*)inout_shared, r48, r57, r10, r49);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           36 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r69 * r101;
    r10 = r72 * r51;
    r10 = r10 * r84;
    r10 = r10 * r69;
    r101 = r50 * r101;
    r69 = r72 * r51;
    r69 = r69 * r84;
    r69 = r69 * r50;
    WriteSum4<float, float>((float*)inout_shared, r49, r10, r101, r69);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           40 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
}

void ThinPrismFisheyeSplitFixedPrincipalPointFixedPointResJac(
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
  ThinPrismFisheyeSplitFixedPrincipalPointFixedPointResJacKernel<<<n_blocks,
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