#include "kernel_thin_prism_fisheye_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFixedPointResJacKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
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
        float* out_calib_jac,
        unsigned int out_calib_jac_num_alloc,
        float* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        float* const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        float* const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
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
      r129, r130, r131, r132, r133, r134, r135, r136;
  LoadShared<4, float, float>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r0,
                       r1,
                       r2,
                       r3);
  };
  __syncthreads();
  LoadShared<4, float, float>(
      calib, 4 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r4,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r8,
                                         r9,
                                         r10);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r11, r12, r13);
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r14,
                       r15,
                       r16,
                       r17);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r18,
                                         r19,
                                         r20,
                                         r21);
    r22 = r16 * r21;
    r23 = r15 * r18;
    r24 = r22 + r23;
    r25 = r17 * r20;
    r26 = -1.00000000000000000e+00;
    r27 = r14 * r19;
    r24 = r24 + r25;
    r24 = fmaf(r26, r27, r24);
    r28 = r24 * r24;
    r29 = -2.00000000000000000e+00;
    r28 = r28 * r29;
    r30 = 1.00000000000000000e+00;
    r31 = r16 * r18;
    r31 = fmaf(r26, r31, r15 * r21);
    r31 = fmaf(r17, r19, r31);
    r31 = fmaf(r14, r20, r31);
    r32 = r29 * r31;
    r32 = fmaf(r31, r32, r30);
    r33 = r28 + r32;
    r33 = fmaf(r11, r33, r8);
    r8 = 2.00000000000000000e+00;
    r34 = fmaf(r17, r18, r14 * r21);
    r35 = r15 * r20;
    r34 = fmaf(r26, r35, r34);
    r34 = fmaf(r16, r19, r34);
    r35 = r8 * r34;
    r35 = r35 * r31;
    r36 = r24 * r29;
    r37 = fmaf(r15, r19, r14 * r18);
    r37 = fmaf(r16, r20, r37);
    r37 = fmaf(r26, r37, r17 * r21);
    r36 = fmaf(r37, r36, r35);
    r38 = r8 * r24;
    r38 = r38 * r34;
    r39 = r8 * r31;
    r39 = fmaf(r37, r39, r38);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r40,
                       r41,
                       r42);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r43 = r18 * r20;
    r43 = r43 * r8;
    r44 = r19 * r21;
    r44 = fmaf(r8, r44, r43);
    r45 = r20 * r21;
    r46 = r18 * r19;
    r46 = r46 * r8;
    r45 = fmaf(r29, r45, r46);
    r47 = r19 * r19;
    r47 = r47 * r29;
    r48 = r30 + r47;
    r49 = r20 * r20;
    r49 = r49 * r29;
    r48 = r48 + r49;
    r33 = fmaf(r12, r36, r33);
    r33 = fmaf(r13, r39, r33);
    r33 = fmaf(r42, r44, r33);
    r33 = fmaf(r41, r45, r33);
    r33 = fmaf(r40, r48, r33);
    r39 = r33 * r33;
    r36 = 9.99999999999999955e-07;
    r50 = r29 * r31;
    r50 = fmaf(r37, r50, r38);
    r50 = fmaf(r11, r50, r10);
    r10 = r19 * r21;
    r10 = fmaf(r29, r10, r43);
    r43 = r18 * r18;
    r43 = r43 * r29;
    r38 = r30 + r43;
    r38 = r38 + r47;
    r47 = r19 * r20;
    r47 = r47 * r8;
    r51 = r18 * r21;
    r51 = fmaf(r8, r51, r47);
    r52 = r8 * r24;
    r52 = r52 * r31;
    r53 = r8 * r34;
    r53 = fmaf(r37, r53, r52);
    r54 = r34 * r34;
    r54 = r54 * r29;
    r32 = r54 + r32;
    r50 = fmaf(r40, r10, r50);
    r50 = fmaf(r42, r38, r50);
    r50 = fmaf(r41, r51, r50);
    r50 = fmaf(r12, r53, r50);
    r50 = fmaf(r13, r32, r50);
    r32 = copysign(1.0, r50);
    r32 = fmaf(r36, r32, r50);
    r50 = r32 * r32;
    r53 = 1.0 / r50;
    r55 = r8 * r24;
    r55 = fmaf(r37, r55, r35);
    r55 = fmaf(r11, r55, r9);
    r9 = r20 * r21;
    r9 = fmaf(r8, r9, r46);
    r43 = r30 + r43;
    r43 = r43 + r49;
    r49 = r18 * r21;
    r49 = fmaf(r29, r49, r47);
    r47 = r34 * r29;
    r47 = fmaf(r37, r47, r52);
    r54 = r30 + r54;
    r54 = r54 + r28;
    r55 = fmaf(r40, r9, r55);
    r55 = fmaf(r41, r43, r55);
    r55 = fmaf(r42, r49, r55);
    r55 = fmaf(r13, r47, r55);
    r55 = fmaf(r12, r54, r55);
    r54 = r55 * r55;
    r54 = fmaf(r53, r54, r53 * r39);
    r39 = sqrtf(r54);
    r47 = atanf(r39);
    r42 = r55 * r47;
    r41 = copysign(1.0, r39);
    r41 = fmaf(r36, r41, r39);
    r36 = r41 * r41;
    r39 = 1.0 / r36;
    r40 = r53 * r39;
    r28 = r42 * r40;
    r52 = r55 * r28;
    r46 = r47 * r52;
    r35 = r33 * r47;
    r56 = 3.00000000000000000e+00;
    r57 = r33 * r47;
    r35 = r35 * r56;
    r35 = r35 * r40;
    r35 = fmaf(r57, r35, r46);
  };
  LoadShared<4, float, float>(
      calib, 8 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r58,
                       r59,
                       r60,
                       r61);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r62 = r33 * r47;
    r62 = r62 * r40;
    r62 = r62 * r57;
    r46 = r46 + r62;
    r63 = fmaf(r60, r46, r7 * r35);
    r64 = r6 * r8;
    r64 = r64 * r57;
    r63 = fmaf(r28, r64, r63);
    r65 = r46 * r46;
    r66 = fmaf(r5, r65, r4 * r46);
    r67 = r65 * r65;
    r68 = r46 * r65;
    r66 = fmaf(r59, r67, r66);
    r66 = fmaf(r58, r68, r66);
    r69 = 1.0 / r32;
    r70 = 1.0 / r41;
    r71 = r69 * r70;
    r72 = r66 * r71;
    r63 = fmaf(r57, r72, r63);
    r63 = fmaf(r71, r57, r63);
    r2 = fmaf(r0, r63, r2);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r64, r73);
    r2 = fmaf(r64, r26, r2);
    r64 = r47 * r56;
    r64 = fmaf(r52, r64, r62);
    r62 = fmaf(r61, r46, r6 * r64);
    r74 = r7 * r8;
    r74 = r74 * r57;
    r62 = fmaf(r28, r74, r62);
    r62 = fmaf(r42, r71, r62);
    r62 = fmaf(r42, r72, r62);
    r3 = fmaf(r1, r62, r3);
    r3 = fmaf(r73, r26, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r73 = r34 * r29;
    r74 = r14 * r21;
    r75 = -5.00000000000000000e-01;
    r76 = r17 * r18;
    r76 = fmaf(r75, r76, r75 * r74);
    r74 = r16 * r19;
    r76 = fmaf(r75, r74, r76);
    r77 = r15 * r20;
    r78 = 5.00000000000000000e-01;
    r76 = fmaf(r78, r77, r76);
    r77 = r17 * r21;
    r74 = r14 * r18;
    r74 = fmaf(r75, r74, r78 * r77);
    r77 = r15 * r19;
    r74 = fmaf(r75, r77, r74);
    r79 = r16 * r20;
    r74 = fmaf(r75, r79, r74);
    r79 = r37 * r74;
    r77 = r29 * r79;
    r73 = fmaf(r76, r73, r77);
    r80 = r8 * r31;
    r81 = r15 * r21;
    r82 = r16 * r18;
    r82 = fmaf(r78, r82, r75 * r81);
    r81 = r17 * r19;
    r82 = fmaf(r75, r81, r82);
    r83 = r14 * r20;
    r82 = fmaf(r75, r83, r82);
    r80 = r80 * r82;
    r83 = r8 * r24;
    r81 = fmaf(r78, r23, r78 * r22);
    r81 = fmaf(r75, r27, r81);
    r81 = fmaf(r78, r25, r81);
    r83 = fmaf(r81, r83, r80);
    r73 = r73 + r83;
    r84 = r34 * r74;
    r85 = -4.00000000000000000e+00;
    r84 = r84 * r85;
    r86 = r24 * r82;
    r87 = r85 * r86;
    r88 = r84 + r87;
    r88 = fmaf(r12, r88, r13 * r73);
    r73 = r8 * r34;
    r73 = r73 * r81;
    r89 = r8 * r37;
    r89 = fmaf(r82, r89, r73);
    r90 = r8 * r31;
    r90 = r90 * r74;
    r91 = r8 * r24;
    r91 = fmaf(r76, r91, r90);
    r89 = r89 + r91;
    r88 = fmaf(r11, r89, r88);
    r89 = r88 * r28;
    r92 = r8 * r47;
    r93 = r30 + r54;
    r93 = 1.0 / r93;
    r54 = rsqrtf(r54);
    r94 = r93 * r54;
    r95 = r94 * r52;
    r96 = r8 * r55;
    r96 = r96 * r88;
    r97 = r8 * r33;
    r98 = r8 * r37;
    r99 = r31 * r76;
    r98 = fmaf(r8, r99, r81 * r98);
    r100 = r8 * r34;
    r101 = r8 * r24;
    r101 = r101 * r74;
    r100 = fmaf(r82, r100, r101);
    r98 = r98 + r100;
    r73 = r90 + r73;
    r90 = r24 * r29;
    r73 = fmaf(r76, r90, r73);
    r102 = r29 * r37;
    r73 = fmaf(r82, r102, r73);
    r73 = fmaf(r12, r73, r13 * r98);
    r98 = r31 * r81;
    r98 = r98 * r85;
    r87 = r98 + r87;
    r73 = fmaf(r11, r87, r73);
    r97 = r97 * r73;
    r97 = fmaf(r53, r97, r53 * r96);
    r96 = r55 * r55;
    r87 = r8 * r34;
    r87 = r87 * r76;
    r79 = r8 * r79;
    r102 = r87 + r79;
    r83 = r83 + r102;
    r90 = r29 * r37;
    r90 = fmaf(r29, r99, r81 * r90);
    r90 = r90 + r100;
    r90 = fmaf(r11, r90, r12 * r83);
    r98 = r84 + r98;
    r90 = fmaf(r13, r98, r90);
    r50 = r32 * r50;
    r98 = 1.0 / r50;
    r84 = r29 * r98;
    r96 = r96 * r90;
    r97 = fmaf(r84, r96, r97);
    r83 = r33 * r33;
    r83 = r83 * r84;
    r97 = fmaf(r90, r83, r97);
    r89 = fmaf(r97, r95, r92 * r89);
    r96 = r26 * r55;
    r36 = r41 * r36;
    r81 = 1.0 / r36;
    r96 = r96 * r47;
    r96 = r96 * r97;
    r96 = r96 * r53;
    r96 = r96 * r54;
    r96 = r96 * r81;
    r89 = fmaf(r42, r96, r89);
    r103 = r55 * r47;
    r103 = r103 * r90;
    r103 = r103 * r39;
    r103 = r103 * r42;
    r89 = fmaf(r84, r103, r89);
    r103 = r40 * r57;
    r96 = r92 * r103;
    r104 = r97 * r103;
    r105 = r33 * r94;
    r104 = fmaf(r105, r104, r73 * r96);
    r106 = r47 * r83;
    r107 = r47 * r39;
    r107 = r107 * r84;
    r107 = r106 * r107;
    r108 = r26 * r33;
    r108 = r108 * r33;
    r108 = r108 * r47;
    r108 = r108 * r47;
    r108 = r108 * r97;
    r108 = r108 * r53;
    r108 = r108 * r54;
    r104 = fmaf(r81, r108, r104);
    r104 = fmaf(r90, r107, r104);
    r108 = r89 + r104;
    r109 = r47 * r73;
    r110 = 6.00000000000000000e+00;
    r109 = r109 * r110;
    r109 = r109 * r40;
    r111 = r56 * r97;
    r111 = r111 * r103;
    r111 = fmaf(r105, r111, r57 * r109);
    r109 = r33 * r33;
    r112 = r47 * r90;
    r113 = -6.00000000000000000e+00;
    r112 = r112 * r113;
    r112 = r112 * r39;
    r112 = r112 * r98;
    r109 = r109 * r47;
    r111 = fmaf(r112, r109, r111);
    r114 = -3.00000000000000000e+00;
    r114 = r47 * r114;
    r114 = r114 * r53;
    r114 = r114 * r54;
    r114 = r114 * r81;
    r115 = r97 * r114;
    r111 = fmaf(r106, r115, r111);
    r111 = r111 + r89;
    r111 = fmaf(r7, r111, r60 * r108);
    r89 = r78 * r72;
    r115 = r105 * r89;
    r109 = r6 * r88;
    r111 = fmaf(r96, r109, r111);
    r116 = r47 * r73;
    r111 = fmaf(r72, r116, r111);
    r117 = r75 * r97;
    r117 = r117 * r39;
    r117 = r117 * r69;
    r117 = r117 * r54;
    r118 = r6 * r97;
    r119 = r8 * r28;
    r119 = r119 * r105;
    r111 = fmaf(r119, r118, r111);
    r120 = r47 * r73;
    r111 = fmaf(r71, r120, r111);
    r121 = r26 * r33;
    r121 = r121 * r47;
    r121 = r121 * r90;
    r121 = r121 * r53;
    r111 = fmaf(r70, r121, r111);
    r122 = r5 * r8;
    r122 = r122 * r46;
    r122 = fmaf(r108, r122, r4 * r108);
    r58 = r58 * r56;
    r58 = r58 * r65;
    r123 = 4.00000000000000000e+00;
    r59 = r59 * r123;
    r59 = r59 * r68;
    r122 = fmaf(r108, r58, r122);
    r122 = fmaf(r108, r59, r122);
    r124 = r122 * r71;
    r111 = fmaf(r57, r124, r111);
    r125 = r6 * r73;
    r125 = r125 * r28;
    r111 = fmaf(r92, r125, r111);
    r126 = r85 * r39;
    r126 = r126 * r98;
    r126 = r126 * r42;
    r126 = r126 * r57;
    r127 = r6 * r126;
    r128 = r66 * r117;
    r129 = r26 * r33;
    r129 = r129 * r47;
    r129 = r129 * r66;
    r129 = r129 * r90;
    r129 = r129 * r53;
    r111 = fmaf(r70, r129, r111);
    r130 = r78 * r97;
    r130 = r130 * r71;
    r111 = fmaf(r105, r130, r111);
    r131 = r6 * r29;
    r131 = r131 * r97;
    r131 = r131 * r53;
    r131 = r131 * r54;
    r131 = r131 * r81;
    r131 = r131 * r42;
    r111 = fmaf(r57, r131, r111);
    r111 = fmaf(r97, r115, r111);
    r111 = fmaf(r117, r57, r111);
    r111 = fmaf(r90, r127, r111);
    r111 = fmaf(r57, r128, r111);
    r131 = r0 * r111;
    r130 = r47 * r88;
    r130 = r130 * r110;
    r129 = r56 * r97;
    r129 = fmaf(r95, r129, r28 * r130);
    r130 = r55 * r42;
    r130 = r130 * r114;
    r125 = r55 * r42;
    r129 = fmaf(r112, r125, r129);
    r129 = fmaf(r97, r130, r129);
    r129 = r129 + r104;
    r129 = fmaf(r6, r129, r61 * r108);
    r108 = r55 * r78;
    r108 = r108 * r97;
    r108 = r108 * r71;
    r129 = fmaf(r94, r108, r129);
    r104 = r7 * r96;
    r125 = r7 * r97;
    r129 = fmaf(r119, r125, r129);
    r112 = r55 * r97;
    r112 = r112 * r94;
    r129 = fmaf(r89, r112, r129);
    r124 = r26 * r66;
    r124 = r124 * r90;
    r124 = r124 * r53;
    r124 = r124 * r70;
    r129 = fmaf(r42, r124, r129);
    r121 = r122 * r42;
    r129 = fmaf(r71, r121, r129);
    r120 = r7 * r73;
    r120 = r120 * r28;
    r129 = fmaf(r92, r120, r129);
    r118 = r7 * r90;
    r129 = fmaf(r126, r118, r129);
    r116 = r26 * r90;
    r116 = r116 * r53;
    r116 = r116 * r70;
    r129 = fmaf(r42, r116, r129);
    r109 = r7 * r29;
    r109 = r109 * r97;
    r109 = r109 * r53;
    r109 = r109 * r54;
    r109 = r109 * r81;
    r109 = r109 * r42;
    r129 = fmaf(r57, r109, r129);
    r132 = r47 * r88;
    r129 = fmaf(r72, r132, r129);
    r133 = r47 * r88;
    r129 = fmaf(r71, r133, r129);
    r129 = fmaf(r42, r128, r129);
    r129 = fmaf(r88, r104, r129);
    r129 = fmaf(r42, r117, r129);
    r133 = r1 * r129;
    r117 = r55 * r55;
    r117 = r53 * r117;
    r132 = r8 * r33;
    r79 = r80 + r79;
    r80 = r8 * r24;
    r23 = fmaf(r75, r23, r75 * r22);
    r23 = fmaf(r78, r27, r23);
    r23 = fmaf(r75, r25, r23);
    r80 = r80 * r23;
    r25 = r8 * r34;
    r27 = r14 * r21;
    r22 = r17 * r18;
    r22 = fmaf(r78, r22, r78 * r27);
    r27 = r16 * r19;
    r22 = fmaf(r78, r27, r22);
    r109 = r15 * r20;
    r22 = fmaf(r75, r109, r22);
    r25 = fmaf(r22, r25, r80);
    r79 = r79 + r25;
    r109 = r31 * r74;
    r109 = r109 * r85;
    r27 = r24 * r85;
    r27 = r27 * r22;
    r116 = r109 + r27;
    r116 = fmaf(r11, r116, r13 * r79);
    r79 = r29 * r37;
    r79 = fmaf(r29, r86, r22 * r79);
    r118 = r8 * r34;
    r118 = r118 * r74;
    r120 = r8 * r31;
    r120 = fmaf(r23, r120, r118);
    r79 = r79 + r120;
    r116 = fmaf(r12, r79, r116);
    r132 = r132 * r116;
    r79 = r29 * r31;
    r79 = fmaf(r82, r79, r77);
    r79 = r79 + r25;
    r25 = r8 * r31;
    r25 = r25 * r22;
    r121 = r8 * r37;
    r121 = fmaf(r23, r121, r25);
    r121 = r121 + r100;
    r121 = fmaf(r12, r121, r11 * r79);
    r79 = r34 * r23;
    r100 = r85 * r79;
    r109 = r109 + r100;
    r121 = fmaf(r13, r109, r121);
    r132 = fmaf(r121, r83, r53 * r132);
    r109 = r55 * r55;
    r109 = r109 * r121;
    r132 = fmaf(r84, r109, r132);
    r124 = r8 * r55;
    r25 = r101 + r25;
    r101 = r34 * r29;
    r25 = fmaf(r82, r101, r25);
    r82 = r29 * r37;
    r25 = fmaf(r23, r82, r25);
    r82 = r8 * r37;
    r86 = fmaf(r8, r86, r22 * r82);
    r86 = r86 + r120;
    r86 = fmaf(r11, r86, r13 * r25);
    r100 = r27 + r100;
    r86 = fmaf(r12, r100, r86);
    r124 = r124 * r86;
    r132 = fmaf(r53, r124, r132);
    r117 = r117 * r39;
    r117 = r117 * r47;
    r117 = r117 * r93;
    r117 = r117 * r54;
    r117 = r117 * r132;
    r93 = r86 * r28;
    r93 = fmaf(r92, r93, r117);
    r124 = r26 * r55;
    r124 = r124 * r47;
    r124 = r124 * r132;
    r124 = r124 * r53;
    r124 = r124 * r54;
    r124 = r124 * r81;
    r93 = fmaf(r42, r124, r93);
    r109 = r55 * r47;
    r109 = r109 * r121;
    r109 = r109 * r39;
    r109 = r109 * r42;
    r93 = fmaf(r84, r109, r93);
    r109 = r132 * r103;
    r124 = r26 * r33;
    r124 = r124 * r33;
    r124 = r124 * r47;
    r124 = r124 * r47;
    r124 = r124 * r132;
    r124 = r124 * r53;
    r124 = r124 * r54;
    r124 = fmaf(r81, r124, r105 * r109);
    r124 = fmaf(r121, r107, r124);
    r124 = fmaf(r116, r96, r124);
    r109 = r93 + r124;
    r100 = r56 * r132;
    r100 = r100 * r103;
    r27 = r132 * r114;
    r27 = fmaf(r106, r27, r105 * r100);
    r100 = r33 * r33;
    r100 = r100 * r47;
    r100 = r100 * r47;
    r100 = r100 * r113;
    r100 = r100 * r121;
    r100 = r100 * r39;
    r27 = fmaf(r98, r100, r27);
    r25 = r47 * r110;
    r25 = r25 * r116;
    r25 = r25 * r40;
    r27 = fmaf(r57, r25, r27);
    r27 = r27 + r93;
    r27 = fmaf(r7, r27, r60 * r109);
    r93 = r6 * r116;
    r93 = r93 * r28;
    r27 = fmaf(r92, r93, r27);
    r25 = r33 * r47;
    r25 = r25 * r75;
    r25 = r25 * r132;
    r25 = r25 * r39;
    r25 = r25 * r69;
    r27 = fmaf(r54, r25, r27);
    r100 = r26 * r33;
    r100 = r100 * r47;
    r100 = r100 * r121;
    r100 = r100 * r53;
    r27 = fmaf(r70, r100, r27);
    r82 = r33 * r47;
    r82 = r82 * r66;
    r82 = r82 * r75;
    r82 = r82 * r132;
    r82 = r82 * r39;
    r82 = r82 * r69;
    r27 = fmaf(r54, r82, r27);
    r22 = r47 * r116;
    r27 = fmaf(r72, r22, r27);
    r101 = r6 * r132;
    r27 = fmaf(r119, r101, r27);
    r112 = r26 * r33;
    r112 = r112 * r47;
    r112 = r112 * r66;
    r112 = r112 * r121;
    r112 = r112 * r53;
    r27 = fmaf(r70, r112, r27);
    r125 = r5 * r8;
    r125 = r125 * r46;
    r125 = fmaf(r4, r109, r109 * r125);
    r125 = fmaf(r109, r59, r125);
    r125 = fmaf(r109, r58, r125);
    r108 = r125 * r71;
    r27 = fmaf(r57, r108, r27);
    r128 = r78 * r132;
    r128 = r128 * r71;
    r27 = fmaf(r105, r128, r27);
    r134 = r6 * r29;
    r134 = r134 * r132;
    r134 = r134 * r53;
    r134 = r134 * r54;
    r134 = r134 * r81;
    r134 = r134 * r42;
    r27 = fmaf(r57, r134, r27);
    r135 = r6 * r86;
    r27 = fmaf(r96, r135, r27);
    r136 = r47 * r116;
    r27 = fmaf(r71, r136, r27);
    r27 = fmaf(r121, r127, r27);
    r27 = fmaf(r132, r115, r27);
    r136 = r0 * r27;
    r135 = r47 * r110;
    r135 = r135 * r86;
    r135 = fmaf(r132, r130, r28 * r135);
    r134 = r55 * r47;
    r134 = r134 * r113;
    r134 = r134 * r121;
    r134 = r134 * r39;
    r134 = r134 * r98;
    r135 = fmaf(r42, r134, r135);
    r135 = fmaf(r56, r117, r135);
    r135 = r135 + r124;
    r135 = fmaf(r6, r135, r61 * r109);
    r109 = r26 * r121;
    r109 = r109 * r53;
    r109 = r109 * r70;
    r135 = fmaf(r42, r109, r135);
    r124 = r47 * r86;
    r135 = fmaf(r72, r124, r135);
    r117 = r7 * r116;
    r117 = r117 * r28;
    r135 = fmaf(r92, r117, r135);
    r134 = r125 * r42;
    r135 = fmaf(r71, r134, r135);
    r128 = r55 * r132;
    r128 = r128 * r94;
    r135 = fmaf(r89, r128, r135);
    r108 = r66 * r75;
    r108 = r108 * r132;
    r108 = r108 * r39;
    r108 = r108 * r69;
    r108 = r108 * r54;
    r135 = fmaf(r42, r108, r135);
    r112 = r7 * r132;
    r135 = fmaf(r119, r112, r135);
    r101 = r26 * r66;
    r101 = r101 * r121;
    r101 = r101 * r53;
    r101 = r101 * r70;
    r135 = fmaf(r42, r101, r135);
    r22 = r55 * r78;
    r22 = r22 * r132;
    r22 = r22 * r71;
    r135 = fmaf(r94, r22, r135);
    r82 = r75 * r132;
    r82 = r82 * r39;
    r82 = r82 * r69;
    r82 = r82 * r54;
    r135 = fmaf(r42, r82, r135);
    r100 = r7 * r121;
    r135 = fmaf(r126, r100, r135);
    r25 = r7 * r29;
    r25 = r25 * r132;
    r25 = r25 * r53;
    r25 = r25 * r54;
    r25 = r25 * r81;
    r25 = r25 * r42;
    r135 = fmaf(r57, r25, r135);
    r93 = r47 * r86;
    r135 = fmaf(r71, r93, r135);
    r135 = fmaf(r86, r104, r135);
    r93 = r1 * r135;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r131,
                                          r133,
                                          r136,
                                          r93);
    r93 = r34 * r85;
    r136 = r15 * r21;
    r133 = r16 * r18;
    r133 = fmaf(r75, r133, r78 * r136);
    r136 = r17 * r19;
    r133 = fmaf(r78, r136, r133);
    r131 = r14 * r20;
    r133 = fmaf(r78, r131, r133);
    r93 = r93 * r133;
    r99 = r85 * r99;
    r131 = r93 + r99;
    r136 = r8 * r24;
    r136 = r136 * r133;
    r118 = r118 + r136;
    r25 = r29 * r31;
    r118 = fmaf(r23, r25, r118);
    r100 = r29 * r37;
    r118 = fmaf(r76, r100, r118);
    r118 = fmaf(r11, r118, r13 * r131);
    r131 = r8 * r37;
    r131 = fmaf(r8, r79, r133 * r131);
    r131 = r131 + r91;
    r118 = fmaf(r12, r131, r118);
    r131 = r8 * r33;
    r100 = r8 * r31;
    r100 = r100 * r133;
    r87 = r87 + r100;
    r25 = r24 * r29;
    r87 = fmaf(r23, r25, r87);
    r87 = r87 + r77;
    r74 = r24 * r74;
    r74 = r74 * r85;
    r99 = r74 + r99;
    r99 = fmaf(r11, r99, r12 * r87);
    r87 = r8 * r37;
    r87 = fmaf(r76, r87, r136);
    r87 = r87 + r120;
    r99 = fmaf(r13, r87, r99);
    r131 = r131 * r99;
    r131 = fmaf(r53, r131, r118 * r83);
    r87 = r55 * r55;
    r87 = r87 * r118;
    r131 = fmaf(r84, r87, r131);
    r120 = r8 * r55;
    r100 = r80 + r100;
    r100 = r100 + r102;
    r102 = r29 * r37;
    r79 = fmaf(r29, r79, r133 * r102);
    r79 = r79 + r91;
    r79 = fmaf(r13, r79, r11 * r100);
    r93 = r74 + r93;
    r79 = fmaf(r12, r93, r79);
    r120 = r120 * r79;
    r131 = fmaf(r53, r120, r131);
    r120 = r79 * r28;
    r120 = fmaf(r92, r120, r131 * r95);
    r87 = r55 * r47;
    r87 = r87 * r118;
    r87 = r87 * r39;
    r87 = r87 * r42;
    r120 = fmaf(r84, r87, r120);
    r93 = r26 * r55;
    r93 = r93 * r47;
    r93 = r93 * r131;
    r93 = r93 * r53;
    r93 = r93 * r54;
    r93 = r93 * r81;
    r120 = fmaf(r42, r93, r120);
    r93 = r131 * r103;
    r93 = fmaf(r105, r93, r118 * r107);
    r87 = r26 * r33;
    r87 = r87 * r33;
    r87 = r87 * r47;
    r87 = r87 * r47;
    r87 = r87 * r131;
    r87 = r87 * r53;
    r87 = r87 * r54;
    r93 = fmaf(r81, r87, r93);
    r93 = fmaf(r99, r96, r93);
    r87 = r120 + r93;
    r12 = r33 * r33;
    r12 = r12 * r47;
    r12 = r12 * r47;
    r12 = r12 * r113;
    r12 = r12 * r118;
    r12 = r12 * r39;
    r74 = r56 * r131;
    r74 = r74 * r103;
    r74 = fmaf(r105, r74, r98 * r12);
    r12 = r131 * r114;
    r74 = fmaf(r106, r12, r74);
    r13 = r47 * r110;
    r13 = r13 * r99;
    r13 = r13 * r40;
    r74 = fmaf(r57, r13, r74);
    r74 = r74 + r120;
    r74 = fmaf(r7, r74, r60 * r87);
    r120 = r6 * r79;
    r74 = fmaf(r96, r120, r74);
    r13 = r47 * r99;
    r74 = fmaf(r72, r13, r74);
    r12 = r78 * r131;
    r12 = r12 * r71;
    r74 = fmaf(r105, r12, r74);
    r100 = r5 * r8;
    r100 = r100 * r46;
    r100 = fmaf(r87, r100, r4 * r87);
    r100 = fmaf(r87, r58, r100);
    r100 = fmaf(r87, r59, r100);
    r11 = r100 * r71;
    r74 = fmaf(r57, r11, r74);
    r91 = r26 * r33;
    r91 = r91 * r47;
    r91 = r91 * r118;
    r91 = r91 * r53;
    r74 = fmaf(r70, r91, r74);
    r102 = r6 * r131;
    r74 = fmaf(r119, r102, r74);
    r133 = r33 * r47;
    r133 = r133 * r66;
    r133 = r133 * r75;
    r133 = r133 * r131;
    r133 = r133 * r39;
    r133 = r133 * r69;
    r74 = fmaf(r54, r133, r74);
    r80 = r26 * r33;
    r80 = r80 * r47;
    r80 = r80 * r66;
    r80 = r80 * r118;
    r80 = r80 * r53;
    r74 = fmaf(r70, r80, r74);
    r136 = r33 * r47;
    r136 = r136 * r75;
    r136 = r136 * r131;
    r136 = r136 * r39;
    r136 = r136 * r69;
    r74 = fmaf(r54, r136, r74);
    r76 = r6 * r29;
    r76 = r76 * r131;
    r76 = r76 * r53;
    r76 = r76 * r54;
    r76 = r76 * r81;
    r76 = r76 * r42;
    r74 = fmaf(r57, r76, r74);
    r85 = r6 * r99;
    r85 = r85 * r28;
    r74 = fmaf(r92, r85, r74);
    r77 = r47 * r99;
    r74 = fmaf(r71, r77, r74);
    r74 = fmaf(r118, r127, r74);
    r74 = fmaf(r131, r115, r74);
    r77 = r0 * r74;
    r85 = r56 * r131;
    r76 = r47 * r110;
    r76 = r76 * r79;
    r76 = fmaf(r28, r76, r95 * r85);
    r85 = r55 * r47;
    r85 = r85 * r113;
    r85 = r85 * r118;
    r85 = r85 * r39;
    r85 = r85 * r98;
    r76 = fmaf(r42, r85, r76);
    r76 = fmaf(r131, r130, r76);
    r76 = r76 + r93;
    r76 = fmaf(r6, r76, r61 * r87);
    r87 = r100 * r42;
    r76 = fmaf(r71, r87, r76);
    r93 = r75 * r131;
    r93 = r93 * r39;
    r93 = r93 * r69;
    r93 = r93 * r54;
    r76 = fmaf(r42, r93, r76);
    r85 = r55 * r78;
    r85 = r85 * r131;
    r85 = r85 * r71;
    r76 = fmaf(r94, r85, r76);
    r136 = r47 * r79;
    r76 = fmaf(r72, r136, r76);
    r80 = r7 * r131;
    r76 = fmaf(r119, r80, r76);
    r133 = r7 * r118;
    r76 = fmaf(r126, r133, r76);
    r102 = r26 * r66;
    r102 = r102 * r118;
    r102 = r102 * r53;
    r102 = r102 * r70;
    r76 = fmaf(r42, r102, r76);
    r91 = r55 * r131;
    r91 = r91 * r94;
    r76 = fmaf(r89, r91, r76);
    r11 = r26 * r118;
    r11 = r11 * r53;
    r11 = r11 * r70;
    r76 = fmaf(r42, r11, r76);
    r12 = r7 * r29;
    r12 = r12 * r131;
    r12 = r12 * r53;
    r12 = r12 * r54;
    r12 = r12 * r81;
    r12 = r12 * r42;
    r76 = fmaf(r57, r12, r76);
    r13 = r7 * r99;
    r13 = r13 * r28;
    r76 = fmaf(r92, r13, r76);
    r120 = r66 * r75;
    r120 = r120 * r131;
    r120 = r120 * r39;
    r120 = r120 * r69;
    r120 = r120 * r54;
    r76 = fmaf(r42, r120, r76);
    r25 = r47 * r79;
    r76 = fmaf(r71, r25, r76);
    r76 = fmaf(r79, r104, r76);
    r25 = r1 * r76;
    r120 = r10 * r33;
    r120 = r120 * r33;
    r120 = r120 * r47;
    r120 = r120 * r47;
    r120 = r120 * r113;
    r120 = r120 * r39;
    r13 = r48 * r47;
    r13 = r13 * r110;
    r13 = r13 * r40;
    r13 = fmaf(r57, r13, r98 * r120);
    r120 = r10 * r55;
    r120 = r120 * r55;
    r12 = r8 * r9;
    r12 = r12 * r55;
    r12 = fmaf(r53, r12, r84 * r120);
    r120 = r8 * r48;
    r120 = r120 * r33;
    r12 = fmaf(r53, r120, r12);
    r12 = fmaf(r10, r83, r12);
    r120 = r12 * r114;
    r13 = fmaf(r106, r120, r13);
    r11 = r56 * r12;
    r11 = r11 * r103;
    r13 = fmaf(r105, r11, r13);
    r91 = r10 * r55;
    r91 = r91 * r47;
    r91 = r91 * r39;
    r91 = r91 * r42;
    r102 = r9 * r28;
    r102 = fmaf(r92, r102, r84 * r91);
    r91 = r26 * r55;
    r91 = r91 * r47;
    r91 = r91 * r12;
    r91 = r91 * r53;
    r91 = r91 * r54;
    r91 = r91 * r81;
    r102 = fmaf(r42, r91, r102);
    r102 = fmaf(r12, r95, r102);
    r13 = r13 + r102;
    r11 = fmaf(r48, r96, r10 * r107);
    r120 = r26 * r33;
    r120 = r120 * r33;
    r120 = r120 * r47;
    r120 = r120 * r47;
    r120 = r120 * r12;
    r120 = r120 * r53;
    r120 = r120 * r54;
    r11 = fmaf(r81, r120, r11);
    r91 = r12 * r103;
    r11 = fmaf(r105, r91, r11);
    r102 = r102 + r11;
    r13 = fmaf(r60, r102, r7 * r13);
    r91 = r26 * r10;
    r91 = r91 * r33;
    r91 = r91 * r47;
    r91 = r91 * r53;
    r13 = fmaf(r70, r91, r13);
    r120 = r33 * r47;
    r120 = r120 * r66;
    r120 = r120 * r75;
    r120 = r120 * r12;
    r120 = r120 * r39;
    r120 = r120 * r69;
    r13 = fmaf(r54, r120, r13);
    r133 = r26 * r10;
    r133 = r133 * r33;
    r133 = r133 * r47;
    r133 = r133 * r66;
    r133 = r133 * r53;
    r13 = fmaf(r70, r133, r13);
    r80 = r48 * r47;
    r13 = fmaf(r71, r80, r13);
    r136 = r12 * r119;
    r85 = r6 * r48;
    r85 = r85 * r28;
    r13 = fmaf(r92, r85, r13);
    r93 = r48 * r47;
    r13 = fmaf(r72, r93, r13);
    r87 = r33 * r47;
    r87 = r87 * r75;
    r87 = r87 * r12;
    r87 = r87 * r39;
    r87 = r87 * r69;
    r13 = fmaf(r54, r87, r13);
    r23 = r78 * r12;
    r23 = r23 * r71;
    r13 = fmaf(r105, r23, r13);
    r82 = r5 * r8;
    r82 = r82 * r46;
    r82 = fmaf(r4, r102, r102 * r82);
    r82 = fmaf(r102, r59, r82);
    r82 = fmaf(r102, r58, r82);
    r22 = r82 * r71;
    r13 = fmaf(r57, r22, r13);
    r101 = r6 * r9;
    r13 = fmaf(r96, r101, r13);
    r112 = r6 * r29;
    r112 = r112 * r12;
    r112 = r112 * r53;
    r112 = r112 * r54;
    r112 = r112 * r81;
    r112 = r112 * r42;
    r13 = fmaf(r57, r112, r13);
    r13 = fmaf(r12, r115, r13);
    r13 = fmaf(r6, r136, r13);
    r13 = fmaf(r10, r127, r13);
    r112 = r0 * r13;
    r101 = r10 * r55;
    r101 = r101 * r47;
    r101 = r101 * r113;
    r101 = r101 * r39;
    r101 = r101 * r98;
    r22 = r9 * r47;
    r22 = r22 * r110;
    r22 = fmaf(r28, r22, r42 * r101);
    r101 = r56 * r12;
    r22 = fmaf(r95, r101, r22);
    r22 = fmaf(r12, r130, r22);
    r22 = r22 + r11;
    r102 = fmaf(r61, r102, r6 * r22);
    r22 = r82 * r42;
    r102 = fmaf(r71, r22, r102);
    r11 = r26 * r10;
    r11 = r11 * r53;
    r11 = r11 * r70;
    r102 = fmaf(r42, r11, r102);
    r101 = r55 * r12;
    r101 = r101 * r94;
    r102 = fmaf(r89, r101, r102);
    r23 = r75 * r12;
    r23 = r23 * r39;
    r23 = r23 * r69;
    r23 = r23 * r54;
    r102 = fmaf(r42, r23, r102);
    r87 = r9 * r47;
    r102 = fmaf(r71, r87, r102);
    r93 = r26 * r10;
    r93 = r93 * r66;
    r93 = r93 * r53;
    r93 = r93 * r70;
    r102 = fmaf(r42, r93, r102);
    r85 = r7 * r48;
    r85 = r85 * r28;
    r102 = fmaf(r92, r85, r102);
    r80 = r9 * r47;
    r102 = fmaf(r72, r80, r102);
    r133 = r55 * r78;
    r133 = r133 * r12;
    r133 = r133 * r71;
    r102 = fmaf(r94, r133, r102);
    r120 = r66 * r75;
    r120 = r120 * r12;
    r120 = r120 * r39;
    r120 = r120 * r69;
    r120 = r120 * r54;
    r102 = fmaf(r42, r120, r102);
    r91 = r7 * r10;
    r102 = fmaf(r126, r91, r102);
    r108 = r7 * r29;
    r108 = r108 * r12;
    r108 = r108 * r53;
    r108 = r108 * r54;
    r108 = r108 * r81;
    r108 = r108 * r42;
    r102 = fmaf(r57, r108, r102);
    r102 = fmaf(r7, r136, r102);
    r102 = fmaf(r9, r104, r102);
    r108 = r1 * r102;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r77,
                                          r25,
                                          r112,
                                          r108);
    r108 = r51 * r55;
    r108 = r108 * r55;
    r112 = r8 * r43;
    r112 = r112 * r55;
    r112 = fmaf(r53, r112, r84 * r108);
    r108 = r8 * r45;
    r108 = r108 * r33;
    r112 = fmaf(r53, r108, r112);
    r112 = fmaf(r51, r83, r112);
    r108 = r51 * r55;
    r108 = r108 * r47;
    r108 = r108 * r39;
    r108 = r108 * r42;
    r108 = fmaf(r84, r108, r112 * r95);
    r25 = r43 * r28;
    r108 = fmaf(r92, r25, r108);
    r77 = r26 * r55;
    r77 = r77 * r47;
    r77 = r77 * r112;
    r77 = r77 * r53;
    r77 = r77 * r54;
    r77 = r77 * r81;
    r108 = fmaf(r42, r77, r108);
    r77 = r112 * r103;
    r77 = fmaf(r51, r107, r105 * r77);
    r25 = r26 * r33;
    r25 = r25 * r33;
    r25 = r25 * r47;
    r25 = r25 * r47;
    r25 = r25 * r112;
    r25 = r25 * r53;
    r25 = r25 * r54;
    r77 = fmaf(r81, r25, r77);
    r77 = fmaf(r45, r96, r77);
    r25 = r108 + r77;
    r91 = r56 * r112;
    r91 = r91 * r103;
    r120 = r51 * r33;
    r120 = r120 * r33;
    r120 = r120 * r47;
    r120 = r120 * r47;
    r120 = r120 * r113;
    r120 = r120 * r39;
    r120 = fmaf(r98, r120, r105 * r91);
    r91 = r45 * r47;
    r91 = r91 * r110;
    r91 = r91 * r40;
    r120 = fmaf(r57, r91, r120);
    r133 = r112 * r114;
    r120 = fmaf(r106, r133, r120);
    r120 = r120 + r108;
    r120 = fmaf(r7, r120, r60 * r25);
    r108 = r6 * r45;
    r108 = r108 * r28;
    r120 = fmaf(r92, r108, r120);
    r133 = r45 * r47;
    r120 = fmaf(r72, r133, r120);
    r91 = r26 * r51;
    r91 = r91 * r33;
    r91 = r91 * r47;
    r91 = r91 * r66;
    r91 = r91 * r53;
    r120 = fmaf(r70, r91, r120);
    r80 = r33 * r47;
    r80 = r80 * r75;
    r80 = r80 * r112;
    r80 = r80 * r39;
    r80 = r80 * r69;
    r120 = fmaf(r54, r80, r120);
    r85 = r5 * r8;
    r85 = r85 * r46;
    r85 = fmaf(r4, r25, r25 * r85);
    r85 = fmaf(r25, r59, r85);
    r85 = fmaf(r25, r58, r85);
    r93 = r85 * r71;
    r120 = fmaf(r57, r93, r120);
    r87 = r6 * r29;
    r87 = r87 * r112;
    r87 = r87 * r53;
    r87 = r87 * r54;
    r87 = r87 * r81;
    r87 = r87 * r42;
    r120 = fmaf(r57, r87, r120);
    r136 = r45 * r47;
    r120 = fmaf(r71, r136, r120);
    r23 = r78 * r112;
    r23 = r23 * r71;
    r120 = fmaf(r105, r23, r120);
    r101 = r6 * r43;
    r120 = fmaf(r96, r101, r120);
    r11 = r6 * r112;
    r120 = fmaf(r119, r11, r120);
    r22 = r26 * r51;
    r22 = r22 * r33;
    r22 = r22 * r47;
    r22 = r22 * r53;
    r120 = fmaf(r70, r22, r120);
    r128 = r33 * r47;
    r128 = r128 * r66;
    r128 = r128 * r75;
    r128 = r128 * r112;
    r128 = r128 * r39;
    r128 = r128 * r69;
    r120 = fmaf(r54, r128, r120);
    r120 = fmaf(r112, r115, r120);
    r120 = fmaf(r51, r127, r120);
    r128 = r0 * r120;
    r22 = r56 * r112;
    r11 = r51 * r55;
    r11 = r11 * r47;
    r11 = r11 * r113;
    r11 = r11 * r39;
    r11 = r11 * r98;
    r11 = fmaf(r42, r11, r95 * r22);
    r22 = r43 * r47;
    r22 = r22 * r110;
    r11 = fmaf(r28, r22, r11);
    r11 = fmaf(r112, r130, r11);
    r11 = r11 + r77;
    r11 = fmaf(r6, r11, r61 * r25);
    r25 = r26 * r51;
    r25 = r25 * r53;
    r25 = r25 * r70;
    r11 = fmaf(r42, r25, r11);
    r77 = r7 * r45;
    r77 = r77 * r28;
    r11 = fmaf(r92, r77, r11);
    r22 = r55 * r112;
    r22 = r22 * r94;
    r11 = fmaf(r89, r22, r11);
    r101 = r55 * r78;
    r101 = r101 * r112;
    r101 = r101 * r71;
    r11 = fmaf(r94, r101, r11);
    r23 = r66 * r75;
    r23 = r23 * r112;
    r23 = r23 * r39;
    r23 = r23 * r69;
    r23 = r23 * r54;
    r11 = fmaf(r42, r23, r11);
    r136 = r7 * r29;
    r136 = r136 * r112;
    r136 = r136 * r53;
    r136 = r136 * r54;
    r136 = r136 * r81;
    r136 = r136 * r42;
    r11 = fmaf(r57, r136, r11);
    r87 = r85 * r42;
    r11 = fmaf(r71, r87, r11);
    r93 = r7 * r112;
    r11 = fmaf(r119, r93, r11);
    r80 = r43 * r47;
    r11 = fmaf(r71, r80, r11);
    r91 = r26 * r51;
    r91 = r91 * r66;
    r91 = r91 * r53;
    r91 = r91 * r70;
    r11 = fmaf(r42, r91, r11);
    r133 = r43 * r47;
    r11 = fmaf(r72, r133, r11);
    r108 = r7 * r51;
    r11 = fmaf(r126, r108, r11);
    r134 = r75 * r112;
    r134 = r134 * r39;
    r134 = r134 * r69;
    r134 = r134 * r54;
    r11 = fmaf(r42, r134, r11);
    r11 = fmaf(r43, r104, r11);
    r134 = r1 * r11;
    r108 = r38 * r33;
    r108 = r108 * r33;
    r108 = r108 * r47;
    r108 = r108 * r47;
    r108 = r108 * r113;
    r108 = r108 * r39;
    r133 = r8 * r49;
    r133 = r133 * r55;
    r91 = r38 * r55;
    r91 = r91 * r55;
    r91 = fmaf(r84, r91, r53 * r133);
    r133 = r8 * r44;
    r133 = r133 * r33;
    r91 = fmaf(r53, r133, r91);
    r91 = fmaf(r38, r83, r91);
    r83 = r91 * r114;
    r83 = fmaf(r106, r83, r98 * r108);
    r108 = r44 * r47;
    r108 = r108 * r110;
    r108 = r108 * r40;
    r83 = fmaf(r57, r108, r83);
    r40 = r56 * r91;
    r40 = r40 * r103;
    r83 = fmaf(r105, r40, r83);
    r106 = r49 * r28;
    r106 = fmaf(r92, r106, r91 * r95);
    r133 = r26 * r55;
    r133 = r133 * r47;
    r133 = r133 * r91;
    r133 = r133 * r53;
    r133 = r133 * r54;
    r133 = r133 * r81;
    r106 = fmaf(r42, r133, r106);
    r80 = r38 * r55;
    r80 = r80 * r47;
    r80 = r80 * r39;
    r80 = r80 * r42;
    r106 = fmaf(r84, r80, r106);
    r83 = r83 + r106;
    r40 = r26 * r33;
    r40 = r40 * r33;
    r40 = r40 * r47;
    r40 = r40 * r47;
    r40 = r40 * r91;
    r40 = r40 * r53;
    r40 = r40 * r54;
    r40 = fmaf(r81, r40, r38 * r107);
    r107 = r91 * r103;
    r40 = fmaf(r105, r107, r40);
    r40 = fmaf(r44, r96, r40);
    r106 = r106 + r40;
    r60 = fmaf(r60, r106, r7 * r83);
    r83 = r33 * r47;
    r83 = r83 * r66;
    r83 = r83 * r75;
    r83 = r83 * r91;
    r83 = r83 * r39;
    r83 = r83 * r69;
    r60 = fmaf(r54, r83, r60);
    r107 = r6 * r49;
    r60 = fmaf(r96, r107, r60);
    r96 = r5 * r8;
    r96 = r96 * r46;
    r96 = fmaf(r106, r96, r4 * r106);
    r96 = fmaf(r106, r59, r96);
    r96 = fmaf(r106, r58, r96);
    r58 = r96 * r71;
    r60 = fmaf(r57, r58, r60);
    r59 = r44 * r47;
    r60 = fmaf(r71, r59, r60);
    r4 = r26 * r38;
    r4 = r4 * r33;
    r4 = r4 * r47;
    r4 = r4 * r53;
    r60 = fmaf(r70, r4, r60);
    r108 = r6 * r91;
    r60 = fmaf(r119, r108, r60);
    r80 = r33 * r47;
    r80 = r80 * r75;
    r80 = r80 * r91;
    r80 = r80 * r39;
    r80 = r80 * r69;
    r60 = fmaf(r54, r80, r60);
    r133 = r6 * r29;
    r133 = r133 * r91;
    r133 = r133 * r53;
    r133 = r133 * r54;
    r133 = r133 * r81;
    r133 = r133 * r42;
    r60 = fmaf(r57, r133, r60);
    r84 = r26 * r38;
    r84 = r84 * r33;
    r84 = r84 * r47;
    r84 = r84 * r66;
    r84 = r84 * r53;
    r60 = fmaf(r70, r84, r60);
    r93 = r6 * r44;
    r93 = r93 * r28;
    r60 = fmaf(r92, r93, r60);
    r87 = r44 * r47;
    r60 = fmaf(r72, r87, r60);
    r136 = r78 * r91;
    r136 = r136 * r71;
    r60 = fmaf(r105, r136, r60);
    r60 = fmaf(r38, r127, r60);
    r60 = fmaf(r91, r115, r60);
    r136 = r0 * r60;
    r87 = r56 * r91;
    r93 = r49 * r47;
    r93 = r93 * r110;
    r93 = fmaf(r28, r93, r95 * r87);
    r87 = r38 * r55;
    r87 = r87 * r47;
    r87 = r87 * r113;
    r87 = r87 * r39;
    r87 = r87 * r98;
    r93 = fmaf(r42, r87, r93);
    r93 = fmaf(r91, r130, r93);
    r93 = r93 + r40;
    r106 = fmaf(r61, r106, r6 * r93);
    r61 = r55 * r78;
    r61 = r61 * r91;
    r61 = r61 * r71;
    r106 = fmaf(r94, r61, r106);
    r93 = r7 * r38;
    r106 = fmaf(r126, r93, r106);
    r126 = r55 * r91;
    r126 = r126 * r94;
    r106 = fmaf(r89, r126, r106);
    r89 = r66 * r75;
    r89 = r89 * r91;
    r89 = r89 * r39;
    r89 = r89 * r69;
    r89 = r89 * r54;
    r106 = fmaf(r42, r89, r106);
    r94 = r7 * r91;
    r106 = fmaf(r119, r94, r106);
    r119 = r75 * r91;
    r119 = r119 * r39;
    r119 = r119 * r69;
    r119 = r119 * r54;
    r106 = fmaf(r42, r119, r106);
    r69 = r49 * r47;
    r106 = fmaf(r72, r69, r106);
    r72 = r7 * r29;
    r72 = r72 * r91;
    r72 = r72 * r53;
    r72 = r72 * r54;
    r72 = r72 * r81;
    r72 = r72 * r42;
    r106 = fmaf(r57, r72, r106);
    r54 = r26 * r38;
    r54 = r54 * r66;
    r54 = r54 * r53;
    r54 = r54 * r70;
    r106 = fmaf(r42, r54, r106);
    r39 = r96 * r42;
    r106 = fmaf(r71, r39, r106);
    r40 = r49 * r47;
    r106 = fmaf(r71, r40, r106);
    r87 = r26 * r38;
    r87 = r87 * r53;
    r87 = r87 * r70;
    r106 = fmaf(r42, r87, r106);
    r70 = r7 * r44;
    r70 = r70 * r28;
    r106 = fmaf(r92, r70, r106);
    r106 = fmaf(r49, r104, r106);
    r70 = r1 * r106;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r128,
                                          r134,
                                          r136,
                                          r70);
    r70 = r0 * r26;
    r70 = r70 * r2;
    r136 = r26 * r3;
    r134 = r1 * r136;
    r70 = fmaf(r129, r134, r111 * r70);
    r128 = r0 * r26;
    r128 = r128 * r2;
    r128 = fmaf(r135, r134, r27 * r128);
    r87 = r0 * r26;
    r87 = r87 * r2;
    r87 = fmaf(r76, r134, r74 * r87);
    r40 = r0 * r26;
    r40 = r40 * r2;
    r40 = fmaf(r102, r134, r13 * r40);
    WriteSum4<float, float>((float*)inout_shared, r70, r128, r87, r40);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r0 * r26;
    r40 = r40 * r2;
    r40 = fmaf(r11, r134, r120 * r40);
    r87 = r0 * r26;
    r87 = r87 * r2;
    r87 = fmaf(r106, r134, r60 * r87);
    WriteSum2<float, float>((float*)inout_shared, r40, r87);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r87 = r1 * r1;
    r40 = r129 * r129;
    r128 = r111 * r111;
    r70 = r0 * r0;
    r128 = fmaf(r70, r128, r87 * r40);
    r40 = r27 * r27;
    r39 = r135 * r135;
    r39 = fmaf(r87, r39, r70 * r40);
    r40 = r74 * r74;
    r54 = r76 * r76;
    r54 = fmaf(r87, r54, r70 * r40);
    r40 = r102 * r102;
    r72 = r13 * r13;
    r72 = fmaf(r70, r72, r87 * r40);
    WriteSum4<float, float>((float*)inout_shared, r128, r39, r54, r72);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r11 * r11;
    r54 = r120 * r120;
    r54 = fmaf(r70, r54, r87 * r72);
    r72 = r106 * r106;
    r39 = r60 * r60;
    r39 = fmaf(r70, r39, r87 * r72);
    WriteSum2<float, float>((float*)inout_shared, r54, r39);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r111 * r27;
    r54 = r129 * r135;
    r54 = fmaf(r87, r54, r70 * r39);
    r39 = r111 * r74;
    r72 = r129 * r76;
    r72 = fmaf(r87, r72, r70 * r39);
    r39 = r129 * r102;
    r128 = r111 * r13;
    r128 = fmaf(r70, r128, r87 * r39);
    r39 = r111 * r120;
    r40 = r129 * r11;
    r40 = fmaf(r87, r40, r70 * r39);
    WriteSum4<float, float>((float*)inout_shared, r54, r72, r128, r40);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r111 * r60;
    r128 = r129 * r106;
    r128 = fmaf(r87, r128, r70 * r40);
    r40 = r135 * r76;
    r72 = r27 * r74;
    r72 = fmaf(r70, r72, r87 * r40);
    r40 = r135 * r102;
    r54 = r27 * r13;
    r54 = fmaf(r70, r54, r87 * r40);
    r40 = r135 * r11;
    r39 = r27 * r120;
    r39 = fmaf(r70, r39, r87 * r40);
    WriteSum4<float, float>((float*)inout_shared, r128, r72, r54, r39);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r135 * r106;
    r54 = r27 * r60;
    r54 = fmaf(r70, r54, r87 * r39);
    r39 = r76 * r102;
    r72 = r74 * r13;
    r72 = fmaf(r70, r72, r87 * r39);
    r39 = r76 * r11;
    r128 = r74 * r120;
    r128 = fmaf(r70, r128, r87 * r39);
    r39 = r74 * r60;
    r40 = r76 * r106;
    r40 = fmaf(r87, r40, r70 * r39);
    WriteSum4<float, float>((float*)inout_shared, r54, r72, r128, r40);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r13 * r120;
    r128 = r102 * r11;
    r128 = fmaf(r87, r128, r70 * r40);
    r40 = r13 * r60;
    r72 = r102 * r106;
    r72 = fmaf(r87, r72, r70 * r40);
    r40 = r11 * r106;
    r54 = r120 * r60;
    r54 = fmaf(r70, r54, r87 * r40);
    WriteSum3<float, float>((float*)inout_shared, r128, r72, r54);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = r0 * r46;
    r54 = r54 * r71;
    r54 = r54 * r57;
    r72 = r1 * r46;
    r72 = r72 * r42;
    r72 = r72 * r71;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          0 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r63,
                                          r62,
                                          r54,
                                          r72);
    r128 = r1 * r64;
    r40 = r0 * r71;
    r40 = r40 * r57;
    r40 = r40 * r65;
    r39 = r1 * r42;
    r39 = r39 * r71;
    r39 = r39 * r65;
    r69 = r0 * r8;
    r69 = r69 * r57;
    r69 = r69 * r28;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          4 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r40,
                                          r39,
                                          r69,
                                          r128);
    r119 = r0 * r35;
    r94 = r1 * r8;
    r94 = r94 * r57;
    r94 = r94 * r28;
    r89 = r0 * r71;
    r89 = r89 * r57;
    r89 = r89 * r68;
    r126 = r1 * r42;
    r126 = r126 * r71;
    r126 = r126 * r68;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          8 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r119,
                                          r94,
                                          r89,
                                          r126);
    r93 = r0 * r46;
    r61 = r1 * r46;
    r104 = r0 * r71;
    r104 = r104 * r57;
    r104 = r104 * r67;
    r53 = r1 * r42;
    r53 = r53 * r71;
    r53 = r53 * r67;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          12 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r104,
                                          r53,
                                          r93,
                                          r61);
    r130 = r26 * r63;
    r130 = r130 * r2;
    r113 = r26 * r2;
    r95 = r62 * r136;
    WriteSum4<float, float>((float*)inout_shared, r130, r95, r113, r136);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r136 = r46 * r42;
    r136 = r136 * r71;
    r113 = r0 * r26;
    r113 = r113 * r46;
    r113 = r113 * r2;
    r113 = r113 * r71;
    r113 = fmaf(r57, r113, r134 * r136);
    r136 = r42 * r71;
    r136 = r136 * r65;
    r95 = r0 * r26;
    r95 = r95 * r2;
    r95 = r95 * r71;
    r95 = r95 * r57;
    r95 = fmaf(r65, r95, r134 * r136);
    r136 = r0 * r29;
    r136 = r136 * r2;
    r136 = r136 * r57;
    r136 = fmaf(r28, r136, r64 * r134);
    r130 = r0 * r26;
    r130 = r130 * r35;
    r84 = r1 * r29;
    r84 = r84 * r3;
    r84 = r84 * r57;
    r84 = fmaf(r28, r84, r2 * r130);
    WriteSum4<float, float>((float*)inout_shared, r113, r95, r136, r84);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           4 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r84 = r0 * r26;
    r84 = r84 * r46;
    r84 = r84 * r2;
    r136 = r46 * r134;
    r95 = r42 * r71;
    r95 = r95 * r68;
    r113 = r0 * r26;
    r113 = r113 * r2;
    r113 = r113 * r71;
    r113 = r113 * r57;
    r113 = fmaf(r68, r113, r134 * r95);
    r95 = r42 * r71;
    r95 = r95 * r67;
    r130 = r0 * r26;
    r130 = r130 * r2;
    r130 = r130 * r71;
    r130 = r130 * r57;
    r130 = fmaf(r67, r130, r134 * r95);
    WriteSum4<float, float>((float*)inout_shared, r113, r130, r84, r136);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           8 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r136 = r63 * r63;
    r84 = r62 * r62;
    WriteSum4<float, float>((float*)inout_shared, r136, r84, r30, r30);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = r33 * r47;
    r30 = r30 * r65;
    r30 = r30 * r70;
    r84 = r47 * r65;
    r84 = r84 * r87;
    r84 = fmaf(r52, r84, r103 * r30);
    r30 = r33 * r47;
    r30 = r30 * r70;
    r30 = r30 * r103;
    r136 = r47 * r87;
    r136 = r136 * r67;
    r136 = fmaf(r52, r136, r67 * r30);
    r30 = r64 * r64;
    r130 = r42 * r70;
    r113 = r33 * r55;
    r50 = r32 * r50;
    r50 = 1.0 / r50;
    r36 = r41 * r36;
    r36 = 1.0 / r36;
    r113 = r113 * r47;
    r113 = r113 * r47;
    r113 = r113 * r123;
    r113 = r113 * r50;
    r113 = r113 * r36;
    r113 = r113 * r57;
    r130 = fmaf(r113, r130, r87 * r30);
    r30 = r35 * r70;
    r36 = r42 * r87;
    r113 = fmaf(r36, r113, r35 * r30);
    WriteSum4<float, float>((float*)inout_shared, r84, r136, r130, r113);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           4 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r113 = r65 * r70;
    r130 = r65 * r87;
    r84 = r33 * r47;
    r50 = r68 * r68;
    r84 = r84 * r70;
    r84 = r84 * r103;
    r123 = r47 * r87;
    r123 = r123 * r52;
    r84 = fmaf(r50, r123, r50 * r84);
    r41 = r67 * r67;
    r32 = r70 * r103;
    r32 = r32 * r57;
    r41 = fmaf(r123, r41, r41 * r32);
    WriteSum4<float, float>((float*)inout_shared, r84, r41, r113, r130);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           8 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r130 = 0.00000000000000000e+00;
    r113 = r0 * r46;
    r113 = r113 * r63;
    r113 = r113 * r71;
    r113 = r113 * r57;
    WriteSum4<float, float>((float*)inout_shared, r130, r63, r130, r113);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r0 * r35;
    r35 = r35 * r63;
    r113 = r0 * r63;
    r113 = r113 * r71;
    r113 = r113 * r57;
    r113 = r113 * r65;
    r41 = r0 * r8;
    r41 = r41 * r63;
    r41 = r41 * r57;
    r41 = r41 * r28;
    r95 = r0 * r63;
    r95 = r95 * r71;
    r95 = r95 * r57;
    r95 = r95 * r68;
    WriteSum4<float, float>((float*)inout_shared, r113, r41, r35, r95);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = r0 * r46;
    r95 = r95 * r63;
    r63 = r0 * r63;
    r63 = r63 * r71;
    r63 = r63 * r57;
    r63 = r63 * r67;
    WriteSum4<float, float>((float*)inout_shared, r63, r95, r130, r130);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           8 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = r1 * r64;
    r95 = r95 * r62;
    r63 = r1 * r46;
    r63 = r63 * r62;
    r63 = r63 * r42;
    r63 = r63 * r71;
    r35 = r1 * r62;
    r35 = r35 * r42;
    r35 = r35 * r71;
    r35 = r35 * r65;
    WriteSum4<float, float>((float*)inout_shared, r62, r63, r35, r95);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           12 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = r1 * r8;
    r95 = r95 * r62;
    r95 = r95 * r57;
    r95 = r95 * r28;
    r35 = r1 * r62;
    r35 = r35 * r42;
    r35 = r35 * r71;
    r35 = r35 * r68;
    r63 = r1 * r62;
    r63 = r63 * r42;
    r63 = r63 * r71;
    r63 = r63 * r67;
    WriteSum4<float, float>((float*)inout_shared, r95, r35, r63, r130);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           16 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = r1 * r46;
    r63 = r63 * r62;
    WriteSum4<float, float>((float*)inout_shared, r63, r130, r54, r40);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           20 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r69, r119, r89, r104);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           24 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r93, r130, r72, r39);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           28 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r128, r94, r126, r53);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           32 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r33 * r47;
    r53 = r53 * r68;
    r53 = r53 * r70;
    r126 = r47 * r68;
    r126 = r126 * r87;
    r126 = fmaf(r52, r126, r103 * r53);
    r53 = r33 * r46;
    r53 = r53 * r98;
    r53 = r53 * r81;
    r53 = r53 * r42;
    r53 = r53 * r57;
    r53 = r53 * r70;
    r94 = r71 * r36;
    r128 = r64 * r94;
    r53 = fmaf(r46, r128, r92 * r53);
    WriteSum4<float, float>((float*)inout_shared, r130, r61, r126, r53);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           36 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r71 * r57;
    r53 = r53 * r65;
    r53 = r53 * r70;
    r126 = r57 * r30;
    r61 = r71 * r126;
    r39 = r55 * r46;
    r39 = r39 * r98;
    r39 = r39 * r81;
    r39 = r39 * r57;
    r39 = r39 * r92;
    r39 = fmaf(r36, r39, r46 * r61);
    r72 = r33 * r47;
    r93 = r46 * r67;
    r72 = r72 * r70;
    r72 = r72 * r103;
    r104 = r47 * r87;
    r104 = r104 * r52;
    r104 = fmaf(r93, r104, r93 * r72);
    WriteSum4<float, float>((float*)inout_shared, r39, r136, r104, r53);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           40 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r65 * r94;
    r136 = r33 * r98;
    r136 = r136 * r81;
    r136 = r136 * r42;
    r136 = r136 * r57;
    r136 = r136 * r65;
    r136 = r136 * r70;
    r136 = fmaf(r65, r128, r92 * r136);
    r39 = r55 * r98;
    r39 = r39 * r81;
    r39 = r39 * r57;
    r39 = r39 * r65;
    r39 = r39 * r92;
    r39 = fmaf(r36, r39, r65 * r61);
    WriteSum4<float, float>((float*)inout_shared, r53, r136, r39, r104);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           44 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r104 = r71 * r57;
    r104 = r104 * r68;
    r104 = r104 * r70;
    r39 = r68 * r94;
    r136 = r8 * r28;
    r53 = r8 * r64;
    r53 = r53 * r57;
    r53 = r53 * r28;
    r53 = fmaf(r87, r53, r126 * r136);
    WriteSum4<float, float>((float*)inout_shared, r84, r104, r39, r53);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           48 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r8 * r46;
    r53 = r53 * r57;
    r53 = r53 * r28;
    r53 = r53 * r70;
    r39 = r46 * r64;
    r39 = r39 * r87;
    r104 = r33 * r98;
    r104 = r104 * r81;
    r104 = r104 * r42;
    r104 = r104 * r57;
    r104 = r104 * r68;
    r104 = r104 * r70;
    r104 = fmaf(r68, r128, r92 * r104);
    r84 = r33 * r98;
    r84 = r84 * r81;
    r84 = r84 * r42;
    r84 = r84 * r57;
    r84 = r84 * r70;
    r84 = r84 * r92;
    r128 = fmaf(r67, r128, r67 * r84);
    WriteSum4<float, float>((float*)inout_shared, r104, r128, r53, r39);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           52 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = r46 * r30;
    r39 = r8 * r46;
    r39 = r39 * r57;
    r39 = r39 * r28;
    r39 = r39 * r87;
    r53 = r55 * r98;
    r53 = r53 * r81;
    r53 = r53 * r57;
    r53 = r53 * r68;
    r53 = r53 * r92;
    r53 = fmaf(r36, r53, r68 * r61);
    r128 = r55 * r98;
    r128 = r128 * r81;
    r128 = r128 * r57;
    r128 = r128 * r92;
    r128 = r128 * r67;
    r128 = fmaf(r36, r128, r67 * r61);
    WriteSum4<float, float>((float*)inout_shared, r53, r128, r30, r39);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           56 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r71 * r57;
    r39 = r39 * r70;
    r39 = r39 * r67;
    r67 = r67 * r94;
    r57 = r71 * r57;
    r57 = r57 * r70;
    r57 = r57 * r93;
    r50 = r46 * r50;
    r123 = fmaf(r50, r123, r50 * r32);
    WriteSum4<float, float>((float*)inout_shared, r123, r39, r67, r57);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           60 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r94 = r93 * r94;
    WriteSum2<float, float>((float*)inout_shared, r94, r130);
  };
  FlushSumShared<2, float>(out_calib_precond_tril,
                           64 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
}

void ThinPrismFisheyeFixedPointResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
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
    float* out_calib_jac,
    unsigned int out_calib_jac_num_alloc,
    float* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    float* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFixedPointResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
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
      out_calib_jac,
      out_calib_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar