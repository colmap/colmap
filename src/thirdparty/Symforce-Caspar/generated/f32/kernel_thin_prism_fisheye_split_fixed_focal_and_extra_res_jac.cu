#include "kernel_thin_prism_fisheye_split_fixed_focal_and_extra_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedFocalAndExtraResJacKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
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
        float* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        float* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
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

  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
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
      r141, r142;
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
    ReadIdx4<1024, float, float, float4>(focal_and_extra,
                                         0 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r5,
                                         r6,
                                         r7);
    ReadIdx2<1024, float, float, float2>(focal_and_extra,
                                         8 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r8,
                                         r9);
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
    ReadIdx4<1024, float, float, float4>(focal_and_extra,
                                         4 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r67,
                                         r68,
                                         r69,
                                         r70);
    r71 = 3.00000000000000000e+00;
    r72 = r71 * r62;
    r72 = fmaf(r64, r72, r52);
    r72 = fmaf(r68, r72, r8 * r66);
    r52 = r67 * r11;
    r73 = r16 * r62;
    r52 = r52 * r63;
    r72 = fmaf(r73, r52, r72);
    r74 = r66 * r66;
    r75 = r66 * r74;
    r76 = fmaf(r69, r75, r6 * r66);
    r75 = r70 * r75;
    r76 = fmaf(r66, r75, r76);
    r76 = fmaf(r7, r74, r76);
    r70 = 1.0 / r36;
    r77 = 1.0 / r51;
    r78 = r70 * r77;
    r79 = r60 * r78;
    r80 = r76 * r79;
    r72 = fmaf(r10, r80, r72);
    r72 = fmaf(r10, r79, r72);
    r2 = fmaf(r0, r72, r2);
    r72 = r11 * r11;
    r72 = r72 * r71;
    r72 = r72 * r47;
    r72 = fmaf(r62, r72, r65);
    r72 = fmaf(r67, r72, r9 * r66);
    r65 = r68 * r11;
    r65 = r65 * r63;
    r72 = fmaf(r73, r65, r72);
    r72 = fmaf(r11, r80, r72);
    r72 = fmaf(r11, r79, r72);
    r72 = fmaf(r5, r72, r1);
    r72 = fmaf(r3, r4, r72);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r72);
    r3 = r10 * r47;
    r1 = -5.00000000000000000e-01;
    r65 = rsqrtf(r35);
    r52 = r16 * r11;
    r81 = r25 * r41;
    r82 = r17 * r24;
    r83 = r20 * r21;
    r83 = fmaf(r1, r83, r1 * r82);
    r82 = r19 * r22;
    r83 = fmaf(r1, r82, r83);
    r84 = r18 * r23;
    r85 = 5.00000000000000000e-01;
    r83 = fmaf(r85, r84, r83);
    r84 = r20 * r24;
    r82 = r17 * r21;
    r82 = fmaf(r1, r82, r85 * r84);
    r84 = r18 * r22;
    r82 = fmaf(r1, r84, r82);
    r86 = r19 * r23;
    r82 = fmaf(r1, r86, r82);
    r86 = r34 * r82;
    r84 = r41 * r86;
    r81 = fmaf(r83, r81, r84);
    r87 = r16 * r29;
    r88 = fmaf(r85, r31, r1 * r27);
    r88 = fmaf(r1, r28, r88);
    r88 = fmaf(r1, r30, r88);
    r87 = r87 * r88;
    r89 = r16 * r32;
    r90 = r19 * r24;
    r91 = r18 * r21;
    r91 = fmaf(r85, r91, r85 * r90);
    r90 = r17 * r22;
    r91 = fmaf(r1, r90, r91);
    r92 = r20 * r23;
    r91 = fmaf(r85, r92, r91);
    r89 = fmaf(r91, r89, r87);
    r81 = r81 + r89;
    r92 = r25 * r82;
    r90 = -4.00000000000000000e+00;
    r92 = r92 * r90;
    r93 = r32 * r88;
    r94 = r90 * r93;
    r95 = r92 + r94;
    r95 = fmaf(r14, r95, r15 * r81);
    r81 = r16 * r25;
    r81 = r81 * r91;
    r96 = r16 * r34;
    r96 = fmaf(r88, r96, r81);
    r97 = r16 * r29;
    r97 = r97 * r82;
    r98 = r16 * r32;
    r98 = fmaf(r83, r98, r97);
    r96 = r96 + r98;
    r95 = fmaf(r13, r96, r95);
    r52 = r52 * r95;
    r96 = r16 * r10;
    r99 = r16 * r34;
    r100 = r29 * r83;
    r99 = fmaf(r16, r100, r91 * r99);
    r101 = r16 * r25;
    r102 = r16 * r32;
    r102 = r102 * r82;
    r101 = fmaf(r88, r101, r102);
    r99 = r99 + r101;
    r81 = r97 + r81;
    r97 = r32 * r41;
    r81 = fmaf(r83, r97, r81);
    r103 = r41 * r34;
    r81 = fmaf(r88, r103, r81);
    r81 = fmaf(r14, r81, r15 * r99);
    r99 = r29 * r91;
    r99 = r99 * r90;
    r94 = r99 + r94;
    r81 = fmaf(r13, r94, r81);
    r96 = r96 * r81;
    r96 = fmaf(r37, r96, r37 * r52);
    r52 = r16 * r25;
    r52 = r52 * r83;
    r86 = r16 * r86;
    r94 = r52 + r86;
    r89 = r89 + r94;
    r103 = r41 * r34;
    r103 = fmaf(r41, r100, r91 * r103);
    r103 = r103 + r101;
    r103 = fmaf(r13, r103, r14 * r89);
    r99 = r92 + r99;
    r103 = fmaf(r15, r99, r103);
    r99 = r11 * r11;
    r12 = r36 * r12;
    r12 = 1.0 / r12;
    r36 = r41 * r12;
    r99 = r99 * r36;
    r92 = r10 * r10;
    r92 = r92 * r103;
    r96 = fmaf(r36, r92, r96);
    r96 = fmaf(r103, r99, r96);
    r3 = r3 * r60;
    r3 = r3 * r70;
    r3 = r3 * r1;
    r3 = r3 * r65;
    r3 = r3 * r96;
    r92 = 6.00000000000000000e+00;
    r89 = r81 * r92;
    r89 = r89 * r62;
    r91 = r71 * r96;
    r35 = r42 + r35;
    r35 = 1.0 / r35;
    r97 = r35 * r57;
    r91 = r91 * r65;
    r91 = r91 * r97;
    r91 = fmaf(r64, r91, r63 * r89);
    r89 = -6.00000000000000000e+00;
    r104 = r89 * r12;
    r105 = r60 * r60;
    r106 = r64 * r105;
    r104 = r104 * r106;
    r107 = r10 * r10;
    r108 = -3.00000000000000000e+00;
    r53 = r51 * r53;
    r53 = 1.0 / r53;
    r107 = r107 * r108;
    r107 = r107 * r96;
    r107 = r107 * r65;
    r107 = r107 * r53;
    r91 = fmaf(r62, r107, r91);
    r51 = r11 * r47;
    r109 = r73 * r51;
    r110 = r11 * r65;
    r111 = r96 * r110;
    r111 = r111 * r97;
    r111 = fmaf(r51, r111, r95 * r109);
    r112 = r4 * r11;
    r113 = r53 * r62;
    r113 = r113 * r110;
    r112 = r112 * r96;
    r111 = fmaf(r113, r112, r111);
    r105 = r47 * r105;
    r105 = r105 * r99;
    r111 = fmaf(r103, r105, r111);
    r91 = fmaf(r103, r104, r91);
    r91 = r91 + r111;
    r91 = fmaf(r68, r91, r3);
    r107 = r81 * r63;
    r112 = r96 * r65;
    r112 = r112 * r97;
    r112 = fmaf(r64, r112, r73 * r107);
    r107 = r103 * r36;
    r112 = fmaf(r106, r107, r112);
    r114 = r4 * r10;
    r114 = r114 * r10;
    r114 = r114 * r96;
    r114 = r114 * r65;
    r114 = r114 * r53;
    r112 = fmaf(r62, r114, r112);
    r111 = r111 + r112;
    r114 = r67 * r96;
    r107 = r16 * r63;
    r107 = r107 * r110;
    r107 = r107 * r97;
    r91 = fmaf(r107, r114, r91);
    r115 = r67 * r11;
    r115 = r115 * r60;
    r115 = r115 * r60;
    r115 = r115 * r90;
    r115 = r115 * r12;
    r115 = r115 * r63;
    r116 = r10 * r85;
    r116 = r116 * r65;
    r116 = r116 * r35;
    r116 = r116 * r78;
    r117 = r76 * r116;
    r118 = r67 * r81;
    r91 = fmaf(r109, r118, r91);
    r119 = r4 * r10;
    r119 = r119 * r103;
    r119 = r119 * r77;
    r91 = fmaf(r57, r119, r91);
    r120 = r4 * r10;
    r120 = r120 * r76;
    r120 = r120 * r103;
    r120 = r120 * r77;
    r91 = fmaf(r57, r120, r91);
    r121 = r67 * r41;
    r121 = r121 * r10;
    r121 = r121 * r96;
    r91 = fmaf(r113, r121, r91);
    r122 = r7 * r16;
    r122 = r122 * r66;
    r122 = fmaf(r111, r122, r6 * r111);
    r123 = 4.00000000000000000e+00;
    r75 = r123 * r75;
    r69 = r69 * r71;
    r69 = r69 * r74;
    r122 = fmaf(r111, r75, r122);
    r122 = fmaf(r111, r69, r122);
    r74 = r10 * r122;
    r91 = fmaf(r79, r74, r91);
    r123 = r67 * r95;
    r123 = r123 * r63;
    r91 = fmaf(r73, r123, r91);
    r91 = fmaf(r8, r111, r91);
    r91 = fmaf(r76, r3, r91);
    r91 = fmaf(r103, r115, r91);
    r91 = fmaf(r96, r117, r91);
    r91 = fmaf(r81, r79, r91);
    r91 = fmaf(r96, r116, r91);
    r91 = fmaf(r81, r80, r91);
    r123 = r0 * r91;
    r74 = r11 * r95;
    r74 = r74 * r92;
    r74 = r74 * r47;
    r121 = r71 * r96;
    r121 = r121 * r110;
    r121 = r121 * r97;
    r121 = fmaf(r51, r121, r62 * r74);
    r74 = r11 * r108;
    r74 = r74 * r96;
    r121 = fmaf(r113, r74, r121);
    r120 = r11 * r11;
    r120 = r120 * r60;
    r120 = r120 * r60;
    r120 = r120 * r103;
    r120 = r120 * r89;
    r120 = r120 * r47;
    r121 = fmaf(r12, r120, r121);
    r121 = r121 + r112;
    r111 = fmaf(r9, r111, r67 * r121);
    r121 = r11 * r47;
    r121 = r121 * r60;
    r121 = r121 * r70;
    r121 = r121 * r1;
    r121 = r121 * r65;
    r121 = r121 * r96;
    r112 = r85 * r96;
    r112 = r112 * r35;
    r112 = r112 * r110;
    r111 = fmaf(r78, r112, r111);
    r120 = r11 * r122;
    r111 = fmaf(r79, r120, r111);
    r74 = r4 * r11;
    r74 = r74 * r76;
    r74 = r74 * r103;
    r74 = r74 * r77;
    r111 = fmaf(r57, r74, r111);
    r119 = r68 * r11;
    r119 = r119 * r60;
    r119 = r119 * r60;
    r119 = r119 * r90;
    r119 = r119 * r103;
    r119 = r119 * r12;
    r111 = fmaf(r63, r119, r111);
    r118 = r68 * r81;
    r111 = fmaf(r109, r118, r111);
    r114 = r68 * r96;
    r111 = fmaf(r107, r114, r111);
    r3 = r4 * r11;
    r3 = r3 * r103;
    r3 = r3 * r77;
    r111 = fmaf(r57, r3, r111);
    r124 = r68 * r41;
    r124 = r124 * r10;
    r124 = r124 * r113;
    r125 = r85 * r76;
    r125 = r125 * r96;
    r125 = r125 * r35;
    r125 = r125 * r110;
    r111 = fmaf(r78, r125, r111);
    r126 = r68 * r95;
    r126 = r126 * r63;
    r111 = fmaf(r73, r126, r111);
    r111 = r111 + r121;
    r111 = fmaf(r76, r121, r111);
    r111 = fmaf(r96, r124, r111);
    r111 = fmaf(r95, r80, r111);
    r111 = fmaf(r95, r79, r111);
    r126 = r5 * r111;
    r125 = r16 * r10;
    r86 = r87 + r86;
    r87 = r16 * r32;
    r3 = r19 * r24;
    r114 = r18 * r21;
    r114 = fmaf(r1, r114, r1 * r3);
    r3 = r17 * r22;
    r114 = fmaf(r85, r3, r114);
    r118 = r20 * r23;
    r114 = fmaf(r1, r118, r114);
    r87 = r87 * r114;
    r118 = r16 * r25;
    r3 = r17 * r24;
    r119 = r20 * r21;
    r119 = fmaf(r85, r119, r85 * r3);
    r3 = r19 * r22;
    r119 = fmaf(r85, r3, r119);
    r74 = r18 * r23;
    r119 = fmaf(r1, r74, r119);
    r118 = fmaf(r119, r118, r87);
    r86 = r86 + r118;
    r74 = r32 * r90;
    r74 = r74 * r119;
    r3 = r29 * r82;
    r3 = r3 * r90;
    r120 = r74 + r3;
    r120 = fmaf(r13, r120, r15 * r86);
    r86 = r41 * r34;
    r86 = fmaf(r41, r93, r119 * r86);
    r121 = r16 * r25;
    r121 = r121 * r82;
    r112 = r16 * r29;
    r112 = fmaf(r114, r112, r121);
    r86 = r86 + r112;
    r120 = fmaf(r14, r86, r120);
    r125 = r125 * r120;
    r86 = r10 * r10;
    r127 = r41 * r29;
    r127 = fmaf(r88, r127, r84);
    r127 = r127 + r118;
    r118 = r16 * r29;
    r118 = r118 * r119;
    r128 = r16 * r34;
    r128 = fmaf(r114, r128, r118);
    r128 = r128 + r101;
    r128 = fmaf(r14, r128, r13 * r127);
    r127 = r25 * r114;
    r101 = r90 * r127;
    r3 = r3 + r101;
    r128 = fmaf(r15, r3, r128);
    r86 = r86 * r128;
    r86 = fmaf(r36, r86, r37 * r125);
    r125 = r16 * r11;
    r3 = r25 * r41;
    r3 = fmaf(r88, r3, r102);
    r102 = r41 * r34;
    r3 = fmaf(r114, r102, r3);
    r3 = r3 + r118;
    r102 = r16 * r34;
    r93 = fmaf(r16, r93, r119 * r102);
    r93 = r93 + r112;
    r93 = fmaf(r13, r93, r15 * r3);
    r101 = r74 + r101;
    r93 = fmaf(r14, r101, r93);
    r125 = r125 * r93;
    r86 = fmaf(r37, r125, r86);
    r86 = fmaf(r128, r99, r86);
    r125 = r86 * r65;
    r125 = r125 * r97;
    r101 = r4 * r10;
    r101 = r101 * r10;
    r101 = r101 * r86;
    r101 = r101 * r65;
    r101 = r101 * r53;
    r101 = fmaf(r62, r101, r64 * r125);
    r125 = r128 * r36;
    r101 = fmaf(r106, r125, r101);
    r74 = r120 * r63;
    r101 = fmaf(r73, r74, r101);
    r74 = r4 * r11;
    r74 = r74 * r86;
    r74 = fmaf(r113, r74, r93 * r109);
    r125 = r86 * r110;
    r125 = r125 * r97;
    r74 = fmaf(r51, r125, r74);
    r74 = fmaf(r128, r105, r74);
    r125 = r101 + r74;
    r3 = r71 * r86;
    r3 = r3 * r65;
    r3 = r3 * r97;
    r102 = r10 * r10;
    r119 = r108 * r86;
    r102 = r102 * r65;
    r102 = r102 * r53;
    r102 = r102 * r62;
    r102 = fmaf(r119, r102, r64 * r3);
    r3 = r92 * r120;
    r3 = r3 * r62;
    r102 = fmaf(r63, r3, r102);
    r102 = fmaf(r128, r104, r102);
    r102 = r102 + r74;
    r102 = fmaf(r68, r102, r8 * r125);
    r74 = r67 * r93;
    r74 = r74 * r63;
    r102 = fmaf(r73, r74, r102);
    r3 = r60 * r1;
    r3 = r3 * r86;
    r3 = r3 * r70;
    r3 = r3 * r65;
    r102 = fmaf(r63, r3, r102);
    r118 = r4 * r10;
    r118 = r118 * r128;
    r118 = r118 * r77;
    r102 = fmaf(r57, r118, r102);
    r88 = r67 * r41;
    r88 = r88 * r10;
    r88 = r88 * r86;
    r102 = fmaf(r113, r88, r102);
    r129 = r67 * r86;
    r102 = fmaf(r107, r129, r102);
    r130 = r4 * r10;
    r130 = r130 * r76;
    r130 = r130 * r128;
    r130 = r130 * r77;
    r102 = fmaf(r57, r130, r102);
    r131 = r67 * r120;
    r102 = fmaf(r109, r131, r102);
    r132 = r7 * r16;
    r132 = r132 * r66;
    r132 = fmaf(r125, r132, r6 * r125);
    r132 = fmaf(r125, r75, r132);
    r132 = fmaf(r125, r69, r132);
    r133 = r10 * r132;
    r102 = fmaf(r79, r133, r102);
    r134 = r60 * r76;
    r134 = r134 * r1;
    r134 = r134 * r86;
    r134 = r134 * r70;
    r134 = r134 * r65;
    r102 = fmaf(r63, r134, r102);
    r102 = fmaf(r128, r115, r102);
    r102 = fmaf(r86, r117, r102);
    r102 = fmaf(r120, r80, r102);
    r102 = fmaf(r86, r116, r102);
    r102 = fmaf(r120, r79, r102);
    r134 = r0 * r102;
    r133 = r11 * r92;
    r133 = r133 * r93;
    r133 = r133 * r47;
    r131 = r11 * r113;
    r131 = fmaf(r119, r131, r62 * r133);
    r133 = r11 * r11;
    r133 = r133 * r60;
    r133 = r133 * r60;
    r133 = r133 * r89;
    r133 = r133 * r128;
    r133 = r133 * r47;
    r131 = fmaf(r12, r133, r131);
    r119 = r71 * r86;
    r119 = r119 * r110;
    r119 = r119 * r97;
    r131 = fmaf(r51, r119, r131);
    r131 = r131 + r101;
    r131 = fmaf(r67, r131, r9 * r125);
    r125 = r4 * r11;
    r125 = r125 * r128;
    r125 = r125 * r77;
    r131 = fmaf(r57, r125, r131);
    r101 = r68 * r93;
    r101 = r101 * r63;
    r131 = fmaf(r73, r101, r131);
    r119 = r68 * r11;
    r119 = r119 * r60;
    r119 = r119 * r60;
    r119 = r119 * r90;
    r119 = r119 * r128;
    r119 = r119 * r12;
    r131 = fmaf(r63, r119, r131);
    r133 = r85 * r76;
    r133 = r133 * r86;
    r133 = r133 * r35;
    r133 = r133 * r110;
    r131 = fmaf(r78, r133, r131);
    r130 = r60 * r76;
    r130 = r130 * r1;
    r130 = r130 * r86;
    r130 = r130 * r47;
    r130 = r130 * r70;
    r131 = fmaf(r110, r130, r131);
    r129 = r11 * r132;
    r131 = fmaf(r79, r129, r131);
    r88 = r85 * r86;
    r88 = r88 * r35;
    r88 = r88 * r110;
    r131 = fmaf(r78, r88, r131);
    r118 = r60 * r1;
    r118 = r118 * r86;
    r118 = r118 * r47;
    r118 = r118 * r70;
    r131 = fmaf(r110, r118, r131);
    r3 = r68 * r86;
    r131 = fmaf(r107, r3, r131);
    r74 = r68 * r120;
    r131 = fmaf(r109, r74, r131);
    r135 = r4 * r11;
    r135 = r135 * r76;
    r135 = r135 * r128;
    r135 = r135 * r77;
    r131 = fmaf(r57, r135, r131);
    r131 = fmaf(r86, r124, r131);
    r131 = fmaf(r93, r80, r131);
    r131 = fmaf(r93, r79, r131);
    r135 = r5 * r131;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r123,
                                          r126,
                                          r134,
                                          r135);
    r135 = r10 * r10;
    r134 = r25 * r90;
    r31 = fmaf(r1, r31, r85 * r27);
    r31 = fmaf(r85, r28, r31);
    r31 = fmaf(r85, r30, r31);
    r134 = r134 * r31;
    r100 = r90 * r100;
    r30 = r134 + r100;
    r28 = r16 * r32;
    r28 = r28 * r31;
    r121 = r121 + r28;
    r27 = r41 * r29;
    r121 = fmaf(r114, r27, r121);
    r126 = r41 * r34;
    r121 = fmaf(r83, r126, r121);
    r121 = fmaf(r13, r121, r15 * r30);
    r30 = r16 * r34;
    r30 = fmaf(r16, r127, r31 * r30);
    r30 = r30 + r98;
    r121 = fmaf(r14, r30, r121);
    r135 = r135 * r121;
    r30 = r16 * r10;
    r126 = r16 * r29;
    r126 = r126 * r31;
    r52 = r52 + r126;
    r27 = r32 * r41;
    r52 = fmaf(r114, r27, r52);
    r52 = r52 + r84;
    r82 = r32 * r82;
    r82 = r82 * r90;
    r100 = r82 + r100;
    r100 = fmaf(r13, r100, r14 * r52);
    r52 = r16 * r34;
    r52 = fmaf(r83, r52, r28);
    r52 = r52 + r112;
    r100 = fmaf(r15, r52, r100);
    r30 = r30 * r100;
    r30 = fmaf(r37, r30, r36 * r135);
    r135 = r16 * r11;
    r126 = r87 + r126;
    r126 = r126 + r94;
    r94 = r41 * r34;
    r127 = fmaf(r41, r127, r31 * r94);
    r127 = r127 + r98;
    r127 = fmaf(r15, r127, r13 * r126);
    r82 = r134 + r82;
    r127 = fmaf(r14, r82, r127);
    r135 = r135 * r127;
    r30 = fmaf(r37, r135, r30);
    r30 = fmaf(r121, r99, r30);
    r135 = r30 * r110;
    r135 = r135 * r97;
    r135 = fmaf(r127, r109, r51 * r135);
    r82 = r4 * r11;
    r82 = r82 * r30;
    r135 = fmaf(r113, r82, r135);
    r135 = fmaf(r121, r105, r135);
    r82 = r121 * r36;
    r14 = r30 * r65;
    r14 = r14 * r97;
    r14 = fmaf(r64, r14, r106 * r82);
    r82 = r4 * r10;
    r82 = r82 * r10;
    r82 = r82 * r30;
    r82 = r82 * r65;
    r82 = r82 * r53;
    r14 = fmaf(r62, r82, r14);
    r134 = r100 * r63;
    r14 = fmaf(r73, r134, r14);
    r134 = r135 + r14;
    r82 = r71 * r30;
    r82 = r82 * r65;
    r82 = r82 * r97;
    r82 = fmaf(r64, r82, r121 * r104);
    r15 = r10 * r10;
    r15 = r15 * r108;
    r15 = r15 * r30;
    r15 = r15 * r65;
    r15 = r15 * r53;
    r82 = fmaf(r62, r15, r82);
    r126 = r92 * r100;
    r126 = r126 * r62;
    r82 = fmaf(r63, r126, r82);
    r82 = r82 + r135;
    r82 = fmaf(r68, r82, r8 * r134);
    r135 = r7 * r16;
    r135 = r135 * r66;
    r135 = fmaf(r134, r135, r6 * r134);
    r135 = fmaf(r134, r69, r135);
    r135 = fmaf(r134, r75, r135);
    r126 = r10 * r135;
    r82 = fmaf(r79, r126, r82);
    r15 = r60 * r76;
    r15 = r15 * r1;
    r15 = r15 * r30;
    r15 = r15 * r70;
    r15 = r15 * r65;
    r82 = fmaf(r63, r15, r82);
    r13 = r4 * r10;
    r13 = r13 * r121;
    r13 = r13 * r77;
    r82 = fmaf(r57, r13, r82);
    r98 = r67 * r41;
    r98 = r98 * r10;
    r98 = r98 * r30;
    r82 = fmaf(r113, r98, r82);
    r94 = r67 * r100;
    r82 = fmaf(r109, r94, r82);
    r31 = r67 * r127;
    r31 = r31 * r63;
    r82 = fmaf(r73, r31, r82);
    r87 = r60 * r1;
    r87 = r87 * r30;
    r87 = r87 * r70;
    r87 = r87 * r65;
    r82 = fmaf(r63, r87, r82);
    r52 = r4 * r10;
    r52 = r52 * r76;
    r52 = r52 * r121;
    r52 = r52 * r77;
    r82 = fmaf(r57, r52, r82);
    r112 = r67 * r30;
    r82 = fmaf(r107, r112, r82);
    r82 = fmaf(r121, r115, r82);
    r82 = fmaf(r30, r116, r82);
    r82 = fmaf(r100, r80, r82);
    r82 = fmaf(r30, r117, r82);
    r82 = fmaf(r100, r79, r82);
    r112 = r0 * r82;
    r52 = r71 * r30;
    r52 = r52 * r110;
    r52 = r52 * r97;
    r87 = r11 * r92;
    r87 = r87 * r127;
    r87 = r87 * r47;
    r87 = fmaf(r62, r87, r51 * r52);
    r52 = r11 * r11;
    r52 = r52 * r60;
    r52 = r52 * r60;
    r52 = r52 * r89;
    r52 = r52 * r121;
    r52 = r52 * r47;
    r87 = fmaf(r12, r52, r87);
    r31 = r11 * r108;
    r31 = r31 * r30;
    r87 = fmaf(r113, r31, r87);
    r87 = r87 + r14;
    r87 = fmaf(r67, r87, r9 * r134);
    r134 = r68 * r11;
    r134 = r134 * r60;
    r134 = r134 * r60;
    r134 = r134 * r90;
    r134 = r134 * r121;
    r134 = r134 * r12;
    r87 = fmaf(r63, r134, r87);
    r14 = r60 * r1;
    r14 = r14 * r30;
    r14 = r14 * r47;
    r14 = r14 * r70;
    r87 = fmaf(r110, r14, r87);
    r31 = r85 * r30;
    r31 = r31 * r35;
    r31 = r31 * r110;
    r87 = fmaf(r78, r31, r87);
    r52 = r4 * r11;
    r52 = r52 * r76;
    r52 = r52 * r121;
    r52 = r52 * r77;
    r87 = fmaf(r57, r52, r87);
    r94 = r68 * r100;
    r87 = fmaf(r109, r94, r87);
    r98 = r60 * r76;
    r98 = r98 * r1;
    r98 = r98 * r30;
    r98 = r98 * r47;
    r98 = r98 * r70;
    r87 = fmaf(r110, r98, r87);
    r13 = r68 * r127;
    r13 = r13 * r63;
    r87 = fmaf(r73, r13, r87);
    r15 = r4 * r11;
    r15 = r15 * r121;
    r15 = r15 * r77;
    r87 = fmaf(r57, r15, r87);
    r126 = r68 * r30;
    r87 = fmaf(r107, r126, r87);
    r28 = r11 * r135;
    r87 = fmaf(r79, r28, r87);
    r83 = r85 * r76;
    r83 = r83 * r30;
    r83 = r83 * r35;
    r83 = r83 * r110;
    r87 = fmaf(r78, r83, r87);
    r87 = fmaf(r127, r80, r87);
    r87 = fmaf(r30, r124, r87);
    r87 = fmaf(r127, r79, r87);
    r83 = r5 * r87;
    r28 = r43 * r92;
    r28 = r28 * r62;
    r28 = fmaf(r63, r28, r26 * r104);
    r126 = r10 * r10;
    r15 = r16 * r39;
    r15 = r15 * r11;
    r15 = fmaf(r37, r15, r26 * r99);
    r13 = r26 * r10;
    r13 = r13 * r10;
    r15 = fmaf(r36, r13, r15);
    r98 = r16 * r43;
    r98 = r98 * r10;
    r15 = fmaf(r37, r98, r15);
    r126 = r126 * r108;
    r126 = r126 * r15;
    r126 = r126 * r65;
    r126 = r126 * r53;
    r28 = fmaf(r62, r126, r28);
    r98 = r71 * r15;
    r98 = r98 * r65;
    r98 = r98 * r97;
    r28 = fmaf(r64, r98, r28);
    r13 = fmaf(r39, r109, r26 * r105);
    r94 = r4 * r11;
    r94 = r94 * r15;
    r13 = fmaf(r113, r94, r13);
    r52 = r15 * r110;
    r52 = r52 * r97;
    r13 = fmaf(r51, r52, r13);
    r28 = r28 + r13;
    r98 = r26 * r36;
    r126 = r43 * r63;
    r126 = fmaf(r73, r126, r106 * r98);
    r98 = r4 * r10;
    r98 = r98 * r10;
    r98 = r98 * r15;
    r98 = r98 * r65;
    r98 = r98 * r53;
    r126 = fmaf(r62, r98, r126);
    r52 = r15 * r65;
    r52 = r52 * r97;
    r126 = fmaf(r64, r52, r126);
    r13 = r13 + r126;
    r28 = fmaf(r8, r13, r68 * r28);
    r52 = r4 * r26;
    r52 = r52 * r10;
    r52 = r52 * r77;
    r28 = fmaf(r57, r52, r28);
    r98 = r60 * r76;
    r98 = r98 * r1;
    r98 = r98 * r15;
    r98 = r98 * r70;
    r98 = r98 * r65;
    r28 = fmaf(r63, r98, r28);
    r94 = r67 * r41;
    r94 = r94 * r10;
    r94 = r94 * r15;
    r28 = fmaf(r113, r94, r28);
    r31 = r67 * r15;
    r28 = fmaf(r107, r31, r28);
    r14 = r43 * r109;
    r134 = r60 * r1;
    r134 = r134 * r15;
    r134 = r134 * r70;
    r134 = r134 * r65;
    r28 = fmaf(r63, r134, r28);
    r84 = r67 * r39;
    r84 = r84 * r63;
    r28 = fmaf(r73, r84, r28);
    r27 = r7 * r16;
    r27 = r27 * r66;
    r27 = fmaf(r6, r13, r13 * r27);
    r27 = fmaf(r13, r75, r27);
    r27 = fmaf(r13, r69, r27);
    r114 = r10 * r27;
    r28 = fmaf(r79, r114, r28);
    r123 = r4 * r26;
    r123 = r123 * r10;
    r123 = r123 * r76;
    r123 = r123 * r77;
    r28 = fmaf(r57, r123, r28);
    r28 = fmaf(r43, r79, r28);
    r28 = fmaf(r43, r80, r28);
    r28 = fmaf(r67, r14, r28);
    r28 = fmaf(r15, r117, r28);
    r28 = fmaf(r26, r115, r28);
    r28 = fmaf(r15, r116, r28);
    r123 = r0 * r28;
    r114 = r26 * r11;
    r114 = r114 * r11;
    r114 = r114 * r60;
    r114 = r114 * r60;
    r114 = r114 * r89;
    r114 = r114 * r47;
    r84 = r39 * r11;
    r84 = r84 * r92;
    r84 = r84 * r47;
    r84 = fmaf(r62, r84, r12 * r114);
    r114 = r11 * r108;
    r114 = r114 * r15;
    r84 = fmaf(r113, r114, r84);
    r134 = r71 * r15;
    r134 = r134 * r110;
    r134 = r134 * r97;
    r84 = fmaf(r51, r134, r84);
    r84 = r84 + r126;
    r13 = fmaf(r9, r13, r67 * r84);
    r84 = r4 * r26;
    r84 = r84 * r11;
    r84 = r84 * r77;
    r13 = fmaf(r57, r84, r13);
    r126 = r4 * r26;
    r126 = r126 * r11;
    r126 = r126 * r76;
    r126 = r126 * r77;
    r13 = fmaf(r57, r126, r13);
    r134 = r60 * r1;
    r134 = r134 * r15;
    r134 = r134 * r47;
    r134 = r134 * r70;
    r13 = fmaf(r110, r134, r13);
    r114 = r60 * r76;
    r114 = r114 * r1;
    r114 = r114 * r15;
    r114 = r114 * r47;
    r114 = r114 * r70;
    r13 = fmaf(r110, r114, r13);
    r31 = r68 * r15;
    r13 = fmaf(r107, r31, r13);
    r94 = r85 * r15;
    r94 = r94 * r35;
    r94 = r94 * r110;
    r13 = fmaf(r78, r94, r13);
    r98 = r68 * r26;
    r98 = r98 * r11;
    r98 = r98 * r60;
    r98 = r98 * r60;
    r98 = r98 * r90;
    r98 = r98 * r12;
    r13 = fmaf(r63, r98, r13);
    r52 = r68 * r39;
    r52 = r52 * r63;
    r13 = fmaf(r73, r52, r13);
    r74 = r85 * r76;
    r74 = r74 * r15;
    r74 = r74 * r35;
    r74 = r74 * r110;
    r13 = fmaf(r78, r74, r13);
    r3 = r11 * r27;
    r13 = fmaf(r79, r3, r13);
    r13 = fmaf(r39, r79, r13);
    r13 = fmaf(r39, r80, r13);
    r13 = fmaf(r15, r124, r13);
    r13 = fmaf(r68, r14, r13);
    r3 = r5 * r13;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r112,
                                          r83,
                                          r123,
                                          r3);
    r3 = r16 * r44;
    r3 = r3 * r11;
    r3 = fmaf(r37, r3, r40 * r99);
    r123 = r40 * r10;
    r123 = r123 * r10;
    r3 = fmaf(r36, r123, r3);
    r83 = r16 * r59;
    r83 = r83 * r10;
    r3 = fmaf(r37, r83, r3);
    r83 = r3 * r110;
    r83 = r83 * r97;
    r83 = fmaf(r40, r105, r51 * r83);
    r123 = r4 * r11;
    r123 = r123 * r3;
    r83 = fmaf(r113, r123, r83);
    r83 = fmaf(r44, r109, r83);
    r123 = r3 * r65;
    r123 = r123 * r97;
    r112 = r40 * r36;
    r112 = fmaf(r106, r112, r64 * r123);
    r123 = r59 * r63;
    r112 = fmaf(r73, r123, r112);
    r74 = r4 * r10;
    r74 = r74 * r10;
    r74 = r74 * r3;
    r74 = r74 * r65;
    r74 = r74 * r53;
    r112 = fmaf(r62, r74, r112);
    r74 = r83 + r112;
    r123 = r71 * r3;
    r123 = r123 * r65;
    r123 = r123 * r97;
    r123 = fmaf(r40, r104, r64 * r123);
    r52 = r59 * r92;
    r52 = r52 * r62;
    r123 = fmaf(r63, r52, r123);
    r98 = r10 * r10;
    r98 = r98 * r108;
    r98 = r98 * r3;
    r98 = r98 * r65;
    r98 = r98 * r53;
    r123 = fmaf(r62, r98, r123);
    r123 = r123 + r83;
    r123 = fmaf(r68, r123, r8 * r74);
    r83 = r7 * r16;
    r83 = r83 * r66;
    r83 = fmaf(r74, r83, r6 * r74);
    r83 = fmaf(r74, r75, r83);
    r83 = fmaf(r74, r69, r83);
    r98 = r10 * r83;
    r123 = fmaf(r79, r98, r123);
    r52 = r67 * r44;
    r52 = r52 * r63;
    r123 = fmaf(r73, r52, r123);
    r94 = r60 * r1;
    r94 = r94 * r3;
    r94 = r94 * r70;
    r94 = r94 * r65;
    r123 = fmaf(r63, r94, r123);
    r14 = r67 * r3;
    r123 = fmaf(r107, r14, r123);
    r31 = r67 * r41;
    r31 = r31 * r10;
    r31 = r31 * r3;
    r123 = fmaf(r113, r31, r123);
    r114 = r60 * r76;
    r114 = r114 * r1;
    r114 = r114 * r3;
    r114 = r114 * r70;
    r114 = r114 * r65;
    r123 = fmaf(r63, r114, r123);
    r134 = r4 * r40;
    r134 = r134 * r10;
    r134 = r134 * r77;
    r123 = fmaf(r57, r134, r123);
    r126 = r67 * r59;
    r123 = fmaf(r109, r126, r123);
    r84 = r4 * r40;
    r84 = r84 * r10;
    r84 = r84 * r76;
    r84 = r84 * r77;
    r123 = fmaf(r57, r84, r123);
    r123 = fmaf(r59, r80, r123);
    r123 = fmaf(r59, r79, r123);
    r123 = fmaf(r3, r116, r123);
    r123 = fmaf(r40, r115, r123);
    r123 = fmaf(r3, r117, r123);
    r84 = r0 * r123;
    r126 = r71 * r3;
    r126 = r126 * r110;
    r126 = r126 * r97;
    r134 = r40 * r11;
    r134 = r134 * r11;
    r134 = r134 * r60;
    r134 = r134 * r60;
    r134 = r134 * r89;
    r134 = r134 * r47;
    r134 = fmaf(r12, r134, r51 * r126);
    r126 = r44 * r11;
    r126 = r126 * r92;
    r126 = r126 * r47;
    r134 = fmaf(r62, r126, r134);
    r114 = r11 * r108;
    r114 = r114 * r3;
    r134 = fmaf(r113, r114, r134);
    r134 = r134 + r112;
    r134 = fmaf(r67, r134, r9 * r74);
    r74 = r68 * r44;
    r74 = r74 * r63;
    r134 = fmaf(r73, r74, r134);
    r112 = r4 * r40;
    r112 = r112 * r11;
    r112 = r112 * r77;
    r134 = fmaf(r57, r112, r134);
    r114 = r85 * r3;
    r114 = r114 * r35;
    r114 = r114 * r110;
    r134 = fmaf(r78, r114, r134);
    r126 = r68 * r3;
    r134 = fmaf(r107, r126, r134);
    r31 = r85 * r76;
    r31 = r31 * r3;
    r31 = r31 * r35;
    r31 = r31 * r110;
    r134 = fmaf(r78, r31, r134);
    r14 = r68 * r40;
    r14 = r14 * r11;
    r14 = r14 * r60;
    r14 = r14 * r60;
    r14 = r14 * r90;
    r14 = r14 * r12;
    r134 = fmaf(r63, r14, r134);
    r94 = r68 * r59;
    r134 = fmaf(r109, r94, r134);
    r52 = r4 * r40;
    r52 = r52 * r11;
    r52 = r52 * r76;
    r52 = r52 * r77;
    r134 = fmaf(r57, r52, r134);
    r98 = r60 * r76;
    r98 = r98 * r1;
    r98 = r98 * r3;
    r98 = r98 * r47;
    r98 = r98 * r70;
    r134 = fmaf(r110, r98, r134);
    r118 = r11 * r83;
    r134 = fmaf(r79, r118, r134);
    r88 = r60 * r1;
    r88 = r88 * r3;
    r88 = r88 * r47;
    r88 = r88 * r70;
    r134 = fmaf(r110, r88, r134);
    r134 = fmaf(r44, r80, r134);
    r134 = fmaf(r3, r124, r134);
    r134 = fmaf(r44, r79, r134);
    r88 = r5 * r134;
    r118 = r16 * r46;
    r118 = r118 * r11;
    r118 = fmaf(r38, r99, r37 * r118);
    r98 = r16 * r58;
    r98 = r98 * r10;
    r118 = fmaf(r37, r98, r118);
    r52 = r38 * r10;
    r52 = r52 * r10;
    r118 = fmaf(r36, r52, r118);
    r52 = r118 * r110;
    r52 = r52 * r97;
    r52 = fmaf(r46, r109, r51 * r52);
    r98 = r4 * r11;
    r98 = r98 * r118;
    r52 = fmaf(r113, r98, r52);
    r52 = fmaf(r38, r105, r52);
    r98 = r38 * r36;
    r94 = r4 * r10;
    r94 = r94 * r10;
    r94 = r94 * r118;
    r94 = r94 * r65;
    r94 = r94 * r53;
    r94 = fmaf(r62, r94, r106 * r98);
    r98 = r58 * r63;
    r94 = fmaf(r73, r98, r94);
    r14 = r118 * r65;
    r14 = r14 * r97;
    r94 = fmaf(r64, r14, r94);
    r14 = r52 + r94;
    r98 = r10 * r10;
    r98 = r98 * r108;
    r98 = r98 * r118;
    r98 = r98 * r65;
    r98 = r98 * r53;
    r98 = fmaf(r62, r98, r38 * r104);
    r31 = r58 * r92;
    r31 = r31 * r62;
    r98 = fmaf(r63, r31, r98);
    r126 = r71 * r118;
    r126 = r126 * r65;
    r126 = r126 * r97;
    r98 = fmaf(r64, r126, r98);
    r98 = r98 + r52;
    r98 = fmaf(r68, r98, r8 * r14);
    r52 = r4 * r38;
    r52 = r52 * r10;
    r52 = r52 * r76;
    r52 = r52 * r77;
    r98 = fmaf(r57, r52, r98);
    r126 = r4 * r38;
    r126 = r126 * r10;
    r126 = r126 * r77;
    r98 = fmaf(r57, r126, r98);
    r31 = r60 * r1;
    r31 = r31 * r118;
    r31 = r31 * r70;
    r31 = r31 * r65;
    r98 = fmaf(r63, r31, r98);
    r114 = r67 * r41;
    r114 = r114 * r10;
    r114 = r114 * r118;
    r98 = fmaf(r113, r114, r98);
    r112 = r67 * r118;
    r98 = fmaf(r107, r112, r98);
    r74 = r67 * r46;
    r74 = r74 * r63;
    r98 = fmaf(r73, r74, r98);
    r129 = r67 * r58;
    r98 = fmaf(r109, r129, r98);
    r130 = r60 * r76;
    r130 = r130 * r1;
    r130 = r130 * r118;
    r130 = r130 * r70;
    r130 = r130 * r65;
    r98 = fmaf(r63, r130, r98);
    r133 = r7 * r16;
    r133 = r133 * r66;
    r133 = fmaf(r6, r14, r14 * r133);
    r133 = fmaf(r14, r75, r133);
    r133 = fmaf(r14, r69, r133);
    r119 = r10 * r133;
    r98 = fmaf(r79, r119, r98);
    r98 = fmaf(r58, r79, r98);
    r98 = fmaf(r58, r80, r98);
    r98 = fmaf(r118, r117, r98);
    r98 = fmaf(r118, r116, r98);
    r98 = fmaf(r38, r115, r98);
    r119 = r0 * r98;
    r130 = r71 * r118;
    r130 = r130 * r110;
    r130 = r130 * r97;
    r129 = r46 * r11;
    r129 = r129 * r92;
    r129 = r129 * r47;
    r129 = fmaf(r62, r129, r51 * r130);
    r130 = r11 * r108;
    r130 = r130 * r118;
    r129 = fmaf(r113, r130, r129);
    r74 = r38 * r11;
    r74 = r74 * r11;
    r74 = r74 * r60;
    r74 = r74 * r60;
    r74 = r74 * r89;
    r74 = r74 * r47;
    r129 = fmaf(r12, r74, r129);
    r129 = r129 + r94;
    r129 = fmaf(r67, r129, r9 * r14);
    r14 = r85 * r76;
    r14 = r14 * r118;
    r14 = r14 * r35;
    r14 = r14 * r110;
    r129 = fmaf(r78, r14, r129);
    r94 = r85 * r118;
    r94 = r94 * r35;
    r94 = r94 * r110;
    r129 = fmaf(r78, r94, r129);
    r74 = r60 * r76;
    r74 = r74 * r1;
    r74 = r74 * r118;
    r74 = r74 * r47;
    r74 = r74 * r70;
    r129 = fmaf(r110, r74, r129);
    r130 = r60 * r1;
    r130 = r130 * r118;
    r130 = r130 * r47;
    r130 = r130 * r70;
    r129 = fmaf(r110, r130, r129);
    r112 = r68 * r118;
    r129 = fmaf(r107, r112, r129);
    r114 = r68 * r46;
    r114 = r114 * r63;
    r129 = fmaf(r73, r114, r129);
    r31 = r4 * r38;
    r31 = r31 * r11;
    r31 = r31 * r76;
    r31 = r31 * r77;
    r129 = fmaf(r57, r31, r129);
    r126 = r68 * r58;
    r129 = fmaf(r109, r126, r129);
    r52 = r4 * r38;
    r52 = r52 * r11;
    r52 = r52 * r77;
    r129 = fmaf(r57, r52, r129);
    r101 = r11 * r133;
    r129 = fmaf(r79, r101, r129);
    r125 = r68 * r38;
    r125 = r125 * r11;
    r125 = r125 * r60;
    r125 = r125 * r60;
    r125 = r125 * r90;
    r125 = r125 * r12;
    r129 = fmaf(r63, r125, r129);
    r129 = fmaf(r46, r80, r129);
    r129 = fmaf(r118, r124, r129);
    r129 = fmaf(r46, r79, r129);
    r125 = r5 * r129;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r84,
                                          r88,
                                          r119,
                                          r125);
    r125 = r0 * r4;
    r125 = r125 * r2;
    r72 = r4 * r72;
    r119 = r5 * r72;
    r125 = fmaf(r111, r119, r91 * r125);
    r88 = r0 * r4;
    r88 = r88 * r2;
    r88 = fmaf(r131, r119, r102 * r88);
    r84 = r0 * r4;
    r84 = r84 * r2;
    r84 = fmaf(r87, r119, r82 * r84);
    r101 = r0 * r4;
    r101 = r101 * r2;
    r101 = fmaf(r13, r119, r28 * r101);
    WriteSum4<float, float>((float*)inout_shared, r125, r88, r84, r101);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r101 = r0 * r4;
    r101 = r101 * r2;
    r101 = fmaf(r134, r119, r123 * r101);
    r84 = r0 * r4;
    r84 = r84 * r2;
    r84 = fmaf(r129, r119, r98 * r84);
    WriteSum2<float, float>((float*)inout_shared, r101, r84);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r84 = r0 * r0;
    r101 = r91 * r84;
    r88 = r5 * r5;
    r125 = r111 * r88;
    r111 = fmaf(r111, r125, r91 * r101);
    r91 = r102 * r102;
    r52 = r131 * r131;
    r52 = fmaf(r88, r52, r84 * r91);
    r91 = r87 * r87;
    r126 = r82 * r82;
    r126 = fmaf(r84, r126, r88 * r91);
    r91 = r28 * r28;
    r31 = r13 * r13;
    r31 = fmaf(r88, r31, r84 * r91);
    WriteSum4<float, float>((float*)inout_shared, r111, r52, r126, r31);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r123 * r123;
    r126 = r134 * r134;
    r126 = fmaf(r88, r126, r84 * r31);
    r31 = r129 * r129;
    r52 = r98 * r98;
    r52 = fmaf(r84, r52, r88 * r31);
    WriteSum2<float, float>((float*)inout_shared, r126, r52);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fmaf(r102, r101, r131 * r125);
    r126 = fmaf(r87, r125, r82 * r101);
    r31 = fmaf(r28, r101, r13 * r125);
    r111 = fmaf(r123, r101, r134 * r125);
    WriteSum4<float, float>((float*)inout_shared, r52, r126, r31, r111);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r125 = fmaf(r129, r125, r98 * r101);
    r101 = r131 * r87;
    r111 = r102 * r82;
    r111 = fmaf(r84, r111, r88 * r101);
    r101 = r102 * r28;
    r31 = r131 * r13;
    r31 = fmaf(r88, r31, r84 * r101);
    r101 = r102 * r123;
    r126 = r131 * r134;
    r126 = fmaf(r88, r126, r84 * r101);
    WriteSum4<float, float>((float*)inout_shared, r125, r111, r31, r126);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r131 * r129;
    r31 = r102 * r98;
    r31 = fmaf(r84, r31, r88 * r126);
    r126 = r87 * r13;
    r111 = r82 * r28;
    r111 = fmaf(r84, r111, r88 * r126);
    r126 = r82 * r123;
    r125 = r87 * r134;
    r125 = fmaf(r88, r125, r84 * r126);
    r126 = r87 * r129;
    r101 = r82 * r98;
    r101 = fmaf(r84, r101, r88 * r126);
    WriteSum4<float, float>((float*)inout_shared, r31, r111, r125, r101);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r101 = r13 * r134;
    r125 = r28 * r123;
    r125 = fmaf(r84, r125, r88 * r101);
    r101 = r28 * r98;
    r111 = r13 * r129;
    r111 = fmaf(r88, r111, r84 * r101);
    r101 = r134 * r129;
    r31 = r123 * r98;
    r31 = fmaf(r84, r31, r88 * r101);
    WriteSum3<float, float>((float*)inout_shared, r125, r111, r31);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r4 * r2;
    WriteSum2<float, float>((float*)inout_shared, r31, r72);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float*)inout_shared, r42, r42);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r55 * r92;
    r42 = r42 * r62;
    r42 = fmaf(r61, r104, r63 * r42);
    r72 = r16 * r33;
    r72 = r72 * r11;
    r31 = r16 * r55;
    r31 = r31 * r10;
    r31 = fmaf(r37, r31, r37 * r72);
    r72 = r61 * r10;
    r72 = r72 * r10;
    r31 = fmaf(r36, r72, r31);
    r31 = fmaf(r61, r99, r31);
    r72 = r71 * r31;
    r72 = r72 * r65;
    r72 = r72 * r97;
    r42 = fmaf(r64, r72, r42);
    r111 = r10 * r10;
    r111 = r111 * r108;
    r111 = r111 * r31;
    r111 = r111 * r65;
    r111 = r111 * r53;
    r42 = fmaf(r62, r111, r42);
    r125 = r4 * r11;
    r125 = r125 * r31;
    r101 = r31 * r110;
    r101 = r101 * r97;
    r101 = fmaf(r51, r101, r113 * r125);
    r101 = fmaf(r61, r105, r101);
    r101 = fmaf(r33, r109, r101);
    r42 = r42 + r101;
    r111 = r55 * r63;
    r72 = r61 * r36;
    r72 = fmaf(r106, r72, r73 * r111);
    r111 = r31 * r65;
    r111 = r111 * r97;
    r72 = fmaf(r64, r111, r72);
    r125 = r4 * r10;
    r125 = r125 * r10;
    r125 = r125 * r31;
    r125 = r125 * r65;
    r125 = r125 * r53;
    r72 = fmaf(r62, r125, r72);
    r101 = r101 + r72;
    r42 = fmaf(r8, r101, r68 * r42);
    r125 = r60 * r76;
    r125 = r125 * r1;
    r125 = r125 * r31;
    r125 = r125 * r70;
    r125 = r125 * r65;
    r42 = fmaf(r63, r125, r42);
    r111 = r4 * r61;
    r111 = r111 * r10;
    r111 = r111 * r76;
    r111 = r111 * r77;
    r42 = fmaf(r57, r111, r42);
    r126 = r4 * r61;
    r126 = r126 * r10;
    r126 = r126 * r77;
    r42 = fmaf(r57, r126, r42);
    r52 = r67 * r33;
    r52 = r52 * r63;
    r42 = fmaf(r73, r52, r42);
    r91 = r7 * r16;
    r91 = r91 * r66;
    r91 = fmaf(r6, r101, r101 * r91);
    r91 = fmaf(r101, r75, r91);
    r91 = fmaf(r101, r69, r91);
    r114 = r10 * r91;
    r42 = fmaf(r79, r114, r42);
    r112 = r60 * r1;
    r112 = r112 * r31;
    r112 = r112 * r70;
    r112 = r112 * r65;
    r42 = fmaf(r63, r112, r42);
    r130 = r67 * r41;
    r130 = r130 * r10;
    r130 = r130 * r31;
    r42 = fmaf(r113, r130, r42);
    r74 = r67 * r55;
    r42 = fmaf(r109, r74, r42);
    r94 = r31 * r107;
    r42 = fmaf(r55, r80, r42);
    r42 = fmaf(r31, r117, r42);
    r42 = fmaf(r55, r79, r42);
    r42 = fmaf(r61, r115, r42);
    r42 = fmaf(r67, r94, r42);
    r42 = fmaf(r31, r116, r42);
    r74 = r0 * r42;
    r130 = r11 * r108;
    r130 = r130 * r31;
    r112 = r71 * r31;
    r112 = r112 * r110;
    r112 = r112 * r97;
    r112 = fmaf(r51, r112, r113 * r130);
    r130 = r61 * r11;
    r130 = r130 * r11;
    r130 = r130 * r60;
    r130 = r130 * r60;
    r130 = r130 * r89;
    r130 = r130 * r47;
    r112 = fmaf(r12, r130, r112);
    r114 = r33 * r11;
    r114 = r114 * r92;
    r114 = r114 * r47;
    r112 = fmaf(r62, r114, r112);
    r112 = r112 + r72;
    r101 = fmaf(r9, r101, r67 * r112);
    r112 = r85 * r76;
    r112 = r112 * r31;
    r112 = r112 * r35;
    r112 = r112 * r110;
    r101 = fmaf(r78, r112, r101);
    r72 = r60 * r76;
    r72 = r72 * r1;
    r72 = r72 * r31;
    r72 = r72 * r47;
    r72 = r72 * r70;
    r101 = fmaf(r110, r72, r101);
    r114 = r4 * r61;
    r114 = r114 * r11;
    r114 = r114 * r76;
    r114 = r114 * r77;
    r101 = fmaf(r57, r114, r101);
    r130 = r68 * r33;
    r130 = r130 * r63;
    r101 = fmaf(r73, r130, r101);
    r52 = r68 * r61;
    r52 = r52 * r11;
    r52 = r52 * r60;
    r52 = r52 * r60;
    r52 = r52 * r90;
    r52 = r52 * r12;
    r101 = fmaf(r63, r52, r101);
    r126 = r60 * r1;
    r126 = r126 * r31;
    r126 = r126 * r47;
    r126 = r126 * r70;
    r101 = fmaf(r110, r126, r101);
    r111 = r68 * r55;
    r101 = fmaf(r109, r111, r101);
    r125 = r85 * r31;
    r125 = r125 * r35;
    r125 = r125 * r110;
    r101 = fmaf(r78, r125, r101);
    r14 = r4 * r61;
    r14 = r14 * r11;
    r14 = r14 * r77;
    r101 = fmaf(r57, r14, r101);
    r136 = r11 * r91;
    r101 = fmaf(r79, r136, r101);
    r101 = fmaf(r31, r124, r101);
    r101 = fmaf(r33, r80, r101);
    r101 = fmaf(r68, r94, r101);
    r101 = fmaf(r33, r79, r101);
    r136 = r5 * r101;
    r94 = r16 * r49;
    r94 = r94 * r10;
    r94 = fmaf(r37, r94, r45 * r99);
    r14 = r16 * r50;
    r14 = r14 * r11;
    r94 = fmaf(r37, r14, r94);
    r125 = r45 * r10;
    r125 = r125 * r10;
    r94 = fmaf(r36, r125, r94);
    r125 = r94 * r65;
    r125 = r125 * r97;
    r14 = r45 * r36;
    r14 = fmaf(r106, r14, r64 * r125);
    r125 = r4 * r10;
    r125 = r125 * r10;
    r125 = r125 * r94;
    r125 = r125 * r65;
    r125 = r125 * r53;
    r14 = fmaf(r62, r125, r14);
    r111 = r49 * r63;
    r14 = fmaf(r73, r111, r14);
    r111 = r4 * r11;
    r111 = r111 * r94;
    r111 = fmaf(r50, r109, r113 * r111);
    r125 = r94 * r110;
    r125 = r125 * r97;
    r111 = fmaf(r51, r125, r111);
    r111 = fmaf(r45, r105, r111);
    r125 = r14 + r111;
    r126 = r71 * r94;
    r126 = r126 * r65;
    r126 = r126 * r97;
    r126 = fmaf(r45, r104, r64 * r126);
    r52 = r10 * r10;
    r52 = r52 * r108;
    r52 = r52 * r94;
    r52 = r52 * r65;
    r52 = r52 * r53;
    r126 = fmaf(r62, r52, r126);
    r130 = r49 * r92;
    r130 = r130 * r62;
    r126 = fmaf(r63, r130, r126);
    r126 = r126 + r111;
    r126 = fmaf(r68, r126, r8 * r125);
    r111 = r67 * r49;
    r126 = fmaf(r109, r111, r126);
    r130 = r67 * r41;
    r130 = r130 * r10;
    r130 = r130 * r94;
    r126 = fmaf(r113, r130, r126);
    r52 = r60 * r1;
    r52 = r52 * r94;
    r52 = r52 * r70;
    r52 = r52 * r65;
    r126 = fmaf(r63, r52, r126);
    r114 = r4 * r45;
    r114 = r114 * r10;
    r114 = r114 * r77;
    r126 = fmaf(r57, r114, r126);
    r72 = r4 * r45;
    r72 = r72 * r10;
    r72 = r72 * r76;
    r72 = r72 * r77;
    r126 = fmaf(r57, r72, r126);
    r112 = r7 * r16;
    r112 = r112 * r66;
    r112 = fmaf(r125, r112, r6 * r125);
    r112 = fmaf(r125, r75, r112);
    r112 = fmaf(r125, r69, r112);
    r137 = r10 * r112;
    r126 = fmaf(r79, r137, r126);
    r138 = r60 * r76;
    r138 = r138 * r1;
    r138 = r138 * r94;
    r138 = r138 * r70;
    r138 = r138 * r65;
    r126 = fmaf(r63, r138, r126);
    r139 = r67 * r50;
    r139 = r139 * r63;
    r126 = fmaf(r73, r139, r126);
    r140 = r67 * r94;
    r126 = fmaf(r107, r140, r126);
    r126 = fmaf(r94, r116, r126);
    r126 = fmaf(r94, r117, r126);
    r126 = fmaf(r49, r80, r126);
    r126 = fmaf(r49, r79, r126);
    r126 = fmaf(r45, r115, r126);
    r140 = r0 * r126;
    r139 = r11 * r108;
    r139 = r139 * r94;
    r138 = r50 * r11;
    r138 = r138 * r92;
    r138 = r138 * r47;
    r138 = fmaf(r62, r138, r113 * r139);
    r139 = r71 * r94;
    r139 = r139 * r110;
    r139 = r139 * r97;
    r138 = fmaf(r51, r139, r138);
    r137 = r45 * r11;
    r137 = r137 * r11;
    r137 = r137 * r60;
    r137 = r137 * r60;
    r137 = r137 * r89;
    r137 = r137 * r47;
    r138 = fmaf(r12, r137, r138);
    r138 = r138 + r14;
    r138 = fmaf(r67, r138, r9 * r125);
    r125 = r68 * r49;
    r138 = fmaf(r109, r125, r138);
    r14 = r68 * r94;
    r138 = fmaf(r107, r14, r138);
    r137 = r60 * r76;
    r137 = r137 * r1;
    r137 = r137 * r94;
    r137 = r137 * r47;
    r137 = r137 * r70;
    r138 = fmaf(r110, r137, r138);
    r139 = r60 * r1;
    r139 = r139 * r94;
    r139 = r139 * r47;
    r139 = r139 * r70;
    r138 = fmaf(r110, r139, r138);
    r72 = r11 * r112;
    r138 = fmaf(r79, r72, r138);
    r114 = r4 * r45;
    r114 = r114 * r11;
    r114 = r114 * r76;
    r114 = r114 * r77;
    r138 = fmaf(r57, r114, r138);
    r52 = r85 * r94;
    r52 = r52 * r35;
    r52 = r52 * r110;
    r138 = fmaf(r78, r52, r138);
    r130 = r68 * r50;
    r130 = r130 * r63;
    r138 = fmaf(r73, r130, r138);
    r111 = r85 * r76;
    r111 = r111 * r94;
    r111 = r111 * r35;
    r111 = r111 * r110;
    r138 = fmaf(r78, r111, r138);
    r141 = r4 * r45;
    r141 = r141 * r11;
    r141 = r141 * r77;
    r138 = fmaf(r57, r141, r138);
    r142 = r68 * r45;
    r142 = r142 * r11;
    r142 = r142 * r60;
    r142 = r142 * r60;
    r142 = r142 * r90;
    r142 = r142 * r12;
    r138 = fmaf(r63, r142, r138);
    r138 = fmaf(r94, r124, r138);
    r138 = fmaf(r50, r79, r138);
    r138 = fmaf(r50, r80, r138);
    r142 = r5 * r138;
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r74,
                                          r136,
                                          r140,
                                          r142);
    r142 = r4 * r11;
    r140 = r16 * r48;
    r140 = r140 * r11;
    r99 = fmaf(r54, r99, r37 * r140);
    r140 = r54 * r10;
    r140 = r140 * r10;
    r99 = fmaf(r36, r140, r99);
    r136 = r16 * r56;
    r136 = r136 * r10;
    r99 = fmaf(r37, r136, r99);
    r142 = r142 * r99;
    r142 = fmaf(r48, r109, r113 * r142);
    r136 = r99 * r110;
    r136 = r136 * r97;
    r142 = fmaf(r51, r136, r142);
    r142 = fmaf(r54, r105, r142);
    r105 = r56 * r63;
    r136 = r99 * r65;
    r136 = r136 * r97;
    r136 = fmaf(r64, r136, r73 * r105);
    r105 = r54 * r36;
    r136 = fmaf(r106, r105, r136);
    r106 = r4 * r10;
    r106 = r106 * r10;
    r106 = r106 * r99;
    r106 = r106 * r65;
    r106 = r106 * r53;
    r136 = fmaf(r62, r106, r136);
    r106 = r142 + r136;
    r105 = r56 * r92;
    r105 = r105 * r62;
    r140 = r71 * r99;
    r140 = r140 * r65;
    r140 = r140 * r97;
    r140 = fmaf(r64, r140, r63 * r105);
    r105 = r10 * r10;
    r105 = r105 * r108;
    r105 = r105 * r99;
    r105 = r105 * r65;
    r105 = r105 * r53;
    r140 = fmaf(r62, r105, r140);
    r140 = fmaf(r54, r104, r140);
    r140 = r140 + r142;
    r140 = fmaf(r68, r140, r8 * r106);
    r8 = r67 * r48;
    r8 = r8 * r63;
    r140 = fmaf(r73, r8, r140);
    r142 = r67 * r99;
    r140 = fmaf(r107, r142, r140);
    r105 = r60 * r76;
    r105 = r105 * r1;
    r105 = r105 * r99;
    r105 = r105 * r70;
    r105 = r105 * r65;
    r140 = fmaf(r63, r105, r140);
    r104 = r4 * r54;
    r104 = r104 * r10;
    r104 = r104 * r76;
    r104 = r104 * r77;
    r140 = fmaf(r57, r104, r140);
    r53 = r7 * r16;
    r53 = r53 * r66;
    r53 = fmaf(r106, r53, r6 * r106);
    r53 = fmaf(r106, r75, r53);
    r53 = fmaf(r106, r69, r53);
    r69 = r10 * r53;
    r140 = fmaf(r79, r69, r140);
    r75 = r4 * r54;
    r75 = r75 * r10;
    r75 = r75 * r77;
    r140 = fmaf(r57, r75, r140);
    r6 = r60 * r1;
    r6 = r6 * r99;
    r6 = r6 * r70;
    r6 = r6 * r65;
    r140 = fmaf(r63, r6, r140);
    r66 = r67 * r41;
    r66 = r66 * r10;
    r66 = r66 * r99;
    r140 = fmaf(r113, r66, r140);
    r64 = r67 * r56;
    r140 = fmaf(r109, r64, r140);
    r140 = fmaf(r99, r117, r140);
    r140 = fmaf(r56, r79, r140);
    r140 = fmaf(r56, r80, r140);
    r140 = fmaf(r99, r116, r140);
    r140 = fmaf(r54, r115, r140);
    r64 = r0 * r140;
    r115 = r11 * r108;
    r115 = r115 * r99;
    r116 = r48 * r11;
    r116 = r116 * r92;
    r116 = r116 * r47;
    r116 = fmaf(r62, r116, r113 * r115);
    r115 = r71 * r99;
    r115 = r115 * r110;
    r115 = r115 * r97;
    r116 = fmaf(r51, r115, r116);
    r51 = r54 * r11;
    r51 = r51 * r11;
    r51 = r51 * r60;
    r51 = r51 * r60;
    r51 = r51 * r89;
    r51 = r51 * r47;
    r116 = fmaf(r12, r51, r116);
    r116 = r116 + r136;
    r116 = fmaf(r67, r116, r9 * r106);
    r106 = r68 * r48;
    r106 = r106 * r63;
    r116 = fmaf(r73, r106, r116);
    r73 = r4 * r54;
    r73 = r73 * r11;
    r73 = r73 * r76;
    r73 = r73 * r77;
    r116 = fmaf(r57, r73, r116);
    r9 = r60 * r1;
    r9 = r9 * r99;
    r9 = r9 * r47;
    r9 = r9 * r70;
    r116 = fmaf(r110, r9, r116);
    r136 = r68 * r99;
    r116 = fmaf(r107, r136, r116);
    r107 = r11 * r53;
    r116 = fmaf(r79, r107, r116);
    r51 = r68 * r54;
    r51 = r51 * r11;
    r51 = r51 * r60;
    r51 = r51 * r60;
    r51 = r51 * r90;
    r51 = r51 * r12;
    r116 = fmaf(r63, r51, r116);
    r12 = r60 * r76;
    r12 = r12 * r1;
    r12 = r12 * r99;
    r12 = r12 * r47;
    r12 = r12 * r70;
    r116 = fmaf(r110, r12, r116);
    r70 = r85 * r76;
    r70 = r70 * r99;
    r70 = r70 * r35;
    r70 = r70 * r110;
    r116 = fmaf(r78, r70, r116);
    r47 = r68 * r56;
    r116 = fmaf(r109, r47, r116);
    r109 = r4 * r54;
    r109 = r109 * r11;
    r109 = r109 * r77;
    r116 = fmaf(r57, r109, r116);
    r57 = r85 * r99;
    r57 = r57 * r35;
    r57 = r57 * r110;
    r116 = fmaf(r78, r57, r116);
    r116 = fmaf(r48, r79, r116);
    r116 = fmaf(r99, r124, r116);
    r116 = fmaf(r48, r80, r116);
    r5 = r5 * r116;
    WriteIdx2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r64, r5);
    r5 = r0 * r4;
    r5 = r5 * r2;
    r5 = fmaf(r101, r119, r42 * r5);
    r64 = r0 * r4;
    r64 = r64 * r2;
    r64 = fmaf(r138, r119, r126 * r64);
    r57 = r0 * r4;
    r57 = r57 * r2;
    r119 = fmaf(r116, r119, r140 * r57);
    WriteSum3<float, float>((float*)inout_shared, r5, r64, r119);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r119 = r42 * r42;
    r64 = r101 * r101;
    r64 = fmaf(r88, r64, r84 * r119);
    r119 = r138 * r138;
    r5 = r126 * r126;
    r5 = fmaf(r84, r5, r88 * r119);
    r119 = r116 * r116;
    r57 = r140 * r140;
    r57 = fmaf(r84, r57, r88 * r119);
    WriteSum3<float, float>((float*)inout_shared, r64, r5, r57);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = r101 * r138;
    r5 = r42 * r126;
    r5 = fmaf(r84, r5, r88 * r57);
    r57 = r42 * r140;
    r64 = r101 * r116;
    r64 = fmaf(r88, r64, r84 * r57);
    r57 = r138 * r116;
    r119 = r126 * r140;
    r119 = fmaf(r84, r119, r88 * r57);
    WriteSum3<float, float>((float*)inout_shared, r5, r64, r119);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void ThinPrismFisheyeSplitFixedFocalAndExtraResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
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
    float* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
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
  ThinPrismFisheyeSplitFixedFocalAndExtraResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
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
      out_principal_point_jac,
      out_principal_point_jac_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
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