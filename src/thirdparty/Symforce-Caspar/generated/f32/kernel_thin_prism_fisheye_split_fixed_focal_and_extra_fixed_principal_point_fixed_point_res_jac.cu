#include "kernel_thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104,
      r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116,
      r117, r118, r119, r120, r121, r122;

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
    r49 = r33 * r33;
    r47 = r40 * r29;
    r47 = r47 * r29;
    r51 = r41 + r47;
    r51 = r51 + r48;
    r51 = fmaf(r13, r51, r10);
    r10 = r32 * r40;
    r10 = fmaf(r34, r10, r26);
    r26 = r16 * r32;
    r26 = r26 * r25;
    r48 = r16 * r29;
    r48 = fmaf(r34, r48, r26);
    r52 = r21 * r23;
    r52 = r52 * r16;
    r53 = r22 * r24;
    r53 = fmaf(r16, r53, r52);
    r54 = r23 * r24;
    r54 = fmaf(r40, r54, r37);
    r37 = r22 * r22;
    r37 = r37 * r40;
    r42 = r37 + r42;
    r51 = fmaf(r14, r10, r51);
    r51 = fmaf(r15, r48, r51);
    r51 = fmaf(r36, r53, r51);
    r51 = fmaf(r35, r54, r51);
    r51 = fmaf(r11, r42, r51);
    r48 = r51 * r51;
    r10 = 9.99999999999999955e-07;
    r55 = r40 * r29;
    r55 = fmaf(r34, r55, r26);
    r55 = fmaf(r13, r55, r12);
    r12 = r22 * r24;
    r12 = fmaf(r40, r12, r52);
    r37 = r41 + r37;
    r37 = r37 + r39;
    r39 = r21 * r24;
    r39 = fmaf(r16, r39, r44);
    r44 = r16 * r25;
    r44 = fmaf(r34, r44, r46);
    r47 = r41 + r47;
    r47 = r47 + r50;
    r55 = fmaf(r11, r12, r55);
    r55 = fmaf(r36, r37, r55);
    r55 = fmaf(r35, r39, r55);
    r55 = fmaf(r14, r44, r55);
    r55 = fmaf(r15, r47, r55);
    r47 = copysign(1.0, r55);
    r47 = fmaf(r10, r47, r55);
    r55 = r47 * r47;
    r44 = 1.0 / r55;
    r35 = r33 * r33;
    r35 = fmaf(r44, r35, r44 * r48);
    r48 = sqrtf(r35);
    r36 = atanf(r48);
    r11 = copysign(1.0, r48);
    r11 = fmaf(r10, r11, r48);
    r10 = r11 * r11;
    r48 = 1.0 / r10;
    r50 = r36 * r44;
    r46 = r48 * r50;
    r49 = r49 * r36;
    r49 = r49 * r46;
    r52 = r51 * r46;
    r26 = r51 * r36;
    r52 = r52 * r26;
    r56 = r49 + r52;
    ReadIdx4<1024, float, float, float4>(focal_and_extra,
                                         4 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r57,
                                         r58,
                                         r59,
                                         r60);
    r61 = 3.00000000000000000e+00;
    r62 = r51 * r61;
    r62 = r62 * r46;
    r62 = fmaf(r26, r62, r49);
    r62 = fmaf(r58, r62, r8 * r56);
    r49 = r16 * r46;
    r63 = r26 * r49;
    r64 = r57 * r63;
    r65 = r56 * r56;
    r66 = r56 * r65;
    r67 = fmaf(r59, r66, r6 * r56);
    r66 = r60 * r66;
    r67 = fmaf(r56, r66, r67);
    r67 = fmaf(r7, r65, r67);
    r60 = 1.0 / r47;
    r68 = 1.0 / r11;
    r69 = r60 * r68;
    r70 = r67 * r69;
    r62 = fmaf(r33, r64, r62);
    r62 = fmaf(r26, r70, r62);
    r62 = fmaf(r69, r26, r62);
    r2 = fmaf(r0, r62, r2);
    r62 = r33 * r33;
    r62 = r62 * r36;
    r62 = r62 * r61;
    r62 = fmaf(r46, r62, r52);
    r62 = fmaf(r57, r62, r9 * r56);
    r52 = r36 * r70;
    r71 = r58 * r33;
    r62 = fmaf(r63, r71, r62);
    r72 = r33 * r36;
    r62 = fmaf(r69, r72, r62);
    r62 = fmaf(r33, r52, r62);
    r62 = fmaf(r5, r62, r1);
    r62 = fmaf(r3, r4, r62);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r62);
    r3 = r0 * r4;
    r1 = r16 * r34;
    r72 = r19 * r24;
    r71 = 5.00000000000000000e-01;
    r73 = r18 * r21;
    r73 = fmaf(r71, r73, r71 * r72);
    r72 = r17 * r22;
    r74 = -5.00000000000000000e-01;
    r73 = fmaf(r74, r72, r73);
    r75 = r20 * r23;
    r73 = fmaf(r71, r75, r73);
    r75 = r17 * r24;
    r72 = r20 * r21;
    r72 = fmaf(r74, r72, r74 * r75);
    r75 = r19 * r22;
    r72 = fmaf(r74, r75, r72);
    r76 = r18 * r23;
    r72 = fmaf(r71, r76, r72);
    r76 = r29 * r72;
    r1 = fmaf(r16, r76, r73 * r1);
    r75 = r16 * r25;
    r77 = fmaf(r71, r31, r74 * r27);
    r77 = fmaf(r74, r28, r77);
    r77 = fmaf(r74, r30, r77);
    r78 = r16 * r32;
    r79 = r20 * r24;
    r80 = r17 * r21;
    r80 = fmaf(r74, r80, r71 * r79);
    r79 = r18 * r22;
    r80 = fmaf(r74, r79, r80);
    r81 = r19 * r23;
    r80 = fmaf(r74, r81, r80);
    r78 = r78 * r80;
    r75 = fmaf(r77, r75, r78);
    r1 = r1 + r75;
    r81 = r16 * r29;
    r81 = r81 * r80;
    r79 = r16 * r25;
    r79 = r79 * r73;
    r82 = r81 + r79;
    r83 = r32 * r40;
    r82 = fmaf(r72, r83, r82);
    r84 = r40 * r34;
    r82 = fmaf(r77, r84, r82);
    r82 = fmaf(r14, r82, r15 * r1);
    r1 = r29 * r73;
    r84 = -4.00000000000000000e+00;
    r1 = r1 * r84;
    r83 = r32 * r77;
    r85 = r84 * r83;
    r86 = r1 + r85;
    r82 = fmaf(r13, r86, r82);
    r86 = 6.00000000000000000e+00;
    r87 = r82 * r86;
    r87 = r87 * r46;
    r88 = r51 * r51;
    r89 = r16 * r33;
    r90 = r25 * r40;
    r91 = r34 * r80;
    r92 = r40 * r91;
    r90 = fmaf(r72, r90, r92);
    r93 = r16 * r29;
    r93 = r93 * r77;
    r94 = r16 * r32;
    r94 = fmaf(r73, r94, r93);
    r90 = r90 + r94;
    r95 = r25 * r80;
    r95 = r95 * r84;
    r85 = r95 + r85;
    r85 = fmaf(r14, r85, r15 * r90);
    r90 = r16 * r34;
    r90 = fmaf(r77, r90, r79);
    r79 = r16 * r32;
    r79 = fmaf(r72, r79, r81);
    r90 = r90 + r79;
    r85 = fmaf(r13, r90, r85);
    r89 = r89 * r85;
    r90 = r16 * r51;
    r90 = r90 * r82;
    r90 = fmaf(r44, r90, r44 * r89);
    r89 = r33 * r33;
    r81 = r16 * r25;
    r81 = r81 * r72;
    r91 = r16 * r91;
    r96 = r81 + r91;
    r94 = r94 + r96;
    r97 = r40 * r34;
    r97 = fmaf(r40, r76, r73 * r97);
    r97 = r97 + r75;
    r97 = fmaf(r13, r97, r14 * r94);
    r1 = r95 + r1;
    r97 = fmaf(r15, r1, r97);
    r55 = r47 * r55;
    r55 = 1.0 / r55;
    r47 = r40 * r55;
    r89 = r89 * r97;
    r90 = fmaf(r47, r89, r90);
    r1 = r51 * r51;
    r1 = r1 * r97;
    r90 = fmaf(r47, r1, r90);
    r41 = r41 + r35;
    r41 = 1.0 / r41;
    r35 = rsqrtf(r35);
    r88 = r88 * r61;
    r88 = r88 * r90;
    r88 = r88 * r41;
    r88 = r88 * r35;
    r88 = fmaf(r46, r88, r26 * r87);
    r87 = r40 * r12;
    r1 = r51 * r51;
    r87 = r87 * r1;
    r87 = r87 * r55;
    r1 = r36 * r36;
    r89 = -6.00000000000000000e+00;
    r1 = r1 * r97;
    r1 = r1 * r89;
    r1 = r1 * r48;
    r1 = r1 * r55;
    r95 = -3.00000000000000000e+00;
    r10 = r11 * r10;
    r10 = 1.0 / r10;
    r10 = r10 * r50;
    r11 = r26 * r10;
    r94 = r51 * r35;
    r11 = r11 * r94;
    r73 = r90 * r11;
    r98 = r85 * r49;
    r99 = r33 * r36;
    r100 = r33 * r46;
    r101 = r33 * r35;
    r102 = r41 * r101;
    r100 = r100 * r102;
    r98 = fmaf(r90, r100, r99 * r98);
    r103 = r4 * r90;
    r10 = r101 * r10;
    r103 = r103 * r99;
    r98 = fmaf(r10, r103, r98);
    r104 = r33 * r33;
    r104 = r104 * r36;
    r104 = r104 * r36;
    r104 = r104 * r97;
    r104 = r104 * r48;
    r98 = fmaf(r47, r104, r98);
    r88 = fmaf(r1, r87, r88);
    r88 = fmaf(r95, r73, r88);
    r88 = r88 + r98;
    r104 = r51 * r51;
    r104 = r104 * r90;
    r104 = r104 * r41;
    r104 = r104 * r35;
    r104 = fmaf(r46, r104, r82 * r63);
    r103 = r51 * r51;
    r103 = r103 * r36;
    r103 = r103 * r36;
    r103 = r103 * r97;
    r103 = r103 * r48;
    r104 = fmaf(r47, r103, r104);
    r104 = fmaf(r4, r73, r104);
    r98 = r98 + r104;
    r88 = fmaf(r8, r98, r58 * r88);
    r73 = r51 * r36;
    r73 = r73 * r67;
    r73 = r73 * r74;
    r73 = r73 * r90;
    r73 = r73 * r48;
    r73 = r73 * r60;
    r88 = fmaf(r35, r73, r88);
    r103 = r57 * r51;
    r103 = r103 * r90;
    r103 = r103 * r49;
    r88 = fmaf(r102, r103, r88);
    r105 = r51 * r36;
    r105 = r105 * r74;
    r105 = r105 * r90;
    r105 = r105 * r48;
    r105 = r105 * r60;
    r88 = fmaf(r35, r105, r88);
    r106 = r57 * r51;
    r106 = r106 * r33;
    r106 = r106 * r36;
    r106 = r106 * r36;
    r106 = r106 * r84;
    r106 = r106 * r97;
    r106 = r106 * r48;
    r88 = fmaf(r55, r106, r88);
    r107 = r71 * r41;
    r107 = r107 * r70;
    r107 = r107 * r94;
    r94 = r57 * r82;
    r94 = r94 * r49;
    r88 = fmaf(r99, r94, r88);
    r108 = r36 * r82;
    r88 = fmaf(r69, r108, r88);
    r109 = r4 * r51;
    r109 = r109 * r97;
    r109 = r109 * r68;
    r88 = fmaf(r50, r109, r88);
    r110 = r4 * r51;
    r110 = r110 * r67;
    r110 = r110 * r97;
    r110 = r110 * r68;
    r88 = fmaf(r50, r110, r88);
    r111 = r51 * r71;
    r111 = r111 * r90;
    r111 = r111 * r41;
    r111 = r111 * r35;
    r88 = fmaf(r69, r111, r88);
    r112 = r57 * r90;
    r113 = r40 * r26;
    r113 = r113 * r10;
    r88 = fmaf(r113, r112, r88);
    r114 = r7 * r16;
    r114 = r114 * r56;
    r114 = fmaf(r98, r114, r6 * r98);
    r115 = 4.00000000000000000e+00;
    r66 = r115 * r66;
    r59 = r59 * r61;
    r59 = r59 * r65;
    r114 = fmaf(r98, r66, r114);
    r114 = fmaf(r98, r59, r114);
    r65 = r114 * r69;
    r88 = fmaf(r26, r65, r88);
    r88 = fmaf(r90, r107, r88);
    r88 = fmaf(r85, r64, r88);
    r88 = fmaf(r82, r52, r88);
    r3 = r3 * r2;
    r65 = r5 * r4;
    r65 = r65 * r62;
    r62 = r33 * r36;
    r62 = r62 * r85;
    r62 = r62 * r86;
    r112 = r61 * r90;
    r112 = fmaf(r100, r112, r46 * r62);
    r62 = r95 * r99;
    r62 = r62 * r10;
    r111 = r33 * r33;
    r111 = r111 * r47;
    r112 = fmaf(r90, r62, r112);
    r112 = fmaf(r1, r111, r112);
    r112 = r112 + r104;
    r98 = fmaf(r9, r98, r57 * r112);
    r112 = r71 * r90;
    r112 = r112 * r69;
    r98 = fmaf(r102, r112, r98);
    r104 = r36 * r74;
    r104 = r104 * r48;
    r104 = r104 * r60;
    r104 = r104 * r101;
    r101 = r67 * r104;
    r111 = r33 * r36;
    r111 = r111 * r114;
    r98 = fmaf(r69, r111, r98);
    r1 = r4 * r33;
    r1 = r1 * r67;
    r1 = r1 * r97;
    r1 = r1 * r68;
    r98 = fmaf(r50, r1, r98);
    r110 = r58 * r51;
    r110 = r110 * r33;
    r110 = r110 * r36;
    r110 = r110 * r36;
    r110 = r110 * r84;
    r110 = r110 * r48;
    r110 = r110 * r55;
    r109 = r58 * r82;
    r109 = r109 * r49;
    r98 = fmaf(r99, r109, r98);
    r108 = r58 * r51;
    r108 = r108 * r90;
    r108 = r108 * r49;
    r98 = fmaf(r102, r108, r98);
    r94 = r4 * r33;
    r94 = r94 * r97;
    r94 = r94 * r68;
    r98 = fmaf(r50, r94, r98);
    r106 = r58 * r90;
    r98 = fmaf(r113, r106, r98);
    r105 = r71 * r90;
    r105 = r105 * r102;
    r98 = fmaf(r70, r105, r98);
    r103 = r58 * r85;
    r98 = fmaf(r63, r103, r98);
    r73 = r36 * r85;
    r98 = fmaf(r69, r73, r98);
    r98 = fmaf(r90, r101, r98);
    r98 = fmaf(r97, r110, r98);
    r98 = fmaf(r90, r104, r98);
    r98 = fmaf(r85, r52, r98);
    r3 = fmaf(r98, r65, r88 * r3);
    r73 = r0 * r4;
    r103 = r51 * r51;
    r105 = r16 * r51;
    r91 = r93 + r91;
    r93 = r16 * r32;
    r106 = r19 * r24;
    r94 = r18 * r21;
    r94 = fmaf(r74, r94, r74 * r106);
    r106 = r17 * r22;
    r94 = fmaf(r71, r106, r94);
    r108 = r20 * r23;
    r94 = fmaf(r74, r108, r94);
    r93 = r93 * r94;
    r108 = r16 * r25;
    r106 = r17 * r24;
    r109 = r20 * r21;
    r109 = fmaf(r71, r109, r71 * r106);
    r106 = r19 * r22;
    r109 = fmaf(r71, r106, r109);
    r97 = r18 * r23;
    r109 = fmaf(r74, r97, r109);
    r108 = fmaf(r109, r108, r93);
    r91 = r91 + r108;
    r97 = r29 * r80;
    r97 = r97 * r84;
    r106 = r32 * r84;
    r106 = r106 * r109;
    r1 = r97 + r106;
    r1 = fmaf(r13, r1, r15 * r91);
    r91 = r40 * r34;
    r91 = fmaf(r40, r83, r109 * r91);
    r111 = r16 * r25;
    r111 = r111 * r80;
    r112 = r16 * r29;
    r112 = fmaf(r94, r112, r111);
    r91 = r91 + r112;
    r1 = fmaf(r14, r91, r1);
    r105 = r105 * r1;
    r91 = r51 * r51;
    r115 = r40 * r29;
    r115 = fmaf(r77, r115, r92);
    r115 = r115 + r108;
    r108 = r16 * r29;
    r108 = r108 * r109;
    r116 = r16 * r34;
    r116 = fmaf(r94, r116, r108);
    r116 = r116 + r75;
    r116 = fmaf(r14, r116, r13 * r115);
    r115 = r25 * r94;
    r75 = r84 * r115;
    r97 = r97 + r75;
    r116 = fmaf(r15, r97, r116);
    r91 = r91 * r116;
    r91 = fmaf(r47, r91, r44 * r105);
    r105 = r33 * r33;
    r105 = r105 * r116;
    r91 = fmaf(r47, r105, r91);
    r97 = r16 * r33;
    r108 = r78 + r108;
    r78 = r25 * r40;
    r108 = fmaf(r77, r78, r108);
    r77 = r40 * r34;
    r108 = fmaf(r94, r77, r108);
    r77 = r16 * r34;
    r83 = fmaf(r16, r83, r109 * r77);
    r83 = r83 + r112;
    r83 = fmaf(r13, r83, r15 * r108);
    r75 = r106 + r75;
    r83 = fmaf(r14, r75, r83);
    r97 = r97 * r83;
    r91 = fmaf(r44, r97, r91);
    r103 = r103 * r91;
    r103 = r103 * r41;
    r103 = r103 * r35;
    r97 = r4 * r91;
    r97 = fmaf(r11, r97, r46 * r103);
    r103 = r51 * r51;
    r103 = r103 * r36;
    r103 = r103 * r36;
    r103 = r103 * r116;
    r103 = r103 * r48;
    r97 = fmaf(r47, r103, r97);
    r97 = fmaf(r1, r63, r97);
    r103 = r83 * r49;
    r105 = r4 * r91;
    r105 = r105 * r99;
    r105 = fmaf(r10, r105, r99 * r103);
    r103 = r33 * r33;
    r103 = r103 * r36;
    r103 = r103 * r36;
    r103 = r103 * r116;
    r103 = r103 * r48;
    r105 = fmaf(r47, r103, r105);
    r105 = fmaf(r91, r100, r105);
    r103 = r97 + r105;
    r75 = r51 * r51;
    r75 = r75 * r61;
    r75 = r75 * r91;
    r75 = r75 * r41;
    r75 = r75 * r35;
    r106 = r95 * r91;
    r106 = fmaf(r11, r106, r46 * r75);
    r75 = r51 * r51;
    r75 = r75 * r36;
    r75 = r75 * r36;
    r75 = r75 * r89;
    r75 = r75 * r116;
    r75 = r75 * r48;
    r106 = fmaf(r55, r75, r106);
    r108 = r86 * r1;
    r108 = r108 * r46;
    r106 = fmaf(r26, r108, r106);
    r106 = r106 + r105;
    r106 = fmaf(r58, r106, r8 * r103);
    r105 = r57 * r51;
    r105 = r105 * r33;
    r105 = r105 * r36;
    r105 = r105 * r36;
    r105 = r105 * r84;
    r105 = r105 * r116;
    r105 = r105 * r48;
    r106 = fmaf(r55, r105, r106);
    r108 = r51 * r36;
    r108 = r108 * r74;
    r108 = r108 * r91;
    r108 = r108 * r48;
    r108 = r108 * r60;
    r106 = fmaf(r35, r108, r106);
    r75 = r4 * r51;
    r75 = r75 * r116;
    r75 = r75 * r68;
    r106 = fmaf(r50, r75, r106);
    r77 = r57 * r91;
    r106 = fmaf(r113, r77, r106);
    r109 = r57 * r51;
    r109 = r109 * r91;
    r109 = r109 * r49;
    r106 = fmaf(r102, r109, r106);
    r78 = r51 * r71;
    r78 = r78 * r91;
    r78 = r78 * r41;
    r78 = r78 * r35;
    r106 = fmaf(r69, r78, r106);
    r117 = r4 * r51;
    r117 = r117 * r67;
    r117 = r117 * r116;
    r117 = r117 * r68;
    r106 = fmaf(r50, r117, r106);
    r118 = r57 * r1;
    r118 = r118 * r49;
    r106 = fmaf(r99, r118, r106);
    r119 = r7 * r16;
    r119 = r119 * r56;
    r119 = fmaf(r103, r119, r6 * r103);
    r119 = fmaf(r103, r66, r119);
    r119 = fmaf(r103, r59, r119);
    r120 = r119 * r69;
    r106 = fmaf(r26, r120, r106);
    r121 = r51 * r36;
    r121 = r121 * r67;
    r121 = r121 * r74;
    r121 = r121 * r91;
    r121 = r121 * r48;
    r121 = r121 * r60;
    r106 = fmaf(r35, r121, r106);
    r122 = r36 * r1;
    r106 = fmaf(r69, r122, r106);
    r106 = fmaf(r83, r64, r106);
    r106 = fmaf(r91, r107, r106);
    r106 = fmaf(r1, r52, r106);
    r73 = r73 * r2;
    r122 = r33 * r36;
    r122 = r122 * r86;
    r122 = r122 * r83;
    r122 = fmaf(r91, r62, r46 * r122);
    r121 = r33 * r33;
    r121 = r121 * r36;
    r121 = r121 * r36;
    r121 = r121 * r89;
    r121 = r121 * r116;
    r121 = r121 * r48;
    r122 = fmaf(r55, r121, r122);
    r120 = r61 * r91;
    r122 = fmaf(r100, r120, r122);
    r122 = r122 + r97;
    r122 = fmaf(r57, r122, r9 * r103);
    r103 = r4 * r33;
    r103 = r103 * r116;
    r103 = r103 * r68;
    r122 = fmaf(r50, r103, r122);
    r97 = r58 * r83;
    r122 = fmaf(r63, r97, r122);
    r120 = r71 * r91;
    r120 = r120 * r102;
    r122 = fmaf(r70, r120, r122);
    r121 = r33 * r36;
    r121 = r121 * r119;
    r122 = fmaf(r69, r121, r122);
    r118 = r71 * r91;
    r118 = r118 * r69;
    r122 = fmaf(r102, r118, r122);
    r117 = r58 * r91;
    r122 = fmaf(r113, r117, r122);
    r78 = r58 * r51;
    r78 = r78 * r91;
    r78 = r78 * r49;
    r122 = fmaf(r102, r78, r122);
    r109 = r58 * r1;
    r109 = r109 * r49;
    r122 = fmaf(r99, r109, r122);
    r77 = r4 * r33;
    r77 = r77 * r67;
    r77 = r77 * r116;
    r77 = r77 * r68;
    r122 = fmaf(r50, r77, r122);
    r75 = r36 * r83;
    r122 = fmaf(r69, r75, r122);
    r122 = fmaf(r116, r110, r122);
    r122 = fmaf(r91, r101, r122);
    r122 = fmaf(r91, r104, r122);
    r122 = fmaf(r83, r52, r122);
    r73 = fmaf(r122, r65, r106 * r73);
    r75 = r0 * r4;
    r77 = r51 * r51;
    r109 = r25 * r84;
    r31 = fmaf(r74, r31, r71 * r27);
    r31 = fmaf(r71, r28, r31);
    r31 = fmaf(r71, r30, r31);
    r109 = r109 * r31;
    r76 = r84 * r76;
    r30 = r109 + r76;
    r28 = r16 * r32;
    r28 = r28 * r31;
    r111 = r111 + r28;
    r27 = r40 * r29;
    r111 = fmaf(r94, r27, r111);
    r78 = r40 * r34;
    r111 = fmaf(r72, r78, r111);
    r111 = fmaf(r13, r111, r15 * r30);
    r30 = r16 * r34;
    r30 = fmaf(r16, r115, r31 * r30);
    r30 = r30 + r79;
    r111 = fmaf(r14, r30, r111);
    r77 = r77 * r111;
    r30 = r16 * r51;
    r78 = r16 * r29;
    r78 = r78 * r31;
    r81 = r81 + r78;
    r27 = r32 * r40;
    r81 = fmaf(r94, r27, r81);
    r81 = r81 + r92;
    r80 = r32 * r80;
    r80 = r80 * r84;
    r76 = r80 + r76;
    r76 = fmaf(r13, r76, r14 * r81);
    r81 = r16 * r34;
    r81 = fmaf(r72, r81, r28);
    r81 = r81 + r112;
    r76 = fmaf(r15, r81, r76);
    r30 = r30 * r76;
    r30 = fmaf(r44, r30, r47 * r77);
    r77 = r33 * r33;
    r77 = r77 * r111;
    r30 = fmaf(r47, r77, r30);
    r81 = r16 * r33;
    r78 = r93 + r78;
    r78 = r78 + r96;
    r96 = r40 * r34;
    r115 = fmaf(r40, r115, r31 * r96);
    r115 = r115 + r79;
    r115 = fmaf(r15, r115, r13 * r78);
    r80 = r109 + r80;
    r115 = fmaf(r14, r80, r115);
    r81 = r81 * r115;
    r30 = fmaf(r44, r81, r30);
    r81 = r115 * r49;
    r81 = fmaf(r99, r81, r30 * r100);
    r77 = r33 * r33;
    r77 = r77 * r36;
    r77 = r77 * r36;
    r77 = r77 * r111;
    r77 = r77 * r48;
    r81 = fmaf(r47, r77, r81);
    r80 = r4 * r30;
    r80 = r80 * r99;
    r81 = fmaf(r10, r80, r81);
    r80 = r51 * r51;
    r80 = r80 * r36;
    r80 = r80 * r36;
    r80 = r80 * r111;
    r80 = r80 * r48;
    r77 = r51 * r51;
    r77 = r77 * r30;
    r77 = r77 * r41;
    r77 = r77 * r35;
    r77 = fmaf(r46, r77, r47 * r80);
    r80 = r4 * r30;
    r77 = fmaf(r11, r80, r77);
    r77 = fmaf(r76, r63, r77);
    r80 = r81 + r77;
    r14 = r51 * r51;
    r14 = r14 * r36;
    r14 = r14 * r36;
    r14 = r14 * r89;
    r14 = r14 * r111;
    r14 = r14 * r48;
    r109 = r51 * r51;
    r109 = r109 * r61;
    r109 = r109 * r30;
    r109 = r109 * r41;
    r109 = r109 * r35;
    r109 = fmaf(r46, r109, r55 * r14);
    r14 = r95 * r30;
    r109 = fmaf(r11, r14, r109);
    r15 = r86 * r76;
    r15 = r15 * r46;
    r109 = fmaf(r26, r15, r109);
    r109 = r109 + r81;
    r109 = fmaf(r58, r109, r8 * r80);
    r81 = r7 * r16;
    r81 = r81 * r56;
    r81 = fmaf(r80, r81, r6 * r80);
    r81 = fmaf(r80, r59, r81);
    r81 = fmaf(r80, r66, r81);
    r15 = r81 * r69;
    r109 = fmaf(r26, r15, r109);
    r14 = r57 * r51;
    r14 = r14 * r33;
    r14 = r14 * r36;
    r14 = r14 * r36;
    r14 = r14 * r84;
    r14 = r14 * r111;
    r14 = r14 * r48;
    r109 = fmaf(r55, r14, r109);
    r78 = r51 * r71;
    r78 = r78 * r30;
    r78 = r78 * r41;
    r78 = r78 * r35;
    r109 = fmaf(r69, r78, r109);
    r13 = r51 * r36;
    r13 = r13 * r67;
    r13 = r13 * r74;
    r13 = r13 * r30;
    r13 = r13 * r48;
    r13 = r13 * r60;
    r109 = fmaf(r35, r13, r109);
    r79 = r4 * r51;
    r79 = r79 * r111;
    r79 = r79 * r68;
    r109 = fmaf(r50, r79, r109);
    r96 = r57 * r30;
    r109 = fmaf(r113, r96, r109);
    r31 = r57 * r76;
    r31 = r31 * r49;
    r109 = fmaf(r99, r31, r109);
    r93 = r51 * r36;
    r93 = r93 * r74;
    r93 = r93 * r30;
    r93 = r93 * r48;
    r93 = r93 * r60;
    r109 = fmaf(r35, r93, r109);
    r112 = r4 * r51;
    r112 = r112 * r67;
    r112 = r112 * r111;
    r112 = r112 * r68;
    r109 = fmaf(r50, r112, r109);
    r28 = r36 * r76;
    r109 = fmaf(r69, r28, r109);
    r72 = r57 * r51;
    r72 = r72 * r30;
    r72 = r72 * r49;
    r109 = fmaf(r102, r72, r109);
    r109 = fmaf(r76, r52, r109);
    r109 = fmaf(r30, r107, r109);
    r109 = fmaf(r115, r64, r109);
    r75 = r75 * r2;
    r72 = r61 * r30;
    r28 = r33 * r36;
    r28 = r28 * r86;
    r28 = r28 * r115;
    r28 = fmaf(r46, r28, r100 * r72);
    r72 = r33 * r33;
    r72 = r72 * r36;
    r72 = r72 * r36;
    r72 = r72 * r89;
    r72 = r72 * r111;
    r72 = r72 * r48;
    r28 = fmaf(r55, r72, r28);
    r28 = fmaf(r30, r62, r28);
    r28 = r28 + r77;
    r28 = fmaf(r57, r28, r9 * r80);
    r80 = r71 * r30;
    r80 = r80 * r69;
    r28 = fmaf(r102, r80, r28);
    r77 = r4 * r33;
    r77 = r77 * r67;
    r77 = r77 * r111;
    r77 = r77 * r68;
    r28 = fmaf(r50, r77, r28);
    r72 = r58 * r30;
    r28 = fmaf(r113, r72, r28);
    r112 = r58 * r76;
    r112 = r112 * r49;
    r28 = fmaf(r99, r112, r28);
    r93 = r58 * r115;
    r28 = fmaf(r63, r93, r28);
    r31 = r4 * r33;
    r31 = r31 * r111;
    r31 = r31 * r68;
    r28 = fmaf(r50, r31, r28);
    r96 = r58 * r51;
    r96 = r96 * r30;
    r96 = r96 * r49;
    r28 = fmaf(r102, r96, r28);
    r79 = r36 * r115;
    r28 = fmaf(r69, r79, r28);
    r13 = r33 * r36;
    r13 = r13 * r81;
    r28 = fmaf(r69, r13, r28);
    r78 = r71 * r30;
    r78 = r78 * r102;
    r28 = fmaf(r70, r78, r28);
    r28 = fmaf(r111, r110, r28);
    r28 = fmaf(r30, r104, r28);
    r28 = fmaf(r115, r52, r28);
    r28 = fmaf(r30, r101, r28);
    r75 = fmaf(r28, r65, r109 * r75);
    r78 = r0 * r4;
    r13 = r12 * r51;
    r13 = r13 * r51;
    r13 = r13 * r36;
    r13 = r13 * r36;
    r13 = r13 * r89;
    r13 = r13 * r48;
    r79 = r42 * r86;
    r79 = r79 * r46;
    r79 = fmaf(r26, r79, r55 * r13);
    r13 = r40 * r12;
    r96 = r33 * r33;
    r13 = r13 * r96;
    r13 = r13 * r55;
    r31 = r13 + r87;
    r93 = r16 * r38;
    r93 = r93 * r33;
    r31 = fmaf(r44, r93, r31);
    r112 = r16 * r42;
    r112 = r112 * r51;
    r31 = fmaf(r44, r112, r31);
    r112 = r95 * r31;
    r79 = fmaf(r11, r112, r79);
    r93 = r51 * r51;
    r93 = r93 * r61;
    r93 = r93 * r31;
    r93 = r93 * r41;
    r93 = r93 * r35;
    r79 = fmaf(r46, r93, r79);
    r72 = r36 * r36;
    r72 = r72 * r48;
    r77 = r38 * r49;
    r77 = fmaf(r99, r77, r72 * r13);
    r13 = r4 * r31;
    r13 = r13 * r99;
    r77 = fmaf(r10, r13, r77);
    r77 = fmaf(r31, r100, r77);
    r79 = r79 + r77;
    r72 = fmaf(r42, r63, r87 * r72);
    r87 = r4 * r31;
    r72 = fmaf(r11, r87, r72);
    r93 = r51 * r51;
    r93 = r93 * r31;
    r93 = r93 * r41;
    r93 = r93 * r35;
    r72 = fmaf(r46, r93, r72);
    r77 = r77 + r72;
    r79 = fmaf(r8, r77, r58 * r79);
    r93 = r4 * r12;
    r93 = r93 * r51;
    r93 = r93 * r68;
    r79 = fmaf(r50, r93, r79);
    r87 = r42 * r36;
    r79 = fmaf(r69, r87, r79);
    r112 = r51 * r36;
    r112 = r112 * r67;
    r112 = r112 * r74;
    r112 = r112 * r31;
    r112 = r112 * r48;
    r112 = r112 * r60;
    r79 = fmaf(r35, r112, r79);
    r13 = r31 * r113;
    r80 = r57 * r51;
    r80 = r80 * r31;
    r80 = r80 * r49;
    r79 = fmaf(r102, r80, r79);
    r111 = r57 * r42;
    r111 = r111 * r49;
    r79 = fmaf(r99, r111, r79);
    r14 = r51 * r36;
    r14 = r14 * r74;
    r14 = r14 * r31;
    r14 = r14 * r48;
    r14 = r14 * r60;
    r79 = fmaf(r35, r14, r79);
    r15 = r57 * r12;
    r15 = r15 * r51;
    r15 = r15 * r33;
    r15 = r15 * r36;
    r15 = r15 * r36;
    r15 = r15 * r84;
    r15 = r15 * r48;
    r79 = fmaf(r55, r15, r79);
    r92 = r51 * r71;
    r92 = r92 * r31;
    r92 = r92 * r41;
    r92 = r92 * r35;
    r79 = fmaf(r69, r92, r79);
    r27 = r7 * r16;
    r27 = r27 * r56;
    r27 = fmaf(r6, r77, r77 * r27);
    r27 = fmaf(r77, r66, r27);
    r27 = fmaf(r77, r59, r27);
    r94 = r27 * r69;
    r79 = fmaf(r26, r94, r79);
    r117 = r4 * r12;
    r117 = r117 * r51;
    r117 = r117 * r67;
    r117 = r117 * r68;
    r79 = fmaf(r50, r117, r79);
    r79 = fmaf(r57, r13, r79);
    r79 = fmaf(r42, r52, r79);
    r79 = fmaf(r31, r107, r79);
    r79 = fmaf(r38, r64, r79);
    r78 = r78 * r2;
    r117 = r12 * r33;
    r117 = r117 * r33;
    r117 = r117 * r36;
    r117 = r117 * r36;
    r117 = r117 * r89;
    r117 = r117 * r48;
    r94 = r38 * r33;
    r94 = r94 * r36;
    r94 = r94 * r86;
    r94 = fmaf(r46, r94, r55 * r117);
    r117 = r61 * r31;
    r94 = fmaf(r100, r117, r94);
    r94 = fmaf(r31, r62, r94);
    r94 = r94 + r72;
    r77 = fmaf(r9, r77, r57 * r94);
    r94 = r4 * r12;
    r94 = r94 * r33;
    r94 = r94 * r68;
    r77 = fmaf(r50, r94, r77);
    r72 = r4 * r12;
    r72 = r72 * r33;
    r72 = r72 * r67;
    r72 = r72 * r68;
    r77 = fmaf(r50, r72, r77);
    r117 = r38 * r36;
    r77 = fmaf(r69, r117, r77);
    r92 = r58 * r51;
    r92 = r92 * r31;
    r92 = r92 * r49;
    r77 = fmaf(r102, r92, r77);
    r15 = r58 * r42;
    r15 = r15 * r49;
    r77 = fmaf(r99, r15, r77);
    r14 = r71 * r31;
    r14 = r14 * r69;
    r77 = fmaf(r102, r14, r77);
    r111 = r58 * r38;
    r77 = fmaf(r63, r111, r77);
    r80 = r71 * r31;
    r80 = r80 * r102;
    r77 = fmaf(r70, r80, r77);
    r112 = r33 * r36;
    r112 = r112 * r27;
    r77 = fmaf(r69, r112, r77);
    r77 = fmaf(r31, r104, r77);
    r77 = fmaf(r31, r101, r77);
    r77 = fmaf(r38, r52, r77);
    r77 = fmaf(r58, r13, r77);
    r77 = fmaf(r12, r110, r77);
    r78 = fmaf(r77, r65, r79 * r78);
    WriteSum4<float, float>((float*)inout_shared, r3, r73, r75, r78);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = r0 * r4;
    r75 = r39 * r33;
    r75 = r75 * r33;
    r73 = r16 * r43;
    r73 = r73 * r33;
    r73 = fmaf(r44, r73, r47 * r75);
    r75 = r39 * r51;
    r75 = r75 * r51;
    r73 = fmaf(r47, r75, r73);
    r3 = r16 * r54;
    r3 = r3 * r51;
    r73 = fmaf(r44, r3, r73);
    r3 = r39 * r33;
    r3 = r3 * r33;
    r3 = r3 * r36;
    r3 = r3 * r36;
    r3 = r3 * r48;
    r3 = fmaf(r47, r3, r73 * r100);
    r75 = r43 * r49;
    r3 = fmaf(r99, r75, r3);
    r112 = r4 * r73;
    r112 = r112 * r99;
    r3 = fmaf(r10, r112, r3);
    r112 = r51 * r51;
    r112 = r112 * r73;
    r112 = r112 * r41;
    r112 = r112 * r35;
    r75 = r39 * r51;
    r75 = r75 * r51;
    r75 = r75 * r36;
    r75 = r75 * r36;
    r75 = r75 * r48;
    r75 = fmaf(r47, r75, r46 * r112);
    r112 = r4 * r73;
    r75 = fmaf(r11, r112, r75);
    r75 = fmaf(r54, r63, r75);
    r112 = r3 + r75;
    r80 = r51 * r51;
    r80 = r80 * r61;
    r80 = r80 * r73;
    r80 = r80 * r41;
    r80 = r80 * r35;
    r111 = r39 * r51;
    r111 = r111 * r51;
    r111 = r111 * r36;
    r111 = r111 * r36;
    r111 = r111 * r89;
    r111 = r111 * r48;
    r111 = fmaf(r55, r111, r46 * r80);
    r80 = r54 * r86;
    r80 = r80 * r46;
    r111 = fmaf(r26, r80, r111);
    r14 = r95 * r73;
    r111 = fmaf(r11, r14, r111);
    r111 = r111 + r3;
    r111 = fmaf(r58, r111, r8 * r112);
    r3 = r7 * r16;
    r3 = r3 * r56;
    r3 = fmaf(r112, r3, r6 * r112);
    r3 = fmaf(r112, r66, r3);
    r3 = fmaf(r112, r59, r3);
    r14 = r3 * r69;
    r111 = fmaf(r26, r14, r111);
    r80 = r51 * r36;
    r80 = r80 * r74;
    r80 = r80 * r73;
    r80 = r80 * r48;
    r80 = r80 * r60;
    r111 = fmaf(r35, r80, r111);
    r15 = r54 * r36;
    r111 = fmaf(r69, r15, r111);
    r92 = r57 * r51;
    r92 = r92 * r73;
    r92 = r92 * r49;
    r111 = fmaf(r102, r92, r111);
    r13 = r51 * r71;
    r13 = r13 * r73;
    r13 = r13 * r41;
    r13 = r13 * r35;
    r111 = fmaf(r69, r13, r111);
    r117 = r57 * r39;
    r117 = r117 * r51;
    r117 = r117 * r33;
    r117 = r117 * r36;
    r117 = r117 * r36;
    r117 = r117 * r84;
    r117 = r117 * r48;
    r111 = fmaf(r55, r117, r111);
    r72 = r57 * r73;
    r111 = fmaf(r113, r72, r111);
    r94 = r51 * r36;
    r94 = r94 * r67;
    r94 = r94 * r74;
    r94 = r94 * r73;
    r94 = r94 * r48;
    r94 = r94 * r60;
    r111 = fmaf(r35, r94, r111);
    r87 = r4 * r39;
    r87 = r87 * r51;
    r87 = r87 * r68;
    r111 = fmaf(r50, r87, r111);
    r93 = r57 * r54;
    r93 = r93 * r49;
    r111 = fmaf(r99, r93, r111);
    r118 = r4 * r39;
    r118 = r118 * r51;
    r118 = r118 * r67;
    r118 = r118 * r68;
    r111 = fmaf(r50, r118, r111);
    r111 = fmaf(r54, r52, r111);
    r111 = fmaf(r43, r64, r111);
    r111 = fmaf(r73, r107, r111);
    r78 = r78 * r2;
    r118 = r61 * r73;
    r93 = r39 * r33;
    r93 = r93 * r33;
    r93 = r93 * r36;
    r93 = r93 * r36;
    r93 = r93 * r89;
    r93 = r93 * r48;
    r93 = fmaf(r55, r93, r100 * r118);
    r118 = r43 * r33;
    r118 = r118 * r36;
    r118 = r118 * r86;
    r93 = fmaf(r46, r118, r93);
    r93 = fmaf(r73, r62, r93);
    r93 = r93 + r75;
    r93 = fmaf(r57, r93, r9 * r112);
    r112 = r58 * r43;
    r93 = fmaf(r63, r112, r93);
    r75 = r4 * r39;
    r75 = r75 * r33;
    r75 = r75 * r68;
    r93 = fmaf(r50, r75, r93);
    r118 = r71 * r73;
    r118 = r118 * r69;
    r93 = fmaf(r102, r118, r93);
    r100 = r58 * r51;
    r100 = r100 * r73;
    r100 = r100 * r49;
    r93 = fmaf(r102, r100, r93);
    r87 = r71 * r73;
    r87 = r87 * r102;
    r93 = fmaf(r70, r87, r93);
    r94 = r58 * r73;
    r93 = fmaf(r113, r94, r93);
    r72 = r43 * r36;
    r93 = fmaf(r69, r72, r93);
    r117 = r58 * r54;
    r117 = r117 * r49;
    r93 = fmaf(r99, r117, r93);
    r13 = r4 * r39;
    r13 = r13 * r33;
    r13 = r13 * r67;
    r13 = r13 * r68;
    r93 = fmaf(r50, r13, r93);
    r92 = r33 * r36;
    r92 = r92 * r3;
    r93 = fmaf(r69, r92, r93);
    r93 = fmaf(r43, r52, r93);
    r93 = fmaf(r39, r110, r93);
    r93 = fmaf(r73, r101, r93);
    r93 = fmaf(r73, r104, r93);
    r78 = fmaf(r93, r65, r111 * r78);
    r92 = r0 * r4;
    r96 = r44 * r96;
    r13 = r16 * r45;
    r13 = r13 * r33;
    r117 = r37 * r33;
    r117 = r117 * r33;
    r117 = fmaf(r47, r117, r44 * r13);
    r13 = r16 * r53;
    r13 = r13 * r51;
    r117 = fmaf(r44, r13, r117);
    r44 = r37 * r51;
    r44 = r44 * r51;
    r117 = fmaf(r47, r44, r117);
    r96 = r96 * r48;
    r96 = r96 * r36;
    r96 = r96 * r41;
    r96 = r96 * r35;
    r96 = r96 * r117;
    r44 = r45 * r49;
    r44 = fmaf(r99, r44, r96);
    r13 = r4 * r117;
    r13 = r13 * r99;
    r44 = fmaf(r10, r13, r44);
    r10 = r37 * r33;
    r10 = r10 * r33;
    r10 = r10 * r36;
    r10 = r10 * r36;
    r10 = r10 * r48;
    r44 = fmaf(r47, r10, r44);
    r10 = r37 * r51;
    r10 = r10 * r51;
    r10 = r10 * r36;
    r10 = r10 * r36;
    r10 = r10 * r48;
    r13 = r4 * r117;
    r13 = fmaf(r11, r13, r47 * r10);
    r10 = r51 * r51;
    r10 = r10 * r117;
    r10 = r10 * r41;
    r10 = r10 * r35;
    r13 = fmaf(r46, r10, r13);
    r13 = fmaf(r53, r63, r13);
    r10 = r44 + r13;
    r47 = r37 * r51;
    r47 = r47 * r51;
    r47 = r47 * r36;
    r47 = r47 * r36;
    r47 = r47 * r89;
    r47 = r47 * r48;
    r72 = r95 * r117;
    r72 = fmaf(r11, r72, r55 * r47);
    r47 = r53 * r86;
    r47 = r47 * r46;
    r72 = fmaf(r26, r47, r72);
    r11 = r51 * r51;
    r11 = r11 * r61;
    r11 = r11 * r117;
    r11 = r11 * r41;
    r11 = r11 * r35;
    r72 = fmaf(r46, r11, r72);
    r72 = r72 + r44;
    r72 = fmaf(r58, r72, r8 * r10);
    r8 = r4 * r37;
    r8 = r8 * r51;
    r8 = r8 * r67;
    r8 = r8 * r68;
    r72 = fmaf(r50, r8, r72);
    r44 = r53 * r36;
    r72 = fmaf(r69, r44, r72);
    r11 = r4 * r37;
    r11 = r11 * r51;
    r11 = r11 * r68;
    r72 = fmaf(r50, r11, r72);
    r47 = r51 * r36;
    r47 = r47 * r74;
    r47 = r47 * r117;
    r47 = r47 * r48;
    r47 = r47 * r60;
    r72 = fmaf(r35, r47, r72);
    r94 = r57 * r117;
    r72 = fmaf(r113, r94, r72);
    r87 = r57 * r51;
    r87 = r87 * r117;
    r87 = r87 * r49;
    r72 = fmaf(r102, r87, r72);
    r100 = r57 * r53;
    r100 = r100 * r49;
    r72 = fmaf(r99, r100, r72);
    r118 = r51 * r36;
    r118 = r118 * r67;
    r118 = r118 * r74;
    r118 = r118 * r117;
    r118 = r118 * r48;
    r118 = r118 * r60;
    r72 = fmaf(r35, r118, r72);
    r60 = r51 * r71;
    r60 = r60 * r117;
    r60 = r60 * r41;
    r60 = r60 * r35;
    r72 = fmaf(r69, r60, r72);
    r35 = r7 * r16;
    r35 = r35 * r56;
    r6 = fmaf(r6, r10, r10 * r35);
    r6 = fmaf(r10, r66, r6);
    r6 = fmaf(r10, r59, r6);
    r59 = r6 * r69;
    r72 = fmaf(r26, r59, r72);
    r26 = r57 * r37;
    r26 = r26 * r51;
    r26 = r26 * r33;
    r26 = r26 * r36;
    r26 = r26 * r36;
    r26 = r26 * r84;
    r26 = r26 * r48;
    r72 = fmaf(r55, r26, r72);
    r72 = fmaf(r53, r52, r72);
    r72 = fmaf(r117, r107, r72);
    r72 = fmaf(r45, r64, r72);
    r92 = r92 * r2;
    r2 = r45 * r33;
    r2 = r2 * r36;
    r2 = r2 * r86;
    r2 = fmaf(r46, r2, r61 * r96);
    r96 = r37 * r33;
    r96 = r96 * r33;
    r96 = r96 * r36;
    r96 = r96 * r36;
    r96 = r96 * r89;
    r96 = r96 * r48;
    r2 = fmaf(r55, r96, r2);
    r2 = fmaf(r117, r62, r2);
    r2 = r2 + r13;
    r2 = fmaf(r57, r2, r9 * r10);
    r10 = r71 * r117;
    r10 = r10 * r102;
    r2 = fmaf(r70, r10, r2);
    r70 = r71 * r117;
    r70 = r70 * r69;
    r2 = fmaf(r102, r70, r2);
    r9 = r58 * r117;
    r2 = fmaf(r113, r9, r2);
    r113 = r58 * r51;
    r113 = r113 * r117;
    r113 = r113 * r49;
    r2 = fmaf(r102, r113, r2);
    r102 = r58 * r45;
    r2 = fmaf(r63, r102, r2);
    r63 = r4 * r37;
    r63 = r63 * r33;
    r63 = r63 * r67;
    r63 = r63 * r68;
    r2 = fmaf(r50, r63, r2);
    r67 = r58 * r53;
    r67 = r67 * r49;
    r2 = fmaf(r99, r67, r2);
    r99 = r45 * r36;
    r2 = fmaf(r69, r99, r2);
    r13 = r4 * r37;
    r13 = r13 * r33;
    r13 = r13 * r68;
    r2 = fmaf(r50, r13, r2);
    r50 = r33 * r36;
    r50 = r50 * r6;
    r2 = fmaf(r69, r50, r2);
    r2 = fmaf(r45, r52, r2);
    r2 = fmaf(r117, r101, r2);
    r2 = fmaf(r117, r104, r2);
    r2 = fmaf(r37, r110, r2);
    r65 = fmaf(r2, r65, r72 * r92);
    WriteSum2<float, float>((float*)inout_shared, r78, r65);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r0 * r0;
    r78 = r88 * r65;
    r5 = r5 * r5;
    r92 = r98 * r5;
    r98 = fmaf(r98, r92, r88 * r78);
    r88 = r106 * r106;
    r110 = r122 * r122;
    r110 = fmaf(r5, r110, r65 * r88);
    r88 = r28 * r28;
    r50 = r109 * r109;
    r50 = fmaf(r65, r50, r5 * r88);
    r88 = r79 * r79;
    r13 = r77 * r77;
    r13 = fmaf(r5, r13, r65 * r88);
    WriteSum4<float, float>((float*)inout_shared, r98, r110, r50, r13);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r111 * r111;
    r50 = r93 * r93;
    r50 = fmaf(r5, r50, r65 * r13);
    r13 = r2 * r2;
    r110 = r72 * r72;
    r110 = fmaf(r65, r110, r5 * r13);
    WriteSum2<float, float>((float*)inout_shared, r50, r110);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r110 = fmaf(r106, r78, r122 * r92);
    r50 = fmaf(r28, r92, r109 * r78);
    r13 = fmaf(r79, r78, r77 * r92);
    r98 = fmaf(r111, r78, r93 * r92);
    WriteSum4<float, float>((float*)inout_shared, r110, r50, r13, r98);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r92 = fmaf(r2, r92, r72 * r78);
    r78 = r122 * r28;
    r98 = r106 * r109;
    r98 = fmaf(r65, r98, r5 * r78);
    r78 = r106 * r79;
    r13 = r122 * r77;
    r13 = fmaf(r5, r13, r65 * r78);
    r78 = r106 * r111;
    r50 = r122 * r93;
    r50 = fmaf(r5, r50, r65 * r78);
    WriteSum4<float, float>((float*)inout_shared, r92, r98, r13, r50);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r122 * r2;
    r13 = r106 * r72;
    r13 = fmaf(r65, r13, r5 * r50);
    r50 = r28 * r77;
    r98 = r109 * r79;
    r98 = fmaf(r65, r98, r5 * r50);
    r50 = r109 * r111;
    r92 = r28 * r93;
    r92 = fmaf(r5, r92, r65 * r50);
    r50 = r28 * r2;
    r78 = r109 * r72;
    r78 = fmaf(r65, r78, r5 * r50);
    WriteSum4<float, float>((float*)inout_shared, r13, r98, r92, r78);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = r77 * r93;
    r92 = r79 * r111;
    r92 = fmaf(r65, r92, r5 * r78);
    r78 = r79 * r72;
    r98 = r77 * r2;
    r98 = fmaf(r5, r98, r65 * r78);
    r78 = r93 * r2;
    r13 = r111 * r72;
    r13 = fmaf(r65, r13, r5 * r78);
    WriteSum3<float, float>((float*)inout_shared, r92, r98, r13);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
}

void ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
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
  ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacKernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              sensor_from_rig,
              sensor_from_rig_num_alloc,
              pixel,
              pixel_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              principal_point,
              principal_point_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_pose_njtr,
              out_pose_njtr_num_alloc,
              out_pose_precond_diag,
              out_pose_precond_diag_num_alloc,
              out_pose_precond_tril,
              out_pose_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar