#include "kernel_thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacFirstKernel(
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
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104,
      r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116,
      r117, r118, r119, r120, r121, r122, r123, r124;

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
    r3 = fmaf(r62, r62, r2 * r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r3);
  if (global_thread_idx < problem_size) {
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
    r1 = r33 * r36;
    r1 = r1 * r114;
    r98 = fmaf(r69, r1, r98);
    r110 = r4 * r33;
    r110 = r110 * r67;
    r110 = r110 * r97;
    r110 = r110 * r68;
    r98 = fmaf(r50, r110, r98);
    r109 = r58 * r51;
    r109 = r109 * r33;
    r109 = r109 * r36;
    r109 = r109 * r36;
    r109 = r109 * r84;
    r109 = r109 * r48;
    r109 = r109 * r55;
    r108 = r58 * r82;
    r108 = r108 * r49;
    r98 = fmaf(r99, r108, r98);
    r94 = r58 * r51;
    r94 = r94 * r90;
    r94 = r94 * r49;
    r98 = fmaf(r102, r94, r98);
    r106 = r4 * r33;
    r106 = r106 * r97;
    r106 = r106 * r68;
    r98 = fmaf(r50, r106, r98);
    r105 = r58 * r90;
    r98 = fmaf(r113, r105, r98);
    r103 = r71 * r90;
    r103 = r103 * r102;
    r98 = fmaf(r70, r103, r98);
    r73 = r58 * r85;
    r98 = fmaf(r63, r73, r98);
    r115 = r36 * r85;
    r98 = fmaf(r69, r115, r98);
    r98 = fmaf(r90, r101, r98);
    r98 = fmaf(r97, r109, r98);
    r98 = fmaf(r90, r104, r98);
    r98 = fmaf(r85, r52, r98);
    r3 = fmaf(r98, r65, r88 * r3);
    r115 = r0 * r4;
    r73 = r33 * r33;
    r73 = r44 * r73;
    r103 = r16 * r51;
    r91 = r93 + r91;
    r93 = r16 * r32;
    r105 = r19 * r24;
    r106 = r18 * r21;
    r106 = fmaf(r74, r106, r74 * r105);
    r105 = r17 * r22;
    r106 = fmaf(r71, r105, r106);
    r94 = r20 * r23;
    r106 = fmaf(r74, r94, r106);
    r93 = r93 * r106;
    r94 = r16 * r25;
    r105 = r17 * r24;
    r108 = r20 * r21;
    r108 = fmaf(r71, r108, r71 * r105);
    r105 = r19 * r22;
    r108 = fmaf(r71, r105, r108);
    r97 = r18 * r23;
    r108 = fmaf(r74, r97, r108);
    r94 = fmaf(r108, r94, r93);
    r91 = r91 + r94;
    r97 = r29 * r80;
    r97 = r97 * r84;
    r105 = r32 * r84;
    r105 = r105 * r108;
    r110 = r97 + r105;
    r110 = fmaf(r13, r110, r15 * r91);
    r91 = r40 * r34;
    r91 = fmaf(r40, r83, r108 * r91);
    r1 = r16 * r25;
    r1 = r1 * r80;
    r112 = r16 * r29;
    r112 = fmaf(r106, r112, r1);
    r91 = r91 + r112;
    r110 = fmaf(r14, r91, r110);
    r103 = r103 * r110;
    r91 = r51 * r51;
    r116 = r40 * r29;
    r116 = fmaf(r77, r116, r92);
    r116 = r116 + r94;
    r94 = r16 * r29;
    r94 = r94 * r108;
    r117 = r16 * r34;
    r117 = fmaf(r106, r117, r94);
    r117 = r117 + r75;
    r117 = fmaf(r14, r117, r13 * r116);
    r116 = r25 * r106;
    r75 = r84 * r116;
    r97 = r97 + r75;
    r117 = fmaf(r15, r97, r117);
    r91 = r91 * r117;
    r91 = fmaf(r47, r91, r44 * r103);
    r103 = r33 * r33;
    r103 = r103 * r117;
    r91 = fmaf(r47, r103, r91);
    r97 = r16 * r33;
    r94 = r78 + r94;
    r78 = r25 * r40;
    r94 = fmaf(r77, r78, r94);
    r77 = r40 * r34;
    r94 = fmaf(r106, r77, r94);
    r77 = r16 * r34;
    r83 = fmaf(r16, r83, r108 * r77);
    r83 = r83 + r112;
    r83 = fmaf(r13, r83, r15 * r94);
    r75 = r105 + r75;
    r83 = fmaf(r14, r75, r83);
    r97 = r97 * r83;
    r91 = fmaf(r44, r97, r91);
    r73 = r73 * r48;
    r73 = r73 * r36;
    r73 = r73 * r41;
    r73 = r73 * r35;
    r73 = r73 * r91;
    r97 = r83 * r49;
    r97 = fmaf(r99, r97, r73);
    r103 = r4 * r91;
    r103 = r103 * r99;
    r97 = fmaf(r10, r103, r97);
    r75 = r33 * r33;
    r75 = r75 * r36;
    r75 = r75 * r36;
    r75 = r75 * r117;
    r75 = r75 * r48;
    r97 = fmaf(r47, r75, r97);
    r75 = r51 * r51;
    r75 = r75 * r91;
    r75 = r75 * r41;
    r75 = r75 * r35;
    r103 = r4 * r91;
    r103 = fmaf(r11, r103, r46 * r75);
    r75 = r51 * r51;
    r75 = r75 * r36;
    r75 = r75 * r36;
    r75 = r75 * r117;
    r75 = r75 * r48;
    r103 = fmaf(r47, r75, r103);
    r103 = fmaf(r110, r63, r103);
    r75 = r97 + r103;
    r105 = r51 * r51;
    r105 = r105 * r61;
    r105 = r105 * r91;
    r105 = r105 * r41;
    r105 = r105 * r35;
    r94 = r95 * r91;
    r94 = fmaf(r11, r94, r46 * r105);
    r105 = r51 * r51;
    r105 = r105 * r36;
    r105 = r105 * r36;
    r105 = r105 * r89;
    r105 = r105 * r117;
    r105 = r105 * r48;
    r94 = fmaf(r55, r105, r94);
    r77 = r86 * r110;
    r77 = r77 * r46;
    r94 = fmaf(r26, r77, r94);
    r94 = r94 + r97;
    r94 = fmaf(r58, r94, r8 * r75);
    r97 = r57 * r51;
    r97 = r97 * r33;
    r97 = r97 * r36;
    r97 = r97 * r36;
    r97 = r97 * r84;
    r97 = r97 * r117;
    r97 = r97 * r48;
    r94 = fmaf(r55, r97, r94);
    r77 = r51 * r36;
    r77 = r77 * r74;
    r77 = r77 * r91;
    r77 = r77 * r48;
    r77 = r77 * r60;
    r94 = fmaf(r35, r77, r94);
    r105 = r4 * r51;
    r105 = r105 * r117;
    r105 = r105 * r68;
    r94 = fmaf(r50, r105, r94);
    r108 = r57 * r91;
    r94 = fmaf(r113, r108, r94);
    r78 = r57 * r51;
    r78 = r78 * r91;
    r78 = r78 * r49;
    r94 = fmaf(r102, r78, r94);
    r118 = r51 * r71;
    r118 = r118 * r91;
    r118 = r118 * r41;
    r118 = r118 * r35;
    r94 = fmaf(r69, r118, r94);
    r119 = r4 * r51;
    r119 = r119 * r67;
    r119 = r119 * r117;
    r119 = r119 * r68;
    r94 = fmaf(r50, r119, r94);
    r120 = r57 * r110;
    r120 = r120 * r49;
    r94 = fmaf(r99, r120, r94);
    r121 = r7 * r16;
    r121 = r121 * r56;
    r121 = fmaf(r75, r121, r6 * r75);
    r121 = fmaf(r75, r66, r121);
    r121 = fmaf(r75, r59, r121);
    r122 = r121 * r69;
    r94 = fmaf(r26, r122, r94);
    r123 = r51 * r36;
    r123 = r123 * r67;
    r123 = r123 * r74;
    r123 = r123 * r91;
    r123 = r123 * r48;
    r123 = r123 * r60;
    r94 = fmaf(r35, r123, r94);
    r124 = r36 * r110;
    r94 = fmaf(r69, r124, r94);
    r94 = fmaf(r83, r64, r94);
    r94 = fmaf(r91, r107, r94);
    r94 = fmaf(r110, r52, r94);
    r115 = r115 * r2;
    r124 = r33 * r36;
    r124 = r124 * r86;
    r124 = r124 * r83;
    r124 = fmaf(r91, r62, r46 * r124);
    r123 = r33 * r33;
    r123 = r123 * r36;
    r123 = r123 * r36;
    r123 = r123 * r89;
    r123 = r123 * r117;
    r123 = r123 * r48;
    r124 = fmaf(r55, r123, r124);
    r124 = fmaf(r61, r73, r124);
    r124 = r124 + r103;
    r124 = fmaf(r57, r124, r9 * r75);
    r75 = r4 * r33;
    r75 = r75 * r117;
    r75 = r75 * r68;
    r124 = fmaf(r50, r75, r124);
    r103 = r58 * r83;
    r124 = fmaf(r63, r103, r124);
    r73 = r71 * r91;
    r73 = r73 * r102;
    r124 = fmaf(r70, r73, r124);
    r123 = r33 * r36;
    r123 = r123 * r121;
    r124 = fmaf(r69, r123, r124);
    r122 = r71 * r91;
    r122 = r122 * r69;
    r124 = fmaf(r102, r122, r124);
    r120 = r58 * r91;
    r124 = fmaf(r113, r120, r124);
    r119 = r58 * r51;
    r119 = r119 * r91;
    r119 = r119 * r49;
    r124 = fmaf(r102, r119, r124);
    r118 = r58 * r110;
    r118 = r118 * r49;
    r124 = fmaf(r99, r118, r124);
    r78 = r4 * r33;
    r78 = r78 * r67;
    r78 = r78 * r117;
    r78 = r78 * r68;
    r124 = fmaf(r50, r78, r124);
    r108 = r36 * r83;
    r124 = fmaf(r69, r108, r124);
    r124 = fmaf(r117, r109, r124);
    r124 = fmaf(r91, r101, r124);
    r124 = fmaf(r91, r104, r124);
    r124 = fmaf(r83, r52, r124);
    r115 = fmaf(r124, r65, r94 * r115);
    r108 = r0 * r4;
    r78 = r51 * r51;
    r118 = r25 * r84;
    r31 = fmaf(r74, r31, r71 * r27);
    r31 = fmaf(r71, r28, r31);
    r31 = fmaf(r71, r30, r31);
    r118 = r118 * r31;
    r76 = r84 * r76;
    r30 = r118 + r76;
    r28 = r16 * r32;
    r28 = r28 * r31;
    r1 = r1 + r28;
    r27 = r40 * r29;
    r1 = fmaf(r106, r27, r1);
    r119 = r40 * r34;
    r1 = fmaf(r72, r119, r1);
    r1 = fmaf(r13, r1, r15 * r30);
    r30 = r16 * r34;
    r30 = fmaf(r16, r116, r31 * r30);
    r30 = r30 + r79;
    r1 = fmaf(r14, r30, r1);
    r78 = r78 * r1;
    r30 = r16 * r51;
    r119 = r16 * r29;
    r119 = r119 * r31;
    r81 = r81 + r119;
    r27 = r32 * r40;
    r81 = fmaf(r106, r27, r81);
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
    r30 = fmaf(r44, r30, r47 * r78);
    r78 = r33 * r33;
    r78 = r78 * r1;
    r30 = fmaf(r47, r78, r30);
    r81 = r16 * r33;
    r119 = r93 + r119;
    r119 = r119 + r96;
    r96 = r40 * r34;
    r116 = fmaf(r40, r116, r31 * r96);
    r116 = r116 + r79;
    r116 = fmaf(r15, r116, r13 * r119);
    r80 = r118 + r80;
    r116 = fmaf(r14, r80, r116);
    r81 = r81 * r116;
    r30 = fmaf(r44, r81, r30);
    r81 = r116 * r49;
    r81 = fmaf(r99, r81, r30 * r100);
    r78 = r33 * r33;
    r78 = r78 * r36;
    r78 = r78 * r36;
    r78 = r78 * r1;
    r78 = r78 * r48;
    r81 = fmaf(r47, r78, r81);
    r80 = r4 * r30;
    r80 = r80 * r99;
    r81 = fmaf(r10, r80, r81);
    r80 = r51 * r51;
    r80 = r80 * r36;
    r80 = r80 * r36;
    r80 = r80 * r1;
    r80 = r80 * r48;
    r78 = r51 * r51;
    r78 = r78 * r30;
    r78 = r78 * r41;
    r78 = r78 * r35;
    r78 = fmaf(r46, r78, r47 * r80);
    r80 = r4 * r30;
    r78 = fmaf(r11, r80, r78);
    r78 = fmaf(r76, r63, r78);
    r80 = r81 + r78;
    r14 = r51 * r51;
    r14 = r14 * r36;
    r14 = r14 * r36;
    r14 = r14 * r89;
    r14 = r14 * r1;
    r14 = r14 * r48;
    r118 = r51 * r51;
    r118 = r118 * r61;
    r118 = r118 * r30;
    r118 = r118 * r41;
    r118 = r118 * r35;
    r118 = fmaf(r46, r118, r55 * r14);
    r14 = r95 * r30;
    r118 = fmaf(r11, r14, r118);
    r15 = r86 * r76;
    r15 = r15 * r46;
    r118 = fmaf(r26, r15, r118);
    r118 = r118 + r81;
    r118 = fmaf(r58, r118, r8 * r80);
    r81 = r7 * r16;
    r81 = r81 * r56;
    r81 = fmaf(r80, r81, r6 * r80);
    r81 = fmaf(r80, r59, r81);
    r81 = fmaf(r80, r66, r81);
    r15 = r81 * r69;
    r118 = fmaf(r26, r15, r118);
    r14 = r57 * r51;
    r14 = r14 * r33;
    r14 = r14 * r36;
    r14 = r14 * r36;
    r14 = r14 * r84;
    r14 = r14 * r1;
    r14 = r14 * r48;
    r118 = fmaf(r55, r14, r118);
    r119 = r51 * r71;
    r119 = r119 * r30;
    r119 = r119 * r41;
    r119 = r119 * r35;
    r118 = fmaf(r69, r119, r118);
    r13 = r51 * r36;
    r13 = r13 * r67;
    r13 = r13 * r74;
    r13 = r13 * r30;
    r13 = r13 * r48;
    r13 = r13 * r60;
    r118 = fmaf(r35, r13, r118);
    r79 = r4 * r51;
    r79 = r79 * r1;
    r79 = r79 * r68;
    r118 = fmaf(r50, r79, r118);
    r96 = r57 * r30;
    r118 = fmaf(r113, r96, r118);
    r31 = r57 * r76;
    r31 = r31 * r49;
    r118 = fmaf(r99, r31, r118);
    r93 = r51 * r36;
    r93 = r93 * r74;
    r93 = r93 * r30;
    r93 = r93 * r48;
    r93 = r93 * r60;
    r118 = fmaf(r35, r93, r118);
    r112 = r4 * r51;
    r112 = r112 * r67;
    r112 = r112 * r1;
    r112 = r112 * r68;
    r118 = fmaf(r50, r112, r118);
    r28 = r36 * r76;
    r118 = fmaf(r69, r28, r118);
    r72 = r57 * r51;
    r72 = r72 * r30;
    r72 = r72 * r49;
    r118 = fmaf(r102, r72, r118);
    r118 = fmaf(r76, r52, r118);
    r118 = fmaf(r30, r107, r118);
    r118 = fmaf(r116, r64, r118);
    r108 = r108 * r2;
    r72 = r61 * r30;
    r28 = r33 * r36;
    r28 = r28 * r86;
    r28 = r28 * r116;
    r28 = fmaf(r46, r28, r100 * r72);
    r72 = r33 * r33;
    r72 = r72 * r36;
    r72 = r72 * r36;
    r72 = r72 * r89;
    r72 = r72 * r1;
    r72 = r72 * r48;
    r28 = fmaf(r55, r72, r28);
    r28 = fmaf(r30, r62, r28);
    r28 = r28 + r78;
    r28 = fmaf(r57, r28, r9 * r80);
    r80 = r71 * r30;
    r80 = r80 * r69;
    r28 = fmaf(r102, r80, r28);
    r78 = r4 * r33;
    r78 = r78 * r67;
    r78 = r78 * r1;
    r78 = r78 * r68;
    r28 = fmaf(r50, r78, r28);
    r72 = r58 * r30;
    r28 = fmaf(r113, r72, r28);
    r112 = r58 * r76;
    r112 = r112 * r49;
    r28 = fmaf(r99, r112, r28);
    r93 = r58 * r116;
    r28 = fmaf(r63, r93, r28);
    r31 = r4 * r33;
    r31 = r31 * r1;
    r31 = r31 * r68;
    r28 = fmaf(r50, r31, r28);
    r96 = r58 * r51;
    r96 = r96 * r30;
    r96 = r96 * r49;
    r28 = fmaf(r102, r96, r28);
    r79 = r36 * r116;
    r28 = fmaf(r69, r79, r28);
    r13 = r33 * r36;
    r13 = r13 * r81;
    r28 = fmaf(r69, r13, r28);
    r119 = r71 * r30;
    r119 = r119 * r102;
    r28 = fmaf(r70, r119, r28);
    r28 = fmaf(r1, r109, r28);
    r28 = fmaf(r30, r104, r28);
    r28 = fmaf(r116, r52, r28);
    r28 = fmaf(r30, r101, r28);
    r108 = fmaf(r28, r65, r118 * r108);
    r119 = r0 * r4;
    r13 = r12 * r51;
    r13 = r13 * r51;
    r13 = r13 * r36;
    r13 = r13 * r36;
    r13 = r13 * r89;
    r13 = r13 * r48;
    r79 = r42 * r86;
    r79 = r79 * r46;
    r79 = fmaf(r26, r79, r55 * r13);
    r13 = r12 * r47;
    r96 = fmaf(r13, r111, r87);
    r31 = r16 * r38;
    r31 = r31 * r33;
    r96 = fmaf(r44, r31, r96);
    r93 = r16 * r42;
    r93 = r93 * r51;
    r96 = fmaf(r44, r93, r96);
    r93 = r95 * r96;
    r79 = fmaf(r11, r93, r79);
    r31 = r51 * r51;
    r31 = r31 * r61;
    r31 = r31 * r96;
    r31 = r31 * r41;
    r31 = r31 * r35;
    r79 = fmaf(r46, r31, r79);
    r112 = r13 * r111;
    r72 = r36 * r36;
    r72 = r72 * r48;
    r78 = r38 * r49;
    r78 = fmaf(r99, r78, r72 * r112);
    r112 = r4 * r96;
    r112 = r112 * r99;
    r78 = fmaf(r10, r112, r78);
    r78 = fmaf(r96, r100, r78);
    r79 = r79 + r78;
    r72 = fmaf(r42, r63, r87 * r72);
    r87 = r4 * r96;
    r72 = fmaf(r11, r87, r72);
    r31 = r51 * r51;
    r31 = r31 * r96;
    r31 = r31 * r41;
    r31 = r31 * r35;
    r72 = fmaf(r46, r31, r72);
    r78 = r78 + r72;
    r79 = fmaf(r8, r78, r58 * r79);
    r31 = r4 * r12;
    r31 = r31 * r51;
    r31 = r31 * r68;
    r79 = fmaf(r50, r31, r79);
    r87 = r42 * r36;
    r79 = fmaf(r69, r87, r79);
    r93 = r51 * r36;
    r93 = r93 * r67;
    r93 = r93 * r74;
    r93 = r93 * r96;
    r93 = r93 * r48;
    r93 = r93 * r60;
    r79 = fmaf(r35, r93, r79);
    r112 = r96 * r113;
    r80 = r57 * r51;
    r80 = r80 * r96;
    r80 = r80 * r49;
    r79 = fmaf(r102, r80, r79);
    r1 = r57 * r42;
    r1 = r1 * r49;
    r79 = fmaf(r99, r1, r79);
    r14 = r51 * r36;
    r14 = r14 * r74;
    r14 = r14 * r96;
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
    r92 = r92 * r96;
    r92 = r92 * r41;
    r92 = r92 * r35;
    r79 = fmaf(r69, r92, r79);
    r27 = r7 * r16;
    r27 = r27 * r56;
    r27 = fmaf(r6, r78, r78 * r27);
    r27 = fmaf(r78, r66, r27);
    r27 = fmaf(r78, r59, r27);
    r106 = r27 * r69;
    r79 = fmaf(r26, r106, r79);
    r120 = r4 * r12;
    r120 = r120 * r51;
    r120 = r120 * r67;
    r120 = r120 * r68;
    r79 = fmaf(r50, r120, r79);
    r79 = fmaf(r57, r112, r79);
    r79 = fmaf(r42, r52, r79);
    r79 = fmaf(r96, r107, r79);
    r79 = fmaf(r38, r64, r79);
    r119 = r119 * r2;
    r120 = r12 * r33;
    r120 = r120 * r33;
    r120 = r120 * r36;
    r120 = r120 * r36;
    r120 = r120 * r89;
    r120 = r120 * r48;
    r106 = r38 * r33;
    r106 = r106 * r36;
    r106 = r106 * r86;
    r106 = fmaf(r46, r106, r55 * r120);
    r120 = r61 * r96;
    r106 = fmaf(r100, r120, r106);
    r106 = fmaf(r96, r62, r106);
    r106 = r106 + r72;
    r78 = fmaf(r9, r78, r57 * r106);
    r106 = r4 * r12;
    r106 = r106 * r33;
    r106 = r106 * r68;
    r78 = fmaf(r50, r106, r78);
    r72 = r4 * r12;
    r72 = r72 * r33;
    r72 = r72 * r67;
    r72 = r72 * r68;
    r78 = fmaf(r50, r72, r78);
    r120 = r38 * r36;
    r78 = fmaf(r69, r120, r78);
    r92 = r58 * r51;
    r92 = r92 * r96;
    r92 = r92 * r49;
    r78 = fmaf(r102, r92, r78);
    r15 = r58 * r42;
    r15 = r15 * r49;
    r78 = fmaf(r99, r15, r78);
    r14 = r71 * r96;
    r14 = r14 * r69;
    r78 = fmaf(r102, r14, r78);
    r1 = r58 * r38;
    r78 = fmaf(r63, r1, r78);
    r80 = r71 * r96;
    r80 = r80 * r102;
    r78 = fmaf(r70, r80, r78);
    r93 = r33 * r36;
    r93 = r93 * r27;
    r78 = fmaf(r69, r93, r78);
    r78 = fmaf(r96, r104, r78);
    r78 = fmaf(r96, r101, r78);
    r78 = fmaf(r38, r52, r78);
    r78 = fmaf(r58, r112, r78);
    r78 = fmaf(r12, r109, r78);
    r119 = fmaf(r78, r65, r79 * r119);
    WriteSum4<float, float>((float*)inout_shared, r3, r115, r108, r119);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r119 = r0 * r4;
    r108 = r39 * r33;
    r108 = r108 * r33;
    r115 = r16 * r43;
    r115 = r115 * r33;
    r115 = fmaf(r44, r115, r47 * r108);
    r108 = r39 * r51;
    r108 = r108 * r51;
    r115 = fmaf(r47, r108, r115);
    r3 = r16 * r54;
    r3 = r3 * r51;
    r115 = fmaf(r44, r3, r115);
    r3 = r39 * r33;
    r3 = r3 * r33;
    r3 = r3 * r36;
    r3 = r3 * r36;
    r3 = r3 * r48;
    r3 = fmaf(r47, r3, r115 * r100);
    r108 = r43 * r49;
    r3 = fmaf(r99, r108, r3);
    r93 = r4 * r115;
    r93 = r93 * r99;
    r3 = fmaf(r10, r93, r3);
    r93 = r51 * r51;
    r93 = r93 * r115;
    r93 = r93 * r41;
    r93 = r93 * r35;
    r108 = r39 * r51;
    r108 = r108 * r51;
    r108 = r108 * r36;
    r108 = r108 * r36;
    r108 = r108 * r48;
    r108 = fmaf(r47, r108, r46 * r93);
    r93 = r4 * r115;
    r108 = fmaf(r11, r93, r108);
    r108 = fmaf(r54, r63, r108);
    r93 = r3 + r108;
    r80 = r51 * r51;
    r80 = r80 * r61;
    r80 = r80 * r115;
    r80 = r80 * r41;
    r80 = r80 * r35;
    r1 = r39 * r51;
    r1 = r1 * r51;
    r1 = r1 * r36;
    r1 = r1 * r36;
    r1 = r1 * r89;
    r1 = r1 * r48;
    r1 = fmaf(r55, r1, r46 * r80);
    r80 = r54 * r86;
    r80 = r80 * r46;
    r1 = fmaf(r26, r80, r1);
    r14 = r95 * r115;
    r1 = fmaf(r11, r14, r1);
    r1 = r1 + r3;
    r1 = fmaf(r58, r1, r8 * r93);
    r3 = r7 * r16;
    r3 = r3 * r56;
    r3 = fmaf(r93, r3, r6 * r93);
    r3 = fmaf(r93, r66, r3);
    r3 = fmaf(r93, r59, r3);
    r14 = r3 * r69;
    r1 = fmaf(r26, r14, r1);
    r80 = r51 * r36;
    r80 = r80 * r74;
    r80 = r80 * r115;
    r80 = r80 * r48;
    r80 = r80 * r60;
    r1 = fmaf(r35, r80, r1);
    r15 = r54 * r36;
    r1 = fmaf(r69, r15, r1);
    r92 = r57 * r51;
    r92 = r92 * r115;
    r92 = r92 * r49;
    r1 = fmaf(r102, r92, r1);
    r112 = r51 * r71;
    r112 = r112 * r115;
    r112 = r112 * r41;
    r112 = r112 * r35;
    r1 = fmaf(r69, r112, r1);
    r120 = r57 * r39;
    r120 = r120 * r51;
    r120 = r120 * r33;
    r120 = r120 * r36;
    r120 = r120 * r36;
    r120 = r120 * r84;
    r120 = r120 * r48;
    r1 = fmaf(r55, r120, r1);
    r72 = r57 * r115;
    r1 = fmaf(r113, r72, r1);
    r106 = r51 * r36;
    r106 = r106 * r67;
    r106 = r106 * r74;
    r106 = r106 * r115;
    r106 = r106 * r48;
    r106 = r106 * r60;
    r1 = fmaf(r35, r106, r1);
    r87 = r4 * r39;
    r87 = r87 * r51;
    r87 = r87 * r68;
    r1 = fmaf(r50, r87, r1);
    r31 = r57 * r54;
    r31 = r31 * r49;
    r1 = fmaf(r99, r31, r1);
    r122 = r4 * r39;
    r122 = r122 * r51;
    r122 = r122 * r67;
    r122 = r122 * r68;
    r1 = fmaf(r50, r122, r1);
    r1 = fmaf(r54, r52, r1);
    r1 = fmaf(r43, r64, r1);
    r1 = fmaf(r115, r107, r1);
    r119 = r119 * r2;
    r122 = r61 * r115;
    r31 = r39 * r33;
    r31 = r31 * r33;
    r31 = r31 * r36;
    r31 = r31 * r36;
    r31 = r31 * r89;
    r31 = r31 * r48;
    r31 = fmaf(r55, r31, r100 * r122);
    r122 = r43 * r33;
    r122 = r122 * r36;
    r122 = r122 * r86;
    r31 = fmaf(r46, r122, r31);
    r31 = fmaf(r115, r62, r31);
    r31 = r31 + r108;
    r31 = fmaf(r57, r31, r9 * r93);
    r93 = r58 * r43;
    r31 = fmaf(r63, r93, r31);
    r108 = r4 * r39;
    r108 = r108 * r33;
    r108 = r108 * r68;
    r31 = fmaf(r50, r108, r31);
    r122 = r71 * r115;
    r122 = r122 * r69;
    r31 = fmaf(r102, r122, r31);
    r87 = r58 * r51;
    r87 = r87 * r115;
    r87 = r87 * r49;
    r31 = fmaf(r102, r87, r31);
    r106 = r71 * r115;
    r106 = r106 * r102;
    r31 = fmaf(r70, r106, r31);
    r72 = r58 * r115;
    r31 = fmaf(r113, r72, r31);
    r120 = r43 * r36;
    r31 = fmaf(r69, r120, r31);
    r112 = r58 * r54;
    r112 = r112 * r49;
    r31 = fmaf(r99, r112, r31);
    r92 = r4 * r39;
    r92 = r92 * r33;
    r92 = r92 * r67;
    r92 = r92 * r68;
    r31 = fmaf(r50, r92, r31);
    r15 = r33 * r36;
    r15 = r15 * r3;
    r31 = fmaf(r69, r15, r31);
    r31 = fmaf(r43, r52, r31);
    r31 = fmaf(r39, r109, r31);
    r31 = fmaf(r115, r101, r31);
    r31 = fmaf(r115, r104, r31);
    r119 = fmaf(r31, r65, r1 * r119);
    r15 = r0 * r4;
    r92 = r16 * r45;
    r92 = r92 * r33;
    r112 = r37 * r33;
    r112 = r112 * r33;
    r112 = fmaf(r47, r112, r44 * r92);
    r92 = r16 * r53;
    r92 = r92 * r51;
    r112 = fmaf(r44, r92, r112);
    r44 = r37 * r51;
    r44 = r44 * r51;
    r112 = fmaf(r47, r44, r112);
    r44 = r45 * r49;
    r44 = fmaf(r99, r44, r112 * r100);
    r92 = r4 * r112;
    r92 = r92 * r99;
    r44 = fmaf(r10, r92, r44);
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
    r92 = r4 * r112;
    r92 = fmaf(r11, r92, r47 * r10);
    r10 = r51 * r51;
    r10 = r10 * r112;
    r10 = r10 * r41;
    r10 = r10 * r35;
    r92 = fmaf(r46, r10, r92);
    r92 = fmaf(r53, r63, r92);
    r10 = r44 + r92;
    r47 = r37 * r51;
    r47 = r47 * r51;
    r47 = r47 * r36;
    r47 = r47 * r36;
    r47 = r47 * r89;
    r47 = r47 * r48;
    r120 = r95 * r112;
    r120 = fmaf(r11, r120, r55 * r47);
    r47 = r53 * r86;
    r47 = r47 * r46;
    r120 = fmaf(r26, r47, r120);
    r11 = r51 * r51;
    r11 = r11 * r61;
    r11 = r11 * r112;
    r11 = r11 * r41;
    r11 = r11 * r35;
    r120 = fmaf(r46, r11, r120);
    r120 = r120 + r44;
    r120 = fmaf(r58, r120, r8 * r10);
    r8 = r4 * r37;
    r8 = r8 * r51;
    r8 = r8 * r67;
    r8 = r8 * r68;
    r120 = fmaf(r50, r8, r120);
    r44 = r53 * r36;
    r120 = fmaf(r69, r44, r120);
    r11 = r4 * r37;
    r11 = r11 * r51;
    r11 = r11 * r68;
    r120 = fmaf(r50, r11, r120);
    r47 = r51 * r36;
    r47 = r47 * r74;
    r47 = r47 * r112;
    r47 = r47 * r48;
    r47 = r47 * r60;
    r120 = fmaf(r35, r47, r120);
    r72 = r57 * r112;
    r120 = fmaf(r113, r72, r120);
    r106 = r57 * r51;
    r106 = r106 * r112;
    r106 = r106 * r49;
    r120 = fmaf(r102, r106, r120);
    r87 = r57 * r53;
    r87 = r87 * r49;
    r120 = fmaf(r99, r87, r120);
    r122 = r51 * r36;
    r122 = r122 * r67;
    r122 = r122 * r74;
    r122 = r122 * r112;
    r122 = r122 * r48;
    r122 = r122 * r60;
    r120 = fmaf(r35, r122, r120);
    r60 = r51 * r71;
    r60 = r60 * r112;
    r60 = r60 * r41;
    r60 = r60 * r35;
    r120 = fmaf(r69, r60, r120);
    r35 = r7 * r16;
    r35 = r35 * r56;
    r6 = fmaf(r6, r10, r10 * r35);
    r6 = fmaf(r10, r66, r6);
    r6 = fmaf(r10, r59, r6);
    r59 = r6 * r69;
    r120 = fmaf(r26, r59, r120);
    r26 = r57 * r37;
    r26 = r26 * r51;
    r26 = r26 * r33;
    r26 = r26 * r36;
    r26 = r26 * r36;
    r26 = r26 * r84;
    r26 = r26 * r48;
    r120 = fmaf(r55, r26, r120);
    r120 = fmaf(r53, r52, r120);
    r120 = fmaf(r112, r107, r120);
    r120 = fmaf(r45, r64, r120);
    r15 = r15 * r2;
    r2 = r61 * r112;
    r26 = r45 * r33;
    r26 = r26 * r36;
    r26 = r26 * r86;
    r26 = fmaf(r46, r26, r100 * r2);
    r2 = r37 * r33;
    r2 = r2 * r33;
    r2 = r2 * r36;
    r2 = r2 * r36;
    r2 = r2 * r89;
    r2 = r2 * r48;
    r26 = fmaf(r55, r2, r26);
    r26 = fmaf(r112, r62, r26);
    r26 = r26 + r92;
    r26 = fmaf(r57, r26, r9 * r10);
    r10 = r71 * r112;
    r10 = r10 * r102;
    r26 = fmaf(r70, r10, r26);
    r70 = r71 * r112;
    r70 = r70 * r69;
    r26 = fmaf(r102, r70, r26);
    r9 = r58 * r112;
    r26 = fmaf(r113, r9, r26);
    r113 = r58 * r51;
    r113 = r113 * r112;
    r113 = r113 * r49;
    r26 = fmaf(r102, r113, r26);
    r102 = r58 * r45;
    r26 = fmaf(r63, r102, r26);
    r63 = r4 * r37;
    r63 = r63 * r33;
    r63 = r63 * r67;
    r63 = r63 * r68;
    r26 = fmaf(r50, r63, r26);
    r67 = r58 * r53;
    r67 = r67 * r49;
    r26 = fmaf(r99, r67, r26);
    r99 = r45 * r36;
    r26 = fmaf(r69, r99, r26);
    r92 = r4 * r37;
    r92 = r92 * r33;
    r92 = r92 * r68;
    r26 = fmaf(r50, r92, r26);
    r50 = r33 * r36;
    r50 = r50 * r6;
    r26 = fmaf(r69, r50, r26);
    r26 = fmaf(r45, r52, r26);
    r26 = fmaf(r112, r101, r26);
    r26 = fmaf(r112, r104, r26);
    r26 = fmaf(r37, r109, r26);
    r65 = fmaf(r26, r65, r120 * r15);
    WriteSum2<float, float>((float*)inout_shared, r119, r65);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r0 * r0;
    r119 = r88 * r65;
    r5 = r5 * r5;
    r15 = r98 * r5;
    r98 = fmaf(r98, r15, r88 * r119);
    r88 = r94 * r94;
    r109 = r124 * r124;
    r109 = fmaf(r5, r109, r65 * r88);
    r88 = r28 * r28;
    r50 = r118 * r118;
    r50 = fmaf(r65, r50, r5 * r88);
    r88 = r79 * r79;
    r92 = r78 * r78;
    r92 = fmaf(r5, r92, r65 * r88);
    WriteSum4<float, float>((float*)inout_shared, r98, r109, r50, r92);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r92 = r1 * r1;
    r50 = r31 * r31;
    r50 = fmaf(r5, r50, r65 * r92);
    r92 = r26 * r26;
    r109 = r120 * r120;
    r109 = fmaf(r65, r109, r5 * r92);
    WriteSum2<float, float>((float*)inout_shared, r50, r109);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r109 = fmaf(r94, r119, r124 * r15);
    r50 = fmaf(r28, r15, r118 * r119);
    r92 = fmaf(r79, r119, r78 * r15);
    r98 = fmaf(r1, r119, r31 * r15);
    WriteSum4<float, float>((float*)inout_shared, r109, r50, r92, r98);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r26, r15, r120 * r119);
    r119 = r124 * r28;
    r98 = r94 * r118;
    r98 = fmaf(r65, r98, r5 * r119);
    r119 = r94 * r79;
    r92 = r124 * r78;
    r92 = fmaf(r5, r92, r65 * r119);
    r119 = r94 * r1;
    r50 = r124 * r31;
    r50 = fmaf(r5, r50, r65 * r119);
    WriteSum4<float, float>((float*)inout_shared, r15, r98, r92, r50);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r124 * r26;
    r92 = r94 * r120;
    r92 = fmaf(r65, r92, r5 * r50);
    r50 = r28 * r78;
    r98 = r118 * r79;
    r98 = fmaf(r65, r98, r5 * r50);
    r50 = r118 * r1;
    r15 = r28 * r31;
    r15 = fmaf(r5, r15, r65 * r50);
    r50 = r28 * r26;
    r119 = r118 * r120;
    r119 = fmaf(r65, r119, r5 * r50);
    WriteSum4<float, float>((float*)inout_shared, r92, r98, r15, r119);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r119 = r78 * r31;
    r15 = r79 * r1;
    r15 = fmaf(r65, r15, r5 * r119);
    r119 = r79 * r120;
    r98 = r78 * r26;
    r98 = fmaf(r5, r98, r65 * r119);
    r119 = r31 * r26;
    r92 = r1 * r120;
    r92 = fmaf(r65, r92, r5 * r119);
    WriteSum3<float, float>((float*)inout_shared, r15, r98, r92);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacFirst(
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
  ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacFirstKernel<<<
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