#include "kernel_thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedFocalAndExtraFixedPointResJacKernel(
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
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104,
      r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116,
      r117, r118, r119, r120, r121, r122, r123, r124, r125, r126;
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
    r47 = 9.99999999999999955e-07;
    r51 = r40 * r29;
    r51 = r51 * r29;
    r52 = r41 + r51;
    r52 = r52 + r48;
    r52 = fmaf(r13, r52, r10);
    r10 = r32 * r40;
    r10 = fmaf(r34, r10, r26);
    r26 = r16 * r32;
    r26 = r26 * r25;
    r48 = r16 * r29;
    r48 = fmaf(r34, r48, r26);
    r53 = r21 * r23;
    r53 = r53 * r16;
    r54 = r22 * r24;
    r54 = fmaf(r16, r54, r53);
    r55 = r23 * r24;
    r55 = fmaf(r40, r55, r37);
    r37 = r22 * r22;
    r37 = r37 * r40;
    r42 = r37 + r42;
    r52 = fmaf(r14, r10, r52);
    r52 = fmaf(r15, r48, r52);
    r52 = fmaf(r36, r54, r52);
    r52 = fmaf(r35, r55, r52);
    r52 = fmaf(r11, r42, r52);
    r48 = r52 * r52;
    r10 = r40 * r29;
    r10 = fmaf(r34, r10, r26);
    r10 = fmaf(r13, r10, r12);
    r12 = r22 * r24;
    r12 = fmaf(r40, r12, r53);
    r37 = r41 + r37;
    r37 = r37 + r39;
    r39 = r21 * r24;
    r39 = fmaf(r16, r39, r44);
    r44 = r16 * r25;
    r44 = fmaf(r34, r44, r46);
    r51 = r41 + r51;
    r51 = r51 + r50;
    r10 = fmaf(r11, r12, r10);
    r10 = fmaf(r36, r37, r10);
    r10 = fmaf(r35, r39, r10);
    r10 = fmaf(r14, r44, r10);
    r10 = fmaf(r15, r51, r10);
    r51 = copysign(1.0, r10);
    r51 = fmaf(r47, r51, r10);
    r10 = r51 * r51;
    r44 = 1.0 / r10;
    r35 = r33 * r33;
    r35 = fmaf(r44, r35, r44 * r48);
    r48 = sqrtf(r35);
    r36 = copysign(1.0, r48);
    r36 = fmaf(r47, r36, r48);
    r47 = r36 * r36;
    r11 = 1.0 / r47;
    r48 = atanf(r48);
    r50 = r48 * r44;
    r46 = r48 * r50;
    r49 = r49 * r11;
    r49 = r49 * r46;
    r53 = r52 * r11;
    r26 = r52 * r53;
    r56 = r46 * r26;
    r57 = r49 + r56;
    ReadIdx4<1024, float, float, float4>(focal_and_extra,
                                         4 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r58,
                                         r59,
                                         r60,
                                         r61);
    r62 = 3.00000000000000000e+00;
    r63 = r62 * r46;
    r63 = fmaf(r26, r63, r49);
    r63 = fmaf(r59, r63, r8 * r57);
    r49 = r58 * r33;
    r64 = r16 * r46;
    r49 = r49 * r53;
    r63 = fmaf(r64, r49, r63);
    r65 = r57 * r57;
    r66 = r57 * r65;
    r67 = fmaf(r60, r66, r6 * r57);
    r66 = r61 * r66;
    r67 = fmaf(r57, r66, r67);
    r67 = fmaf(r7, r65, r67);
    r61 = 1.0 / r51;
    r68 = 1.0 / r36;
    r69 = r61 * r68;
    r70 = r48 * r69;
    r71 = r67 * r70;
    r63 = fmaf(r52, r71, r63);
    r63 = fmaf(r52, r70, r63);
    r2 = fmaf(r0, r63, r2);
    r63 = r33 * r33;
    r63 = r63 * r62;
    r63 = r63 * r11;
    r63 = fmaf(r46, r63, r56);
    r63 = fmaf(r58, r63, r9 * r57);
    r56 = r59 * r33;
    r56 = r56 * r53;
    r63 = fmaf(r64, r56, r63);
    r63 = fmaf(r33, r71, r63);
    r63 = fmaf(r33, r70, r63);
    r63 = fmaf(r5, r63, r1);
    r63 = fmaf(r3, r4, r63);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r63);
    r3 = r52 * r11;
    r1 = -5.00000000000000000e-01;
    r56 = rsqrtf(r35);
    r49 = r16 * r33;
    r72 = r25 * r40;
    r73 = r17 * r24;
    r74 = r20 * r21;
    r74 = fmaf(r1, r74, r1 * r73);
    r73 = r19 * r22;
    r74 = fmaf(r1, r73, r74);
    r75 = r18 * r23;
    r76 = 5.00000000000000000e-01;
    r74 = fmaf(r76, r75, r74);
    r75 = r20 * r24;
    r73 = r17 * r21;
    r73 = fmaf(r1, r73, r76 * r75);
    r75 = r18 * r22;
    r73 = fmaf(r1, r75, r73);
    r77 = r19 * r23;
    r73 = fmaf(r1, r77, r73);
    r77 = r34 * r73;
    r75 = r40 * r77;
    r72 = fmaf(r74, r72, r75);
    r78 = r16 * r29;
    r79 = fmaf(r76, r31, r1 * r27);
    r79 = fmaf(r1, r28, r79);
    r79 = fmaf(r1, r30, r79);
    r78 = r78 * r79;
    r80 = r16 * r32;
    r81 = r19 * r24;
    r82 = r18 * r21;
    r82 = fmaf(r76, r82, r76 * r81);
    r81 = r17 * r22;
    r82 = fmaf(r1, r81, r82);
    r83 = r20 * r23;
    r82 = fmaf(r76, r83, r82);
    r80 = fmaf(r82, r80, r78);
    r72 = r72 + r80;
    r83 = r25 * r73;
    r81 = -4.00000000000000000e+00;
    r83 = r83 * r81;
    r84 = r32 * r79;
    r85 = r81 * r84;
    r86 = r83 + r85;
    r86 = fmaf(r14, r86, r15 * r72);
    r72 = r16 * r25;
    r72 = r72 * r82;
    r87 = r16 * r34;
    r87 = fmaf(r79, r87, r72);
    r88 = r16 * r29;
    r88 = r88 * r73;
    r89 = r16 * r32;
    r89 = fmaf(r74, r89, r88);
    r87 = r87 + r89;
    r86 = fmaf(r13, r87, r86);
    r49 = r49 * r86;
    r87 = r16 * r52;
    r90 = r16 * r34;
    r91 = r29 * r74;
    r90 = fmaf(r16, r91, r82 * r90);
    r92 = r16 * r25;
    r93 = r16 * r32;
    r93 = r93 * r73;
    r92 = fmaf(r79, r92, r93);
    r90 = r90 + r92;
    r72 = r88 + r72;
    r88 = r32 * r40;
    r72 = fmaf(r74, r88, r72);
    r94 = r40 * r34;
    r72 = fmaf(r79, r94, r72);
    r72 = fmaf(r14, r72, r15 * r90);
    r90 = r29 * r82;
    r90 = r90 * r81;
    r85 = r90 + r85;
    r72 = fmaf(r13, r85, r72);
    r87 = r87 * r72;
    r87 = fmaf(r44, r87, r44 * r49);
    r49 = r16 * r25;
    r49 = r49 * r74;
    r77 = r16 * r77;
    r85 = r49 + r77;
    r80 = r80 + r85;
    r94 = r40 * r34;
    r94 = fmaf(r40, r91, r82 * r94);
    r94 = r94 + r92;
    r94 = fmaf(r13, r94, r14 * r80);
    r90 = r83 + r90;
    r94 = fmaf(r15, r90, r94);
    r90 = r33 * r33;
    r10 = r51 * r10;
    r10 = 1.0 / r10;
    r51 = r40 * r10;
    r90 = r90 * r51;
    r83 = r52 * r52;
    r83 = r83 * r94;
    r87 = fmaf(r51, r83, r87);
    r87 = fmaf(r94, r90, r87);
    r3 = r3 * r48;
    r3 = r3 * r61;
    r3 = r3 * r1;
    r3 = r3 * r56;
    r3 = r3 * r87;
    r83 = 6.00000000000000000e+00;
    r80 = r72 * r83;
    r80 = r80 * r46;
    r82 = r62 * r87;
    r35 = r41 + r35;
    r35 = 1.0 / r35;
    r88 = r35 * r50;
    r82 = r82 * r56;
    r82 = r82 * r88;
    r82 = fmaf(r26, r82, r53 * r80);
    r80 = -6.00000000000000000e+00;
    r95 = r80 * r10;
    r96 = r48 * r48;
    r97 = r26 * r96;
    r95 = r95 * r97;
    r98 = r52 * r52;
    r99 = -3.00000000000000000e+00;
    r47 = r36 * r47;
    r47 = 1.0 / r47;
    r98 = r98 * r99;
    r98 = r98 * r87;
    r98 = r98 * r56;
    r98 = r98 * r47;
    r82 = fmaf(r46, r98, r82);
    r36 = r33 * r11;
    r100 = r64 * r36;
    r101 = r33 * r56;
    r102 = r87 * r101;
    r102 = r102 * r88;
    r102 = fmaf(r36, r102, r86 * r100);
    r103 = r4 * r33;
    r104 = r47 * r46;
    r104 = r104 * r101;
    r103 = r103 * r87;
    r102 = fmaf(r104, r103, r102);
    r96 = r11 * r96;
    r96 = r96 * r90;
    r102 = fmaf(r94, r96, r102);
    r82 = fmaf(r94, r95, r82);
    r82 = r82 + r102;
    r82 = fmaf(r59, r82, r3);
    r98 = r72 * r53;
    r103 = r87 * r56;
    r103 = r103 * r88;
    r103 = fmaf(r26, r103, r64 * r98);
    r98 = r94 * r51;
    r103 = fmaf(r97, r98, r103);
    r105 = r4 * r52;
    r105 = r105 * r52;
    r105 = r105 * r87;
    r105 = r105 * r56;
    r105 = r105 * r47;
    r103 = fmaf(r46, r105, r103);
    r102 = r102 + r103;
    r105 = r58 * r87;
    r98 = r16 * r53;
    r98 = r98 * r101;
    r98 = r98 * r88;
    r82 = fmaf(r98, r105, r82);
    r106 = r58 * r33;
    r106 = r106 * r48;
    r106 = r106 * r48;
    r106 = r106 * r81;
    r106 = r106 * r10;
    r106 = r106 * r53;
    r107 = r52 * r76;
    r107 = r107 * r56;
    r107 = r107 * r35;
    r107 = r107 * r69;
    r108 = r67 * r107;
    r109 = r58 * r72;
    r82 = fmaf(r100, r109, r82);
    r110 = r4 * r52;
    r110 = r110 * r94;
    r110 = r110 * r68;
    r82 = fmaf(r50, r110, r82);
    r111 = r4 * r52;
    r111 = r111 * r67;
    r111 = r111 * r94;
    r111 = r111 * r68;
    r82 = fmaf(r50, r111, r82);
    r112 = r58 * r40;
    r112 = r112 * r52;
    r112 = r112 * r87;
    r82 = fmaf(r104, r112, r82);
    r113 = r7 * r16;
    r113 = r113 * r57;
    r113 = fmaf(r102, r113, r6 * r102);
    r114 = 4.00000000000000000e+00;
    r66 = r114 * r66;
    r60 = r60 * r62;
    r60 = r60 * r65;
    r113 = fmaf(r102, r66, r113);
    r113 = fmaf(r102, r60, r113);
    r65 = r52 * r113;
    r82 = fmaf(r70, r65, r82);
    r114 = r58 * r86;
    r114 = r114 * r53;
    r82 = fmaf(r64, r114, r82);
    r82 = fmaf(r8, r102, r82);
    r82 = fmaf(r67, r3, r82);
    r82 = fmaf(r94, r106, r82);
    r82 = fmaf(r87, r108, r82);
    r82 = fmaf(r72, r70, r82);
    r82 = fmaf(r87, r107, r82);
    r82 = fmaf(r72, r71, r82);
    r114 = r0 * r82;
    r65 = r33 * r86;
    r65 = r65 * r83;
    r65 = r65 * r11;
    r112 = r62 * r87;
    r112 = r112 * r101;
    r112 = r112 * r88;
    r112 = fmaf(r36, r112, r46 * r65);
    r65 = r33 * r99;
    r65 = r65 * r87;
    r112 = fmaf(r104, r65, r112);
    r111 = r33 * r33;
    r111 = r111 * r48;
    r111 = r111 * r48;
    r111 = r111 * r94;
    r111 = r111 * r80;
    r111 = r111 * r11;
    r112 = fmaf(r10, r111, r112);
    r112 = r112 + r103;
    r102 = fmaf(r9, r102, r58 * r112);
    r112 = r33 * r11;
    r112 = r112 * r48;
    r112 = r112 * r61;
    r112 = r112 * r1;
    r112 = r112 * r56;
    r112 = r112 * r87;
    r103 = r76 * r87;
    r103 = r103 * r35;
    r103 = r103 * r101;
    r102 = fmaf(r69, r103, r102);
    r111 = r33 * r113;
    r102 = fmaf(r70, r111, r102);
    r65 = r4 * r33;
    r65 = r65 * r67;
    r65 = r65 * r94;
    r65 = r65 * r68;
    r102 = fmaf(r50, r65, r102);
    r110 = r59 * r33;
    r110 = r110 * r48;
    r110 = r110 * r48;
    r110 = r110 * r81;
    r110 = r110 * r94;
    r110 = r110 * r10;
    r102 = fmaf(r53, r110, r102);
    r109 = r59 * r72;
    r102 = fmaf(r100, r109, r102);
    r105 = r59 * r87;
    r102 = fmaf(r98, r105, r102);
    r3 = r4 * r33;
    r3 = r3 * r94;
    r3 = r3 * r68;
    r102 = fmaf(r50, r3, r102);
    r115 = r59 * r40;
    r115 = r115 * r52;
    r115 = r115 * r104;
    r116 = r76 * r67;
    r116 = r116 * r87;
    r116 = r116 * r35;
    r116 = r116 * r101;
    r102 = fmaf(r69, r116, r102);
    r117 = r59 * r86;
    r117 = r117 * r53;
    r102 = fmaf(r64, r117, r102);
    r102 = r102 + r112;
    r102 = fmaf(r67, r112, r102);
    r102 = fmaf(r87, r115, r102);
    r102 = fmaf(r86, r71, r102);
    r102 = fmaf(r86, r70, r102);
    r117 = r5 * r102;
    r116 = r16 * r52;
    r77 = r78 + r77;
    r78 = r16 * r32;
    r3 = r19 * r24;
    r105 = r18 * r21;
    r105 = fmaf(r1, r105, r1 * r3);
    r3 = r17 * r22;
    r105 = fmaf(r76, r3, r105);
    r109 = r20 * r23;
    r105 = fmaf(r1, r109, r105);
    r78 = r78 * r105;
    r109 = r16 * r25;
    r3 = r17 * r24;
    r110 = r20 * r21;
    r110 = fmaf(r76, r110, r76 * r3);
    r3 = r19 * r22;
    r110 = fmaf(r76, r3, r110);
    r65 = r18 * r23;
    r110 = fmaf(r1, r65, r110);
    r109 = fmaf(r110, r109, r78);
    r77 = r77 + r109;
    r65 = r32 * r81;
    r65 = r65 * r110;
    r3 = r29 * r73;
    r3 = r3 * r81;
    r111 = r65 + r3;
    r111 = fmaf(r13, r111, r15 * r77);
    r77 = r40 * r34;
    r77 = fmaf(r40, r84, r110 * r77);
    r112 = r16 * r25;
    r112 = r112 * r73;
    r103 = r16 * r29;
    r103 = fmaf(r105, r103, r112);
    r77 = r77 + r103;
    r111 = fmaf(r14, r77, r111);
    r116 = r116 * r111;
    r77 = r52 * r52;
    r118 = r40 * r29;
    r118 = fmaf(r79, r118, r75);
    r118 = r118 + r109;
    r109 = r16 * r29;
    r109 = r109 * r110;
    r119 = r16 * r34;
    r119 = fmaf(r105, r119, r109);
    r119 = r119 + r92;
    r119 = fmaf(r14, r119, r13 * r118);
    r118 = r25 * r105;
    r92 = r81 * r118;
    r3 = r3 + r92;
    r119 = fmaf(r15, r3, r119);
    r77 = r77 * r119;
    r77 = fmaf(r51, r77, r44 * r116);
    r116 = r16 * r33;
    r3 = r25 * r40;
    r3 = fmaf(r79, r3, r93);
    r93 = r40 * r34;
    r3 = fmaf(r105, r93, r3);
    r3 = r3 + r109;
    r93 = r16 * r34;
    r84 = fmaf(r16, r84, r110 * r93);
    r84 = r84 + r103;
    r84 = fmaf(r13, r84, r15 * r3);
    r92 = r65 + r92;
    r84 = fmaf(r14, r92, r84);
    r116 = r116 * r84;
    r77 = fmaf(r44, r116, r77);
    r77 = fmaf(r119, r90, r77);
    r116 = r77 * r56;
    r116 = r116 * r88;
    r92 = r4 * r52;
    r92 = r92 * r52;
    r92 = r92 * r77;
    r92 = r92 * r56;
    r92 = r92 * r47;
    r92 = fmaf(r46, r92, r26 * r116);
    r116 = r119 * r51;
    r92 = fmaf(r97, r116, r92);
    r65 = r111 * r53;
    r92 = fmaf(r64, r65, r92);
    r65 = r4 * r33;
    r65 = r65 * r77;
    r65 = fmaf(r104, r65, r84 * r100);
    r116 = r77 * r101;
    r116 = r116 * r88;
    r65 = fmaf(r36, r116, r65);
    r65 = fmaf(r119, r96, r65);
    r116 = r92 + r65;
    r3 = r62 * r77;
    r3 = r3 * r56;
    r3 = r3 * r88;
    r93 = r52 * r52;
    r110 = r99 * r77;
    r93 = r93 * r56;
    r93 = r93 * r47;
    r93 = r93 * r46;
    r93 = fmaf(r110, r93, r26 * r3);
    r3 = r83 * r111;
    r3 = r3 * r46;
    r93 = fmaf(r53, r3, r93);
    r93 = fmaf(r119, r95, r93);
    r93 = r93 + r65;
    r93 = fmaf(r59, r93, r8 * r116);
    r65 = r58 * r84;
    r65 = r65 * r53;
    r93 = fmaf(r64, r65, r93);
    r3 = r48 * r1;
    r3 = r3 * r77;
    r3 = r3 * r61;
    r3 = r3 * r56;
    r93 = fmaf(r53, r3, r93);
    r109 = r4 * r52;
    r109 = r109 * r119;
    r109 = r109 * r68;
    r93 = fmaf(r50, r109, r93);
    r79 = r58 * r40;
    r79 = r79 * r52;
    r79 = r79 * r77;
    r93 = fmaf(r104, r79, r93);
    r120 = r58 * r77;
    r93 = fmaf(r98, r120, r93);
    r121 = r4 * r52;
    r121 = r121 * r67;
    r121 = r121 * r119;
    r121 = r121 * r68;
    r93 = fmaf(r50, r121, r93);
    r122 = r111 * r100;
    r123 = r7 * r16;
    r123 = r123 * r57;
    r123 = fmaf(r116, r123, r6 * r116);
    r123 = fmaf(r116, r66, r123);
    r123 = fmaf(r116, r60, r123);
    r124 = r52 * r123;
    r93 = fmaf(r70, r124, r93);
    r125 = r48 * r67;
    r125 = r125 * r1;
    r125 = r125 * r77;
    r125 = r125 * r61;
    r125 = r125 * r56;
    r93 = fmaf(r53, r125, r93);
    r93 = fmaf(r119, r106, r93);
    r93 = fmaf(r77, r108, r93);
    r93 = fmaf(r111, r71, r93);
    r93 = fmaf(r77, r107, r93);
    r93 = fmaf(r58, r122, r93);
    r93 = fmaf(r111, r70, r93);
    r125 = r0 * r93;
    r124 = r33 * r83;
    r124 = r124 * r84;
    r124 = r124 * r11;
    r121 = r33 * r104;
    r121 = fmaf(r110, r121, r46 * r124);
    r124 = r33 * r33;
    r124 = r124 * r48;
    r124 = r124 * r48;
    r124 = r124 * r80;
    r124 = r124 * r119;
    r124 = r124 * r11;
    r121 = fmaf(r10, r124, r121);
    r110 = r62 * r77;
    r110 = r110 * r101;
    r110 = r110 * r88;
    r121 = fmaf(r36, r110, r121);
    r121 = r121 + r92;
    r121 = fmaf(r58, r121, r9 * r116);
    r116 = r4 * r33;
    r116 = r116 * r119;
    r116 = r116 * r68;
    r121 = fmaf(r50, r116, r121);
    r92 = r59 * r84;
    r92 = r92 * r53;
    r121 = fmaf(r64, r92, r121);
    r110 = r59 * r33;
    r110 = r110 * r48;
    r110 = r110 * r48;
    r110 = r110 * r81;
    r110 = r110 * r119;
    r110 = r110 * r10;
    r121 = fmaf(r53, r110, r121);
    r124 = r76 * r67;
    r124 = r124 * r77;
    r124 = r124 * r35;
    r124 = r124 * r101;
    r121 = fmaf(r69, r124, r121);
    r120 = r48 * r67;
    r120 = r120 * r1;
    r120 = r120 * r77;
    r120 = r120 * r11;
    r120 = r120 * r61;
    r121 = fmaf(r101, r120, r121);
    r79 = r33 * r123;
    r121 = fmaf(r70, r79, r121);
    r109 = r76 * r77;
    r109 = r109 * r35;
    r109 = r109 * r101;
    r121 = fmaf(r69, r109, r121);
    r3 = r48 * r1;
    r3 = r3 * r77;
    r3 = r3 * r11;
    r3 = r3 * r61;
    r121 = fmaf(r101, r3, r121);
    r65 = r59 * r77;
    r121 = fmaf(r98, r65, r121);
    r126 = r4 * r33;
    r126 = r126 * r67;
    r126 = r126 * r119;
    r126 = r126 * r68;
    r121 = fmaf(r50, r126, r121);
    r121 = fmaf(r77, r115, r121);
    r121 = fmaf(r59, r122, r121);
    r121 = fmaf(r84, r71, r121);
    r121 = fmaf(r84, r70, r121);
    r126 = r5 * r121;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r114,
                                          r117,
                                          r125,
                                          r126);
    r126 = r52 * r52;
    r125 = r25 * r81;
    r31 = fmaf(r1, r31, r76 * r27);
    r31 = fmaf(r76, r28, r31);
    r31 = fmaf(r76, r30, r31);
    r125 = r125 * r31;
    r91 = r81 * r91;
    r30 = r125 + r91;
    r28 = r16 * r32;
    r28 = r28 * r31;
    r112 = r112 + r28;
    r27 = r40 * r29;
    r112 = fmaf(r105, r27, r112);
    r117 = r40 * r34;
    r112 = fmaf(r74, r117, r112);
    r112 = fmaf(r13, r112, r15 * r30);
    r30 = r16 * r34;
    r30 = fmaf(r16, r118, r31 * r30);
    r30 = r30 + r89;
    r112 = fmaf(r14, r30, r112);
    r126 = r126 * r112;
    r30 = r16 * r52;
    r117 = r16 * r29;
    r117 = r117 * r31;
    r49 = r49 + r117;
    r27 = r32 * r40;
    r49 = fmaf(r105, r27, r49);
    r49 = r49 + r75;
    r73 = r32 * r73;
    r73 = r73 * r81;
    r91 = r73 + r91;
    r91 = fmaf(r13, r91, r14 * r49);
    r49 = r16 * r34;
    r49 = fmaf(r74, r49, r28);
    r49 = r49 + r103;
    r91 = fmaf(r15, r49, r91);
    r30 = r30 * r91;
    r30 = fmaf(r44, r30, r51 * r126);
    r126 = r16 * r33;
    r117 = r78 + r117;
    r117 = r117 + r85;
    r85 = r40 * r34;
    r118 = fmaf(r40, r118, r31 * r85);
    r118 = r118 + r89;
    r118 = fmaf(r15, r118, r13 * r117);
    r73 = r125 + r73;
    r118 = fmaf(r14, r73, r118);
    r126 = r126 * r118;
    r30 = fmaf(r44, r126, r30);
    r30 = fmaf(r112, r90, r30);
    r126 = r30 * r101;
    r126 = r126 * r88;
    r126 = fmaf(r118, r100, r36 * r126);
    r73 = r4 * r33;
    r73 = r73 * r30;
    r126 = fmaf(r104, r73, r126);
    r126 = fmaf(r112, r96, r126);
    r73 = r112 * r51;
    r14 = r30 * r56;
    r14 = r14 * r88;
    r14 = fmaf(r26, r14, r97 * r73);
    r73 = r4 * r52;
    r73 = r73 * r52;
    r73 = r73 * r30;
    r73 = r73 * r56;
    r73 = r73 * r47;
    r14 = fmaf(r46, r73, r14);
    r125 = r91 * r53;
    r14 = fmaf(r64, r125, r14);
    r125 = r126 + r14;
    r73 = r62 * r30;
    r73 = r73 * r56;
    r73 = r73 * r88;
    r73 = fmaf(r26, r73, r112 * r95);
    r15 = r52 * r52;
    r15 = r15 * r99;
    r15 = r15 * r30;
    r15 = r15 * r56;
    r15 = r15 * r47;
    r73 = fmaf(r46, r15, r73);
    r117 = r83 * r91;
    r117 = r117 * r46;
    r73 = fmaf(r53, r117, r73);
    r73 = r73 + r126;
    r73 = fmaf(r59, r73, r8 * r125);
    r126 = r7 * r16;
    r126 = r126 * r57;
    r126 = fmaf(r125, r126, r6 * r125);
    r126 = fmaf(r125, r60, r126);
    r126 = fmaf(r125, r66, r126);
    r117 = r52 * r126;
    r73 = fmaf(r70, r117, r73);
    r15 = r48 * r67;
    r15 = r15 * r1;
    r15 = r15 * r30;
    r15 = r15 * r61;
    r15 = r15 * r56;
    r73 = fmaf(r53, r15, r73);
    r13 = r4 * r52;
    r13 = r13 * r112;
    r13 = r13 * r68;
    r73 = fmaf(r50, r13, r73);
    r89 = r58 * r40;
    r89 = r89 * r52;
    r89 = r89 * r30;
    r73 = fmaf(r104, r89, r73);
    r85 = r58 * r91;
    r73 = fmaf(r100, r85, r73);
    r31 = r58 * r118;
    r31 = r31 * r53;
    r73 = fmaf(r64, r31, r73);
    r78 = r48 * r1;
    r78 = r78 * r30;
    r78 = r78 * r61;
    r78 = r78 * r56;
    r73 = fmaf(r53, r78, r73);
    r49 = r4 * r52;
    r49 = r49 * r67;
    r49 = r49 * r112;
    r49 = r49 * r68;
    r73 = fmaf(r50, r49, r73);
    r103 = r58 * r30;
    r73 = fmaf(r98, r103, r73);
    r73 = fmaf(r112, r106, r73);
    r73 = fmaf(r30, r107, r73);
    r73 = fmaf(r91, r71, r73);
    r73 = fmaf(r30, r108, r73);
    r73 = fmaf(r91, r70, r73);
    r103 = r0 * r73;
    r49 = r62 * r30;
    r49 = r49 * r101;
    r49 = r49 * r88;
    r78 = r33 * r83;
    r78 = r78 * r118;
    r78 = r78 * r11;
    r78 = fmaf(r46, r78, r36 * r49);
    r49 = r33 * r33;
    r49 = r49 * r48;
    r49 = r49 * r48;
    r49 = r49 * r80;
    r49 = r49 * r112;
    r49 = r49 * r11;
    r78 = fmaf(r10, r49, r78);
    r31 = r33 * r99;
    r31 = r31 * r30;
    r78 = fmaf(r104, r31, r78);
    r78 = r78 + r14;
    r78 = fmaf(r58, r78, r9 * r125);
    r125 = r59 * r33;
    r125 = r125 * r48;
    r125 = r125 * r48;
    r125 = r125 * r81;
    r125 = r125 * r112;
    r125 = r125 * r10;
    r78 = fmaf(r53, r125, r78);
    r14 = r48 * r1;
    r14 = r14 * r30;
    r14 = r14 * r11;
    r14 = r14 * r61;
    r78 = fmaf(r101, r14, r78);
    r31 = r76 * r30;
    r31 = r31 * r35;
    r31 = r31 * r101;
    r78 = fmaf(r69, r31, r78);
    r49 = r4 * r33;
    r49 = r49 * r67;
    r49 = r49 * r112;
    r49 = r49 * r68;
    r78 = fmaf(r50, r49, r78);
    r85 = r59 * r91;
    r78 = fmaf(r100, r85, r78);
    r89 = r48 * r67;
    r89 = r89 * r1;
    r89 = r89 * r30;
    r89 = r89 * r11;
    r89 = r89 * r61;
    r78 = fmaf(r101, r89, r78);
    r13 = r59 * r118;
    r13 = r13 * r53;
    r78 = fmaf(r64, r13, r78);
    r15 = r4 * r33;
    r15 = r15 * r112;
    r15 = r15 * r68;
    r78 = fmaf(r50, r15, r78);
    r117 = r59 * r30;
    r78 = fmaf(r98, r117, r78);
    r28 = r33 * r126;
    r78 = fmaf(r70, r28, r78);
    r74 = r76 * r67;
    r74 = r74 * r30;
    r74 = r74 * r35;
    r74 = r74 * r101;
    r78 = fmaf(r69, r74, r78);
    r78 = fmaf(r118, r71, r78);
    r78 = fmaf(r30, r115, r78);
    r78 = fmaf(r118, r70, r78);
    r74 = r5 * r78;
    r28 = r42 * r83;
    r28 = r28 * r46;
    r28 = fmaf(r53, r28, r12 * r95);
    r117 = r52 * r52;
    r15 = r16 * r38;
    r15 = r15 * r33;
    r15 = fmaf(r44, r15, r12 * r90);
    r13 = r12 * r52;
    r13 = r13 * r52;
    r15 = fmaf(r51, r13, r15);
    r89 = r16 * r42;
    r89 = r89 * r52;
    r15 = fmaf(r44, r89, r15);
    r117 = r117 * r99;
    r117 = r117 * r15;
    r117 = r117 * r56;
    r117 = r117 * r47;
    r28 = fmaf(r46, r117, r28);
    r89 = r62 * r15;
    r89 = r89 * r56;
    r89 = r89 * r88;
    r28 = fmaf(r26, r89, r28);
    r13 = fmaf(r38, r100, r12 * r96);
    r85 = r4 * r33;
    r85 = r85 * r15;
    r13 = fmaf(r104, r85, r13);
    r49 = r15 * r101;
    r49 = r49 * r88;
    r13 = fmaf(r36, r49, r13);
    r28 = r28 + r13;
    r89 = r12 * r51;
    r117 = r42 * r53;
    r117 = fmaf(r64, r117, r97 * r89);
    r89 = r4 * r52;
    r89 = r89 * r52;
    r89 = r89 * r15;
    r89 = r89 * r56;
    r89 = r89 * r47;
    r117 = fmaf(r46, r89, r117);
    r49 = r15 * r56;
    r49 = r49 * r88;
    r117 = fmaf(r26, r49, r117);
    r13 = r13 + r117;
    r28 = fmaf(r8, r13, r59 * r28);
    r49 = r4 * r12;
    r49 = r49 * r52;
    r49 = r49 * r68;
    r28 = fmaf(r50, r49, r28);
    r89 = r48 * r67;
    r89 = r89 * r1;
    r89 = r89 * r15;
    r89 = r89 * r61;
    r89 = r89 * r56;
    r28 = fmaf(r53, r89, r28);
    r85 = r58 * r40;
    r85 = r85 * r52;
    r85 = r85 * r15;
    r28 = fmaf(r104, r85, r28);
    r31 = r15 * r98;
    r14 = r58 * r42;
    r28 = fmaf(r100, r14, r28);
    r125 = r48 * r1;
    r125 = r125 * r15;
    r125 = r125 * r61;
    r125 = r125 * r56;
    r28 = fmaf(r53, r125, r28);
    r75 = r58 * r38;
    r75 = r75 * r53;
    r28 = fmaf(r64, r75, r28);
    r27 = r7 * r16;
    r27 = r27 * r57;
    r27 = fmaf(r6, r13, r13 * r27);
    r27 = fmaf(r13, r66, r27);
    r27 = fmaf(r13, r60, r27);
    r105 = r52 * r27;
    r28 = fmaf(r70, r105, r28);
    r114 = r4 * r12;
    r114 = r114 * r52;
    r114 = r114 * r67;
    r114 = r114 * r68;
    r28 = fmaf(r50, r114, r28);
    r28 = fmaf(r42, r70, r28);
    r28 = fmaf(r42, r71, r28);
    r28 = fmaf(r58, r31, r28);
    r28 = fmaf(r15, r108, r28);
    r28 = fmaf(r12, r106, r28);
    r28 = fmaf(r15, r107, r28);
    r114 = r0 * r28;
    r105 = r12 * r33;
    r105 = r105 * r33;
    r105 = r105 * r48;
    r105 = r105 * r48;
    r105 = r105 * r80;
    r105 = r105 * r11;
    r75 = r38 * r33;
    r75 = r75 * r83;
    r75 = r75 * r11;
    r75 = fmaf(r46, r75, r10 * r105);
    r105 = r33 * r99;
    r105 = r105 * r15;
    r75 = fmaf(r104, r105, r75);
    r125 = r62 * r15;
    r125 = r125 * r101;
    r125 = r125 * r88;
    r75 = fmaf(r36, r125, r75);
    r75 = r75 + r117;
    r13 = fmaf(r9, r13, r58 * r75);
    r75 = r4 * r12;
    r75 = r75 * r33;
    r75 = r75 * r68;
    r13 = fmaf(r50, r75, r13);
    r117 = r4 * r12;
    r117 = r117 * r33;
    r117 = r117 * r67;
    r117 = r117 * r68;
    r13 = fmaf(r50, r117, r13);
    r125 = r48 * r1;
    r125 = r125 * r15;
    r125 = r125 * r11;
    r125 = r125 * r61;
    r13 = fmaf(r101, r125, r13);
    r105 = r48 * r67;
    r105 = r105 * r1;
    r105 = r105 * r15;
    r105 = r105 * r11;
    r105 = r105 * r61;
    r13 = fmaf(r101, r105, r13);
    r14 = r59 * r42;
    r13 = fmaf(r100, r14, r13);
    r85 = r76 * r15;
    r85 = r85 * r35;
    r85 = r85 * r101;
    r13 = fmaf(r69, r85, r13);
    r89 = r59 * r12;
    r89 = r89 * r33;
    r89 = r89 * r48;
    r89 = r89 * r48;
    r89 = r89 * r81;
    r89 = r89 * r10;
    r13 = fmaf(r53, r89, r13);
    r49 = r59 * r38;
    r49 = r49 * r53;
    r13 = fmaf(r64, r49, r13);
    r122 = r76 * r67;
    r122 = r122 * r15;
    r122 = r122 * r35;
    r122 = r122 * r101;
    r13 = fmaf(r69, r122, r13);
    r65 = r33 * r27;
    r13 = fmaf(r70, r65, r13);
    r13 = fmaf(r38, r70, r13);
    r13 = fmaf(r38, r71, r13);
    r13 = fmaf(r15, r115, r13);
    r13 = fmaf(r59, r31, r13);
    r65 = r5 * r13;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r103,
                                          r74,
                                          r114,
                                          r65);
    r65 = r16 * r43;
    r65 = r65 * r33;
    r65 = fmaf(r44, r65, r39 * r90);
    r114 = r39 * r52;
    r114 = r114 * r52;
    r65 = fmaf(r51, r114, r65);
    r74 = r16 * r55;
    r74 = r74 * r52;
    r65 = fmaf(r44, r74, r65);
    r74 = r65 * r101;
    r74 = r74 * r88;
    r74 = fmaf(r39, r96, r36 * r74);
    r114 = r4 * r33;
    r114 = r114 * r65;
    r74 = fmaf(r104, r114, r74);
    r74 = fmaf(r43, r100, r74);
    r114 = r65 * r56;
    r114 = r114 * r88;
    r103 = r39 * r51;
    r103 = fmaf(r97, r103, r26 * r114);
    r114 = r55 * r53;
    r103 = fmaf(r64, r114, r103);
    r122 = r4 * r52;
    r122 = r122 * r52;
    r122 = r122 * r65;
    r122 = r122 * r56;
    r122 = r122 * r47;
    r103 = fmaf(r46, r122, r103);
    r122 = r74 + r103;
    r114 = r62 * r65;
    r114 = r114 * r56;
    r114 = r114 * r88;
    r114 = fmaf(r39, r95, r26 * r114);
    r49 = r55 * r83;
    r49 = r49 * r46;
    r114 = fmaf(r53, r49, r114);
    r89 = r52 * r52;
    r89 = r89 * r99;
    r89 = r89 * r65;
    r89 = r89 * r56;
    r89 = r89 * r47;
    r114 = fmaf(r46, r89, r114);
    r114 = r114 + r74;
    r114 = fmaf(r59, r114, r8 * r122);
    r74 = r7 * r16;
    r74 = r74 * r57;
    r74 = fmaf(r122, r74, r6 * r122);
    r74 = fmaf(r122, r66, r74);
    r74 = fmaf(r122, r60, r74);
    r89 = r52 * r74;
    r114 = fmaf(r70, r89, r114);
    r49 = r58 * r43;
    r49 = r49 * r53;
    r114 = fmaf(r64, r49, r114);
    r85 = r48 * r1;
    r85 = r85 * r65;
    r85 = r85 * r61;
    r85 = r85 * r56;
    r114 = fmaf(r53, r85, r114);
    r14 = r58 * r65;
    r114 = fmaf(r98, r14, r114);
    r31 = r58 * r40;
    r31 = r31 * r52;
    r31 = r31 * r65;
    r114 = fmaf(r104, r31, r114);
    r105 = r48 * r67;
    r105 = r105 * r1;
    r105 = r105 * r65;
    r105 = r105 * r61;
    r105 = r105 * r56;
    r114 = fmaf(r53, r105, r114);
    r125 = r4 * r39;
    r125 = r125 * r52;
    r125 = r125 * r68;
    r114 = fmaf(r50, r125, r114);
    r117 = r58 * r55;
    r114 = fmaf(r100, r117, r114);
    r75 = r4 * r39;
    r75 = r75 * r52;
    r75 = r75 * r67;
    r75 = r75 * r68;
    r114 = fmaf(r50, r75, r114);
    r114 = fmaf(r55, r71, r114);
    r114 = fmaf(r55, r70, r114);
    r114 = fmaf(r65, r107, r114);
    r114 = fmaf(r39, r106, r114);
    r114 = fmaf(r65, r108, r114);
    r75 = r0 * r114;
    r117 = r62 * r65;
    r117 = r117 * r101;
    r117 = r117 * r88;
    r125 = r39 * r33;
    r125 = r125 * r33;
    r125 = r125 * r48;
    r125 = r125 * r48;
    r125 = r125 * r80;
    r125 = r125 * r11;
    r125 = fmaf(r10, r125, r36 * r117);
    r117 = r43 * r33;
    r117 = r117 * r83;
    r117 = r117 * r11;
    r125 = fmaf(r46, r117, r125);
    r105 = r33 * r99;
    r105 = r105 * r65;
    r125 = fmaf(r104, r105, r125);
    r125 = r125 + r103;
    r125 = fmaf(r58, r125, r9 * r122);
    r122 = r59 * r43;
    r122 = r122 * r53;
    r125 = fmaf(r64, r122, r125);
    r103 = r4 * r39;
    r103 = r103 * r33;
    r103 = r103 * r68;
    r125 = fmaf(r50, r103, r125);
    r105 = r76 * r65;
    r105 = r105 * r35;
    r105 = r105 * r101;
    r125 = fmaf(r69, r105, r125);
    r117 = r59 * r65;
    r125 = fmaf(r98, r117, r125);
    r31 = r76 * r67;
    r31 = r31 * r65;
    r31 = r31 * r35;
    r31 = r31 * r101;
    r125 = fmaf(r69, r31, r125);
    r14 = r59 * r39;
    r14 = r14 * r33;
    r14 = r14 * r48;
    r14 = r14 * r48;
    r14 = r14 * r81;
    r14 = r14 * r10;
    r125 = fmaf(r53, r14, r125);
    r85 = r59 * r55;
    r125 = fmaf(r100, r85, r125);
    r49 = r4 * r39;
    r49 = r49 * r33;
    r49 = r49 * r67;
    r49 = r49 * r68;
    r125 = fmaf(r50, r49, r125);
    r89 = r48 * r67;
    r89 = r89 * r1;
    r89 = r89 * r65;
    r89 = r89 * r11;
    r89 = r89 * r61;
    r125 = fmaf(r101, r89, r125);
    r3 = r33 * r74;
    r125 = fmaf(r70, r3, r125);
    r109 = r48 * r1;
    r109 = r109 * r65;
    r109 = r109 * r11;
    r109 = r109 * r61;
    r125 = fmaf(r101, r109, r125);
    r125 = fmaf(r43, r71, r125);
    r125 = fmaf(r65, r115, r125);
    r125 = fmaf(r43, r70, r125);
    r109 = r5 * r125;
    r3 = r16 * r45;
    r3 = r3 * r33;
    r90 = fmaf(r37, r90, r44 * r3);
    r3 = r16 * r54;
    r3 = r3 * r52;
    r90 = fmaf(r44, r3, r90);
    r44 = r37 * r52;
    r44 = r44 * r52;
    r90 = fmaf(r51, r44, r90);
    r44 = r90 * r101;
    r44 = r44 * r88;
    r44 = fmaf(r45, r100, r36 * r44);
    r3 = r4 * r33;
    r3 = r3 * r90;
    r44 = fmaf(r104, r3, r44);
    r44 = fmaf(r37, r96, r44);
    r96 = r37 * r51;
    r3 = r4 * r52;
    r3 = r3 * r52;
    r3 = r3 * r90;
    r3 = r3 * r56;
    r3 = r3 * r47;
    r3 = fmaf(r46, r3, r97 * r96);
    r96 = r54 * r53;
    r3 = fmaf(r64, r96, r3);
    r97 = r90 * r56;
    r97 = r97 * r88;
    r3 = fmaf(r26, r97, r3);
    r97 = r44 + r3;
    r96 = r52 * r52;
    r96 = r96 * r99;
    r96 = r96 * r90;
    r96 = r96 * r56;
    r96 = r96 * r47;
    r96 = fmaf(r46, r96, r37 * r95);
    r95 = r54 * r83;
    r95 = r95 * r46;
    r96 = fmaf(r53, r95, r96);
    r47 = r62 * r90;
    r47 = r47 * r56;
    r47 = r47 * r88;
    r96 = fmaf(r26, r47, r96);
    r96 = r96 + r44;
    r96 = fmaf(r59, r96, r8 * r97);
    r8 = r4 * r37;
    r8 = r8 * r52;
    r8 = r8 * r67;
    r8 = r8 * r68;
    r96 = fmaf(r50, r8, r96);
    r44 = r4 * r37;
    r44 = r44 * r52;
    r44 = r44 * r68;
    r96 = fmaf(r50, r44, r96);
    r47 = r48 * r1;
    r47 = r47 * r90;
    r47 = r47 * r61;
    r47 = r47 * r56;
    r96 = fmaf(r53, r47, r96);
    r95 = r58 * r40;
    r95 = r95 * r52;
    r95 = r95 * r90;
    r96 = fmaf(r104, r95, r96);
    r26 = r58 * r90;
    r96 = fmaf(r98, r26, r96);
    r89 = r58 * r45;
    r89 = r89 * r53;
    r96 = fmaf(r64, r89, r96);
    r49 = r58 * r54;
    r96 = fmaf(r100, r49, r96);
    r85 = r48 * r67;
    r85 = r85 * r1;
    r85 = r85 * r90;
    r85 = r85 * r61;
    r85 = r85 * r56;
    r96 = fmaf(r53, r85, r96);
    r14 = r7 * r16;
    r14 = r14 * r57;
    r6 = fmaf(r6, r97, r97 * r14);
    r6 = fmaf(r97, r66, r6);
    r6 = fmaf(r97, r60, r6);
    r60 = r52 * r6;
    r96 = fmaf(r70, r60, r96);
    r96 = fmaf(r54, r70, r96);
    r96 = fmaf(r54, r71, r96);
    r96 = fmaf(r90, r108, r96);
    r96 = fmaf(r90, r107, r96);
    r96 = fmaf(r37, r106, r96);
    r106 = r0 * r96;
    r60 = r62 * r90;
    r60 = r60 * r101;
    r60 = r60 * r88;
    r88 = r45 * r33;
    r88 = r88 * r83;
    r88 = r88 * r11;
    r88 = fmaf(r46, r88, r36 * r60);
    r60 = r33 * r99;
    r60 = r60 * r90;
    r88 = fmaf(r104, r60, r88);
    r36 = r37 * r33;
    r36 = r36 * r33;
    r36 = r36 * r48;
    r36 = r36 * r48;
    r36 = r36 * r80;
    r36 = r36 * r11;
    r88 = fmaf(r10, r36, r88);
    r88 = r88 + r3;
    r88 = fmaf(r58, r88, r9 * r97);
    r97 = r76 * r67;
    r97 = r97 * r90;
    r97 = r97 * r35;
    r97 = r97 * r101;
    r88 = fmaf(r69, r97, r88);
    r9 = r76 * r90;
    r9 = r9 * r35;
    r9 = r9 * r101;
    r88 = fmaf(r69, r9, r88);
    r69 = r48 * r67;
    r69 = r69 * r1;
    r69 = r69 * r90;
    r69 = r69 * r11;
    r69 = r69 * r61;
    r88 = fmaf(r101, r69, r88);
    r35 = r48 * r1;
    r35 = r35 * r90;
    r35 = r35 * r11;
    r35 = r35 * r61;
    r88 = fmaf(r101, r35, r88);
    r61 = r59 * r90;
    r88 = fmaf(r98, r61, r88);
    r98 = r59 * r45;
    r98 = r98 * r53;
    r88 = fmaf(r64, r98, r88);
    r64 = r4 * r37;
    r64 = r64 * r33;
    r64 = r64 * r67;
    r64 = r64 * r68;
    r88 = fmaf(r50, r64, r88);
    r11 = r59 * r54;
    r88 = fmaf(r100, r11, r88);
    r100 = r4 * r37;
    r100 = r100 * r33;
    r100 = r100 * r68;
    r88 = fmaf(r50, r100, r88);
    r50 = r33 * r6;
    r88 = fmaf(r70, r50, r88);
    r68 = r59 * r37;
    r68 = r68 * r33;
    r68 = r68 * r48;
    r68 = r68 * r48;
    r68 = r68 * r81;
    r68 = r68 * r10;
    r88 = fmaf(r53, r68, r88);
    r88 = fmaf(r45, r71, r88);
    r88 = fmaf(r90, r115, r88);
    r88 = fmaf(r45, r70, r88);
    r68 = r5 * r88;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r75,
                                          r109,
                                          r106,
                                          r68);
    r68 = r0 * r4;
    r68 = r68 * r2;
    r63 = r4 * r63;
    r106 = r5 * r63;
    r68 = fmaf(r102, r106, r82 * r68);
    r109 = r0 * r4;
    r109 = r109 * r2;
    r109 = fmaf(r121, r106, r93 * r109);
    r75 = r0 * r4;
    r75 = r75 * r2;
    r75 = fmaf(r78, r106, r73 * r75);
    r50 = r0 * r4;
    r50 = r50 * r2;
    r50 = fmaf(r13, r106, r28 * r50);
    WriteSum4<float, float>((float*)inout_shared, r68, r109, r75, r50);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r0 * r4;
    r50 = r50 * r2;
    r50 = fmaf(r125, r106, r114 * r50);
    r75 = r0 * r4;
    r75 = r75 * r2;
    r106 = fmaf(r88, r106, r96 * r75);
    WriteSum2<float, float>((float*)inout_shared, r50, r106);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r106 = r0 * r0;
    r50 = r82 * r106;
    r5 = r5 * r5;
    r75 = r102 * r5;
    r102 = fmaf(r102, r75, r82 * r50);
    r82 = r93 * r93;
    r109 = r121 * r121;
    r109 = fmaf(r5, r109, r106 * r82);
    r82 = r78 * r78;
    r68 = r73 * r73;
    r68 = fmaf(r106, r68, r5 * r82);
    r82 = r28 * r28;
    r100 = r13 * r13;
    r100 = fmaf(r5, r100, r106 * r82);
    WriteSum4<float, float>((float*)inout_shared, r102, r109, r68, r100);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r100 = r114 * r114;
    r68 = r125 * r125;
    r68 = fmaf(r5, r68, r106 * r100);
    r100 = r88 * r88;
    r109 = r96 * r96;
    r109 = fmaf(r106, r109, r5 * r100);
    WriteSum2<float, float>((float*)inout_shared, r68, r109);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r109 = fmaf(r93, r50, r121 * r75);
    r68 = fmaf(r78, r75, r73 * r50);
    r100 = fmaf(r28, r50, r13 * r75);
    r102 = fmaf(r114, r50, r125 * r75);
    WriteSum4<float, float>((float*)inout_shared, r109, r68, r100, r102);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = fmaf(r88, r75, r96 * r50);
    r50 = r121 * r78;
    r102 = r93 * r73;
    r102 = fmaf(r106, r102, r5 * r50);
    r50 = r93 * r28;
    r100 = r121 * r13;
    r100 = fmaf(r5, r100, r106 * r50);
    r50 = r93 * r114;
    r68 = r121 * r125;
    r68 = fmaf(r5, r68, r106 * r50);
    WriteSum4<float, float>((float*)inout_shared, r75, r102, r100, r68);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r121 * r88;
    r100 = r93 * r96;
    r100 = fmaf(r106, r100, r5 * r68);
    r68 = r78 * r13;
    r102 = r73 * r28;
    r102 = fmaf(r106, r102, r5 * r68);
    r68 = r73 * r114;
    r75 = r78 * r125;
    r75 = fmaf(r5, r75, r106 * r68);
    r68 = r78 * r88;
    r50 = r73 * r96;
    r50 = fmaf(r106, r50, r5 * r68);
    WriteSum4<float, float>((float*)inout_shared, r100, r102, r75, r50);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r13 * r125;
    r75 = r28 * r114;
    r75 = fmaf(r106, r75, r5 * r50);
    r50 = r28 * r96;
    r102 = r13 * r88;
    r102 = fmaf(r5, r102, r106 * r50);
    r50 = r125 * r88;
    r100 = r114 * r96;
    r100 = fmaf(r106, r100, r5 * r50);
    WriteSum3<float, float>((float*)inout_shared, r75, r102, r100);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r4 * r2;
    WriteSum2<float, float>((float*)inout_shared, r2, r63);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float*)inout_shared, r41, r41);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
}

void ThinPrismFisheyeSplitFixedFocalAndExtraFixedPointResJac(
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
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
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
  ThinPrismFisheyeSplitFixedFocalAndExtraFixedPointResJacKernel<<<n_blocks,
                                                                  1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      pixel,
      pixel_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
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