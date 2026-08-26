#include "kernel_thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedFocalAndExtraFixedPointResJacFirstKernel(
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
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104,
      r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116,
      r117, r118, r119, r120, r121, r122, r123, r124;
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
    r49 = 9.99999999999999955e-07;
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
    r10 = r40 * r29;
    r10 = fmaf(r34, r10, r26);
    r10 = fmaf(r13, r10, r12);
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
    r10 = fmaf(r11, r12, r10);
    r10 = fmaf(r36, r37, r10);
    r10 = fmaf(r35, r39, r10);
    r10 = fmaf(r14, r44, r10);
    r10 = fmaf(r15, r47, r10);
    r47 = copysign(1.0, r10);
    r47 = fmaf(r49, r47, r10);
    r10 = r47 * r47;
    r44 = 1.0 / r10;
    r35 = r33 * r33;
    r35 = fmaf(r44, r35, r44 * r48);
    r48 = sqrtf(r35);
    r36 = copysign(1.0, r48);
    r36 = fmaf(r49, r36, r48);
    r49 = r36 * r36;
    r11 = 1.0 / r49;
    r48 = atanf(r48);
    r50 = r48 * r44;
    r46 = r11 * r50;
    r52 = r33 * r46;
    r26 = r33 * r48;
    r52 = r52 * r26;
    r55 = r51 * r51;
    r55 = r55 * r48;
    r55 = r55 * r46;
    r56 = r52 + r55;
    ReadIdx4<1024, float, float, float4>(focal_and_extra,
                                         4 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r57,
                                         r58,
                                         r59,
                                         r60);
    r61 = r51 * r51;
    r62 = 3.00000000000000000e+00;
    r61 = r61 * r48;
    r61 = r61 * r62;
    r61 = fmaf(r46, r61, r52);
    r61 = fmaf(r58, r61, r8 * r56);
    r52 = r16 * r46;
    r63 = r26 * r52;
    r64 = r57 * r63;
    r65 = r56 * r56;
    r66 = r56 * r65;
    r67 = fmaf(r59, r66, r6 * r56);
    r66 = r60 * r66;
    r67 = fmaf(r56, r66, r67);
    r67 = fmaf(r7, r65, r67);
    r60 = 1.0 / r47;
    r68 = 1.0 / r36;
    r69 = r60 * r68;
    r70 = r67 * r69;
    r71 = r48 * r70;
    r72 = r51 * r48;
    r61 = fmaf(r69, r72, r61);
    r61 = fmaf(r51, r64, r61);
    r61 = fmaf(r51, r71, r61);
    r2 = fmaf(r0, r61, r2);
    r61 = r33 * r62;
    r61 = r61 * r46;
    r61 = fmaf(r26, r61, r55);
    r61 = fmaf(r57, r61, r9 * r56);
    r55 = r58 * r51;
    r61 = fmaf(r63, r55, r61);
    r61 = fmaf(r26, r70, r61);
    r61 = fmaf(r69, r26, r61);
    r61 = fmaf(r5, r61, r1);
    r61 = fmaf(r3, r4, r61);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r61);
    r3 = fmaf(r61, r61, r2 * r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r3);
  if (global_thread_idx < problem_size) {
    r3 = r51 * r48;
    r1 = r16 * r34;
    r55 = r19 * r24;
    r72 = 5.00000000000000000e-01;
    r73 = r18 * r21;
    r73 = fmaf(r72, r73, r72 * r55);
    r55 = r17 * r22;
    r74 = -5.00000000000000000e-01;
    r73 = fmaf(r74, r55, r73);
    r75 = r20 * r23;
    r73 = fmaf(r72, r75, r73);
    r75 = r17 * r24;
    r55 = r20 * r21;
    r55 = fmaf(r74, r55, r74 * r75);
    r75 = r19 * r22;
    r55 = fmaf(r74, r75, r55);
    r76 = r18 * r23;
    r55 = fmaf(r72, r76, r55);
    r76 = r29 * r55;
    r1 = fmaf(r16, r76, r73 * r1);
    r75 = r16 * r25;
    r77 = fmaf(r72, r31, r74 * r27);
    r77 = fmaf(r74, r28, r77);
    r77 = fmaf(r74, r30, r77);
    r78 = r16 * r32;
    r79 = r20 * r24;
    r80 = r17 * r21;
    r80 = fmaf(r74, r80, r72 * r79);
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
    r82 = fmaf(r55, r83, r82);
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
    r3 = r3 * r82;
    r3 = r3 * r86;
    r87 = r16 * r33;
    r88 = r25 * r40;
    r89 = r34 * r80;
    r90 = r40 * r89;
    r88 = fmaf(r55, r88, r90);
    r91 = r16 * r29;
    r91 = r91 * r77;
    r92 = r16 * r32;
    r92 = fmaf(r73, r92, r91);
    r88 = r88 + r92;
    r93 = r25 * r80;
    r93 = r93 * r84;
    r85 = r93 + r85;
    r85 = fmaf(r14, r85, r15 * r88);
    r88 = r16 * r34;
    r88 = fmaf(r77, r88, r79);
    r79 = r16 * r32;
    r79 = fmaf(r55, r79, r81);
    r88 = r88 + r79;
    r85 = fmaf(r13, r88, r85);
    r87 = r87 * r85;
    r88 = r16 * r51;
    r88 = r88 * r82;
    r88 = fmaf(r44, r88, r44 * r87);
    r87 = r33 * r33;
    r81 = r16 * r25;
    r81 = r81 * r55;
    r89 = r16 * r89;
    r94 = r81 + r89;
    r92 = r92 + r94;
    r95 = r40 * r34;
    r95 = fmaf(r40, r76, r73 * r95);
    r95 = r95 + r75;
    r95 = fmaf(r13, r95, r14 * r92);
    r1 = r93 + r1;
    r95 = fmaf(r15, r1, r95);
    r10 = r47 * r10;
    r10 = 1.0 / r10;
    r47 = r40 * r10;
    r87 = r87 * r95;
    r88 = fmaf(r47, r87, r88);
    r1 = r51 * r51;
    r1 = r1 * r95;
    r88 = fmaf(r47, r1, r88);
    r1 = r62 * r88;
    r87 = r51 * r46;
    r93 = r41 + r35;
    r93 = 1.0 / r93;
    r35 = rsqrtf(r35);
    r92 = r51 * r35;
    r73 = r93 * r92;
    r87 = r87 * r73;
    r1 = fmaf(r87, r1, r46 * r3);
    r3 = r48 * r48;
    r96 = -6.00000000000000000e+00;
    r3 = r3 * r95;
    r3 = r3 * r96;
    r3 = r3 * r11;
    r3 = r3 * r10;
    r97 = r51 * r51;
    r97 = r97 * r47;
    r98 = -3.00000000000000000e+00;
    r99 = r51 * r48;
    r100 = r98 * r99;
    r49 = r36 * r49;
    r49 = 1.0 / r49;
    r49 = r49 * r50;
    r36 = r92 * r49;
    r100 = r100 * r36;
    r101 = r33 * r33;
    r101 = r101 * r88;
    r101 = r101 * r35;
    r101 = r101 * r93;
    r101 = fmaf(r46, r101, r85 * r63);
    r49 = r26 * r49;
    r102 = r33 * r35;
    r49 = r49 * r102;
    r103 = r88 * r49;
    r104 = r33 * r33;
    r104 = r104 * r48;
    r104 = r104 * r48;
    r104 = r104 * r95;
    r104 = r104 * r11;
    r101 = fmaf(r47, r104, r101);
    r101 = fmaf(r4, r103, r101);
    r1 = fmaf(r3, r97, r1);
    r1 = fmaf(r88, r100, r1);
    r1 = r1 + r101;
    r97 = r82 * r52;
    r97 = fmaf(r88, r87, r99 * r97);
    r104 = r51 * r51;
    r104 = r104 * r48;
    r104 = r104 * r48;
    r104 = r104 * r95;
    r104 = r104 * r11;
    r97 = fmaf(r47, r104, r97);
    r105 = r4 * r88;
    r105 = r105 * r99;
    r97 = fmaf(r36, r105, r97);
    r101 = r101 + r97;
    r1 = fmaf(r8, r101, r58 * r1);
    r105 = r48 * r74;
    r105 = r105 * r11;
    r105 = r105 * r60;
    r105 = r105 * r92;
    r92 = r67 * r105;
    r104 = r57 * r33;
    r104 = r104 * r88;
    r104 = r104 * r52;
    r1 = fmaf(r73, r104, r1);
    r106 = r57 * r51;
    r106 = r106 * r33;
    r106 = r106 * r48;
    r106 = r106 * r48;
    r106 = r106 * r84;
    r106 = r106 * r95;
    r106 = r106 * r11;
    r1 = fmaf(r10, r106, r1);
    r107 = r72 * r88;
    r107 = r107 * r73;
    r1 = fmaf(r70, r107, r1);
    r108 = r48 * r82;
    r1 = fmaf(r69, r108, r1);
    r109 = r4 * r51;
    r109 = r109 * r95;
    r109 = r109 * r68;
    r1 = fmaf(r50, r109, r1);
    r110 = r4 * r51;
    r110 = r110 * r67;
    r110 = r110 * r95;
    r110 = r110 * r68;
    r1 = fmaf(r50, r110, r1);
    r111 = r72 * r88;
    r111 = r111 * r69;
    r1 = fmaf(r73, r111, r1);
    r112 = r57 * r88;
    r113 = r40 * r26;
    r113 = r113 * r36;
    r1 = fmaf(r113, r112, r1);
    r114 = r51 * r48;
    r115 = r7 * r16;
    r115 = r115 * r56;
    r115 = fmaf(r101, r115, r6 * r101);
    r116 = 4.00000000000000000e+00;
    r66 = r116 * r66;
    r59 = r59 * r62;
    r59 = r59 * r65;
    r115 = fmaf(r101, r66, r115);
    r115 = fmaf(r101, r59, r115);
    r114 = r114 * r115;
    r1 = fmaf(r69, r114, r1);
    r65 = r57 * r85;
    r65 = r65 * r52;
    r1 = fmaf(r99, r65, r1);
    r1 = fmaf(r88, r92, r1);
    r1 = fmaf(r88, r105, r1);
    r1 = fmaf(r82, r64, r1);
    r1 = fmaf(r82, r71, r1);
    r65 = r0 * r1;
    r114 = r85 * r86;
    r114 = r114 * r46;
    r112 = r33 * r33;
    r112 = r112 * r62;
    r112 = r112 * r88;
    r112 = r112 * r35;
    r112 = r112 * r93;
    r112 = fmaf(r46, r112, r26 * r114);
    r114 = r40 * r12;
    r111 = r33 * r33;
    r114 = r114 * r111;
    r114 = r114 * r10;
    r112 = fmaf(r98, r103, r112);
    r112 = fmaf(r3, r114, r112);
    r112 = r112 + r97;
    r101 = fmaf(r9, r101, r57 * r112);
    r112 = r33 * r72;
    r112 = r112 * r88;
    r112 = r112 * r35;
    r112 = r112 * r93;
    r101 = fmaf(r69, r112, r101);
    r97 = r33 * r48;
    r97 = r97 * r67;
    r97 = r97 * r74;
    r97 = r97 * r88;
    r97 = r97 * r11;
    r97 = r97 * r60;
    r101 = fmaf(r35, r97, r101);
    r3 = r115 * r69;
    r101 = fmaf(r26, r3, r101);
    r103 = r4 * r33;
    r103 = r103 * r67;
    r103 = r103 * r95;
    r103 = r103 * r68;
    r101 = fmaf(r50, r103, r101);
    r111 = r58 * r51;
    r111 = r111 * r33;
    r111 = r111 * r48;
    r111 = r111 * r48;
    r111 = r111 * r84;
    r111 = r111 * r11;
    r111 = r111 * r10;
    r110 = r58 * r82;
    r101 = fmaf(r63, r110, r101);
    r109 = r58 * r33;
    r109 = r109 * r88;
    r109 = r109 * r52;
    r101 = fmaf(r73, r109, r101);
    r108 = r4 * r33;
    r108 = r108 * r95;
    r108 = r108 * r68;
    r101 = fmaf(r50, r108, r101);
    r107 = r58 * r88;
    r101 = fmaf(r113, r107, r101);
    r106 = r72 * r93;
    r106 = r106 * r70;
    r106 = r106 * r102;
    r102 = r58 * r85;
    r102 = r102 * r52;
    r101 = fmaf(r99, r102, r101);
    r104 = r33 * r48;
    r104 = r104 * r74;
    r104 = r104 * r88;
    r104 = r104 * r11;
    r104 = r104 * r60;
    r101 = fmaf(r35, r104, r101);
    r116 = r48 * r85;
    r101 = fmaf(r69, r116, r101);
    r101 = fmaf(r95, r111, r101);
    r101 = fmaf(r88, r106, r101);
    r101 = fmaf(r85, r71, r101);
    r116 = r5 * r101;
    r104 = r51 * r51;
    r102 = r44 * r104;
    r107 = r16 * r51;
    r89 = r91 + r89;
    r91 = r16 * r32;
    r108 = r19 * r24;
    r109 = r18 * r21;
    r109 = fmaf(r74, r109, r74 * r108);
    r108 = r17 * r22;
    r109 = fmaf(r72, r108, r109);
    r110 = r20 * r23;
    r109 = fmaf(r74, r110, r109);
    r91 = r91 * r109;
    r110 = r16 * r25;
    r108 = r17 * r24;
    r95 = r20 * r21;
    r95 = fmaf(r72, r95, r72 * r108);
    r108 = r19 * r22;
    r95 = fmaf(r72, r108, r95);
    r103 = r18 * r23;
    r95 = fmaf(r74, r103, r95);
    r110 = fmaf(r95, r110, r91);
    r89 = r89 + r110;
    r103 = r32 * r84;
    r103 = r103 * r95;
    r108 = r29 * r80;
    r108 = r108 * r84;
    r3 = r103 + r108;
    r3 = fmaf(r13, r3, r15 * r89);
    r89 = r40 * r34;
    r89 = fmaf(r40, r83, r95 * r89);
    r97 = r16 * r25;
    r97 = r97 * r80;
    r112 = r16 * r29;
    r112 = fmaf(r109, r112, r97);
    r89 = r89 + r112;
    r3 = fmaf(r14, r89, r3);
    r107 = r107 * r3;
    r89 = r51 * r51;
    r117 = r40 * r29;
    r117 = fmaf(r77, r117, r90);
    r117 = r117 + r110;
    r110 = r16 * r29;
    r110 = r110 * r95;
    r118 = r16 * r34;
    r118 = fmaf(r109, r118, r110);
    r118 = r118 + r75;
    r118 = fmaf(r14, r118, r13 * r117);
    r117 = r25 * r109;
    r75 = r84 * r117;
    r108 = r108 + r75;
    r118 = fmaf(r15, r108, r118);
    r89 = r89 * r118;
    r89 = fmaf(r47, r89, r44 * r107);
    r107 = r33 * r33;
    r107 = r107 * r118;
    r89 = fmaf(r47, r107, r89);
    r108 = r16 * r33;
    r119 = r25 * r40;
    r119 = fmaf(r77, r119, r78);
    r78 = r40 * r34;
    r119 = fmaf(r109, r78, r119);
    r119 = r119 + r110;
    r78 = r16 * r34;
    r83 = fmaf(r16, r83, r95 * r78);
    r83 = r83 + r112;
    r83 = fmaf(r13, r83, r15 * r119);
    r75 = r103 + r75;
    r83 = fmaf(r14, r75, r83);
    r108 = r108 * r83;
    r89 = fmaf(r44, r108, r89);
    r102 = r102 * r11;
    r102 = r102 * r48;
    r102 = r102 * r35;
    r102 = r102 * r93;
    r102 = r102 * r89;
    r108 = r4 * r89;
    r108 = r108 * r99;
    r108 = fmaf(r36, r108, r102);
    r107 = r51 * r51;
    r107 = r107 * r48;
    r107 = r107 * r48;
    r107 = r107 * r118;
    r107 = r107 * r11;
    r108 = fmaf(r47, r107, r108);
    r75 = r3 * r52;
    r108 = fmaf(r99, r75, r108);
    r75 = r4 * r89;
    r75 = fmaf(r49, r75, r83 * r63);
    r107 = r33 * r33;
    r107 = r107 * r48;
    r107 = r107 * r48;
    r107 = r107 * r118;
    r107 = r107 * r11;
    r75 = fmaf(r47, r107, r75);
    r103 = r33 * r33;
    r103 = r103 * r89;
    r103 = r103 * r35;
    r103 = r103 * r93;
    r75 = fmaf(r46, r103, r75);
    r103 = r108 + r75;
    r102 = fmaf(r89, r100, r62 * r102);
    r107 = r51 * r51;
    r107 = r107 * r48;
    r107 = r107 * r48;
    r107 = r107 * r96;
    r107 = r107 * r118;
    r107 = r107 * r11;
    r102 = fmaf(r10, r107, r102);
    r119 = r51 * r48;
    r119 = r119 * r86;
    r119 = r119 * r3;
    r102 = fmaf(r46, r119, r102);
    r102 = r102 + r75;
    r102 = fmaf(r58, r102, r8 * r103);
    r75 = r57 * r83;
    r75 = r75 * r52;
    r102 = fmaf(r99, r75, r102);
    r119 = r57 * r51;
    r119 = r119 * r33;
    r119 = r119 * r48;
    r119 = r119 * r48;
    r119 = r119 * r84;
    r119 = r119 * r118;
    r119 = r119 * r11;
    r102 = fmaf(r10, r119, r102);
    r107 = r4 * r51;
    r107 = r107 * r118;
    r107 = r107 * r68;
    r102 = fmaf(r50, r107, r102);
    r78 = r72 * r89;
    r78 = r78 * r73;
    r102 = fmaf(r70, r78, r102);
    r95 = r57 * r89;
    r102 = fmaf(r113, r95, r102);
    r110 = r57 * r33;
    r110 = r110 * r89;
    r110 = r110 * r52;
    r102 = fmaf(r73, r110, r102);
    r77 = r72 * r89;
    r77 = r77 * r69;
    r102 = fmaf(r73, r77, r102);
    r120 = r4 * r51;
    r120 = r120 * r67;
    r120 = r120 * r118;
    r120 = r120 * r68;
    r102 = fmaf(r50, r120, r102);
    r121 = r51 * r48;
    r122 = r7 * r16;
    r122 = r122 * r56;
    r122 = fmaf(r103, r122, r6 * r103);
    r122 = fmaf(r103, r66, r122);
    r122 = fmaf(r103, r59, r122);
    r121 = r121 * r122;
    r102 = fmaf(r69, r121, r102);
    r123 = r48 * r3;
    r102 = fmaf(r69, r123, r102);
    r102 = fmaf(r89, r105, r102);
    r102 = fmaf(r3, r71, r102);
    r102 = fmaf(r3, r64, r102);
    r102 = fmaf(r89, r92, r102);
    r123 = r0 * r102;
    r121 = r86 * r83;
    r121 = r121 * r46;
    r120 = r98 * r89;
    r120 = fmaf(r49, r120, r26 * r121);
    r121 = r33 * r33;
    r121 = r121 * r48;
    r121 = r121 * r48;
    r121 = r121 * r96;
    r121 = r121 * r118;
    r121 = r121 * r11;
    r120 = fmaf(r10, r121, r120);
    r77 = r33 * r33;
    r77 = r77 * r62;
    r77 = r77 * r89;
    r77 = r77 * r35;
    r77 = r77 * r93;
    r120 = fmaf(r46, r77, r120);
    r120 = r120 + r108;
    r120 = fmaf(r57, r120, r9 * r103);
    r103 = r4 * r33;
    r103 = r103 * r118;
    r103 = r103 * r68;
    r120 = fmaf(r50, r103, r120);
    r108 = r58 * r83;
    r108 = r108 * r52;
    r120 = fmaf(r99, r108, r120);
    r77 = r33 * r48;
    r77 = r77 * r67;
    r77 = r77 * r74;
    r77 = r77 * r89;
    r77 = r77 * r11;
    r77 = r77 * r60;
    r120 = fmaf(r35, r77, r120);
    r121 = r122 * r69;
    r120 = fmaf(r26, r121, r120);
    r110 = r33 * r72;
    r110 = r110 * r89;
    r110 = r110 * r35;
    r110 = r110 * r93;
    r120 = fmaf(r69, r110, r120);
    r95 = r33 * r48;
    r95 = r95 * r74;
    r95 = r95 * r89;
    r95 = r95 * r11;
    r95 = r95 * r60;
    r120 = fmaf(r35, r95, r120);
    r78 = r58 * r89;
    r120 = fmaf(r113, r78, r120);
    r107 = r58 * r33;
    r107 = r107 * r89;
    r107 = r107 * r52;
    r120 = fmaf(r73, r107, r120);
    r119 = r58 * r3;
    r120 = fmaf(r63, r119, r120);
    r75 = r4 * r33;
    r75 = r75 * r67;
    r75 = r75 * r118;
    r75 = r75 * r68;
    r120 = fmaf(r50, r75, r120);
    r124 = r48 * r83;
    r120 = fmaf(r69, r124, r120);
    r120 = fmaf(r118, r111, r120);
    r120 = fmaf(r89, r106, r120);
    r120 = fmaf(r83, r71, r120);
    r124 = r5 * r120;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r65,
                                          r116,
                                          r123,
                                          r124);
    r124 = r33 * r33;
    r123 = r51 * r51;
    r116 = r25 * r84;
    r31 = fmaf(r74, r31, r72 * r27);
    r31 = fmaf(r72, r28, r31);
    r31 = fmaf(r72, r30, r31);
    r116 = r116 * r31;
    r76 = r84 * r76;
    r30 = r116 + r76;
    r28 = r16 * r32;
    r28 = r28 * r31;
    r97 = r97 + r28;
    r27 = r40 * r29;
    r97 = fmaf(r109, r27, r97);
    r65 = r40 * r34;
    r97 = fmaf(r55, r65, r97);
    r97 = fmaf(r13, r97, r15 * r30);
    r30 = r16 * r34;
    r30 = fmaf(r16, r117, r31 * r30);
    r30 = r30 + r79;
    r97 = fmaf(r14, r30, r97);
    r123 = r123 * r97;
    r30 = r16 * r51;
    r65 = r16 * r29;
    r65 = r65 * r31;
    r81 = r81 + r65;
    r27 = r32 * r40;
    r81 = fmaf(r109, r27, r81);
    r81 = r81 + r90;
    r80 = r32 * r80;
    r80 = r80 * r84;
    r76 = r80 + r76;
    r76 = fmaf(r13, r76, r14 * r81);
    r81 = r16 * r34;
    r81 = fmaf(r55, r81, r28);
    r81 = r81 + r112;
    r76 = fmaf(r15, r81, r76);
    r30 = r30 * r76;
    r30 = fmaf(r44, r30, r47 * r123);
    r123 = r33 * r33;
    r123 = r123 * r97;
    r30 = fmaf(r47, r123, r30);
    r81 = r16 * r33;
    r65 = r91 + r65;
    r65 = r65 + r94;
    r94 = r40 * r34;
    r117 = fmaf(r40, r117, r31 * r94);
    r117 = r117 + r79;
    r117 = fmaf(r15, r117, r13 * r65);
    r80 = r116 + r80;
    r117 = fmaf(r14, r80, r117);
    r81 = r81 * r117;
    r30 = fmaf(r44, r81, r30);
    r124 = r124 * r30;
    r124 = r124 * r35;
    r124 = r124 * r93;
    r124 = fmaf(r117, r63, r46 * r124);
    r81 = r33 * r33;
    r81 = r81 * r48;
    r81 = r81 * r48;
    r81 = r81 * r97;
    r81 = r81 * r11;
    r124 = fmaf(r47, r81, r124);
    r123 = r4 * r30;
    r124 = fmaf(r49, r123, r124);
    r123 = r51 * r51;
    r123 = r123 * r48;
    r123 = r123 * r48;
    r123 = r123 * r97;
    r123 = r123 * r11;
    r123 = fmaf(r30, r87, r47 * r123);
    r81 = r4 * r30;
    r81 = r81 * r99;
    r123 = fmaf(r36, r81, r123);
    r80 = r76 * r52;
    r123 = fmaf(r99, r80, r123);
    r80 = r124 + r123;
    r81 = r51 * r51;
    r81 = r81 * r48;
    r81 = r81 * r48;
    r81 = r81 * r96;
    r81 = r81 * r97;
    r81 = r81 * r11;
    r14 = r62 * r30;
    r14 = fmaf(r87, r14, r10 * r81);
    r81 = r51 * r48;
    r81 = r81 * r86;
    r81 = r81 * r76;
    r14 = fmaf(r46, r81, r14);
    r14 = fmaf(r30, r100, r14);
    r14 = r14 + r124;
    r14 = fmaf(r58, r14, r8 * r80);
    r124 = r51 * r48;
    r81 = r7 * r16;
    r81 = r81 * r56;
    r81 = fmaf(r80, r81, r6 * r80);
    r81 = fmaf(r80, r59, r81);
    r81 = fmaf(r80, r66, r81);
    r124 = r124 * r81;
    r14 = fmaf(r69, r124, r14);
    r116 = r57 * r51;
    r116 = r116 * r33;
    r116 = r116 * r48;
    r116 = r116 * r48;
    r116 = r116 * r84;
    r116 = r116 * r97;
    r116 = r116 * r11;
    r14 = fmaf(r10, r116, r14);
    r15 = r72 * r30;
    r15 = r15 * r69;
    r14 = fmaf(r73, r15, r14);
    r65 = r4 * r51;
    r65 = r65 * r97;
    r65 = r65 * r68;
    r14 = fmaf(r50, r65, r14);
    r13 = r57 * r30;
    r14 = fmaf(r113, r13, r14);
    r79 = r72 * r30;
    r79 = r79 * r73;
    r14 = fmaf(r70, r79, r14);
    r94 = r57 * r117;
    r94 = r94 * r52;
    r14 = fmaf(r99, r94, r14);
    r31 = r4 * r51;
    r31 = r31 * r67;
    r31 = r31 * r97;
    r31 = r31 * r68;
    r14 = fmaf(r50, r31, r14);
    r91 = r48 * r76;
    r14 = fmaf(r69, r91, r14);
    r112 = r57 * r33;
    r112 = r112 * r30;
    r112 = r112 * r52;
    r14 = fmaf(r73, r112, r14);
    r14 = fmaf(r30, r92, r14);
    r14 = fmaf(r76, r71, r14);
    r14 = fmaf(r76, r64, r14);
    r14 = fmaf(r30, r105, r14);
    r112 = r0 * r14;
    r91 = r33 * r33;
    r91 = r91 * r62;
    r91 = r91 * r30;
    r91 = r91 * r35;
    r91 = r91 * r93;
    r31 = r86 * r117;
    r31 = r31 * r46;
    r31 = fmaf(r26, r31, r46 * r91);
    r91 = r33 * r33;
    r91 = r91 * r48;
    r91 = r91 * r48;
    r91 = r91 * r96;
    r91 = r91 * r97;
    r91 = r91 * r11;
    r31 = fmaf(r10, r91, r31);
    r94 = r98 * r30;
    r31 = fmaf(r49, r94, r31);
    r31 = r31 + r123;
    r31 = fmaf(r57, r31, r9 * r80);
    r80 = r33 * r48;
    r80 = r80 * r74;
    r80 = r80 * r30;
    r80 = r80 * r11;
    r80 = r80 * r60;
    r31 = fmaf(r35, r80, r31);
    r123 = r33 * r72;
    r123 = r123 * r30;
    r123 = r123 * r35;
    r123 = r123 * r93;
    r31 = fmaf(r69, r123, r31);
    r94 = r4 * r33;
    r94 = r94 * r67;
    r94 = r94 * r97;
    r94 = r94 * r68;
    r31 = fmaf(r50, r94, r31);
    r91 = r58 * r30;
    r31 = fmaf(r113, r91, r31);
    r79 = r58 * r76;
    r31 = fmaf(r63, r79, r31);
    r13 = r33 * r48;
    r13 = r13 * r67;
    r13 = r13 * r74;
    r13 = r13 * r30;
    r13 = r13 * r11;
    r13 = r13 * r60;
    r31 = fmaf(r35, r13, r31);
    r65 = r58 * r117;
    r65 = r65 * r52;
    r31 = fmaf(r99, r65, r31);
    r15 = r4 * r33;
    r15 = r15 * r97;
    r15 = r15 * r68;
    r31 = fmaf(r50, r15, r31);
    r116 = r58 * r33;
    r116 = r116 * r30;
    r116 = r116 * r52;
    r31 = fmaf(r73, r116, r31);
    r124 = r48 * r117;
    r31 = fmaf(r69, r124, r31);
    r28 = r81 * r69;
    r31 = fmaf(r26, r28, r31);
    r31 = fmaf(r97, r111, r31);
    r31 = fmaf(r117, r71, r31);
    r31 = fmaf(r30, r106, r31);
    r28 = r5 * r31;
    r124 = r12 * r51;
    r124 = r124 * r51;
    r124 = r124 * r48;
    r124 = r124 * r48;
    r124 = r124 * r96;
    r124 = r124 * r11;
    r116 = r42 * r51;
    r116 = r116 * r48;
    r116 = r116 * r86;
    r116 = fmaf(r46, r116, r10 * r124);
    r124 = r40 * r12;
    r124 = r124 * r104;
    r124 = r124 * r10;
    r104 = r114 + r124;
    r15 = r16 * r38;
    r15 = r15 * r33;
    r104 = fmaf(r44, r15, r104);
    r65 = r16 * r42;
    r65 = r65 * r51;
    r104 = fmaf(r44, r65, r104);
    r65 = r62 * r104;
    r116 = fmaf(r87, r65, r116);
    r15 = r48 * r48;
    r15 = r15 * r11;
    r114 = fmaf(r38, r63, r114 * r15);
    r13 = r4 * r104;
    r114 = fmaf(r49, r13, r114);
    r79 = r33 * r33;
    r79 = r79 * r104;
    r79 = r79 * r35;
    r79 = r79 * r93;
    r114 = fmaf(r46, r79, r114);
    r116 = fmaf(r104, r100, r116);
    r116 = r116 + r114;
    r65 = r42 * r52;
    r65 = fmaf(r99, r65, r15 * r124);
    r124 = r4 * r104;
    r124 = r124 * r99;
    r65 = fmaf(r36, r124, r65);
    r65 = fmaf(r104, r87, r65);
    r114 = r114 + r65;
    r116 = fmaf(r8, r114, r58 * r116);
    r124 = r4 * r12;
    r124 = r124 * r51;
    r124 = r124 * r68;
    r116 = fmaf(r50, r124, r116);
    r15 = r42 * r48;
    r116 = fmaf(r69, r15, r116);
    r79 = r104 * r113;
    r13 = r57 * r33;
    r13 = r13 * r104;
    r13 = r13 * r52;
    r116 = fmaf(r73, r13, r116);
    r91 = r72 * r104;
    r91 = r91 * r73;
    r116 = fmaf(r70, r91, r116);
    r94 = r57 * r12;
    r94 = r94 * r51;
    r94 = r94 * r33;
    r94 = r94 * r48;
    r94 = r94 * r48;
    r94 = r94 * r84;
    r94 = r94 * r11;
    r116 = fmaf(r10, r94, r116);
    r123 = r72 * r104;
    r123 = r123 * r69;
    r116 = fmaf(r73, r123, r116);
    r80 = r57 * r38;
    r80 = r80 * r52;
    r116 = fmaf(r99, r80, r116);
    r97 = r51 * r48;
    r55 = r7 * r16;
    r55 = r55 * r56;
    r55 = fmaf(r6, r114, r114 * r55);
    r55 = fmaf(r114, r66, r55);
    r55 = fmaf(r114, r59, r55);
    r97 = r97 * r55;
    r116 = fmaf(r69, r97, r116);
    r90 = r4 * r12;
    r90 = r90 * r51;
    r90 = r90 * r67;
    r90 = r90 * r68;
    r116 = fmaf(r50, r90, r116);
    r116 = fmaf(r104, r92, r116);
    r116 = fmaf(r57, r79, r116);
    r116 = fmaf(r42, r71, r116);
    r116 = fmaf(r42, r64, r116);
    r116 = fmaf(r104, r105, r116);
    r90 = r0 * r116;
    r97 = r12 * r33;
    r97 = r97 * r33;
    r97 = r97 * r48;
    r97 = r97 * r48;
    r97 = r97 * r96;
    r97 = r97 * r11;
    r80 = r38 * r86;
    r80 = r80 * r46;
    r80 = fmaf(r26, r80, r10 * r97);
    r97 = r98 * r104;
    r80 = fmaf(r49, r97, r80);
    r123 = r33 * r33;
    r123 = r123 * r62;
    r123 = r123 * r104;
    r123 = r123 * r35;
    r123 = r123 * r93;
    r80 = fmaf(r46, r123, r80);
    r80 = r80 + r65;
    r114 = fmaf(r9, r114, r57 * r80);
    r80 = r4 * r12;
    r80 = r80 * r33;
    r80 = r80 * r68;
    r114 = fmaf(r50, r80, r114);
    r65 = r4 * r12;
    r65 = r65 * r33;
    r65 = r65 * r67;
    r65 = r65 * r68;
    r114 = fmaf(r50, r65, r114);
    r123 = r33 * r48;
    r123 = r123 * r74;
    r123 = r123 * r104;
    r123 = r123 * r11;
    r123 = r123 * r60;
    r114 = fmaf(r35, r123, r114);
    r97 = r33 * r48;
    r97 = r97 * r67;
    r97 = r97 * r74;
    r97 = r97 * r104;
    r97 = r97 * r11;
    r97 = r97 * r60;
    r114 = fmaf(r35, r97, r114);
    r94 = r38 * r48;
    r114 = fmaf(r69, r94, r114);
    r91 = r58 * r33;
    r91 = r91 * r104;
    r91 = r91 * r52;
    r114 = fmaf(r73, r91, r114);
    r13 = r58 * r42;
    r114 = fmaf(r63, r13, r114);
    r15 = r33 * r72;
    r15 = r15 * r104;
    r15 = r15 * r35;
    r15 = r15 * r93;
    r114 = fmaf(r69, r15, r114);
    r124 = r58 * r38;
    r124 = r124 * r52;
    r114 = fmaf(r99, r124, r114);
    r27 = r55 * r69;
    r114 = fmaf(r26, r27, r114);
    r114 = fmaf(r38, r71, r114);
    r114 = fmaf(r58, r79, r114);
    r114 = fmaf(r12, r111, r114);
    r114 = fmaf(r104, r106, r114);
    r27 = r5 * r114;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r112,
                                          r28,
                                          r90,
                                          r27);
    r27 = r33 * r33;
    r90 = r39 * r33;
    r90 = r90 * r33;
    r28 = r16 * r43;
    r28 = r28 * r33;
    r28 = fmaf(r44, r28, r47 * r90);
    r90 = r39 * r51;
    r90 = r90 * r51;
    r28 = fmaf(r47, r90, r28);
    r112 = r16 * r54;
    r112 = r112 * r51;
    r28 = fmaf(r44, r112, r28);
    r27 = r27 * r28;
    r27 = r27 * r35;
    r27 = r27 * r93;
    r112 = r39 * r33;
    r112 = r112 * r33;
    r112 = r112 * r48;
    r112 = r112 * r48;
    r112 = r112 * r11;
    r112 = fmaf(r47, r112, r46 * r27);
    r27 = r4 * r28;
    r112 = fmaf(r49, r27, r112);
    r112 = fmaf(r43, r63, r112);
    r27 = r39 * r51;
    r27 = r27 * r51;
    r27 = r27 * r48;
    r27 = r27 * r48;
    r27 = r27 * r11;
    r27 = fmaf(r47, r27, r28 * r87);
    r90 = r54 * r52;
    r27 = fmaf(r99, r90, r27);
    r124 = r4 * r28;
    r124 = r124 * r99;
    r27 = fmaf(r36, r124, r27);
    r124 = r112 + r27;
    r90 = r62 * r28;
    r15 = r39 * r51;
    r15 = r15 * r51;
    r15 = r15 * r48;
    r15 = r15 * r48;
    r15 = r15 * r96;
    r15 = r15 * r11;
    r15 = fmaf(r10, r15, r87 * r90);
    r90 = r54 * r51;
    r90 = r90 * r48;
    r90 = r90 * r86;
    r15 = fmaf(r46, r90, r15);
    r15 = fmaf(r28, r100, r15);
    r15 = r15 + r112;
    r15 = fmaf(r58, r15, r8 * r124);
    r112 = r51 * r48;
    r90 = r7 * r16;
    r90 = r90 * r56;
    r90 = fmaf(r124, r90, r6 * r124);
    r90 = fmaf(r124, r66, r90);
    r90 = fmaf(r124, r59, r90);
    r112 = r112 * r90;
    r15 = fmaf(r69, r112, r15);
    r13 = r57 * r43;
    r13 = r13 * r52;
    r15 = fmaf(r99, r13, r15);
    r91 = r54 * r48;
    r15 = fmaf(r69, r91, r15);
    r79 = r57 * r33;
    r79 = r79 * r28;
    r79 = r79 * r52;
    r15 = fmaf(r73, r79, r15);
    r94 = r72 * r28;
    r94 = r94 * r69;
    r15 = fmaf(r73, r94, r15);
    r97 = r57 * r39;
    r97 = r97 * r51;
    r97 = r97 * r33;
    r97 = r97 * r48;
    r97 = r97 * r48;
    r97 = r97 * r84;
    r97 = r97 * r11;
    r15 = fmaf(r10, r97, r15);
    r123 = r57 * r28;
    r15 = fmaf(r113, r123, r15);
    r65 = r72 * r28;
    r65 = r65 * r73;
    r15 = fmaf(r70, r65, r15);
    r80 = r4 * r39;
    r80 = r80 * r51;
    r80 = r80 * r68;
    r15 = fmaf(r50, r80, r15);
    r109 = r4 * r39;
    r109 = r109 * r51;
    r109 = r109 * r67;
    r109 = r109 * r68;
    r15 = fmaf(r50, r109, r15);
    r15 = fmaf(r54, r71, r15);
    r15 = fmaf(r28, r105, r15);
    r15 = fmaf(r28, r92, r15);
    r15 = fmaf(r54, r64, r15);
    r109 = r0 * r15;
    r80 = r33 * r33;
    r80 = r80 * r62;
    r80 = r80 * r28;
    r80 = r80 * r35;
    r80 = r80 * r93;
    r65 = r39 * r33;
    r65 = r65 * r33;
    r65 = r65 * r48;
    r65 = r65 * r48;
    r65 = r65 * r96;
    r65 = r65 * r11;
    r65 = fmaf(r10, r65, r46 * r80);
    r80 = r43 * r86;
    r80 = r80 * r46;
    r65 = fmaf(r26, r80, r65);
    r123 = r98 * r28;
    r65 = fmaf(r49, r123, r65);
    r65 = r65 + r27;
    r65 = fmaf(r57, r65, r9 * r124);
    r124 = r58 * r43;
    r124 = r124 * r52;
    r65 = fmaf(r99, r124, r65);
    r27 = r4 * r39;
    r27 = r27 * r33;
    r27 = r27 * r68;
    r65 = fmaf(r50, r27, r65);
    r123 = r33 * r72;
    r123 = r123 * r28;
    r123 = r123 * r35;
    r123 = r123 * r93;
    r65 = fmaf(r69, r123, r65);
    r80 = r58 * r33;
    r80 = r80 * r28;
    r80 = r80 * r52;
    r65 = fmaf(r73, r80, r65);
    r97 = r58 * r28;
    r65 = fmaf(r113, r97, r65);
    r94 = r43 * r48;
    r65 = fmaf(r69, r94, r65);
    r79 = r58 * r54;
    r65 = fmaf(r63, r79, r65);
    r91 = r4 * r39;
    r91 = r91 * r33;
    r91 = r91 * r67;
    r91 = r91 * r68;
    r65 = fmaf(r50, r91, r65);
    r13 = r33 * r48;
    r13 = r13 * r67;
    r13 = r13 * r74;
    r13 = r13 * r28;
    r13 = r13 * r11;
    r13 = r13 * r60;
    r65 = fmaf(r35, r13, r65);
    r112 = r90 * r69;
    r65 = fmaf(r26, r112, r65);
    r75 = r33 * r48;
    r75 = r75 * r74;
    r75 = r75 * r28;
    r75 = r75 * r11;
    r75 = r75 * r60;
    r65 = fmaf(r35, r75, r65);
    r65 = fmaf(r28, r106, r65);
    r65 = fmaf(r43, r71, r65);
    r65 = fmaf(r39, r111, r65);
    r75 = r5 * r65;
    r112 = r33 * r33;
    r13 = r16 * r45;
    r13 = r13 * r33;
    r91 = r37 * r33;
    r91 = r91 * r33;
    r91 = fmaf(r47, r91, r44 * r13);
    r13 = r16 * r53;
    r13 = r13 * r51;
    r91 = fmaf(r44, r13, r91);
    r44 = r37 * r51;
    r44 = r44 * r51;
    r91 = fmaf(r47, r44, r91);
    r112 = r112 * r91;
    r112 = r112 * r35;
    r112 = r112 * r93;
    r112 = fmaf(r45, r63, r46 * r112);
    r44 = r4 * r91;
    r112 = fmaf(r49, r44, r112);
    r13 = r37 * r33;
    r13 = r13 * r33;
    r13 = r13 * r48;
    r13 = r13 * r48;
    r13 = r13 * r11;
    r112 = fmaf(r47, r13, r112);
    r13 = r37 * r51;
    r13 = r13 * r51;
    r13 = r13 * r48;
    r13 = r13 * r48;
    r13 = r13 * r11;
    r44 = r4 * r91;
    r44 = r44 * r99;
    r44 = fmaf(r36, r44, r47 * r13);
    r13 = r53 * r52;
    r44 = fmaf(r99, r13, r44);
    r44 = fmaf(r91, r87, r44);
    r13 = r112 + r44;
    r36 = r37 * r51;
    r36 = r36 * r51;
    r36 = r36 * r48;
    r36 = r36 * r48;
    r36 = r36 * r96;
    r36 = r36 * r11;
    r100 = fmaf(r91, r100, r10 * r36);
    r36 = r53 * r51;
    r36 = r36 * r48;
    r36 = r36 * r86;
    r100 = fmaf(r46, r36, r100);
    r47 = r62 * r91;
    r100 = fmaf(r87, r47, r100);
    r100 = r100 + r112;
    r100 = fmaf(r58, r100, r8 * r13);
    r8 = r4 * r37;
    r8 = r8 * r51;
    r8 = r8 * r67;
    r8 = r8 * r68;
    r100 = fmaf(r50, r8, r100);
    r112 = r53 * r48;
    r100 = fmaf(r69, r112, r100);
    r47 = r4 * r37;
    r47 = r47 * r51;
    r47 = r47 * r68;
    r100 = fmaf(r50, r47, r100);
    r36 = r72 * r91;
    r36 = r36 * r73;
    r100 = fmaf(r70, r36, r100);
    r70 = r57 * r91;
    r100 = fmaf(r113, r70, r100);
    r87 = r57 * r33;
    r87 = r87 * r91;
    r87 = r87 * r52;
    r100 = fmaf(r73, r87, r100);
    r79 = r57 * r45;
    r79 = r79 * r52;
    r100 = fmaf(r99, r79, r100);
    r94 = r72 * r91;
    r94 = r94 * r69;
    r100 = fmaf(r73, r94, r100);
    r97 = r51 * r48;
    r80 = r7 * r16;
    r80 = r80 * r56;
    r6 = fmaf(r6, r13, r13 * r80);
    r6 = fmaf(r13, r66, r6);
    r6 = fmaf(r13, r59, r6);
    r97 = r97 * r6;
    r100 = fmaf(r69, r97, r100);
    r59 = r57 * r37;
    r59 = r59 * r51;
    r59 = r59 * r33;
    r59 = r59 * r48;
    r59 = r59 * r48;
    r59 = r59 * r84;
    r59 = r59 * r11;
    r100 = fmaf(r10, r59, r100);
    r100 = fmaf(r53, r71, r100);
    r100 = fmaf(r91, r105, r100);
    r100 = fmaf(r53, r64, r100);
    r100 = fmaf(r91, r92, r100);
    r59 = r0 * r100;
    r97 = r33 * r33;
    r97 = r97 * r62;
    r97 = r97 * r91;
    r97 = r97 * r35;
    r97 = r97 * r93;
    r94 = r45 * r86;
    r94 = r94 * r46;
    r94 = fmaf(r26, r94, r46 * r97);
    r97 = r98 * r91;
    r94 = fmaf(r49, r97, r94);
    r49 = r37 * r33;
    r49 = r49 * r33;
    r49 = r49 * r48;
    r49 = r49 * r48;
    r49 = r49 * r96;
    r49 = r49 * r11;
    r94 = fmaf(r10, r49, r94);
    r94 = r94 + r44;
    r94 = fmaf(r57, r94, r9 * r13);
    r13 = r33 * r72;
    r13 = r13 * r91;
    r13 = r13 * r35;
    r13 = r13 * r93;
    r94 = fmaf(r69, r13, r94);
    r93 = r33 * r48;
    r93 = r93 * r67;
    r93 = r93 * r74;
    r93 = r93 * r91;
    r93 = r93 * r11;
    r93 = r93 * r60;
    r94 = fmaf(r35, r93, r94);
    r9 = r33 * r48;
    r9 = r9 * r74;
    r9 = r9 * r91;
    r9 = r9 * r11;
    r9 = r9 * r60;
    r94 = fmaf(r35, r9, r94);
    r35 = r58 * r91;
    r94 = fmaf(r113, r35, r94);
    r113 = r58 * r33;
    r113 = r113 * r91;
    r113 = r113 * r52;
    r94 = fmaf(r73, r113, r94);
    r73 = r58 * r45;
    r73 = r73 * r52;
    r94 = fmaf(r99, r73, r94);
    r99 = r4 * r37;
    r99 = r99 * r33;
    r99 = r99 * r67;
    r99 = r99 * r68;
    r94 = fmaf(r50, r99, r94);
    r67 = r58 * r53;
    r94 = fmaf(r63, r67, r94);
    r63 = r45 * r48;
    r94 = fmaf(r69, r63, r94);
    r60 = r4 * r37;
    r60 = r60 * r33;
    r60 = r60 * r68;
    r94 = fmaf(r50, r60, r94);
    r50 = r6 * r69;
    r94 = fmaf(r26, r50, r94);
    r94 = fmaf(r91, r106, r94);
    r94 = fmaf(r45, r71, r94);
    r94 = fmaf(r37, r111, r94);
    r111 = r5 * r94;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r109,
                                          r75,
                                          r59,
                                          r111);
    r111 = r0 * r4;
    r111 = r111 * r2;
    r61 = r4 * r61;
    r59 = r5 * r61;
    r111 = fmaf(r101, r59, r1 * r111);
    r75 = r0 * r4;
    r75 = r75 * r2;
    r75 = fmaf(r120, r59, r102 * r75);
    r109 = r0 * r4;
    r109 = r109 * r2;
    r109 = fmaf(r31, r59, r14 * r109);
    r50 = r0 * r4;
    r50 = r50 * r2;
    r50 = fmaf(r114, r59, r116 * r50);
    WriteSum4<float, float>((float*)inout_shared, r111, r75, r109, r50);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r0 * r4;
    r50 = r50 * r2;
    r50 = fmaf(r65, r59, r15 * r50);
    r109 = r0 * r4;
    r109 = r109 * r2;
    r59 = fmaf(r94, r59, r100 * r109);
    WriteSum2<float, float>((float*)inout_shared, r50, r59);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = r0 * r0;
    r50 = r1 * r59;
    r5 = r5 * r5;
    r109 = r101 * r5;
    r101 = fmaf(r101, r109, r1 * r50);
    r1 = r102 * r102;
    r75 = r120 * r120;
    r75 = fmaf(r5, r75, r59 * r1);
    r1 = r31 * r31;
    r111 = r14 * r14;
    r111 = fmaf(r59, r111, r5 * r1);
    r1 = r116 * r116;
    r60 = r114 * r114;
    r60 = fmaf(r5, r60, r59 * r1);
    WriteSum4<float, float>((float*)inout_shared, r101, r75, r111, r60);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = r15 * r15;
    r111 = r65 * r65;
    r111 = fmaf(r5, r111, r59 * r60);
    r60 = r94 * r94;
    r75 = r100 * r100;
    r75 = fmaf(r59, r75, r5 * r60);
    WriteSum2<float, float>((float*)inout_shared, r111, r75);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = fmaf(r102, r50, r120 * r109);
    r111 = fmaf(r31, r109, r14 * r50);
    r60 = fmaf(r116, r50, r114 * r109);
    r101 = fmaf(r15, r50, r65 * r109);
    WriteSum4<float, float>((float*)inout_shared, r75, r111, r60, r101);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r109 = fmaf(r94, r109, r100 * r50);
    r50 = r120 * r31;
    r101 = r102 * r14;
    r101 = fmaf(r59, r101, r5 * r50);
    r50 = r102 * r116;
    r60 = r120 * r114;
    r60 = fmaf(r5, r60, r59 * r50);
    r50 = r102 * r15;
    r111 = r120 * r65;
    r111 = fmaf(r5, r111, r59 * r50);
    WriteSum4<float, float>((float*)inout_shared, r109, r101, r60, r111);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r111 = r120 * r94;
    r60 = r102 * r100;
    r60 = fmaf(r59, r60, r5 * r111);
    r111 = r31 * r114;
    r101 = r14 * r116;
    r101 = fmaf(r59, r101, r5 * r111);
    r111 = r14 * r15;
    r109 = r31 * r65;
    r109 = fmaf(r5, r109, r59 * r111);
    r111 = r31 * r94;
    r50 = r14 * r100;
    r50 = fmaf(r59, r50, r5 * r111);
    WriteSum4<float, float>((float*)inout_shared, r60, r101, r109, r50);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r114 * r65;
    r109 = r116 * r15;
    r109 = fmaf(r59, r109, r5 * r50);
    r50 = r116 * r100;
    r101 = r114 * r94;
    r101 = fmaf(r5, r101, r59 * r50);
    r50 = r65 * r94;
    r60 = r15 * r100;
    r60 = fmaf(r59, r60, r5 * r50);
    WriteSum3<float, float>((float*)inout_shared, r109, r101, r60);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r4 * r2;
    WriteSum2<float, float>((float*)inout_shared, r2, r61);
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
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void ThinPrismFisheyeSplitFixedFocalAndExtraFixedPointResJacFirst(
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
  ThinPrismFisheyeSplitFixedFocalAndExtraFixedPointResJacFirstKernel<<<n_blocks,
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