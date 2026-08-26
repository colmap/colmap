#include "kernel_thin_prism_fisheye_split_fixed_focal_and_extra_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedFocalAndExtraResJacFirstKernel(
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
    ReadIdx4<1024, float, float, float4>(focal_and_extra,
                                         4 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r66,
                                         r67,
                                         r68,
                                         r69);
    r70 = r10 * r10;
    r71 = 3.00000000000000000e+00;
    r70 = r70 * r59;
    r70 = r70 * r71;
    r70 = fmaf(r61, r70, r62);
    r70 = fmaf(r67, r70, r8 * r65);
    r62 = r16 * r61;
    r72 = r66 * r62;
    r73 = r63 * r72;
    r74 = r65 * r65;
    r75 = r65 * r74;
    r76 = fmaf(r68, r75, r6 * r65);
    r75 = r69 * r75;
    r76 = fmaf(r65, r75, r76);
    r76 = fmaf(r7, r74, r76);
    r69 = 1.0 / r36;
    r77 = 1.0 / r51;
    r78 = r69 * r77;
    r79 = r76 * r78;
    r80 = r59 * r79;
    r81 = r10 * r59;
    r70 = fmaf(r78, r81, r70);
    r70 = fmaf(r10, r73, r70);
    r70 = fmaf(r10, r80, r70);
    r2 = fmaf(r0, r70, r2);
    r70 = r11 * r71;
    r70 = r70 * r61;
    r70 = fmaf(r63, r70, r64);
    r70 = fmaf(r66, r70, r9 * r65);
    r64 = r67 * r10;
    r64 = r64 * r63;
    r70 = fmaf(r62, r64, r70);
    r70 = fmaf(r63, r79, r70);
    r70 = fmaf(r78, r63, r70);
    r70 = fmaf(r5, r70, r1);
    r70 = fmaf(r3, r4, r70);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r70);
    r3 = fmaf(r70, r70, r2 * r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r3);
  if (global_thread_idx < problem_size) {
    r3 = r10 * r59;
    r1 = r16 * r34;
    r64 = r19 * r24;
    r81 = 5.00000000000000000e-01;
    r82 = r18 * r21;
    r82 = fmaf(r81, r82, r81 * r64);
    r64 = r17 * r22;
    r83 = -5.00000000000000000e-01;
    r82 = fmaf(r83, r64, r82);
    r84 = r20 * r23;
    r82 = fmaf(r81, r84, r82);
    r84 = r17 * r24;
    r64 = r20 * r21;
    r64 = fmaf(r83, r64, r83 * r84);
    r84 = r19 * r22;
    r64 = fmaf(r83, r84, r64);
    r85 = r18 * r23;
    r64 = fmaf(r81, r85, r64);
    r85 = r29 * r64;
    r1 = fmaf(r16, r85, r82 * r1);
    r84 = r16 * r25;
    r86 = fmaf(r81, r31, r83 * r27);
    r86 = fmaf(r83, r28, r86);
    r86 = fmaf(r83, r30, r86);
    r87 = r16 * r32;
    r88 = r20 * r24;
    r89 = r17 * r21;
    r89 = fmaf(r83, r89, r81 * r88);
    r88 = r18 * r22;
    r89 = fmaf(r83, r88, r89);
    r90 = r19 * r23;
    r89 = fmaf(r83, r90, r89);
    r87 = r87 * r89;
    r84 = fmaf(r86, r84, r87);
    r1 = r1 + r84;
    r90 = r16 * r29;
    r90 = r90 * r89;
    r88 = r16 * r25;
    r88 = r88 * r82;
    r91 = r90 + r88;
    r92 = r32 * r41;
    r91 = fmaf(r64, r92, r91);
    r93 = r41 * r34;
    r91 = fmaf(r86, r93, r91);
    r91 = fmaf(r14, r91, r15 * r1);
    r1 = r29 * r82;
    r93 = -4.00000000000000000e+00;
    r1 = r1 * r93;
    r92 = r32 * r86;
    r94 = r93 * r92;
    r95 = r1 + r94;
    r91 = fmaf(r13, r95, r91);
    r95 = 6.00000000000000000e+00;
    r3 = r3 * r91;
    r3 = r3 * r95;
    r96 = r16 * r11;
    r97 = r25 * r41;
    r98 = r34 * r89;
    r99 = r41 * r98;
    r97 = fmaf(r64, r97, r99);
    r100 = r16 * r29;
    r100 = r100 * r86;
    r101 = r16 * r32;
    r101 = fmaf(r82, r101, r100);
    r97 = r97 + r101;
    r102 = r25 * r89;
    r102 = r102 * r93;
    r94 = r102 + r94;
    r94 = fmaf(r14, r94, r15 * r97);
    r97 = r16 * r34;
    r97 = fmaf(r86, r97, r88);
    r88 = r16 * r32;
    r88 = fmaf(r64, r88, r90);
    r97 = r97 + r88;
    r94 = fmaf(r13, r97, r94);
    r96 = r96 * r94;
    r97 = r16 * r10;
    r97 = r97 * r91;
    r97 = fmaf(r37, r97, r37 * r96);
    r96 = r11 * r11;
    r90 = r16 * r25;
    r90 = r90 * r64;
    r98 = r16 * r98;
    r103 = r90 + r98;
    r101 = r101 + r103;
    r104 = r41 * r34;
    r104 = fmaf(r41, r85, r82 * r104);
    r104 = r104 + r84;
    r104 = fmaf(r13, r104, r14 * r101);
    r1 = r102 + r1;
    r104 = fmaf(r15, r1, r104);
    r12 = r36 * r12;
    r12 = 1.0 / r12;
    r36 = r41 * r12;
    r96 = r96 * r104;
    r97 = fmaf(r36, r96, r97);
    r1 = r10 * r10;
    r1 = r1 * r104;
    r97 = fmaf(r36, r1, r97);
    r1 = r71 * r97;
    r96 = r10 * r61;
    r102 = r42 + r35;
    r102 = 1.0 / r102;
    r35 = rsqrtf(r35);
    r101 = r10 * r35;
    r82 = r102 * r101;
    r96 = r96 * r82;
    r1 = fmaf(r96, r1, r61 * r3);
    r3 = r59 * r59;
    r105 = -6.00000000000000000e+00;
    r3 = r3 * r104;
    r3 = r3 * r105;
    r3 = r3 * r47;
    r3 = r3 * r12;
    r106 = r10 * r10;
    r106 = r106 * r36;
    r107 = -3.00000000000000000e+00;
    r108 = r107 * r101;
    r52 = r51 * r52;
    r52 = 1.0 / r52;
    r52 = r52 * r56;
    r51 = r10 * r59;
    r108 = r108 * r52;
    r108 = r108 * r51;
    r109 = r94 * r63;
    r110 = r11 * r11;
    r110 = r110 * r97;
    r110 = r110 * r35;
    r110 = r110 * r102;
    r110 = fmaf(r61, r110, r62 * r109);
    r109 = r4 * r97;
    r111 = r63 * r52;
    r112 = r11 * r35;
    r109 = r109 * r111;
    r110 = fmaf(r112, r109, r110);
    r113 = r11 * r11;
    r113 = r113 * r59;
    r113 = r113 * r59;
    r113 = r113 * r104;
    r113 = r113 * r47;
    r110 = fmaf(r36, r113, r110);
    r1 = fmaf(r3, r106, r1);
    r1 = fmaf(r97, r108, r1);
    r1 = r1 + r110;
    r106 = r91 * r62;
    r106 = fmaf(r97, r96, r51 * r106);
    r113 = r10 * r10;
    r113 = r113 * r59;
    r113 = r113 * r59;
    r113 = r113 * r104;
    r113 = r113 * r47;
    r106 = fmaf(r36, r113, r106);
    r109 = r4 * r97;
    r109 = r109 * r101;
    r109 = r109 * r52;
    r106 = fmaf(r51, r109, r106);
    r110 = r110 + r106;
    r1 = fmaf(r8, r110, r67 * r1);
    r109 = r59 * r83;
    r109 = r109 * r47;
    r109 = r109 * r69;
    r109 = r109 * r101;
    r113 = r76 * r109;
    r114 = r11 * r97;
    r114 = r114 * r82;
    r1 = fmaf(r72, r114, r1);
    r115 = r66 * r10;
    r115 = r115 * r11;
    r115 = r115 * r59;
    r115 = r115 * r59;
    r115 = r115 * r93;
    r115 = r115 * r104;
    r115 = r115 * r47;
    r1 = fmaf(r12, r115, r1);
    r116 = r81 * r97;
    r116 = r116 * r82;
    r1 = fmaf(r79, r116, r1);
    r117 = r59 * r91;
    r1 = fmaf(r78, r117, r1);
    r118 = r4 * r10;
    r118 = r118 * r104;
    r118 = r118 * r77;
    r1 = fmaf(r56, r118, r1);
    r119 = r4 * r10;
    r119 = r119 * r76;
    r119 = r119 * r104;
    r119 = r119 * r77;
    r1 = fmaf(r56, r119, r1);
    r120 = r81 * r97;
    r120 = r120 * r78;
    r1 = fmaf(r82, r120, r1);
    r121 = r41 * r101;
    r121 = r121 * r111;
    r122 = r66 * r121;
    r123 = r10 * r59;
    r124 = r7 * r16;
    r124 = r124 * r65;
    r124 = fmaf(r110, r124, r6 * r110);
    r125 = 4.00000000000000000e+00;
    r75 = r125 * r75;
    r68 = r68 * r71;
    r68 = r68 * r74;
    r124 = fmaf(r110, r75, r124);
    r124 = fmaf(r110, r68, r124);
    r123 = r123 * r124;
    r1 = fmaf(r78, r123, r1);
    r74 = r94 * r51;
    r1 = fmaf(r72, r74, r1);
    r1 = fmaf(r97, r113, r1);
    r1 = fmaf(r97, r109, r1);
    r1 = fmaf(r91, r73, r1);
    r1 = fmaf(r97, r122, r1);
    r1 = fmaf(r91, r80, r1);
    r74 = r0 * r1;
    r123 = r94 * r95;
    r123 = r123 * r61;
    r120 = r11 * r11;
    r120 = r120 * r71;
    r120 = r120 * r97;
    r120 = r120 * r35;
    r120 = r120 * r102;
    r120 = fmaf(r61, r120, r63 * r123);
    r123 = r107 * r97;
    r123 = r123 * r111;
    r120 = fmaf(r112, r123, r120);
    r119 = r41 * r26;
    r118 = r11 * r11;
    r119 = r119 * r118;
    r119 = r119 * r12;
    r120 = fmaf(r3, r119, r120);
    r120 = r120 + r106;
    r110 = fmaf(r9, r110, r66 * r120);
    r120 = r11 * r81;
    r120 = r120 * r97;
    r120 = r120 * r35;
    r120 = r120 * r102;
    r110 = fmaf(r78, r120, r110);
    r106 = r11 * r59;
    r106 = r106 * r76;
    r106 = r106 * r83;
    r106 = r106 * r97;
    r106 = r106 * r47;
    r106 = r106 * r69;
    r110 = fmaf(r35, r106, r110);
    r3 = r124 * r78;
    r110 = fmaf(r63, r3, r110);
    r123 = r4 * r11;
    r123 = r123 * r76;
    r123 = r123 * r104;
    r123 = r123 * r77;
    r110 = fmaf(r56, r123, r110);
    r118 = r67 * r10;
    r118 = r118 * r11;
    r118 = r118 * r59;
    r118 = r118 * r59;
    r118 = r118 * r93;
    r118 = r118 * r47;
    r118 = r118 * r12;
    r117 = r67 * r91;
    r117 = r117 * r63;
    r110 = fmaf(r62, r117, r110);
    r116 = r67 * r11;
    r116 = r116 * r97;
    r116 = r116 * r62;
    r110 = fmaf(r82, r116, r110);
    r115 = r4 * r11;
    r115 = r115 * r104;
    r115 = r115 * r77;
    r110 = fmaf(r56, r115, r110);
    r114 = r67 * r97;
    r110 = fmaf(r121, r114, r110);
    r125 = r81 * r102;
    r125 = r125 * r79;
    r125 = r125 * r112;
    r126 = r67 * r94;
    r126 = r126 * r62;
    r110 = fmaf(r51, r126, r110);
    r127 = r11 * r59;
    r127 = r127 * r83;
    r127 = r127 * r97;
    r127 = r127 * r47;
    r127 = r127 * r69;
    r110 = fmaf(r35, r127, r110);
    r128 = r59 * r94;
    r110 = fmaf(r78, r128, r110);
    r110 = fmaf(r104, r118, r110);
    r110 = fmaf(r97, r125, r110);
    r110 = fmaf(r94, r80, r110);
    r128 = r5 * r110;
    r127 = r10 * r10;
    r126 = r37 * r127;
    r114 = r16 * r10;
    r98 = r100 + r98;
    r100 = r16 * r32;
    r115 = r19 * r24;
    r116 = r18 * r21;
    r116 = fmaf(r83, r116, r83 * r115);
    r115 = r17 * r22;
    r116 = fmaf(r81, r115, r116);
    r117 = r20 * r23;
    r116 = fmaf(r83, r117, r116);
    r100 = r100 * r116;
    r117 = r16 * r25;
    r115 = r17 * r24;
    r104 = r20 * r21;
    r104 = fmaf(r81, r104, r81 * r115);
    r115 = r19 * r22;
    r104 = fmaf(r81, r115, r104);
    r123 = r18 * r23;
    r104 = fmaf(r83, r123, r104);
    r117 = fmaf(r104, r117, r100);
    r98 = r98 + r117;
    r123 = r32 * r93;
    r123 = r123 * r104;
    r115 = r29 * r89;
    r115 = r115 * r93;
    r3 = r123 + r115;
    r3 = fmaf(r13, r3, r15 * r98);
    r98 = r41 * r34;
    r98 = fmaf(r41, r92, r104 * r98);
    r106 = r16 * r25;
    r106 = r106 * r89;
    r120 = r16 * r29;
    r120 = fmaf(r116, r120, r106);
    r98 = r98 + r120;
    r3 = fmaf(r14, r98, r3);
    r114 = r114 * r3;
    r98 = r10 * r10;
    r129 = r41 * r29;
    r129 = fmaf(r86, r129, r99);
    r129 = r129 + r117;
    r117 = r16 * r29;
    r117 = r117 * r104;
    r130 = r16 * r34;
    r130 = fmaf(r116, r130, r117);
    r130 = r130 + r84;
    r130 = fmaf(r14, r130, r13 * r129);
    r129 = r25 * r116;
    r84 = r93 * r129;
    r115 = r115 + r84;
    r130 = fmaf(r15, r115, r130);
    r98 = r98 * r130;
    r98 = fmaf(r36, r98, r37 * r114);
    r114 = r11 * r11;
    r114 = r114 * r130;
    r98 = fmaf(r36, r114, r98);
    r115 = r16 * r11;
    r131 = r25 * r41;
    r131 = fmaf(r86, r131, r87);
    r87 = r41 * r34;
    r131 = fmaf(r116, r87, r131);
    r131 = r131 + r117;
    r87 = r16 * r34;
    r92 = fmaf(r16, r92, r104 * r87);
    r92 = r92 + r120;
    r92 = fmaf(r13, r92, r15 * r131);
    r84 = r123 + r84;
    r92 = fmaf(r14, r84, r92);
    r115 = r115 * r92;
    r98 = fmaf(r37, r115, r98);
    r126 = r126 * r47;
    r126 = r126 * r59;
    r126 = r126 * r35;
    r126 = r126 * r102;
    r126 = r126 * r98;
    r115 = r4 * r98;
    r115 = r115 * r101;
    r115 = r115 * r52;
    r115 = fmaf(r51, r115, r126);
    r114 = r10 * r10;
    r114 = r114 * r59;
    r114 = r114 * r59;
    r114 = r114 * r130;
    r114 = r114 * r47;
    r115 = fmaf(r36, r114, r115);
    r84 = r3 * r62;
    r115 = fmaf(r51, r84, r115);
    r84 = r92 * r63;
    r114 = r4 * r98;
    r114 = r114 * r111;
    r114 = fmaf(r112, r114, r62 * r84);
    r84 = r11 * r11;
    r84 = r84 * r59;
    r84 = r84 * r59;
    r84 = r84 * r130;
    r84 = r84 * r47;
    r114 = fmaf(r36, r84, r114);
    r123 = r11 * r11;
    r123 = r123 * r98;
    r123 = r123 * r35;
    r123 = r123 * r102;
    r114 = fmaf(r61, r123, r114);
    r123 = r115 + r114;
    r126 = fmaf(r98, r108, r71 * r126);
    r84 = r10 * r10;
    r84 = r84 * r59;
    r84 = r84 * r59;
    r84 = r84 * r105;
    r84 = r84 * r130;
    r84 = r84 * r47;
    r126 = fmaf(r12, r84, r126);
    r131 = r10 * r59;
    r131 = r131 * r95;
    r131 = r131 * r3;
    r126 = fmaf(r61, r131, r126);
    r126 = r126 + r114;
    r126 = fmaf(r67, r126, r8 * r123);
    r114 = r92 * r51;
    r126 = fmaf(r72, r114, r126);
    r131 = r66 * r10;
    r131 = r131 * r11;
    r131 = r131 * r59;
    r131 = r131 * r59;
    r131 = r131 * r93;
    r131 = r131 * r130;
    r131 = r131 * r47;
    r126 = fmaf(r12, r131, r126);
    r84 = r4 * r10;
    r84 = r84 * r130;
    r84 = r84 * r77;
    r126 = fmaf(r56, r84, r126);
    r87 = r81 * r98;
    r87 = r87 * r82;
    r126 = fmaf(r79, r87, r126);
    r104 = r11 * r98;
    r104 = r104 * r82;
    r126 = fmaf(r72, r104, r126);
    r117 = r81 * r98;
    r117 = r117 * r78;
    r126 = fmaf(r82, r117, r126);
    r86 = r4 * r10;
    r86 = r86 * r76;
    r86 = r86 * r130;
    r86 = r86 * r77;
    r126 = fmaf(r56, r86, r126);
    r132 = r10 * r59;
    r133 = r7 * r16;
    r133 = r133 * r65;
    r133 = fmaf(r123, r133, r6 * r123);
    r133 = fmaf(r123, r75, r133);
    r133 = fmaf(r123, r68, r133);
    r132 = r132 * r133;
    r126 = fmaf(r78, r132, r126);
    r134 = r59 * r3;
    r126 = fmaf(r78, r134, r126);
    r126 = fmaf(r98, r109, r126);
    r126 = fmaf(r3, r80, r126);
    r126 = fmaf(r98, r122, r126);
    r126 = fmaf(r3, r73, r126);
    r126 = fmaf(r98, r113, r126);
    r134 = r0 * r126;
    r132 = r95 * r92;
    r132 = r132 * r61;
    r86 = r107 * r98;
    r86 = r86 * r111;
    r86 = fmaf(r112, r86, r63 * r132);
    r132 = r11 * r11;
    r132 = r132 * r59;
    r132 = r132 * r59;
    r132 = r132 * r105;
    r132 = r132 * r130;
    r132 = r132 * r47;
    r86 = fmaf(r12, r132, r86);
    r117 = r11 * r11;
    r117 = r117 * r71;
    r117 = r117 * r98;
    r117 = r117 * r35;
    r117 = r117 * r102;
    r86 = fmaf(r61, r117, r86);
    r86 = r86 + r115;
    r86 = fmaf(r66, r86, r9 * r123);
    r123 = r4 * r11;
    r123 = r123 * r130;
    r123 = r123 * r77;
    r86 = fmaf(r56, r123, r86);
    r115 = r67 * r92;
    r115 = r115 * r62;
    r86 = fmaf(r51, r115, r86);
    r117 = r11 * r59;
    r117 = r117 * r76;
    r117 = r117 * r83;
    r117 = r117 * r98;
    r117 = r117 * r47;
    r117 = r117 * r69;
    r86 = fmaf(r35, r117, r86);
    r132 = r133 * r78;
    r86 = fmaf(r63, r132, r86);
    r104 = r11 * r81;
    r104 = r104 * r98;
    r104 = r104 * r35;
    r104 = r104 * r102;
    r86 = fmaf(r78, r104, r86);
    r87 = r11 * r59;
    r87 = r87 * r83;
    r87 = r87 * r98;
    r87 = r87 * r47;
    r87 = r87 * r69;
    r86 = fmaf(r35, r87, r86);
    r84 = r67 * r98;
    r86 = fmaf(r121, r84, r86);
    r131 = r67 * r11;
    r131 = r131 * r98;
    r131 = r131 * r62;
    r86 = fmaf(r82, r131, r86);
    r114 = r67 * r3;
    r114 = r114 * r63;
    r86 = fmaf(r62, r114, r86);
    r135 = r4 * r11;
    r135 = r135 * r76;
    r135 = r135 * r130;
    r135 = r135 * r77;
    r86 = fmaf(r56, r135, r86);
    r136 = r59 * r92;
    r86 = fmaf(r78, r136, r86);
    r86 = fmaf(r130, r118, r86);
    r86 = fmaf(r98, r125, r86);
    r86 = fmaf(r92, r80, r86);
    r136 = r5 * r86;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r74,
                                          r128,
                                          r134,
                                          r136);
    r136 = r11 * r11;
    r134 = r10 * r10;
    r128 = r25 * r93;
    r31 = fmaf(r83, r31, r81 * r27);
    r31 = fmaf(r81, r28, r31);
    r31 = fmaf(r81, r30, r31);
    r128 = r128 * r31;
    r85 = r93 * r85;
    r30 = r128 + r85;
    r28 = r16 * r32;
    r28 = r28 * r31;
    r106 = r106 + r28;
    r27 = r41 * r29;
    r106 = fmaf(r116, r27, r106);
    r74 = r41 * r34;
    r106 = fmaf(r64, r74, r106);
    r106 = fmaf(r13, r106, r15 * r30);
    r30 = r16 * r34;
    r30 = fmaf(r16, r129, r31 * r30);
    r30 = r30 + r88;
    r106 = fmaf(r14, r30, r106);
    r134 = r134 * r106;
    r30 = r16 * r10;
    r74 = r16 * r29;
    r74 = r74 * r31;
    r90 = r90 + r74;
    r27 = r32 * r41;
    r90 = fmaf(r116, r27, r90);
    r90 = r90 + r99;
    r89 = r32 * r89;
    r89 = r89 * r93;
    r85 = r89 + r85;
    r85 = fmaf(r13, r85, r14 * r90);
    r90 = r16 * r34;
    r90 = fmaf(r64, r90, r28);
    r90 = r90 + r120;
    r85 = fmaf(r15, r90, r85);
    r30 = r30 * r85;
    r30 = fmaf(r37, r30, r36 * r134);
    r134 = r11 * r11;
    r134 = r134 * r106;
    r30 = fmaf(r36, r134, r30);
    r90 = r16 * r11;
    r74 = r100 + r74;
    r74 = r74 + r103;
    r103 = r41 * r34;
    r129 = fmaf(r41, r129, r31 * r103);
    r129 = r129 + r88;
    r129 = fmaf(r15, r129, r13 * r74);
    r89 = r128 + r89;
    r129 = fmaf(r14, r89, r129);
    r90 = r90 * r129;
    r30 = fmaf(r37, r90, r30);
    r136 = r136 * r30;
    r136 = r136 * r35;
    r136 = r136 * r102;
    r90 = r129 * r63;
    r90 = fmaf(r62, r90, r61 * r136);
    r136 = r11 * r11;
    r136 = r136 * r59;
    r136 = r136 * r59;
    r136 = r136 * r106;
    r136 = r136 * r47;
    r90 = fmaf(r36, r136, r90);
    r134 = r4 * r30;
    r134 = r134 * r111;
    r90 = fmaf(r112, r134, r90);
    r134 = r10 * r10;
    r134 = r134 * r59;
    r134 = r134 * r59;
    r134 = r134 * r106;
    r134 = r134 * r47;
    r134 = fmaf(r30, r96, r36 * r134);
    r136 = r4 * r30;
    r136 = r136 * r101;
    r136 = r136 * r52;
    r134 = fmaf(r51, r136, r134);
    r89 = r85 * r62;
    r134 = fmaf(r51, r89, r134);
    r89 = r90 + r134;
    r136 = r10 * r10;
    r136 = r136 * r59;
    r136 = r136 * r59;
    r136 = r136 * r105;
    r136 = r136 * r106;
    r136 = r136 * r47;
    r14 = r71 * r30;
    r14 = fmaf(r96, r14, r12 * r136);
    r136 = r10 * r59;
    r136 = r136 * r95;
    r136 = r136 * r85;
    r14 = fmaf(r61, r136, r14);
    r14 = fmaf(r30, r108, r14);
    r14 = r14 + r90;
    r14 = fmaf(r67, r14, r8 * r89);
    r90 = r10 * r59;
    r136 = r7 * r16;
    r136 = r136 * r65;
    r136 = fmaf(r89, r136, r6 * r89);
    r136 = fmaf(r89, r68, r136);
    r136 = fmaf(r89, r75, r136);
    r90 = r90 * r136;
    r14 = fmaf(r78, r90, r14);
    r128 = r66 * r10;
    r128 = r128 * r11;
    r128 = r128 * r59;
    r128 = r128 * r59;
    r128 = r128 * r93;
    r128 = r128 * r106;
    r128 = r128 * r47;
    r14 = fmaf(r12, r128, r14);
    r15 = r81 * r30;
    r15 = r15 * r78;
    r14 = fmaf(r82, r15, r14);
    r74 = r4 * r10;
    r74 = r74 * r106;
    r74 = r74 * r77;
    r14 = fmaf(r56, r74, r14);
    r13 = r81 * r30;
    r13 = r13 * r82;
    r14 = fmaf(r79, r13, r14);
    r88 = r129 * r51;
    r14 = fmaf(r72, r88, r14);
    r103 = r4 * r10;
    r103 = r103 * r76;
    r103 = r103 * r106;
    r103 = r103 * r77;
    r14 = fmaf(r56, r103, r14);
    r31 = r59 * r85;
    r14 = fmaf(r78, r31, r14);
    r100 = r11 * r30;
    r100 = r100 * r82;
    r14 = fmaf(r72, r100, r14);
    r14 = fmaf(r30, r113, r14);
    r14 = fmaf(r30, r122, r14);
    r14 = fmaf(r85, r80, r14);
    r14 = fmaf(r85, r73, r14);
    r14 = fmaf(r30, r109, r14);
    r100 = r0 * r14;
    r31 = r11 * r11;
    r31 = r31 * r71;
    r31 = r31 * r30;
    r31 = r31 * r35;
    r31 = r31 * r102;
    r103 = r95 * r129;
    r103 = r103 * r61;
    r103 = fmaf(r63, r103, r61 * r31);
    r31 = r11 * r11;
    r31 = r31 * r59;
    r31 = r31 * r59;
    r31 = r31 * r105;
    r31 = r31 * r106;
    r31 = r31 * r47;
    r103 = fmaf(r12, r31, r103);
    r88 = r107 * r30;
    r88 = r88 * r111;
    r103 = fmaf(r112, r88, r103);
    r103 = r103 + r134;
    r103 = fmaf(r66, r103, r9 * r89);
    r89 = r11 * r59;
    r89 = r89 * r83;
    r89 = r89 * r30;
    r89 = r89 * r47;
    r89 = r89 * r69;
    r103 = fmaf(r35, r89, r103);
    r134 = r11 * r81;
    r134 = r134 * r30;
    r134 = r134 * r35;
    r134 = r134 * r102;
    r103 = fmaf(r78, r134, r103);
    r88 = r4 * r11;
    r88 = r88 * r76;
    r88 = r88 * r106;
    r88 = r88 * r77;
    r103 = fmaf(r56, r88, r103);
    r31 = r67 * r30;
    r103 = fmaf(r121, r31, r103);
    r13 = r67 * r85;
    r13 = r13 * r63;
    r103 = fmaf(r62, r13, r103);
    r74 = r11 * r59;
    r74 = r74 * r76;
    r74 = r74 * r83;
    r74 = r74 * r30;
    r74 = r74 * r47;
    r74 = r74 * r69;
    r103 = fmaf(r35, r74, r103);
    r15 = r67 * r129;
    r15 = r15 * r62;
    r103 = fmaf(r51, r15, r103);
    r128 = r4 * r11;
    r128 = r128 * r106;
    r128 = r128 * r77;
    r103 = fmaf(r56, r128, r103);
    r90 = r67 * r11;
    r90 = r90 * r30;
    r90 = r90 * r62;
    r103 = fmaf(r82, r90, r103);
    r120 = r59 * r129;
    r103 = fmaf(r78, r120, r103);
    r28 = r136 * r78;
    r103 = fmaf(r63, r28, r103);
    r103 = fmaf(r106, r118, r103);
    r103 = fmaf(r129, r80, r103);
    r103 = fmaf(r30, r125, r103);
    r28 = r5 * r103;
    r120 = r26 * r10;
    r120 = r120 * r10;
    r120 = r120 * r59;
    r120 = r120 * r59;
    r120 = r120 * r105;
    r120 = r120 * r47;
    r90 = r43 * r10;
    r90 = r90 * r59;
    r90 = r90 * r95;
    r90 = fmaf(r61, r90, r12 * r120);
    r120 = r41 * r26;
    r120 = r120 * r127;
    r120 = r120 * r12;
    r127 = r119 + r120;
    r128 = r16 * r39;
    r128 = r128 * r11;
    r127 = fmaf(r37, r128, r127);
    r15 = r16 * r43;
    r15 = r15 * r10;
    r127 = fmaf(r37, r15, r127);
    r15 = r71 * r127;
    r90 = fmaf(r96, r15, r90);
    r128 = r59 * r59;
    r128 = r128 * r47;
    r74 = r39 * r63;
    r74 = fmaf(r62, r74, r119 * r128);
    r119 = r4 * r127;
    r119 = r119 * r111;
    r74 = fmaf(r112, r119, r74);
    r13 = r11 * r11;
    r13 = r13 * r127;
    r13 = r13 * r35;
    r13 = r13 * r102;
    r74 = fmaf(r61, r13, r74);
    r90 = fmaf(r127, r108, r90);
    r90 = r90 + r74;
    r15 = r43 * r62;
    r15 = fmaf(r51, r15, r128 * r120);
    r120 = r4 * r127;
    r120 = r120 * r101;
    r120 = r120 * r52;
    r15 = fmaf(r51, r120, r15);
    r15 = fmaf(r127, r96, r15);
    r74 = r74 + r15;
    r90 = fmaf(r8, r74, r67 * r90);
    r120 = r4 * r26;
    r120 = r120 * r10;
    r120 = r120 * r77;
    r90 = fmaf(r56, r120, r90);
    r128 = r43 * r59;
    r90 = fmaf(r78, r128, r90);
    r13 = r11 * r127;
    r13 = r13 * r82;
    r90 = fmaf(r72, r13, r90);
    r119 = r81 * r127;
    r119 = r119 * r82;
    r90 = fmaf(r79, r119, r90);
    r31 = r66 * r26;
    r31 = r31 * r10;
    r31 = r31 * r11;
    r31 = r31 * r59;
    r31 = r31 * r59;
    r31 = r31 * r93;
    r31 = r31 * r47;
    r90 = fmaf(r12, r31, r90);
    r88 = r81 * r127;
    r88 = r88 * r78;
    r90 = fmaf(r82, r88, r90);
    r134 = r39 * r51;
    r90 = fmaf(r72, r134, r90);
    r89 = r10 * r59;
    r106 = r7 * r16;
    r106 = r106 * r65;
    r106 = fmaf(r6, r74, r74 * r106);
    r106 = fmaf(r74, r75, r106);
    r106 = fmaf(r74, r68, r106);
    r89 = r89 * r106;
    r90 = fmaf(r78, r89, r90);
    r64 = r4 * r26;
    r64 = r64 * r10;
    r64 = r64 * r76;
    r64 = r64 * r77;
    r90 = fmaf(r56, r64, r90);
    r90 = fmaf(r127, r113, r90);
    r90 = fmaf(r127, r122, r90);
    r90 = fmaf(r43, r80, r90);
    r90 = fmaf(r43, r73, r90);
    r90 = fmaf(r127, r109, r90);
    r64 = r0 * r90;
    r89 = r26 * r11;
    r89 = r89 * r11;
    r89 = r89 * r59;
    r89 = r89 * r59;
    r89 = r89 * r105;
    r89 = r89 * r47;
    r134 = r39 * r95;
    r134 = r134 * r61;
    r134 = fmaf(r63, r134, r12 * r89);
    r89 = r107 * r127;
    r89 = r89 * r111;
    r134 = fmaf(r112, r89, r134);
    r88 = r11 * r11;
    r88 = r88 * r71;
    r88 = r88 * r127;
    r88 = r88 * r35;
    r88 = r88 * r102;
    r134 = fmaf(r61, r88, r134);
    r134 = r134 + r15;
    r74 = fmaf(r9, r74, r66 * r134);
    r134 = r4 * r26;
    r134 = r134 * r11;
    r134 = r134 * r77;
    r74 = fmaf(r56, r134, r74);
    r15 = r4 * r26;
    r15 = r15 * r11;
    r15 = r15 * r76;
    r15 = r15 * r77;
    r74 = fmaf(r56, r15, r74);
    r88 = r11 * r59;
    r88 = r88 * r83;
    r88 = r88 * r127;
    r88 = r88 * r47;
    r88 = r88 * r69;
    r74 = fmaf(r35, r88, r74);
    r89 = r11 * r59;
    r89 = r89 * r76;
    r89 = r89 * r83;
    r89 = r89 * r127;
    r89 = r89 * r47;
    r89 = r89 * r69;
    r74 = fmaf(r35, r89, r74);
    r31 = r39 * r59;
    r74 = fmaf(r78, r31, r74);
    r119 = r67 * r127;
    r74 = fmaf(r121, r119, r74);
    r13 = r67 * r11;
    r13 = r13 * r127;
    r13 = r13 * r62;
    r74 = fmaf(r82, r13, r74);
    r128 = r67 * r43;
    r128 = r128 * r63;
    r74 = fmaf(r62, r128, r74);
    r120 = r11 * r81;
    r120 = r120 * r127;
    r120 = r120 * r35;
    r120 = r120 * r102;
    r74 = fmaf(r78, r120, r74);
    r99 = r67 * r39;
    r99 = r99 * r62;
    r74 = fmaf(r51, r99, r74);
    r27 = r106 * r78;
    r74 = fmaf(r63, r27, r74);
    r74 = fmaf(r39, r80, r74);
    r74 = fmaf(r26, r118, r74);
    r74 = fmaf(r127, r125, r74);
    r27 = r5 * r74;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r100,
                                          r28,
                                          r64,
                                          r27);
    r27 = r11 * r11;
    r64 = r40 * r11;
    r64 = r64 * r11;
    r28 = r16 * r44;
    r28 = r28 * r11;
    r28 = fmaf(r37, r28, r36 * r64);
    r64 = r40 * r10;
    r64 = r64 * r10;
    r28 = fmaf(r36, r64, r28);
    r100 = r16 * r58;
    r100 = r100 * r10;
    r28 = fmaf(r37, r100, r28);
    r27 = r27 * r28;
    r27 = r27 * r35;
    r27 = r27 * r102;
    r100 = r40 * r11;
    r100 = r100 * r11;
    r100 = r100 * r59;
    r100 = r100 * r59;
    r100 = r100 * r47;
    r100 = fmaf(r36, r100, r61 * r27);
    r27 = r44 * r63;
    r100 = fmaf(r62, r27, r100);
    r64 = r4 * r28;
    r64 = r64 * r111;
    r100 = fmaf(r112, r64, r100);
    r64 = r40 * r10;
    r64 = r64 * r10;
    r64 = r64 * r59;
    r64 = r64 * r59;
    r64 = r64 * r47;
    r64 = fmaf(r36, r64, r28 * r96);
    r27 = r58 * r62;
    r64 = fmaf(r51, r27, r64);
    r99 = r4 * r28;
    r99 = r99 * r101;
    r99 = r99 * r52;
    r64 = fmaf(r51, r99, r64);
    r99 = r100 + r64;
    r27 = r71 * r28;
    r120 = r40 * r10;
    r120 = r120 * r10;
    r120 = r120 * r59;
    r120 = r120 * r59;
    r120 = r120 * r105;
    r120 = r120 * r47;
    r120 = fmaf(r12, r120, r96 * r27);
    r27 = r58 * r10;
    r27 = r27 * r59;
    r27 = r27 * r95;
    r120 = fmaf(r61, r27, r120);
    r120 = fmaf(r28, r108, r120);
    r120 = r120 + r100;
    r120 = fmaf(r67, r120, r8 * r99);
    r100 = r10 * r59;
    r27 = r7 * r16;
    r27 = r27 * r65;
    r27 = fmaf(r99, r27, r6 * r99);
    r27 = fmaf(r99, r75, r27);
    r27 = fmaf(r99, r68, r27);
    r100 = r100 * r27;
    r120 = fmaf(r78, r100, r120);
    r128 = r44 * r51;
    r120 = fmaf(r72, r128, r120);
    r13 = r58 * r59;
    r120 = fmaf(r78, r13, r120);
    r119 = r11 * r28;
    r119 = r119 * r82;
    r120 = fmaf(r72, r119, r120);
    r31 = r81 * r28;
    r31 = r31 * r78;
    r120 = fmaf(r82, r31, r120);
    r89 = r66 * r40;
    r89 = r89 * r10;
    r89 = r89 * r11;
    r89 = r89 * r59;
    r89 = r89 * r59;
    r89 = r89 * r93;
    r89 = r89 * r47;
    r120 = fmaf(r12, r89, r120);
    r88 = r81 * r28;
    r88 = r88 * r82;
    r120 = fmaf(r79, r88, r120);
    r15 = r4 * r40;
    r15 = r15 * r10;
    r15 = r15 * r77;
    r120 = fmaf(r56, r15, r120);
    r134 = r4 * r40;
    r134 = r134 * r10;
    r134 = r134 * r76;
    r134 = r134 * r77;
    r120 = fmaf(r56, r134, r120);
    r120 = fmaf(r58, r80, r120);
    r120 = fmaf(r28, r109, r120);
    r120 = fmaf(r28, r122, r120);
    r120 = fmaf(r28, r113, r120);
    r120 = fmaf(r58, r73, r120);
    r134 = r0 * r120;
    r15 = r11 * r11;
    r15 = r15 * r71;
    r15 = r15 * r28;
    r15 = r15 * r35;
    r15 = r15 * r102;
    r88 = r40 * r11;
    r88 = r88 * r11;
    r88 = r88 * r59;
    r88 = r88 * r59;
    r88 = r88 * r105;
    r88 = r88 * r47;
    r88 = fmaf(r12, r88, r61 * r15);
    r15 = r44 * r95;
    r15 = r15 * r61;
    r88 = fmaf(r63, r15, r88);
    r89 = r107 * r28;
    r89 = r89 * r111;
    r88 = fmaf(r112, r89, r88);
    r88 = r88 + r64;
    r88 = fmaf(r66, r88, r9 * r99);
    r99 = r67 * r44;
    r99 = r99 * r62;
    r88 = fmaf(r51, r99, r88);
    r64 = r4 * r40;
    r64 = r64 * r11;
    r64 = r64 * r77;
    r88 = fmaf(r56, r64, r88);
    r89 = r11 * r81;
    r89 = r89 * r28;
    r89 = r89 * r35;
    r89 = r89 * r102;
    r88 = fmaf(r78, r89, r88);
    r15 = r67 * r11;
    r15 = r15 * r28;
    r15 = r15 * r62;
    r88 = fmaf(r82, r15, r88);
    r31 = r67 * r28;
    r88 = fmaf(r121, r31, r88);
    r119 = r44 * r59;
    r88 = fmaf(r78, r119, r88);
    r13 = r67 * r58;
    r13 = r13 * r63;
    r88 = fmaf(r62, r13, r88);
    r128 = r4 * r40;
    r128 = r128 * r11;
    r128 = r128 * r76;
    r128 = r128 * r77;
    r88 = fmaf(r56, r128, r88);
    r100 = r11 * r59;
    r100 = r100 * r76;
    r100 = r100 * r83;
    r100 = r100 * r28;
    r100 = r100 * r47;
    r100 = r100 * r69;
    r88 = fmaf(r35, r100, r88);
    r116 = r27 * r78;
    r88 = fmaf(r63, r116, r88);
    r135 = r11 * r59;
    r135 = r135 * r83;
    r135 = r135 * r28;
    r135 = r135 * r47;
    r135 = r135 * r69;
    r88 = fmaf(r35, r135, r88);
    r88 = fmaf(r28, r125, r88);
    r88 = fmaf(r44, r80, r88);
    r88 = fmaf(r40, r118, r88);
    r135 = r5 * r88;
    r116 = r11 * r11;
    r100 = r16 * r46;
    r100 = r100 * r11;
    r128 = r38 * r11;
    r128 = r128 * r11;
    r128 = fmaf(r36, r128, r37 * r100);
    r100 = r16 * r57;
    r100 = r100 * r10;
    r128 = fmaf(r37, r100, r128);
    r13 = r38 * r10;
    r13 = r13 * r10;
    r128 = fmaf(r36, r13, r128);
    r116 = r116 * r128;
    r116 = r116 * r35;
    r116 = r116 * r102;
    r13 = r46 * r63;
    r13 = fmaf(r62, r13, r61 * r116);
    r116 = r4 * r128;
    r116 = r116 * r111;
    r13 = fmaf(r112, r116, r13);
    r100 = r38 * r11;
    r100 = r100 * r11;
    r100 = r100 * r59;
    r100 = r100 * r59;
    r100 = r100 * r47;
    r13 = fmaf(r36, r100, r13);
    r100 = r38 * r10;
    r100 = r100 * r10;
    r100 = r100 * r59;
    r100 = r100 * r59;
    r100 = r100 * r47;
    r116 = r4 * r128;
    r116 = r116 * r101;
    r116 = r116 * r52;
    r116 = fmaf(r51, r116, r36 * r100);
    r100 = r57 * r62;
    r116 = fmaf(r51, r100, r116);
    r116 = fmaf(r128, r96, r116);
    r100 = r13 + r116;
    r119 = r38 * r10;
    r119 = r119 * r10;
    r119 = r119 * r59;
    r119 = r119 * r59;
    r119 = r119 * r105;
    r119 = r119 * r47;
    r119 = fmaf(r128, r108, r12 * r119);
    r31 = r57 * r10;
    r31 = r31 * r59;
    r31 = r31 * r95;
    r119 = fmaf(r61, r31, r119);
    r15 = r71 * r128;
    r119 = fmaf(r96, r15, r119);
    r119 = r119 + r13;
    r119 = fmaf(r67, r119, r8 * r100);
    r13 = r4 * r38;
    r13 = r13 * r10;
    r13 = r13 * r76;
    r13 = r13 * r77;
    r119 = fmaf(r56, r13, r119);
    r15 = r57 * r59;
    r119 = fmaf(r78, r15, r119);
    r31 = r4 * r38;
    r31 = r31 * r10;
    r31 = r31 * r77;
    r119 = fmaf(r56, r31, r119);
    r89 = r81 * r128;
    r89 = r89 * r82;
    r119 = fmaf(r79, r89, r119);
    r64 = r11 * r128;
    r64 = r64 * r82;
    r119 = fmaf(r72, r64, r119);
    r99 = r46 * r51;
    r119 = fmaf(r72, r99, r119);
    r114 = r81 * r128;
    r114 = r114 * r78;
    r119 = fmaf(r82, r114, r119);
    r131 = r10 * r59;
    r84 = r7 * r16;
    r84 = r84 * r65;
    r84 = fmaf(r6, r100, r100 * r84);
    r84 = fmaf(r100, r75, r84);
    r84 = fmaf(r100, r68, r84);
    r131 = r131 * r84;
    r119 = fmaf(r78, r131, r119);
    r87 = r66 * r38;
    r87 = r87 * r10;
    r87 = r87 * r11;
    r87 = r87 * r59;
    r87 = r87 * r59;
    r87 = r87 * r93;
    r87 = r87 * r47;
    r119 = fmaf(r12, r87, r119);
    r119 = fmaf(r57, r80, r119);
    r119 = fmaf(r128, r109, r119);
    r119 = fmaf(r128, r122, r119);
    r119 = fmaf(r57, r73, r119);
    r119 = fmaf(r128, r113, r119);
    r87 = r0 * r119;
    r131 = r11 * r11;
    r131 = r131 * r71;
    r131 = r131 * r128;
    r131 = r131 * r35;
    r131 = r131 * r102;
    r114 = r46 * r95;
    r114 = r114 * r61;
    r114 = fmaf(r63, r114, r61 * r131);
    r131 = r107 * r128;
    r131 = r131 * r111;
    r114 = fmaf(r112, r131, r114);
    r99 = r38 * r11;
    r99 = r99 * r11;
    r99 = r99 * r59;
    r99 = r99 * r59;
    r99 = r99 * r105;
    r99 = r99 * r47;
    r114 = fmaf(r12, r99, r114);
    r114 = r114 + r116;
    r114 = fmaf(r66, r114, r9 * r100);
    r100 = r11 * r81;
    r100 = r100 * r128;
    r100 = r100 * r35;
    r100 = r100 * r102;
    r114 = fmaf(r78, r100, r114);
    r116 = r11 * r59;
    r116 = r116 * r76;
    r116 = r116 * r83;
    r116 = r116 * r128;
    r116 = r116 * r47;
    r116 = r116 * r69;
    r114 = fmaf(r35, r116, r114);
    r99 = r11 * r59;
    r99 = r99 * r83;
    r99 = r99 * r128;
    r99 = r99 * r47;
    r99 = r99 * r69;
    r114 = fmaf(r35, r99, r114);
    r131 = r67 * r128;
    r114 = fmaf(r121, r131, r114);
    r64 = r67 * r11;
    r64 = r64 * r128;
    r64 = r64 * r62;
    r114 = fmaf(r82, r64, r114);
    r89 = r67 * r46;
    r89 = r89 * r62;
    r114 = fmaf(r51, r89, r114);
    r31 = r4 * r38;
    r31 = r31 * r11;
    r31 = r31 * r76;
    r31 = r31 * r77;
    r114 = fmaf(r56, r31, r114);
    r15 = r67 * r57;
    r15 = r15 * r63;
    r114 = fmaf(r62, r15, r114);
    r13 = r46 * r59;
    r114 = fmaf(r78, r13, r114);
    r104 = r4 * r38;
    r104 = r104 * r11;
    r104 = r104 * r77;
    r114 = fmaf(r56, r104, r114);
    r132 = r84 * r78;
    r114 = fmaf(r63, r132, r114);
    r114 = fmaf(r128, r125, r114);
    r114 = fmaf(r46, r80, r114);
    r114 = fmaf(r38, r118, r114);
    r132 = r5 * r114;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r134,
                                          r135,
                                          r87,
                                          r132);
    r132 = r0 * r4;
    r132 = r132 * r2;
    r70 = r4 * r70;
    r87 = r5 * r70;
    r132 = fmaf(r110, r87, r1 * r132);
    r135 = r0 * r4;
    r135 = r135 * r2;
    r135 = fmaf(r86, r87, r126 * r135);
    r134 = r0 * r4;
    r134 = r134 * r2;
    r134 = fmaf(r103, r87, r14 * r134);
    r104 = r0 * r4;
    r104 = r104 * r2;
    r104 = fmaf(r74, r87, r90 * r104);
    WriteSum4<float, float>((float*)inout_shared, r132, r135, r134, r104);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r104 = r0 * r4;
    r104 = r104 * r2;
    r104 = fmaf(r88, r87, r120 * r104);
    r134 = r0 * r4;
    r134 = r134 * r2;
    r134 = fmaf(r114, r87, r119 * r134);
    WriteSum2<float, float>((float*)inout_shared, r104, r134);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r134 = r0 * r0;
    r104 = r1 * r134;
    r135 = r5 * r5;
    r132 = r110 * r135;
    r110 = fmaf(r110, r132, r1 * r104);
    r1 = r126 * r126;
    r13 = r86 * r86;
    r13 = fmaf(r135, r13, r134 * r1);
    r1 = r103 * r103;
    r15 = r14 * r14;
    r15 = fmaf(r134, r15, r135 * r1);
    r1 = r90 * r90;
    r31 = r74 * r74;
    r31 = fmaf(r135, r31, r134 * r1);
    WriteSum4<float, float>((float*)inout_shared, r110, r13, r15, r31);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r120 * r120;
    r15 = r88 * r88;
    r15 = fmaf(r135, r15, r134 * r31);
    r31 = r114 * r114;
    r13 = r119 * r119;
    r13 = fmaf(r134, r13, r135 * r31);
    WriteSum2<float, float>((float*)inout_shared, r15, r13);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fmaf(r126, r104, r86 * r132);
    r15 = fmaf(r103, r132, r14 * r104);
    r31 = fmaf(r90, r104, r74 * r132);
    r110 = fmaf(r120, r104, r88 * r132);
    WriteSum4<float, float>((float*)inout_shared, r13, r15, r31, r110);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r132 = fmaf(r114, r132, r119 * r104);
    r104 = r86 * r103;
    r110 = r126 * r14;
    r110 = fmaf(r134, r110, r135 * r104);
    r104 = r126 * r90;
    r31 = r86 * r74;
    r31 = fmaf(r135, r31, r134 * r104);
    r104 = r126 * r120;
    r15 = r86 * r88;
    r15 = fmaf(r135, r15, r134 * r104);
    WriteSum4<float, float>((float*)inout_shared, r132, r110, r31, r15);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = r86 * r114;
    r31 = r126 * r119;
    r31 = fmaf(r134, r31, r135 * r15);
    r15 = r103 * r74;
    r110 = r14 * r90;
    r110 = fmaf(r134, r110, r135 * r15);
    r15 = r14 * r120;
    r132 = r103 * r88;
    r132 = fmaf(r135, r132, r134 * r15);
    r15 = r103 * r114;
    r104 = r14 * r119;
    r104 = fmaf(r134, r104, r135 * r15);
    WriteSum4<float, float>((float*)inout_shared, r31, r110, r132, r104);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r104 = r74 * r88;
    r132 = r90 * r120;
    r132 = fmaf(r134, r132, r135 * r104);
    r104 = r90 * r119;
    r110 = r74 * r114;
    r110 = fmaf(r135, r110, r134 * r104);
    r104 = r88 * r114;
    r31 = r120 * r119;
    r31 = fmaf(r134, r31, r135 * r104);
    WriteSum3<float, float>((float*)inout_shared, r132, r110, r31);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r4 * r2;
    WriteSum2<float, float>((float*)inout_shared, r31, r70);
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
    r42 = r54 * r10;
    r42 = r42 * r59;
    r42 = r42 * r95;
    r70 = r60 * r10;
    r70 = r70 * r10;
    r70 = r70 * r59;
    r70 = r70 * r59;
    r70 = r70 * r105;
    r70 = r70 * r47;
    r70 = fmaf(r12, r70, r61 * r42);
    r42 = r16 * r33;
    r42 = r42 * r11;
    r31 = r16 * r54;
    r31 = r31 * r10;
    r31 = fmaf(r37, r31, r37 * r42);
    r42 = r60 * r10;
    r42 = r42 * r10;
    r31 = fmaf(r36, r42, r31);
    r110 = r60 * r11;
    r110 = r110 * r11;
    r31 = fmaf(r36, r110, r31);
    r110 = r71 * r31;
    r70 = fmaf(r96, r110, r70);
    r42 = r4 * r31;
    r42 = r42 * r111;
    r132 = r11 * r11;
    r132 = r132 * r31;
    r132 = r132 * r35;
    r132 = r132 * r102;
    r132 = fmaf(r61, r132, r112 * r42);
    r42 = r60 * r11;
    r42 = r42 * r11;
    r42 = r42 * r59;
    r42 = r42 * r59;
    r42 = r42 * r47;
    r132 = fmaf(r36, r42, r132);
    r104 = r33 * r63;
    r132 = fmaf(r62, r104, r132);
    r70 = fmaf(r31, r108, r70);
    r70 = r70 + r132;
    r110 = r54 * r62;
    r104 = r60 * r10;
    r104 = r104 * r10;
    r104 = r104 * r59;
    r104 = r104 * r59;
    r104 = r104 * r47;
    r104 = fmaf(r36, r104, r51 * r110);
    r110 = r4 * r31;
    r110 = r110 * r101;
    r110 = r110 * r52;
    r104 = fmaf(r51, r110, r104);
    r104 = fmaf(r31, r96, r104);
    r132 = r132 + r104;
    r70 = fmaf(r8, r132, r67 * r70);
    r110 = r4 * r60;
    r110 = r110 * r10;
    r110 = r110 * r76;
    r110 = r110 * r77;
    r70 = fmaf(r56, r110, r70);
    r42 = r4 * r60;
    r42 = r42 * r10;
    r42 = r42 * r77;
    r70 = fmaf(r56, r42, r70);
    r15 = r33 * r51;
    r70 = fmaf(r72, r15, r70);
    r13 = r10 * r59;
    r1 = r7 * r16;
    r1 = r1 * r65;
    r1 = fmaf(r6, r132, r132 * r1);
    r1 = fmaf(r132, r75, r1);
    r1 = fmaf(r132, r68, r1);
    r13 = r13 * r1;
    r70 = fmaf(r78, r13, r70);
    r89 = r81 * r31;
    r89 = r89 * r82;
    r70 = fmaf(r79, r89, r70);
    r64 = r54 * r59;
    r70 = fmaf(r78, r64, r70);
    r131 = r66 * r60;
    r131 = r131 * r10;
    r131 = r131 * r11;
    r131 = r131 * r59;
    r131 = r131 * r59;
    r131 = r131 * r93;
    r131 = r131 * r47;
    r70 = fmaf(r12, r131, r70);
    r99 = r11 * r31;
    r99 = r99 * r82;
    r70 = fmaf(r72, r99, r70);
    r116 = r81 * r31;
    r116 = r116 * r78;
    r70 = fmaf(r82, r116, r70);
    r70 = fmaf(r54, r80, r70);
    r70 = fmaf(r31, r113, r70);
    r70 = fmaf(r31, r109, r70);
    r70 = fmaf(r31, r122, r70);
    r70 = fmaf(r54, r73, r70);
    r116 = r0 * r70;
    r99 = r107 * r31;
    r99 = r99 * r111;
    r131 = r11 * r11;
    r131 = r131 * r71;
    r131 = r131 * r31;
    r131 = r131 * r35;
    r131 = r131 * r102;
    r131 = fmaf(r61, r131, r112 * r99);
    r99 = r60 * r11;
    r99 = r99 * r11;
    r99 = r99 * r59;
    r99 = r99 * r59;
    r99 = r99 * r105;
    r99 = r99 * r47;
    r131 = fmaf(r12, r99, r131);
    r64 = r33 * r95;
    r64 = r64 * r61;
    r131 = fmaf(r63, r64, r131);
    r131 = r131 + r104;
    r132 = fmaf(r9, r132, r66 * r131);
    r131 = r11 * r59;
    r131 = r131 * r76;
    r131 = r131 * r83;
    r131 = r131 * r31;
    r131 = r131 * r47;
    r131 = r131 * r69;
    r132 = fmaf(r35, r131, r132);
    r104 = r4 * r60;
    r104 = r104 * r11;
    r104 = r104 * r76;
    r104 = r104 * r77;
    r132 = fmaf(r56, r104, r132);
    r64 = r67 * r33;
    r64 = r64 * r62;
    r132 = fmaf(r51, r64, r132);
    r99 = r11 * r59;
    r99 = r99 * r83;
    r99 = r99 * r31;
    r99 = r99 * r47;
    r99 = r99 * r69;
    r132 = fmaf(r35, r99, r132);
    r89 = r67 * r31;
    r132 = fmaf(r121, r89, r132);
    r13 = r67 * r54;
    r13 = r13 * r63;
    r132 = fmaf(r62, r13, r132);
    r15 = r11 * r81;
    r15 = r15 * r31;
    r15 = r15 * r35;
    r15 = r15 * r102;
    r132 = fmaf(r78, r15, r132);
    r42 = r4 * r60;
    r42 = r42 * r11;
    r42 = r42 * r77;
    r132 = fmaf(r56, r42, r132);
    r110 = r67 * r11;
    r110 = r110 * r31;
    r110 = r110 * r62;
    r132 = fmaf(r82, r110, r132);
    r100 = r33 * r59;
    r132 = fmaf(r78, r100, r132);
    r117 = r1 * r78;
    r132 = fmaf(r63, r117, r132);
    r132 = fmaf(r31, r125, r132);
    r132 = fmaf(r60, r118, r132);
    r132 = fmaf(r33, r80, r132);
    r117 = r5 * r132;
    r100 = r45 * r11;
    r100 = r100 * r11;
    r110 = r16 * r49;
    r110 = r110 * r10;
    r110 = fmaf(r37, r110, r36 * r100);
    r100 = r16 * r50;
    r100 = r100 * r11;
    r110 = fmaf(r37, r100, r110);
    r42 = r45 * r10;
    r42 = r42 * r10;
    r110 = fmaf(r36, r42, r110);
    r42 = r45 * r10;
    r42 = r42 * r10;
    r42 = r42 * r59;
    r42 = r42 * r59;
    r42 = r42 * r47;
    r42 = fmaf(r36, r42, r110 * r96);
    r100 = r4 * r110;
    r100 = r100 * r101;
    r100 = r100 * r52;
    r42 = fmaf(r51, r100, r42);
    r15 = r49 * r62;
    r42 = fmaf(r51, r15, r42);
    r15 = r4 * r110;
    r15 = r15 * r111;
    r100 = r50 * r63;
    r100 = fmaf(r62, r100, r112 * r15);
    r15 = r11 * r11;
    r15 = r15 * r110;
    r15 = r15 * r35;
    r15 = r15 * r102;
    r100 = fmaf(r61, r15, r100);
    r13 = r45 * r11;
    r13 = r13 * r11;
    r13 = r13 * r59;
    r13 = r13 * r59;
    r13 = r13 * r47;
    r100 = fmaf(r36, r13, r100);
    r13 = r42 + r100;
    r15 = r71 * r110;
    r89 = r45 * r10;
    r89 = r89 * r10;
    r89 = r89 * r59;
    r89 = r89 * r59;
    r89 = r89 * r105;
    r89 = r89 * r47;
    r89 = fmaf(r12, r89, r96 * r15);
    r15 = r49 * r10;
    r15 = r15 * r59;
    r15 = r15 * r95;
    r89 = fmaf(r61, r15, r89);
    r89 = fmaf(r110, r108, r89);
    r89 = r89 + r100;
    r89 = fmaf(r67, r89, r8 * r13);
    r100 = r4 * r45;
    r100 = r100 * r10;
    r100 = r100 * r77;
    r89 = fmaf(r56, r100, r89);
    r15 = r4 * r45;
    r15 = r15 * r10;
    r15 = r15 * r76;
    r15 = r15 * r77;
    r89 = fmaf(r56, r15, r89);
    r99 = r10 * r59;
    r64 = r7 * r16;
    r64 = r64 * r65;
    r64 = fmaf(r13, r64, r6 * r13);
    r64 = fmaf(r13, r75, r64);
    r64 = fmaf(r13, r68, r64);
    r99 = r99 * r64;
    r89 = fmaf(r78, r99, r89);
    r104 = r81 * r110;
    r104 = r104 * r78;
    r89 = fmaf(r82, r104, r89);
    r131 = r81 * r110;
    r131 = r131 * r82;
    r89 = fmaf(r79, r131, r89);
    r130 = r50 * r51;
    r89 = fmaf(r72, r130, r89);
    r115 = r49 * r59;
    r89 = fmaf(r78, r115, r89);
    r123 = r66 * r45;
    r123 = r123 * r10;
    r123 = r123 * r11;
    r123 = r123 * r59;
    r123 = r123 * r59;
    r123 = r123 * r93;
    r123 = r123 * r47;
    r89 = fmaf(r12, r123, r89);
    r137 = r11 * r110;
    r137 = r137 * r82;
    r89 = fmaf(r72, r137, r89);
    r89 = fmaf(r49, r73, r89);
    r89 = fmaf(r110, r122, r89);
    r89 = fmaf(r110, r109, r89);
    r89 = fmaf(r110, r113, r89);
    r89 = fmaf(r49, r80, r89);
    r137 = r0 * r89;
    r123 = r107 * r110;
    r123 = r123 * r111;
    r115 = r50 * r95;
    r115 = r115 * r61;
    r115 = fmaf(r63, r115, r112 * r123);
    r123 = r11 * r11;
    r123 = r123 * r71;
    r123 = r123 * r110;
    r123 = r123 * r35;
    r123 = r123 * r102;
    r115 = fmaf(r61, r123, r115);
    r130 = r45 * r11;
    r130 = r130 * r11;
    r130 = r130 * r59;
    r130 = r130 * r59;
    r130 = r130 * r105;
    r130 = r130 * r47;
    r115 = fmaf(r12, r130, r115);
    r115 = r115 + r42;
    r115 = fmaf(r66, r115, r9 * r13);
    r13 = r67 * r49;
    r13 = r13 * r63;
    r115 = fmaf(r62, r13, r115);
    r42 = r67 * r11;
    r42 = r42 * r110;
    r42 = r42 * r62;
    r115 = fmaf(r82, r42, r115);
    r130 = r67 * r110;
    r115 = fmaf(r121, r130, r115);
    r123 = r11 * r59;
    r123 = r123 * r76;
    r123 = r123 * r83;
    r123 = r123 * r110;
    r123 = r123 * r47;
    r123 = r123 * r69;
    r115 = fmaf(r35, r123, r115);
    r131 = r11 * r59;
    r131 = r131 * r83;
    r131 = r131 * r110;
    r131 = r131 * r47;
    r131 = r131 * r69;
    r115 = fmaf(r35, r131, r115);
    r104 = r64 * r78;
    r115 = fmaf(r63, r104, r115);
    r99 = r50 * r59;
    r115 = fmaf(r78, r99, r115);
    r15 = r4 * r45;
    r15 = r15 * r11;
    r15 = r15 * r76;
    r15 = r15 * r77;
    r115 = fmaf(r56, r15, r115);
    r100 = r11 * r81;
    r100 = r100 * r110;
    r100 = r100 * r35;
    r100 = r100 * r102;
    r115 = fmaf(r78, r100, r115);
    r138 = r67 * r50;
    r138 = r138 * r62;
    r115 = fmaf(r51, r138, r115);
    r139 = r4 * r45;
    r139 = r139 * r11;
    r139 = r139 * r77;
    r115 = fmaf(r56, r139, r115);
    r115 = fmaf(r50, r80, r115);
    r115 = fmaf(r110, r125, r115);
    r115 = fmaf(r45, r118, r115);
    r139 = r5 * r115;
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r116,
                                          r117,
                                          r137,
                                          r139);
    r139 = r16 * r48;
    r139 = r139 * r11;
    r137 = r53 * r11;
    r137 = r137 * r11;
    r137 = fmaf(r36, r137, r37 * r139);
    r139 = r53 * r10;
    r139 = r139 * r10;
    r137 = fmaf(r36, r139, r137);
    r117 = r16 * r55;
    r117 = r117 * r10;
    r137 = fmaf(r37, r117, r137);
    r117 = r4 * r137;
    r117 = r117 * r111;
    r139 = r48 * r63;
    r139 = fmaf(r62, r139, r112 * r117);
    r117 = r11 * r11;
    r117 = r117 * r137;
    r117 = r117 * r35;
    r117 = r117 * r102;
    r139 = fmaf(r61, r117, r139);
    r37 = r53 * r11;
    r37 = r37 * r11;
    r37 = r37 * r59;
    r37 = r37 * r59;
    r37 = r37 * r47;
    r139 = fmaf(r36, r37, r139);
    r37 = r55 * r62;
    r37 = fmaf(r137, r96, r51 * r37);
    r117 = r53 * r10;
    r117 = r117 * r10;
    r117 = r117 * r59;
    r117 = r117 * r59;
    r117 = r117 * r47;
    r37 = fmaf(r36, r117, r37);
    r36 = r4 * r137;
    r36 = r36 * r101;
    r36 = r36 * r52;
    r37 = fmaf(r51, r36, r37);
    r36 = r139 + r37;
    r117 = r55 * r10;
    r117 = r117 * r59;
    r117 = r117 * r95;
    r52 = r71 * r137;
    r52 = fmaf(r96, r52, r61 * r117);
    r117 = r53 * r10;
    r117 = r117 * r10;
    r117 = r117 * r59;
    r117 = r117 * r59;
    r117 = r117 * r105;
    r117 = r117 * r47;
    r52 = fmaf(r12, r117, r52);
    r52 = fmaf(r137, r108, r52);
    r52 = r52 + r139;
    r52 = fmaf(r67, r52, r8 * r36);
    r8 = r48 * r51;
    r52 = fmaf(r72, r8, r52);
    r139 = r11 * r137;
    r139 = r139 * r82;
    r52 = fmaf(r72, r139, r52);
    r72 = r4 * r53;
    r72 = r72 * r10;
    r72 = r72 * r76;
    r72 = r72 * r77;
    r52 = fmaf(r56, r72, r52);
    r108 = r10 * r59;
    r117 = r7 * r16;
    r117 = r117 * r65;
    r117 = fmaf(r36, r117, r6 * r36);
    r117 = fmaf(r36, r75, r117);
    r117 = fmaf(r36, r68, r117);
    r108 = r108 * r117;
    r52 = fmaf(r78, r108, r52);
    r68 = r81 * r137;
    r68 = r68 * r82;
    r52 = fmaf(r79, r68, r52);
    r79 = r4 * r53;
    r79 = r79 * r10;
    r79 = r79 * r77;
    r52 = fmaf(r56, r79, r52);
    r75 = r55 * r59;
    r52 = fmaf(r78, r75, r52);
    r6 = r81 * r137;
    r6 = r6 * r78;
    r52 = fmaf(r82, r6, r52);
    r65 = r66 * r53;
    r65 = r65 * r10;
    r65 = r65 * r11;
    r65 = r65 * r59;
    r65 = r65 * r59;
    r65 = r65 * r93;
    r65 = r65 * r47;
    r52 = fmaf(r12, r65, r52);
    r52 = fmaf(r137, r113, r52);
    r52 = fmaf(r137, r109, r52);
    r52 = fmaf(r55, r80, r52);
    r52 = fmaf(r137, r122, r52);
    r52 = fmaf(r55, r73, r52);
    r73 = r0 * r52;
    r65 = r107 * r137;
    r65 = r65 * r111;
    r111 = r48 * r95;
    r111 = r111 * r61;
    r111 = fmaf(r63, r111, r112 * r65);
    r65 = r11 * r11;
    r65 = r65 * r71;
    r65 = r65 * r137;
    r65 = r65 * r35;
    r65 = r65 * r102;
    r111 = fmaf(r61, r65, r111);
    r61 = r53 * r11;
    r61 = r61 * r11;
    r61 = r61 * r59;
    r61 = r61 * r59;
    r61 = r61 * r105;
    r61 = r61 * r47;
    r111 = fmaf(r12, r61, r111);
    r111 = r111 + r37;
    r111 = fmaf(r66, r111, r9 * r36);
    r36 = r67 * r48;
    r36 = r36 * r62;
    r111 = fmaf(r51, r36, r111);
    r9 = r4 * r53;
    r9 = r9 * r11;
    r9 = r9 * r76;
    r9 = r9 * r77;
    r111 = fmaf(r56, r9, r111);
    r37 = r11 * r59;
    r37 = r37 * r83;
    r37 = r37 * r137;
    r37 = r37 * r47;
    r37 = r37 * r69;
    r111 = fmaf(r35, r37, r111);
    r61 = r67 * r11;
    r61 = r61 * r137;
    r61 = r61 * r62;
    r111 = fmaf(r82, r61, r111);
    r82 = r48 * r59;
    r111 = fmaf(r78, r82, r111);
    r65 = r117 * r78;
    r111 = fmaf(r63, r65, r111);
    r12 = r67 * r137;
    r111 = fmaf(r121, r12, r111);
    r121 = r11 * r59;
    r121 = r121 * r76;
    r121 = r121 * r83;
    r121 = r121 * r137;
    r121 = r121 * r47;
    r121 = r121 * r69;
    r111 = fmaf(r35, r121, r111);
    r69 = r67 * r55;
    r69 = r69 * r63;
    r111 = fmaf(r62, r69, r111);
    r47 = r4 * r53;
    r47 = r47 * r11;
    r47 = r47 * r77;
    r111 = fmaf(r56, r47, r111);
    r56 = r11 * r81;
    r56 = r56 * r137;
    r56 = r56 * r35;
    r56 = r56 * r102;
    r111 = fmaf(r78, r56, r111);
    r111 = fmaf(r53, r118, r111);
    r111 = fmaf(r137, r125, r111);
    r111 = fmaf(r48, r80, r111);
    r5 = r5 * r111;
    WriteIdx2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r73, r5);
    r5 = r0 * r4;
    r5 = r5 * r2;
    r5 = fmaf(r132, r87, r70 * r5);
    r73 = r0 * r4;
    r73 = r73 * r2;
    r73 = fmaf(r115, r87, r89 * r73);
    r56 = r0 * r4;
    r56 = r56 * r2;
    r87 = fmaf(r111, r87, r52 * r56);
    WriteSum3<float, float>((float*)inout_shared, r5, r73, r87);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r87 = r70 * r70;
    r73 = r132 * r132;
    r73 = fmaf(r135, r73, r134 * r87);
    r87 = r115 * r115;
    r5 = r89 * r89;
    r5 = fmaf(r134, r5, r135 * r87);
    r87 = r111 * r111;
    r56 = r52 * r52;
    r56 = fmaf(r134, r56, r135 * r87);
    WriteSum3<float, float>((float*)inout_shared, r73, r5, r56);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = r132 * r115;
    r5 = r70 * r89;
    r5 = fmaf(r134, r5, r135 * r56);
    r56 = r70 * r52;
    r73 = r132 * r111;
    r73 = fmaf(r135, r73, r134 * r56);
    r56 = r115 * r111;
    r87 = r89 * r52;
    r87 = fmaf(r134, r87, r135 * r56);
    WriteSum3<float, float>((float*)inout_shared, r5, r73, r87);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void ThinPrismFisheyeSplitFixedFocalAndExtraResJacFirst(
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
  ThinPrismFisheyeSplitFixedFocalAndExtraResJacFirstKernel<<<n_blocks, 1024>>>(
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