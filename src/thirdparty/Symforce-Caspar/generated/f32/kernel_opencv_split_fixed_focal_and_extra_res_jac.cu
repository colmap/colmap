#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_split_fixed_focal_and_extra_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedFocalAndExtraResJacKernel(
        float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        float *principal_point, unsigned int principal_point_num_alloc,
        SharedIndex *principal_point_indices, float *point,
        unsigned int point_num_alloc, SharedIndex *point_indices, float *pixel,
        unsigned int pixel_num_alloc, float *focal_and_extra,
        unsigned int focal_and_extra_num_alloc, float *out_res,
        unsigned int out_res_num_alloc, float *out_pose_jac,
        unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float *const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float *const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        float *out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        float *const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float *const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float *const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        float *out_point_jac, unsigned int out_point_jac_num_alloc,
        float *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
        float *const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float *const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
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
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101;
  LoadShared<2, float, float>(principal_point, 0 * principal_point_num_alloc,
                              principal_point_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float *)inout_shared,
                       principal_point_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(focal_and_extra,
                                         0 * focal_and_extra_num_alloc,
                                         global_thread_idx, r2, r3, r4, r5);
    ReadIdx2<1024, float, float, float2>(focal_and_extra,
                                         4 * focal_and_extra_num_alloc,
                                         global_thread_idx, r6, r7);
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r8, r9, r10);
  };
  LoadShared<3, float, float>(point, 0 * point_num_alloc, point_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       point_indices_loc[threadIdx.x].target, r11, r12, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r14 = 2.00000000000000000e+00;
  };
  LoadShared<4, float, float>(pose, 0 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r15, r16, r17,
                       r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r19, r20, r21, r22);
    r23 = fmaf(r18, r19, r15 * r22);
    r24 = r16 * r21;
    r25 = -1.00000000000000000e+00;
    r23 = fmaf(r25, r24, r23);
    r23 = fmaf(r17, r20, r23);
    r24 = r14 * r23;
    r26 = r18 * r20;
    r27 = fmaf(r16, r22, r26);
    r28 = r15 * r21;
    r29 = r17 * r19;
    r27 = r27 + r28;
    r27 = fmaf(r25, r29, r27);
    r24 = r24 * r27;
    r30 = fmaf(r16, r19, r17 * r22);
    r31 = r15 * r20;
    r30 = fmaf(r25, r31, r30);
    r30 = fmaf(r18, r21, r30);
    r31 = r14 * r30;
    r32 = fmaf(r16, r20, r15 * r19);
    r32 = fmaf(r17, r21, r32);
    r32 = fmaf(r25, r32, r18 * r22);
    r31 = fmaf(r32, r31, r24);
    r9 = fmaf(r11, r31, r9);
  };
  LoadShared<3, float, float>(pose, 4 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r33, r34, r35);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r36 = r19 * r20;
    r36 = r36 * r14;
    r37 = r21 * r22;
    r37 = fmaf(r14, r37, r36);
    r38 = r19 * r19;
    r39 = -2.00000000000000000e+00;
    r38 = r38 * r39;
    r40 = 1.00000000000000000e+00;
    r41 = r21 * r21;
    r41 = fmaf(r39, r41, r40);
    r42 = r38 + r41;
    r43 = r20 * r21;
    r43 = r43 * r14;
    r44 = r19 * r22;
    r44 = fmaf(r39, r44, r43);
    r45 = r14 * r30;
    r45 = r45 * r27;
    r46 = r39 * r32;
    r47 = fmaf(r23, r46, r45);
    r48 = r30 * r30;
    r48 = r48 * r39;
    r49 = r40 + r48;
    r50 = r23 * r23;
    r50 = r50 * r39;
    r49 = r49 + r50;
    r9 = fmaf(r33, r37, r9);
    r9 = fmaf(r34, r42, r9);
    r9 = fmaf(r35, r44, r9);
    r9 = fmaf(r13, r47, r9);
    r9 = fmaf(r12, r49, r9);
    r51 = r9 * r9;
    r52 = 9.99999999999999955e-07;
    r53 = r14 * r30;
    r53 = r53 * r23;
    r54 = fmaf(r27, r46, r53);
    r10 = fmaf(r11, r54, r10);
    r55 = r19 * r21;
    r55 = r55 * r14;
    r56 = r20 * r22;
    r56 = fmaf(r39, r56, r55);
    r57 = r20 * r20;
    r57 = r57 * r39;
    r58 = r40 + r57;
    r58 = r58 + r38;
    r38 = r19 * r22;
    r38 = fmaf(r14, r38, r43);
    r43 = r14 * r23;
    r43 = fmaf(r32, r43, r45);
    r45 = r39 * r27;
    r45 = r45 * r27;
    r59 = r40 + r45;
    r59 = r59 + r50;
    r10 = fmaf(r33, r56, r10);
    r10 = fmaf(r35, r58, r10);
    r10 = fmaf(r34, r38, r10);
    r10 = fmaf(r12, r43, r10);
    r10 = fmaf(r13, r59, r10);
    r50 = copysign(1.0, r10);
    r50 = fmaf(r52, r50, r10);
    r52 = r50 * r50;
    r10 = 1.0 / r52;
    r51 = r51 * r10;
    r45 = r40 + r45;
    r45 = r45 + r48;
    r8 = fmaf(r11, r45, r8);
    r24 = fmaf(r30, r46, r24);
    r48 = r14 * r27;
    r48 = fmaf(r32, r48, r53);
    r53 = r20 * r22;
    r53 = fmaf(r14, r53, r55);
    r55 = r21 * r22;
    r55 = fmaf(r39, r55, r36);
    r41 = r57 + r41;
    r8 = fmaf(r12, r24, r8);
    r8 = fmaf(r13, r48, r8);
    r8 = fmaf(r35, r53, r8);
    r8 = fmaf(r34, r55, r8);
    r8 = fmaf(r33, r41, r8);
    r33 = 3.00000000000000000e+00;
    r34 = r8 * r33;
    r35 = r8 * r10;
    r34 = fmaf(r35, r34, r51);
    r57 = 1.0 / r50;
    r34 = fmaf(r8, r57, r7 * r34);
    r36 = r8 * r35;
    r51 = r36 + r51;
    r5 = r5 * r51;
    r51 = fmaf(r51, r5, r4 * r51);
    r60 = r51 * r57;
    r61 = r14 * r35;
    r62 = r6 * r61;
    r34 = fmaf(r8, r60, r34);
    r34 = fmaf(r9, r62, r34);
    r34 = fmaf(r2, r34, r0);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r0, r63);
    r34 = fmaf(r0, r25, r34);
    r0 = r9 * r9;
    r0 = r0 * r33;
    r0 = fmaf(r10, r0, r36);
    r0 = fmaf(r9, r57, r6 * r0);
    r36 = r7 * r9;
    r0 = fmaf(r61, r36, r0);
    r0 = fmaf(r9, r60, r0);
    r0 = fmaf(r3, r0, r1);
    r0 = fmaf(r63, r25, r0);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r34, r0);
    r63 = 6.00000000000000000e+00;
    r1 = r14 * r32;
    r36 = r16 * r19;
    r64 = 5.00000000000000000e-01;
    r65 = r15 * r20;
    r66 = -5.00000000000000000e-01;
    r65 = fmaf(r66, r65, r64 * r36);
    r36 = r18 * r21;
    r65 = fmaf(r64, r36, r65);
    r67 = r22 * r64;
    r65 = fmaf(r17, r67, r65);
    r36 = r15 * r22;
    r68 = r18 * r19;
    r68 = fmaf(r66, r68, r66 * r36);
    r36 = r17 * r20;
    r68 = fmaf(r66, r36, r68);
    r69 = r16 * r21;
    r68 = fmaf(r64, r69, r68);
    r69 = r27 * r68;
    r1 = fmaf(r14, r69, r65 * r1);
    r36 = r14 * r23;
    r70 = r16 * r66;
    r71 = fmaf(r64, r29, r22 * r70);
    r71 = fmaf(r66, r26, r71);
    r71 = fmaf(r66, r28, r71);
    r72 = r14 * r30;
    r73 = r15 * r19;
    r74 = r17 * r21;
    r74 = fmaf(r66, r74, r66 * r73);
    r74 = fmaf(r18, r67, r74);
    r74 = fmaf(r20, r70, r74);
    r72 = r72 * r74;
    r36 = fmaf(r71, r36, r72);
    r1 = r1 + r36;
    r73 = r14 * r27;
    r73 = r73 * r74;
    r75 = r14 * r23;
    r75 = r75 * r65;
    r76 = r73 + r75;
    r77 = r30 * r39;
    r76 = fmaf(r68, r77, r76);
    r76 = fmaf(r71, r46, r76);
    r76 = fmaf(r12, r76, r13 * r1);
    r1 = r27 * r65;
    r77 = -4.00000000000000000e+00;
    r1 = r1 * r77;
    r78 = r30 * r71;
    r79 = r77 * r78;
    r80 = r1 + r79;
    r76 = fmaf(r11, r80, r76);
    r80 = r63 * r76;
    r81 = r8 * r8;
    r82 = r14 * r27;
    r82 = r82 * r71;
    r83 = r14 * r30;
    r83 = fmaf(r65, r83, r82);
    r84 = r14 * r23;
    r84 = r84 * r68;
    r85 = r14 * r32;
    r85 = r85 * r74;
    r86 = r84 + r85;
    r87 = r83 + r86;
    r65 = fmaf(r65, r46, r39 * r69);
    r65 = r65 + r36;
    r65 = fmaf(r11, r65, r12 * r87);
    r87 = r23 * r74;
    r87 = r87 * r77;
    r1 = r87 + r1;
    r65 = fmaf(r13, r1, r65);
    r1 = -6.00000000000000000e+00;
    r52 = r50 * r52;
    r52 = 1.0 / r52;
    r81 = r81 * r65;
    r81 = r81 * r1;
    r81 = fmaf(r52, r81, r35 * r80);
    r80 = r14 * r9;
    r50 = r23 * r39;
    r88 = r74 * r46;
    r50 = fmaf(r68, r50, r88);
    r50 = r50 + r83;
    r79 = r87 + r79;
    r79 = fmaf(r12, r79, r13 * r50);
    r50 = r14 * r32;
    r50 = fmaf(r71, r50, r75);
    r75 = r14 * r30;
    r75 = fmaf(r68, r75, r73);
    r50 = r50 + r75;
    r79 = fmaf(r11, r50, r79);
    r80 = r80 * r79;
    r50 = r39 * r9;
    r73 = r9 * r52;
    r50 = r50 * r65;
    r50 = fmaf(r73, r50, r10 * r80);
    r81 = r81 + r50;
    r80 = r25 * r51;
    r80 = r80 * r65;
    r80 = fmaf(r35, r80, r7 * r81);
    r81 = r6 * r65;
    r87 = r8 * r77;
    r87 = r87 * r73;
    r80 = fmaf(r87, r81, r80);
    r83 = r6 * r14;
    r83 = r83 * r9;
    r83 = r83 * r76;
    r80 = fmaf(r10, r83, r80);
    r89 = r25 * r65;
    r80 = fmaf(r35, r89, r80);
    r90 = r39 * r8;
    r90 = r90 * r8;
    r90 = r90 * r65;
    r90 = fmaf(r52, r90, r76 * r61);
    r50 = r50 + r90;
    r5 = r14 * r5;
    r50 = fmaf(r50, r5, r4 * r50);
    r91 = r8 * r50;
    r80 = fmaf(r57, r91, r80);
    r80 = fmaf(r79, r62, r80);
    r80 = fmaf(r76, r60, r80);
    r80 = fmaf(r76, r57, r80);
    r91 = r2 * r80;
    r89 = r9 * r79;
    r89 = r89 * r63;
    r83 = r9 * r1;
    r83 = r83 * r73;
    r89 = fmaf(r65, r83, r10 * r89);
    r89 = r89 + r90;
    r89 = fmaf(r79, r57, r6 * r89);
    r90 = r25 * r9;
    r90 = r90 * r65;
    r89 = fmaf(r10, r90, r89);
    r81 = r7 * r79;
    r89 = fmaf(r61, r81, r89);
    r92 = r9 * r50;
    r89 = fmaf(r57, r92, r89);
    r93 = r7 * r87;
    r94 = r7 * r14;
    r94 = r94 * r9;
    r94 = r94 * r76;
    r89 = fmaf(r10, r94, r89);
    r95 = r25 * r9;
    r95 = r95 * r51;
    r95 = r95 * r65;
    r89 = fmaf(r10, r95, r89);
    r89 = fmaf(r79, r60, r89);
    r89 = fmaf(r65, r93, r89);
    r95 = r3 * r89;
    r85 = r82 + r85;
    r82 = r14 * r30;
    r94 = r17 * r22;
    r92 = r15 * r20;
    r92 = fmaf(r64, r92, r66 * r94);
    r94 = r18 * r21;
    r92 = fmaf(r66, r94, r92);
    r92 = fmaf(r19, r70, r92);
    r82 = r82 * r92;
    r94 = r14 * r23;
    r81 = r18 * r19;
    r90 = r17 * r20;
    r90 = fmaf(r64, r90, r64 * r81);
    r90 = fmaf(r15, r67, r90);
    r90 = fmaf(r21, r70, r90);
    r94 = fmaf(r90, r94, r82);
    r85 = r85 + r94;
    r70 = r27 * r74;
    r70 = r70 * r77;
    r81 = r30 * r77;
    r81 = r81 * r90;
    r96 = r70 + r81;
    r96 = fmaf(r11, r96, r13 * r85);
    r85 = fmaf(r90, r46, r39 * r78);
    r97 = r14 * r23;
    r97 = r97 * r74;
    r98 = r14 * r27;
    r98 = fmaf(r92, r98, r97);
    r85 = r85 + r98;
    r96 = fmaf(r12, r85, r96);
    r85 = r63 * r96;
    r99 = r8 * r8;
    r100 = r39 * r27;
    r100 = fmaf(r71, r100, r88);
    r100 = r100 + r94;
    r94 = r14 * r27;
    r94 = r94 * r90;
    r101 = r14 * r32;
    r101 = fmaf(r92, r101, r94);
    r101 = r101 + r36;
    r101 = fmaf(r12, r101, r11 * r100);
    r100 = r23 * r92;
    r36 = r77 * r100;
    r70 = r70 + r36;
    r101 = fmaf(r13, r70, r101);
    r99 = r99 * r1;
    r99 = r99 * r101;
    r99 = fmaf(r52, r99, r35 * r85);
    r85 = r39 * r9;
    r85 = r85 * r101;
    r70 = r14 * r9;
    r94 = r72 + r94;
    r72 = r23 * r39;
    r94 = fmaf(r71, r72, r94);
    r94 = fmaf(r92, r46, r94);
    r72 = r14 * r32;
    r78 = fmaf(r14, r78, r90 * r72);
    r78 = r78 + r98;
    r78 = fmaf(r11, r78, r13 * r94);
    r36 = r81 + r36;
    r78 = fmaf(r12, r36, r78);
    r70 = r70 * r78;
    r70 = fmaf(r10, r70, r73 * r85);
    r99 = r99 + r70;
    r99 = fmaf(r96, r60, r7 * r99);
    r85 = r25 * r101;
    r99 = fmaf(r35, r85, r99);
    r36 = r6 * r101;
    r99 = fmaf(r87, r36, r99);
    r81 = r6 * r14;
    r81 = r81 * r9;
    r81 = r81 * r96;
    r99 = fmaf(r10, r81, r99);
    r94 = r25 * r51;
    r94 = r94 * r101;
    r99 = fmaf(r35, r94, r99);
    r72 = r39 * r8;
    r72 = r72 * r8;
    r72 = r72 * r101;
    r72 = fmaf(r52, r72, r96 * r61);
    r70 = r70 + r72;
    r70 = fmaf(r70, r5, r4 * r70);
    r90 = r8 * r70;
    r99 = fmaf(r57, r90, r99);
    r99 = fmaf(r96, r57, r99);
    r99 = fmaf(r78, r62, r99);
    r90 = r2 * r99;
    r94 = r9 * r63;
    r94 = r94 * r78;
    r94 = fmaf(r10, r94, r101 * r83);
    r94 = r94 + r72;
    r72 = r25 * r9;
    r72 = r72 * r51;
    r72 = r72 * r101;
    r72 = fmaf(r10, r72, r6 * r94);
    r94 = r9 * r70;
    r72 = fmaf(r57, r94, r72);
    r81 = r25 * r9;
    r81 = r81 * r101;
    r72 = fmaf(r10, r81, r72);
    r36 = r7 * r14;
    r36 = r36 * r9;
    r36 = r36 * r96;
    r72 = fmaf(r10, r36, r72);
    r85 = r7 * r78;
    r72 = fmaf(r61, r85, r72);
    r72 = fmaf(r78, r57, r72);
    r72 = fmaf(r101, r93, r72);
    r72 = fmaf(r78, r60, r72);
    r85 = r3 * r72;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r91, r95,
        r90, r85);
    r85 = r8 * r8;
    r90 = r23 * r77;
    r29 = fmaf(r66, r29, r16 * r67);
    r29 = fmaf(r64, r26, r29);
    r29 = fmaf(r64, r28, r29);
    r90 = r90 * r29;
    r69 = r77 * r69;
    r28 = r90 + r69;
    r64 = r14 * r30;
    r64 = r64 * r29;
    r97 = r97 + r64;
    r26 = r39 * r27;
    r97 = fmaf(r92, r26, r97);
    r97 = fmaf(r68, r46, r97);
    r97 = fmaf(r11, r97, r13 * r28);
    r28 = r14 * r32;
    r28 = fmaf(r14, r100, r29 * r28);
    r28 = r28 + r75;
    r97 = fmaf(r12, r28, r97);
    r85 = r85 * r1;
    r85 = r85 * r97;
    r28 = r14 * r27;
    r28 = r28 * r29;
    r84 = r84 + r28;
    r26 = r30 * r39;
    r84 = fmaf(r92, r26, r84);
    r84 = r84 + r88;
    r74 = r30 * r74;
    r74 = r74 * r77;
    r69 = r74 + r69;
    r69 = fmaf(r11, r69, r12 * r84);
    r84 = r14 * r32;
    r84 = fmaf(r68, r84, r64);
    r84 = r84 + r98;
    r69 = fmaf(r13, r84, r69);
    r84 = r63 * r69;
    r84 = fmaf(r35, r84, r52 * r85);
    r85 = r39 * r9;
    r85 = r85 * r97;
    r98 = r14 * r9;
    r28 = r82 + r28;
    r28 = r28 + r86;
    r46 = fmaf(r29, r46, r39 * r100);
    r46 = r46 + r75;
    r46 = fmaf(r13, r46, r11 * r28);
    r90 = r74 + r90;
    r46 = fmaf(r12, r90, r46);
    r98 = r98 * r46;
    r98 = fmaf(r10, r98, r73 * r85);
    r84 = r84 + r98;
    r84 = fmaf(r69, r57, r7 * r84);
    r85 = r25 * r51;
    r85 = r85 * r97;
    r84 = fmaf(r35, r85, r84);
    r90 = r25 * r97;
    r84 = fmaf(r35, r90, r84);
    r12 = r6 * r14;
    r12 = r12 * r9;
    r12 = r12 * r69;
    r84 = fmaf(r10, r12, r84);
    r74 = r6 * r97;
    r84 = fmaf(r87, r74, r84);
    r13 = r39 * r8;
    r13 = r13 * r8;
    r13 = r13 * r97;
    r13 = fmaf(r69, r61, r52 * r13);
    r98 = r98 + r13;
    r98 = fmaf(r98, r5, r4 * r98);
    r28 = r8 * r98;
    r84 = fmaf(r57, r28, r84);
    r84 = fmaf(r46, r62, r84);
    r84 = fmaf(r69, r60, r84);
    r28 = r2 * r84;
    r74 = r9 * r63;
    r74 = r74 * r46;
    r74 = fmaf(r10, r74, r97 * r83);
    r74 = r74 + r13;
    r74 = fmaf(r46, r57, r6 * r74);
    r13 = r7 * r14;
    r13 = r13 * r9;
    r13 = r13 * r69;
    r74 = fmaf(r10, r13, r74);
    r12 = r9 * r98;
    r74 = fmaf(r57, r12, r74);
    r90 = r7 * r46;
    r74 = fmaf(r61, r90, r74);
    r85 = r25 * r9;
    r85 = r85 * r97;
    r74 = fmaf(r10, r85, r74);
    r11 = r25 * r9;
    r11 = r11 * r51;
    r11 = r11 * r97;
    r74 = fmaf(r10, r11, r74);
    r74 = fmaf(r97, r93, r74);
    r74 = fmaf(r46, r60, r74);
    r11 = r3 * r74;
    r85 = r56 * r8;
    r85 = r85 * r8;
    r85 = r85 * r1;
    r90 = r41 * r63;
    r90 = fmaf(r35, r90, r52 * r85);
    r85 = r39 * r56;
    r85 = r85 * r9;
    r12 = r14 * r37;
    r12 = r12 * r9;
    r12 = fmaf(r10, r12, r73 * r85);
    r90 = r90 + r12;
    r90 = fmaf(r41, r57, r7 * r90);
    r85 = r6 * r14;
    r85 = r85 * r41;
    r85 = r85 * r9;
    r90 = fmaf(r10, r85, r90);
    r13 = r6 * r56;
    r90 = fmaf(r87, r13, r90);
    r75 = r39 * r56;
    r75 = r75 * r8;
    r75 = r75 * r8;
    r75 = fmaf(r41, r61, r52 * r75);
    r12 = r12 + r75;
    r12 = fmaf(r12, r5, r4 * r12);
    r29 = r8 * r12;
    r90 = fmaf(r57, r29, r90);
    r100 = r25 * r56;
    r90 = fmaf(r35, r100, r90);
    r86 = r25 * r56;
    r86 = r86 * r51;
    r90 = fmaf(r35, r86, r90);
    r90 = fmaf(r41, r60, r90);
    r90 = fmaf(r37, r62, r90);
    r86 = r2 * r90;
    r100 = r37 * r9;
    r100 = r100 * r63;
    r100 = fmaf(r10, r100, r56 * r83);
    r100 = r100 + r75;
    r100 = fmaf(r37, r57, r6 * r100);
    r75 = r25 * r56;
    r75 = r75 * r9;
    r75 = r75 * r51;
    r100 = fmaf(r10, r75, r100);
    r29 = r7 * r14;
    r29 = r29 * r41;
    r29 = r29 * r9;
    r100 = fmaf(r10, r29, r100);
    r13 = r9 * r12;
    r100 = fmaf(r57, r13, r100);
    r85 = r25 * r56;
    r85 = r85 * r9;
    r100 = fmaf(r10, r85, r100);
    r82 = r7 * r37;
    r100 = fmaf(r61, r82, r100);
    r100 = fmaf(r56, r93, r100);
    r100 = fmaf(r37, r60, r100);
    r82 = r3 * r100;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r28, r11,
        r86, r82);
    r82 = r38 * r8;
    r82 = r82 * r8;
    r82 = r82 * r1;
    r86 = r55 * r63;
    r86 = fmaf(r35, r86, r52 * r82);
    r82 = r39 * r38;
    r82 = r82 * r9;
    r11 = r14 * r42;
    r11 = r11 * r9;
    r11 = fmaf(r10, r11, r73 * r82);
    r86 = r86 + r11;
    r82 = r25 * r38;
    r82 = r82 * r51;
    r82 = fmaf(r35, r82, r7 * r86);
    r86 = r6 * r14;
    r86 = r86 * r55;
    r86 = r86 * r9;
    r82 = fmaf(r10, r86, r82);
    r28 = r39 * r38;
    r28 = r28 * r8;
    r28 = r28 * r8;
    r28 = fmaf(r55, r61, r52 * r28);
    r11 = r11 + r28;
    r11 = fmaf(r11, r5, r4 * r11);
    r85 = r8 * r11;
    r82 = fmaf(r57, r85, r82);
    r13 = r6 * r38;
    r82 = fmaf(r87, r13, r82);
    r29 = r25 * r38;
    r82 = fmaf(r35, r29, r82);
    r82 = fmaf(r55, r57, r82);
    r82 = fmaf(r42, r62, r82);
    r82 = fmaf(r55, r60, r82);
    r29 = r2 * r82;
    r13 = r42 * r9;
    r13 = r13 * r63;
    r13 = fmaf(r10, r13, r38 * r83);
    r13 = r13 + r28;
    r13 = fmaf(r42, r57, r6 * r13);
    r28 = r7 * r14;
    r28 = r28 * r55;
    r28 = r28 * r9;
    r13 = fmaf(r10, r28, r13);
    r85 = r7 * r42;
    r13 = fmaf(r61, r85, r13);
    r86 = r25 * r38;
    r86 = r86 * r9;
    r13 = fmaf(r10, r86, r13);
    r75 = r9 * r11;
    r13 = fmaf(r57, r75, r13);
    r64 = r25 * r38;
    r64 = r64 * r9;
    r64 = r64 * r51;
    r13 = fmaf(r10, r64, r13);
    r13 = fmaf(r42, r60, r13);
    r13 = fmaf(r38, r93, r13);
    r64 = r3 * r13;
    r75 = r53 * r63;
    r86 = r58 * r8;
    r86 = r86 * r8;
    r86 = r86 * r1;
    r86 = fmaf(r52, r86, r35 * r75);
    r75 = r14 * r44;
    r75 = r75 * r9;
    r85 = r39 * r58;
    r85 = r85 * r9;
    r85 = fmaf(r73, r85, r10 * r75);
    r86 = r86 + r85;
    r75 = r6 * r58;
    r75 = fmaf(r87, r75, r7 * r86);
    r86 = r39 * r58;
    r86 = r86 * r8;
    r86 = r86 * r8;
    r86 = fmaf(r52, r86, r53 * r61);
    r85 = r85 + r86;
    r85 = fmaf(r85, r5, r4 * r85);
    r28 = r8 * r85;
    r75 = fmaf(r57, r28, r75);
    r68 = r25 * r58;
    r68 = r68 * r51;
    r75 = fmaf(r35, r68, r75);
    r77 = r25 * r58;
    r75 = fmaf(r35, r77, r75);
    r88 = r6 * r14;
    r88 = r88 * r53;
    r88 = r88 * r9;
    r75 = fmaf(r10, r88, r75);
    r75 = fmaf(r53, r57, r75);
    r75 = fmaf(r44, r62, r75);
    r75 = fmaf(r53, r60, r75);
    r88 = r2 * r75;
    r77 = r44 * r9;
    r77 = r77 * r63;
    r77 = fmaf(r58, r83, r10 * r77);
    r77 = r77 + r86;
    r77 = fmaf(r58, r93, r6 * r77);
    r86 = r25 * r58;
    r86 = r86 * r9;
    r86 = r86 * r51;
    r77 = fmaf(r10, r86, r77);
    r68 = r7 * r44;
    r77 = fmaf(r61, r68, r77);
    r28 = r25 * r58;
    r28 = r28 * r9;
    r77 = fmaf(r10, r28, r77);
    r26 = r9 * r85;
    r77 = fmaf(r57, r26, r77);
    r92 = r7 * r14;
    r92 = r92 * r53;
    r92 = r92 * r9;
    r77 = fmaf(r10, r92, r77);
    r77 = fmaf(r44, r57, r77);
    r77 = fmaf(r44, r60, r77);
    r92 = r3 * r77;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r29, r64,
        r88, r92);
    r92 = r3 * r25;
    r92 = r92 * r0;
    r34 = r25 * r34;
    r88 = r2 * r34;
    r92 = fmaf(r80, r88, r89 * r92);
    r64 = r3 * r25;
    r64 = r64 * r0;
    r64 = fmaf(r99, r88, r72 * r64);
    r29 = r3 * r25;
    r29 = r29 * r0;
    r29 = fmaf(r84, r88, r74 * r29);
    r26 = r3 * r25;
    r26 = r26 * r0;
    r26 = fmaf(r90, r88, r100 * r26);
    WriteSum4<float, float>((float *)inout_shared, r92, r64, r29, r26);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = r3 * r25;
    r26 = r26 * r0;
    r26 = fmaf(r82, r88, r13 * r26);
    r29 = r3 * r25;
    r29 = r29 * r0;
    r29 = fmaf(r75, r88, r77 * r29);
    WriteSum2<float, float>((float *)inout_shared, r26, r29);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = r3 * r3;
    r26 = r89 * r29;
    r64 = r2 * r2;
    r92 = r80 * r64;
    r80 = fmaf(r80, r92, r89 * r26);
    r89 = r99 * r99;
    r28 = r72 * r72;
    r28 = fmaf(r29, r28, r64 * r89);
    r89 = r74 * r74;
    r68 = r84 * r84;
    r68 = fmaf(r64, r68, r29 * r89);
    r89 = r90 * r90;
    r86 = r100 * r100;
    r86 = fmaf(r29, r86, r64 * r89);
    WriteSum4<float, float>((float *)inout_shared, r80, r28, r68, r86);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = r13 * r13;
    r68 = r82 * r82;
    r68 = fmaf(r64, r68, r29 * r86);
    r86 = r77 * r77;
    r28 = r75 * r75;
    r28 = fmaf(r64, r28, r29 * r86);
    WriteSum2<float, float>((float *)inout_shared, r68, r28);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fmaf(r99, r92, r72 * r26);
    r68 = fmaf(r74, r26, r84 * r92);
    r86 = fmaf(r100, r26, r90 * r92);
    r80 = fmaf(r82, r92, r13 * r26);
    WriteSum4<float, float>((float *)inout_shared, r28, r68, r86, r80);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r92 = fmaf(r75, r92, r77 * r26);
    r26 = r99 * r84;
    r80 = r72 * r74;
    r80 = fmaf(r29, r80, r64 * r26);
    r26 = r72 * r100;
    r86 = r99 * r90;
    r86 = fmaf(r64, r86, r29 * r26);
    r26 = r72 * r13;
    r68 = r99 * r82;
    r68 = fmaf(r64, r68, r29 * r26);
    WriteSum4<float, float>((float *)inout_shared, r92, r80, r86, r68);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r99 * r75;
    r86 = r72 * r77;
    r86 = fmaf(r29, r86, r64 * r68);
    r68 = r84 * r90;
    r80 = r74 * r100;
    r80 = fmaf(r29, r80, r64 * r68);
    r68 = r84 * r82;
    r92 = r74 * r13;
    r92 = fmaf(r29, r92, r64 * r68);
    r68 = r84 * r75;
    r26 = r74 * r77;
    r26 = fmaf(r29, r26, r64 * r68);
    WriteSum4<float, float>((float *)inout_shared, r86, r80, r92, r26);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = r100 * r13;
    r92 = r90 * r82;
    r92 = fmaf(r64, r92, r29 * r26);
    r26 = r100 * r77;
    r80 = r90 * r75;
    r80 = fmaf(r64, r80, r29 * r26);
    r26 = r82 * r75;
    r86 = r13 * r77;
    r86 = fmaf(r29, r86, r64 * r26);
    WriteSum3<float, float>((float *)inout_shared, r92, r80, r86);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = r25 * r0;
    WriteSum2<float, float>((float *)inout_shared, r34, r86);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float *)inout_shared, r40, r40);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r45 * r63;
    r86 = r54 * r8;
    r86 = r86 * r8;
    r86 = r86 * r1;
    r86 = fmaf(r52, r86, r35 * r40);
    r40 = r14 * r31;
    r40 = r40 * r9;
    r34 = r39 * r54;
    r34 = r34 * r9;
    r34 = fmaf(r73, r34, r10 * r40);
    r86 = r86 + r34;
    r86 = fmaf(r45, r57, r7 * r86);
    r40 = r6 * r54;
    r86 = fmaf(r87, r40, r86);
    r80 = r6 * r14;
    r80 = r80 * r45;
    r80 = r80 * r9;
    r86 = fmaf(r10, r80, r86);
    r92 = r25 * r54;
    r86 = fmaf(r35, r92, r86);
    r26 = r39 * r54;
    r26 = r26 * r8;
    r26 = r26 * r8;
    r26 = fmaf(r52, r26, r45 * r61);
    r34 = r34 + r26;
    r34 = fmaf(r34, r5, r4 * r34);
    r68 = r8 * r34;
    r86 = fmaf(r57, r68, r86);
    r28 = r25 * r54;
    r28 = r28 * r51;
    r86 = fmaf(r35, r28, r86);
    r86 = fmaf(r31, r62, r86);
    r86 = fmaf(r45, r60, r86);
    r28 = r2 * r86;
    r68 = r31 * r9;
    r68 = r68 * r63;
    r68 = fmaf(r54, r83, r10 * r68);
    r68 = r68 + r26;
    r68 = fmaf(r54, r93, r6 * r68);
    r26 = r7 * r31;
    r68 = fmaf(r61, r26, r68);
    r92 = r7 * r14;
    r92 = r92 * r45;
    r92 = r92 * r9;
    r68 = fmaf(r10, r92, r68);
    r80 = r25 * r54;
    r80 = r80 * r9;
    r68 = fmaf(r10, r80, r68);
    r40 = r9 * r34;
    r68 = fmaf(r57, r40, r68);
    r89 = r25 * r54;
    r89 = r89 * r9;
    r89 = r89 * r51;
    r68 = fmaf(r10, r89, r68);
    r68 = fmaf(r31, r57, r68);
    r68 = fmaf(r31, r60, r68);
    r89 = r3 * r68;
    r40 = r24 * r63;
    r80 = r43 * r8;
    r80 = r80 * r8;
    r80 = r80 * r1;
    r80 = fmaf(r52, r80, r35 * r40);
    r40 = r39 * r43;
    r40 = r40 * r9;
    r92 = r14 * r49;
    r92 = r92 * r9;
    r92 = fmaf(r10, r92, r73 * r40);
    r80 = r80 + r92;
    r40 = r6 * r43;
    r40 = fmaf(r87, r40, r7 * r80);
    r80 = r39 * r43;
    r80 = r80 * r8;
    r80 = r80 * r8;
    r80 = fmaf(r52, r80, r24 * r61);
    r92 = r92 + r80;
    r92 = fmaf(r92, r5, r4 * r92);
    r26 = r8 * r92;
    r40 = fmaf(r57, r26, r40);
    r66 = r6 * r14;
    r66 = r66 * r24;
    r66 = r66 * r9;
    r40 = fmaf(r10, r66, r40);
    r67 = r25 * r43;
    r67 = r67 * r51;
    r40 = fmaf(r35, r67, r40);
    r95 = r25 * r43;
    r40 = fmaf(r35, r95, r40);
    r40 = fmaf(r24, r57, r40);
    r40 = fmaf(r24, r60, r40);
    r40 = fmaf(r49, r62, r40);
    r95 = r2 * r40;
    r67 = r49 * r9;
    r67 = r67 * r63;
    r67 = fmaf(r10, r67, r43 * r83);
    r67 = r67 + r80;
    r67 = fmaf(r49, r57, r6 * r67);
    r80 = r9 * r92;
    r67 = fmaf(r57, r80, r67);
    r66 = r25 * r43;
    r66 = r66 * r9;
    r67 = fmaf(r10, r66, r67);
    r26 = r25 * r43;
    r26 = r26 * r9;
    r26 = r26 * r51;
    r67 = fmaf(r10, r26, r67);
    r91 = r7 * r14;
    r91 = r91 * r24;
    r91 = r91 * r9;
    r67 = fmaf(r10, r91, r67);
    r36 = r7 * r49;
    r67 = fmaf(r61, r36, r67);
    r67 = fmaf(r43, r93, r67);
    r67 = fmaf(r49, r60, r67);
    r36 = r3 * r67;
    WriteIdx4<1024, float, float, float4>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r28, r89,
        r95, r36);
    r36 = r59 * r8;
    r36 = r36 * r8;
    r36 = r36 * r1;
    r1 = r48 * r63;
    r1 = fmaf(r35, r1, r52 * r36);
    r36 = r14 * r47;
    r36 = r36 * r9;
    r95 = r39 * r59;
    r95 = r95 * r9;
    r95 = fmaf(r73, r95, r10 * r36);
    r1 = r1 + r95;
    r62 = fmaf(r47, r62, r7 * r1);
    r1 = r25 * r59;
    r62 = fmaf(r35, r1, r62);
    r36 = r6 * r59;
    r62 = fmaf(r87, r36, r62);
    r87 = r39 * r59;
    r87 = r87 * r8;
    r87 = r87 * r8;
    r87 = fmaf(r48, r61, r52 * r87);
    r95 = r95 + r87;
    r5 = fmaf(r95, r5, r4 * r95);
    r95 = r8 * r5;
    r62 = fmaf(r57, r95, r62);
    r4 = r6 * r14;
    r4 = r4 * r48;
    r4 = r4 * r9;
    r62 = fmaf(r10, r4, r62);
    r52 = r25 * r59;
    r52 = r52 * r51;
    r62 = fmaf(r35, r52, r62);
    r62 = fmaf(r48, r57, r62);
    r62 = fmaf(r48, r60, r62);
    r2 = r2 * r62;
    r52 = r47 * r9;
    r52 = r52 * r63;
    r83 = fmaf(r59, r83, r10 * r52);
    r83 = r83 + r87;
    r87 = r7 * r47;
    r87 = fmaf(r61, r87, r6 * r83);
    r83 = r25 * r59;
    r83 = r83 * r9;
    r87 = fmaf(r10, r83, r87);
    r61 = r25 * r59;
    r61 = r61 * r9;
    r61 = r61 * r51;
    r87 = fmaf(r10, r61, r87);
    r52 = r9 * r5;
    r87 = fmaf(r57, r52, r87);
    r4 = r7 * r14;
    r4 = r4 * r48;
    r4 = r4 * r9;
    r87 = fmaf(r10, r4, r87);
    r87 = fmaf(r47, r57, r87);
    r87 = fmaf(r59, r93, r87);
    r87 = fmaf(r47, r60, r87);
    r60 = r3 * r87;
    WriteIdx2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r2, r60);
    r60 = r3 * r25;
    r60 = r60 * r0;
    r60 = fmaf(r86, r88, r68 * r60);
    r2 = r3 * r25;
    r2 = r2 * r0;
    r2 = fmaf(r40, r88, r67 * r2);
    r4 = r3 * r25;
    r4 = r4 * r0;
    r88 = fmaf(r62, r88, r87 * r4);
    WriteSum3<float, float>((float *)inout_shared, r60, r2, r88);
  };
  FlushSumShared<3, float>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r88 = r68 * r68;
    r2 = r86 * r86;
    r2 = fmaf(r64, r2, r29 * r88);
    r88 = r40 * r40;
    r60 = r67 * r67;
    r60 = fmaf(r29, r60, r64 * r88);
    r88 = r62 * r62;
    r4 = r87 * r87;
    r4 = fmaf(r29, r4, r64 * r88);
    WriteSum3<float, float>((float *)inout_shared, r2, r60, r4);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = r68 * r67;
    r60 = r86 * r40;
    r60 = fmaf(r64, r60, r29 * r4);
    r4 = r68 * r87;
    r2 = r86 * r62;
    r2 = fmaf(r64, r2, r29 * r4);
    r4 = r40 * r62;
    r88 = r67 * r87;
    r88 = fmaf(r29, r88, r64 * r4);
    WriteSum3<float, float>((float *)inout_shared, r60, r2, r88);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc, (float *)inout_shared);
}

void OpencvSplitFixedFocalAndExtraResJac(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *principal_point, unsigned int principal_point_num_alloc,
    SharedIndex *principal_point_indices, float *point,
    unsigned int point_num_alloc, SharedIndex *point_indices, float *pixel,
    unsigned int pixel_num_alloc, float *focal_and_extra,
    unsigned int focal_and_extra_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float *out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    float *const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float *const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float *const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    float *out_point_jac, unsigned int out_point_jac_num_alloc,
    float *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
    float *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvSplitFixedFocalAndExtraResJacKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, principal_point, principal_point_num_alloc,
      principal_point_indices, point, point_num_alloc, point_indices, pixel,
      pixel_num_alloc, focal_and_extra, focal_and_extra_num_alloc, out_res,
      out_res_num_alloc, out_pose_jac, out_pose_jac_num_alloc, out_pose_njtr,
      out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, out_principal_point_jac,
      out_principal_point_jac_num_alloc, out_principal_point_njtr,
      out_principal_point_njtr_num_alloc, out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc, out_point_jac,
      out_point_jac_num_alloc, out_point_njtr, out_point_njtr_num_alloc,
      out_point_precond_diag, out_point_precond_diag_num_alloc,
      out_point_precond_tril, out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar