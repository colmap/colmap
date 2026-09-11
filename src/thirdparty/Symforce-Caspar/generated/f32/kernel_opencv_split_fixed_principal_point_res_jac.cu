#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_split_fixed_principal_point_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedPrincipalPointResJacKernel(
        float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        float *focal_and_extra, unsigned int focal_and_extra_num_alloc,
        SharedIndex *focal_and_extra_indices, float *point,
        unsigned int point_num_alloc, SharedIndex *point_indices, float *pixel,
        unsigned int pixel_num_alloc, float *principal_point,
        unsigned int principal_point_num_alloc, float *out_res,
        unsigned int out_res_num_alloc, float *out_pose_jac,
        unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float *const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float *const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        float *out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        float *const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        float *const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        float *const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
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

  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
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
      r105, r106, r107, r108, r109;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx, r0, r1);
  };
  LoadShared<4, float, float>(focal_and_extra, 0 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target, r2, r3,
                       r4, r5);
  };
  __syncthreads();
  LoadShared<2, float, float>(focal_and_extra, 4 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float *)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r36 = fmaf(r8, r57, r7 * r34);
    r40 = r8 * r35;
    r51 = r40 + r51;
    r60 = r51 * r51;
    r61 = fmaf(r5, r60, r4 * r51);
    r62 = r61 * r57;
    r63 = r14 * r35;
    r64 = r6 * r63;
    r36 = fmaf(r8, r62, r36);
    r36 = fmaf(r9, r64, r36);
    r0 = fmaf(r2, r36, r0);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r65, r66);
    r0 = fmaf(r65, r25, r0);
    r65 = r9 * r9;
    r65 = r65 * r33;
    r65 = fmaf(r10, r65, r40);
    r40 = fmaf(r9, r57, r6 * r65);
    r67 = r7 * r9;
    r40 = fmaf(r63, r67, r40);
    r40 = fmaf(r9, r62, r40);
    r1 = fmaf(r3, r40, r1);
    r1 = fmaf(r66, r25, r1);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r0, r1);
    r66 = 6.00000000000000000e+00;
    r67 = r14 * r32;
    r68 = r16 * r19;
    r69 = 5.00000000000000000e-01;
    r70 = r15 * r20;
    r71 = -5.00000000000000000e-01;
    r70 = fmaf(r71, r70, r69 * r68);
    r68 = r18 * r21;
    r70 = fmaf(r69, r68, r70);
    r72 = r22 * r69;
    r70 = fmaf(r17, r72, r70);
    r68 = r15 * r22;
    r73 = r18 * r19;
    r73 = fmaf(r71, r73, r71 * r68);
    r68 = r17 * r20;
    r73 = fmaf(r71, r68, r73);
    r74 = r16 * r21;
    r73 = fmaf(r69, r74, r73);
    r74 = r27 * r73;
    r67 = fmaf(r14, r74, r70 * r67);
    r68 = r14 * r23;
    r75 = r16 * r71;
    r76 = fmaf(r69, r29, r22 * r75);
    r76 = fmaf(r71, r26, r76);
    r76 = fmaf(r71, r28, r76);
    r77 = r14 * r30;
    r78 = r15 * r19;
    r79 = r17 * r21;
    r79 = fmaf(r71, r79, r71 * r78);
    r79 = fmaf(r18, r72, r79);
    r79 = fmaf(r20, r75, r79);
    r77 = r77 * r79;
    r68 = fmaf(r76, r68, r77);
    r67 = r67 + r68;
    r78 = r14 * r27;
    r78 = r78 * r79;
    r80 = r14 * r23;
    r80 = r80 * r70;
    r81 = r78 + r80;
    r82 = r30 * r39;
    r81 = fmaf(r73, r82, r81);
    r81 = fmaf(r76, r46, r81);
    r81 = fmaf(r12, r81, r13 * r67);
    r67 = r27 * r70;
    r82 = -4.00000000000000000e+00;
    r67 = r67 * r82;
    r83 = r30 * r76;
    r84 = r82 * r83;
    r85 = r67 + r84;
    r81 = fmaf(r11, r85, r81);
    r85 = r66 * r81;
    r86 = r8 * r8;
    r87 = r14 * r27;
    r87 = r87 * r76;
    r88 = r14 * r30;
    r88 = fmaf(r70, r88, r87);
    r89 = r14 * r23;
    r89 = r89 * r73;
    r90 = r14 * r32;
    r90 = r90 * r79;
    r91 = r89 + r90;
    r92 = r88 + r91;
    r70 = fmaf(r70, r46, r39 * r74);
    r70 = r70 + r68;
    r70 = fmaf(r11, r70, r12 * r92);
    r92 = r23 * r79;
    r92 = r92 * r82;
    r67 = r92 + r67;
    r70 = fmaf(r13, r67, r70);
    r67 = -6.00000000000000000e+00;
    r52 = r50 * r52;
    r93 = 1.0 / r52;
    r86 = r86 * r70;
    r86 = r86 * r67;
    r86 = fmaf(r93, r86, r35 * r85);
    r85 = r14 * r9;
    r94 = r23 * r39;
    r95 = r79 * r46;
    r94 = fmaf(r73, r94, r95);
    r94 = r94 + r88;
    r84 = r92 + r84;
    r84 = fmaf(r12, r84, r13 * r94);
    r94 = r14 * r32;
    r94 = fmaf(r76, r94, r80);
    r80 = r14 * r30;
    r80 = fmaf(r73, r80, r78);
    r94 = r94 + r80;
    r84 = fmaf(r11, r94, r84);
    r85 = r85 * r84;
    r94 = r39 * r9;
    r78 = r9 * r93;
    r94 = r94 * r70;
    r94 = fmaf(r78, r94, r10 * r85);
    r86 = r86 + r94;
    r85 = r25 * r61;
    r85 = r85 * r70;
    r85 = fmaf(r35, r85, r7 * r86);
    r86 = r6 * r70;
    r92 = r8 * r78;
    r88 = r82 * r92;
    r85 = fmaf(r88, r86, r85);
    r96 = r6 * r14;
    r96 = r96 * r9;
    r96 = r96 * r81;
    r85 = fmaf(r10, r96, r85);
    r97 = r25 * r70;
    r85 = fmaf(r35, r97, r85);
    r98 = r39 * r8;
    r98 = r98 * r8;
    r98 = r98 * r70;
    r98 = fmaf(r93, r98, r81 * r63);
    r94 = r94 + r98;
    r99 = r5 * r14;
    r99 = r99 * r51;
    r99 = fmaf(r94, r99, r4 * r94);
    r94 = r8 * r99;
    r85 = fmaf(r57, r94, r85);
    r85 = fmaf(r84, r64, r85);
    r85 = fmaf(r81, r62, r85);
    r85 = fmaf(r81, r57, r85);
    r94 = r2 * r85;
    r97 = r9 * r84;
    r97 = r97 * r66;
    r96 = r9 * r67;
    r96 = r96 * r78;
    r97 = fmaf(r70, r96, r10 * r97);
    r97 = r97 + r98;
    r97 = fmaf(r84, r57, r6 * r97);
    r98 = r25 * r9;
    r98 = r98 * r70;
    r97 = fmaf(r10, r98, r97);
    r86 = r7 * r84;
    r97 = fmaf(r63, r86, r97);
    r100 = r9 * r99;
    r97 = fmaf(r57, r100, r97);
    r101 = r7 * r88;
    r102 = r7 * r14;
    r102 = r102 * r9;
    r102 = r102 * r81;
    r97 = fmaf(r10, r102, r97);
    r103 = r25 * r9;
    r103 = r103 * r61;
    r103 = r103 * r70;
    r97 = fmaf(r10, r103, r97);
    r97 = fmaf(r84, r62, r97);
    r97 = fmaf(r70, r101, r97);
    r103 = r3 * r97;
    r90 = r87 + r90;
    r87 = r14 * r30;
    r102 = r17 * r22;
    r100 = r15 * r20;
    r100 = fmaf(r69, r100, r71 * r102);
    r102 = r18 * r21;
    r100 = fmaf(r71, r102, r100);
    r100 = fmaf(r19, r75, r100);
    r87 = r87 * r100;
    r102 = r14 * r23;
    r86 = r18 * r19;
    r98 = r17 * r20;
    r98 = fmaf(r69, r98, r69 * r86);
    r98 = fmaf(r15, r72, r98);
    r98 = fmaf(r21, r75, r98);
    r102 = fmaf(r98, r102, r87);
    r90 = r90 + r102;
    r75 = r27 * r79;
    r75 = r75 * r82;
    r86 = r30 * r82;
    r86 = r86 * r98;
    r104 = r75 + r86;
    r104 = fmaf(r11, r104, r13 * r90);
    r90 = fmaf(r98, r46, r39 * r83);
    r105 = r14 * r23;
    r105 = r105 * r79;
    r106 = r14 * r27;
    r106 = fmaf(r100, r106, r105);
    r90 = r90 + r106;
    r104 = fmaf(r12, r90, r104);
    r90 = r66 * r104;
    r107 = r8 * r8;
    r108 = r39 * r27;
    r108 = fmaf(r76, r108, r95);
    r108 = r108 + r102;
    r102 = r14 * r27;
    r102 = r102 * r98;
    r109 = r14 * r32;
    r109 = fmaf(r100, r109, r102);
    r109 = r109 + r68;
    r109 = fmaf(r12, r109, r11 * r108);
    r108 = r23 * r100;
    r68 = r82 * r108;
    r75 = r75 + r68;
    r109 = fmaf(r13, r75, r109);
    r107 = r107 * r67;
    r107 = r107 * r109;
    r107 = fmaf(r93, r107, r35 * r90);
    r90 = r39 * r9;
    r90 = r90 * r109;
    r75 = r14 * r9;
    r102 = r77 + r102;
    r77 = r23 * r39;
    r102 = fmaf(r76, r77, r102);
    r102 = fmaf(r100, r46, r102);
    r77 = r14 * r32;
    r83 = fmaf(r14, r83, r98 * r77);
    r83 = r83 + r106;
    r83 = fmaf(r11, r83, r13 * r102);
    r68 = r86 + r68;
    r83 = fmaf(r12, r68, r83);
    r75 = r75 * r83;
    r75 = fmaf(r10, r75, r78 * r90);
    r107 = r107 + r75;
    r107 = fmaf(r104, r62, r7 * r107);
    r90 = r25 * r109;
    r107 = fmaf(r35, r90, r107);
    r68 = r6 * r109;
    r107 = fmaf(r88, r68, r107);
    r86 = r6 * r14;
    r86 = r86 * r9;
    r86 = r86 * r104;
    r107 = fmaf(r10, r86, r107);
    r102 = r25 * r61;
    r102 = r102 * r109;
    r107 = fmaf(r35, r102, r107);
    r77 = r5 * r14;
    r98 = r39 * r8;
    r98 = r98 * r8;
    r98 = r98 * r109;
    r98 = fmaf(r93, r98, r104 * r63);
    r75 = r75 + r98;
    r77 = r77 * r51;
    r75 = fmaf(r4, r75, r75 * r77);
    r77 = r8 * r75;
    r107 = fmaf(r57, r77, r107);
    r107 = fmaf(r104, r57, r107);
    r107 = fmaf(r83, r64, r107);
    r77 = r2 * r107;
    r102 = r9 * r66;
    r102 = r102 * r83;
    r102 = fmaf(r10, r102, r109 * r96);
    r102 = r102 + r98;
    r98 = r25 * r9;
    r98 = r98 * r61;
    r98 = r98 * r109;
    r98 = fmaf(r10, r98, r6 * r102);
    r102 = r9 * r75;
    r98 = fmaf(r57, r102, r98);
    r86 = r25 * r9;
    r86 = r86 * r109;
    r98 = fmaf(r10, r86, r98);
    r68 = r7 * r14;
    r68 = r68 * r9;
    r68 = r68 * r104;
    r98 = fmaf(r10, r68, r98);
    r90 = r7 * r83;
    r98 = fmaf(r63, r90, r98);
    r98 = fmaf(r83, r57, r98);
    r98 = fmaf(r109, r101, r98);
    r98 = fmaf(r83, r62, r98);
    r90 = r3 * r98;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r94, r103,
        r77, r90);
    r90 = r8 * r8;
    r77 = r23 * r82;
    r29 = fmaf(r71, r29, r16 * r72);
    r29 = fmaf(r69, r26, r29);
    r29 = fmaf(r69, r28, r29);
    r77 = r77 * r29;
    r74 = r82 * r74;
    r28 = r77 + r74;
    r69 = r14 * r30;
    r69 = r69 * r29;
    r105 = r105 + r69;
    r26 = r39 * r27;
    r105 = fmaf(r100, r26, r105);
    r105 = fmaf(r73, r46, r105);
    r105 = fmaf(r11, r105, r13 * r28);
    r28 = r14 * r32;
    r28 = fmaf(r14, r108, r29 * r28);
    r28 = r28 + r80;
    r105 = fmaf(r12, r28, r105);
    r90 = r90 * r67;
    r90 = r90 * r105;
    r28 = r14 * r27;
    r28 = r28 * r29;
    r89 = r89 + r28;
    r26 = r30 * r39;
    r89 = fmaf(r100, r26, r89);
    r89 = r89 + r95;
    r79 = r30 * r79;
    r79 = r79 * r82;
    r74 = r79 + r74;
    r74 = fmaf(r11, r74, r12 * r89);
    r89 = r14 * r32;
    r89 = fmaf(r73, r89, r69);
    r89 = r89 + r106;
    r74 = fmaf(r13, r89, r74);
    r89 = r66 * r74;
    r89 = fmaf(r35, r89, r93 * r90);
    r90 = r39 * r9;
    r90 = r90 * r105;
    r106 = r14 * r9;
    r28 = r87 + r28;
    r28 = r28 + r91;
    r46 = fmaf(r29, r46, r39 * r108);
    r46 = r46 + r80;
    r46 = fmaf(r13, r46, r11 * r28);
    r77 = r79 + r77;
    r46 = fmaf(r12, r77, r46);
    r106 = r106 * r46;
    r106 = fmaf(r10, r106, r78 * r90);
    r89 = r89 + r106;
    r89 = fmaf(r74, r57, r7 * r89);
    r90 = r25 * r61;
    r90 = r90 * r105;
    r89 = fmaf(r35, r90, r89);
    r77 = r25 * r105;
    r89 = fmaf(r35, r77, r89);
    r12 = r6 * r14;
    r12 = r12 * r9;
    r12 = r12 * r74;
    r89 = fmaf(r10, r12, r89);
    r79 = r6 * r105;
    r89 = fmaf(r88, r79, r89);
    r13 = r5 * r14;
    r28 = r39 * r8;
    r28 = r28 * r8;
    r28 = r28 * r105;
    r28 = fmaf(r74, r63, r93 * r28);
    r106 = r106 + r28;
    r13 = r13 * r51;
    r106 = fmaf(r4, r106, r106 * r13);
    r13 = r8 * r106;
    r89 = fmaf(r57, r13, r89);
    r89 = fmaf(r46, r64, r89);
    r89 = fmaf(r74, r62, r89);
    r13 = r2 * r89;
    r79 = r9 * r66;
    r79 = r79 * r46;
    r79 = fmaf(r10, r79, r105 * r96);
    r79 = r79 + r28;
    r79 = fmaf(r46, r57, r6 * r79);
    r28 = r7 * r14;
    r28 = r28 * r9;
    r28 = r28 * r74;
    r79 = fmaf(r10, r28, r79);
    r12 = r9 * r106;
    r79 = fmaf(r57, r12, r79);
    r77 = r7 * r46;
    r79 = fmaf(r63, r77, r79);
    r90 = r25 * r9;
    r90 = r90 * r105;
    r79 = fmaf(r10, r90, r79);
    r11 = r25 * r9;
    r11 = r11 * r61;
    r11 = r11 * r105;
    r79 = fmaf(r10, r11, r79);
    r79 = fmaf(r105, r101, r79);
    r79 = fmaf(r46, r62, r79);
    r11 = r3 * r79;
    r90 = r56 * r8;
    r90 = r90 * r8;
    r90 = r90 * r67;
    r77 = r41 * r66;
    r77 = fmaf(r35, r77, r93 * r90);
    r90 = r39 * r56;
    r90 = r90 * r9;
    r12 = r14 * r37;
    r12 = r12 * r9;
    r12 = fmaf(r10, r12, r78 * r90);
    r77 = r77 + r12;
    r77 = fmaf(r41, r57, r7 * r77);
    r90 = r6 * r14;
    r90 = r90 * r41;
    r90 = r90 * r9;
    r77 = fmaf(r10, r90, r77);
    r28 = r6 * r56;
    r77 = fmaf(r88, r28, r77);
    r80 = r39 * r56;
    r80 = r80 * r8;
    r80 = r80 * r8;
    r80 = fmaf(r41, r63, r93 * r80);
    r12 = r12 + r80;
    r29 = r5 * r14;
    r29 = r29 * r51;
    r29 = fmaf(r12, r29, r4 * r12);
    r12 = r8 * r29;
    r77 = fmaf(r57, r12, r77);
    r108 = r25 * r56;
    r77 = fmaf(r35, r108, r77);
    r91 = r25 * r56;
    r91 = r91 * r61;
    r77 = fmaf(r35, r91, r77);
    r77 = fmaf(r41, r62, r77);
    r77 = fmaf(r37, r64, r77);
    r91 = r2 * r77;
    r108 = r37 * r9;
    r108 = r108 * r66;
    r108 = fmaf(r10, r108, r56 * r96);
    r108 = r108 + r80;
    r108 = fmaf(r37, r57, r6 * r108);
    r80 = r25 * r56;
    r80 = r80 * r9;
    r80 = r80 * r61;
    r108 = fmaf(r10, r80, r108);
    r12 = r7 * r14;
    r12 = r12 * r41;
    r12 = r12 * r9;
    r108 = fmaf(r10, r12, r108);
    r28 = r9 * r29;
    r108 = fmaf(r57, r28, r108);
    r90 = r25 * r56;
    r90 = r90 * r9;
    r108 = fmaf(r10, r90, r108);
    r87 = r7 * r37;
    r108 = fmaf(r63, r87, r108);
    r108 = fmaf(r56, r101, r108);
    r108 = fmaf(r37, r62, r108);
    r87 = r3 * r108;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r13, r11,
        r91, r87);
    r87 = r38 * r8;
    r87 = r87 * r8;
    r87 = r87 * r67;
    r91 = r55 * r66;
    r91 = fmaf(r35, r91, r93 * r87);
    r87 = r39 * r38;
    r87 = r87 * r9;
    r11 = r14 * r42;
    r11 = r11 * r9;
    r11 = fmaf(r10, r11, r78 * r87);
    r91 = r91 + r11;
    r87 = r25 * r38;
    r87 = r87 * r61;
    r87 = fmaf(r35, r87, r7 * r91);
    r91 = r6 * r14;
    r91 = r91 * r55;
    r91 = r91 * r9;
    r87 = fmaf(r10, r91, r87);
    r13 = r5 * r14;
    r90 = r39 * r38;
    r90 = r90 * r8;
    r90 = r90 * r8;
    r90 = fmaf(r55, r63, r93 * r90);
    r11 = r11 + r90;
    r13 = r13 * r51;
    r11 = fmaf(r4, r11, r11 * r13);
    r13 = r8 * r11;
    r87 = fmaf(r57, r13, r87);
    r28 = r6 * r38;
    r87 = fmaf(r88, r28, r87);
    r12 = r25 * r38;
    r87 = fmaf(r35, r12, r87);
    r87 = fmaf(r55, r57, r87);
    r87 = fmaf(r42, r64, r87);
    r87 = fmaf(r55, r62, r87);
    r12 = r2 * r87;
    r28 = r42 * r9;
    r28 = r28 * r66;
    r28 = fmaf(r10, r28, r38 * r96);
    r28 = r28 + r90;
    r28 = fmaf(r42, r57, r6 * r28);
    r90 = r7 * r14;
    r90 = r90 * r55;
    r90 = r90 * r9;
    r28 = fmaf(r10, r90, r28);
    r13 = r7 * r42;
    r28 = fmaf(r63, r13, r28);
    r91 = r25 * r38;
    r91 = r91 * r9;
    r28 = fmaf(r10, r91, r28);
    r80 = r9 * r11;
    r28 = fmaf(r57, r80, r28);
    r69 = r25 * r38;
    r69 = r69 * r9;
    r69 = r69 * r61;
    r28 = fmaf(r10, r69, r28);
    r28 = fmaf(r42, r62, r28);
    r28 = fmaf(r38, r101, r28);
    r69 = r3 * r28;
    r80 = r53 * r66;
    r91 = r58 * r8;
    r91 = r91 * r8;
    r91 = r91 * r67;
    r91 = fmaf(r93, r91, r35 * r80);
    r80 = r14 * r44;
    r80 = r80 * r9;
    r13 = r39 * r58;
    r13 = r13 * r9;
    r13 = fmaf(r78, r13, r10 * r80);
    r91 = r91 + r13;
    r80 = r6 * r58;
    r80 = fmaf(r88, r80, r7 * r91);
    r91 = r39 * r58;
    r91 = r91 * r8;
    r91 = r91 * r8;
    r91 = fmaf(r93, r91, r53 * r63);
    r13 = r13 + r91;
    r90 = r5 * r14;
    r90 = r90 * r51;
    r90 = fmaf(r13, r90, r4 * r13);
    r13 = r8 * r90;
    r80 = fmaf(r57, r13, r80);
    r73 = r25 * r58;
    r73 = r73 * r61;
    r80 = fmaf(r35, r73, r80);
    r82 = r25 * r58;
    r80 = fmaf(r35, r82, r80);
    r95 = r6 * r14;
    r95 = r95 * r53;
    r95 = r95 * r9;
    r80 = fmaf(r10, r95, r80);
    r80 = fmaf(r53, r57, r80);
    r80 = fmaf(r44, r64, r80);
    r80 = fmaf(r53, r62, r80);
    r95 = r2 * r80;
    r82 = r44 * r9;
    r82 = r82 * r66;
    r82 = fmaf(r58, r96, r10 * r82);
    r82 = r82 + r91;
    r82 = fmaf(r58, r101, r6 * r82);
    r91 = r25 * r58;
    r91 = r91 * r9;
    r91 = r91 * r61;
    r82 = fmaf(r10, r91, r82);
    r73 = r7 * r44;
    r82 = fmaf(r63, r73, r82);
    r13 = r25 * r58;
    r13 = r13 * r9;
    r82 = fmaf(r10, r13, r82);
    r26 = r9 * r90;
    r82 = fmaf(r57, r26, r82);
    r100 = r7 * r14;
    r100 = r100 * r53;
    r100 = r100 * r9;
    r82 = fmaf(r10, r100, r82);
    r82 = fmaf(r44, r57, r82);
    r82 = fmaf(r44, r62, r82);
    r100 = r3 * r82;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r12, r69,
        r95, r100);
    r100 = r3 * r25;
    r100 = r100 * r1;
    r95 = r25 * r0;
    r69 = r2 * r95;
    r100 = fmaf(r85, r69, r97 * r100);
    r12 = r3 * r25;
    r12 = r12 * r1;
    r12 = fmaf(r107, r69, r98 * r12);
    r26 = r3 * r25;
    r26 = r26 * r1;
    r26 = fmaf(r89, r69, r79 * r26);
    r13 = r3 * r25;
    r13 = r13 * r1;
    r13 = fmaf(r77, r69, r108 * r13);
    WriteSum4<float, float>((float *)inout_shared, r100, r12, r26, r13);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r3 * r25;
    r13 = r13 * r1;
    r13 = fmaf(r87, r69, r28 * r13);
    r26 = r3 * r25;
    r26 = r26 * r1;
    r26 = fmaf(r80, r69, r82 * r26);
    WriteSum2<float, float>((float *)inout_shared, r13, r26);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = r3 * r3;
    r13 = r97 * r26;
    r12 = r2 * r2;
    r100 = r85 * r12;
    r85 = fmaf(r85, r100, r97 * r13);
    r97 = r107 * r107;
    r73 = r98 * r98;
    r73 = fmaf(r26, r73, r12 * r97);
    r97 = r79 * r79;
    r91 = r89 * r89;
    r91 = fmaf(r12, r91, r26 * r97);
    r97 = r77 * r77;
    r71 = r108 * r108;
    r71 = fmaf(r26, r71, r12 * r97);
    WriteSum4<float, float>((float *)inout_shared, r85, r73, r91, r71);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = r28 * r28;
    r91 = r87 * r87;
    r91 = fmaf(r12, r91, r26 * r71);
    r71 = r82 * r82;
    r73 = r80 * r80;
    r73 = fmaf(r12, r73, r26 * r71);
    WriteSum2<float, float>((float *)inout_shared, r91, r73);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fmaf(r107, r100, r98 * r13);
    r91 = fmaf(r79, r13, r89 * r100);
    r71 = fmaf(r108, r13, r77 * r100);
    r85 = fmaf(r87, r100, r28 * r13);
    WriteSum4<float, float>((float *)inout_shared, r73, r91, r71, r85);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r100 = fmaf(r80, r100, r82 * r13);
    r13 = r107 * r89;
    r85 = r98 * r79;
    r85 = fmaf(r26, r85, r12 * r13);
    r13 = r98 * r108;
    r71 = r107 * r77;
    r71 = fmaf(r12, r71, r26 * r13);
    r13 = r98 * r28;
    r91 = r107 * r87;
    r91 = fmaf(r12, r91, r26 * r13);
    WriteSum4<float, float>((float *)inout_shared, r100, r85, r71, r91);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = r107 * r80;
    r71 = r98 * r82;
    r71 = fmaf(r26, r71, r12 * r91);
    r91 = r89 * r77;
    r85 = r79 * r108;
    r85 = fmaf(r26, r85, r12 * r91);
    r91 = r89 * r87;
    r100 = r79 * r28;
    r100 = fmaf(r26, r100, r12 * r91);
    r91 = r89 * r80;
    r13 = r79 * r82;
    r13 = fmaf(r26, r13, r12 * r91);
    WriteSum4<float, float>((float *)inout_shared, r71, r85, r100, r13);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r108 * r28;
    r100 = r77 * r87;
    r100 = fmaf(r12, r100, r26 * r13);
    r13 = r108 * r82;
    r85 = r77 * r80;
    r85 = fmaf(r12, r85, r26 * r13);
    r13 = r87 * r80;
    r71 = r28 * r82;
    r71 = fmaf(r26, r71, r12 * r13);
    WriteSum3<float, float>((float *)inout_shared, r100, r85, r71);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = r2 * r8;
    r71 = r71 * r51;
    r71 = r71 * r57;
    r85 = r3 * r9;
    r85 = r85 * r51;
    r85 = r85 * r57;
    WriteIdx4<1024, float, float, float4>(
        out_focal_and_extra_jac, 0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx, r36, r40, r71, r85);
    r85 = r2 * r57;
    r85 = r85 * r8;
    r85 = r85 * r60;
    r71 = r3 * r65;
    r100 = r3 * r9;
    r100 = r100 * r57;
    r100 = r100 * r60;
    r13 = r2 * r9;
    r13 = r13 * r63;
    WriteIdx4<1024, float, float, float4>(
        out_focal_and_extra_jac, 4 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx, r85, r100, r13, r71);
    r71 = r2 * r34;
    r13 = r3 * r9;
    r13 = r13 * r63;
    WriteIdx2<1024, float, float, float2>(out_focal_and_extra_jac,
                                          8 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx, r71, r13);
    r13 = r25 * r40;
    r13 = r13 * r1;
    r95 = r36 * r95;
    r71 = r3 * r25;
    r71 = r71 * r9;
    r71 = r71 * r51;
    r71 = r71 * r1;
    r100 = r8 * r51;
    r100 = r100 * r57;
    r100 = fmaf(r69, r100, r57 * r71);
    r71 = r3 * r25;
    r71 = r71 * r9;
    r71 = r71 * r1;
    r71 = r71 * r57;
    r91 = r8 * r60;
    r73 = r57 * r91;
    r71 = fmaf(r69, r73, r60 * r71);
    WriteSum4<float, float>((float *)inout_shared, r95, r13, r100, r71);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           0 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = r3 * r25;
    r71 = r71 * r65;
    r100 = r2 * r39;
    r100 = r100 * r9;
    r100 = r100 * r0;
    r100 = fmaf(r35, r100, r1 * r71);
    r71 = r3 * r39;
    r71 = r71 * r9;
    r71 = r71 * r1;
    r71 = fmaf(r34, r69, r35 * r71);
    WriteSum2<float, float>((float *)inout_shared, r100, r71);
  };
  FlushSumShared<2, float>(out_focal_and_extra_njtr,
                           4 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = r36 * r36;
    r100 = r40 * r40;
    r0 = r9 * r9;
    r0 = r0 * r10;
    r0 = r0 * r26;
    r13 = r35 * r12;
    r13 = fmaf(r91, r13, r60 * r0);
    r0 = r9 * r9;
    r95 = r60 * r60;
    r0 = r0 * r10;
    r0 = r0 * r26;
    r97 = r35 * r12;
    r97 = r97 * r91;
    r97 = fmaf(r60, r97, r95 * r0);
    WriteSum4<float, float>((float *)inout_shared, r71, r100, r13, r97);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_diag,
                           0 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r97 = r8 * r8;
    r13 = 4.00000000000000000e+00;
    r52 = r50 * r52;
    r52 = 1.0 / r52;
    r97 = r97 * r9;
    r97 = r97 * r9;
    r97 = r97 * r13;
    r97 = r97 * r52;
    r52 = r65 * r65;
    r52 = fmaf(r26, r52, r12 * r97);
    r13 = r34 * r34;
    r13 = fmaf(r12, r13, r26 * r97);
    WriteSum2<float, float>((float *)inout_shared, r52, r13);
  };
  FlushSumShared<2, float>(out_focal_and_extra_precond_diag,
                           4 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = 0.00000000000000000e+00;
    r52 = r2 * r8;
    r52 = r52 * r51;
    r52 = r52 * r36;
    r52 = r52 * r57;
    r85 = r36 * r85;
    r97 = r2 * r9;
    r97 = r97 * r36;
    r97 = r97 * r63;
    WriteSum4<float, float>((float *)inout_shared, r13, r52, r85, r97);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           0 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r97 = r2 * r34;
    r97 = r97 * r36;
    r36 = r3 * r65;
    r36 = r36 * r40;
    r85 = r3 * r9;
    r85 = r85 * r51;
    r85 = r85 * r40;
    r85 = r85 * r57;
    r52 = r57 * r60;
    r13 = r3 * r9;
    r13 = r13 * r40;
    r52 = r52 * r13;
    WriteSum4<float, float>((float *)inout_shared, r97, r85, r52, r36);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           4 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r63 * r13;
    r36 = r9 * r9;
    r36 = r36 * r51;
    r36 = r36 * r10;
    r36 = r36 * r26;
    r52 = r51 * r35;
    r52 = r52 * r12;
    r52 = fmaf(r91, r52, r60 * r36);
    r36 = r14 * r8;
    r36 = r36 * r51;
    r36 = r36 * r12;
    r85 = r9 * r51;
    r85 = r85 * r65;
    r85 = r85 * r57;
    r85 = fmaf(r26, r85, r92 * r36);
    r36 = r14 * r9;
    r36 = r36 * r51;
    r36 = r36 * r26;
    r97 = r8 * r51;
    r97 = r97 * r34;
    r97 = r97 * r57;
    r97 = fmaf(r12, r97, r92 * r36);
    WriteSum4<float, float>((float *)inout_shared, r13, r52, r85, r97);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           8 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r97 = r14 * r12;
    r97 = r97 * r92;
    r85 = r9 * r65;
    r85 = r85 * r57;
    r85 = r85 * r26;
    r85 = fmaf(r60, r85, r91 * r97);
    r97 = r14 * r9;
    r97 = r97 * r26;
    r97 = r97 * r92;
    r92 = r34 * r12;
    r92 = fmaf(r73, r92, r60 * r97);
    r97 = r9 * r65;
    r97 = r97 * r26;
    r73 = r9 * r34;
    r73 = r73 * r12;
    r73 = fmaf(r63, r73, r63 * r97);
    WriteSum3<float, float>((float *)inout_shared, r85, r92, r73);
  };
  FlushSumShared<3, float>(out_focal_and_extra_precond_tril,
                           12 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r45 * r66;
    r92 = r54 * r8;
    r92 = r92 * r8;
    r92 = r92 * r67;
    r92 = fmaf(r93, r92, r35 * r73);
    r73 = r14 * r31;
    r73 = r73 * r9;
    r85 = r39 * r54;
    r85 = r85 * r9;
    r85 = fmaf(r78, r85, r10 * r73);
    r92 = r92 + r85;
    r92 = fmaf(r45, r57, r7 * r92);
    r73 = r6 * r54;
    r92 = fmaf(r88, r73, r92);
    r97 = r6 * r14;
    r97 = r97 * r45;
    r97 = r97 * r9;
    r92 = fmaf(r10, r97, r92);
    r60 = r25 * r54;
    r92 = fmaf(r35, r60, r92);
    r91 = r39 * r54;
    r91 = r91 * r8;
    r91 = r91 * r8;
    r91 = fmaf(r93, r91, r45 * r63);
    r85 = r85 + r91;
    r52 = r5 * r14;
    r52 = r52 * r51;
    r52 = fmaf(r85, r52, r4 * r85);
    r85 = r8 * r52;
    r92 = fmaf(r57, r85, r92);
    r13 = r25 * r54;
    r13 = r13 * r61;
    r92 = fmaf(r35, r13, r92);
    r92 = fmaf(r31, r64, r92);
    r92 = fmaf(r45, r62, r92);
    r13 = r2 * r92;
    r85 = r31 * r9;
    r85 = r85 * r66;
    r85 = fmaf(r54, r96, r10 * r85);
    r85 = r85 + r91;
    r85 = fmaf(r54, r101, r6 * r85);
    r91 = r7 * r31;
    r85 = fmaf(r63, r91, r85);
    r60 = r7 * r14;
    r60 = r60 * r45;
    r60 = r60 * r9;
    r85 = fmaf(r10, r60, r85);
    r97 = r25 * r54;
    r97 = r97 * r9;
    r85 = fmaf(r10, r97, r85);
    r73 = r9 * r52;
    r85 = fmaf(r57, r73, r85);
    r36 = r25 * r54;
    r36 = r36 * r9;
    r36 = r36 * r61;
    r85 = fmaf(r10, r36, r85);
    r85 = fmaf(r31, r57, r85);
    r85 = fmaf(r31, r62, r85);
    r36 = r3 * r85;
    r73 = r24 * r66;
    r97 = r43 * r8;
    r97 = r97 * r8;
    r97 = r97 * r67;
    r97 = fmaf(r93, r97, r35 * r73);
    r73 = r39 * r43;
    r73 = r73 * r9;
    r60 = r14 * r49;
    r60 = r60 * r9;
    r60 = fmaf(r10, r60, r78 * r73);
    r97 = r97 + r60;
    r73 = r6 * r43;
    r73 = fmaf(r88, r73, r7 * r97);
    r97 = r39 * r43;
    r97 = r97 * r8;
    r97 = r97 * r8;
    r97 = fmaf(r93, r97, r24 * r63);
    r60 = r60 + r97;
    r91 = r5 * r14;
    r91 = r91 * r51;
    r91 = fmaf(r60, r91, r4 * r60);
    r60 = r8 * r91;
    r73 = fmaf(r57, r60, r73);
    r40 = r6 * r14;
    r40 = r40 * r24;
    r40 = r40 * r9;
    r73 = fmaf(r10, r40, r73);
    r50 = r25 * r43;
    r50 = r50 * r61;
    r73 = fmaf(r35, r50, r73);
    r100 = r25 * r43;
    r73 = fmaf(r35, r100, r73);
    r73 = fmaf(r24, r57, r73);
    r73 = fmaf(r24, r62, r73);
    r73 = fmaf(r49, r64, r73);
    r100 = r2 * r73;
    r50 = r49 * r9;
    r50 = r50 * r66;
    r50 = fmaf(r10, r50, r43 * r96);
    r50 = r50 + r97;
    r50 = fmaf(r49, r57, r6 * r50);
    r97 = r9 * r91;
    r50 = fmaf(r57, r97, r50);
    r40 = r25 * r43;
    r40 = r40 * r9;
    r50 = fmaf(r10, r40, r50);
    r60 = r25 * r43;
    r60 = r60 * r9;
    r60 = r60 * r61;
    r50 = fmaf(r10, r60, r50);
    r71 = r7 * r14;
    r71 = r71 * r24;
    r71 = r71 * r9;
    r50 = fmaf(r10, r71, r50);
    r0 = r7 * r49;
    r50 = fmaf(r63, r0, r50);
    r50 = fmaf(r43, r101, r50);
    r50 = fmaf(r49, r62, r50);
    r0 = r3 * r50;
    WriteIdx4<1024, float, float, float4>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r13, r36,
        r100, r0);
    r0 = r59 * r8;
    r0 = r0 * r8;
    r0 = r0 * r67;
    r67 = r48 * r66;
    r67 = fmaf(r35, r67, r93 * r0);
    r0 = r14 * r47;
    r0 = r0 * r9;
    r100 = r39 * r59;
    r100 = r100 * r9;
    r100 = fmaf(r78, r100, r10 * r0);
    r67 = r67 + r100;
    r64 = fmaf(r47, r64, r7 * r67);
    r67 = r25 * r59;
    r64 = fmaf(r35, r67, r64);
    r0 = r6 * r59;
    r64 = fmaf(r88, r0, r64);
    r88 = r5 * r14;
    r78 = r39 * r59;
    r78 = r78 * r8;
    r78 = r78 * r8;
    r78 = fmaf(r48, r63, r93 * r78);
    r100 = r100 + r78;
    r88 = r88 * r51;
    r100 = fmaf(r4, r100, r100 * r88);
    r4 = r8 * r100;
    r64 = fmaf(r57, r4, r64);
    r88 = r6 * r14;
    r88 = r88 * r48;
    r88 = r88 * r9;
    r64 = fmaf(r10, r88, r64);
    r93 = r25 * r59;
    r93 = r93 * r61;
    r64 = fmaf(r35, r93, r64);
    r64 = fmaf(r48, r57, r64);
    r64 = fmaf(r48, r62, r64);
    r93 = r2 * r64;
    r88 = r47 * r9;
    r88 = r88 * r66;
    r96 = fmaf(r59, r96, r10 * r88);
    r96 = r96 + r78;
    r78 = r7 * r47;
    r78 = fmaf(r63, r78, r6 * r96);
    r96 = r25 * r59;
    r96 = r96 * r9;
    r78 = fmaf(r10, r96, r78);
    r63 = r25 * r59;
    r63 = r63 * r9;
    r63 = r63 * r61;
    r78 = fmaf(r10, r63, r78);
    r88 = r9 * r100;
    r78 = fmaf(r57, r88, r78);
    r4 = r7 * r14;
    r4 = r4 * r48;
    r4 = r4 * r9;
    r78 = fmaf(r10, r4, r78);
    r78 = fmaf(r47, r57, r78);
    r78 = fmaf(r59, r101, r78);
    r78 = fmaf(r47, r62, r78);
    r62 = r3 * r78;
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx, r93, r62);
    r62 = r3 * r25;
    r62 = r62 * r1;
    r62 = fmaf(r92, r69, r85 * r62);
    r93 = r3 * r25;
    r93 = r93 * r1;
    r93 = fmaf(r73, r69, r50 * r93);
    r4 = r3 * r25;
    r4 = r4 * r1;
    r69 = fmaf(r64, r69, r78 * r4);
    WriteSum3<float, float>((float *)inout_shared, r62, r93, r69);
  };
  FlushSumShared<3, float>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r85 * r85;
    r93 = r92 * r92;
    r93 = fmaf(r12, r93, r26 * r69);
    r69 = r73 * r73;
    r62 = r50 * r50;
    r62 = fmaf(r26, r62, r12 * r69);
    r69 = r64 * r64;
    r4 = r78 * r78;
    r4 = fmaf(r26, r4, r12 * r69);
    WriteSum3<float, float>((float *)inout_shared, r93, r62, r4);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = r85 * r50;
    r62 = r92 * r73;
    r62 = fmaf(r12, r62, r26 * r4);
    r4 = r85 * r78;
    r93 = r92 * r64;
    r93 = fmaf(r12, r93, r26 * r4);
    r4 = r73 * r64;
    r69 = r50 * r78;
    r69 = fmaf(r26, r69, r12 * r4);
    WriteSum3<float, float>((float *)inout_shared, r62, r93, r69);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc, (float *)inout_shared);
}

void OpencvSplitFixedPrincipalPointResJac(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *focal_and_extra, unsigned int focal_and_extra_num_alloc,
    SharedIndex *focal_and_extra_indices, float *point,
    unsigned int point_num_alloc, SharedIndex *point_indices, float *pixel,
    unsigned int pixel_num_alloc, float *principal_point,
    unsigned int principal_point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float *out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    float *const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    float *const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    float *const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
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
  OpencvSplitFixedPrincipalPointResJacKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, focal_and_extra, focal_and_extra_num_alloc,
      focal_and_extra_indices, point, point_num_alloc, point_indices, pixel,
      pixel_num_alloc, principal_point, principal_point_num_alloc, out_res,
      out_res_num_alloc, out_pose_jac, out_pose_jac_num_alloc, out_pose_njtr,
      out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc, out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc, out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc, out_point_jac,
      out_point_jac_num_alloc, out_point_njtr, out_point_njtr_num_alloc,
      out_point_precond_diag, out_point_precond_diag_num_alloc,
      out_point_precond_tril, out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar