#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_split_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedPrincipalPointFixedPointResJacFirstKernel(
        float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        float *focal_and_extra, unsigned int focal_and_extra_num_alloc,
        SharedIndex *focal_and_extra_indices, float *pixel,
        unsigned int pixel_num_alloc, float *principal_point,
        unsigned int principal_point_num_alloc, float *point,
        unsigned int point_num_alloc, float *out_res,
        unsigned int out_res_num_alloc, float *const out_rTr,
        float *out_pose_jac, unsigned int out_pose_jac_num_alloc,
        float *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
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

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102;

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
    ReadIdx3<1024, float, float, float4>(point, 0 * point_num_alloc,
                                         global_thread_idx, r11, r12, r13);
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
    r31 = fmaf(r11, r31, r9);
  };
  LoadShared<3, float, float>(pose, 4 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r9, r33, r34);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r35 = r19 * r20;
    r35 = r35 * r14;
    r36 = r21 * r22;
    r36 = fmaf(r14, r36, r35);
    r37 = r19 * r19;
    r38 = -2.00000000000000000e+00;
    r37 = r37 * r38;
    r39 = 1.00000000000000000e+00;
    r40 = r21 * r21;
    r40 = fmaf(r38, r40, r39);
    r41 = r37 + r40;
    r42 = r20 * r21;
    r42 = r42 * r14;
    r43 = r19 * r22;
    r43 = fmaf(r38, r43, r42);
    r44 = r14 * r30;
    r44 = r44 * r27;
    r45 = r38 * r32;
    r46 = fmaf(r23, r45, r44);
    r47 = r30 * r30;
    r47 = r47 * r38;
    r48 = r39 + r47;
    r49 = r23 * r23;
    r49 = r49 * r38;
    r48 = r48 + r49;
    r31 = fmaf(r9, r36, r31);
    r31 = fmaf(r33, r41, r31);
    r31 = fmaf(r34, r43, r31);
    r31 = fmaf(r13, r46, r31);
    r31 = fmaf(r12, r48, r31);
    r48 = r31 * r31;
    r46 = 9.99999999999999955e-07;
    r50 = r14 * r30;
    r50 = r50 * r23;
    r51 = fmaf(r27, r45, r50);
    r51 = fmaf(r11, r51, r10);
    r10 = r19 * r21;
    r10 = r10 * r14;
    r52 = r20 * r22;
    r52 = fmaf(r38, r52, r10);
    r53 = r20 * r20;
    r53 = r53 * r38;
    r54 = r39 + r53;
    r54 = r54 + r37;
    r37 = r19 * r22;
    r37 = fmaf(r14, r37, r42);
    r42 = r14 * r23;
    r42 = fmaf(r32, r42, r44);
    r44 = r38 * r27;
    r44 = r44 * r27;
    r55 = r39 + r44;
    r55 = r55 + r49;
    r51 = fmaf(r9, r52, r51);
    r51 = fmaf(r34, r54, r51);
    r51 = fmaf(r33, r37, r51);
    r51 = fmaf(r12, r42, r51);
    r51 = fmaf(r13, r55, r51);
    r55 = copysign(1.0, r51);
    r55 = fmaf(r46, r55, r51);
    r46 = r55 * r55;
    r51 = 1.0 / r46;
    r48 = r48 * r51;
    r44 = r39 + r44;
    r44 = r44 + r47;
    r44 = fmaf(r11, r44, r8);
    r24 = fmaf(r30, r45, r24);
    r8 = r14 * r27;
    r8 = fmaf(r32, r8, r50);
    r50 = r20 * r22;
    r50 = fmaf(r14, r50, r10);
    r10 = r21 * r22;
    r10 = fmaf(r38, r10, r35);
    r40 = r53 + r40;
    r44 = fmaf(r12, r24, r44);
    r44 = fmaf(r13, r8, r44);
    r44 = fmaf(r34, r50, r44);
    r44 = fmaf(r33, r10, r44);
    r44 = fmaf(r9, r40, r44);
    r9 = 3.00000000000000000e+00;
    r33 = r44 * r9;
    r34 = r44 * r51;
    r33 = fmaf(r34, r33, r48);
    r8 = 1.0 / r55;
    r24 = fmaf(r44, r8, r7 * r33);
    r53 = r44 * r34;
    r48 = r53 + r48;
    r35 = r48 * r48;
    r47 = fmaf(r5, r35, r4 * r48);
    r39 = r47 * r8;
    r42 = r6 * r34;
    r49 = r14 * r31;
    r24 = fmaf(r49, r42, r24);
    r24 = fmaf(r44, r39, r24);
    r42 = r2 * r24;
    r0 = r0 + r42;
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r56, r57);
    r0 = fmaf(r56, r25, r0);
    r56 = r31 * r31;
    r56 = r56 * r9;
    r56 = fmaf(r51, r56, r53);
    r53 = fmaf(r31, r8, r6 * r56);
    r58 = r7 * r34;
    r53 = fmaf(r49, r58, r53);
    r53 = fmaf(r31, r39, r53);
    r58 = r3 * r53;
    r1 = r1 + r58;
    r1 = fmaf(r57, r25, r1);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r0, r1);
    r57 = fmaf(r0, r0, r1 * r1);
  };
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r57);
  if (global_thread_idx < problem_size) {
    r57 = 6.00000000000000000e+00;
    r59 = r14 * r32;
    r60 = r16 * r19;
    r61 = 5.00000000000000000e-01;
    r62 = r15 * r20;
    r63 = -5.00000000000000000e-01;
    r62 = fmaf(r63, r62, r61 * r60);
    r60 = r18 * r21;
    r62 = fmaf(r61, r60, r62);
    r64 = r22 * r61;
    r62 = fmaf(r17, r64, r62);
    r60 = r15 * r22;
    r65 = r18 * r19;
    r65 = fmaf(r63, r65, r63 * r60);
    r60 = r17 * r20;
    r65 = fmaf(r63, r60, r65);
    r66 = r16 * r21;
    r65 = fmaf(r61, r66, r65);
    r66 = r27 * r65;
    r59 = fmaf(r14, r66, r62 * r59);
    r60 = r14 * r23;
    r67 = r16 * r63;
    r68 = fmaf(r61, r29, r22 * r67);
    r68 = fmaf(r63, r26, r68);
    r68 = fmaf(r63, r28, r68);
    r69 = r14 * r30;
    r70 = r15 * r19;
    r71 = r17 * r21;
    r71 = fmaf(r63, r71, r63 * r70);
    r71 = fmaf(r18, r64, r71);
    r71 = fmaf(r20, r67, r71);
    r69 = r69 * r71;
    r60 = fmaf(r68, r60, r69);
    r59 = r59 + r60;
    r70 = r14 * r27;
    r70 = r70 * r71;
    r72 = r14 * r23;
    r72 = r72 * r62;
    r73 = r70 + r72;
    r74 = r30 * r38;
    r73 = fmaf(r65, r74, r73);
    r73 = fmaf(r68, r45, r73);
    r73 = fmaf(r12, r73, r13 * r59);
    r59 = r27 * r62;
    r74 = -4.00000000000000000e+00;
    r59 = r59 * r74;
    r75 = r30 * r68;
    r76 = r74 * r75;
    r77 = r59 + r76;
    r73 = fmaf(r11, r77, r73);
    r77 = r57 * r73;
    r78 = r44 * r44;
    r79 = r14 * r27;
    r79 = r79 * r68;
    r80 = r14 * r30;
    r80 = fmaf(r62, r80, r79);
    r81 = r14 * r23;
    r81 = r81 * r65;
    r82 = r14 * r32;
    r82 = r82 * r71;
    r83 = r81 + r82;
    r84 = r80 + r83;
    r62 = fmaf(r62, r45, r38 * r66);
    r62 = r62 + r60;
    r62 = fmaf(r11, r62, r12 * r84);
    r84 = r23 * r71;
    r84 = r84 * r74;
    r59 = r84 + r59;
    r62 = fmaf(r13, r59, r62);
    r59 = -6.00000000000000000e+00;
    r85 = r62 * r59;
    r46 = r55 * r46;
    r86 = 1.0 / r46;
    r85 = r85 * r86;
    r78 = fmaf(r85, r78, r34 * r77);
    r77 = r23 * r38;
    r87 = r71 * r45;
    r77 = fmaf(r65, r77, r87);
    r77 = r77 + r80;
    r76 = r84 + r76;
    r76 = fmaf(r12, r76, r13 * r77);
    r77 = r14 * r32;
    r77 = fmaf(r68, r77, r72);
    r72 = r14 * r30;
    r72 = fmaf(r65, r72, r70);
    r77 = r77 + r72;
    r76 = fmaf(r11, r77, r76);
    r77 = r51 * r49;
    r70 = r38 * r31;
    r70 = r70 * r31;
    r70 = r70 * r62;
    r70 = fmaf(r86, r70, r76 * r77);
    r78 = r78 + r70;
    r84 = r25 * r34;
    r80 = r47 * r84;
    r78 = fmaf(r62, r80, r7 * r78);
    r88 = r6 * r14;
    r88 = r88 * r76;
    r78 = fmaf(r34, r88, r78);
    r89 = r44 * r31;
    r89 = r89 * r74;
    r89 = r89 * r86;
    r90 = r6 * r89;
    r91 = r6 * r73;
    r78 = fmaf(r77, r91, r78);
    r92 = r14 * r73;
    r93 = r38 * r44;
    r93 = r93 * r44;
    r93 = r93 * r62;
    r93 = fmaf(r86, r93, r34 * r92);
    r70 = r70 + r93;
    r92 = r5 * r14;
    r92 = r92 * r48;
    r92 = fmaf(r70, r92, r4 * r70);
    r70 = r44 * r92;
    r78 = fmaf(r8, r70, r78);
    r78 = fmaf(r73, r39, r78);
    r78 = fmaf(r62, r90, r78);
    r78 = fmaf(r62, r84, r78);
    r78 = fmaf(r73, r8, r78);
    r70 = r2 * r78;
    r91 = r31 * r76;
    r91 = r91 * r57;
    r88 = r31 * r31;
    r88 = fmaf(r85, r88, r51 * r91);
    r88 = r88 + r93;
    r88 = fmaf(r76, r8, r6 * r88);
    r93 = r25 * r31;
    r93 = r93 * r62;
    r88 = fmaf(r51, r93, r88);
    r91 = r7 * r14;
    r91 = r91 * r76;
    r88 = fmaf(r34, r91, r88);
    r85 = r31 * r92;
    r88 = fmaf(r8, r85, r88);
    r94 = r7 * r62;
    r88 = fmaf(r89, r94, r88);
    r95 = r7 * r77;
    r96 = r25 * r31;
    r96 = r96 * r47;
    r96 = r96 * r62;
    r88 = fmaf(r51, r96, r88);
    r88 = fmaf(r76, r39, r88);
    r88 = fmaf(r73, r95, r88);
    r96 = r3 * r88;
    r82 = r79 + r82;
    r79 = r14 * r30;
    r94 = r17 * r22;
    r85 = r15 * r20;
    r85 = fmaf(r61, r85, r63 * r94);
    r94 = r18 * r21;
    r85 = fmaf(r63, r94, r85);
    r85 = fmaf(r19, r67, r85);
    r79 = r79 * r85;
    r94 = r14 * r23;
    r91 = r18 * r19;
    r93 = r17 * r20;
    r93 = fmaf(r61, r93, r61 * r91);
    r93 = fmaf(r15, r64, r93);
    r93 = fmaf(r21, r67, r93);
    r94 = fmaf(r93, r94, r79);
    r82 = r82 + r94;
    r67 = r27 * r71;
    r67 = r67 * r74;
    r91 = r30 * r74;
    r91 = r91 * r93;
    r97 = r67 + r91;
    r97 = fmaf(r11, r97, r13 * r82);
    r82 = fmaf(r93, r45, r38 * r75);
    r98 = r14 * r23;
    r98 = r98 * r71;
    r99 = r14 * r27;
    r99 = fmaf(r85, r99, r98);
    r82 = r82 + r99;
    r97 = fmaf(r12, r82, r97);
    r82 = r57 * r97;
    r100 = r44 * r44;
    r101 = r38 * r27;
    r101 = fmaf(r68, r101, r87);
    r101 = r101 + r94;
    r94 = r14 * r27;
    r94 = r94 * r93;
    r102 = r14 * r32;
    r102 = fmaf(r85, r102, r94);
    r102 = r102 + r60;
    r102 = fmaf(r12, r102, r11 * r101);
    r101 = r23 * r85;
    r60 = r74 * r101;
    r67 = r67 + r60;
    r102 = fmaf(r13, r67, r102);
    r100 = r100 * r59;
    r100 = r100 * r102;
    r100 = fmaf(r86, r100, r34 * r82);
    r82 = r38 * r31;
    r82 = r82 * r31;
    r82 = r82 * r102;
    r94 = r69 + r94;
    r69 = r23 * r38;
    r94 = fmaf(r68, r69, r94);
    r94 = fmaf(r85, r45, r94);
    r69 = r14 * r32;
    r75 = fmaf(r14, r75, r93 * r69);
    r75 = r75 + r99;
    r75 = fmaf(r11, r75, r13 * r94);
    r60 = r91 + r60;
    r75 = fmaf(r12, r60, r75);
    r82 = fmaf(r75, r77, r86 * r82);
    r100 = r100 + r82;
    r100 = fmaf(r97, r39, r7 * r100);
    r60 = r6 * r97;
    r100 = fmaf(r77, r60, r100);
    r91 = r6 * r14;
    r91 = r91 * r75;
    r100 = fmaf(r34, r91, r100);
    r94 = r5 * r14;
    r69 = r14 * r97;
    r93 = r38 * r44;
    r93 = r93 * r44;
    r93 = r93 * r102;
    r93 = fmaf(r86, r93, r34 * r69);
    r82 = r82 + r93;
    r94 = r94 * r48;
    r82 = fmaf(r4, r82, r82 * r94);
    r94 = r44 * r82;
    r100 = fmaf(r8, r94, r100);
    r100 = fmaf(r102, r84, r100);
    r100 = fmaf(r97, r8, r100);
    r100 = fmaf(r102, r90, r100);
    r100 = fmaf(r102, r80, r100);
    r94 = r2 * r100;
    r91 = r31 * r31;
    r91 = r91 * r59;
    r91 = r91 * r102;
    r60 = r31 * r57;
    r60 = r60 * r75;
    r60 = fmaf(r51, r60, r86 * r91);
    r60 = r60 + r93;
    r93 = r25 * r31;
    r93 = r93 * r47;
    r93 = r93 * r102;
    r93 = fmaf(r51, r93, r6 * r60);
    r60 = r31 * r82;
    r93 = fmaf(r8, r60, r93);
    r91 = r25 * r31;
    r91 = r91 * r102;
    r93 = fmaf(r51, r91, r93);
    r69 = r7 * r102;
    r93 = fmaf(r89, r69, r93);
    r68 = r7 * r14;
    r68 = r68 * r75;
    r93 = fmaf(r34, r68, r93);
    r93 = fmaf(r75, r8, r93);
    r93 = fmaf(r97, r95, r93);
    r93 = fmaf(r75, r39, r93);
    r75 = r3 * r93;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r70, r96,
        r94, r75);
    r75 = r44 * r44;
    r94 = r23 * r74;
    r29 = fmaf(r63, r29, r16 * r64);
    r29 = fmaf(r61, r26, r29);
    r29 = fmaf(r61, r28, r29);
    r94 = r94 * r29;
    r66 = r74 * r66;
    r28 = r94 + r66;
    r61 = r14 * r30;
    r61 = r61 * r29;
    r98 = r98 + r61;
    r26 = r38 * r27;
    r98 = fmaf(r85, r26, r98);
    r98 = fmaf(r65, r45, r98);
    r98 = fmaf(r11, r98, r13 * r28);
    r28 = r14 * r32;
    r28 = fmaf(r14, r101, r29 * r28);
    r28 = r28 + r72;
    r98 = fmaf(r12, r28, r98);
    r75 = r75 * r59;
    r75 = r75 * r98;
    r28 = r14 * r27;
    r28 = r28 * r29;
    r81 = r81 + r28;
    r26 = r30 * r38;
    r81 = fmaf(r85, r26, r81);
    r81 = r81 + r87;
    r71 = r30 * r71;
    r71 = r71 * r74;
    r66 = r71 + r66;
    r66 = fmaf(r11, r66, r12 * r81);
    r81 = r14 * r32;
    r81 = fmaf(r65, r81, r61);
    r81 = r81 + r99;
    r66 = fmaf(r13, r81, r66);
    r81 = r57 * r66;
    r81 = fmaf(r34, r81, r86 * r75);
    r75 = r38 * r31;
    r75 = r75 * r31;
    r75 = r75 * r98;
    r28 = r79 + r28;
    r28 = r28 + r83;
    r45 = fmaf(r29, r45, r38 * r101);
    r45 = r45 + r72;
    r45 = fmaf(r13, r45, r11 * r28);
    r94 = r71 + r94;
    r45 = fmaf(r12, r94, r45);
    r75 = fmaf(r45, r77, r86 * r75);
    r81 = r81 + r75;
    r81 = fmaf(r66, r8, r7 * r81);
    r94 = r6 * r66;
    r81 = fmaf(r77, r94, r81);
    r12 = r6 * r14;
    r12 = r12 * r45;
    r81 = fmaf(r34, r12, r81);
    r71 = r5 * r14;
    r13 = r38 * r44;
    r13 = r13 * r44;
    r13 = r13 * r98;
    r28 = r14 * r66;
    r28 = fmaf(r34, r28, r86 * r13);
    r75 = r75 + r28;
    r71 = r71 * r48;
    r75 = fmaf(r4, r75, r75 * r71);
    r71 = r44 * r75;
    r81 = fmaf(r8, r71, r81);
    r81 = fmaf(r98, r80, r81);
    r81 = fmaf(r98, r84, r81);
    r81 = fmaf(r98, r90, r81);
    r81 = fmaf(r66, r39, r81);
    r71 = r2 * r81;
    r12 = r31 * r31;
    r12 = r12 * r59;
    r12 = r12 * r98;
    r94 = r31 * r57;
    r94 = r94 * r45;
    r94 = fmaf(r51, r94, r86 * r12);
    r94 = r94 + r28;
    r94 = fmaf(r45, r8, r6 * r94);
    r28 = r7 * r98;
    r94 = fmaf(r89, r28, r94);
    r12 = r31 * r75;
    r94 = fmaf(r8, r12, r94);
    r13 = r7 * r14;
    r13 = r13 * r45;
    r94 = fmaf(r34, r13, r94);
    r11 = r25 * r31;
    r11 = r11 * r98;
    r94 = fmaf(r51, r11, r94);
    r72 = r25 * r31;
    r72 = r72 * r47;
    r72 = r72 * r98;
    r94 = fmaf(r51, r72, r94);
    r94 = fmaf(r66, r95, r94);
    r94 = fmaf(r45, r39, r94);
    r72 = r3 * r94;
    r11 = r52 * r44;
    r11 = r11 * r44;
    r11 = r11 * r59;
    r45 = r40 * r57;
    r45 = fmaf(r34, r45, r86 * r11);
    r11 = r38 * r52;
    r11 = r11 * r31;
    r11 = r11 * r31;
    r11 = fmaf(r36, r77, r86 * r11);
    r45 = r45 + r11;
    r45 = fmaf(r40, r8, r7 * r45);
    r13 = r6 * r40;
    r45 = fmaf(r77, r13, r45);
    r12 = r38 * r52;
    r12 = r12 * r44;
    r12 = r12 * r44;
    r28 = r14 * r40;
    r28 = fmaf(r34, r28, r86 * r12);
    r11 = r11 + r28;
    r12 = r5 * r14;
    r12 = r12 * r48;
    r12 = fmaf(r11, r12, r4 * r11);
    r11 = r44 * r12;
    r45 = fmaf(r8, r11, r45);
    r29 = r6 * r14;
    r29 = r29 * r36;
    r45 = fmaf(r34, r29, r45);
    r45 = fmaf(r52, r90, r45);
    r45 = fmaf(r40, r39, r45);
    r45 = fmaf(r52, r84, r45);
    r45 = fmaf(r52, r80, r45);
    r29 = r2 * r45;
    r11 = r52 * r31;
    r11 = r11 * r31;
    r11 = r11 * r59;
    r13 = r36 * r31;
    r13 = r13 * r57;
    r13 = fmaf(r51, r13, r86 * r11);
    r13 = r13 + r28;
    r13 = fmaf(r36, r8, r6 * r13);
    r28 = r25 * r52;
    r28 = r28 * r31;
    r28 = r28 * r47;
    r13 = fmaf(r51, r28, r13);
    r11 = r7 * r52;
    r13 = fmaf(r89, r11, r13);
    r101 = r31 * r12;
    r13 = fmaf(r8, r101, r13);
    r83 = r25 * r52;
    r83 = r83 * r31;
    r13 = fmaf(r51, r83, r13);
    r79 = r7 * r14;
    r79 = r79 * r36;
    r13 = fmaf(r34, r79, r13);
    r13 = fmaf(r40, r95, r13);
    r13 = fmaf(r36, r39, r13);
    r79 = r3 * r13;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r71, r72,
        r29, r79);
    r79 = r37 * r44;
    r79 = r79 * r44;
    r79 = r79 * r59;
    r29 = r10 * r57;
    r29 = fmaf(r34, r29, r86 * r79);
    r79 = r38 * r37;
    r79 = r79 * r31;
    r79 = r79 * r31;
    r79 = fmaf(r41, r77, r86 * r79);
    r29 = r29 + r79;
    r29 = fmaf(r37, r80, r7 * r29);
    r72 = r6 * r10;
    r29 = fmaf(r77, r72, r29);
    r71 = r6 * r14;
    r71 = r71 * r41;
    r29 = fmaf(r34, r71, r29);
    r83 = r5 * r14;
    r101 = r38 * r37;
    r101 = r101 * r44;
    r101 = r101 * r44;
    r11 = r14 * r10;
    r11 = fmaf(r34, r11, r86 * r101);
    r79 = r79 + r11;
    r83 = r83 * r48;
    r79 = fmaf(r4, r79, r79 * r83);
    r83 = r44 * r79;
    r29 = fmaf(r8, r83, r29);
    r29 = fmaf(r10, r8, r29);
    r29 = fmaf(r10, r39, r29);
    r29 = fmaf(r37, r90, r29);
    r29 = fmaf(r37, r84, r29);
    r83 = r2 * r29;
    r71 = r37 * r31;
    r71 = r71 * r31;
    r71 = r71 * r59;
    r72 = r41 * r31;
    r72 = r72 * r57;
    r72 = fmaf(r51, r72, r86 * r71);
    r72 = r72 + r11;
    r72 = fmaf(r41, r8, r6 * r72);
    r11 = r7 * r14;
    r11 = r11 * r41;
    r72 = fmaf(r34, r11, r72);
    r71 = r7 * r37;
    r72 = fmaf(r89, r71, r72);
    r101 = r25 * r37;
    r101 = r101 * r31;
    r72 = fmaf(r51, r101, r72);
    r28 = r31 * r79;
    r72 = fmaf(r8, r28, r72);
    r99 = r25 * r37;
    r99 = r99 * r31;
    r99 = r99 * r47;
    r72 = fmaf(r51, r99, r72);
    r72 = fmaf(r10, r95, r72);
    r72 = fmaf(r41, r39, r72);
    r99 = r3 * r72;
    r28 = r50 * r57;
    r101 = r54 * r44;
    r101 = r101 * r44;
    r101 = r101 * r59;
    r101 = fmaf(r86, r101, r34 * r28);
    r28 = r38 * r54;
    r28 = r28 * r31;
    r28 = r28 * r31;
    r28 = fmaf(r86, r28, r43 * r77);
    r101 = r101 + r28;
    r90 = fmaf(r54, r90, r7 * r101);
    r101 = r6 * r14;
    r101 = r101 * r43;
    r90 = fmaf(r34, r101, r90);
    r71 = r14 * r50;
    r11 = r38 * r54;
    r11 = r11 * r44;
    r11 = r11 * r44;
    r11 = fmaf(r86, r11, r34 * r71);
    r28 = r28 + r11;
    r71 = r5 * r14;
    r71 = r71 * r48;
    r71 = fmaf(r28, r71, r4 * r28);
    r28 = r44 * r71;
    r90 = fmaf(r8, r28, r90);
    r4 = r6 * r50;
    r90 = fmaf(r77, r4, r90);
    r90 = fmaf(r50, r8, r90);
    r90 = fmaf(r50, r39, r90);
    r90 = fmaf(r54, r80, r90);
    r90 = fmaf(r54, r84, r90);
    r4 = r2 * r90;
    r84 = r43 * r31;
    r84 = r84 * r57;
    r80 = r54 * r31;
    r80 = r80 * r31;
    r80 = r80 * r59;
    r80 = fmaf(r86, r80, r51 * r84);
    r80 = r80 + r11;
    r11 = r7 * r54;
    r11 = fmaf(r89, r11, r6 * r80);
    r80 = r25 * r54;
    r80 = r80 * r31;
    r80 = r80 * r47;
    r11 = fmaf(r51, r80, r11);
    r47 = r7 * r14;
    r47 = r47 * r43;
    r11 = fmaf(r34, r47, r11);
    r89 = r25 * r54;
    r89 = r89 * r31;
    r11 = fmaf(r51, r89, r11);
    r84 = r31 * r71;
    r11 = fmaf(r8, r84, r11);
    r11 = fmaf(r43, r8, r11);
    r11 = fmaf(r43, r39, r11);
    r11 = fmaf(r50, r95, r11);
    r95 = r3 * r11;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx, r83, r99, r4, r95);
    r95 = r2 * r25;
    r95 = r95 * r0;
    r4 = r3 * r25;
    r4 = r4 * r1;
    r4 = fmaf(r88, r4, r78 * r95);
    r95 = r3 * r25;
    r95 = r95 * r1;
    r99 = r2 * r25;
    r99 = r99 * r0;
    r99 = fmaf(r100, r99, r93 * r95);
    r95 = r3 * r25;
    r95 = r95 * r1;
    r83 = r2 * r25;
    r83 = r83 * r0;
    r83 = fmaf(r81, r83, r94 * r95);
    r95 = r3 * r25;
    r95 = r95 * r1;
    r84 = r2 * r25;
    r84 = r84 * r0;
    r84 = fmaf(r45, r84, r13 * r95);
    WriteSum4<float, float>((float *)inout_shared, r4, r99, r83, r84);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r84 = r2 * r25;
    r84 = r84 * r0;
    r83 = r3 * r25;
    r83 = r83 * r1;
    r83 = fmaf(r72, r83, r29 * r84);
    r84 = r3 * r25;
    r84 = r84 * r1;
    r99 = r2 * r25;
    r99 = r99 * r0;
    r99 = fmaf(r90, r99, r11 * r84);
    WriteSum2<float, float>((float *)inout_shared, r83, r99);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r99 = r3 * r3;
    r83 = r88 * r99;
    r84 = r2 * r2;
    r4 = r78 * r84;
    r78 = fmaf(r78, r4, r88 * r83);
    r88 = r100 * r100;
    r95 = r93 * r93;
    r95 = fmaf(r99, r95, r84 * r88);
    r88 = r94 * r94;
    r89 = r81 * r81;
    r89 = fmaf(r84, r89, r99 * r88);
    r88 = r45 * r45;
    r47 = r13 * r13;
    r47 = fmaf(r99, r47, r84 * r88);
    WriteSum4<float, float>((float *)inout_shared, r78, r95, r89, r47);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = r72 * r72;
    r89 = r29 * r29;
    r89 = fmaf(r84, r89, r99 * r47);
    r47 = r11 * r11;
    r95 = r90 * r90;
    r95 = fmaf(r84, r95, r99 * r47);
    WriteSum2<float, float>((float *)inout_shared, r89, r95);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = fmaf(r100, r4, r93 * r83);
    r89 = fmaf(r94, r83, r81 * r4);
    r47 = fmaf(r13, r83, r45 * r4);
    r78 = fmaf(r29, r4, r72 * r83);
    WriteSum4<float, float>((float *)inout_shared, r95, r89, r47, r78);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = fmaf(r90, r4, r11 * r83);
    r83 = r100 * r81;
    r78 = r93 * r94;
    r78 = fmaf(r99, r78, r84 * r83);
    r83 = r93 * r13;
    r47 = r100 * r45;
    r47 = fmaf(r84, r47, r99 * r83);
    r83 = r93 * r72;
    r89 = r100 * r29;
    r89 = fmaf(r84, r89, r99 * r83);
    WriteSum4<float, float>((float *)inout_shared, r4, r78, r47, r89);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r89 = r100 * r90;
    r47 = r93 * r11;
    r47 = fmaf(r99, r47, r84 * r89);
    r89 = r81 * r45;
    r78 = r94 * r13;
    r78 = fmaf(r99, r78, r84 * r89);
    r89 = r81 * r29;
    r4 = r94 * r72;
    r4 = fmaf(r99, r4, r84 * r89);
    r89 = r81 * r90;
    r83 = r94 * r11;
    r83 = fmaf(r99, r83, r84 * r89);
    WriteSum4<float, float>((float *)inout_shared, r47, r78, r4, r83);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = r13 * r72;
    r4 = r45 * r29;
    r4 = fmaf(r84, r4, r99 * r83);
    r83 = r13 * r11;
    r78 = r45 * r90;
    r78 = fmaf(r84, r78, r99 * r83);
    r83 = r29 * r90;
    r47 = r72 * r11;
    r47 = fmaf(r99, r47, r84 * r83);
    WriteSum3<float, float>((float *)inout_shared, r4, r78, r47);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = r2 * r44;
    r47 = r47 * r48;
    r47 = r47 * r8;
    r78 = r3 * r31;
    r78 = r78 * r48;
    r78 = r78 * r8;
    WriteIdx4<1024, float, float, float4>(
        out_focal_and_extra_jac, 0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx, r24, r53, r47, r78);
    r78 = r3 * r56;
    r47 = r2 * r44;
    r47 = r47 * r8;
    r47 = r47 * r35;
    r4 = r3 * r31;
    r4 = r4 * r8;
    r4 = r4 * r35;
    r83 = r2 * r34;
    r83 = r83 * r49;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          4 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx, r47, r4, r83, r78);
    r78 = r2 * r33;
    r83 = r3 * r34;
    r83 = r83 * r49;
    WriteIdx2<1024, float, float, float2>(out_focal_and_extra_jac,
                                          8 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx, r78, r83);
    r83 = r25 * r24;
    r83 = r83 * r0;
    r78 = r25 * r53;
    r78 = r78 * r1;
    r4 = r3 * r25;
    r4 = r4 * r31;
    r4 = r4 * r48;
    r4 = r4 * r1;
    r47 = r2 * r25;
    r47 = r47 * r44;
    r47 = r47 * r48;
    r47 = r47 * r0;
    r47 = fmaf(r8, r47, r8 * r4);
    r4 = r3 * r25;
    r4 = r4 * r31;
    r4 = r4 * r1;
    r4 = r4 * r8;
    r89 = r2 * r25;
    r89 = r89 * r44;
    r89 = r89 * r0;
    r89 = r89 * r8;
    r89 = fmaf(r35, r89, r35 * r4);
    WriteSum4<float, float>((float *)inout_shared, r83, r78, r47, r89);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           0 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r89 = r3 * r25;
    r89 = r89 * r56;
    r47 = r2 * r38;
    r47 = r47 * r31;
    r47 = r47 * r0;
    r47 = fmaf(r34, r47, r1 * r89);
    r89 = r2 * r25;
    r89 = r89 * r33;
    r78 = r3 * r38;
    r78 = r78 * r31;
    r78 = r78 * r1;
    r78 = fmaf(r34, r78, r0 * r89);
    WriteSum2<float, float>((float *)inout_shared, r47, r78);
  };
  FlushSumShared<2, float>(out_focal_and_extra_njtr,
                           4 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = r24 * r24;
    r47 = r53 * r53;
    r89 = r31 * r31;
    r89 = r89 * r51;
    r89 = r89 * r99;
    r0 = r44 * r34;
    r0 = r0 * r84;
    r0 = fmaf(r35, r0, r35 * r89);
    r89 = r35 * r35;
    r1 = r31 * r31;
    r1 = r1 * r51;
    r1 = r1 * r99;
    r51 = r44 * r34;
    r51 = r51 * r84;
    r89 = fmaf(r89, r51, r89 * r1);
    WriteSum4<float, float>((float *)inout_shared, r78, r47, r0, r89);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_diag,
                           0 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r89 = r44 * r44;
    r0 = 4.00000000000000000e+00;
    r46 = r55 * r46;
    r46 = 1.0 / r46;
    r89 = r89 * r31;
    r89 = r89 * r31;
    r89 = r89 * r0;
    r89 = r89 * r46;
    r46 = r56 * r56;
    r46 = fmaf(r99, r46, r84 * r89);
    r0 = r33 * r33;
    r0 = fmaf(r84, r0, r99 * r89);
    WriteSum2<float, float>((float *)inout_shared, r46, r0);
  };
  FlushSumShared<2, float>(out_focal_and_extra_precond_diag,
                           4 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
    r46 = r2 * r44;
    r46 = r46 * r48;
    r46 = r46 * r24;
    r46 = r46 * r8;
    r89 = r44 * r42;
    r55 = r8 * r35;
    r89 = r89 * r55;
    r47 = r34 * r49;
    r42 = r42 * r47;
    WriteSum4<float, float>((float *)inout_shared, r0, r46, r89, r42);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           0 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r2 * r33;
    r42 = r42 * r24;
    r24 = r3 * r56;
    r24 = r24 * r53;
    r89 = r3 * r31;
    r89 = r89 * r48;
    r89 = r89 * r53;
    r89 = r89 * r8;
    r53 = r31 * r58;
    r53 = r53 * r55;
    WriteSum4<float, float>((float *)inout_shared, r42, r89, r53, r24);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           4 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = r58 * r47;
    r58 = r48 * r35;
    r51 = fmaf(r58, r51, r58 * r1);
    r58 = r44 * r44;
    r58 = r58 * r48;
    r58 = r58 * r86;
    r58 = r58 * r84;
    r1 = r31 * r48;
    r1 = r1 * r56;
    r1 = r1 * r8;
    r1 = fmaf(r99, r1, r49 * r58);
    r58 = r44 * r31;
    r58 = r58 * r48;
    r58 = r58 * r86;
    r58 = r58 * r99;
    r24 = r44 * r48;
    r24 = r24 * r33;
    r24 = r24 * r8;
    r24 = fmaf(r84, r24, r49 * r58);
    WriteSum4<float, float>((float *)inout_shared, r47, r51, r1, r24);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           8 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r44 * r44;
    r24 = r24 * r86;
    r24 = r24 * r84;
    r24 = r24 * r49;
    r1 = r31 * r56;
    r1 = r1 * r8;
    r1 = r1 * r99;
    r1 = fmaf(r35, r1, r35 * r24);
    r24 = r44 * r31;
    r24 = r24 * r86;
    r24 = r24 * r99;
    r24 = r24 * r49;
    r86 = r44 * r33;
    r86 = r86 * r8;
    r86 = r86 * r84;
    r86 = fmaf(r35, r86, r35 * r24);
    r24 = r56 * r34;
    r24 = r24 * r99;
    r99 = r33 * r34;
    r99 = r99 * r84;
    r99 = fmaf(r49, r99, r49 * r24);
    WriteSum3<float, float>((float *)inout_shared, r1, r86, r99);
  };
  FlushSumShared<3, float>(out_focal_and_extra_precond_tril,
                           12 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvSplitFixedPrincipalPointFixedPointResJacFirst(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *focal_and_extra, unsigned int focal_and_extra_num_alloc,
    SharedIndex *focal_and_extra_indices, float *pixel,
    unsigned int pixel_num_alloc, float *principal_point,
    unsigned int principal_point_num_alloc, float *point,
    unsigned int point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *const out_rTr, float *out_pose_jac,
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
    size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvSplitFixedPrincipalPointFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, focal_and_extra, focal_and_extra_num_alloc,
      focal_and_extra_indices, pixel, pixel_num_alloc, principal_point,
      principal_point_num_alloc, point, point_num_alloc, out_res,
      out_res_num_alloc, out_rTr, out_pose_jac, out_pose_jac_num_alloc,
      out_pose_njtr, out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc, out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc, out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc, problem_size);
}

} // namespace caspar